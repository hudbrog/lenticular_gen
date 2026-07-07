"""Segmentation: auto subject cutout + interactive object masking.

Two complementary tools:

* :func:`auto_subject_mask` — Rembg with the ``isnet-anime`` model produces a
  fast, anime-tuned subject cutout. Written to ``masks/character.png``.
* :class:`Sam2Segmenter` — wraps SAM 2.1 so the user can click points or drag a
  box per object (back props, foreground frame, ...) to get a clean mask.

The HF transformers SAM2 API is model-forward based (no ``ImagePredictor``
class): prompts go through ``Sam2Processor``, the model returns ``pred_masks``
+ ``iou_scores``, and ``processor.post_process_masks`` upscales to full res.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rembg background-removal model.
# NOTE: ``isnet-anime`` is the anime-tuned IS-Net model and is the ideal choice,
# but its ONNX export in rembg 2.0.69 emits a near-empty/inverted mask on many
# inputs. ``isnet-general-use`` is the same architecture trained on general
# subjects and produces correct, clean cutouts on cartoon art. Switch back to
# ``isnet-anime`` once the export is fixed.
REMBG_DEFAULT_MODEL = "isnet-general-use"
REMBG_ANIME_MODEL = "isnet-anime"
# SAM 2.1 checkpoint. base_plus is a good speed/quality balance; large for
# trickier boundaries (hair, thin frames).
SAM2_MODEL_ID_DEFAULT = "facebook/sam2.1-hiera-base-plus"


def auto_subject_mask(rgb: np.ndarray, model: str = REMBG_DEFAULT_MODEL) -> np.ndarray:
    """Background-removal subject cutout. Returns a boolean ``(H, W)`` mask.

    Uses Rembg (CPU onnxruntime). The ``isnet-anime`` model is specifically
    trained on anime/illustration subjects and gives cleaner edges than the
    general U2-Net on cel-shaded art.
    """
    from PIL import Image
    from rembg import new_session, remove

    # rembg 2.x takes a pre-built session, not a `model=` kwarg (which is
    # silently ignored and falls back to u2net).
    session = new_session(model)
    image = Image.fromarray(rgb.astype(np.uint8))
    mask_pil = remove(image, session=session, only_mask=True)
    mask = np.array(mask_pil.convert("L"), dtype=np.uint8)
    return mask >= 128


@dataclass
class Sam2Segmenter:
    """Stateful SAM 2.1 segmenter: loads the model once, predicts many times.

    The image embedding is recomputed on every :meth:`predict` (simplest correct
    behavior for an interactive picker that may switch images).
    """

    model_id: str = SAM2_MODEL_ID_DEFAULT

    def __post_init__(self) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = Sam2Model.from_pretrained(self.model_id).to(self._device).eval()
        self._processor = Sam2Processor.from_pretrained(self.model_id)

    def predict(
        self,
        rgb: np.ndarray,
        points: list[tuple[float, float]],
        labels: list[int],
    ) -> np.ndarray:
        """Predict a mask from foreground/background point prompts.

        Args:
            rgb: source image ``(H, W, 3)`` uint8.
            points: list of ``(x, y)`` pixel coordinates.
            labels: parallel list; 1 = foreground point, 0 = background point.

        Returns:
            Boolean ``(H, W)`` mask of the highest-IoU prediction.
        """
        from PIL import Image

        image = Image.fromarray(rgb.astype(np.uint8))
        # SAM2 processor expects nested lists: [image][object][points][coords].
        input_points = [[[[float(x), float(y)] for x, y in points]]]
        input_labels = [[list(labels)]]

        inputs = self._processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        inputs_gpu = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs_gpu, multimask_output=True)

        masks = self._processor.post_process_masks(  # type: ignore[no-untyped-call]
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]  # (num_objects=1, num_masks=3, H, W) bool
        scores = outputs.iou_scores.cpu()  # (1, point_batch=1, num_masks=3)
        best = int(scores.view(-1).argmax())
        mask = masks[0, best].numpy().astype(bool)
        return np.asarray(mask)
