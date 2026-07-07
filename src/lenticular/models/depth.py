"""Monocular depth estimation.

Default backend: Depth Anything V2 (relative, Large) via ``transformers``.
Returns a depth map normalized to ``[0, 1]`` with the convention
**0 = far, 1 = near** so that ordering back-to-front is ``argsort(depth)``.

Depth Anything outputs *near = bright* in its raw predictions, but the model's
metric semantics are inverted relative to our convention, so we negate and
re-normalize here. DAV2 is trained on photorealistic data and over-smooths flat
cel-shaded regions; downstream code must treat the map as an *ordering prior*
(mask-median level), not a per-pixel ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthResult:
    """Output of a depth estimator.

    Attributes:
        depth: ``(H, W)`` float64 in ``[0, 1]``. 0 = far, 1 = near.
        preview: ``(H, W, 3)`` uint8 colorized depth (turormap) for humans.
    """

    depth: np.ndarray
    preview: np.ndarray


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` depth to an RGB turbo-ish preview (far=blue, near=red)."""
    # matplotlib turbo colormap is always available in this project; cv2 is not
    # a dependency, so we use the matplotlib path directly.
    import matplotlib.cm as mpl_cm

    rgba = mpl_cm.turbo(np.clip(depth, 0.0, 1.0))  # (H, W, 4) float
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return np.asarray(rgb)


def estimate_depth_dav2(
    rgb: np.ndarray,
    model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
    device: str | None = None,
) -> DepthResult:
    """Run Depth Anything V2 (relative) and return a normalized ``[0,1]`` map.

    DAV2 predicts *disparity-like* values where larger = nearer. We invert to
    our convention (1 = near) and min-max normalize per image.
    """
    import torch  # local import: heavy, and only needed when this backend runs
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_id)  # type: ignore[no-untyped-call]
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    image = Image.fromarray(rgb.astype(np.uint8))
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # DAV2 outputs (1, 1, H, W); interpolate back to source resolution.
    predicted = outputs.predicted_depth
    h, w = rgb.shape[:2]
    predicted = torch.nn.functional.interpolate(
        predicted.unsqueeze(0) if predicted.ndim == 3 else predicted,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    raw = predicted.cpu().numpy().astype(np.float64)

    # DAV2: larger value = nearer. Normalize then map near -> 1.0.
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo if hi > lo else 1.0
    disp = np.asarray((raw - lo) / span)  # 0=far, 1=near in DAV2's own terms
    return DepthResult(depth=disp, preview=colorize_depth(disp))


def estimate_depth_marigold(
    rgb: np.ndarray,
    model_id: str = "prs-eth/marigold-depth-v1-1",
    device: str | None = None,
    num_inference_steps: int = 4,
) -> DepthResult:
    """Run Marigold (diffusion-based) and return a normalized ``[0,1]`` map.

    Marigold's SD-derived prior captures finer structure on flat cel-shaded
    regions than DAV2, at the cost of more VRAM/time. Uses the LCM-style few-step
    schedule (``num_inference_steps=4``) by default for a usable pipeline speed.
    Marigold predicts *metric depth* (meters); we invert to disparity-like
    (near=high) and min-max normalize.
    """
    import torch
    from diffusers import MarigoldDepthPipeline
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = MarigoldDepthPipeline.from_pretrained(model_id)  # type: ignore[no-untyped-call]
    pipe = pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:  # some diffusers versions lack this
        pass

    image = Image.fromarray(rgb.astype(np.uint8))
    with torch.inference_mode():
        result = pipe(
            image,
            num_inference_steps=num_inference_steps,
            processing_resolution=768,
            match_input_resolution=True,
        )
    pred = result.prediction
    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()
    raw = np.asarray(pred).squeeze().astype(np.float64)

    # Marigold: larger value = farther (depth in meters). Invert to near=high.
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo if hi > lo else 1.0
    disp = np.asarray(1.0 - (raw - lo) / span)  # near=1, far=0
    return DepthResult(depth=disp, preview=colorize_depth(disp))


def estimate_depth(
    rgb: np.ndarray, model: str = "dav2", model_id: str | None = None, device: str | None = None
) -> DepthResult:
    """Dispatch to the configured depth backend.

    Args:
        rgb: source image ``(H, W, 3)`` uint8.
        model: ``"dav2"`` (Depth Anything V2, default, fast) or ``"marigold"``
            (finer detail on flat regions, slower).
        model_id: override the HF model id; None = backend default.
    """
    if model == "marigold":
        return estimate_depth_marigold(
            rgb, model_id=model_id or "prs-eth/marigold-depth-v1-1", device=device
        )
    if model == "dav2":
        return estimate_depth_dav2(
            rgb,
            model_id=model_id or "depth-anything/Depth-Anything-V2-Large-hf",
            device=device,
        )
    raise ValueError(f"Unknown depth backend: {model!r}. Use 'dav2' or 'marigold'.")
