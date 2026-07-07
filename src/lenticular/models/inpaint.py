"""Multi-provider disocclusion inpainting.

Strategy: **full-background-once**. We mask the union of every foreground
silhouette (everything in front of the background plane), dilated horizontally
by ±max shift so the disocclusion bands revealed at every parallax view are
already painted. The inpainter runs once and produces a clean, unoccluded
background plane; every parallax view then composites shifted layers over it
with no per-view inpainting — which keeps the background temporally consistent.

Backends (selectable via ``--inpainter`` or config ``inpaint.backend``):

* ``lama`` (default) — local LaMa GAN. Fast, deterministic, offline. Good on
  smooth cartoon backgrounds; can blur fine linework.
* ``fal`` — fal.ai FLUX Fill (strict binary mask). FLUX-class quality, ~$0.05/img.
  Supports LoRAs for style locking (future). Needs ``FAL_KEY``.
* ``replicate`` — Replicate FLUX Fill Pro (strict binary mask). Same FLUX Fill
  quality, ~$0.05/img. Needs ``REPLICATE_API_TOKEN``.
* ``openrouter`` — OpenRouter /images (Nano Banana Pro). **Experimental** — no
  hard mask, so foreground may drift; the prompt must instruct preservation.
  Reuses an existing ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import io
import os
from typing import Protocol

import numpy as np


class Inpainter(Protocol):
    """Inpaint the masked region of an RGB image, returning the full image."""

    def inpaint(
        self, rgb: np.ndarray, mask: np.ndarray, prompt: str = ""
    ) -> np.ndarray:
        """Fill the masked region.

        Args:
            rgb: ``(H, W, 3)`` uint8 source image.
            mask: boolean ``(H, W)`` mask; True = region to fill.
            prompt: optional background description (used by diffusion backends;
                empty = pure context completion).

        Returns:
            ``(H, W, 3)`` uint8 image with the masked region filled, cropped to
            the source dimensions.
        """
        ...


def _crop_to_source(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crop an output array back to the source (H, W) — cloud/diffusion models
    may return a different resolution."""
    return arr[:h, :w]


def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (PNG/JPEG) into an RGB uint8 array."""
    from PIL import Image

    return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def _pil_to_png_bytes(rgb: np.ndarray, mask: np.ndarray | None = None) -> bytes:
    """Encode an RGB array (optionally with an alpha mask) as PNG bytes.

    If ``mask`` is given, produces an RGBA PNG where masked pixels are fully
    transparent — the OpenAI/fal convention for "edit these pixels".
    """
    from PIL import Image

    if mask is None:
        Image.fromarray(rgb.astype(np.uint8)).save(buf := io.BytesIO(), format="PNG")
    else:
        alpha = (mask.astype(np.uint8)) * 255
        rgba = np.dstack([rgb.astype(np.uint8), alpha])
        Image.fromarray(rgba).save(buf := io.BytesIO(), format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# LaMa (local, default)
# ---------------------------------------------------------------------------


class LamaInpainter:
    """Local LaMa GAN inpainter. Fast, deterministic, offline."""

    def inpaint(
        self, rgb: np.ndarray, mask: np.ndarray, prompt: str = ""
    ) -> np.ndarray:
        from PIL import Image
        from simple_lama_inpainting import SimpleLama

        h, w = rgb.shape[:2]
        # SimpleLama's prepare_img_and_mask does np.array(image), so the image
        # must be RGB (3 ch) — RGBA yields 4 ch and the model errors on 5 ch.
        image_pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).convert("L")
        lama = SimpleLama()
        result = lama(image_pil, mask_pil)
        out = np.array(result.convert("RGB"), dtype=np.uint8)
        return _crop_to_source(out, h, w)


# ---------------------------------------------------------------------------
# fal.ai FLUX Fill (strict mask)
# ---------------------------------------------------------------------------


class FalFillInpainter:
    """fal.ai FLUX Fill inpainter. Strict binary mask, FLUX-class quality."""

    def __init__(self, model: str = "fal-ai/flux-pro/v1.1/fill") -> None:
        self.model = model
        if not os.environ.get("FAL_KEY"):
            raise RuntimeError(
                "FAL_KEY environment variable not set. "
                "Get a key at https://fal.ai/dashboard/keys and export FAL_KEY=..."
            )

    def inpaint(
        self, rgb: np.ndarray, mask: np.ndarray, prompt: str = ""
    ) -> np.ndarray:
        import fal
        from PIL import Image

        h, w = rgb.shape[:2]
        img_buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(img_buf, format="PNG")
        mask_buf = io.BytesIO()
        # fal convention: mask white = fill.
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_buf, format="PNG")

        result = fal.subscribe(
            self.model,
            arguments={
                "image_url": fal.upload_file(img_buf.getvalue()),
                "mask_url": fal.upload_file(mask_buf.getvalue()),
                "prompt": prompt or " ",
            },
        )
        url = result["images"][0]["url"]
        resp = __import__("requests").get(url, timeout=120)
        resp.raise_for_status()
        out = _decode_image_bytes(resp.content)
        return _crop_to_source(out, h, w)


# ---------------------------------------------------------------------------
# Replicate FLUX Fill Pro (strict mask)
# ---------------------------------------------------------------------------


class ReplicateFillInpainter:
    """Replicate FLUX Fill Pro. Strict binary mask, FLUX-class quality."""

    def __init__(self, model: str = "black-forest-labs/flux-fill-pro") -> None:
        self.model = model
        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise RuntimeError(
                "REPLICATE_API_TOKEN environment variable not set. "
                "Get a token at https://replicate.com/account/api-tokens "
                "and export REPLICATE_API_TOKEN=..."
            )

    def inpaint(
        self, rgb: np.ndarray, mask: np.ndarray, prompt: str = ""
    ) -> np.ndarray:
        import replicate

        h, w = rgb.shape[:2]
        img_bytes = _pil_to_png_bytes(rgb)
        # Replicate convention: mask white = fill.
        mask_bytes = _pil_to_png_bytes(
            np.zeros_like(rgb), mask.astype(np.uint8) * 255
        )

        output = replicate.run(
            self.model,
            input={
                "image": io.BytesIO(img_bytes),
                "mask": io.BytesIO(mask_bytes),
                "prompt": prompt or " ",
                "output_format": "png",
            },
        )
        # output is a URL string, a file-like, or an iterator of file-likes.
        if isinstance(output, str):
            resp = __import__("requests").get(output, timeout=120)
            resp.raise_for_status()
            out = _decode_image_bytes(resp.content)
        elif hasattr(output, "read"):
            out = _decode_image_bytes(output.read())
        else:
            # Iterator: take the first element.
            first = next(iter(output))
            out = _decode_image_bytes(first.read() if hasattr(first, "read") else first)
        return _crop_to_source(out, h, w)


# ---------------------------------------------------------------------------
# OpenRouter (semantic, experimental — no hard mask)
# ---------------------------------------------------------------------------


class OpenRouterInpainter:
    """OpenRouter /images endpoint (Nano Banana Pro). Experimental.

    No strict mask support — the prompt must instruct the model to preserve the
    background and fill only the removed-subject area. Foreground may drift;
    not production-safe for exact lenticular alignment.
    """

    def __init__(self, model: str = "google/gemini-3-pro-image-preview") -> None:
        self.model = model
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get a key at https://openrouter.ai/keys and export OPENROUTER_API_KEY=..."
            )

    def inpaint(
        self, rgb: np.ndarray, mask: np.ndarray, prompt: str = ""
    ) -> np.ndarray:
        import base64

        import requests

        h, w = rgb.shape[:2]
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        # Build a prompt that asks for background fill only.
        full_prompt = (
            prompt.strip()
            or "Fill the area where the subject was removed with matching background. "
            "Preserve all existing background exactly. Do not change any unmodified pixels."
        )

        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "lenticular-gen inpaint",
        }
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": full_prompt,
            "input_references": [
                {"type": "image_url", "image_url": {"url": data_url}}
            ],
            "n": 1,
            "output_format": "png",
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/images",
            headers=headers,
            json=payload,  # type: ignore[arg-type]
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        img_b64 = data["data"][0]["b64_json"]
        out = _decode_image_bytes(base64.b64decode(img_b64))
        return _crop_to_source(out, h, w)


# ---------------------------------------------------------------------------
# Factory / dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {
    "lama": LamaInpainter,
    "fal": FalFillInpainter,
    "replicate": ReplicateFillInpainter,
    "openrouter": OpenRouterInpainter,
}


def get_inpainter(
    backend: str = "lama",
    *,
    model: str | None = None,
) -> Inpainter:
    """Construct an inpainter by backend name.

    Args:
        backend: ``"lama"`` (default, offline), ``"fal"``, ``"replicate"``, or
            ``"openrouter"``.
        model: optional model ID override for cloud backends.
    """
    if backend == "lama":
        return LamaInpainter()
    if backend == "fal":
        return FalFillInpainter(model=model) if model else FalFillInpainter()
    if backend == "replicate":
        return ReplicateFillInpainter(model=model) if model else ReplicateFillInpainter()
    if backend == "openrouter":
        return OpenRouterInpainter(model=model) if model else OpenRouterInpainter()
    raise ValueError(
        f"Unknown inpaint backend: {backend!r}. "
        f"Use one of: {', '.join(_BACKENDS)}"
    )


def inpaint_background(
    rgb: np.ndarray,
    mask: np.ndarray,
    backend: str = "lama",
    prompt: str = "",
    model: str | None = None,
) -> np.ndarray:
    """Convenience wrapper: inpaint with the given backend.

    Kept for backward compatibility with code that called the old single
    function directly.
    """
    return get_inpainter(backend, model=model).inpaint(rgb, mask, prompt)
