"""Image and mask I/O helpers (RGBA images, single-channel masks, 16-bit depth)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Threshold separating "inside the mask" (255) from "outside" (0) on disk.
MASK_ON = 255


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as ``uint8`` RGB ``(H, W, 3)``."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def load_rgba(path: Path) -> np.ndarray:
    """Load an image as ``uint8`` RGBA ``(H, W, 4)``."""
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"), dtype=np.uint8)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    """Save an ``uint8`` RGB ``(H, W, 3)`` array."""
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_rgba(arr: np.ndarray, path: Path) -> None:
    """Save an ``uint8`` RGBA ``(H, W, 4)`` array."""
    Image.fromarray(arr.astype(np.uint8)).save(path)


def load_mask(path: Path) -> np.ndarray:
    """Load a mask as boolean ``(H, W)`` (True where the layer is present)."""
    with Image.open(path) as im:
        gray = np.array(im.convert("L"), dtype=np.uint8)
    return gray >= (MASK_ON // 2 + 1)


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean mask as 0/255 single-channel PNG."""
    Image.fromarray(mask.astype(np.uint8) * MASK_ON).save(path)


def load_depth(path: Path) -> np.ndarray:
    """Load a 16-bit depth PNG as ``float64`` normalized to ``[0, 1]``.

    Convention: 0 = far, 1 = near (matches the in-memory depth map).
    """
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.uint16)
    return arr.astype(np.float64) / 65535.0


def save_depth(depth: np.ndarray, path: Path) -> None:
    """Save a ``[0, 1]`` float depth map as a 16-bit grayscale PNG."""
    clipped = np.clip(depth, 0.0, 1.0)
    arr16 = (clipped * 65535.0).round().astype(np.uint16)
    Image.fromarray(arr16).save(path)


def feather_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Feather a boolean mask by ``px`` pixels (soft outer edge).

    Uses a box blur on the alpha; the core stays hard to avoid halos.
    Returns a float32 alpha map in ``[0, 1]``.
    """
    if px <= 0:
        return mask.astype(np.float32)
    alpha = mask.astype(np.float32)
    # Simple separable box blur of the requested radius.
    from PIL import ImageFilter

    im = Image.fromarray((alpha * 255.0).round().astype(np.uint8))
    blurred = im.filter(ImageFilter.BoxBlur(radius=px))
    return np.array(blurred, dtype=np.float32) / 255.0
