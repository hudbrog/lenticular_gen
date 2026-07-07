"""Preview outputs: animated GIF and a side-by-side view strip.

The preview composites views with disocclusion holes flagged magenta and no
inpainting, so you can sanity-check the parallax/shift settings before the
expensive inpaint step. The GIF sweeps left-right-left for a continuous loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .layers import LayerSet
from .pipeline.render import render_all_views


def make_gif(views: list[np.ndarray], path: Path, duration_ms: int = 120) -> None:
    """Write an animated GIF that loops left->right->left for continuity."""
    pil_views = [Image.fromarray(v.astype(np.uint8)) for v in views]
    frames = pil_views + list(reversed(pil_views[1:-1]))  # ping-pong
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def make_strip(views: list[np.ndarray], path: Path) -> None:
    """Write all views side by side into one wide PNG."""
    strip = np.hstack([v.astype(np.uint8) for v in views])
    Image.fromarray(strip).save(path)


def render_preview(
    layerset: LayerSet,
    source_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    n_views: int,
    s_max: float,
    gif_path: Path,
    strip_path: Path,
    feather_px: int = 1,
    flip: bool = False,
    duration_ms: int = 120,
) -> None:
    """Render the preview GIF + strip (holes flagged magenta, no inpaint)."""
    views = render_all_views(
        layerset,
        source_rgb,
        masks,
        n_views,
        s_max,
        bg_rgb=source_rgb,
        feather_px=feather_px,
        flip=flip,
        flag_holes=True,
    )
    make_gif(views, gif_path, duration_ms=duration_ms)
    make_strip(views, strip_path)
