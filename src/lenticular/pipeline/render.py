"""Multi-view rendering by horizontal layer shift.

For each view ``v in [0, N-1]`` we composite the layers back-to-front, each
pasted at its per-view horizontal offset. The background plane is either:

* the inpainted background (``bg_rgb``) in the full render path, or
* the *original* source pixels with disocclusion holes flagged magenta in the
  preview path — so you can see exactly what inpainting must fill.

All views are vertically registered (only horizontal shift conveys depth);
any vertical drift would read as flicker on the lenticular print.

Parallax shift per layer uses the formula
``shift_i(v) = s_max * (D_i - D_focus) * t_v`` where ``t_v`` spans ``[-1, +1]``.
Pinned layers (e.g. the focus anchor) are never shifted.
"""

from __future__ import annotations

import numpy as np

from ..io_utils import feather_mask
from ..layers import LayerSet
from ..parallax import layer_shift

# Sentinel color for disocclusion holes in preview mode.
HOLE_COLOR = np.array([255, 0, 255], dtype=np.uint8)  # magenta


def _shift_paste(canvas: np.ndarray, tile_rgba: np.ndarray, shift_px: float) -> np.ndarray:
    """Alpha-composite an RGBA tile onto an RGB canvas at a horizontal offset.

    ``shift_px`` is rounded to integer columns — sufficient for parallax
    preview, and the later interlace stage re-samples per-column anyway.
    """
    w = canvas.shape[1]
    shift = int(round(shift_px))

    src_x0 = max(0, -shift)
    src_x1 = min(w, w - shift)
    if src_x1 <= src_x0:
        return canvas
    dst_x0 = src_x0 + shift

    tile = tile_rgba[:, src_x0:src_x1]
    alpha = (tile[..., 3].astype(np.float32) / 255.0)[..., None]
    dst = canvas[:, dst_x0 : dst_x0 + (src_x1 - src_x0)].astype(np.float32)
    canvas[:, dst_x0 : dst_x0 + (src_x1 - src_x0)] = (
        tile[..., :3].astype(np.float32) * alpha + dst * (1.0 - alpha)
    ).astype(np.uint8)
    return canvas


def _coverage_at_shift(mask: np.ndarray, shift_px: float, w: int) -> np.ndarray:
    """Return a coverage map (h, w) for a mask placed at a horizontal shift."""
    h = mask.shape[0]
    cov = np.zeros((h, w), dtype=bool)
    s = int(round(shift_px))
    x0, x1 = max(0, -s), min(w, w - s)
    cov[:, (x0 + s) : (x1 + s)] = mask[:, x0:x1]
    return cov


def render_view(
    layerset: LayerSet,
    source_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    view_index: int,
    n_views: int,
    s_max: float,
    bg_rgb: np.ndarray,
    feather_px: int = 1,
    flip: bool = False,
    flag_holes: bool = False,
) -> np.ndarray:
    """Render one parallax view.

    Args:
        layerset: ordered layers (back-to-front) + focus index.
        source_rgb: original image ``(H, W, 3)`` uint8 — pixels sampled per layer.
        masks: ``{name: bool mask}``, overlap-resolved.
        view_index: ``v in [0, n_views-1]``.
        n_views: total view count.
        s_max: max parallax shift (px) between extreme views.
        bg_rgb: background plane ``(H, W, 3)``. Inpainted BG for render; the
            source image for preview (with ``flag_holes=True``).
        feather_px: mask edge feather radius.
        flip: reverse parallax direction (fixes pseudoscopic prints).
        flag_holes: paint revealed/disoccluded regions magenta (preview only).
    """
    h, w = source_rgb.shape[:2]
    focus_depth = layerset.focus.depth

    canvas = bg_rgb.astype(np.uint8).copy()
    coverage = np.zeros((h, w), dtype=bool)

    for layer in layerset:  # back-to-front
        mask = masks.get(layer.name)
        if mask is None or not mask.any():
            continue
        shift = (
            0.0
            if layer.pinned
            else float(layer_shift(layer.depth, focus_depth, s_max, n_views, flip)[view_index])
        )

        alpha = feather_mask(mask, feather_px)
        rgba = np.dstack([source_rgb.astype(np.float32), alpha * 255.0]).astype(np.uint8)
        canvas = _shift_paste(canvas, rgba, shift)
        coverage |= _coverage_at_shift(mask, shift, w)

    if flag_holes:
        # Disocclusion = pixels that should show the (inpainted) background but
        # were NOT part of the original source composition. We approximate this
        # as the background region (union of all masks' original footprint)
        # that is uncovered at this view's shifts.
        original_union = np.zeros((h, w), dtype=bool)
        for layer in layerset:
            m = masks.get(layer.name)
            if m is not None:
                original_union |= m
        # Background-visible + not covered by any shifted layer = exposed bg.
        exposed_bg = (~original_union) & (~coverage)
        # But only flag the part that is *disocclusion* (was covered originally,
        # now revealed) — i.e. within the max-shift band of foreground layers.
        reveal_band = _reveal_band(
            layerset, masks, view_index, n_views, s_max, w, h, flip, focus_depth
        )
        holes = exposed_bg & reveal_band
        canvas[holes] = HOLE_COLOR

    return canvas


def _reveal_band(
    layerset: LayerSet,
    masks: dict[str, np.ndarray],
    view_index: int,
    n_views: int,
    s_max: float,
    w: int,
    h: int,
    flip: bool,
    focus_depth: float,
) -> np.ndarray:
    """The band where foreground shift exposes previously-covered background."""
    band = np.zeros((h, w), dtype=bool)
    for layer in layerset:
        mask = masks.get(layer.name)
        if mask is None or layer.pinned:
            continue
        shift = float(layer_shift(layer.depth, focus_depth, s_max, n_views, flip)[view_index])
        band |= _coverage_at_shift(mask, shift, w)
    return band


def render_all_views(
    layerset: LayerSet,
    source_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    n_views: int,
    s_max: float,
    bg_rgb: np.ndarray,
    feather_px: int = 1,
    flip: bool = False,
    flag_holes: bool = False,
) -> list[np.ndarray]:
    """Render all N parallax views, left-to-right."""
    return [
        render_view(
            layerset,
            source_rgb,
            masks,
            v,
            n_views,
            s_max,
            bg_rgb,
            feather_px,
            flip,
            flag_holes,
        )
        for v in range(n_views)
    ]
