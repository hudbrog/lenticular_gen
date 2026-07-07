"""Layer assembly: combine masks + depth into an ordered LayerSet.

Algorithm
---------
1. Load every mask in ``masks/``. The character mask is special (zero-parallax
   anchor candidate); everything else is an additional layer.
2. For each mask, compute a representative depth = median of the depth map over
   the mask region (robust to DAV2's per-pixel noise on flat cel regions).
3. The background layer is the residual: pixels not covered by any mask. Its
   depth is the median over that residual region. If no residual exists, no
   explicit background layer is emitted.
4. Sort all layers back-to-front by depth (deepest = index 0).
5. Pick the focus (zero-parallax) layer: the character if present, else the
   layer whose depth is closest to the median of all layer depths.
6. Compute per-layer shift ``shift_px_per_view`` from the parallax formula at
   the extreme view (``t = +1``): ``s_max * (D_i - D_focus)``.

The user may then hand-edit ``layers.json`` to reorder, pin a layer, or tweak
shifts before preview/render.

Overlap resolution (back-to-front occlusion) happens at render time in
``pipeline.render``, not here — assemble only decides *ordering and shift*, so
that ``layers.json`` stays a pure, editable description.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io_utils import load_depth, load_mask
from ..layers import (
    BACKGROUND_LAYER_NAME,
    CHARACTER_LAYER_NAME,
    Layer,
    LayerSet,
)
from ..parallax import default_s_max


def _median_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    """Median depth over a mask region (0 if mask is empty)."""
    if not mask.any():
        return 0.0
    return float(np.median(depth[mask]))


def load_effective_masks(
    masks_dir: Path,
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Load masks from a directory and resolve overlaps back-to-front.

    Returns a dict ``{name: bool mask}`` where every pixel belongs to exactly
    one layer (front layers occlude layers behind them). Sorting is by filename
    stem; the caller (render) re-sorts by the layer depths in ``layers.json``.
    Background is synthesized as the residual if not provided explicitly.

    NOTE: the front/back order used here is the *raw mask pile* order. The
    canonical depth ordering is applied in render via ``layers.json``; this
    helper is only for cutout-style composites. For parallax rendering use
    :func:`resolve_overlaps` which takes the ordered LayerSet.
    """
    raw_masks: dict[str, np.ndarray] = {}
    for p in sorted(Path(masks_dir).glob("*.png")):
        raw_masks[p.stem] = load_mask(p)

    has_explicit_bg = BACKGROUND_LAYER_NAME in raw_masks
    if not has_explicit_bg:
        union = np.zeros(image_shape, dtype=bool)
        for m in raw_masks.values():
            union |= m
        residual = ~union
        # Only synthesize a background layer if the residual is non-trivial.
        # Tiny residuals (<1%) come from morphological cleanup of cluster masks
        # and are not a real background plane.
        if residual.mean() > 0.01:
            raw_masks[BACKGROUND_LAYER_NAME] = residual
    return raw_masks


def resolve_overlaps(
    raw_masks: dict[str, np.ndarray],
    back_to_front: list[str],
) -> dict[str, np.ndarray]:
    """Make masks non-overlapping: each front layer occludes layers behind.

    Args:
        raw_masks: ``{name: bool mask}``, possibly overlapping.
        back_to_front: layer names ordered deepest-first.

    Returns:
        ``{name: bool mask}`` where every pixel belongs to exactly one layer.
    """
    effective: dict[str, np.ndarray] = {}
    occluded = np.zeros_like(next(iter(raw_masks.values())), dtype=bool)
    for name in reversed(back_to_front):
        remaining = raw_masks[name] & ~occluded
        effective[name] = remaining
        occluded |= raw_masks[name]
    return effective


def build_inpaint_mask(
    layerset: LayerSet,
    masks: dict[str, np.ndarray],
    s_max: float,
    vertical_margin_px: int = 2,
) -> np.ndarray:
    """Build the mask of background to inpaint (full-background-once strategy).

    The mask is the union of every layer *at or in front of* the focus depth —
    i.e. the occluding layers whose shift reveals hidden background. Layers
    behind the focus (the far background) are kept and shifted, not inpainted.

    The mask is dilated horizontally by ±``s_max`` (the maximum disocclusion
    reveal over the symmetric view range) plus a small vertical margin.

    For mask-based segmentation (character + background), this inpaints the
    character silhouette. For cluster-based segmentation (5 depth slabs), this
    inpaints the focus layer and everything nearer, keeping the deeper slabs
    as the revealed background.
    """
    h, w = next(iter(masks.values())).shape
    focus_depth = layerset.focus.depth

    # Union of layers at or in front of the focus (depth >= focus_depth).
    # The farthest layer is never inpainted (it IS the background plane).
    ordered = sorted(layerset, key=lambda ly: ly.depth)
    if len(ordered) <= 1:
        return np.zeros((h, w), dtype=bool)
    background_name = ordered[0].name  # deepest layer is the kept background

    union = np.zeros((h, w), dtype=bool)
    for layer in layerset:
        if layer.name == background_name:
            continue  # the kept background plane
        if layer.depth < focus_depth and layer.name != layerset.focus.name:
            continue  # behind focus: kept & shifted, not inpainted
        m = masks.get(layer.name)
        if m is not None:
            union |= m

    # Horizontal dilation by ±s_max: shift the union left and right and OR.
    s = int(round(abs(s_max)))
    dilated = union.copy()
    for shift in range(1, s + 1):
        # Shift right
        dilated[:, shift:] |= union[:, : w - shift]
        # Shift left
        dilated[:, : w - shift] |= union[:, shift:]

    # Vertical margin (a few rows top/bottom) for anti-flicker safety.
    vm = max(0, vertical_margin_px)
    if vm:
        dilated[:vm, :] |= union[:vm, :]
        # Propagate the top/bottom rows outward.
        for _ in range(vm):
            dilated[1:, :] |= dilated[:-1, :]
            dilated[:-1, :] |= dilated[1:, :]

    return dilated


def assemble_layers(
    masks_dir: Path,
    depth_path: Path,
    image_shape: tuple[int, int],
    focus_layer_name: str | None = None,
    s_max: int | None = None,
    overrides: dict[str, dict[str, object]] | None = None,
) -> LayerSet:
    """Build an ordered LayerSet from a masks directory + depth map.

    See module docstring for the algorithm. Returns the LayerSet only; overlap
    resolution is deferred to render time.

    Args:
        overrides: optional ``{layer_name: {"shift_px_per_view": float, "pinned": bool}}``
            applied on top of computed defaults. A pinned layer keeps its
            override shift and ignores the depth-derived value.
    """
    depth = load_depth(depth_path)
    if s_max is None:
        s_max = default_s_max(image_shape[1])

    raw_masks = load_effective_masks(masks_dir, image_shape)
    has_character = CHARACTER_LAYER_NAME in raw_masks

    # Representative depth per layer.
    depths = {name: _median_depth(depth, m) for name, m in raw_masks.items()}

    # Sort back-to-front (deepest first). Ties broken by name for determinism.
    ordered_names = sorted(raw_masks, key=lambda n: (depths[n], n))

    # Focus layer selection.
    if focus_layer_name is not None and focus_layer_name in depths:
        focus_name = focus_layer_name
    elif has_character:
        focus_name = CHARACTER_LAYER_NAME
    else:
        all_depths = np.array([depths[n] for n in ordered_names])
        focus_name = ordered_names[int(np.argmin(np.abs(all_depths - np.median(all_depths))))]

    focus_depth = depths[focus_name]
    focus_index = ordered_names.index(focus_name)
    overrides = overrides or {}

    layers = []
    for name in ordered_names:
        pinned = name == focus_name
        shift = round(float(s_max) * (depths[name] - focus_depth), 2)
        ov = overrides.get(name)
        if ov:
            if "shift_px_per_view" in ov:
                shift = float(ov["shift_px_per_view"])  # type: ignore[arg-type]
            if "pinned" in ov:
                pinned = bool(ov["pinned"])
                if pinned:
                    shift = 0.0
        layers.append(
            Layer(name=name, depth=depths[name], pinned=pinned, shift_px_per_view=shift)
        )
    return LayerSet(layers=layers, focus_index=focus_index)
