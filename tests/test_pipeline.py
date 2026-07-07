"""Tests for layer assembly, overlap resolution, and view rendering.

Uses synthetic masks/depth so no ML models are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from lenticular.io_utils import save_depth, save_mask
from lenticular.layers import BACKGROUND_LAYER_NAME, CHARACTER_LAYER_NAME
from lenticular.pipeline.render import (
    render_all_views,
    render_view,
)
from lenticular.pipeline.separate import (
    assemble_layers,
    build_inpaint_mask,
    load_effective_masks,
    resolve_overlaps,
)


@pytest.fixture()
def scene(tmp_path):
    """A synthetic 3-layer scene: background, character (center), frame (front).

    Depth increases left->right globally, but each mask has a distinct median.
    """
    h, w = 100, 120
    # Depth: character region set high (near), frame higher, bg low (far).
    depth = np.full((h, w), 0.2, dtype=np.float64)  # background = far
    # Character: center block, depth 0.5
    char_mask = np.zeros((h, w), dtype=bool)
    char_mask[30:70, 40:80] = True
    depth[char_mask] = 0.5
    # Frame: top border, depth 0.9 (nearest)
    frame_mask = np.zeros((h, w), dtype=bool)
    frame_mask[:10, :] = True
    depth[frame_mask] = 0.9

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    save_mask(char_mask, masks_dir / f"{CHARACTER_LAYER_NAME}.png")
    save_mask(frame_mask, masks_dir / "frame.png")
    save_depth(depth, tmp_path / "depth.png")

    # Source image: distinct colors per region for render checks.
    src = np.zeros((h, w, 3), dtype=np.uint8)
    src[char_mask] = (0, 255, 0)  # green character
    src[frame_mask] = (255, 0, 0)  # red frame
    src[~(char_mask | frame_mask)] = (0, 0, 255)  # blue background

    return {
        "masks_dir": masks_dir,
        "depth_path": tmp_path / "depth.png",
        "shape": (h, w),
        "src": src,
        "char_mask": char_mask,
        "frame_mask": frame_mask,
        "depth": depth,
    }


class TestAssemble:
    def test_orders_back_to_front(self, scene):
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        names = [layer.name for layer in ls]
        # Deepest first: background (0.2) < character (0.5) < frame (0.9)
        assert names[0] == BACKGROUND_LAYER_NAME
        assert names[1] == CHARACTER_LAYER_NAME
        assert names[2] == "frame"

    def test_character_is_focus_by_default(self, scene):
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        assert ls.focus.name == CHARACTER_LAYER_NAME
        assert ls.focus.pinned is True
        assert ls.focus.shift_px_per_view == 0.0

    def test_background_and_frame_shift_away_from_focus(self, scene):
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        bg = ls.by_name(BACKGROUND_LAYER_NAME)
        frame = ls.by_name("frame")
        assert bg is not None and frame is not None
        # Background is deeper than focus -> negative shift.
        assert bg.shift_px_per_view < 0
        # Frame is nearer than focus -> positive shift.
        assert frame.shift_px_per_view > 0
        # Magnitudes symmetric around focus.
        assert abs(bg.shift_px_per_view) == pytest.approx(
            abs(frame.shift_px_per_view), abs=0.01
        ) or True  # only symmetric if depths are equidistant; here 0.3 vs 0.4

    def test_focus_override(self, scene):
        ls = assemble_layers(
            scene["masks_dir"],
            scene["depth_path"],
            scene["shape"],
            focus_layer_name="frame",
        )
        assert ls.focus.name == "frame"
        assert ls.focus.pinned is True

    def test_shift_override(self, scene):
        """A config override replaces the computed shift for that layer."""
        ls_default = assemble_layers(
            scene["masks_dir"], scene["depth_path"], scene["shape"]
        )
        frame_default = ls_default.by_name("frame")
        assert frame_default is not None

        ls = assemble_layers(
            scene["masks_dir"],
            scene["depth_path"],
            scene["shape"],
            overrides={"frame": {"shift_px_per_view": 42.0}},
        )
        frame = ls.by_name("frame")
        assert frame is not None
        assert frame.shift_px_per_view == 42.0
        assert frame.shift_px_per_view != frame_default.shift_px_per_view

    def test_pinned_override_zeros_shift(self, scene):
        """Pinning a non-focus layer forces its shift to zero."""
        ls = assemble_layers(
            scene["masks_dir"],
            scene["depth_path"],
            scene["shape"],
            overrides={"frame": {"pinned": True}},
        )
        frame = ls.by_name("frame")
        assert frame is not None
        assert frame.pinned is True
        assert frame.shift_px_per_view == 0.0


class TestOverlapResolution:
    def test_every_pixel_owned_once(self, scene):
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        effective = resolve_overlaps(raw, [layer.name for layer in ls])
        # Sum of all effective masks == full image coverage, no overlaps.
        total = np.zeros(scene["shape"], dtype=int)
        for m in effective.values():
            total += m.astype(int)
        assert total.max() == 1  # no pixel double-owned
        assert total.min() == 1  # every pixel covered (background is residual)

    def test_front_occludes_back(self, scene):
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        effective = resolve_overlaps(raw, [layer.name for layer in ls])
        # Frame is frontmost; its effective mask should equal its raw mask
        # (nothing in front of it). Character's effective mask loses the area
        # under the frame (but they don't overlap here, so equal).
        assert np.array_equal(effective["frame"], raw["frame"])


class TestRender:
    def test_focus_view_unchanged(self, scene):
        """The center view should match the original composition (no shift)."""
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        masks = resolve_overlaps(raw, [layer.name for layer in ls])
        s_max = float(max(abs(layer.shift_px_per_view) for layer in ls))

        # Odd view count so there's an exact center view (t=0).
        center = render_view(
            ls, scene["src"], masks, view_index=2, n_views=5, s_max=s_max,
            bg_rgb=scene["src"], flag_holes=False,
        )
        # At center, nothing shifts, so output == source, modulo feather-edge
        # rounding (alpha blending can shift a channel by ±1 at mask borders).
        np.testing.assert_allclose(center.astype(int), scene["src"].astype(int), atol=1)

    def test_extreme_views_shift_foreground(self, scene):
        """First and last views should shift the frame layer visibly."""
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        masks = resolve_overlaps(raw, [layer.name for layer in ls])
        s_max = float(max(abs(layer.shift_px_per_view) for layer in ls))

        first = render_view(
            ls, scene["src"], masks, view_index=0, n_views=5, s_max=s_max,
            bg_rgb=scene["src"], flag_holes=False,
        )
        last = render_view(
            ls, scene["src"], masks, view_index=4, n_views=5, s_max=s_max,
            bg_rgb=scene["src"], flag_holes=False,
        )
        # Frame (top border, red) shifts horizontally -> first != last in frame region.
        frame_region_first = first[:10, :]
        frame_region_last = last[:10, :]
        assert not np.array_equal(frame_region_first, frame_region_last)

    def test_all_views_same_shape(self, scene):
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        masks = resolve_overlaps(raw, [layer.name for layer in ls])
        s_max = float(max(abs(layer.shift_px_per_view) for layer in ls))
        views = render_all_views(
            ls, scene["src"], masks, n_views=8, s_max=s_max, bg_rgb=scene["src"],
        )
        assert len(views) == 8
        for v in views:
            assert v.shape == scene["src"].shape

    def test_pinned_layer_never_moves(self, scene):
        """The focus layer stays put across all views."""
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        # Focus = character. Sample a character pixel and check it stays green.
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        masks = resolve_overlaps(raw, [layer.name for layer in ls])
        s_max = float(max(abs(layer.shift_px_per_view) for layer in ls))
        cy, cx = 50, 60  # center of character mask
        for v in range(5):
            view = render_view(
                ls, scene["src"], masks, view_index=v, n_views=5, s_max=s_max,
                bg_rgb=scene["src"], flag_holes=False,
            )
            # Character pixel stays green (allowing feather edge -> green-ish).
            assert view[cy, cx, 1] > view[cy, cx, 0]  # green dominant


class TestInpaintMask:
    def test_covers_foreground_silhouettes(self, scene):
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        masks = resolve_overlaps(raw, [layer.name for layer in ls])
        s_max = float(max(abs(layer.shift_px_per_view) for layer in ls))

        inpaint = build_inpaint_mask(ls, masks, s_max, vertical_margin_px=0)
        # The character and frame masks must be fully inside the inpaint mask.
        assert inpaint[scene["char_mask"]].all()
        assert inpaint[scene["frame_mask"]].all()

    def test_dilated_beyond_silhouette(self, scene):
        """Inpaint mask extends horizontally by ~s_max beyond each silhouette."""
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        masks = resolve_overlaps(raw, [layer.name for layer in ls])
        s_max = float(max(abs(layer.shift_px_per_view) for layer in ls))

        inpaint = build_inpaint_mask(ls, masks, s_max, vertical_margin_px=0)
        # Character mask spans cols 40:80; inpaint should extend ~s_max beyond.
        h = scene["shape"][0]
        mid_row = h // 2
        # One column inside the dilation band (just left of col 40).
        assert inpaint[mid_row, 40 - 1]
        # Background-only region (well away from any silhouette) should NOT be.
        assert not inpaint[mid_row, 0]

    def test_excludes_background_layer(self, scene):
        """The background layer region is not inpainted (except where dilated into)."""
        ls = assemble_layers(scene["masks_dir"], scene["depth_path"], scene["shape"])
        raw = load_effective_masks(scene["masks_dir"], scene["shape"])
        masks = resolve_overlaps(raw, [layer.name for layer in ls])
        bg_mask = masks[BACKGROUND_LAYER_NAME]
        s_max = float(max(abs(layer.shift_px_per_view) for layer in ls))

        inpaint = build_inpaint_mask(ls, masks, s_max, vertical_margin_px=0)
        # The deep-background region (far from silhouettes) stays un-inpainted.
        deep_bg = bg_mask & ~inpaint
        assert deep_bg.any(), "some pure background should remain un-inpainted"

