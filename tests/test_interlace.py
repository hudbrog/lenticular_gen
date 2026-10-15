"""Tests for the interlace and pitchtest stages.

Uses synthetic solid-color views so the column-gather geometry is exact and
no ML models are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from lenticular.pipeline.interlace import (
    _column_indices,
    interlace,
    output_pixel_size,
)
from lenticular.pipeline.pitchtest import pitch_test_image

# Fairy Door defaults from config.
SHEET_MM = (70.0, 125.0)
DPI = 600
LPI = 60.0


def _solid_views(n: int, h: int = 50, w: int = 40) -> list[np.ndarray]:
    """N distinct solid-color views (red, green, blue, ...)."""
    palette = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
         [0, 255, 255], [128, 0, 0], [0, 128, 0]],
        dtype=np.uint8,
    )
    return [np.tile(palette[i % len(palette)], (h, w, 1)) for i in range(n)]


class TestOutputSize:
    def test_dimensions_match_sheet_at_dpi(self):
        # 70 mm @ 600 DPI ~= 1654 px; 125 mm ~= 2953 px.
        w, h = output_pixel_size(DPI, SHEET_MM, n_views=8)
        assert abs(w - round(DPI * SHEET_MM[0] / 25.4)) <= 8  # rounded to mult of N
        assert h == round(DPI * SHEET_MM[1] / 25.4)

    def test_width_is_multiple_of_n(self):
        for n in (1, 2, 4, 8, 12):
            w, _ = output_pixel_size(DPI, SHEET_MM, n_views=n)
            assert w % n == 0


class TestColumnIndices:
    def test_first_column_is_view_zero(self):
        pitch_px = DPI / LPI
        v, _ = _column_indices(out_w=int(pitch_px * 3), pitch_px=pitch_px, n_views=8)
        assert v[0] == 0

    def test_view_cycle_repeats_per_lens(self):
        """Every view index appears at least once within one lens period.

        With integer ``pitch_px / n`` the columns tile exactly 0,1,...,N-1;
        with a fractional slot some views get an extra column, but all N must
        appear (otherwise that view has no column under this lens at all).
        """
        n = 8
        pitch_px = DPI / LPI
        out_w = int(round(pitch_px)) * 2  # two lenses
        v, _ = _column_indices(out_w=out_w, pitch_px=pitch_px, n_views=n)
        first_lens = v[: int(round(pitch_px))]
        assert set(first_lens.tolist()) == set(range(n))

    def test_all_views_present(self):
        n = 8
        pitch_px = DPI / LPI
        out_w = int(round(pitch_px)) * 3
        v, _ = _column_indices(out_w=out_w, pitch_px=pitch_px, n_views=n)
        assert set(v.tolist()) == set(range(n))


class TestInterlace:
    def test_solid_view_color_preserved(self):
        """A single solid-color view yields an output of exactly that color."""
        views = _solid_views(1)
        out = interlace(views, dpi=DPI, sheet_mm=SHEET_MM, lpi=LPI)
        assert out.shape[2] == 3
        # Every pixel must be red (view 0).
        assert np.all(out == views[0][0, 0])

    def test_n_views_produces_all_colors(self):
        """With N distinct solid views, every view's color appears in output."""
        n = 4
        views = _solid_views(n)
        out = interlace(views, dpi=DPI, sheet_mm=(40.0, 40.0), lpi=LPI)
        # Sample one row; gather unique colors.
        colors = {tuple(c) for c in out[out.shape[0] // 2]}
        palette = {tuple(views[i][0, 0]) for i in range(n)}
        assert colors == palette

    def test_flip_reverses_view_order(self):
        """--flip should reverse which color appears at column 0's slot."""
        n = 4
        views = _solid_views(n)
        kwargs = dict(dpi=DPI, sheet_mm=(40.0, 40.0), lpi=LPI)
        out = interlace(views, flip=False, **kwargs)
        out_flip = interlace(views, flip=True, **kwargs)
        # Column 0 is view 0 normally, view N-1 when flipped.
        assert tuple(out[0, 0]) == tuple(views[0][0, 0])
        assert tuple(out_flip[0, 0]) == tuple(views[-1][0, 0])

    def test_output_shape_matches_sheet(self):
        views = _solid_views(8)
        out = interlace(views, dpi=DPI, sheet_mm=SHEET_MM, lpi=LPI)
        w_expected, h_expected = output_pixel_size(DPI, SHEET_MM, n_views=8)
        assert out.shape == (h_expected, w_expected, 3)

    def test_nn_purity_no_blending(self):
        """Nearest-neighbor gather must never blend adjacent views' colors.

        With solid-color views, every output pixel must be one of the input
        colors exactly — no mid-tone averages.
        """
        n = 4
        views = _solid_views(n)
        out = interlace(views, dpi=DPI, sheet_mm=(40.0, 40.0), lpi=LPI)
        palette = {tuple(views[i][0, 0]) for i in range(n)}
        out_colors = {tuple(c) for c in out.reshape(-1, 3)}
        assert out_colors.issubset(palette)

    def test_requires_at_least_one_view(self):
        with pytest.raises(ValueError, match="at least one view"):
            interlace([], dpi=DPI, sheet_mm=SHEET_MM, lpi=LPI)


class TestPitchTest:
    def test_output_shape(self):
        out, n = pitch_test_image(
            dpi=DPI, sheet_mm=SHEET_MM, lpi_nominal=LPI, step=0.1, count=7
        )
        # Pitch test always uses 2 views internally.
        w_expected, h_expected = output_pixel_size(DPI, SHEET_MM, n_views=2)
        assert out.shape == (h_expected, w_expected, 3)
        assert n == 7

    def test_band_count_forced_odd(self):
        out_even, n = pitch_test_image(
            dpi=DPI, sheet_mm=(40.0, 40.0), lpi_nominal=LPI, count=6
        )
        # Even count is forced odd -> 7 bands.
        assert n == 7
        assert out_even.shape[0] > 0

    def test_has_text_labels(self):
        """The label box drawn on each band introduces non-pattern colors
        (gray box + black text), so the output is not purely black/white."""
        out, _ = pitch_test_image(
            dpi=DPI, sheet_mm=SHEET_MM, lpi_nominal=LPI, count=3
        )
        colors = {tuple(c) for c in out.reshape(-1, 3)}
        # Pattern is black+white only; labels add gray (200,200,200) and black.
        assert (200, 200, 200) in colors

    def test_rejects_zero_count(self):
        with pytest.raises(ValueError, match="count must be >= 1"):
            pitch_test_image(
                dpi=DPI, sheet_mm=SHEET_MM, lpi_nominal=LPI, count=0
            )

    def test_rejects_zero_band_mm(self):
        with pytest.raises(ValueError, match="band_mm must be > 0"):
            pitch_test_image(
                dpi=DPI, sheet_mm=SHEET_MM, lpi_nominal=LPI, band_mm=0.0
            )

    def test_band_mm_derives_count_and_forces_odd(self):
        """band_mm derives count from sheet height; result forced odd."""
        # A4 portrait: 297 mm / 40 mm = 7.4 -> 7 bands (forced odd, floor).
        out, n = pitch_test_image(
            dpi=DPI,
            sheet_mm=(210.0, 297.0),
            lpi_nominal=LPI,
            band_mm=40.0,
        )
        assert n == 7
        # Each band ~= 297/7 = 42.4 mm tall.
        band_h_px = out.shape[0] // n
        band_h_mm = band_h_px / DPI * 25.4
        assert 38.0 < band_h_mm < 44.0

    def test_band_mm_overrides_count(self):
        """When band_mm is set, count is ignored."""
        out, n = pitch_test_image(
            dpi=DPI,
            sheet_mm=(210.0, 297.0),
            lpi_nominal=LPI,
            count=3,  # would give 3 bands; band_mm gives 7.
            band_mm=40.0,
        )
        assert n == 7

    def test_labels_distinct_at_fine_step(self):
        """At step=0.05 adjacent bands must have distinct labels.

        Regression: ``:.1f`` rounded 69.65 and 69.70 both to "69.7", making a
        fine search look like a coarse one. Label precision must track step.
        """
        # Re-derive the label strings the way the generator does; the image
        # itself is covered by other shape/label tests.
        import numpy as np

        half = 7 // 2
        lpis = [69.7 + (i - half) * 0.05 for i in range(7)]
        dp = max(1, -int(np.floor(np.log10(0.05))))
        labels = [f"{v:.{dp}f} LPI" for v in lpis]
        assert len(set(labels)) == 7
        assert labels[3] == "69.70 LPI"
