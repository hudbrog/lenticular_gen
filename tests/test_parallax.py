"""Tests for the parallax shift math."""

from __future__ import annotations

import numpy as np
import pytest

from lenticular import parallax


class TestViewParam:
    def test_single_view_is_center(self):
        assert parallax.view_t(0, 1) == 0.0

    def test_odd_n_center_is_zero(self):
        # N=5 -> views at t = -1, -0.5, 0, 0.5, 1
        ts = parallax.view_params(5)
        np.testing.assert_allclose(ts, [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_even_n_symmetric_about_zero(self):
        ts = parallax.view_params(4)
        assert len(ts) == 4
        # Symmetric: t[v] == -t[N-1-v]
        for v in range(4):
            assert ts[v] == pytest.approx(-ts[3 - v])

    def test_extremes_are_pm_one(self):
        ts = parallax.view_params(8)
        assert ts[0] == pytest.approx(-1.0)
        assert ts[-1] == pytest.approx(1.0)

    def test_rejects_zero_views(self):
        with pytest.raises(ValueError):
            parallax.view_params(0)


class TestLayerShift:
    def test_focus_layer_does_not_move(self):
        # A layer at the focus depth is pinned at 0 across all views.
        shifts = parallax.layer_shift(depth=0.5, focus_depth=0.5, s_max=20, n_views=8)
        np.testing.assert_allclose(shifts, np.zeros(8))

    def test_foreground_shifts_opposite_to_background(self):
        # Foreground (D > focus) and background (D < focus) get opposite signs.
        fg = parallax.layer_shift(0.9, 0.5, s_max=20, n_views=5)
        bg = parallax.layer_shift(0.1, 0.5, s_max=20, n_views=5)
        # At the same |t|, signs are opposite.
        assert np.sign(fg[0]) == -np.sign(bg[0])
        # Magnitudes scale with depth distance from focus.
        np.testing.assert_allclose(np.abs(fg), np.abs(bg), rtol=1e-9)

    def test_shift_scales_with_s_max(self):
        s1 = parallax.layer_shift(0.8, 0.5, s_max=10, n_views=5)
        s2 = parallax.layer_shift(0.8, 0.5, s_max=20, n_views=5)
        np.testing.assert_allclose(s2, 2 * s1)

    def test_center_view_is_zero(self):
        shifts = parallax.layer_shift(0.9, 0.5, s_max=20, n_views=5)
        # Center view (v=2 for N=5) is the zero-parallax view.
        assert shifts[2] == pytest.approx(0.0)

    def test_extreme_shift_equals_s_max_times_depth_delta(self):
        # At t = +1 (last view), shift == s_max * (D - D_focus).
        shifts = parallax.layer_shift(0.8, 0.4, s_max=15, n_views=7)
        assert shifts[-1] == pytest.approx(15 * (0.8 - 0.4))

    def test_flip_negates(self):
        s = parallax.layer_shift(0.8, 0.4, s_max=15, n_views=5)
        s_flip = parallax.layer_shift(0.8, 0.4, s_max=15, n_views=5, flip=True)
        np.testing.assert_allclose(s_flip, -s)


class TestDisocclusion:
    def test_focus_layer_reveals_nothing(self):
        # The anchor never moves -> no hidden background exposed.
        assert parallax.max_disocclusion_px(0.5, 0.5, s_max=20) == 0.0

    def test_scales_with_depth_gap_and_s_max(self):
        assert parallax.max_disocclusion_px(1.0, 0.0, s_max=20) == pytest.approx(20.0)
        assert parallax.max_disocclusion_px(0.75, 0.25, s_max=20) == pytest.approx(10.0)

    def test_symmetric_in_depth(self):
        # A layer 0.2 above focus reveals the same band as one 0.2 below.
        assert parallax.max_disocclusion_px(0.7, 0.5, s_max=20) == pytest.approx(
            parallax.max_disocclusion_px(0.3, 0.5, s_max=20)
        )


class TestDefaultSMax:
    def test_about_four_percent(self):
        assert parallax.default_s_max(100) == 4
        assert parallax.default_s_max(250) == 10

    def test_minimum_one(self):
        assert parallax.default_s_max(1) == 1
