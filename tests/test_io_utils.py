"""Tests for image/mask/depth I/O round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from lenticular import io_utils


def test_rgb_roundtrip(sample_rgb):
    arr = io_utils.load_rgb(sample_rgb)
    assert arr.shape[2] == 3 and arr.dtype == np.uint8


def test_mask_roundtrip(tmp_path):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 3:8] = True
    path = tmp_path / "m.png"
    io_utils.save_mask(mask, path)
    loaded = io_utils.load_mask(path)
    np.testing.assert_array_equal(loaded, mask)


def test_depth_roundtrip(tmp_path):
    depth = np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
    path = tmp_path / "d.png"
    io_utils.save_depth(depth, path)
    loaded = io_utils.load_depth(path)
    # 16-bit quantization: tolerance of one level.
    np.testing.assert_allclose(loaded, depth, atol=1.5 / 65535.0)


def test_feather_returns_float_in_unit_range():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    alpha = io_utils.feather_mask(mask, px=2)
    assert alpha.dtype == np.float32
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    # Core still hard inside.
    assert alpha[10, 10] == pytest.approx(1.0)


def test_feather_zero_px_returns_binary_float():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    alpha = io_utils.feather_mask(mask, px=0)
    np.testing.assert_array_equal(alpha, mask.astype(np.float32))
