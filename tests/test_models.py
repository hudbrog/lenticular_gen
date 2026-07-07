"""Tests for pure helper functions in the model wrappers (no GPU/models needed)."""

from __future__ import annotations

import numpy as np
import pytest

from lenticular.models.depth import colorize_depth, estimate_depth
from lenticular.models.segment import REMBG_DEFAULT_MODEL


def test_colorize_depth_shape_and_range():
    depth = np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
    rgb = colorize_depth(depth)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8
    assert rgb.min() >= 0 and rgb.max() <= 255


def test_colorize_depth_far_and_near_differ():
    # Far (0.0) and near (1.0) should map to visibly different colors.
    far = colorize_depth(np.array([[0.0]]))[0, 0]
    near = colorize_depth(np.array([[1.0]]))[0, 0]
    assert not np.array_equal(far, near)


def test_default_rembg_model_is_general_use():
    # isnet-anime's ONNX export is broken in rembg 2.0.69; we default to the
    # general IS-Net. This test guards against silently switching back.
    assert REMBG_DEFAULT_MODEL == "isnet-general-use"


def test_depth_dispatcher_rejects_unknown_backend():
    dummy = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown depth backend"):
        estimate_depth(dummy, model="nonexistent")
