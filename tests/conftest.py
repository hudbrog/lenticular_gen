"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def sample_rgb(tmp_path):
    """A small synthetic RGB image (3 vertical color bands)."""
    arr = np.zeros((40, 60, 3), dtype=np.uint8)
    arr[:, :20] = (255, 0, 0)
    arr[:, 20:40] = (0, 255, 0)
    arr[:, 40:] = (0, 0, 255)
    path = tmp_path / "source.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture()
def sample_depth():
    """A depth map that increases left->right (left = far, right = near)."""
    w = 60
    grad = np.linspace(0.1, 0.9, w, dtype=np.float32)
    return np.tile(grad[None, :], (40, 1))
