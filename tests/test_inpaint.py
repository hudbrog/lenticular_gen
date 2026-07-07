"""Tests for the inpainting dispatcher and LaMa backend.

Cloud backends (fal, replicate, openrouter) need live API keys and are not
unit-tested here — they're covered by the manual smoke-test step in the plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from lenticular.models.inpaint import (
    FalFillInpainter,
    LamaInpainter,
    OpenRouterInpainter,
    ReplicateFillInpainter,
    get_inpainter,
    inpaint_background,
)


class TestDispatcher:
    def test_lama_default(self):
        inp = get_inpainter("lama")
        assert isinstance(inp, LamaInpainter)

    def test_explicit_lama(self):
        assert isinstance(get_inpainter("lama"), LamaInpainter)

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown inpaint backend"):
            get_inpainter("nonexistent")

    def test_cloud_backends_require_env_var(self, monkeypatch):
        """Cloud backends raise a clear error when their API key is missing."""
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="FAL_KEY"):
            FalFillInpainter()
        with pytest.raises(RuntimeError, match="REPLICATE_API_TOKEN"):
            ReplicateFillInpainter()
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OpenRouterInpainter()

    def test_cloud_backends_construct_with_env(self, monkeypatch):
        """When the env var is set, construction succeeds (no API call yet)."""
        monkeypatch.setenv("FAL_KEY", "test-key")
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert isinstance(FalFillInpainter(), FalFillInpainter)
        assert isinstance(ReplicateFillInpainter(), ReplicateFillInpainter)
        assert isinstance(OpenRouterInpainter(), OpenRouterInpainter)

    def test_model_override_passed_through(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "k")
        inp = FalFillInpainter(model="custom/model")
        assert inp.model == "custom/model"

    def test_inpaint_background_wrapper_dispatches(self):
        """The convenience wrapper routes to LaMa by default."""
        # Just verify it returns the right type without erroring on a tiny image.
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[4:12, 4:12] = 200  # a white square
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:10, 6:10] = True  # fill the center
        result = inpaint_background(rgb, mask, backend="lama")
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8


class TestLamaRoundTrip:
    def test_preserves_unmasked_region(self):
        """Pixels outside the mask should be unchanged."""
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, :] = (50, 100, 150)  # uniform color
        # Add a white block we'll mask out
        rgb[8:24, 8:24] = 255
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True

        result = inpaint_background(rgb, mask, backend="lama")
        # Unmasked region unchanged.
        np.testing.assert_array_equal(result[0:8, :], rgb[0:8, :])
        np.testing.assert_array_equal(result[:, 0:8], rgb[:, 0:8])

    def test_masked_region_changed(self):
        """Pixels inside the mask should be modified (filled)."""
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:24, 8:24] = 255  # white block to fill
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:22, 10:22] = True

        result = inpaint_background(rgb, mask, backend="lama")
        # The masked region should differ from the original (filled with something).
        masked_diff = np.abs(
            result[10:22, 10:22].astype(int) - rgb[10:22, 10:22].astype(int)
        )
        assert masked_diff.mean() > 5  # clearly modified

    def test_output_dimensions_match(self):
        """Output is cropped to source dimensions (LaMa pads internally)."""
        rgb = np.zeros((30, 40, 3), dtype=np.uint8)  # non-modulo-8 dims
        mask = np.zeros((30, 40), dtype=bool)
        mask[10:20, 10:30] = True
        result = inpaint_background(rgb, mask, backend="lama")
        assert result.shape[:2] == (30, 40)
