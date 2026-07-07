"""Tests for the Layer/LayerSet JSON round-trip and ordering."""

from __future__ import annotations

from lenticular.layers import Layer, LayerSet


def test_layerset_roundtrip(tmp_path):
    layers = LayerSet(
        layers=[
            Layer(name="background", depth=0.1, shift_px_per_view=-8.0),
            Layer(name="character", depth=0.5, pinned=True),
            Layer(name="frame", depth=0.9, shift_px_per_view=8.0),
        ],
        focus_index=1,
    )
    path = tmp_path / "layers.json"
    layers.save(path)
    loaded = LayerSet.load(path)

    assert len(loaded) == 3
    assert loaded.focus_index == 1
    assert loaded.focus.name == "character"
    assert [layer.name for layer in loaded] == ["background", "character", "frame"]
    assert loaded.by_name("frame").depth == 0.9  # type: ignore[union-attr]
    assert loaded.by_name("frame").shift_px_per_view == 8.0
    assert loaded.by_name("missing") is None


def test_mask_filename_matches_name():
    assert Layer(name="character", depth=0.5).mask_filename == "character.png"


def test_back_to_front_order_preserved():
    # layers list is always back-to-front; the deepest layer is index 0.
    layers = LayerSet(
        layers=[
            Layer(name="background", depth=0.0),
            Layer(name="frame", depth=1.0),
        ]
    )
    assert layers[0].name == "background"
    assert layers[1].name == "frame"
