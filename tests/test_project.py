"""Tests for Project workdir lifecycle and config loading."""

from __future__ import annotations

import pytest

from lenticular.config import deep_merge, load_default
from lenticular.project import Project


def test_init_creates_layout(sample_rgb, tmp_path):
    proj = Project.init(sample_rgb, tmp_path, name="fairy_door")
    assert proj.path == tmp_path / "fairy_door"
    assert proj.source.exists()
    assert proj.config.exists()
    assert proj.masks_dir.is_dir()
    assert proj.views_dir.is_dir()
    # Source copied faithfully.
    assert proj.source.read_bytes() == sample_rgb.read_bytes()


def test_init_defaults_name_from_stem(sample_rgb, tmp_path):
    proj = Project.init(sample_rgb, tmp_path)
    assert proj.path.name == "source"


def test_init_rejects_existing(sample_rgb, tmp_path):
    Project.init(sample_rgb, tmp_path, name="dup")
    with pytest.raises(FileExistsError):
        Project.init(sample_rgb, tmp_path, name="dup")


def test_load_config_merges_defaults(sample_rgb, tmp_path):
    proj = Project.init(sample_rgb, tmp_path, name="p")
    cfg = proj.load_config()
    # Defaults present.
    assert cfg["views"]["count"] == 8
    assert cfg["depth"]["model"] == "dav2"


def test_deep_merge_preserves_unmentioned_keys():
    base = {"a": {"b": 1, "c": 2}, "x": 9}
    over = {"a": {"b": 99}}
    merged = deep_merge(base, over)
    assert merged == {"a": {"b": 99, "c": 2}, "x": 9}
    # Inputs untouched.
    assert base["a"]["b"] == 1


def test_default_config_loads():
    cfg = load_default()
    assert "views" in cfg and "layers" in cfg and "depth" in cfg


def test_list_masks_empty(sample_rgb, tmp_path):
    proj = Project.init(sample_rgb, tmp_path, name="p")
    assert proj.list_masks() == []
