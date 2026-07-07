"""Configuration loading and merging.

The default config lives at ``config/default.yaml`` (shipped with the package
source). On ``lenticular init`` it is copied into each project workdir as
``config.yaml`` and may be edited there. Runtime values are the deep-merge of
default < project < CLI flags.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Default config ships alongside the source tree (repo root / config).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict. Empty file -> empty dict."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (new dict; inputs untouched)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_default() -> dict[str, Any]:
    """Load the shipped default config."""
    return load_yaml(DEFAULT_CONFIG_PATH)


def load_project(project_config_path: Path) -> dict[str, Any]:
    """Load default merged with a project's ``config.yaml``."""
    return deep_merge(load_default(), load_yaml(project_config_path))
