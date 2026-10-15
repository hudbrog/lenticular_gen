"""Project workdir: every stage reads/writes artifacts under a project folder.

Layout created by :meth:`Project.init`::

    <workdir>/<name>/
        source.png            # original image (copied)
        config.yaml           # per-project config (copy of defaults)
        depth.png             # 16-bit normalized depth (far=0, near=1)
        depth_preview.png     # colorized depth, for humans
        bg_inpaint.png        # LaMa full-background-once (after inpaint stage)
        preview.gif           # animated parallax sanity check
        preview_strip.png     # all views side by side
        render_meta.json      # final shift table + view params
        layers.json           # ordered layers + focus index
        masks/                # <layer>.png boolean masks (0/255)
        views/                # view_00.png ... view_{N-1}.png
        interlaced.tif        # N views column-gathered for printing (after interlace)
        pitch_test.tif        # banded LPI calibration print (after pitchtest)
    """

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, load_yaml
from .layers import LayerSet

SOURCE_NAME = "source.png"
CONFIG_NAME = "config.yaml"
LAYERS_NAME = "layers.json"


@dataclass
class Project:
    """A single lenticular project rooted at ``path``."""

    path: Path

    # --- paths to standard artifacts ----------------------------------------
    @property
    def source(self) -> Path:
        return self.path / SOURCE_NAME

    @property
    def config(self) -> Path:
        return self.path / CONFIG_NAME

    @property
    def layers(self) -> Path:
        return self.path / LAYERS_NAME

    @property
    def depth(self) -> Path:
        return self.path / "depth.png"

    @property
    def depth_preview(self) -> Path:
        return self.path / "depth_preview.png"

    @property
    def bg_inpaint(self) -> Path:
        return self.path / "bg_inpaint.png"

    @property
    def preview_gif(self) -> Path:
        return self.path / "preview.gif"

    @property
    def preview_strip(self) -> Path:
        return self.path / "preview_strip.png"

    @property
    def render_meta(self) -> Path:
        return self.path / "render_meta.json"

    @property
    def masks_dir(self) -> Path:
        return self.path / "masks"

    @property
    def views_dir(self) -> Path:
        return self.path / "views"

    @property
    def interlaced(self) -> Path:
        return self.path / "interlaced.tif"

    @property
    def pitch_test(self) -> Path:
        return self.path / "pitch_test.tif"

    def mask(self, name: str) -> Path:
        return self.masks_dir / f"{name}.png"

    # --- lifecycle ----------------------------------------------------------
    @classmethod
    def init(cls, source_image: Path, workdir: Path, name: str | None = None) -> Project:
        """Create a project workdir and copy in the source image + default config."""
        if name is None:
            name = source_image.stem
        project = cls(path=workdir / name)
        if project.path.exists():
            raise FileExistsError(f"Project already exists: {project.path}")
        project.path.mkdir(parents=True)
        project.masks_dir.mkdir()
        project.views_dir.mkdir()
        shutil.copy2(source_image, project.source)
        shutil.copy2(DEFAULT_CONFIG_PATH, project.config)
        return project

    def load_config(self) -> dict[str, Any]:
        """Load this project's merged config (default < project config.yaml)."""
        from .config import deep_merge, load_default

        project_cfg = load_yaml(self.config)
        return deep_merge(load_default(), project_cfg)

    def load_layers(self) -> LayerSet:
        return LayerSet.load(self.layers)

    def save_layers(self, layers: LayerSet) -> None:
        layers.save(self.layers)

    def list_masks(self) -> list[Path]:
        if not self.masks_dir.exists():
            return []
        return sorted(self.masks_dir.glob("*.png"))
