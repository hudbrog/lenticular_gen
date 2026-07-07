"""Layer model and JSON serialization.

A :class:`Layer` is one depth slab of the scene (background, back props,
character, foreground frame, ...). A :class:`LayerSet` is the ordered,
back-to-front collection that drives parallax rendering and inpainting.

The layer order (``LayerSet.layers``) is always **back-to-front**: index 0 is
the deepest (far background), the last index is the nearest (foreground).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Reserved layer name for the auto-generated subject cutout (Rembg isnet-anime).
CHARACTER_LAYER_NAME = "character"
# Reserved layer name for the residual background plane (everything else).
BACKGROUND_LAYER_NAME = "background"


@dataclass
class Layer:
    """One depth slab.

    Attributes:
        name: Stable identifier; also the mask filename stem (``<name>.png``).
        depth: Normalized depth in [0, 1]. 0 = far, 1 = near.
        pinned: If True, the layer keeps this exact per-view shift and ignores
            the depth-derived value (used to force the foreground frame forward,
            or pin the character to the zero-parallax plane).
        shift_px_per_view: Computed per-view step (px) at the extreme view
            (``t_v = +1``). Filled by assemble/render; negative shifts move a
            layer left across views, positive moves it right. The shift for an
            arbitrary view ``v`` is ``shift_px_per_view * t_v``.
    """

    name: str
    depth: float
    pinned: bool = False
    shift_px_per_view: float = 0.0

    @property
    def mask_filename(self) -> str:
        return f"{self.name}.png"


@dataclass
class LayerSet:
    """Ordered (back-to-front) collection of layers."""

    layers: list[Layer] = field(default_factory=list)
    # Index into ``layers`` of the zero-parallax anchor (the print-plane layer).
    focus_index: int = 0

    def __iter__(self) -> Any:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]

    @property
    def focus(self) -> Layer:
        return self.layers[self.focus_index]

    def by_name(self, name: str) -> Layer | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_index": self.focus_index,
            "layers": [asdict(layer) for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerSet:
        layers = [Layer(**layer) for layer in data["layers"]]
        return cls(layers=layers, focus_index=int(data.get("focus_index", 0)))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LayerSet:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
