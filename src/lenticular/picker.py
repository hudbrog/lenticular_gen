"""Interactive matplotlib picker for SAM 2 segmentation.

Workflow inside the window:

* Type a layer name in the terminal prompt (e.g. ``foreground_frame``,
  ``back_props``) — or press Enter to finish.
* Left-click  = add a FOREGROUND point (inside the object).
* Right-click = add a BACKGROUND point (outside, to refine).
* The SAM mask updates live; press ``a`` to accept and save the mask, ``c`` to
  clear points and retry.
* Repeat for the next layer name.

The character cutout is produced automatically by Rembg beforehand; this picker
is for the *additional* layers (back props, foreground frame, ...).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# Input coordinate is (x, y) in image pixels; label 1 = fg, 0 = bg.
PointPredictFn = Callable[[np.ndarray, list[tuple[float, float]], list[int]], np.ndarray]


class InteractivePicker:
    """Matplotlib-backed point picker driving a SAM predictor.

    Args:
        rgb: source image ``(H, W, 3)`` uint8.
        predict_fn: callable ``(rgb, points, labels) -> mask (H, W) bool``.
        on_save: callable ``(name, mask) -> None`` invoked when the user
            accepts a mask. Typically writes the mask PNG to the project.
    """

    FG = 1
    BG = 0

    def __init__(
        self,
        rgb: np.ndarray,
        predict_fn: PointPredictFn,
        on_save: Callable[[str, np.ndarray], None],
    ) -> None:
        self.rgb = rgb
        self.predict_fn = predict_fn
        self.on_save = on_save

        self.points: list[tuple[float, float]] = []
        self.labels: list[int] = []
        self.current_mask: np.ndarray | None = None
        self.layer_name: str | None = None

        # Lazily created matplotlib state (kept out of module import time).
        # Typed Any to avoid importing matplotlib artists at module scope.
        self._fig: Any = None
        self._ax: Any = None
        self._img_artist: Any = None
        self._mask_artist: Any = None
        self._scatter_artist: Any = None

    # --- matplotlib plumbing -------------------------------------------------
    def _setup_figure(self) -> None:  # pragma: no cover - UI
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        self._fig = fig
        self._ax = ax
        ax.set_title("Layer segmentation — see terminal for prompts")
        self._img_artist = ax.imshow(self.rgb)
        # Overlay artists (updated in place).
        transparent = np.zeros((*self.rgb.shape[:2], 4), dtype=np.uint8)
        self._mask_artist = ax.imshow(transparent, alpha=0.5)
        self._scatter_artist = None

        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show(block=False)

    def _on_click(self, event: Any) -> None:  # pragma: no cover - UI
        if event.xdata is None or self.layer_name is None:
            return
        label = self.FG if event.button == 1 else self.BG
        self.points.append((float(event.xdata), float(event.ydata)))
        self.labels.append(label)
        self._update_mask()
        self._draw_points()

    def _on_key(self, event: Any) -> None:  # pragma: no cover - UI
        if event.key == "a":
            self._accept()
        elif event.key == "c":
            self._clear_points()
        elif event.key == "escape":
            self.layer_name = None

    def _update_mask(self) -> None:  # pragma: no cover - UI
        if len(self.points) < 1 or self.layer_name is None or self._mask_artist is None:
            return
        self.current_mask = self.predict_fn(self.rgb, list(self.points), list(self.labels))
        overlay = np.zeros((*self.current_mask.shape, 4), dtype=np.uint8)
        overlay[self.current_mask] = (0, 255, 0, 120)  # green tint
        self._mask_artist.set_data(overlay)
        assert self._fig is not None
        self._fig.canvas.draw_idle()

    def _draw_points(self) -> None:  # pragma: no cover - UI
        assert self._fig is not None and self._ax is not None
        if self._scatter_artist is not None:
            self._scatter_artist.remove()
            self._scatter_artist = None
        if not self.points:
            self._fig.canvas.draw_idle()
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        colors = ["lime" if lbl == self.FG else "red" for lbl in self.labels]
        self._scatter_artist = self._ax.scatter(xs, ys, c=colors, s=40, edgecolors="black")
        self._fig.canvas.draw_idle()

    def _clear_points(self) -> None:  # pragma: no cover - UI
        self.points.clear()
        self.labels.clear()
        self.current_mask = None
        if self._mask_artist is not None:
            transparent = np.zeros((*self.rgb.shape[:2], 4), dtype=np.uint8)
            self._mask_artist.set_data(transparent)
        self._draw_points()

    def _accept(self) -> None:  # pragma: no cover - UI
        if self.current_mask is None or self.layer_name is None:
            return
        self.on_save(self.layer_name, self.current_mask)
        self.layer_name = None
        self._clear_points()

    # --- main loop -----------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - UI
        """Block on a terminal prompt loop, refreshing the figure each round."""
        import matplotlib.pyplot as plt

        self._setup_figure()
        try:
            while True:
                name = input(
                    "\nLayer name (Enter to finish) > "
                ).strip()
                if not name:
                    break
                self.layer_name = name
                self._clear_points()
                print(
                    f"  Click FG (left) / BG (right) points for '{name}'. "
                    "a=accept, c=clear, Esc=cancel."
                )
                # Spin until this layer is accepted or cancelled.
                while self.layer_name is not None:
                    plt.pause(0.1)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            plt.close(self._fig)
