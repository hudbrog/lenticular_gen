"""Parallax shift math and view parameterization.

Conventions
-----------
* Views are indexed ``v in [0, N-1]``. The center view (``t_v = 0``) is the
  zero-parallax view: the anchor layer sits exactly on the print plane and every
  other layer sits at its center-view position.
* ``t_v = 2*v/(N-1) - 1`` maps the view index to ``[-1, +1]``. For odd N the
  center view maps to 0; for even N the two middle views are symmetric about 0.
* Per-view shift of a layer (px), positive = move right as the viewer moves
  right::

      shift_i(v) = s_max * (D_i - D_focus) * t_v

  ``D_focus`` is the depth of the anchor (zero-parallax) layer; layers deeper
  than it (``D < D_focus``) shift opposite to layers in front of it
  (``D > D_focus``), giving correct pop-out around the anchored subject.

Disocclusion: the maximum horizontal reveal for a layer across the full view
range equals ``|shift_i(v)|`` at the extreme views, i.e.
``abs(s_max * (D_i - D_focus))``. This drives the inpaint mask dilation.
"""

from __future__ import annotations

import numpy as np


def view_t(v: int, n_views: int) -> float:
    """Normalized view position in ``[-1, +1]``."""
    if n_views < 2:
        # Single view: trivially the center.
        return 0.0
    return 2.0 * v / (n_views - 1) - 1.0


def view_params(n_views: int) -> np.ndarray:
    """Return the ``t_v`` array for all views, shape ``(n_views,)``."""
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    return np.array([view_t(v, n_views) for v in range(n_views)], dtype=np.float64)


def layer_shift(
    depth: float,
    focus_depth: float,
    s_max: float,
    n_views: int,
    flip: bool = False,
) -> np.ndarray:
    """Per-view horizontal shift (px) for one layer.

    Returns an array of shape ``(n_views,)``. With ``flip=True`` the sign is
    negated — use this when a test print reads pseudoscopically (depth inverted).
    """
    ts = view_params(n_views)
    sign = -1.0 if flip else 1.0
    return sign * s_max * (depth - focus_depth) * ts


def max_disocclusion_px(depth: float, focus_depth: float, s_max: float) -> float:
    """Largest horizontal reveal (px) for a layer over the full view range.

    This is the per-side band that must be present in the inpainted background.
    """
    return abs(s_max * (depth - focus_depth))


def default_s_max(width: int) -> int:
    """Default ``s_max`` (~4% of image width), rounded to an integer."""
    return max(1, round(0.04 * width))
