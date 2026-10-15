"""Lenticular interlacing: N parallax views -> one print-ready image.

A lenticular sheet is a grid of vertical cylindrical lenses, each ``pitch_px =
DPI / LPI`` printer pixels wide. Under each lens we stack N thin vertical
strips — one column-slice from each view — so the lens refracts a different
view's strip to the eye as the viewing angle changes.

The interlace is a two-step column gather:

1. **Compress** each view horizontally to ``out_w / N`` columns (the per-view
   strip width). Vertical extent is resampled to the print height. We use
   Lanczos here — sub-pixel resampling *within* a view is fine and desirable;
   only the *between-view* column gather is nearest-neighbor (step 2).
2. **Gather** columns: for each output column ``x`` we compute which view ``v``
   owns it and which compressed column ``cx`` of that view to copy, then do a
   single vectorized fancy-index into the scaled-view stack. No cross-view
   blending — each output column is one exact source column, which is what the
   lens geometry demands.

Output dimensions are ``round(DPI * sheet_mm / 25.4)``; width is rounded to the
nearest multiple of N so the view cycle tiles cleanly across the sheet.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MM_PER_INCH = 25.4


def output_pixel_size(dpi: int, sheet_mm: tuple[float, float], n_views: int) -> tuple[int, int]:
    """Print output size in pixels.

    Width is rounded to the nearest multiple of ``n_views`` so the per-view
    column cycle tiles cleanly across the sheet (the rounding shifts the
    physical width by at most ``n_views/2`` px — under 0.2 mm at 600 DPI).
    """
    w_mm, h_mm = sheet_mm
    out_w_raw = int(round(dpi * w_mm / MM_PER_INCH))
    out_h = int(round(dpi * h_mm / MM_PER_INCH))
    out_w = int(round(out_w_raw / n_views)) * n_views
    return out_w, out_h


def _scale_view(view: np.ndarray, out_h: int, compressed_w: int) -> np.ndarray:
    """Resize one view to ``(out_h, compressed_w)`` via Lanczos."""
    img = Image.fromarray(view.astype(np.uint8))
    resized = img.resize((compressed_w, out_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def _column_indices(
    out_w: int, pitch_px: float, n_views: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (view_index, compressed_col) for each output column.

    Under each lens (``pitch_px`` output columns wide) the N views appear in
    order, each occupying ``pitch_px / N`` columns. The compressed column
    indexes into the scaled view (width ``out_w / N``): lens ``g`` contributes
    a ``pitch_px / N``-wide block starting at compressed column
    ``round(g * pitch_px / N)``.
    """
    xs = np.arange(out_w, dtype=np.float64)
    view_slot = pitch_px / n_views  # output cols per view within one lens
    lens = np.floor(xs / pitch_px)  # which lens this column falls under
    pos_in_lens = xs - lens * pitch_px  # [0, pitch_px)
    v = np.floor(pos_in_lens / view_slot).astype(np.int64) % n_views
    # Compressed column = lens offset + within-slot offset.
    cx = np.floor(lens * view_slot + (pos_in_lens - v.astype(np.float64) * view_slot))
    cx = cx.astype(np.int64)
    return v, cx


def interlace(
    views: list[np.ndarray],
    dpi: int,
    sheet_mm: tuple[float, float],
    lpi: float,
    flip: bool = False,
) -> np.ndarray:
    """Interleave N parallax views into one print-ready image.

    Args:
        views: N views, each ``(H, W, 3)`` uint8. Need not share the source
            resolution — they are resampled to the print size.
        dpi: Printer resolution (dots per inch).
        sheet_mm: ``(width_mm, height_mm)`` of the physical print.
        lpi: Lens pitch (lenses per inch). Use the *measured* effective LPI
            (from ``lenticular pitchtest``) if available; otherwise the nominal
            sheet spec.
        flip: Reverse view order in the gather (consistent with ``render --flip``;
            use when a test print reads pseudoscopically).

    Returns:
        ``(out_h, out_w, 3)`` uint8 interlaced image.
    """
    n = len(views)
    if n < 1:
        raise ValueError("interlace requires at least one view")
    out_w, out_h = output_pixel_size(dpi, sheet_mm, n)

    pitch_px = float(dpi) / float(lpi)
    compressed_w = max(1, int(round(out_w / n)))

    scaled = np.stack([_scale_view(v, out_h, compressed_w) for v in views])
    if flip:
        scaled = scaled[::-1]

    v_idx, cx = _column_indices(out_w, pitch_px, n)
    cx = np.clip(cx, 0, compressed_w - 1)

    # Gather: out[x, h, c] = scaled[v_idx[x], h, cx[x], c] -> (out_w, out_h, 3)
    out = scaled[v_idx, :, cx]
    return np.ascontiguousarray(out.transpose(1, 0, 2))
