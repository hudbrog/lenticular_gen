"""Pitch-test calibration image generator.

Real lenticular sheets drift from their marked LPI, and printer DPI isn't
exact (driver scaling, paper feed). If ``pitch_px = DPI / LPI`` is off by even
~0.5%, views bleed across lenses and you get ghosting / a muddy flip.

The fix is a **pitch test**: print a banded image where each band is
interlaced at a slightly different candidate LPI. Hold it under the actual
lens at the intended viewing distance; the band that flips cleanly between two
colors with no bleed is your sheet+printer's effective LPI. Feed that number
back as ``print.lpi_effective`` in ``config.yaml``.

Each band interlaces a **2-view** black/white pattern — not the real N views.
Pitch (``DPI/LPI``) is independent of view count, so a 2-view test calibrates
the same LPI as an N-view print, but each color column is ``pitch_px/2`` wide
(~4 px at 600 DPI/70 LPI) instead of ``pitch_px/N`` (~1 px for N=8). The
wider columns are renderable by laser printers; the 1px columns of a full
N-view test would toner-spread into solid gray.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .interlace import output_pixel_size

# The pitch test always uses a 2-view black/white pattern. LPI is independent
# of N, but 2 views gives ~4px columns (laser-safe) vs ~1px for N=8.
_PATTERN_N_VIEWS = 2

# Two highly-contrasting colors for the per-view test pattern. Black/white
# gives the clearest flip signal; saturated colors work too.
_PATTERN_COLORS = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)


def pitch_test_image(
    dpi: int,
    sheet_mm: tuple[float, float],
    lpi_nominal: float,
    step: float = 0.1,
    count: int = 7,
    band_mm: float | None = None,
) -> tuple[np.ndarray, int]:
    """Build a vertical stack of interlaced calibration bands.

    Args:
        dpi: Printer resolution.
        sheet_mm: Physical print size (full sheet; bands divide the height).
        lpi_nominal: Center candidate LPI (usually the sheet's marked LPI).
        step: LPI spacing between adjacent candidate bands.
        count: Number of candidate bands. Forced odd so one band sits exactly
            at ``lpi_nominal``. Ignored if ``band_mm`` is set.
        band_mm: If set, derive the band count from the sheet height divided by
            this band height (each band is exactly this tall in mm). Overrides
            ``count``. Useful for fitting many bands on a large sheet (e.g. A4)
            so you can slide a small lens around to compare candidates.

    Returns:
        ``(image, count)`` where image is ``(out_h, out_w, 3)`` uint8 and
        ``count`` is the actual number of bands (after forcing odd / deriving
        from ``band_mm``). The caller needs the count to print the range summary.
    """
    out_w, out_h = output_pixel_size(dpi, sheet_mm, _PATTERN_N_VIEWS)

    if band_mm is not None:
        if band_mm <= 0:
            raise ValueError("band_mm must be > 0")
        band_h_px = int(round(dpi * band_mm / 25.4))
        count = max(1, out_h // band_h_px)
    if count < 1:
        raise ValueError("count must be >= 1")
    if count % 2 == 0:
        count += 1  # force odd so the nominal LPI is centered

    half = count // 2
    lpis = [lpi_nominal + (i - half) * step for i in range(count)]

    band_h = out_h // count
    if band_h < 1:
        raise ValueError("sheet too small for this many bands at this DPI")

    # Build a full-width per-view pattern: view 0 black, view 1 white.
    # Each "view" is a solid-color slab; interlacing them at the right pitch
    # yields a clean flip; wrong pitches bleed into gray banding.
    pattern_views = [
        np.tile(_PATTERN_COLORS[i % 2], (band_h, out_w, 1)) for i in range(_PATTERN_N_VIEWS)
    ]

    canvas = Image.new("RGB", (out_w, out_h), (128, 128, 128))
    draw = ImageDraw.Draw(canvas)

    font: Any
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, band_h // 4))
    except OSError:
        font = ImageFont.load_default()

    # Label precision must reflect the step, otherwise adjacent bands share a
    # label (e.g. step=0.05 -> 69.65 and 69.70 both format to "69.7").
    label_dp = max(1, -int(np.floor(np.log10(step))) if step > 0 else 1)

    for i, lpi in enumerate(lpis):
        y0 = i * band_h
        # Interlace the pattern at this candidate LPI, then paste onto the band.
        band = _interlace_pattern(pattern_views, dpi, out_w, band_h, lpi)
        canvas.paste(Image.fromarray(band), (0, y0))

        # Label with the candidate LPI (top-left of each band).
        label = f"{lpi:.{label_dp}f} LPI"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = max(4, band_h // 16)
        # Background box for readability over any pattern.
        draw.rectangle([pad, y0 + pad, pad + tw + 8, y0 + pad + th + 4], fill=(200, 200, 200))
        draw.text((pad + 4, y0 + pad + 2), label, fill=(0, 0, 0), font=font)

    return np.asarray(canvas, dtype=np.uint8), count


def _interlace_pattern(
    views: list[np.ndarray],
    dpi: int,
    out_w: int,
    out_h: int,
    lpi: float,
) -> np.ndarray:
    """Interlace the solid-color pattern views at a candidate LPI.

    Views are full-width solid-color slabs, so column ``x`` of the output is
    simply the color of view ``v(x)`` — no source-column math needed.
    """
    n = len(views)
    pitch_px = float(dpi) / float(lpi)
    view_slot = pitch_px / n

    xs = np.arange(out_w, dtype=np.float64)
    lens = np.floor(xs / pitch_px)
    pos_in_lens = xs - lens * pitch_px
    v = np.floor(pos_in_lens / view_slot).astype(np.int64) % n

    # Gather solid colors (n, 3) -> (out_w, 3) -> broadcast down rows.
    palette = np.stack([views[i][0, 0] for i in range(n)])
    row = palette[v]  # (out_w, 3)
    return np.ascontiguousarray(np.broadcast_to(row[None, :, :], (out_h, out_w, 3)))
