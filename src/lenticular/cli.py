"""CLI entry point.

Each pipeline stage is a subcommand operating on a project workdir. Stages not
yet implemented (M2+) print a message and exit non-zero when invoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import typer

from .io_utils import load_rgb, save_depth, save_mask, save_rgb
from .layers import CHARACTER_LAYER_NAME, LayerSet
from .project import Project

app = typer.Typer(
    name="lenticular",
    help="Prepare 2D cartoon images for lenticular parallax printing.",
    no_args_is_help=True,
    add_completion=False,
)


def _resolve_project(project: Path) -> Project:
    if not project.exists():
        raise typer.BadParameter(f"Project not found: {project}")
    return Project(path=project)


@app.command()
def init(
    image: Path = typer.Argument(..., help="Source 2D image to prepare."),  # noqa: B008
    workdir: Path = typer.Option(Path("./workdir"), help="Parent workdir."),  # noqa: B008
    name: str | None = typer.Option(None, help="Project name (defaults to image stem)."),  # noqa: B008
) -> None:
    """Create a new project workdir from a source image."""
    if not image.is_file():
        raise typer.BadParameter(f"Source image not found: {image}")
    proj = Project.init(image, workdir, name=name)
    typer.echo(f"Created project: {proj.path}")


@app.command()
def depth(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    model: str = typer.Option(  # noqa: B008
        None,
        "--model",
        help="Depth backend: 'dav2' (default, fast) or 'marigold' (finer detail).",
    ),
) -> None:
    """Estimate depth -> depth.png + depth_preview.png.

    Backend defaults to config `depth.model` (dav2); override with --model.
    """
    proj = _resolve_project(project)
    cfg = proj.load_config()
    rgb = load_rgb(proj.source)
    backend = model or cfg["depth"]["model"]
    model_id = cfg["depth"].get(f"{backend}_id") if backend != "dav2" else cfg["depth"]["dav2_id"]

    typer.echo(f"Running depth backend '{backend}' ({model_id})...")
    from .models.depth import estimate_depth

    result = estimate_depth(rgb, model=backend, model_id=model_id)
    save_depth(result.depth, proj.depth)
    save_rgb(result.preview, proj.depth_preview)
    typer.echo(f"Wrote {proj.depth} and {proj.depth_preview}")


@app.command()
def segment(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    interactive: bool = typer.Option(  # noqa: B008
        True, "--interactive/--auto-only", help="Open the SAM2 picker for extra layers."
    ),
) -> None:
    """Auto subject cutout (Rembg isnet-anime) + optional SAM2 masking."""
    proj = _resolve_project(project)
    cfg = proj.load_config()
    rgb = load_rgb(proj.source)

    # 1) Auto subject cutout -> masks/character.png
    typer.echo(f"Auto subject cutout (Rembg {cfg['segment']['bg_removal']})...")
    from .models.segment import auto_subject_mask

    character_mask = auto_subject_mask(rgb, model=cfg["segment"]["bg_removal"])
    save_mask(character_mask, proj.mask(CHARACTER_LAYER_NAME))
    typer.echo(f"  -> {proj.mask(CHARACTER_LAYER_NAME)}")

    if not interactive:
        return

    # 2) Interactive SAM2 picker for additional layers (back props, frame, ...).
    typer.echo("Opening interactive SAM2 picker for additional layers...")
    from .models.segment import SAM2_MODEL_ID_DEFAULT, Sam2Segmenter
    from .picker import InteractivePicker

    segmenter = Sam2Segmenter(model_id=SAM2_MODEL_ID_DEFAULT)

    def on_save(name: str, mask: np.ndarray) -> None:
        save_mask(mask, proj.mask(name))
        typer.echo(f"  -> {proj.mask(name)}")

    picker = InteractivePicker(
        rgb=rgb,
        predict_fn=lambda im, pts, lbls: segmenter.predict(im, pts, lbls),
        on_save=on_save,
    )
    picker.run()
    typer.echo("Segmentation complete.")


@app.command()
def cluster(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    k: int = typer.Option(5, "--k", help="Number of depth clusters (4-8 recommended)."),  # noqa: B008
    open_radius: int = typer.Option(  # noqa: B008
        1, "--open-radius", help="Morphological opening radius (px) for edge cleanup."
    ),
    min_component_fraction: float = typer.Option(  # noqa: B008
        0.02,
        "--min-component-fraction",
        help="Drop connected components smaller than this fraction of the cluster.",
    ),
) -> None:
    """Split depth into K clustered layers -> masks/layer_0..K-1.png (far→near).

    A fast, fully-automatic alternative to the SAM2 picker when the depth map
    has distinct modes (typical for composed cartoon scenes). Use the picker to
    refine specific edges afterward if needed.
    """
    proj = _resolve_project(project)
    if not proj.depth.exists():
        raise typer.BadParameter("depth.png not found. Run `lenticular depth` first.")

    from .io_utils import load_depth
    from .pipeline.cluster import cluster_depth

    depth = load_depth(proj.depth)
    result = cluster_depth(
        depth,
        k=k,
        open_radius=open_radius,
        min_component_fraction=min_component_fraction,
    )

    # Remove any stale cluster masks so a re-run doesn't mix old/new layer counts.
    for old in proj.masks_dir.glob("layer_*.png"):
        old.unlink()

    for i, mask in enumerate(result.masks):
        out = proj.mask(f"layer_{i}")
        save_mask(mask, out)
        frac = mask.mean()
        typer.echo(
            f"  layer_{i}: center={result.centers[i]:.3f} "
            f"depth range far→near, {100 * frac:5.1f}% of frame"
        )

    # Colorized cluster map for inspection (far=blue → near=red).
    import numpy as _np

    cmap = _np.array(
        [[30, 60, 180], [60, 180, 220], [100, 220, 120], [255, 200, 40], [220, 60, 40]],
        dtype=_np.uint8,
    )
    # Extend palette if k > 5 by interpolating.
    if k > len(cmap):
        idx = _np.linspace(0, len(cmap) - 1, k)
        cmap = _np.asarray(
            _np.stack(
                [_np.interp(idx, _np.arange(len(cmap)), cmap[:, c]) for c in range(3)], axis=1
            )
        ).astype(_np.uint8)
    vis = cmap[result.labels]
    save_rgb(vis, proj.path / f"depth_clusters_k{k}.png")
    typer.echo(f"Wrote {k} cluster masks to {proj.masks_dir}/ and depth_clusters_k{k}.png")


@app.command()
def assemble(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
) -> None:
    """Order masks by depth into layers.json (auto focus = character)."""
    proj = _resolve_project(project)
    cfg = proj.load_config()
    rgb = load_rgb(proj.source)

    from .pipeline.separate import assemble_layers

    focus_override = cfg.get("layers", {}).get("focus_layer") or None
    s_max_cfg = cfg.get("views", {}).get("s_max_px")
    layers_cfg = cfg.get("layers", {}) if isinstance(cfg.get("layers"), dict) else {}
    overrides = layers_cfg.get("overrides") or None
    layerset = assemble_layers(
        masks_dir=proj.masks_dir,
        depth_path=proj.depth,
        image_shape=rgb.shape[:2],
        focus_layer_name=focus_override,
        s_max=int(s_max_cfg) if s_max_cfg else None,
        overrides=overrides,
    )
    proj.save_layers(layerset)
    typer.echo(f"Wrote {proj.layers}")
    for layer in layerset:
        typer.echo(
            f"  {'*' if layer.pinned else ' '} {layer.name:<20} "
            f"depth={layer.depth:.3f} shift/view={layer.shift_px_per_view:+.1f}px"
        )
    typer.echo(f"  focus: {layerset.focus.name}")


@app.command()
def preview(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    flip: bool = typer.Option(  # noqa: B008
        None,
        "--flip/--no-flip",
        help="Reverse parallax direction (overrides config; fixes pseudoscopic prints).",
    ),
) -> None:
    """Render an animated parallax GIF + strip (holes flagged magenta, no inpaint)."""
    proj = _resolve_project(project)
    cfg = proj.load_config()
    rgb = load_rgb(proj.source)

    if not proj.layers.exists():
        raise typer.BadParameter(
            "layers.json not found. Run `lenticular assemble` first."
        )

    from .pipeline.render import render_all_views
    from .pipeline.separate import load_effective_masks, resolve_overlaps
    from .preview import make_gif, make_strip

    layerset = proj.load_layers()
    n_views = int(cfg["views"]["count"])
    s_max = _resolve_s_max(cfg, layerset)
    flip = bool(cfg["views"].get("flip", False)) if flip is None else flip
    feather_px = int(cfg["layers"].get("feather_px", 1))

    raw_masks = load_effective_masks(proj.masks_dir, rgb.shape[:2])
    back_to_front = [layer.name for layer in layerset]
    masks = resolve_overlaps(raw_masks, back_to_front)

    views = render_all_views(
        layerset,
        rgb,
        masks,
        n_views,
        s_max,
        bg_rgb=rgb,
        feather_px=feather_px,
        flip=flip,
        flag_holes=True,
    )
    make_gif(views, proj.preview_gif)
    make_strip(views, proj.preview_strip)
    typer.echo(f"Wrote {proj.preview_gif} and {proj.preview_strip}")


@app.command()
def inpaint(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    inpainter: str = typer.Option(  # noqa: B008
        None,
        "--inpainter",
        help="Inpaint backend: lama (default, offline) | fal | replicate | openrouter.",
    ),
    prompt: str = typer.Option(  # noqa: B008
        None,
        "--prompt",
        help="Background description for diffusion backends (empty = context completion).",
    ),
) -> None:
    """Inpaint hidden background (full-bg-once) -> bg_inpaint.png."""
    proj = _resolve_project(project)
    cfg = proj.load_config()
    rgb = load_rgb(proj.source)

    if not proj.layers.exists():
        raise typer.BadParameter("layers.json not found. Run `lenticular assemble` first.")

    from .pipeline.separate import (
        build_inpaint_mask,
        load_effective_masks,
        resolve_overlaps,
    )

    layerset = proj.load_layers()
    s_max = _resolve_s_max(cfg, layerset)

    raw_masks = load_effective_masks(proj.masks_dir, rgb.shape[:2])
    masks = resolve_overlaps(raw_masks, [layer.name for layer in layerset])

    inpaint_mask = build_inpaint_mask(
        layerset,
        masks,
        s_max,
        vertical_margin_px=int(cfg["inpaint"].get("vertical_margin_px", 2)),
    )
    fg_pixels = int(inpaint_mask.mean() * rgb.shape[0] * rgb.shape[1])

    inpaint_cfg = cfg["inpaint"] if isinstance(cfg.get("inpaint"), dict) else {}
    backend = inpainter or inpaint_cfg.get("backend", "lama")
    prompt = prompt if prompt is not None else inpaint_cfg.get("prompt", "")
    # Resolve per-backend model ID from config.
    model_id = inpaint_cfg.get(f"{backend}_model") if backend != "lama" else None

    typer.echo(
        f"Inpainting {100 * inpaint_mask.mean():.1f}% of frame "
        f"({fg_pixels} px) with {backend} (full-bg-once)..."
    )
    if backend == "openrouter":
        typer.echo("  WARNING: openrouter has no strict mask; foreground may drift.")

    from .models.inpaint import get_inpainter

    inpainter_obj = get_inpainter(backend, model=model_id)
    bg = inpainter_obj.inpaint(rgb, inpaint_mask, prompt=prompt)
    save_rgb(bg, proj.bg_inpaint)
    typer.echo(f"Wrote {proj.bg_inpaint}")


@app.command()
def render(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    flip: bool = typer.Option(  # noqa: B008
        None,
        "--flip/--no-flip",
        help="Reverse parallax direction (overrides config; fixes pseudoscopic prints).",
    ),
) -> None:
    """Render N parallax views over the inpainted background -> views/."""
    proj = _resolve_project(project)
    cfg = proj.load_config()
    rgb = load_rgb(proj.source)

    if not proj.layers.exists():
        raise typer.BadParameter("layers.json not found. Run `lenticular assemble` first.")
    if not proj.bg_inpaint.exists():
        raise typer.BadParameter(
            "bg_inpaint.png not found. Run `lenticular inpaint` first."
        )

    from .pipeline.render import render_all_views
    from .pipeline.separate import load_effective_masks, resolve_overlaps

    layerset = proj.load_layers()
    n_views = int(cfg["views"]["count"])
    s_max = _resolve_s_max(cfg, layerset)
    flip = bool(cfg["views"].get("flip", False)) if flip is None else flip
    feather_px = int(cfg["layers"].get("feather_px", 1))

    raw_masks = load_effective_masks(proj.masks_dir, rgb.shape[:2])
    masks = resolve_overlaps(raw_masks, [layer.name for layer in layerset])
    bg = load_rgb(proj.bg_inpaint)

    views = render_all_views(
        layerset,
        rgb,
        masks,
        n_views,
        s_max,
        bg_rgb=bg,
        feather_px=feather_px,
        flip=flip,
        flag_holes=False,
    )

    # Clear stale views, then write fresh ones.
    for old in proj.views_dir.glob("view_*.png"):
        old.unlink()
    import json

    shift_table = []
    for v, view in enumerate(views):
        out = proj.views_dir / f"view_{v:02d}.png"
        save_rgb(view, out)
        shift_table.append({"view": v, "file": out.name})
    proj.render_meta.write_text(
        json.dumps(
            {
                "n_views": n_views,
                "s_max": s_max,
                "flip": flip,
                "focus": layerset.focus.name,
                "views": shift_table,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {n_views} views to {proj.views_dir}")


@app.command()
def interlace(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    lpi: float = typer.Option(  # noqa: B008
        None,
        "--lpi",
        help="Lens LPI. Overrides config (use lpi_effective if calibrated).",
    ),
    dpi: int = typer.Option(None, "--dpi", help="Printer DPI. Overrides config."),  # noqa: B008
    flip: bool = typer.Option(  # noqa: B008
        None,
        "--flip/--no-flip",
        help="Reverse view order (overrides render_meta / config).",
    ),
) -> None:
    """Interleave N parallax views into one print-ready TIFF -> interlaced.tif.

    Reads views from ``views/`` and column-gathers them at ``dpi / lpi`` columns
    per lens. Use ``lenticular pitchtest`` first to measure the effective LPI
    of your sheet+printer; feed it back as ``print.lpi_effective`` in config.
    """
    proj = _resolve_project(project)
    cfg = proj.load_config()

    if not proj.render_meta.exists():
        raise typer.BadParameter(
            "render_meta.json not found. Run `lenticular render` first."
        )

    import json

    from .pipeline.interlace import interlace as interlace_fn

    meta = json.loads(proj.render_meta.read_text(encoding="utf-8"))
    n_views = int(meta["n_views"])

    view_files = sorted(proj.views_dir.glob("view_*.png"))
    if len(view_files) != n_views:
        raise typer.BadParameter(
            f"Expected {n_views} views in {proj.views_dir}, found {len(view_files)}."
        )

    print_cfg = cfg["print"] if isinstance(cfg.get("print"), dict) else {}
    dpi_val = int(dpi or print_cfg.get("dpi", 600))
    lpi_eff = print_cfg.get("lpi_effective")
    lpi_val = float(lpi if lpi is not None else (lpi_eff or print_cfg.get("lpi", 60)))
    sheet_mm = tuple(print_cfg.get("sheet_mm", [70, 125]))
    flip_val = bool(flip) if flip is not None else bool(meta.get("flip", False))

    views = [load_rgb(f) for f in view_files]
    out = interlace_fn(views, dpi=dpi_val, sheet_mm=sheet_mm, lpi=lpi_val, flip=flip_val)

    _save_tiff(out, proj.interlaced, dpi=dpi_val)
    pitch_px = dpi_val / lpi_val
    view_slot_px = pitch_px / n_views
    typer.echo(
        f"Interlaced {n_views} views -> {proj.interlaced} "
        f"({out.shape[1]}x{out.shape[0]} px, "
        f"{sheet_mm[0]:.0f}x{sheet_mm[1]:.0f} mm @ {dpi_val} DPI, "
        f"pitch={pitch_px:.3f} px/lens, LPI={lpi_val:.2f})"
    )
    # Feasibility: per-view column width below ~2px toner-spreads / ink-spreads
    # into adjacent views -> ghosting. Lasers are risky below ~2px; inkjets
    # hold down to ~1px. Surface this so the user doesn't waste a print.
    if view_slot_px < 2.0:
        typer.echo(
            f"  WARNING: each view is only {view_slot_px:.2f} px wide "
            f"(pitch {pitch_px:.2f} / {n_views} views). Below ~2px, laser toner "
            f"spreads across views -> ghosting. Use a higher DPI, fewer views, "
            f"or lower LPI for clean laser output; inkjets handle ~1px fine."
        )


@app.command()
def pitchtest(
    project: Path = typer.Argument(..., help="Project workdir."),  # noqa: B008
    lpi: float = typer.Option(  # noqa: B008
        None, "--lpi", help="Center candidate LPI (defaults to config print.lpi)."
    ),
    step: float = typer.Option(  # noqa: B008
        None, "--step", help="LPI spacing between bands (defaults to config)."
    ),
    count: int = typer.Option(  # noqa: B008
        None, "--count", help="Number of bands, odd (defaults to config). Ignored if --band-mm."
    ),
    band_mm: float = typer.Option(  # noqa: B008
        None,
        "--band-mm",
        help="Band height in mm. Derives count from sheet height; overrides --count. "
        "Use for fitting many bands on a large test sheet (e.g. A4).",
    ),
    sheet_mm: str = typer.Option(  # noqa: B008
        None,
        "--sheet-mm",
        help="Test sheet size 'WxH' mm (defaults to A4: 210x297). "
        "Independent of config print.sheet_mm — calibrate on any size sheet.",
    ),
    dpi: int = typer.Option(None, "--dpi", help="Printer DPI. Overrides config."),  # noqa: B008
) -> None:
    """Generate a banded LPI calibration print -> pitch_test.tif.

    Each band is interlaced at a candidate LPI near the nominal value. Print
    it, hold it under the lens, and read off the band that flips cleanly — that
    is your sheet's effective LPI. Set ``print.lpi_effective`` in config.yaml
    and re-run ``lenticular interlace``.

    Uses a 2-view black/white pattern (not the real N views): LPI is
    independent of view count, but 2 views gives ~4px color columns that any
    printer can render, vs the ~1px columns an N=8 test would need.

    The test sheet defaults to **A4** (210x297 mm), not the product size — a
    bigger sheet fits more bands so you can compare candidates side by side by
    sliding the lens around. Use ``--band-mm`` to control band height (count is
    then derived from the sheet height); or ``--count`` for a fixed count.
    """
    proj = _resolve_project(project)
    cfg = proj.load_config()

    from .pipeline.pitchtest import pitch_test_image

    print_cfg = cfg["print"] if isinstance(cfg.get("print"), dict) else {}
    dpi_val = int(dpi or print_cfg.get("dpi", 600))
    lpi_val = float(lpi if lpi is not None else print_cfg.get("lpi", 60))
    # The test sheet defaults to A4 — bigger than the product, so more bands
    # fit and you can slide a small lens around to compare candidates.
    if sheet_mm is not None:
        parts = sheet_mm.lower().split("x")
        if len(parts) != 2:
            raise typer.BadParameter("--sheet-mm must be 'WxH' e.g. '210x297'.")
        sheet = (float(parts[0]), float(parts[1]))
    else:
        sheet = (210.0, 297.0)  # A4
    pt_cfg = print_cfg.get("pitchtest", {}) if isinstance(print_cfg.get("pitchtest"), dict) else {}
    step_val = float(step if step is not None else pt_cfg.get("step", 0.1))
    count_val = int(count if count is not None else pt_cfg.get("count", 7))

    out, actual_count = pitch_test_image(
        dpi=dpi_val,
        sheet_mm=sheet,
        lpi_nominal=lpi_val,
        step=step_val,
        count=count_val,
        band_mm=band_mm,
    )
    _save_tiff(out, proj.pitch_test, dpi=dpi_val)
    lo = lpi_val - (actual_count // 2) * step_val
    hi = lpi_val + (actual_count // 2) * step_val
    band_h_mm = sheet[1] / actual_count
    typer.echo(
        f"Wrote {proj.pitch_test} ({actual_count} bands @ {band_h_mm:.1f} mm each, "
        f"LPI {lo:.2f}..{hi:.2f}, {out.shape[1]}x{out.shape[0]} px, "
        f"{sheet[0]:.0f}x{sheet[1]:.0f} mm @ {dpi_val} DPI)"
    )


def _save_tiff(arr: np.ndarray, path: Path, dpi: int) -> None:
    """Save an RGB array as TIFF with physical DPI metadata."""
    from PIL import Image

    Image.fromarray(arr.astype(np.uint8)).save(
        path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw"
    )


def _resolve_s_max(cfg: dict[str, Any], layerset: LayerSet) -> float:
    """Recover s_max from config or, failing that, the layerset's max shift."""
    views = cfg.get("views")
    if isinstance(views, dict):
        raw = views.get("s_max_px")
        if raw is not None:
            return float(raw)
    return float(max(abs(layer.shift_px_per_view) for layer in layerset))
