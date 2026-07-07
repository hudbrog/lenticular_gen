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


def _resolve_s_max(cfg: dict[str, Any], layerset: LayerSet) -> float:
    """Recover s_max from config or, failing that, the layerset's max shift."""
    views = cfg.get("views")
    if isinstance(views, dict):
        raw = views.get("s_max_px")
        if raw is not None:
            return float(raw)
    return float(max(abs(layer.shift_px_per_view) for layer in layerset))
