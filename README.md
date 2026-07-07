# lenticular-gen

Prepare a plain 2D cartoon image for **lenticular parallax printing** (4–12
views). It splits the image into depth layers, inpaints the background hidden
behind the character/foreground, and renders N horizontally-shifted views ready
for interlacing and printing.

Built for the **Fairy Door** product: a 70×125 mm door with a character scene
inside, where lenticular views give a parallax/3D effect as you move your head.

## Pipeline

```
source.png
   │  lenticular depth       depth.png           (Depth Anything V2 / Marigold)
   │  lenticular segment     masks/*.png         (Rembg auto + SAM2 interactive)
   │  lenticular assemble    layers.json         (depth-ordered layers + shifts)
   │  ── review masks & layers.json ──
   │  lenticular preview     preview.gif         (animated, holes flagged magenta)
   │  lenticular inpaint     bg_inpaint.png      (LaMa, full-bg-once)
   │  lenticular render      views/view_*.png    (N parallax views)
```

Each stage is a CLI subcommand that reads/writes artifacts in a project workdir,
so you can stop, review/edit, and resume at any point.

## Status

- **M0 — Scaffold**: project workdir, config, layer model, parallax math + tests. ✅
- **M1 — Depth + segment**: Depth Anything V2 / Marigold → `depth.png`; Rembg auto cutout + SAM2 interactive picker. ✅
- **M2 — Assemble + preview**: depth-ordered `layers.json` + animated parallax GIF/strip. ✅
- **M3 — Inpaint + render**: LaMa full-bg-once inpaint + N parallax view PNGs. ✅
- **M4 — Hardening**: Marigold backend, `--flip`, per-layer overrides. ✅

Lenticular interlacing (DPI/LPI → print-ready TIFF) and a PySide6 GUI are
deferred to later phases; config fields are reserved.

## Install

Core pipeline (math, IO, project management):

```bash
uv sync
```

With the ML model backends (torch CUDA, transformers, diffusers, rembg, LaMa)
on a GPU box:

```bash
uv sync --group ml
```

Requires an NVIDIA GPU (tested on RTX 3090, 24 GB). Each CLI command uses
`uv run --group ml lenticular <stage>`.

## Usage

```bash
# 1. Create a project workdir from a source image
lenticular init path/to/scene.png              # -> workdir/scene/

# 2. Estimate depth (dav2 fast / marigold finer detail)
lenticular depth   workdir/scene
lenticular depth   workdir/scene --model marigold

# 3. Segment: auto subject cutout + interactive SAM2 for extra layers
lenticular segment workdir/scene               # opens matplotlib picker
lenticular segment workdir/scene --auto-only   # character cutout only

# 4. Assemble layers (depth ordering + parallax shifts)
lenticular assemble workdir/scene              # -> layers.json

# 5. Preview parallax (no inpaint; holes shown magenta)
lenticular preview  workdir/scene              # -> preview.gif + preview_strip.png
lenticular preview  workdir/scene --flip       # reverse direction (pseudoscopic fix)

# 6. Inpaint hidden background (full-bg-once)
lenticular inpaint  workdir/scene              # LaMa (default, offline)
lenticular inpaint  workdir/scene --inpainter fal          # fal.ai FLUX Fill
lenticular inpaint  workdir/scene --inpainter replicate    # Replicate FLUX Fill
lenticular inpaint  workdir/scene --inpainter openrouter   # Nano Banana Pro (experimental)
lenticular inpaint  workdir/scene --inpainter fal --prompt "cozy forest interior"

# 7. Render N final views over the inpainted background
lenticular render   workdir/scene              # -> views/view_00..07.png
lenticular render   workdir/scene --flip
```

### Interactive SAM2 picker

`lenticular segment` (without `--auto-only`) opens a matplotlib window:

- **Type a layer name** at the terminal prompt (e.g. `foreground_frame`,
  `back_props`), or press Enter to finish.
- **Left-click** = foreground point (inside the object).
- **Right-click** = background point (outside, to refine).
- **`a`** = accept and save the mask.
- **`c`** = clear points and retry.
- **Esc** = cancel the current layer.

The `character` mask is produced automatically by Rembg; the picker is for
additional layers that add depth richness (frames, props, foreground objects).

## Configuration

Each project has its own `config.yaml` (copied from `config/default.yaml` on
`init`). Edit it to tune the pipeline per-image. Key options:

```yaml
views:
  count: 8            # N parallax views (4-12; 8 suits 40-60 LPI)
  s_max_px: null      # Max shift px between extremes. null => 4% of width.
  flip: false         # Reverse parallax (pseudoscopic fix).

layers:
  feather_px: 1       # Mask edge feather (anti-seam).
  focus_layer: null   # Zero-parallax anchor. null => character.
  overrides:          # Per-layer shift/pin overrides (applied on assemble):
    foreground_frame:
      shift_px_per_view: 12.0   # force a specific shift
    back_props:
      pinned: true              # pin to print plane (shift = 0)

depth:
  model: dav2        # dav2 (fast) | marigold (finer detail, slower)
  dav2_id: depth-anything/Depth-Anything-V2-Large-hf

segment:
  bg_removal: isnet-general-use   # Rembg model for the auto cutout.

inpaint:
  strategy: full-bg-once          # one run, cross-view consistent
  backend: lama                   # lama (default, offline) | fal | replicate | openrouter
  prompt: ""                      # background description for diffusion backends
  vertical_margin_px: 2
```

### Inpainting backends

| Backend | Mask | Speed | Cost | Quality | Setup |
|---|---|---|---|---|---|
| `lama` (default) | strict | ~0.1s | free | good (smooth bg) | none (offline) |
| `fal` | strict | ~5s | ~$0.05/img | excellent (FLUX Fill) | `export FAL_KEY=...` |
| `replicate` | strict | ~5s | ~$0.05/img | excellent (FLUX Fill) | `export REPLICATE_API_TOKEN=...` |
| `openrouter` | none | ~10s | ~$0.04/img | good (may drift) | `export OPENROUTER_API_KEY=...` |

`fal` and `replicate` use FLUX Fill with a strict binary mask (foreground
preserved pixel-exact) — best for textured/patterned backgrounds. `openrouter`
is experimental (no hard mask; foreground may drift). Use `--prompt` to guide
diffusion backends ("cozy forest interior with soft light").

## How it works

**Layering.** Depth estimation (DAV2 or Marigold) gives a per-pixel depth map.
Each mask's representative depth is the *median* over its region (robust to
per-pixel noise on flat cel-shaded art). Layers are ordered back-to-front, and
the character is auto-pinned as the zero-parallax anchor — it sits on the print
plane and stays crisp, while background shifts away and foreground shifts
toward the viewer.

**Parallax math.** Per-view shift of layer `i` at view `v`:
`shift_i(v) = s_max · (D_i − D_focus) · t_v`, where `t_v ∈ [−1, +1]` spans the
view range symmetrically. Background moves least (small `|D_i − D_focus|`),
foreground most. `--flip` negates the sign (fixes pseudoscopic test prints).

**Disocclusion inpainting.** Shifting foreground layers reveals background that
was previously occluded. We inpaint the **full background once** with LaMa —
masking the union of all foreground silhouettes dilated by ±`s_max` — so every
view composites over the same clean background (no per-view flicker).

## Notes & gotchas

- **Rembg `isnet-anime`** is the ideal anime-tuned model but its ONNX export in
  rembg 2.0.69 emits near-empty masks. We default to `isnet-general-use`;
  switch back via `config.yaml` once the export is fixed.
- **Depth on cartoons** is an *ordering prior*, not ground truth — DAV2
  over-smooths flat cel regions. The mask-median approach handles this; if
  layers mis-rank, fix the order in `layers.json` or pin depths via overrides.
- **Pseudoscopic prints** (depth reads inverted): re-render with `--flip`.
- **Vertical registration**: views shift horizontally only; any vertical drift
  reads as flicker. The pipeline enforces horizontal-only shifts.
- **Marigold** gives cleaner subject/background separation on some images but
  is slower (~11s vs ~5s for DAV2 on a 3090) and uses more VRAM.

## Dev

```bash
uv run --group ml pytest        # tests
uv run --group ml ruff check .  # lint
uv run --group ml mypy          # type check (strict)
```
