# mac_Gameleon

Apple Silicon port of [Gameleon](https://github.com/gameleon2026/Gameleon): **geometry-meta** lossless point-cloud codec and **gsplat-mlx** Gaussian rendering. The CUDA reference implementation and model weights live in the sibling repo `../Gameleon/`.

## Requirements

- macOS on Apple Silicon (M series)
- Python ≥ 3.10 (Homebrew `python@3.12` recommended)
- Homebrew: `openblas`, `libomp`, `google-sparsehash`
- Sibling checkout of Gameleon with weights fetched (`git lfs pull`)

Workspace layout:

```text
mac_Gameleon/
  mac_Gameleon/    ← this repo
  Gameleon/        ← Gameleon source + weights/
```

Default test assets: `examples/0519/pcd_0.ply`, `examples/0519/0519.obj`.

## Setup

First-time install (creates `.venv`, installs dependencies, applies patches, builds MinkowskiEngine, prewarms torchac):

```bash
cd mac_Gameleon
chmod +x scripts/*.sh
./scripts/setup_env.sh
```

Activate the environment in each new shell session:

```bash
cd mac_Gameleon
source scripts/env_mac_cpu.sh
```

Re-run setup without rebuilding MinkowskiEngine:

```bash
SKIP_NATIVE=1 ./scripts/setup_env.sh
```

## Usage

### Full pipeline (recommended)

Step 1 attribute encode → Step 2 lossless geometry → Step 3 decode + PLY + bpp + render PSNR:

```bash
python test.py
```

Codec + bpp only, skip rendering:

```bash
python test.py --no-render
```

Common options:

```bash
python test.py --input examples/0519/pcd_0.ply --outdir outputs
python test.py --no-lattice          # disable mlx-lattice in Step 1
python test.py --step3-lattice       # enable mlx-lattice in Step 3 decode
```

### Step by step

```bash
# Step 1: attribute encode
python scripts/step1.py

# Step 2: lossless support geometry
python scripts/step2.py \
  --input outputs/orig_attribute/pcd_0_level_8_geom.ply \
  --sample-name pcd_0

# Step 3: attribute decode, PLY export, bpp, PSNR
python scripts/step3.py
python scripts/step3.py --no-render   # skip rendering
```

After the decoded Gaussian PLY is written, the log prints **编解码结束** (codec finished) with bpp, encode time, and decode time. Rendering and PSNR are reported afterward and are not included in codec timing.

PSNR uses 4 cardinal views (front / left / back / right) at 512×512. GT is mesh ray intersection; decoded views are gsplat-mlx splats.

### Render a 3DGS PLY standalone

```bash
python scripts/render_gaussian_ply.py \
  --ply outputs/render_level_8_seq/decoded_gaussians_seq.ply \
  --output outputs/render.png \
  --width 512 --height 512
```

## Outputs

Results are written under `outputs/` (gitignored):

```text
outputs/
  orig_attribute/          # Step 1: attribute bitstreams + support PLY
  geometry/                # Step 2: geometry bitstream + decoded support PLY
  render_level_8_seq/
    decoded_gaussians_seq.ply
    render/rgb_*.png       # decoded splat renders
  gt_render_mesh/rgb_*.png # mesh GT renders
  summary.json             # bpp, PSNR, and related metrics
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup_env.sh` | Environment setup |
| `scripts/env_mac_cpu.sh` | Activate venv and set `PYTHONPATH` |
| `test.py` | Step 1 + 2 + 3 end-to-end |
| `scripts/step1.py` | Attribute encode |
| `scripts/step2.py` | Lossless geometry encode/decode |
| `scripts/step3.py` | Attribute decode, metrics, PSNR |
| `scripts/render_gaussian_ply.py` | Standalone PLY → PNG render |
| `scripts/install_minkowski_cpu.sh` | MinkowskiEngine CPU build (used by setup) |

## Layout

```text
mac_Gameleon/
  test.py
  examples/0519/
  mac_gameleon/            # codec, render, and metrics library
  scripts/                 # entrypoint scripts
  vendor/
    gsplat-mlx/
    minkowskiengine/
  requirements.txt
```
