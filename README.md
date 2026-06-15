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

### mlx-lattice + mlx_gameleon workspace (next-gen codec)

Experimental Gameleon geometry/attribute codec on the latest [mlx-lattice](https://github.com/utakotoba/mlx-lattice) (requires macOS 26+, **full Xcode** for Metal shader build, no pip wheel). This is separate from the current `test.py` pipeline until integration is complete.

```bash
chmod +x scripts/setup_mlx_lattice_workspace.sh
./scripts/setup_mlx_lattice_workspace.sh
```

If setup fails with `cannot execute tool 'metal' due to missing Metal Toolchain`, install the component:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Or in Xcode: **Settings → Components → Metal Toolchain**. Then re-run `./scripts/setup_mlx_lattice_workspace.sh`.

If setup fails with `unable to find utility "metal"`, install **Xcode** from the App Store (Command Line Tools alone are not enough), then run `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.

Benchmark after setup:

```bash
cd vendor/mlx-lattice
uv run mlx-gameleon-bench --points 100000 --warmup 1 --iters 3
uv run python -c "from mlx_lattice.ops import normalized_cdf, range_decode, range_encode; print('entropy ops OK')"
```

Source for `mlx_gameleon` lives in `vendor/mlx_gameleon/` and is synced into `vendor/mlx-lattice/mlx_gameleon/` by the setup script.

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

After the decoded Gaussian PLY is written, the log prints **编解码结束** (codec finished) with bpp and four codec timings: `attribute_encode_sec`, `geometry_encode_sec`, `attribute_decode_sec`, and `geometry_decode_sec`. Rendering and PSNR are reported afterward and are not included in codec timing.

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
| `scripts/setup_mlx_lattice_workspace.sh` | Clone/build mlx-lattice + mlx_gameleon uv workspace |

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
    mlx_gameleon/          # Gameleon codec on mlx-lattice (synced into mlx-lattice repo)
    mlx-lattice/           # cloned by setup_mlx_lattice_workspace.sh (gitignored)
  requirements.txt
```
