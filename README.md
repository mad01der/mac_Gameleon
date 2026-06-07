# mac_Gameleon

Apple Silicon port of [Gameleon](https://github.com/gameleon2026/Gameleon): **geometry-meta** lossless point-cloud codec and **gsplat-mlx** 3D Gaussian rendering.

| Path | Role |
|------|------|
| Geometry encode/decode | **mlx-lattice** (Metal sparse conv) + MinkowskiEngine (PLY load) |
| Rendering | **gsplat-mlx** (Metal 3DGS) |
| CUDA reference / weights | Sibling repo `../Gameleon/` |

## Requirements

- macOS on Apple Silicon (M series)
- Python ≥ 3.10 (Homebrew `python@3.12` recommended)
- Sibling checkout: `../Gameleon/gameleon/` with geometry checkpoints under `weights/geometry/`
- Homebrew: `openblas`, `libomp`, `google-sparsehash`

Expected workspace:

```text
mac_Gameleon/
  mac_Gameleon/    ← this repo
  Gameleon/        ← gameleon2026/Gameleon (git lfs pull for weights)
```

## Setup

```bash
cd mac_Gameleon
chmod +x scripts/*.sh
./scripts/setup_env.sh
source scripts/env_mac_cpu.sh
```

`setup_env.sh` creates `.venv`, installs PyTorch / Python deps / gsplat-mlx, patches the Gameleon checkout for geometry-meta, builds MinkowskiEngine CPU, and pre-compiles `torchac`.

Skip MinkowskiEngine rebuild on repeat runs:

```bash
SKIP_NATIVE=1 ./scripts/setup_env.sh
```

## Daily shell

```bash
cd mac_Gameleon
source scripts/env_mac_cpu.sh
```

## Geometry-meta encode/decode

Lossless UCM codec on the default test cloud (`examples/0519/pcd_0.ply`, ~562k voxel points):

```bash
python scripts/geometry_meta.py
```

Options:

```bash
python scripts/geometry_meta.py \
  --input examples/0519/pcd_0.ply \
  --ckpt ../Gameleon/gameleon/weights/geometry/gameleon_lossless_geometry/best_model_UCM.pt
```

Output bitstreams: `outputs/geometry_meta/` (e.g. `pcd_0.bin`).

Sparse convolutions run through **mlx-lattice** (`pip install mlx-lattice` or [utakotoba/mlx-lattice](https://github.com/utakotoba/mlx-lattice)). Native TorchSparse is not used.

## Render 3D Gaussian PLY (gsplat-mlx)

```bash
python scripts/render_gaussian_ply.py \
  --ply /path/to/your_3dgs.ply \
  --output outputs/render.png \
  --width 512 --height 512
```

Supports standard 3DGS PLY layout (`x,y,z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`), binary or ASCII.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/env_mac_cpu.sh` | `PYTHONPATH`, `GAMELEON_ROOT`, venv activate |
| `scripts/setup_env.sh` | One-time / idempotent full environment setup |
| `scripts/geometry_meta.py` | Geometry-meta lossless encode/decode |
| `scripts/render_gaussian_ply.py` | Metal 3DGS render to PNG |
| `scripts/install_minkowski_cpu.sh` | MinkowskiEngine v0.5.4 CPU build (called by setup) |

## Layout

```text
mac_Gameleon/
  examples/0519/              # default geometry test PLY + mesh
  outputs/geometry_meta/      # encoded bitstreams (gitignored)
  mac_gameleon/
    mlx_lattice_shim.py       # mlx-lattice backend for Gameleon sparse ops
    geometry_meta_patches.py  # Gameleon source patches (idempotent)
    minkowski_setup.py        # ME Mac build template
    prewarm_torchac.py        # avoid first-run torchac JIT hang
    render_gsplat.py          # gsplat-mlx render helpers
  scripts/                    # entrypoints (see table above)
  vendor/
    gsplat-mlx/               # Metal 3DGS (editable install)
    minkowskiengine/          # cloned + built by install_minkowski_cpu.sh
  requirements-mac-cpu.txt
```
