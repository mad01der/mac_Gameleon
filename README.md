# mac_Gameleon

Mac-side work for Gameleon: **gsplat-mlx** rendering (Metal) and **geometry UCM** on CPU (TorchSparse + MinkowskiEngine).

CUDA reference code stays in `../Gameleon/`. Weights live under `../Gameleon/gameleon/weights/`.

## Requirements

- macOS on Apple Silicon (M series)
- **Python >= 3.10** (Homebrew `python@3.12` recommended)
- Sibling repo: `../Gameleon/` with geometry/attribute checkpoints
- Homebrew: `openblas`, `libomp`, `google-sparsehash`

## One-time setup

```bash
cd mac_Gameleon
chmod +x scripts/*.sh
./scripts/setup_env.sh
source scripts/env_mac_cpu.sh
python scripts/verify_phase0_env.py
```

Re-run without rebuilding TorchSparse / ME:

```bash
SKIP_NATIVE=1 ./scripts/setup_env.sh
```

Default test data: `examples/0519/` (`pcd_0.ply` ~562k voxel coords + `0519.obj`). Use `--max-points` for CPU smoke tests.

## Daily shell

```bash
cd mac_Gameleon
source scripts/env_mac_cpu.sh
```

## Render 3D Gaussian PLY (gsplat-mlx / Metal)

```bash
python scripts/render_gaussian_ply.py \
  --ply /path/to/your_3dgs.ply \
  --output outputs/render.png \
  --width 512 --height 512
```

Supported layout: standard 3DGS (`x,y,z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`), binary or ASCII.

## Phase 1 — Geometry encode/decode (CPU)

Apply patches to sibling `../Gameleon` once (or after pulling Gameleon):

```bash
source scripts/env_mac_cpu.sh
./scripts/apply_gameleon_geometry_cpu_patches.sh
```

Re-apply TorchSparse patches into the active venv after `pip install` (if needed):

```bash
./scripts/apply_torchsparse_patches.sh
```

Smoke test (subsample 8000 points by default):

```bash
python scripts/test_gameleon_geometry_cpu.py
```

Full cloud (~562k points, slow on CPU):

```bash
python scripts/test_gameleon_geometry_cpu.py --max-points 0
```

## Optional: install native libs only

```bash
./scripts/install_torchsparse_cpu.sh   # geometry (TorchSparse v2.0.0 CPU)
python scripts/test_torchsparse_cpu.py

./scripts/install_minkowski_cpu.sh     # attribute path (ME v0.5.4 CPU)
export KMP_DUPLICATE_LIB_OK=TRUE
python scripts/test_minkowski_cpu.py
```

Patches: `patches/torchsparse_v2.0.0/`, `patches/minkowskiengine_v0.5.4/`, `patches/gameleon_geometry_cpu/`.

## Layout

```text
mac_Gameleon/
  examples/0519/           # default geometry test PLY/OBJ
  mac_gameleon/            # paths, device, render helpers
  patches/                 # Mac patches (TorchSparse, ME, Gameleon geometry)
  scripts/                 # setup, env, tests, apply_*_patches.sh
  vendor/gsplat-mlx/       # Metal 3DGS (submodule / vendor)
  vendor/torchsparse/      # v2.0.0 source (after install script)
  vendor/minkowskiengine/  # v0.5.4 source (after install script)
  requirements-mac-cpu.txt
```
