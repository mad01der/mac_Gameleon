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

Default test data: `examples/0519/` (`pcd_0.ply` ~562k voxel coords + `0519.obj`).

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

`setup_env.sh` applies Gameleon geometry CPU patches; `geometry_mac.py` re-applies them idempotently before encode/decode.

Re-copy TorchSparse runtime patches after a venv `pip install` overwrote them (no rebuild):

```bash
PATCHES_ONLY=1 ./scripts/install_torchsparse_cpu.sh
```

Lossless geometry encode/decode on the default full cloud (~562k points, slow on CPU):

```bash
python scripts/geometry_mac.py
```

Encoded bitstreams are written to `outputs/geometry/` (e.g. `pcd_0.bin` for the default input).

Notes:

- `geometry_mac.py` enables MLX-hybrid patches by default for:
  - `spdownsample`
  - `build_kernel_map`
  - `FOG.forward`
  - `conv3d` fast-path (`k=3,s=1,non-transposed`)
- Use `--no-patch-*` flags to disable each patch for ablation.

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
  outputs/geometry/        # geometry bitstreams from geometry_mac.py (gitignored)
  mac_gameleon/            # paths, device, render helpers
  patches/                 # Mac patches (TorchSparse, ME, Gameleon geometry)
  scripts/                 # setup, env, install_*, geometry_mac.py
  vendor/gsplat-mlx/       # Metal 3DGS (submodule / vendor)
  vendor/torchsparse/      # v2.0.0 source (after install script)
  vendor/minkowskiengine/  # v0.5.4 source (after install script)
  requirements-mac-cpu.txt
```
