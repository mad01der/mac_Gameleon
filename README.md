# mac_Gameleon

Mac-side rendering experiments for Gameleon, using [gsplat-mlx](https://github.com/RobotFlow-Labs/gsplat-mlx) (Metal on Apple Silicon).

CUDA reference code stays in `../Gameleon/`. This tree only adds Mac rendering.

## Requirements

- macOS on Apple Silicon (M series)
- **Python >= 3.10** (system 3.9 is not enough)
- Homebrew Python recommended: `brew install python@3.12`

## One-time setup

```bash
cd mac_Gameleon
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
source .venv/bin/activate
```

## Minimal render (demo PLY)

```bash
cd mac_Gameleon
source .venv/bin/activate

python scripts/make_demo_gaussian_ply.py --output examples/data/demo_gaussians.ply

python scripts/render_gaussian_ply.py \
  --ply examples/data/demo_gaussians.ply \
  --output outputs/demo_render.png \
  --width 512 --height 512
open outputs/demo_render.png
```

## Render Gameleon-exported Gaussians

After CUDA `gameleon-test` writes `decoded_gaussians_seq.ply` (or similar 3DGS PLY):

```bash
source .venv/bin/activate
python scripts/render_gaussian_ply.py \
  --ply /path/to/decoded_gaussians_seq.ply \
  --output outputs/longdress_render.png \
  --width 512 --height 512 --fov 60
```

Supported PLY layout: standard 3DGS (`x,y,z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`), binary or ASCII — same as Gameleon `export_decoded_ply`.

## TorchSparse CPU (geometry spike)

Gameleon UCM needs **TorchSparse**. v2.1 master is GPU-only; this repo pins **v2.0.0** with Mac patches.

Prerequisites (Homebrew):

```bash
brew install google-sparsehash libomp
```

Install into the same `.venv` as gsplat-mlx:

```bash
source .venv/bin/activate
chmod +x scripts/install_torchsparse_cpu.sh
./scripts/install_torchsparse_cpu.sh
```

Smoke test only:

```bash
python scripts/test_torchsparse_cpu.py
# expect: CPU conv ok: (100, 16)
```

Patches live in `patches/torchsparse_v2.0.0/` (OpenMP/libomp + CPU kmap fallback).

**Note:** This validates sparse conv on Mac CPU. Wiring Gameleon `CoderIntra`/UCM still requires matching the Linux TorchSparse version and replacing hard-coded `cuda` in Gameleon code.

## Layout

```text
mac_Gameleon/
  vendor/gsplat-mlx/       # Metal 3DGS rasterizer
  vendor/torchsparse/      # v2.0.0 + Mac CPU patches (after install script)
  patches/torchsparse_v2.0.0/
  mac_gameleon/            # PLY loader + render wrapper
  scripts/
    setup_env.sh
    install_torchsparse_cpu.sh
    test_torchsparse_cpu.py
    make_demo_gaussian_ply.py
    render_gaussian_ply.py
```
