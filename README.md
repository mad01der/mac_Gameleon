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

## Layout

```text
mac_Gameleon/
  vendor/gsplat-mlx/     # vendored Metal rasterizer
  mac_gameleon/          # PLY loader + render wrapper
  scripts/
    setup_env.sh
    make_demo_gaussian_ply.py
    render_gaussian_ply.py
```
