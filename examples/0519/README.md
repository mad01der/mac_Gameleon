# 0519 test frame

- `pcd_0.ply` — point cloud (~562k points)
- `0519.obj` — mesh ground truth (+ `material0.mtl` / `material0.jpeg`)

Full pipeline:

```bash
cd ../..
source scripts/env_mac_cpu.sh
python test.py
```

Skip render (bpp + PLY only):

```bash
python test.py --no-render
```
