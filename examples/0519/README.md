# 0519 test frame

- `pcd_0.ply` — point cloud (~562,683 points, integer voxel coords x/y/z ≈ 276–748 / 64–960 / 418–615)
- `0519.obj` — mesh ground truth (+ `material0.mtl` / `material0.jpeg`)

Geometry encode/decode (full cloud):

```bash
cd ../..
source scripts/env_mac_cpu.sh
python scripts/geometry_mac.py
```
