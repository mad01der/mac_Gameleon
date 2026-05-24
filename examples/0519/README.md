# 0519 test frame

- `pcd_0.ply` — point cloud (~562,683 points, integer voxel coords x/y/z ≈ 276–748 / 64–960 / 418–615)
- `0519.obj` — mesh ground truth (+ `material0.mtl` / `material0.jpeg`)

Geometry smoke test (subsample for CPU):

```bash
python scripts/test_gameleon_geometry_cpu.py --max-points 2000
```
