#!/usr/bin/env python3
"""Step2 benchmark: CPU vs MLX sparse voxel aggregation PoC."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from mac_gameleon.mlx_sparse_poc import (
    aggregate_voxels_cpu_reference,
    aggregate_voxels_mlx,
    mlx_device_name,
)
from mac_gameleon.paths import DEFAULT_INPUT_PLY


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


def _load_points(path: Path, max_points: int) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise SystemExit(f"No points loaded from {path}")
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    return np.floor(points).astype(np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse voxel aggregation: CPU reference vs MLX."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PLY)
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Randomly subsample input points before benchmark (0 = full cloud).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=8,
        help="Feature channels for synthetic feature tensor.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=2,
        help="Voxel parent mapping factor (coords // factor).",
    )
    parser.add_argument(
        "--reduce",
        type=str,
        choices=("sum", "mean"),
        default="sum",
        help="Aggregation mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input PLY not found: {args.input}")
    if args.channels <= 0:
        raise SystemExit("--channels must be > 0")
    if args.downsample_factor <= 0:
        raise SystemExit("--downsample-factor must be > 0")

    print("Step2 MLX sparse PoC benchmark")
    print(f"mlx_device={mlx_device_name()}")
    print(f"input={args.input}")
    print(f"max_points={args.max_points} channels={args.channels} reduce={args.reduce}")

    coords = _load_points(args.input, args.max_points)
    rng = np.random.default_rng(1234)
    feats = rng.normal(size=(coords.shape[0], args.channels)).astype(np.float32)
    print(f"loaded_points={coords.shape[0]}")

    cpu_result, cpu_sec = _timed(
        aggregate_voxels_cpu_reference,
        coords,
        feats,
        downsample_factor=args.downsample_factor,
        reduce=args.reduce,
    )
    try:
        mlx_result, mlx_sec = _timed(
            aggregate_voxels_mlx,
            coords,
            feats,
            downsample_factor=args.downsample_factor,
            reduce=args.reduce,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "No Metal device available" in msg:
            raise SystemExit(
                "FAILED: MLX cannot access Metal GPU in this session. "
                "Please run this script from your local terminal (non-headless)."
            ) from exc
        raise

    coords_equal = np.array_equal(cpu_result.parent_coords, mlx_result.parent_coords)
    max_abs_diff = float(np.max(np.abs(cpu_result.parent_feats - mlx_result.parent_feats)))
    mean_abs_diff = float(np.mean(np.abs(cpu_result.parent_feats - mlx_result.parent_feats)))

    print(f"parents_cpu={cpu_result.parent_coords.shape[0]} parents_mlx={mlx_result.parent_coords.shape[0]}")
    print(f"coords_equal={coords_equal}")
    print(f"max_abs_diff={max_abs_diff:.8f} mean_abs_diff={mean_abs_diff:.8f}")
    print(f"cpu_time_sec={cpu_sec:.4f}")
    print(f"mlx_time_sec={mlx_sec:.4f}")
    if mlx_sec > 0:
        print(f"speedup_cpu_over_mlx={cpu_sec / mlx_sec:.3f}x")

    if not coords_equal or max_abs_diff > 1e-5:
        raise SystemExit("FAILED: MLX result mismatch vs CPU reference")
    print("PASS: MLX aggregation matches CPU reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

