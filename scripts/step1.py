#!/usr/bin/env python3
"""Gameleon Step 1: attribute encode (PCML) from RGB point cloud."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

warnings.filterwarnings("ignore", message=".*MinkowskiEngine was compiled with CPU_ONLY.*")

import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mac_gameleon.attribute_bootstrap import (  # noqa: E402
    install_attribute_backend,
    load_attribute_adapter,
    prewarm_attribute_extensions,
    setup_attribute_env,
)
from mac_gameleon.paths import (  # noqa: E402
    ATTRIBUTE_CKPT_LEVEL8,
    DEFAULT_INPUT_PLY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIPELINE_LEVEL,
    pipeline_output_paths,
)
from mac_gameleon.pointcloud import PointCloud  # noqa: E402


@dataclass
class Step1Result:
    bitstream_files: list[str]
    native_support_ply: Path
    geom_points: int
    attribute_bits: int
    attribute_prefix: str
    sparse_points: int
    encode_sec: float
    required_streams: int


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gameleon Step 1: attribute encode.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PLY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--level", type=int, default=DEFAULT_PIPELINE_LEVEL)
    parser.add_argument("--ckpt", type=Path, default=ATTRIBUTE_CKPT_LEVEL8)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Subsample to at most N points (default: 0 = use all points)",
    )
    parser.add_argument(
        "--no-lattice",
        action="store_true",
        help="Disable mlx-lattice ME acceleration (default: lattice enabled)",
    )
    return parser.parse_args(argv)


def _default_log(msg: str, *, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.2f}s] {msg}", flush=True)


def run_step1(
    *,
    input_ply: Path,
    ckpt: Path = ATTRIBUTE_CKPT_LEVEL8,
    outdir: Path = DEFAULT_OUTPUT_DIR,
    level: int = DEFAULT_PIPELINE_LEVEL,
    max_points: int = 0,
    no_lattice: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Step1Result:
    t0 = time.perf_counter()
    log = log or (lambda msg: _default_log(msg, t0=t0))
    setup_attribute_env(no_lattice=no_lattice)

    if not input_ply.is_file():
        raise FileNotFoundError(f"Missing input PLY: {input_ply}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    sample_name = input_ply.stem
    paths = pipeline_output_paths(outdir, sample_name, level=level)
    paths["orig_attribute_dir"].mkdir(parents=True, exist_ok=True)
    prefix = str(paths["attribute_prefix"])
    native_support_ply = paths["native_support_ply"]

    install_attribute_backend(no_lattice=no_lattice, log=log)
    prewarm_attribute_extensions(log=log)
    adapter = load_attribute_adapter(str(ckpt), log=log)

    log("Reading PLY...")
    input_pcd = o3d.io.read_point_cloud(str(input_ply))
    if input_pcd.is_empty():
        raise ValueError(f"Empty point cloud: {input_ply}")
    n_points = len(input_pcd.points)
    if max_points and n_points > max_points:
        indices = np.random.default_rng(0).choice(n_points, max_points, replace=False)
        input_pcd = input_pcd.select_by_index(np.sort(indices))
        print(f"subsampled points {n_points} -> {len(input_pcd.points)}")

    log("Building ME sparse input...")
    pcd = PointCloud.from_o3d_pcd(input_pcd)
    color_sparse = adapter._build_sparse_input(pcd, input_offset=None)
    sparse_points = int(color_sparse.F.shape[0])
    print(f"sparse_points={sparse_points} feat_dim={int(color_sparse.F.shape[1])}")

    for idx in range(4):
        stale = f"{prefix}_level_{idx}.bin"
        if os.path.exists(stale):
            os.remove(stale)
    if native_support_ply.exists():
        native_support_ply.unlink()

    log("Running model.encode(verbose=True)...")
    encode_t0 = time.perf_counter()
    with torch.no_grad():
        encode_result = adapter.model.encode(
            color_sparse=color_sparse,
            bitstr_prefix=prefix,
            required_geom_levels=(level,),
            verbose=True,
            write_geom_files=True,
            return_geom_xyz=True,
        )
    encode_sec = time.perf_counter() - encode_t0
    log(f"model.encode finished in {encode_sec:.3f}s")

    bitstreams = [
        f"{prefix}_level_{idx}.bin"
        for idx in range(4)
        if os.path.exists(f"{prefix}_level_{idx}.bin")
    ]
    required_streams = max(1, int(level) - 6)
    if len(bitstreams) < required_streams:
        raise RuntimeError(
            f"Expected at least {required_streams} attribute bitstreams, got {len(bitstreams)}"
        )
    if not native_support_ply.is_file():
        raise FileNotFoundError(f"Missing native support geometry PLY: {native_support_ply}")

    attribute_bits = sum(
        Path(path).stat().st_size for path in bitstreams[:required_streams]
    ) * 8
    geom_xyz = np.asarray(encode_result["geom_xyz_by_level"][level], dtype=np.int64)
    print(
        f"attribute_bitstreams={bitstreams[:required_streams]} "
        f"attribute_bits={attribute_bits} "
        f"support_points={geom_xyz.shape[0]} "
        f"native_support_ply={native_support_ply}"
    )

    return Step1Result(
        bitstream_files=bitstreams[:required_streams],
        native_support_ply=native_support_ply,
        geom_points=int(geom_xyz.shape[0]),
        attribute_bits=int(attribute_bits),
        attribute_prefix=prefix,
        sparse_points=sparse_points,
        encode_sec=float(encode_sec),
        required_streams=required_streams,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    print("step1 (attribute encode)")
    print(f"input={args.input}")
    print(f"outdir={args.outdir}")
    print(f"ckpt={args.ckpt}")
    print(f"max_points={args.max_points or 'all'}")
    setup_attribute_env(no_lattice=args.no_lattice)
    print(f"me_lattice={os.environ.get('GAMELEON_ME_LATTICE', '1')}")

    run_step1(
        input_ply=args.input,
        ckpt=args.ckpt,
        outdir=args.outdir,
        level=args.level,
        max_points=args.max_points,
        no_lattice=args.no_lattice,
        log=lambda msg: _default_log(msg, t0=t0),
    )
    _default_log("step1 OK.", t0=t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
