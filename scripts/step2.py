#!/usr/bin/env python3
"""Gameleon Step 2: lossless UCM geometry encode/decode on support point cloud."""

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

from mac_gameleon.device import resolve_gameleon_device  # noqa: E402
from mac_gameleon.geometry_meta_patches import apply_geometry_meta_patches  # noqa: E402
from mac_gameleon.paths import (  # noqa: E402
    DEFAULT_INPUT_PLY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIPELINE_LEVEL,
    GAMELEON_PACKAGE_ROOT,
    GEOMETRY_CKPT,
    pipeline_output_paths,
)


@dataclass
class Step2Result:
    bitstream_files: list[Path]
    decoded_support_ply: Path
    geometry_bits: int
    points: int
    encode_sec: float
    decode_sec: float


def _coords_from_sparse(x) -> np.ndarray:
    return x.C[:, 1:].detach().cpu().numpy().astype(np.int64)


def _sort_rows(coords: np.ndarray) -> np.ndarray:
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order]


def _write_support_ply(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)


def _resolve_sample_name(input_ply: Path, sample_name: Optional[str], level: int) -> str:
    if sample_name:
        return sample_name
    stem = input_ply.stem
    suffix = f"_level_{level}_geom"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gameleon Step 2: lossless UCM geometry encode/decode (support PLY).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PLY,
        help="Support geometry PLY (typically orig_attribute/{sample}_level_8_geom.ply)",
    )
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--level", type=int, default=DEFAULT_PIPELINE_LEVEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=GEOMETRY_CKPT,
        help="Geometry UCM checkpoint (best_model_UCM.pt)",
    )
    return parser.parse_args(argv)


def _default_log(msg: str, *, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.2f}s] {msg}", flush=True)


def run_step2(
    *,
    input_ply: Path,
    ckpt: Path = GEOMETRY_CKPT,
    outdir: Path = DEFAULT_OUTPUT_DIR,
    sample_name: Optional[str] = None,
    level: int = DEFAULT_PIPELINE_LEVEL,
    log: Optional[Callable[[str], None]] = None,
) -> Step2Result:
    t0 = time.perf_counter()
    log = log or (lambda msg: _default_log(msg, t0=t0))
    os.environ.setdefault("GAMELEON_DEVICE", "cpu")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = resolve_gameleon_device()
    if not input_ply.is_file():
        raise FileNotFoundError(f"Missing input PLY: {input_ply}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing geometry checkpoint: {ckpt}")

    sample = _resolve_sample_name(input_ply, sample_name, level)
    paths = pipeline_output_paths(outdir, sample, level=level)
    bin_path = paths["geometry_bitstream"]
    decoded_ply = paths["decoded_support_ply"]
    paths["geometry_dir"].mkdir(parents=True, exist_ok=True)

    try:
        import mlx_lattice  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("mlx-lattice is not installed. Run: pip install mlx-lattice") from exc

    from mac_gameleon.mlx_lattice_shim import install_mlx_lattice_geometry_backend  # noqa: E402

    install_mlx_lattice_geometry_backend(force=True)

    log("Applying geometry patches...")
    apply_geometry_meta_patches(quiet=True)

    from mac_gameleon.prewarm_torchac import prewarm_torchac  # noqa: E402

    prewarm_torchac(log=log)

    log("Loading geometry modules...")
    sys.path.insert(0, str(GAMELEON_PACKAGE_ROOT))

    from data_utils.dataloaders.geometry_dataloader import load_sparse_tensor  # noqa: E402
    from lossless_torchsparse.src.coder.coder_intra import CoderIntra  # noqa: E402

    log("Loading support point cloud...")
    sparse_in = load_sparse_tensor(
        str(input_ply), voxel_size=1, quant_mode="floor", device=device
    )
    native_support_points = _sort_rows(_coords_from_sparse(sparse_in))
    points = int(native_support_points.shape[0])
    print(f"points={points}")

    log("Loading UCM model...")
    coder = CoderIntra(
        model_path=str(ckpt),
        device=device,
        lossy_level=0,
        no_lossy_net=False,
        is_data_pre_quantized=False,
        posQ=1,
        preprocess_scale=1.0,
        preprocess_shift=0.0,
        channels=32,
        kernel_size=3,
    )

    log("Compressing...")
    encode_t0 = time.perf_counter()
    byte_stream, _metadata = coder.compress(native_support_points, return_metadata=True)
    encode_sec = time.perf_counter() - encode_t0
    bin_path.write_bytes(byte_stream)
    geometry_bits = bin_path.stat().st_size * 8
    print(f"bitstream={bin_path} bits={geometry_bits}")

    log("Decompressing...")
    decode_t0 = time.perf_counter()
    xyz_dec, _ = coder.decompress(byte_stream, return_torch=True)
    decode_sec = time.perf_counter() - decode_t0
    decoded_points = _sort_rows(xyz_dec.int().cpu().numpy().astype(np.int64))

    log("Verifying...")
    if native_support_points.shape != decoded_points.shape:
        raise RuntimeError(
            f"point count mismatch: in={native_support_points.shape[0]} dec={decoded_points.shape[0]}"
        )
    if not np.array_equal(native_support_points, decoded_points):
        diff = int(np.sum(np.any(native_support_points != decoded_points, axis=1)))
        raise RuntimeError(f"coordinate mismatch on {diff} points")

    coord_to_idx = {tuple(pt.tolist()): idx for idx, pt in enumerate(decoded_points)}
    ordered_indices = np.asarray(
        [coord_to_idx[tuple(pt.tolist())] for pt in native_support_points],
        dtype=np.int64,
    )
    ordered_decoded_points = decoded_points[ordered_indices]
    _write_support_ply(decoded_ply, ordered_decoded_points)
    print(f"decoded_support_ply={decoded_ply}")

    return Step2Result(
        bitstream_files=[bin_path],
        decoded_support_ply=decoded_ply,
        geometry_bits=int(geometry_bits),
        points=points,
        encode_sec=float(encode_sec),
        decode_sec=float(decode_sec),
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    print("step2 (lossless support geometry)")
    print(f"input={args.input}")
    print(f"outdir={args.outdir}")
    print(f"ckpt={args.ckpt}")

    run_step2(
        input_ply=args.input,
        ckpt=args.ckpt,
        outdir=args.outdir,
        sample_name=args.sample_name,
        level=args.level,
        log=lambda msg: _default_log(msg, t0=t0),
    )
    _default_log("step2 OK.", t0=t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
