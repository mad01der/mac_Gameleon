#!/usr/bin/env python3
"""Phase 1: Gameleon lossless geometry encode/decode on Mac CPU (TorchSparse UCM)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mac_gameleon.device import resolve_gameleon_device  # noqa: E402
from mac_gameleon.paths import (  # noqa: E402
    DEFAULT_INPUT_PLY,
    GAMELEON_PACKAGE_ROOT,
    GEOMETRY_CKPT,
)


def _coords_from_sparse(x) -> np.ndarray:
    return x.C[:, 1:].detach().cpu().numpy().astype(np.int64)


def _sort_rows(coords: np.ndarray) -> np.ndarray:
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order]


def _maybe_subsample_ply(input_ply: Path, max_points: int | None, workdir: Path) -> Path:
    if max_points is None or max_points <= 0:
        return input_ply
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(input_ply))
    n = len(pcd.points)
    if n <= max_points:
        return input_ply
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
    subset = pcd.select_by_index(idx.tolist())
    out = workdir / "subset.ply"
    o3d.io.write_point_cloud(str(out), subset, write_ascii=True)
    print(f"subsampled {n} -> {max_points} points for smoke test: {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gameleon geometry UCM on Mac CPU.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PLY,
        help=f"Input point cloud PLY (default: {DEFAULT_INPUT_PLY})",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=GEOMETRY_CKPT,
        help="Geometry UCM checkpoint (best_model_UCM.pt)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=8000,
        help="Random subsample cap for CPU smoke tests (0 = use full cloud)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("GAMELEON_DEVICE", "cpu")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = resolve_gameleon_device()
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"GAMELEON_DEVICE={device}")
    print(f"input={args.input}")
    print(f"ckpt={args.ckpt}")

    if not args.input.is_file():
        raise SystemExit(f"Missing input PLY: {args.input}")
    if not args.ckpt.is_file():
        raise SystemExit(f"Missing geometry checkpoint: {args.ckpt}")

    sys.path.insert(0, str(GAMELEON_PACKAGE_ROOT))
    from data_utils.dataloaders.geometry_dataloader import load_sparse_tensor  # noqa: E402
    from lossless_torchsparse.src.coder.coder_intra import CoderIntra  # noqa: E402

    with tempfile.TemporaryDirectory(prefix="mac_gameleon_geom_") as tmp:
        tmpdir = Path(tmp)
        ply_path = _maybe_subsample_ply(args.input, args.max_points, tmpdir)
        outdir = tmpdir / "bitstreams"
        outdir.mkdir()

        sparse_in = load_sparse_tensor(str(ply_path), voxel_size=1, quant_mode="floor", device=device)
        coords_in = _sort_rows(_coords_from_sparse(sparse_in))
        print(f"input points={coords_in.shape[0]}")

        coder = CoderIntra(
            model_path=str(args.ckpt),
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

        byte_stream, _metadata = coder.compress(coords_in, return_metadata=True)
        bin_path = outdir / "phase1.bin"
        bin_path.write_bytes(byte_stream)
        geometry_bits = bin_path.stat().st_size * 8
        print(f"bitstream={bin_path} geometry_bits={geometry_bits}")

        xyz_dec, _ = coder.decompress(byte_stream, return_torch=True)
        coords_dec = xyz_dec.int().cpu().numpy()
        coords_dec_sorted = _sort_rows(coords_dec.astype(np.int64))
        print(f"decoded points={coords_dec_sorted.shape[0]}")

        if coords_in.shape != coords_dec_sorted.shape:
            raise SystemExit(
                f"point count mismatch: in={coords_in.shape[0]} dec={coords_dec_sorted.shape[0]}"
            )
        if not np.array_equal(coords_in, coords_dec_sorted):
            diff = int(np.sum(np.any(coords_in != coords_dec_sorted, axis=1)))
            raise SystemExit(f"coordinate mismatch on {diff} points")

    print("Phase 1 geometry OK: CPU encode/decode coordinates match.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
