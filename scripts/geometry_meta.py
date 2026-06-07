#!/usr/bin/env python3
"""Gameleon geometry-meta: lossless UCM encode/decode (mlx-lattice + MinkowskiEngine)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*MinkowskiEngine was compiled with CPU_ONLY.*")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mac_gameleon.device import resolve_gameleon_device  # noqa: E402
from mac_gameleon.geometry_meta_patches import apply_geometry_meta_patches  # noqa: E402
from mac_gameleon.paths import (  # noqa: E402
    DEFAULT_INPUT_PLY,
    GAMELEON_PACKAGE_ROOT,
    GEOMETRY_CKPT,
    GEOMETRY_META_OUTPUT_DIR,
)


def _coords_from_sparse(x) -> np.ndarray:
    return x.C[:, 1:].detach().cpu().numpy().astype(np.int64)


def _sort_rows(coords: np.ndarray) -> np.ndarray:
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gameleon geometry-meta UCM encode/decode.")
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
    return parser.parse_args()


def _log(msg: str, *, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"[{elapsed:7.2f}s] {msg}", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    args = parse_args()
    os.environ.setdefault("GAMELEON_DEVICE", "cpu")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = resolve_gameleon_device()
    print(f"geometry-meta")
    print(f"input={args.input}")
    print(f"ckpt={args.ckpt}")

    if not args.input.is_file():
        raise SystemExit(f"Missing input PLY: {args.input}")
    if not args.ckpt.is_file():
        raise SystemExit(f"Missing geometry checkpoint: {args.ckpt}")

    try:
        import mlx_lattice  # noqa: F401
    except ImportError as exc:
        raise SystemExit("mlx-lattice is not installed. Run: pip install mlx-lattice") from exc

    from mac_gameleon.mlx_lattice_shim import install_mlx_lattice_geometry_backend  # noqa: E402

    install_mlx_lattice_geometry_backend(force=True)

    _log("Applying geometry-meta patches...", t0=t0)
    apply_geometry_meta_patches(quiet=True)

    from mac_gameleon.prewarm_torchac import prewarm_torchac  # noqa: E402

    prewarm_torchac(log=lambda msg: _log(msg, t0=t0))

    _log("Loading geometry-meta modules...", t0=t0)
    sys.path.insert(0, str(GAMELEON_PACKAGE_ROOT))

    from data_utils.dataloaders.geometry_dataloader import load_sparse_tensor  # noqa: E402
    from lossless_torchsparse.src.coder.coder_intra import CoderIntra  # noqa: E402

    outdir = GEOMETRY_META_OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    bin_path = outdir / f"{args.input.stem}.bin"

    _log("Loading input point cloud...", t0=t0)
    sparse_in = load_sparse_tensor(
        str(args.input), voxel_size=1, quant_mode="floor", device=device
    )
    coords_in = _sort_rows(_coords_from_sparse(sparse_in))
    print(f"points={coords_in.shape[0]}")

    _log("Loading UCM model...", t0=t0)
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

    _log("Compressing...", t0=t0)
    byte_stream, _metadata = coder.compress(coords_in, return_metadata=True)
    bin_path.write_bytes(byte_stream)
    geometry_bits = bin_path.stat().st_size * 8
    print(f"bitstream={bin_path} bits={geometry_bits}")

    _log("Decompressing...", t0=t0)
    xyz_dec, _ = coder.decompress(byte_stream, return_torch=True)
    coords_dec = xyz_dec.int().cpu().numpy()
    coords_dec_sorted = _sort_rows(coords_dec.astype(np.int64))

    _log("Verifying...", t0=t0)
    if coords_in.shape != coords_dec_sorted.shape:
        raise SystemExit(
            f"point count mismatch: in={coords_in.shape[0]} dec={coords_dec_sorted.shape[0]}"
        )
    if not np.array_equal(coords_in, coords_dec_sorted):
        diff = int(np.sum(np.any(coords_in != coords_dec_sorted, axis=1)))
        raise SystemExit(f"coordinate mismatch on {diff} points")

    _log("geometry-meta encode/decode OK.", t0=t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
