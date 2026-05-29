#!/usr/bin/env python3
"""Gameleon lossless geometry encode/decode on Mac (TorchSparse UCM + MLX patches)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mac_gameleon.device import resolve_gameleon_device  # noqa: E402
from mac_gameleon.paths import (  # noqa: E402
    DEFAULT_INPUT_PLY,
    GAMELEON_PACKAGE_ROOT,
    GAMELEON_ROOT,
    GEOMETRY_CKPT,
    GEOMETRY_OUTPUT_DIR,
)


def _ensure_gameleon_geometry_patches() -> None:
    patch_script = ROOT / "patches" / "gameleon_geometry_cpu" / "apply_patches.py"
    import subprocess

    subprocess.run(
        [sys.executable, str(patch_script), str(GAMELEON_ROOT)],
        check=True,
    )


def _coords_from_sparse(x) -> np.ndarray:
    return x.C[:, 1:].detach().cpu().numpy().astype(np.int64)


def _sort_rows(coords: np.ndarray) -> np.ndarray:
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gameleon geometry UCM encode/decode on Mac.")
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
        "--no-patch-spdownsample-mlx",
        action="store_true",
        help="Disable MLX-backed monkey patch for torchsparse F.spdownsample (enabled by default).",
    )
    parser.add_argument(
        "--no-patch-build-kmap-mlx",
        action="store_true",
        help="Disable MLX-hybrid monkey patch for torchsparse F.build_kernel_map (enabled by default).",
    )
    parser.add_argument(
        "--no-patch-fog-mlx",
        action="store_true",
        help="Disable MLX-hybrid monkey patch for Gameleon FOG.forward (enabled by default).",
    )
    parser.add_argument(
        "--no-patch-conv3d-mlx",
        action="store_true",
        help="Disable MLX-hybrid monkey patch for torchsparse F.conv3d (k=3,s=1 path).",
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
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"GAMELEON_DEVICE={device}")
    print(f"input={args.input}")
    print(f"ckpt={args.ckpt}")

    if not args.input.is_file():
        raise SystemExit(f"Missing input PLY: {args.input}")
    if not args.ckpt.is_file():
        raise SystemExit(f"Missing geometry checkpoint: {args.ckpt}")

    _log("Applying Gameleon geometry CPU patches (idempotent)...", t0=t0)
    _ensure_gameleon_geometry_patches()

    _log("Importing Gameleon geometry modules...", t0=t0)
    sys.path.insert(0, str(GAMELEON_PACKAGE_ROOT))
    from mac_gameleon.mlx_torchsparse_patch import (  # noqa: E402
        patch_gameleon_fog_with_mlx,
        patch_torchsparse_build_kmap_with_mlx,
        patch_torchsparse_conv3d_k3s1_with_mlx,
        patch_torchsparse_spdownsample_with_mlx,
    )
    patch_spdownsample_mlx = not args.no_patch_spdownsample_mlx
    patch_build_kmap_mlx = not args.no_patch_build_kmap_mlx
    patch_fog_mlx = not args.no_patch_fog_mlx
    patch_conv3d_mlx = not args.no_patch_conv3d_mlx
    if patch_spdownsample_mlx:
        patch_torchsparse_spdownsample_with_mlx()
        _log("Patched torchsparse F.spdownsample -> MLX backend.", t0=t0)
    else:
        _log("Using original torchsparse F.spdownsample (MLX patch disabled).", t0=t0)
    if patch_build_kmap_mlx:
        patch_torchsparse_build_kmap_with_mlx()
        _log("Patched torchsparse F.build_kernel_map -> MLX-hybrid backend.", t0=t0)
    else:
        _log("Using original torchsparse F.build_kernel_map (MLX patch disabled).", t0=t0)
    if patch_fog_mlx:
        patch_gameleon_fog_with_mlx()
        _log("Patched Gameleon FOG.forward -> MLX-hybrid backend.", t0=t0)
    else:
        _log("Using original Gameleon FOG.forward (MLX patch disabled).", t0=t0)
    if patch_conv3d_mlx:
        patch_torchsparse_conv3d_k3s1_with_mlx()
        _log("Patched torchsparse F.conv3d (k=3,s=1) -> MLX-hybrid backend.", t0=t0)
    else:
        _log("Using original torchsparse F.conv3d (MLX patch disabled).", t0=t0)

    from data_utils.dataloaders.geometry_dataloader import load_sparse_tensor  # noqa: E402
    from lossless_torchsparse.src.coder.coder_intra import CoderIntra  # noqa: E402

    outdir = GEOMETRY_OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    bin_path = outdir / f"{args.input.stem}.bin"

    _log("Loading and quantizing sparse input...", t0=t0)
    sparse_in = load_sparse_tensor(
        str(args.input), voxel_size=1, quant_mode="floor", device=device
    )
    coords_in = _sort_rows(_coords_from_sparse(sparse_in))
    print(f"input points={coords_in.shape[0]}")

    _log("Initializing CoderIntra (model load)...", t0=t0)
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

    _log("Starting geometry compress()...", t0=t0)
    byte_stream, _metadata = coder.compress(coords_in, return_metadata=True)
    bin_path.write_bytes(byte_stream)
    geometry_bits = bin_path.stat().st_size * 8
    print(f"bitstream={bin_path} geometry_bits={geometry_bits}")

    _log("Starting geometry decompress()...", t0=t0)
    xyz_dec, _ = coder.decompress(byte_stream, return_torch=True)
    coords_dec = xyz_dec.int().cpu().numpy()
    coords_dec_sorted = _sort_rows(coords_dec.astype(np.int64))
    print(f"decoded points={coords_dec_sorted.shape[0]}")

    _log("Verifying decoded coordinates...", t0=t0)
    if coords_in.shape != coords_dec_sorted.shape:
        raise SystemExit(
            f"point count mismatch: in={coords_in.shape[0]} dec={coords_dec_sorted.shape[0]}"
        )
    if not np.array_equal(coords_in, coords_dec_sorted):
        diff = int(np.sum(np.any(coords_in != coords_dec_sorted, axis=1)))
        raise SystemExit(f"coordinate mismatch on {diff} points")

    _log("Phase 1 geometry OK: CPU encode/decode coordinates match.", t0=t0)
    print("Phase 1 geometry OK: CPU encode/decode coordinates match.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
