#!/usr/bin/env python3
"""Decode attribute bitstreams and print decoded_dc stats (Mac step3-aligned).

Use this to compare Mac-produced bitstreams on CUDA vs Mac:
  1. Copy Mac outputs bundle to the CUDA machine (see README section / --help).
  2. Run this script on CUDA with the same ckpt and level as Mac step3.
  3. Compare printed decoded_dc_diagnostics with Mac step3 output.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

warnings.filterwarnings("ignore", message=".*MinkowskiEngine was compiled with CPU_ONLY.*")

import numpy as np
import torch

_SH_DC_SCALE = 0.28209479177387814


def _tensor_channel_stats(tensor: torch.Tensor) -> str:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    parts = []
    for ch in range(arr.shape[-1]):
        col = arr[..., ch].reshape(-1)
        parts.append(
            f"ch{ch}[min={float(col.min()):.4f} max={float(col.max()):.4f} mean={float(col.mean()):.4f}]"
        )
    return " ".join(parts)


def _summarize_decoded_dc(gaussian_dict: dict) -> None:
    dc_list = gaussian_dict.get("decoded_dc") or []
    if not dc_list:
        print("decoded_dc_diagnostics: missing decoded_dc")
        return
    dc = dc_list[0].detach().float().cpu()
    print("decoded_dc_diagnostics (tensor as stored in gaussian_dict)")
    print(f"  shape={tuple(dc.shape)} {_tensor_channel_stats(dc)}")
    dc_np = dc.numpy()
    as_linear_rgb = np.clip(dc_np, 0.0, 1.0)
    print(
        "  as_linear_rgb_clip01 "
        f"R_mean={as_linear_rgb[:, 0].mean():.4f} "
        f"G_mean={as_linear_rgb[:, 1].mean():.4f} "
        f"B_mean={as_linear_rgb[:, 2].mean():.4f}"
    )
    rgb_from_sh = np.clip(dc_np * _SH_DC_SCALE + 0.5, 0.0, 1.0)
    print(
        "  if_sh_fdc_to_rgb "
        f"R_mean={rgb_from_sh[:, 0].mean():.4f} "
        f"G_mean={rgb_from_sh[:, 1].mean():.4f} "
        f"B_mean={rgb_from_sh[:, 2].mean():.4f}"
    )
    o_list = gaussian_dict.get("decoded_o") or []
    if o_list:
        op = o_list[0].detach().float().cpu().numpy().reshape(-1)
        print(
            f"  decoded_o mean={op.mean():.4f} min={op.min():.4f} max={op.max():.4f} "
            f"frac>0.5={float((op > 0.5).mean()):.4f}"
        )
    primitives = gaussian_dict.get("decoded_primitives") or []
    if primitives:
        print(f"  gaussian_points={int(primitives[0].shape[0])}")


def _resolve_gameleon_roots(gameleon_root: Optional[Path]) -> tuple[Path, Path]:
    if gameleon_root is not None:
        package_root = gameleon_root / "gameleon"
        attribute_root = package_root / "gameleon_attribute"
    else:
        mac_root = Path(__file__).resolve().parents[1]
        package_root = mac_root.parent / "Gameleon" / "gameleon"
        attribute_root = package_root / "gameleon_attribute"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Gameleon package root not found: {package_root}")
    if not attribute_root.is_dir():
        raise FileNotFoundError(f"gameleon_attribute not found: {attribute_root}")
    return package_root, attribute_root


def _bootstrap_pythonpath(package_root: Path, attribute_root: Path) -> None:
    for path in (str(package_root), str(attribute_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


@contextlib.contextmanager
def _gameleon_attribute_workdir(attribute_root: Path):
    prev_cwd = os.getcwd()
    os.chdir(attribute_root)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _bitstreams_for_level(prefix: Path, level: int) -> list[Path]:
    required_streams = max(1, int(level) - 6)
    paths = [Path(f"{prefix}_level_{idx}.bin") for idx in range(required_streams)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attribute bitstream(s): {missing}")
    return paths


def _resolve_from_outdir(
    outdir: Path,
    sample_name: str,
    level: int,
) -> tuple[list[Path], Path]:
    orig_dir = outdir / "orig_attribute"
    geometry_dir = outdir / "geometry"
    prefix = orig_dir / sample_name
    bitstreams = _bitstreams_for_level(prefix, level)
    support_ply = geometry_dir / f"{sample_name}_level_{level}_support_dec.ply"
    if not support_ply.is_file():
        raise FileNotFoundError(f"Missing decoded support PLY: {support_ply}")
    return bitstreams, support_ply


def _load_adapter(
    *,
    ckpt: Path,
    package_root: Path,
    attribute_root: Path,
    mac_patches: bool,
    exact_decode_impl: str,
    scale_factor: int,
    offset: int,
):
    if mac_patches:
        mac_root = Path(__file__).resolve().parents[1]
        if str(mac_root) not in sys.path:
            sys.path.insert(0, str(mac_root))
        from mac_gameleon.attribute_bootstrap import (
            gameleon_attribute_workdir,
            install_attribute_backend,
            load_attribute_adapter,
            prewarm_attribute_extensions,
            setup_attribute_env,
        )

        setup_attribute_env(no_lattice=True)
        install_attribute_backend(no_lattice=True)
        prewarm_attribute_extensions()
        adapter = load_attribute_adapter(str(ckpt))
        workdir_cm = gameleon_attribute_workdir()
    else:
        _bootstrap_pythonpath(package_root, attribute_root)
        from core.attribute_adapter import GameleonAttributeAdapter

        adapter = GameleonAttributeAdapter(
            ckpt=str(ckpt),
            scale_factor=scale_factor,
            offset=offset,
            runtime_precision="fp32",
            exact_decode_impl=exact_decode_impl,
            debug=False,
        )
        workdir_cm = _gameleon_attribute_workdir(attribute_root)
    return adapter, workdir_cm


def run_decode_diagnostics(
    *,
    bitstream_files: Sequence[str | Path],
    decoded_support_ply: Path,
    ckpt: Path,
    level: int,
    gameleon_root: Optional[Path] = None,
    device: Optional[str] = None,
    exact_decode_impl: str = "fused_cpu_fastcore",
    scale_factor: int = 512,
    offset: int = 512,
    mac_patches: bool = False,
) -> dict:
    package_root, attribute_root = _resolve_gameleon_roots(gameleon_root)

    if device:
        os.environ["GAMELEON_DEVICE"] = device
    os.environ.setdefault("GAMELEON_ME_LATTICE", "0")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    if not decoded_support_ply.is_file():
        raise FileNotFoundError(f"Missing decoded support PLY: {decoded_support_ply}")

    bitstreams = [str(p) for p in bitstream_files]
    for path in bitstreams:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing attribute bitstream: {path}")

    print(f"gameleon_package_root={package_root}")
    print(f"ckpt={ckpt}")
    print(f"level={level}")
    print(f"bitstreams={bitstreams}")
    print(f"geom_ply={decoded_support_ply}")
    print(f"exact_decode_impl={exact_decode_impl}")
    print(f"mac_patches={mac_patches}")
    print(f"scale_factor={scale_factor} offset={offset}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")

    adapter, workdir_cm = _load_adapter(
        ckpt=ckpt,
        package_root=package_root,
        attribute_root=attribute_root,
        mac_patches=mac_patches,
        exact_decode_impl=exact_decode_impl,
        scale_factor=scale_factor,
        offset=offset,
    )
    print(f"adapter_device={adapter.device}")

    decode_t0 = time.perf_counter()
    with workdir_cm:
        with torch.no_grad():
            gaussian_dict, decode_info = adapter.decode(
                level=level,
                bitstream_files=bitstreams,
                geom_ply_file=str(decoded_support_ply),
                geom_coordinates=None,
                geom_sparse=None,
                fixed_geometry_ply=None,
                compat_mode=False,
                decode_geom_support_dir=None,
                attribute_codec="exact",
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_sec = time.perf_counter() - decode_t0
    print(f"attribute_decode_sec={decode_sec:.3f}")

    _summarize_decoded_dc(gaussian_dict)
    timing = dict(decode_info.get("decode_timing", {}) or {})
    if timing:
        keys = (
            "attribute_entropy_decode",
            "attribute_gaussian_build",
            "attribute_decode_impl",
            "attribute_entropy_decode_impl",
            "runtime_precision",
        )
        print(
            "decode_timing "
            + " ".join(f"{key}={timing.get(key)}" for key in keys if key in timing)
        )
    return gaussian_dict


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    mac_root = Path(__file__).resolve().parents[1]
    default_ckpt = (
        mac_root.parent
        / "Gameleon"
        / "gameleon"
        / "weights"
        / "attribute"
        / "bpp_0.39"
        / "checkpoint"
        / "epoch4.pth"
    )
    parser = argparse.ArgumentParser(
        description="Decode Mac bitstreams and print decoded_dc diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # On Mac (re-baseline from existing outputs):\n"
            "  python scripts/decode_bitstreams_diagnostics.py "
            "--from-outdir outputs --sample pcd_0 --level 8\n\n"
            "  # On CUDA after copying bundle to ~/mac_codec_bundle:\n"
            "  python scripts/decode_bitstreams_diagnostics.py "
            "--bundle ~/mac_codec_bundle --sample pcd_0 --level 8 "
            "--device cuda --gameleon-root /path/to/Gameleon\n"
        ),
    )
    parser.add_argument("--bundle", type=Path, help="Directory with orig_attribute/ and geometry/ subdirs")
    parser.add_argument("--from-outdir", type=Path, help="Mac outputs directory (same layout as test.py)")
    parser.add_argument("--sample", type=str, default="pcd_0", help="Sample prefix (stem of input PLY)")
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--ckpt", type=Path, default=default_ckpt)
    parser.add_argument("--bitstream", type=Path, nargs="*", help="Explicit bitstream paths (overrides bundle)")
    parser.add_argument("--geom-ply", type=Path, help="Decoded geometry support PLY (overrides bundle)")
    parser.add_argument("--gameleon-root", type=Path, help="Path to Gameleon repo root (parent of gameleon/)")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (default: auto)")
    parser.add_argument("--exact-decode-impl", type=str, default="fused_cpu_fastcore")
    parser.add_argument("--scale-factor", type=int, default=512)
    parser.add_argument("--offset", type=int, default=512)
    parser.add_argument(
        "--mac-patches",
        action="store_true",
        help="Apply mac_Gameleon ME aliases/patches (use on Mac; omit on native CUDA Gameleon)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.bitstream and args.geom_ply:
        bitstreams = list(args.bitstream)
        support_ply = args.geom_ply
    elif args.from_outdir is not None:
        bitstreams, support_ply = _resolve_from_outdir(args.from_outdir, args.sample, args.level)
    elif args.bundle is not None:
        bitstreams, support_ply = _resolve_from_outdir(args.bundle, args.sample, args.level)
    else:
        print("Provide --from-outdir, --bundle, or both --bitstream and --geom-ply", file=sys.stderr)
        return 2

    mac_patches = bool(args.mac_patches or args.from_outdir is not None)
    run_decode_diagnostics(
        bitstream_files=bitstreams,
        decoded_support_ply=support_ply,
        ckpt=args.ckpt,
        level=args.level,
        gameleon_root=args.gameleon_root,
        device=args.device or None,
        exact_decode_impl=args.exact_decode_impl,
        scale_factor=args.scale_factor,
        offset=args.offset,
        mac_patches=mac_patches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
