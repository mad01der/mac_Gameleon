#!/usr/bin/env python3
"""Gameleon Step 3: attribute decode and export decoded Gaussian PLY."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

warnings.filterwarnings("ignore", message=".*MinkowskiEngine was compiled with CPU_ONLY.*")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mac_gameleon.attribute_bootstrap import (  # noqa: E402
    gameleon_attribute_workdir,
    install_attribute_backend,
    load_attribute_adapter,
    prewarm_attribute_extensions,
    setup_attribute_env,
)
from mac_gameleon.me_lattice_patch import lattice_stats  # noqa: E402
from mac_gameleon.paths import (  # noqa: E402
    ATTRIBUTE_CKPT_LEVEL8,
    DEFAULT_INPUT_PLY,
    DEFAULT_MESH_GT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIPELINE_LEVEL,
    attribute_bitstream_paths,
    pipeline_output_paths,
    render_output_paths,
)
from mac_gameleon.metrics import (  # noqa: E402
    bitstream_size_bits,
    compute_rate_metrics,
    format_rate_metrics,
    sum_bitstream_bits,
)
from mac_gameleon.render_pipeline import (  # noqa: E402
    DEFAULT_BACKGROUND_COLOR,
    DEFAULT_FOV,
    DEFAULT_HEIGHT,
    DEFAULT_SUPER_SAMPLE_RATE,
    DEFAULT_WIDTH,
    export_decoded_gaussian_ply,
    run_post_decode_render,
    write_decode_summary,
)


@dataclass
class Step3Result:
    gaussian_points: int
    decode_sec: float
    decode_timing: dict
    gaussian_keys: list[str]
    decoded_pcd_path: Optional[str] = None
    render_psnr: Optional[float] = None
    per_view_psnr: Optional[list[float]] = None
    render_dir: Optional[str] = None
    gt_render_dir: Optional[str] = None
    summary_json: Optional[str] = None
    render_sec: Optional[float] = None
    encode_sec: Optional[float] = None
    total_decode_sec: Optional[float] = None
    geometry_bpp: Optional[float] = None
    attribute_bpp: Optional[float] = None
    bpp: Optional[float] = None


def _count_ply_points(ply_path: Path) -> int:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    return int(len(pcd.points))


def _resolve_rate_metrics(
    *,
    outdir: Path,
    sample_name: str,
    level: int,
    total_points: Optional[int] = None,
    attribute_bits: Optional[int] = None,
    geometry_bits: Optional[int] = None,
    support_points: Optional[int] = None,
) -> dict[str, float | int]:
    paths = pipeline_output_paths(outdir, sample_name, level=level)
    bitstreams = attribute_bitstream_paths(outdir, sample_name, level=level)
    if attribute_bits is None:
        attribute_bits = sum_bitstream_bits(bitstreams)
    if geometry_bits is None:
        geometry_bits = bitstream_size_bits(paths["geometry_bitstream"])
    if support_points is None:
        support_points = _count_ply_points(paths["decoded_support_ply"])
    if total_points is None:
        raise ValueError("total_points is required to compute bpp")
    return compute_rate_metrics(
        total_points=total_points,
        attribute_bits=attribute_bits,
        geometry_bits=geometry_bits,
        support_points=support_points,
    )


def _default_log(msg: str, *, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.2f}s] {msg}", flush=True)


def _gaussian_point_count(gaussian_dict: dict) -> int:
    primitives = gaussian_dict.get("decoded_primitives") or []
    if not primitives:
        return 0
    return int(primitives[0].shape[0])


def _summarize_gaussian_dict(gaussian_dict: dict) -> None:
    count = _gaussian_point_count(gaussian_dict)
    print(f"gaussian_points={count}")
    for key in ("decoded_primitives", "decoded_dc", "decoded_r", "decoded_s", "decoded_o"):
        values = gaussian_dict.get(key) or []
        if not values:
            continue
        tensor = values[0].detach().float().cpu()
        print(
            f"  {key}: shape={tuple(tensor.shape)} "
            f"mean={float(tensor.mean()):.6f} std={float(tensor.std()):.6f}"
        )


def _summarize_rate_metrics(rate_metrics: dict[str, float | int]) -> None:
    print(format_rate_metrics(rate_metrics))


def _print_codec_complete(
    *,
    rate_metrics: Optional[dict[str, float | int]],
    encode_sec: Optional[float],
    attribute_decode_sec: float,
    geometry_decode_sec: Optional[float],
) -> None:
    print("\n编解码结束", flush=True)
    if rate_metrics is not None:
        _summarize_rate_metrics(rate_metrics)
    if encode_sec is not None:
        print(f"encode_sec={encode_sec:.3f}")
    total_decode_sec = attribute_decode_sec
    if geometry_decode_sec is not None:
        total_decode_sec = geometry_decode_sec + attribute_decode_sec
    print(f"decode_sec={total_decode_sec:.3f}")


def run_step3(
    *,
    bitstream_files: Sequence[str | Path],
    decoded_support_ply: Path,
    ckpt: Path = ATTRIBUTE_CKPT_LEVEL8,
    level: int = DEFAULT_PIPELINE_LEVEL,
    outdir: Optional[Path] = None,
    mesh_gt: Optional[Path] = None,
    no_lattice: bool = True,
    enable_render: bool = True,
    total_points: Optional[int] = None,
    attribute_bits: Optional[int] = None,
    geometry_bits: Optional[int] = None,
    support_points: Optional[int] = None,
    sample_name: Optional[str] = None,
    encode_sec: Optional[float] = None,
    geometry_decode_sec: Optional[float] = None,
    fov: int = DEFAULT_FOV,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    super_sample_rate: int = DEFAULT_SUPER_SAMPLE_RATE,
    background_color: float = DEFAULT_BACKGROUND_COLOR,
    log: Optional[Callable[[str], None]] = None,
) -> Step3Result:
    t0 = time.perf_counter()
    log = log or (lambda msg: _default_log(msg, t0=t0))
    setup_attribute_env(no_lattice=no_lattice)

    bitstreams = [str(path) for path in bitstream_files]
    for path in bitstreams:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing attribute bitstream: {path}")
    if not decoded_support_ply.is_file():
        raise FileNotFoundError(f"Missing decoded support PLY: {decoded_support_ply}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    if outdir is None:
        raise ValueError("outdir is required to export decoded Gaussian PLY")

    install_attribute_backend(no_lattice=no_lattice, log=log)
    prewarm_attribute_extensions(log=log)
    adapter = load_attribute_adapter(str(ckpt), log=log)

    log(
        f"Decoding level {level} with {len(bitstreams)} bitstream(s) "
        f"on {decoded_support_ply.name}..."
    )
    decode_t0 = time.perf_counter()
    with gameleon_attribute_workdir():
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
    decode_sec = time.perf_counter() - decode_t0
    log(f"attribute decode finished in {decode_sec:.3f}s")

    _summarize_gaussian_dict(gaussian_dict)
    decode_timing = dict(decode_info.get("decode_timing", {}) or {})
    if decode_timing:
        print(
            "decode_timing "
            + " ".join(
                f"{key}={decode_timing.get(key)}"
                for key in (
                    "attribute_geom_prepare",
                    "attribute_entropy_decode",
                    "attribute_gaussian_build",
                    "attribute_decode_total",
                )
                if key in decode_timing
            )
        )
    stats = lattice_stats()
    if stats:
        print(
            "me_lattice_stats "
            + " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
        )

    log("Exporting decoded Gaussian PLY...")
    export_t0 = time.perf_counter()
    decoded_pcd_path = export_decoded_gaussian_ply(gaussian_dict, outdir, level=level)
    log(f"decoded Gaussian PLY written in {time.perf_counter() - export_t0:.3f}s")
    print(f"decoded_gaussian_ply={decoded_pcd_path}")

    rate_metrics: Optional[dict[str, float | int]] = None
    if sample_name is not None:
        rate_metrics = _resolve_rate_metrics(
            outdir=outdir,
            sample_name=sample_name,
            level=level,
            total_points=total_points,
            attribute_bits=attribute_bits,
            geometry_bits=geometry_bits,
            support_points=support_points,
        )

    _print_codec_complete(
        rate_metrics=rate_metrics,
        encode_sec=encode_sec,
        attribute_decode_sec=decode_sec,
        geometry_decode_sec=geometry_decode_sec,
    )

    render_info: dict[str, Any] = {}
    render_sec: Optional[float] = None
    summary_json: Optional[str] = None
    if enable_render:
        mesh_path = mesh_gt or DEFAULT_MESH_GT
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing GT mesh: {mesh_path}")
        print("\n=== Render PSNR (not counted in codec timing) ===", flush=True)
        render_log: Callable[[str], None] = lambda msg: print(msg, flush=True)
        render_t0 = time.perf_counter()
        render_info = run_post_decode_render(
            adapter,
            gaussian_dict,
            outdir=outdir,
            level=level,
            mesh_gt=mesh_path,
            fov=fov,
            width=width,
            height=height,
            super_sample_rate=super_sample_rate,
            background_color=background_color,
            decode_sec=decode_sec,
            decode_timing=decode_timing,
            rate_metrics=rate_metrics,
            write_decoded_ply=False,
            log=render_log,
        )
        render_sec = time.perf_counter() - render_t0
        render_psnr = render_info.get("render_psnr")
        if render_psnr is not None:
            print(f"render_psnr={float(render_psnr):.4f}")
            per_view = render_info.get("per_view_psnr") or []
            if per_view:
                print("per_view_psnr " + " ".join(f"{value:.4f}" for value in per_view))
        print(f"decoded_render_dir={render_info.get('render_dir')}")
        print(f"gt_render_dir={render_info.get('gt_render_dir')}")
        summary_json = render_info.get("summary_json")
    elif rate_metrics is not None:
        summary_path = write_decode_summary(
            outdir,
            level=level,
            rate_metrics=rate_metrics,
            decode_sec=decode_sec,
            decode_timing=decode_timing,
            decoded_pcd_path=str(decoded_pcd_path),
            gaussian_points=_gaussian_point_count(gaussian_dict),
        )
        summary_json = str(summary_path)
        print(f"summary_json={summary_json}")

    total_decode_sec = decode_sec
    if geometry_decode_sec is not None:
        total_decode_sec = geometry_decode_sec + decode_sec

    return Step3Result(
        gaussian_points=_gaussian_point_count(gaussian_dict),
        decode_sec=float(decode_sec),
        decode_timing=decode_timing,
        gaussian_keys=sorted(gaussian_dict.keys()),
        decoded_pcd_path=str(decoded_pcd_path),
        render_psnr=None if not render_info else render_info.get("render_psnr"),
        per_view_psnr=None if not render_info else render_info.get("per_view_psnr"),
        render_dir=None if not render_info else render_info.get("render_dir"),
        gt_render_dir=None if not render_info else render_info.get("gt_render_dir"),
        summary_json=summary_json,
        render_sec=render_sec,
        encode_sec=encode_sec,
        total_decode_sec=total_decode_sec,
        geometry_bpp=None if rate_metrics is None else float(rate_metrics["geometry_bpp"]),
        attribute_bpp=None if rate_metrics is None else float(rate_metrics["attribute_bpp"]),
        bpp=None if rate_metrics is None else float(rate_metrics["bpp"]),
    )


def run_step3_from_outdir(
    *,
    outdir: Path,
    sample_name: str,
    ckpt: Path = ATTRIBUTE_CKPT_LEVEL8,
    level: int = DEFAULT_PIPELINE_LEVEL,
    mesh_gt: Optional[Path] = None,
    no_lattice: bool = True,
    enable_render: bool = True,
    total_points: Optional[int] = None,
    attribute_bits: Optional[int] = None,
    geometry_bits: Optional[int] = None,
    support_points: Optional[int] = None,
    encode_sec: Optional[float] = None,
    geometry_decode_sec: Optional[float] = None,
    fov: int = DEFAULT_FOV,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    super_sample_rate: int = DEFAULT_SUPER_SAMPLE_RATE,
    background_color: float = DEFAULT_BACKGROUND_COLOR,
    log: Optional[Callable[[str], None]] = None,
) -> Step3Result:
    paths = pipeline_output_paths(outdir, sample_name, level=level)
    bitstreams = attribute_bitstream_paths(outdir, sample_name, level=level)
    missing = [path for path in bitstreams if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attribute bitstreams: {missing}")
    if not paths["decoded_support_ply"].is_file():
        raise FileNotFoundError(f"Missing decoded support PLY: {paths['decoded_support_ply']}")
    return run_step3(
        bitstream_files=bitstreams,
        decoded_support_ply=paths["decoded_support_ply"],
        ckpt=ckpt,
        level=level,
        outdir=outdir,
        mesh_gt=mesh_gt,
        no_lattice=no_lattice,
        enable_render=enable_render,
        total_points=total_points,
        attribute_bits=attribute_bits,
        geometry_bits=geometry_bits,
        support_points=support_points,
        sample_name=sample_name,
        encode_sec=encode_sec,
        geometry_decode_sec=geometry_decode_sec,
        fov=fov,
        width=width,
        height=height,
        super_sample_rate=super_sample_rate,
        background_color=background_color,
        log=log,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gameleon Step 3: attribute decode + PLY export.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-name", type=str, default="pcd_0")
    parser.add_argument("--level", type=int, default=DEFAULT_PIPELINE_LEVEL)
    parser.add_argument("--ckpt", type=Path, default=ATTRIBUTE_CKPT_LEVEL8)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PLY,
        help="Original input PLY (bpp denominator)",
    )
    parser.add_argument(
        "--mesh-gt",
        type=Path,
        default=DEFAULT_MESH_GT,
        help="GT mesh for render PSNR (Gameleon official path)",
    )
    parser.add_argument("--fov", type=int, default=DEFAULT_FOV)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--super-sample-rate", type=int, default=DEFAULT_SUPER_SAMPLE_RATE)
    parser.add_argument("--background-color", type=float, default=DEFAULT_BACKGROUND_COLOR)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip gsplat-mlx render/PSNR (still writes bpp to summary.json)",
    )
    parser.add_argument(
        "--lattice",
        action="store_true",
        help="Enable mlx-lattice ME acceleration (default: native ME CPU)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    print("step3 (attribute decode + PLY export + metrics)")
    print(f"outdir={args.outdir}")
    print(f"sample_name={args.sample_name}")
    print(f"level={args.level}")
    setup_attribute_env(no_lattice=not args.lattice)
    print(f"me_lattice={os.environ.get('GAMELEON_ME_LATTICE', '0')}")
    render_paths = render_output_paths(args.outdir, level=args.level)
    print(f"decoded_gaussian_ply={render_paths['decoded_gaussian_ply']}")
    enable_render = not args.no_render
    print(f"render={'1' if enable_render else '0'} mesh_gt={args.mesh_gt}")
    if enable_render:
        print(f"render_root={render_paths['render_root']}")

    run_step3_from_outdir(
        outdir=args.outdir,
        sample_name=args.sample_name,
        ckpt=args.ckpt,
        level=args.level,
        mesh_gt=args.mesh_gt,
        no_lattice=not args.lattice,
        enable_render=enable_render,
        total_points=_count_ply_points(args.input),
        fov=args.fov,
        width=args.width,
        height=args.height,
        super_sample_rate=args.super_sample_rate,
        background_color=args.background_color,
        log=lambda msg: _default_log(msg, t0=t0),
    )
    _default_log("step3 OK.", t0=t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
