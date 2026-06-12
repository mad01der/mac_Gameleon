#!/usr/bin/env python3
"""Gameleon Mac pipeline: Step 1 + Step 2 + Step 3 (decode + Gaussian PLY export)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mac_gameleon.paths import (  # noqa: E402
    ATTRIBUTE_CKPT_LEVEL8,
    DEFAULT_INPUT_PLY,
    DEFAULT_MESH_GT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIPELINE_LEVEL,
    GEOMETRY_CKPT,
    pipeline_output_paths,
    render_output_paths,
)


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 1 + Step 2 + Step 3 with Gameleon-aligned outputs under outdir/.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PLY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--level", type=int, default=DEFAULT_PIPELINE_LEVEL)
    parser.add_argument("--attribute-ckpt", type=Path, default=ATTRIBUTE_CKPT_LEVEL8)
    parser.add_argument("--geometry-ckpt", type=Path, default=GEOMETRY_CKPT)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Subsample Step 1 input to N points (default: 0 = all points)",
    )
    parser.add_argument(
        "--no-lattice",
        action="store_true",
        help="Disable mlx-lattice in Step 1 (default: lattice enabled)",
    )
    parser.add_argument(
        "--step3-lattice",
        action="store_true",
        help="Enable mlx-lattice in Step 3 decode (default: native ME CPU)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip gsplat-mlx render/PSNR (still computes bpp and writes summary.json)",
    )
    return parser.parse_args(argv)


def _log(msg: str, *, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.2f}s] {msg}", flush=True)


def _print_outputs(outdir: Path, sample_name: str, level: int) -> None:
    paths = pipeline_output_paths(outdir, sample_name, level=level)
    render_paths = render_output_paths(outdir, level=level)
    print("\noutputs")
    print(f"  outdir={paths['outdir']}")
    print(f"  orig_attribute/")
    for idx in range(4):
        candidate = Path(f"{paths['attribute_prefix']}_level_{idx}.bin")
        if candidate.is_file():
            print(f"    {candidate.relative_to(outdir)}")
    print(f"    {paths['native_support_ply'].relative_to(outdir)}")
    print(f"  geometry/")
    print(f"    {paths['geometry_bitstream'].relative_to(outdir)}")
    print(f"    {paths['decoded_support_ply'].relative_to(outdir)}")
    print(f"  {render_paths['render_root'].name}/")
    print(f"    {render_paths['decoded_gaussian_ply'].relative_to(outdir)}")
    summary_path = render_paths["summary_json"]
    if summary_path.is_file():
        print(f"  {summary_path.relative_to(outdir)}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    step1 = _load_script_module("step1")
    step2 = _load_script_module("step2")
    step3 = _load_script_module("step3")
    sample_name = args.input.stem

    render_paths = render_output_paths(args.outdir, level=args.level)

    print("test.py: Step 1 + Step 2 + Step 3 (decode + Gaussian PLY export)")
    print(f"input={args.input}")
    print(f"outdir={args.outdir}")
    print(f"level={args.level}")
    print(f"max_points={args.max_points or 'all'}")
    print(f"step1_me_lattice={'0' if args.no_lattice else '1'}")
    print(f"step3_me_lattice={'1' if args.step3_lattice else '0'}")
    print(f"render={'0' if args.no_render else '1'}")
    print(f"decoded_gaussian_ply={render_paths['decoded_gaussian_ply']}")
    if not args.no_render:
        print(f"mesh_gt={DEFAULT_MESH_GT}")

    print("\n=== Step 1: attribute encode ===", flush=True)
    step1_result = step1.run_step1(
        input_ply=args.input,
        ckpt=args.attribute_ckpt,
        outdir=args.outdir,
        level=args.level,
        max_points=args.max_points,
        no_lattice=args.no_lattice,
        log=lambda msg: _log(msg, t0=t0),
    )
    _log(
        f"Step 1 done: streams={len(step1_result.bitstream_files)} "
        f"support={step1_result.native_support_ply.name}",
        t0=t0,
    )

    print("\n=== Step 2: lossless support geometry ===", flush=True)
    step2_result = step2.run_step2(
        input_ply=step1_result.native_support_ply,
        ckpt=args.geometry_ckpt,
        outdir=args.outdir,
        sample_name=sample_name,
        level=args.level,
        log=lambda msg: _log(msg, t0=t0),
    )
    _log(
        f"Step 2 done: bitstream={step2_result.bitstream_files[0].name} "
        f"decoded={step2_result.decoded_support_ply.name}",
        t0=t0,
    )

    print("\n=== Step 3: attribute decode + metrics ===", flush=True)
    enable_render = not args.no_render
    step3_result = step3.run_step3_from_outdir(
        outdir=args.outdir,
        sample_name=sample_name,
        ckpt=args.attribute_ckpt,
        level=args.level,
        mesh_gt=DEFAULT_MESH_GT if enable_render else None,
        no_lattice=not args.step3_lattice,
        enable_render=enable_render,
        total_points=step1_result.sparse_points,
        attribute_bits=step1_result.attribute_bits,
        geometry_bits=step2_result.geometry_bits,
        support_points=step1_result.geom_points,
        encode_sec=step1_result.encode_sec + step2_result.encode_sec,
        geometry_decode_sec=step2_result.decode_sec,
        log=lambda msg: _log(msg, t0=t0),
    )

    _print_outputs(args.outdir, sample_name, args.level)
    _log("test.py OK.", t0=t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
