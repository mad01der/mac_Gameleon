#!/usr/bin/env python3
"""CLI: render a 3D Gaussian Splatting PLY to PNG via gsplat-mlx (Metal)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 3DGS PLY file on Mac using gsplat-mlx (Metal).",
    )
    parser.add_argument("--ply", type=str, required=True, help="Input Gaussian PLY")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov", type=float, default=60.0, help="Vertical FOV (degrees)")
    parser.add_argument("--azimuth", type=float, default=0.0, help="Camera azimuth (degrees)")
    parser.add_argument("--elevation", type=float, default=15.0, help="Camera elevation (degrees)")
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--max-points", type=int, default=None, help="Cap Gaussians for quick tests")
    args = parser.parse_args()

    ply = Path(args.ply)
    if not ply.is_file():
        print(f"PLY not found: {ply}", file=sys.stderr)
        sys.exit(1)

    try:
        import mlx.core as mx  # noqa: F401
    except ImportError as exc:
        print(
            "mlx is not installed. Run: cd mac_Gameleon && ./scripts/setup_env.sh && source .venv/bin/activate",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from mac_gameleon.render_gsplat import render_gaussian_ply_to_png

    _img, elapsed = render_gaussian_ply_to_png(
        ply,
        args.output,
        width=int(args.width),
        height=int(args.height),
        fov_deg=float(args.fov),
        azimuth_deg=float(args.azimuth),
        elevation_deg=float(args.elevation),
        opacity_threshold=float(args.opacity_threshold),
        load_max_points=args.max_points,
    )
    out = Path(args.output).resolve()
    print(f"Rendered {ply} -> {out}")
    print(f"Elapsed: {elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    main()
