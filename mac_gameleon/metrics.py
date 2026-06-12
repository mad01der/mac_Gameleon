"""Gameleon-aligned rate metrics (bpp) for the Mac pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from mac_gameleon.paths import attribute_bitstream_paths, pipeline_output_paths


def bitstream_size_bits(path: Path) -> int:
    return int(path.stat().st_size * 8)


def sum_bitstream_bits(paths: Sequence[Path]) -> int:
    return sum(bitstream_size_bits(path) for path in paths)


def count_ply_points(ply_path: Path) -> int:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    return int(len(pcd.points))


def compute_rate_metrics(
    *,
    total_points: int,
    attribute_bits: int,
    geometry_bits: int,
    support_points: int,
) -> dict[str, float | int]:
    if total_points <= 0:
        raise ValueError(f"total_points must be positive, got {total_points}")
    if support_points <= 0:
        raise ValueError(f"support_points must be positive, got {support_points}")

    total = float(total_points)
    return {
        "total_points": int(total_points),
        "support_points": int(support_points),
        "attribute_bits": int(attribute_bits),
        "geometry_bits": int(geometry_bits),
        "geometry_bpp": float(geometry_bits / total),
        "geometry_bpp_support": float(geometry_bits / float(support_points)),
        "attribute_bpp": float(attribute_bits / total),
        "bpp": float((geometry_bits + attribute_bits) / total),
    }


def resolve_rate_metrics(
    outdir: Path,
    sample_name: str,
    *,
    level: int,
    total_points: int,
    support_points: Optional[int] = None,
    attribute_bits: Optional[int] = None,
    geometry_bits: Optional[int] = None,
) -> dict[str, float | int]:
    paths = pipeline_output_paths(outdir, sample_name, level=level)
    if attribute_bits is None:
        attribute_bits = sum_bitstream_bits(attribute_bitstream_paths(outdir, sample_name, level=level))
    if geometry_bits is None:
        geometry_bits = bitstream_size_bits(paths["geometry_bitstream"])
    if support_points is None:
        support_points = count_ply_points(paths["decoded_support_ply"])
    return compute_rate_metrics(
        total_points=total_points,
        attribute_bits=attribute_bits,
        geometry_bits=geometry_bits,
        support_points=support_points,
    )


def format_rate_metrics(metrics: dict[str, float | int]) -> str:
    return (
        f"geometry_bpp={metrics['geometry_bpp']:.6f} "
        f"attribute_bpp={metrics['attribute_bpp']:.6f} "
        f"bpp={metrics['bpp']:.6f}"
    )
