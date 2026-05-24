"""Mac Gameleon: gsplat-mlx rendering and Gameleon Mac CPU integration helpers."""

from mac_gameleon.paths import (
    ATTRIBUTE_CKPT_LEVEL8,
    ATTRIBUTE_CKPT_LEVEL9,
    GAMELEON_ATTRIBUTE_ROOT,
    GAMELEON_PACKAGE_ROOT,
    GAMELEON_ROOT,
    GEOMETRY_CKPT,
    LONGDRESS_MESH,
    LONGDRESS_PLY,
    MAC_GAMELEON_ROOT,
    required_paths,
)
from mac_gameleon.ply_gaussian import GaussianPlyData, load_gaussian_ply
from mac_gameleon.render_gsplat import render_gaussian_ply_to_png

__all__ = [
    "ATTRIBUTE_CKPT_LEVEL8",
    "ATTRIBUTE_CKPT_LEVEL9",
    "GAMELEON_ATTRIBUTE_ROOT",
    "GAMELEON_PACKAGE_ROOT",
    "GAMELEON_ROOT",
    "GEOMETRY_CKPT",
    "GaussianPlyData",
    "LONGDRESS_MESH",
    "LONGDRESS_PLY",
    "MAC_GAMELEON_ROOT",
    "load_gaussian_ply",
    "render_gaussian_ply_to_png",
    "required_paths",
]
