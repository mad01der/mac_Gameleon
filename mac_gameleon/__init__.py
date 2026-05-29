"""Mac Gameleon: gsplat-mlx rendering and Gameleon Mac CPU integration helpers."""

from mac_gameleon.paths import (
    DEFAULT_EXAMPLE_DIR,
    DEFAULT_INPUT_PLY,
    DEFAULT_MESH_GT,
    GAMELEON_PACKAGE_ROOT,
    GAMELEON_ROOT,
    GEOMETRY_CKPT,
    MAC_GAMELEON_ROOT,
    required_paths,
)
from mac_gameleon.ply_gaussian import GaussianPlyData, load_gaussian_ply
from mac_gameleon.render_gsplat import render_gaussian_ply_to_png

__all__ = [
    "DEFAULT_EXAMPLE_DIR",
    "DEFAULT_INPUT_PLY",
    "DEFAULT_MESH_GT",
    "GAMELEON_PACKAGE_ROOT",
    "GAMELEON_ROOT",
    "GEOMETRY_CKPT",
    "GaussianPlyData",
    "MAC_GAMELEON_ROOT",
    "load_gaussian_ply",
    "render_gaussian_ply_to_png",
    "required_paths",
]
