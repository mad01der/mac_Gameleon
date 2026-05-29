"""Canonical paths for mac_Gameleon ↔ Gameleon integration."""

from __future__ import annotations

from pathlib import Path

MAC_GAMELEON_ROOT = Path(__file__).resolve().parents[1]
GAMELEON_ROOT = MAC_GAMELEON_ROOT.parent / "Gameleon"
GAMELEON_PACKAGE_ROOT = GAMELEON_ROOT / "gameleon"
GAMELEON_ATTRIBUTE_ROOT = GAMELEON_PACKAGE_ROOT / "gameleon_attribute"

MAC_EXAMPLES_ROOT = MAC_GAMELEON_ROOT / "examples"
GEOMETRY_OUTPUT_DIR = MAC_GAMELEON_ROOT / "outputs" / "geometry"

# Default Mac test frame (sequence-style: pcd_0.ply + mesh .obj).
DEFAULT_EXAMPLE_DIR = MAC_EXAMPLES_ROOT / "0519"
DEFAULT_INPUT_PLY = DEFAULT_EXAMPLE_DIR / "pcd_0.ply"
DEFAULT_MESH_GT = DEFAULT_EXAMPLE_DIR / "0519.obj"

GEOMETRY_CKPT = (
    GAMELEON_PACKAGE_ROOT
    / "weights"
    / "geometry"
    / "gameleon_lossless_geometry"
    / "best_model_UCM.pt"
)
ATTRIBUTE_CKPT_LEVEL8 = (
    GAMELEON_PACKAGE_ROOT
    / "weights"
    / "attribute"
    / "bpp_0.39"
    / "checkpoint"
    / "epoch4.pth"
)
ATTRIBUTE_CKPT_LEVEL9 = (
    GAMELEON_PACKAGE_ROOT
    / "weights"
    / "attribute"
    / "level9_w001_epoch23"
    / "checkpoint"
    / "epoch23.pth"
)


def required_paths() -> dict[str, Path]:
    return {
        "gameleon_root": GAMELEON_ROOT,
        "gameleon_package_root": GAMELEON_PACKAGE_ROOT,
        "geometry_ckpt": GEOMETRY_CKPT,
        "attribute_ckpt_level8": ATTRIBUTE_CKPT_LEVEL8,
        "default_input_ply": DEFAULT_INPUT_PLY,
        "default_mesh_gt": DEFAULT_MESH_GT,
    }
