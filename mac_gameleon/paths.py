"""Canonical paths for mac_Gameleon ↔ Gameleon integration."""

from __future__ import annotations

from pathlib import Path

MAC_GAMELEON_ROOT = Path(__file__).resolve().parents[1]
GAMELEON_ROOT = MAC_GAMELEON_ROOT.parent / "Gameleon"
GAMELEON_PACKAGE_ROOT = GAMELEON_ROOT / "gameleon"
GAMELEON_ATTRIBUTE_ROOT = GAMELEON_PACKAGE_ROOT / "gameleon_attribute"

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

LONGDRESS_PLY = GAMELEON_ROOT / "examples" / "data" / "longdress" / "longdress_vox10_1300.ply"
LONGDRESS_MESH = GAMELEON_ROOT / "examples" / "data" / "longdress" / "longdress.obj"


def required_paths() -> dict[str, Path]:
    return {
        "gameleon_root": GAMELEON_ROOT,
        "gameleon_package_root": GAMELEON_PACKAGE_ROOT,
        "geometry_ckpt": GEOMETRY_CKPT,
        "attribute_ckpt_level8": ATTRIBUTE_CKPT_LEVEL8,
        "longdress_ply": LONGDRESS_PLY,
        "longdress_mesh": LONGDRESS_MESH,
    }
