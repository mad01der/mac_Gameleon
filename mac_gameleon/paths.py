"""Canonical paths for mac_Gameleon ↔ Gameleon integration."""

from __future__ import annotations

from pathlib import Path

MAC_GAMELEON_ROOT = Path(__file__).resolve().parents[1]
GAMELEON_ROOT = MAC_GAMELEON_ROOT.parent / "Gameleon"
GAMELEON_PACKAGE_ROOT = GAMELEON_ROOT / "gameleon"
GAMELEON_ATTRIBUTE_ROOT = GAMELEON_PACKAGE_ROOT / "gameleon_attribute"

MAC_EXAMPLES_ROOT = MAC_GAMELEON_ROOT / "examples"
DEFAULT_OUTPUT_DIR = MAC_GAMELEON_ROOT / "outputs"
DEFAULT_PIPELINE_LEVEL = 8

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

# Backward-compatible aliases (same layout as original Gameleon outdir).
ORIG_ATTRIBUTE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "orig_attribute"
GEOMETRY_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "geometry"


def pipeline_output_paths(
    outdir: Path,
    sample_name: str,
    level: int = DEFAULT_PIPELINE_LEVEL,
) -> dict[str, Path]:
    """Gameleon-aligned artifact paths under a single outdir."""
    orig_attribute_dir = outdir / "orig_attribute"
    geometry_dir = outdir / "geometry"
    attribute_prefix = orig_attribute_dir / sample_name
    return {
        "outdir": outdir,
        "orig_attribute_dir": orig_attribute_dir,
        "geometry_dir": geometry_dir,
        "attribute_prefix": attribute_prefix,
        "native_support_ply": Path(f"{attribute_prefix}_level_{level}_geom.ply"),
        "geometry_bitstream": geometry_dir / f"{sample_name}_level_{level}_support.bin",
        "decoded_support_ply": geometry_dir / f"{sample_name}_level_{level}_support_dec.ply",
    }


def render_output_paths(
    outdir: Path,
    level: int = DEFAULT_PIPELINE_LEVEL,
) -> dict[str, Path]:
    """Post-decode render artifacts (Gameleon-aligned layout)."""
    render_root = outdir / f"render_level_{level}_seq"
    return {
        "render_root": render_root,
        "decoded_render_dir": render_root / "render",
        "gt_render_dir": outdir / "gt_render_mesh",
        "decoded_gaussian_ply": render_root / "decoded_gaussians_seq.ply",
        "summary_json": outdir / "summary.json",
    }


def attribute_bitstream_paths(
    outdir: Path,
    sample_name: str,
    level: int = DEFAULT_PIPELINE_LEVEL,
) -> list[Path]:
    paths = pipeline_output_paths(outdir, sample_name, level=level)
    required_streams = max(1, int(level) - 6)
    return [
        Path(f"{paths['attribute_prefix']}_level_{idx}.bin")
        for idx in range(required_streams)
    ]


def required_paths() -> dict[str, Path]:
    return {
        "gameleon_root": GAMELEON_ROOT,
        "gameleon_package_root": GAMELEON_PACKAGE_ROOT,
        "geometry_ckpt": GEOMETRY_CKPT,
        "attribute_ckpt_level8": ATTRIBUTE_CKPT_LEVEL8,
        "default_input_ply": DEFAULT_INPUT_PLY,
        "default_mesh_gt": DEFAULT_MESH_GT,
        "default_output_dir": DEFAULT_OUTPUT_DIR,
    }
