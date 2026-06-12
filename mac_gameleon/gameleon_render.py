"""Gameleon mesh GT rendering on Mac (ray-mesh intersection, no CUDA rasterizer)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from mac_gameleon.attribute_bootstrap import gameleon_attribute_workdir
from mac_gameleon.camera import build_cardinal_cameras
from mac_gameleon.paths import GAMELEON_ATTRIBUTE_ROOT, GAMELEON_PACKAGE_ROOT

DEFAULT_NUM_VIEWS = 4


def _ensure_gameleon_paths() -> None:
    for path in (GAMELEON_PACKAGE_ROOT, GAMELEON_ATTRIBUTE_ROOT):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def build_gameleon_camera(
    *,
    fov: int = 45,
    width: int = 512,
    height: int = 512,
    num_views: int = DEFAULT_NUM_VIEWS,
):
    """Build Gameleon Camera + gsplat view/intrinsic arrays from cardinal side views."""
    _ensure_gameleon_paths()
    from gameleon_attribute.structures import Camera

    viewmats, intrinsics = build_cardinal_cameras(
        num_views=num_views,
        width=width,
        height=height,
        fov_deg=float(fov),
    )
    c2w = np.linalg.inv(viewmats).astype(np.float32)
    camera = Camera(
        H_c2w=torch.from_numpy(c2w[None, ...]),
        intrinsic=torch.from_numpy(intrinsics[None, ...].astype(np.float32)),
        width_px=int(width),
        height_px=int(height),
    )
    return camera, viewmats, intrinsics


def render_mesh_gt_gameleon(
    mesh_path: str | Path,
    output_dir: Path,
    *,
    fov: int = 45,
    width: int = 512,
    height: int = 512,
    num_views: int = DEFAULT_NUM_VIEWS,
    background_color: float = 1.0,
    manual_scale: bool = False,
    mesh_input_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    log: Optional[Callable[[str], None]] = None,
) -> float:
    """Render GT RGB views with Gameleon mesh ray intersection (official eval path)."""
    _ensure_gameleon_paths()
    from gameleon_attribute.simple_raw_render import get_gt, save_pic

    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing GT mesh: {mesh_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if log is not None:
        log(
            f"GT mesh render: {mesh_path.name} "
            f"({num_views} views, {width}x{height}, Gameleon ray-mesh)..."
        )

    camera, _viewmats, _intrinsics = build_gameleon_camera(
        fov=fov,
        width=width,
        height=height,
        num_views=num_views,
    )
    offset = np.array(mesh_input_offset, dtype=np.float32)

    t0 = time.perf_counter()
    with gameleon_attribute_workdir():
        mesh_gt = get_gt(
            str(mesh_path),
            camera,
            manual_scale=manual_scale,
            input_offset=offset,
        )
        mesh_gt_rgb = mesh_gt["ray_rgbs"]
        hit_map = mesh_gt["hit_map"].unsqueeze(-1)
        bg = torch.zeros(3, dtype=mesh_gt_rgb.dtype, device=mesh_gt_rgb.device) + float(background_color)
        mesh_gt_rgb = mesh_gt_rgb + (1.0 - hit_map) * bg
        save_pic(mesh_gt_rgb, str(output_dir), type="rgb")

    elapsed = time.perf_counter() - t0
    if log is not None:
        log(f"GT mesh render done in {elapsed:.1f}s -> {output_dir}")
    return elapsed
