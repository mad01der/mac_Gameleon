"""Render 3D Gaussian PLY files with gsplat-mlx on Apple Silicon."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
from gsplat_mlx.rendering import rasterization
from PIL import Image

from mac_gameleon.camera import orbit_camera_around_scene
from mac_gameleon.ply_gaussian import GaussianPlyData, load_gaussian_ply


def _to_mx(array: np.ndarray) -> mx.array:
    return mx.array(array.astype(np.float32))


def render_gaussian_data(
    data: GaussianPlyData,
    output_png: str | Path,
    *,
    width: int = 512,
    height: int = 512,
    fov_deg: float = 60.0,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 15.0,
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    opacity_threshold: float = 0.01,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    means = data.means
    quats = data.quats
    scales = data.scales
    opacities = data.opacities
    sh_coeffs = data.sh_coeffs

    if max_points is not None and means.shape[0] > max_points:
        means = means[:max_points]
        quats = quats[:max_points]
        scales = scales[:max_points]
        opacities = opacities[:max_points]
        sh_coeffs = sh_coeffs[:max_points]

    if opacity_threshold > 0.0:
        keep = opacities >= float(opacity_threshold)
        means = means[keep]
        quats = quats[keep]
        scales = scales[keep]
        opacities = opacities[keep]
        sh_coeffs = sh_coeffs[keep]

    if means.shape[0] == 0:
        raise ValueError("No Gaussians left after filtering")

    view_np, k_np = orbit_camera_around_scene(
        means,
        width=width,
        height=height,
        fov_deg=fov_deg,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )

    means_mx = _to_mx(means)
    quats_mx = _to_mx(quats)
    scales_mx = _to_mx(scales)
    opacities_mx = _to_mx(opacities)
    colors_mx = _to_mx(sh_coeffs)
    view_mx = _to_mx(view_np)[None, :, :]
    k_mx = _to_mx(k_np)[None, :, :]
    backgrounds = _to_mx(np.array([background], dtype=np.float32))

    t0 = time.perf_counter()
    render_colors, render_alphas, _info = rasterization(
        means=means_mx,
        quats=quats_mx,
        scales=scales_mx,
        opacities=opacities_mx,
        colors=colors_mx,
        viewmats=view_mx,
        Ks=k_mx,
        width=int(width),
        height=int(height),
        backgrounds=backgrounds,
        render_mode="RGB",
        sh_degree=int(data.sh_degree),
        differentiable=False,
    )
    mx.eval(render_colors, render_alphas)
    elapsed = time.perf_counter() - t0

    img = np.array(render_colors[0])
    img = np.clip(img, 0.0, 1.0)
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((img * 255.0).astype(np.uint8)).save(out)
    return img, elapsed


def render_gaussian_ply_to_png(
    ply_path: str | Path,
    output_png: str | Path,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    data = load_gaussian_ply(ply_path, max_points=kwargs.pop("load_max_points", None))
    return render_gaussian_data(data, output_png, **kwargs)
