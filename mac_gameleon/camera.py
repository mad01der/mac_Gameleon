"""Simple pinhole cameras for gsplat-mlx (world-to-camera view matrices)."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return (v / n).astype(np.float32)


def look_at_world_to_camera(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Return world-to-camera 4x4 matrix (OpenCV-style, gsplat-mlx convention)."""
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    eye = eye.astype(np.float32)
    target = target.astype(np.float32)
    up = up.astype(np.float32)

    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    true_up = _normalize(np.cross(right, forward))

    # Camera looks down +Z in camera space; rows are camera axes in world coords.
    R = np.stack([right, true_up, forward], axis=0).astype(np.float32)
    t = -R @ eye
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def intrinsics_from_fov(
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    """Pinhole K matrix from vertical-agnostic symmetric FOV (use min dimension)."""
    fov_rad = math.radians(float(fov_deg))
    focal = 0.5 * float(min(width, height)) / math.tan(0.5 * fov_rad)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def gameleon_intrinsics_from_fov(
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    """Match Gameleon `derive_camera_intrinsics` (focal from width)."""
    focal = 0.5 * float(width) / math.tan(0.5 * math.radians(float(fov_deg)))
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


# Cardinal view azimuths (degrees): front, left, back, right.
CARDINAL_AZIMUTHS_DEG = (0.0, 270.0, 180.0, 90.0)
CARDINAL_VIEW_NAMES = ("front", "left", "back", "right")


def build_cardinal_cameras(
    *,
    num_views: int = 4,
    width: int = 512,
    height: int = 512,
    fov_deg: float = 45.0,
    distance: float = 3.5,
    target: np.ndarray | None = None,
    elevation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Four side views (front/left/back/right) looking at the scene center."""
    if num_views != 4:
        raise ValueError(f"build_cardinal_cameras expects 4 views, got {num_views}")
    center = np.zeros(3, dtype=np.float32) if target is None else target.astype(np.float32)
    intrinsic = gameleon_intrinsics_from_fov(width, height, fov_deg)
    intrinsics = np.repeat(intrinsic[None, :, :], 4, axis=0)

    viewmats = []
    el = math.radians(float(elevation_deg))
    dist = float(distance)
    for az_deg in CARDINAL_AZIMUTHS_DEG:
        az = math.radians(float(az_deg))
        eye = center + np.array(
            [
                dist * math.cos(el) * math.sin(az),
                dist * math.sin(el),
                dist * math.cos(el) * math.cos(az),
            ],
            dtype=np.float32,
        )
        viewmats.append(look_at_world_to_camera(eye, center))

    return np.stack(viewmats, axis=0).astype(np.float32), intrinsics


def orbit_camera_around_scene(
    means: np.ndarray,
    width: int,
    height: int,
    fov_deg: float = 60.0,
    distance_scale: float = 2.5,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place camera on a sphere looking at the scene centroid."""
    center = means.mean(axis=0).astype(np.float32)
    extent = np.linalg.norm(means - center, axis=1)
    radius = float(np.percentile(extent, 90)) if extent.size else 1.0
    radius = max(radius, 1e-3)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    dist = float(distance_scale) * radius / math.tan(math.radians(fov_deg) * 0.5)

    eye = center + np.array(
        [
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el),
            dist * math.cos(el) * math.cos(az),
        ],
        dtype=np.float32,
    )
    viewmat = look_at_world_to_camera(eye, center)
    K = intrinsics_from_fov(width, height, fov_deg)
    return viewmat, K
