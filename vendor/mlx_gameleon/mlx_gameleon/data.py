from __future__ import annotations

import numpy as np


def make_dummy_point_cloud(
    points: int,
    *,
    extent: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate unique integer coordinates for geometry network benchmarks."""
    if points <= 0:
        raise ValueError('points must be positive')
    rng = np.random.default_rng(seed)
    side = int(extent or max(16, np.ceil(points ** (1.0 / 3.0)) * 4))
    coords = rng.integers(0, side, size=(points * 2, 3), dtype=np.int32)
    unique = np.unique(coords, axis=0)
    while unique.shape[0] < points:
        extra = rng.integers(0, side, size=(points, 3), dtype=np.int32)
        unique = np.unique(np.concatenate([unique, extra], axis=0), axis=0)
    return unique[:points].astype(np.float32, copy=False)
