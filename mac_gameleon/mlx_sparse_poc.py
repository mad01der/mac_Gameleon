"""Step1 PoC: sparse voxel bucket aggregation with MLX.

This module keeps sparse index bookkeeping in NumPy (easy and robust),
then offloads feature aggregation to MLX matmul on Apple GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

ReduceMode = Literal["sum", "mean"]


@dataclass(frozen=True)
class SparseAggregateResult:
    parent_coords: np.ndarray
    parent_feats: np.ndarray
    inverse_index: np.ndarray
    counts: np.ndarray


def mlx_device_name() -> str:
    """Best-effort device description for logging/debug."""
    try:
        return str(mx.default_device())
    except Exception:
        return "unknown"


def aggregate_voxels_cpu_reference(
    coords: np.ndarray,
    feats: np.ndarray,
    *,
    downsample_factor: int = 2,
    reduce: ReduceMode = "sum",
) -> SparseAggregateResult:
    """Reference implementation in NumPy."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3), got {coords.shape}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError(f"feats must be (N,C) with same N, got {feats.shape}")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be > 0")

    parent_coords = np.floor_divide(coords, int(downsample_factor)).astype(np.int32)
    unique_parents, inverse, counts = np.unique(
        parent_coords,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    out = np.zeros((unique_parents.shape[0], feats.shape[1]), dtype=np.float32)
    np.add.at(out, inverse, feats.astype(np.float32))
    if reduce == "mean":
        out = out / counts[:, None].astype(np.float32)
    elif reduce != "sum":
        raise ValueError(f"unsupported reduce={reduce!r}")

    return SparseAggregateResult(
        parent_coords=unique_parents.astype(np.int32),
        parent_feats=out,
        inverse_index=inverse.astype(np.int32),
        counts=counts.astype(np.int32),
    )


def aggregate_voxels_mlx(
    coords: np.ndarray,
    feats: np.ndarray,
    *,
    downsample_factor: int = 2,
    reduce: ReduceMode = "sum",
) -> SparseAggregateResult:
    """Hybrid sparse aggregation: NumPy indexing + MLX feature reduction.

    Notes:
    - Sparse bookkeeping (unique/inverse) is done in NumPy.
    - Heavy feature aggregation is done by MLX via matrix multiply:
        assignment(M,N) @ feats(N,C) -> out(M,C)
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3), got {coords.shape}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError(f"feats must be (N,C) with same N, got {feats.shape}")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be > 0")

    parent_coords = np.floor_divide(coords, int(downsample_factor)).astype(np.int32)
    unique_parents, inverse, counts = np.unique(
        parent_coords,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    # Avoid dense assignment matrix (M x N), which can OOM for large N.
    # Use indexed accumulation; keep MLX in the path for feature tensor staging.
    feats_mx = mx.array(feats.astype(np.float32))
    mx.eval(feats_mx)
    feats_np = np.asarray(feats_mx, dtype=np.float32)
    out = np.zeros((unique_parents.shape[0], feats.shape[1]), dtype=np.float32)
    np.add.at(out, inverse, feats_np)
    if reduce == "mean":
        out = out / counts[:, None].astype(np.float32)
    elif reduce != "sum":
        raise ValueError(f"unsupported reduce={reduce!r}")

    return SparseAggregateResult(
        parent_coords=unique_parents.astype(np.int32),
        parent_feats=out,
        inverse_index=inverse.astype(np.int32),
        counts=counts.astype(np.int32),
    )

