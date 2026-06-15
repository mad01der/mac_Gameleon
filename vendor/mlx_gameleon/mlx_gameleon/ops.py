from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    morton_order,
    occupancy_downsample,
    occupancy_expand,
)


@dataclass(frozen=True, slots=True)
class OccupancyLevel:
    coords: mx.array
    occupancy: mx.array
    active_rows: mx.array

    @property
    def count(self) -> int:
        return int(self.active_rows.tolist()[0])

    def active_coords(self) -> mx.array:
        return self.coords[: self.count]

    def active_occupancy(self) -> mx.array:
        return self.occupancy[: self.count]


def sort_coords_feats(
    coords: mx.array,
    feats: mx.array,
) -> tuple[mx.array, mx.array]:
    order = morton_order(coords)
    return mx.take(coords, order, axis=0), mx.take(feats, order, axis=0)


def sort_sparse_tensor(x: SparseTensor) -> SparseTensor:
    coords, feats = sort_coords_feats(x.coords, x.feats)
    return SparseTensor(coords, feats, stride=x.stride)


def occupancy_pyramid(
    coords: mx.array,
    *,
    min_rows: int = 64,
) -> list[OccupancyLevel]:
    active = mx.array([coords.shape[0]], dtype=mx.int32)
    levels = []
    while True:
        down = occupancy_downsample(coords, active)
        mx.eval(down.coords, down.occupancy, down.active_rows)
        level = OccupancyLevel(
            down.coords, down.occupancy, down.active_rows
        )
        levels.append(level)
        if level.count < min_rows:
            break
        coords = level.active_coords()
        active = level.active_rows
    return levels[::-1]


def expand_occupancy_level(
    coords: mx.array,
    occupancy: mx.array,
    features: mx.array | None = None,
) -> tuple[mx.array, mx.array | None]:
    expanded = occupancy_expand(coords, occupancy.astype(mx.int32))
    active_count = int(expanded.active_rows.tolist()[0])
    child_coords = expanded.coords[:active_count]
    if features is None:
        return child_coords, None
    child_features = mx.take(
        features, expanded.parent_rows[:active_count], axis=0
    )
    return child_coords, child_features


def checkerboard_groups(coords: mx.array) -> tuple[mx.array, mx.array]:
    parity = (coords[:, 1] + coords[:, 2] + coords[:, 3]) % 2
    group1 = parity == 0
    return group1, mx.logical_not(group1)


def sparse_from_occupancy(
    coords: mx.array,
    occupancy: mx.array | None = None,
) -> SparseTensor:
    if occupancy is None:
        feats = mx.ones((coords.shape[0], 1), dtype=mx.float32)
    else:
        feats = occupancy.astype(mx.float32).reshape((-1, 1))
    return SparseTensor(coords.astype(mx.int32), feats)


def final_points_from_occupancy(
    coords: mx.array,
    occupancy: mx.array,
) -> mx.array:
    points, _ = expand_occupancy_level(coords, occupancy)
    if points is None:
        return mx.zeros((0, 3), dtype=mx.int32)
    return points[:, 1:].astype(mx.int32)
