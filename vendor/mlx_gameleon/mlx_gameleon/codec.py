from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import pairwise

import mlx.core as mx
import numpy as np

from mlx_gameleon.attributes import (
    AttributeDecodeResult,
    GameleonAttributeDecoder,
    make_dummy_attribute_latents,
)
from mlx_gameleon.bitstream import (
    GeometryPayload,
    pack_payload,
    unpack_payload,
)
from mlx_gameleon.model import GameleonGeometryModel
from mlx_gameleon.ops import (
    final_points_from_occupancy,
    occupancy_pyramid,
    sort_coords_feats,
)


@dataclass(frozen=True, slots=True)
class CodecResult:
    byte_stream: bytes
    seconds: float
    layers: int
    points: int


class GameleonGeometryCodec:
    """Codec-facing wrapper around the MLX Gameleon geometry network."""

    def __init__(
        self,
        *,
        channels: int = 32,
        kernel_size: int = 3,
        pos_q: float = 1.0,
    ) -> None:
        self.model = GameleonGeometryModel(
            channels=channels,
            kernel_size=kernel_size,
        )
        self.pos_q = float(pos_q)

    def compress(self, xyz: np.ndarray | mx.array) -> CodecResult:
        coords = _quantized_coords(xyz, self.pos_q)
        start = time.perf_counter()
        levels = occupancy_pyramid(coords)
        stage_occupancy = []
        for current, target in pairwise(levels):
            _ = self.model.predict_stage(
                current.active_coords(), current.active_occupancy()
            )
            _, target_occupancy = sort_coords_feats(
                target.active_coords(),
                target.active_occupancy().reshape((-1, 1)),
            )
            stage_occupancy.append(
                np.asarray(target_occupancy[:, 0].tolist(), dtype=np.uint8)
            )
        mx.eval()
        seconds = time.perf_counter() - start
        base = levels[0]
        payload = GeometryPayload(
            pos_q=self.pos_q,
            base_coords=np.asarray(
                base.active_coords()[:, 1:].tolist(), dtype=np.int32
            ),
            base_occupancy=np.asarray(
                base.active_occupancy().tolist(), dtype=np.uint8
            ),
            stage_occupancy=stage_occupancy,
        )
        return CodecResult(
            byte_stream=pack_payload(payload),
            seconds=seconds,
            layers=len(stage_occupancy),
            points=int(coords.shape[0]),
        )

    def decompress(self, stream: bytes) -> tuple[np.ndarray, float]:
        payload = unpack_payload(stream)
        coords = _batched_coords(payload.base_coords)
        occupancy = mx.array(payload.base_occupancy, dtype=mx.int32)
        start = time.perf_counter()
        for stage in payload.stage_occupancy:
            prediction = self.model.predict_stage(coords, occupancy)
            coords = prediction.coords
            occupancy = mx.array(stage, dtype=mx.int32)
        points = final_points_from_occupancy(coords, occupancy)
        mx.eval(points)
        seconds = time.perf_counter() - start
        xyz = np.asarray(points.tolist(), dtype=np.float32) * payload.pos_q
        return xyz, seconds

    def network_bpp(self, xyz: np.ndarray | mx.array) -> float:
        coords = _quantized_coords(xyz, self.pos_q)
        bpp = self.model.bits_per_point(coords)
        mx.eval(bpp)
        return float(bpp.tolist())


class GameleonCodec(GameleonGeometryCodec):
    """Combined geometry + attribute decode workload reproduction."""

    def __init__(
        self,
        *,
        channels: int = 32,
        kernel_size: int = 3,
        pos_q: float = 1.0,
        attribute_channels: int | None = None,
        attribute_level: int = 9,
    ) -> None:
        super().__init__(
            channels=channels,
            kernel_size=kernel_size,
            pos_q=pos_q,
        )
        self.attribute_decoder = GameleonAttributeDecoder(
            context_channels=attribute_channels or 64,
            kernel_size=kernel_size,
            level=attribute_level,
        )

    def compress(self, xyz: np.ndarray | mx.array) -> CodecResult:
        start = time.perf_counter()
        geometry = super().compress(xyz)
        coords = _quantized_coords(xyz, self.pos_q)
        support_coords = _attribute_support_coords(
            coords,
            level=self.attribute_decoder.level,
        )
        latent_shapes = self.attribute_decoder.latent_shapes(support_coords)
        attribute_latents = make_dummy_attribute_latents(latent_shapes)
        payload = unpack_payload(geometry.byte_stream)
        combined = GeometryPayload(
            pos_q=payload.pos_q,
            base_coords=payload.base_coords,
            base_occupancy=payload.base_occupancy,
            stage_occupancy=payload.stage_occupancy,
            attribute_latents=attribute_latents,
        )
        return CodecResult(
            byte_stream=pack_payload(combined),
            seconds=time.perf_counter() - start,
            layers=geometry.layers,
            points=geometry.points,
        )

    def decompress(
        self,
        stream: bytes,
    ) -> tuple[np.ndarray, AttributeDecodeResult, float]:
        payload = unpack_payload(stream)
        coords = _batched_coords(payload.base_coords)
        occupancy = mx.array(payload.base_occupancy, dtype=mx.int32)
        start = time.perf_counter()
        support_coords = None
        support_stage = _attribute_support_stage_index(
            self.attribute_decoder.level,
            len(payload.stage_occupancy),
        )
        for stage_index, stage in enumerate(payload.stage_occupancy):
            prediction = self.model.predict_stage(coords, occupancy)
            coords = prediction.coords
            occupancy = mx.array(stage, dtype=mx.int32)
            if stage_index == support_stage:
                support_coords = coords
        points = final_points_from_occupancy(coords, occupancy)
        if support_coords is None:
            support_coords = _attribute_support_coords(
                _batched_coords(points.astype(mx.int32)),
                level=self.attribute_decoder.level,
            )
        latents = payload.attribute_latents
        if latents is None:
            latents = make_dummy_attribute_latents(
                self.attribute_decoder.latent_shapes(support_coords)
            )
        attributes = self.attribute_decoder.decode(support_coords, latents)
        mx.eval(points, attributes.features)
        seconds = time.perf_counter() - start
        xyz = np.asarray(points.tolist(), dtype=np.float32) * payload.pos_q
        return xyz, attributes, seconds


def _quantized_coords(xyz: np.ndarray | mx.array, pos_q: float) -> mx.array:
    arr = xyz if isinstance(xyz, mx.array) else mx.array(xyz)
    coords = mx.round(arr / pos_q).astype(mx.int32)
    return mx.concatenate(
        [mx.zeros((coords.shape[0], 1), dtype=mx.int32), coords],
        axis=1,
    )


def _batched_coords(xyz: np.ndarray) -> mx.array:
    coords = mx.array(xyz, dtype=mx.int32)
    return mx.concatenate(
        [mx.zeros((coords.shape[0], 1), dtype=mx.int32), coords],
        axis=1,
    )


def _attribute_support_coords(coords: mx.array, *, level: int) -> mx.array:
    factor = 1 << (10 - level)
    if factor == 1:
        return coords.astype(mx.int32)
    support = mx.concatenate(
        [coords[:, :1], coords[:, 1:] // factor],
        axis=1,
    ).astype(mx.int32)
    support, _ = sort_coords_feats(
        support,
        mx.zeros((support.shape[0], 1), dtype=mx.float32),
    )
    keep = mx.concatenate(
        [
            mx.array([True]),
            mx.any(support[1:] != support[:-1], axis=1),
        ],
        axis=0,
    )
    count = int(mx.sum(keep).tolist())
    rows = mx.argsort(keep.astype(mx.int32)).astype(mx.int32)[-count:]
    return mx.take(support, rows, axis=0)


def _attribute_support_stage_index(
    attribute_level: int,
    geometry_stages: int,
) -> int:
    return geometry_stages - (9 - attribute_level) - 1
