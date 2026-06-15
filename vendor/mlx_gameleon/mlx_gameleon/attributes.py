from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lattice import SparseTensor
from mlx_lattice.nn import ReLU, SubmConv3d
from mlx_lattice.ops import lookup_coords, occupancy_downsample

from mlx_gameleon.layers import ResidualBlock, SparseSequential
from mlx_gameleon.ops import sort_coords_feats


@dataclass(frozen=True, slots=True)
class AttributeDecodeResult:
    primitives: mx.array
    dc: mx.array
    rotation: mx.array
    scale: mx.array
    opacity: mx.array
    features: mx.array


@dataclass(frozen=True, slots=True)
class AttributeStageInfo:
    rows: tuple[int, ...]
    support_rows: int
    level: int


class AttributeFusion(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        context_channels: int,
        *,
        kernel_size: int,
        block_layers: int,
    ) -> None:
        super().__init__()
        self.in_proj = SubmConv3d(
            latent_channels + context_channels,
            context_channels,
            kernel_size=1,
        )
        self.blocks = SparseSequential(
            *(
                ResidualBlock(context_channels, kernel_size=kernel_size)
                for _ in range(block_layers)
            )
        )
        self.out_proj = SubmConv3d(
            context_channels,
            context_channels,
            kernel_size=1,
        )
        self.relu = ReLU()

    def __call__(
        self,
        decoded: SparseTensor,
        context: SparseTensor,
    ) -> SparseTensor:
        if not decoded.same_coords(context):
            decoded = context.replace(feats=decoded.feats)
        x = context.replace(
            feats=mx.concatenate([decoded.feats, context.feats], axis=1)
        )
        x = self.relu(self.in_proj(x))
        x = self.relu(self.blocks(x))
        return self.out_proj(x)


class AttributeEntropyContext(nn.Module):
    def __init__(
        self,
        context_channels: int,
        latent_channels: int,
        *,
        kernel_size: int,
        block_layers: int,
    ) -> None:
        super().__init__()
        self.context = SparseSequential(
            SubmConv3d(
                context_channels,
                context_channels,
                kernel_size=kernel_size,
            ),
            ReLU(),
            *(
                ResidualBlock(context_channels, kernel_size=kernel_size)
                for _ in range(block_layers)
            ),
        )
        self.mu = SubmConv3d(
            context_channels, latent_channels, kernel_size=1
        )
        self.log_sigma = SubmConv3d(
            context_channels, latent_channels, kernel_size=1
        )

    def __call__(self, context: SparseTensor) -> tuple[mx.array, mx.array]:
        x = self.context(context)
        return self.mu(x).feats, self.log_sigma(x).feats


class AttributeGaussianGenerator(nn.Module):
    def __init__(
        self,
        context_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        block_layers: int,
    ) -> None:
        super().__init__()
        self.net = SparseSequential(
            SubmConv3d(
                context_channels,
                context_channels,
                kernel_size=kernel_size,
            ),
            ReLU(),
            *(
                ResidualBlock(context_channels, kernel_size=kernel_size)
                for _ in range(block_layers)
            ),
            ReLU(),
            SubmConv3d(
                context_channels,
                context_channels,
                kernel_size=kernel_size,
            ),
            ReLU(),
            *(
                ResidualBlock(context_channels, kernel_size=kernel_size)
                for _ in range(block_layers)
            ),
            ReLU(),
            SubmConv3d(context_channels, out_channels, kernel_size=1),
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return self.net(x)


class GameleonAttributeDecoder(nn.Module):
    """MLX decode-side reproduction of Gameleon's attribute network path."""

    def __init__(
        self,
        *,
        context_channels: int = 64,
        latent_channels: int = 8,
        kernel_size: int = 3,
        block_layers: int = 2,
        level: int = 9,
        use_opacity: bool = False,
    ) -> None:
        super().__init__()
        self.context_channels = int(context_channels)
        self.latent_channels = int(latent_channels)
        self.level = int(level)
        self.use_opacity = bool(use_opacity)
        out_channels = 14 if self.use_opacity else 13
        self.entropy_contexts = [
            AttributeEntropyContext(
                self.context_channels,
                self.latent_channels,
                kernel_size=kernel_size,
                block_layers=block_layers,
            )
            for _ in range(self.stage_count)
        ]
        self.fusions = [
            AttributeFusion(
                self.latent_channels,
                self.context_channels,
                kernel_size=kernel_size,
                block_layers=block_layers,
            )
            for _ in range(self.stage_count)
        ]
        self.generator = AttributeGaussianGenerator(
            self.context_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            block_layers=block_layers,
        )

    @property
    def stage_count(self) -> int:
        return max(1, self.level - 6)

    def latent_shapes(self, coords: mx.array) -> list[tuple[int, int]]:
        stage_coords = _attribute_stage_coords(
            coords.astype(mx.int32),
            level=self.level,
            stage_count=self.stage_count,
        )
        return [
            (int(stage.shape[0]), self.latent_channels)
            for stage in stage_coords
        ]

    def decode(
        self,
        coords: mx.array,
        latent_values: list[np.ndarray] | list[mx.array],
    ) -> AttributeDecodeResult:
        stage_coords = _attribute_stage_coords(
            coords.astype(mx.int32),
            level=self.level,
            stage_count=self.stage_count,
        )
        if len(latent_values) != len(stage_coords):
            raise ValueError('attribute latent stage count mismatch')

        coarse = stage_coords[0]
        lower = SparseTensor(
            coarse,
            mx.zeros(
                (coarse.shape[0], self.context_channels),
                dtype=mx.float32,
            ),
        )

        for stage, (fine_coords, latent) in enumerate(
            zip(stage_coords, latent_values, strict=True)
        ):
            if stage == 0:
                context_feats = lower.feats
            else:
                context_feats = _project_parent_context(lower, fine_coords)
            latent_feats = _latent_array(latent, fine_coords.shape[0])
            fine_coords, combined = sort_coords_feats(
                fine_coords,
                mx.concatenate([context_feats, latent_feats], axis=1),
            )
            context_feats = combined[:, : self.context_channels]
            latent_feats = combined[:, self.context_channels :]
            context = SparseTensor(fine_coords, context_feats)
            mu, log_sigma = self.entropy_contexts[stage](context)
            decoded = latent_feats * mx.rsqrt(mx.exp(log_sigma) + 1.0)
            decoded = decoded + mx.tanh(mu) * 0.01
            lower = self.fusions[stage](
                context.replace(feats=decoded), context
            )

        generated = self.generator(lower)
        feats = generated.feats
        rotation = feats[:, 0:4] + mx.array(
            [1.0, 0.0, 0.0, 0.0], dtype=feats.dtype
        )
        scale = mx.maximum(feats[:, 4:7] + 1.0, 0.0)
        if self.use_opacity:
            opacity = mx.sigmoid(feats[:, 7:8])
            offset = feats[:, 8:11]
            dc = feats[:, 11:14]
        else:
            opacity = mx.ones((feats.shape[0], 1), dtype=feats.dtype)
            offset = feats[:, 7:10]
            dc = feats[:, 10:13]
        primitives = generated.coords[:, 1:].astype(mx.float32) + offset
        mx.eval(primitives, dc, rotation, scale, opacity, feats)
        return AttributeDecodeResult(
            primitives=primitives,
            dc=dc,
            rotation=rotation,
            scale=scale,
            opacity=opacity,
            features=feats,
        )

    def stage_info(self, coords: mx.array) -> AttributeStageInfo:
        stage_coords = _attribute_stage_coords(
            coords.astype(mx.int32),
            level=self.level,
            stage_count=self.stage_count,
        )
        return AttributeStageInfo(
            rows=tuple(int(stage.shape[0]) for stage in stage_coords),
            support_rows=int(coords.shape[0]),
            level=self.level,
        )


def make_dummy_attribute_latents(
    shapes: list[tuple[int, int]],
    *,
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(-8, 9, size=shape, dtype=np.int8) for shape in shapes
    ]


def _latent_array(value: np.ndarray | mx.array, rows: int) -> mx.array:
    latent = value if isinstance(value, mx.array) else mx.array(value)
    if latent.shape[0] != rows:
        raise ValueError('attribute latent row count mismatch')
    return latent.astype(mx.float32)


def _attribute_stage_coords(
    coords: mx.array,
    *,
    level: int,
    stage_count: int,
) -> list[mx.array]:
    if level < 7 or level > 10:
        raise ValueError('attribute level must be in [7, 10]')
    if stage_count < 1 or stage_count > 4:
        raise ValueError('attribute stage count must be in [1, 4]')

    geom_factor = 1 << (10 - level)
    active = mx.array([coords.shape[0]], dtype=mx.int32)
    current = coords.astype(mx.int32)
    if geom_factor != 1:
        current = mx.concatenate(
            [current[:, :1], current[:, 1:] * geom_factor],
            axis=1,
        ).astype(mx.int32)

    feature_levels = [current]
    for _ in range(3):
        down = occupancy_downsample(feature_levels[-1], active)
        mx.eval(down.coords, down.active_rows)
        count = int(down.active_rows.tolist()[0])
        feature_levels.append(down.coords[:count])
        active = down.active_rows

    return feature_levels[-stage_count:][::-1]


def _project_parent_context(
    parent: SparseTensor, child_coords: mx.array
) -> mx.array:
    child_parent = mx.concatenate(
        [child_coords[:, :1], child_coords[:, 1:] // 2],
        axis=1,
    ).astype(mx.int32)
    parent_rows = lookup_coords(parent.coords, child_parent)
    parent_rows = mx.maximum(parent_rows, 0)
    return mx.take(parent.feats, parent_rows, axis=0)
