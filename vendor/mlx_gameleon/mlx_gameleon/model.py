from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import mlx.core as mx
import mlx.nn as nn
from mlx_lattice import SparseTensor
from mlx_lattice.nn import ReLU, SubmConv3d

from mlx_gameleon.layers import (
    ResidualBlock,
    SparseFeatureMLP,
    SparseSequential,
    TargetEmbedding,
)
from mlx_gameleon.ops import (
    checkerboard_groups,
    expand_occupancy_level,
    occupancy_pyramid,
    sort_coords_feats,
)


@dataclass(frozen=True, slots=True)
class StagePrediction:
    coords: mx.array
    occupancy: mx.array
    group1_s0: mx.array
    group1_s1: mx.array
    group2_s0: mx.array
    group2_s1: mx.array


class GameleonGeometryModel(nn.Module):
    """MLX reproduction of Gameleon's lossless TorchSparse geometry network."""

    def __init__(
        self,
        *,
        channels: int = 32,
        kernel_size: int = 3,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.prior_embedding = nn.Embedding(256, self.channels)
        self.prior_resnet = SparseSequential(
            SubmConv3d(
                self.channels, self.channels, kernel_size=kernel_size
            ),
            ReLU(),
            ResidualBlock(self.channels, kernel_size=kernel_size),
            ResidualBlock(self.channels, kernel_size=kernel_size),
        )
        self.target_embedding = TargetEmbedding(self.channels)
        self.target_resnet = SparseSequential(
            SubmConv3d(
                self.channels, self.channels, kernel_size=kernel_size
            ),
            ReLU(),
            ResidualBlock(self.channels, kernel_size=kernel_size),
            ResidualBlock(self.channels, kernel_size=kernel_size),
        )

        self.group1_spatial_conv_s0 = _stage_conv(
            self.channels, kernel_size
        )
        self.group1_spatial_conv_s1 = _stage_conv(
            self.channels, kernel_size
        )
        self.group2_spatial_conv_s0 = _stage_conv(
            self.channels, kernel_size
        )
        self.group2_spatial_conv_s1 = _stage_conv(
            self.channels, kernel_size
        )
        self.neighbor_conv = SubmConv3d(
            self.channels, self.channels, kernel_size=kernel_size
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.ReLU(),
        )

        self.group1_pred_head_s0 = SparseFeatureMLP(
            self.channels, hidden_channels, 16, final_softmax=True
        )
        self.group1_pred_head_s1_emb = nn.Embedding(16, self.channels)
        self.group1_pred_head_s1 = SparseFeatureMLP(
            self.channels, hidden_channels, 16, final_softmax=True
        )
        self.group2_pred_head_s0 = SparseFeatureMLP(
            self.channels, hidden_channels, 16, final_softmax=True
        )
        self.group2_pred_head_s1_emb = nn.Embedding(16, self.channels)
        self.group2_pred_head_s1 = SparseFeatureMLP(
            self.channels, hidden_channels, 16, final_softmax=True
        )

    def backbone(
        self,
        coords: mx.array,
        occupancy: mx.array,
    ) -> tuple[mx.array, mx.array]:
        coords, occupancy = sort_coords_feats(
            coords.astype(mx.int32), occupancy.reshape((-1, 1))
        )
        occupancy = occupancy[:, 0].astype(mx.int32)
        features = self.prior_embedding(occupancy).reshape(
            (-1, self.channels)
        )
        x = SparseTensor(coords, features)
        x = self.prior_resnet(x)

        up_coords, up_features = expand_occupancy_level(
            coords, occupancy, x.feats
        )
        if up_features is None:
            return up_coords, mx.zeros(
                (0, self.channels), dtype=x.feats.dtype
            )
        up_coords, up_features = sort_coords_feats(up_coords, up_features)
        up_features = self.target_embedding(up_features, up_coords)
        up = SparseTensor(up_coords, up_features)
        up = self.target_resnet(up)
        return up.coords, up.feats

    def aggregate_group2_features(
        self,
        coords: mx.array,
        features: mx.array,
        group1_mask: mx.array,
        group2_mask: mx.array,
        occupancy: mx.array,
    ) -> mx.array:
        masked = mx.where(
            group2_mask[:, None], mx.zeros_like(features), features
        )
        group1_s0 = (occupancy % 16).astype(mx.int32)
        group1_embedding = self.group1_pred_head_s1_emb(group1_s0)
        masked = mx.where(
            group1_mask[:, None],
            masked + group1_embedding,
            masked,
        )
        aggregated = self.neighbor_conv(SparseTensor(coords, masked)).feats
        combined = mx.concatenate([features, aggregated], axis=1)
        fused = self.feature_fusion(combined)
        return mx.take(fused, _mask_rows(group2_mask), axis=0)

    def predict_stage(
        self,
        coords: mx.array,
        occupancy: mx.array,
    ) -> StagePrediction:
        up_coords, up_features = self.backbone(coords, occupancy)
        group1_mask, group2_mask = checkerboard_groups(up_coords)
        out_occ = mx.zeros((up_coords.shape[0],), dtype=mx.int32)

        g1_s0_prob = mx.zeros((0, 16), dtype=up_features.dtype)
        g1_s1_prob = mx.zeros((0, 16), dtype=up_features.dtype)
        g2_s0_prob = mx.zeros((0, 16), dtype=up_features.dtype)
        g2_s1_prob = mx.zeros((0, 16), dtype=up_features.dtype)

        if int(mx.sum(group1_mask).tolist()) > 0:
            g1_rows = _mask_rows(group1_mask)
            g1_coords = mx.take(up_coords, g1_rows, axis=0)
            g1_features = mx.take(up_features, g1_rows, axis=0)
            g1_sparse = SparseTensor(g1_coords, g1_features)
            g1_s0_prob = self.group1_pred_head_s0(
                self.group1_spatial_conv_s0(g1_sparse).feats
            )
            g1_s0 = mx.argmax(g1_s0_prob, axis=1).astype(mx.int32)
            g1_s1_features = g1_features + self.group1_pred_head_s1_emb(
                g1_s0
            )
            g1_s1_prob = self.group1_pred_head_s1(
                self.group1_spatial_conv_s1(
                    SparseTensor(g1_coords, g1_s1_features)
                ).feats
            )
            g1_occ = (
                g1_s0 + mx.argmax(g1_s1_prob, axis=1).astype(mx.int32) * 16
            )
            out_occ = out_occ.at[g1_rows].add(g1_occ)

        if int(mx.sum(group2_mask).tolist()) > 0:
            g2_rows = _mask_rows(group2_mask)
            if int(mx.sum(group1_mask).tolist()) > 0:
                g2_features = self.aggregate_group2_features(
                    up_coords,
                    up_features,
                    group1_mask,
                    group2_mask,
                    out_occ,
                )
            else:
                g2_features = mx.take(up_features, g2_rows, axis=0)
            g2_coords = mx.take(up_coords, g2_rows, axis=0)
            g2_sparse = SparseTensor(g2_coords, g2_features)
            g2_s0_prob = self.group2_pred_head_s0(
                self.group2_spatial_conv_s0(g2_sparse).feats
            )
            g2_s0 = mx.argmax(g2_s0_prob, axis=1).astype(mx.int32)
            g2_s1_features = g2_features + self.group2_pred_head_s1_emb(
                g2_s0
            )
            g2_s1_prob = self.group2_pred_head_s1(
                self.group2_spatial_conv_s1(
                    SparseTensor(g2_coords, g2_s1_features)
                ).feats
            )
            g2_occ = (
                g2_s0 + mx.argmax(g2_s1_prob, axis=1).astype(mx.int32) * 16
            )
            out_occ = out_occ.at[g2_rows].add(g2_occ)

        mx.eval(out_occ, g1_s0_prob, g1_s1_prob, g2_s0_prob, g2_s1_prob)
        return StagePrediction(
            coords=up_coords,
            occupancy=out_occ,
            group1_s0=g1_s0_prob,
            group1_s1=g1_s1_prob,
            group2_s0=g2_s0_prob,
            group2_s1=g2_s1_prob,
        )

    def encode_forward(self, coords: mx.array) -> list[StagePrediction]:
        levels = occupancy_pyramid(coords)
        predictions = []
        for current, target in pairwise(levels):
            prediction = self.predict_stage(
                current.active_coords(), current.active_occupancy()
            )
            target_coords, target_occupancy = sort_coords_feats(
                target.active_coords(),
                target.active_occupancy().reshape((-1, 1)),
            )
            predictions.append(
                StagePrediction(
                    coords=target_coords,
                    occupancy=target_occupancy[:, 0].astype(mx.int32),
                    group1_s0=prediction.group1_s0,
                    group1_s1=prediction.group1_s1,
                    group2_s0=prediction.group2_s0,
                    group2_s1=prediction.group2_s1,
                )
            )
        return predictions

    def bits_per_point(self, coords: mx.array) -> mx.array:
        predictions = self.encode_forward(coords)
        total = mx.array(0.0, dtype=mx.float32)
        for stage in predictions:
            group1_mask, group2_mask = checkerboard_groups(stage.coords)
            group1_occ = mx.take(
                stage.occupancy, _mask_rows(group1_mask), axis=0
            )
            group2_occ = mx.take(
                stage.occupancy, _mask_rows(group2_mask), axis=0
            )
            total = total + _group_bits(stage.group1_s0, group1_occ % 16)
            total = total + _group_bits(stage.group1_s1, group1_occ // 16)
            total = total + _group_bits(stage.group2_s0, group2_occ % 16)
            total = total + _group_bits(stage.group2_s1, group2_occ // 16)
        return total / mx.maximum(
            mx.array(coords.shape[0], dtype=mx.float32), 1.0
        )


def _stage_conv(channels: int, kernel_size: int) -> SparseSequential:
    return SparseSequential(
        SubmConv3d(channels, channels, kernel_size=kernel_size),
        ReLU(),
        SubmConv3d(channels, channels, kernel_size=kernel_size),
    )


def _group_bits(prob: mx.array, symbols: mx.array) -> mx.array:
    if prob.shape[0] == 0:
        return mx.array(0.0, dtype=mx.float32)
    selected = mx.take_along_axis(
        prob,
        symbols.astype(mx.int32).reshape((-1, 1)),
        axis=1,
    )[:, 0]
    return mx.sum(mx.minimum(-mx.log2(selected + 1e-10), 50.0))


def _mask_rows(mask: mx.array) -> mx.array:
    count = int(mx.sum(mask).tolist())
    if count == 0:
        return mx.array([], dtype=mx.int32)
    order = mx.argsort(mask.astype(mx.int32)).astype(mx.int32)
    return order[-count:]
