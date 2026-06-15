from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lattice import SparseTensor
from mlx_lattice.nn import ReLU, SubmConv3d


class SparseSequential(nn.Module):
    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x: SparseTensor) -> SparseTensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv0 = SubmConv3d(channels, channels, kernel_size=kernel_size)
        self.conv1 = SubmConv3d(channels, channels, kernel_size=kernel_size)
        self.relu = ReLU()

    def __call__(self, x: SparseTensor) -> SparseTensor:
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        return self.relu(x.replace(feats=out.feats + x.feats))


class SparseFeatureMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        final_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.final_softmax = final_softmax

    def __call__(self, feats: mx.array) -> mx.array:
        feats = nn.relu(self.fc0(feats))
        feats = nn.relu(self.fc1(feats))
        feats = self.fc2(feats)
        if self.final_softmax:
            feats = mx.softmax(feats, axis=-1)
        return feats


class TargetEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, channels)

    def __call__(self, feats: mx.array, coords: mx.array) -> mx.array:
        delta = coords[:, 1:] % 2
        indices = (delta[:, 0] + delta[:, 1] * 2 + delta[:, 2] * 4).astype(
            mx.int32
        )
        return feats + self.embedding(indices)
