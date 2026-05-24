#!/usr/bin/env python3
"""Smoke test: TorchSparse CPU Conv3d on macOS."""

from __future__ import annotations

import sys

import torch
from torchsparse import SparseTensor
from torchsparse import nn as spnn


def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    import torchsparse

    print(f"torchsparse={torchsparse.__version__}")

    coords = torch.randint(0, 4, (100, 4), dtype=torch.int32)
    coords[:, 0] = 0
    x = SparseTensor(coords=coords, feats=torch.randn(100, 16))
    conv = spnn.Conv3d(16, 16, 3)
    y = conv(x)
    print("CPU conv ok:", tuple(y.feats.shape))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
