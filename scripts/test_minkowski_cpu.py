#!/usr/bin/env python3
"""Smoke test: MinkowskiEngine CPU SparseTensor + convolution on macOS."""

from __future__ import annotations

import os
import sys

# PyTorch + ME both link libomp on macOS; allow duplicate runtime for smoke tests.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import MinkowskiEngine as ME


def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"minkowski={ME.__version__}")

    device = "cpu"
    coords = torch.randint(0, 4, (100, 3), dtype=torch.int32)
    feats = torch.randn(100, 16)
    batch = torch.zeros(100, 1, dtype=torch.int32)
    coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(
        features=feats_batch,
        coordinates=coords_batch,
        tensor_stride=1,
        device=device,
    )

    conv_cls = getattr(ME, "MinkowskiNormalizedConvolution", ME.MinkowskiConvolution)
    conv = conv_cls(16, 16, kernel_size=3, dimension=3).to(device)
    y = conv(x)
    print("CPU sparse collate ok:", tuple(x.F.shape))
    print("CPU conv ok:", tuple(y.F.shape))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
