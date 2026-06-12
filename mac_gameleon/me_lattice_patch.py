"""MinkowskiEngine acceleration via mlx-lattice (inference-only)."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import torch

_ME_LATTICE_INSTALLED = False
_ORIG_CONV_FORWARD: Optional[Callable[..., Any]] = None
_ORIG_POOL_FORWARD: Optional[Callable[..., Any]] = None
_ORIG_NLN_FORWARD: Optional[Callable[..., Any]] = None

_LATTICE_STATS = {
    "conv": 0,
    "generative_transpose": 0,
    "transpose": 0,
    "pool": 0,
    "pool_transpose": 0,
    "relu": 0,
    "sigmoid": 0,
    "fallback": 0,
}

_SUM_UNPOOL_WEIGHT_CACHE: dict[tuple[int, tuple[int, int, int]], mx.array] = {}
_COUNT_UNPOOL_WEIGHT_CACHE: dict[tuple[int, int, int], mx.array] = {}


def lattice_stats() -> dict[str, int]:
    return dict(_LATTICE_STATS)


def _lattice_enabled() -> bool:
    value = os.environ.get("GAMELEON_ME_LATTICE", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _lattice_strict() -> bool:
    value = os.environ.get("GAMELEON_ME_LATTICE_STRICT", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _triple(value: Union[int, Tuple[int, ...], list[int]]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            item = int(value[0])
            return (item, item, item)
        if len(value) == 3:
            return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"Cannot create 3-tuple from {value!r}")


def _to_mx_coords(coords: torch.Tensor) -> mx.array:
    return mx.array(coords.detach().cpu().numpy().astype(np.int32, copy=False))


def _to_mx_feats(feats: torch.Tensor) -> mx.array:
    return mx.array(feats.detach().cpu().numpy().astype(np.float32, copy=False))


def _to_mx_weight(weight: torch.Tensor) -> mx.array:
    return mx.array(weight.detach().cpu().numpy().astype(np.float32, copy=False))


def _to_mx_bias(bias: Optional[torch.Tensor]) -> Optional[mx.array]:
    if bias is None:
        return None
    flat = bias.detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False)
    return mx.array(flat)


def _can_accelerate_sparse(sparse) -> bool:
    if not _lattice_enabled():
        return False
    if torch.is_grad_enabled():
        return False
    if sparse.D != 3:
        return False
    if sparse.device.type != "cpu":
        return False
    if sparse.F.dtype != torch.float32:
        return False
    return True


def _kernel_generator_ok(kernel_generator) -> bool:
    from MinkowskiEngineBackend._C import RegionType

    dilation = _triple(kernel_generator.kernel_dilation)
    if any(value != 1 for value in dilation):
        return False
    if kernel_generator.region_type != RegionType.HYPER_CUBE:
        return False
    if kernel_generator.region_offsets is not None:
        return False
    return True


def _can_accelerate_module(module, sparse) -> bool:
    if not _can_accelerate_sparse(sparse):
        return False
    if module.dimension != 3:
        return False
    if not _kernel_generator_ok(module.kernel_generator):
        return False
    return True


def _me_target_coords(sparse, coordinates):
    import MinkowskiEngine as ME

    if coordinates is None:
        return None, None
    if isinstance(coordinates, ME.SparseTensor):
        return coordinates.C, coordinates.coordinate_map_key
    if isinstance(coordinates, ME.CoordinateMapKey):
        return sparse._manager.get_coordinates(coordinates), coordinates
    if isinstance(coordinates, torch.Tensor):
        return coordinates, None
    raise TypeError(f"unsupported coordinates type: {type(coordinates)!r}")


def _restrict_lattice_output(lattice_out, sparse, coordinates):
    from mlx_lattice import SparseTensor as LatticeSparseTensor
    from mlx_lattice.point import lookup_coords as lattice_lookup_coords

    target_coords, _ = _me_target_coords(sparse, coordinates)
    if target_coords is None:
        return lattice_out

    target_mx = _to_mx_coords(target_coords)
    rows = lattice_lookup_coords(lattice_out.coords, target_mx)
    mx.eval(rows)
    rows_np = np.asarray(rows, dtype=np.int64)
    if (rows_np < 0).any():
        raise ValueError("lattice output missing requested coordinates")

    feats = mx.take(lattice_out.feats, rows, axis=0)
    mx.eval(feats)
    return LatticeSparseTensor(
        target_mx,
        feats,
        lattice_out.stride,
        coord_manager=lattice_out.coord_manager,
    )


def _me_sparse_from_lattice(
    lattice_out,
    template,
    manager,
    *,
    output_coordinate_map_key=None,
) -> Any:
    import MinkowskiEngine as ME

    mx.eval(lattice_out.coords, lattice_out.feats)
    feats = torch.from_numpy(np.asarray(lattice_out.feats).astype(np.float32, copy=False)).to(
        device=template.device,
        dtype=torch.float32,
    )
    if output_coordinate_map_key is not None:
        return ME.SparseTensor(
            features=feats,
            coordinate_map_key=output_coordinate_map_key,
            coordinate_manager=manager,
        )

    coords = torch.from_numpy(np.asarray(lattice_out.coords).astype(np.int32, copy=False)).to(
        device=template.device
    )
    stride = tuple(int(v) for v in lattice_out.stride)
    return ME.SparseTensor(
        features=feats,
        coordinates=coords,
        tensor_stride=stride,
        coordinate_manager=manager,
    )


def _lattice_input_from_me(sparse) -> Any:
    from mlx_lattice import SparseTensor as LatticeSparseTensor

    cache = getattr(sparse, "_mac_lattice_cache", None)
    feat_stamp = (sparse.F.data_ptr(), int(sparse.F.shape[0]), int(sparse.F.shape[1]))
    if cache is not None and cache.get("feat_stamp") == feat_stamp and cache.get("coord_stamp") == (
        sparse.C.data_ptr(),
        int(sparse.C.shape[0]),
    ):
        return cache["tensor"].replace(feats=_to_mx_feats(sparse.F))

    stride = _triple(sparse.tensor_stride)
    lattice = LatticeSparseTensor(_to_mx_coords(sparse.C), _to_mx_feats(sparse.F), stride=stride)
    sparse._mac_lattice_cache = {
        "feat_stamp": feat_stamp,
        "coord_stamp": (sparse.C.data_ptr(), int(sparse.C.shape[0])),
        "tensor": lattice,
    }
    return lattice


def _sum_unpool_weight(channels: int, kernel_size: Tuple[int, int, int]) -> mx.array:
    key = (channels, kernel_size)
    cached = _SUM_UNPOOL_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    volume = int(np.prod(kernel_size))
    flat = np.zeros((volume * channels, channels), dtype=np.float32)
    for kernel_idx in range(volume):
        for channel_idx in range(channels):
            flat[kernel_idx * channels + channel_idx, channel_idx] = 1.0
    weight = mx.array(flat)
    _SUM_UNPOOL_WEIGHT_CACHE[key] = weight
    return weight


def _count_unpool_weight(kernel_size: Tuple[int, int, int]) -> mx.array:
    cached = _COUNT_UNPOOL_WEIGHT_CACHE.get(kernel_size)
    if cached is not None:
        return cached

    volume = int(np.prod(kernel_size))
    weight = mx.ones((volume, 1), dtype=mx.float32)
    _COUNT_UNPOOL_WEIGHT_CACHE[kernel_size] = weight
    return weight


def _lattice_conv_forward(module, sparse, coordinates=None):
    from mlx_lattice.ops import conv3d as lattice_conv3d
    from mlx_lattice.ops import conv_transpose3d as lattice_conv_transpose3d
    from mlx_lattice.ops import generative_conv_transpose3d as lattice_generative_conv_transpose3d

    kg = module.kernel_generator
    kernel_size = _triple(kg.kernel_size)
    stride = _triple(kg.kernel_stride)
    dilation = _triple(kg.kernel_dilation)
    x_ml = _lattice_input_from_me(sparse)
    weight_mx = _to_mx_weight(module.kernel)
    bias_mx = _to_mx_bias(module.bias)
    _, output_key = _me_target_coords(sparse, coordinates)

    if module.is_transpose and kg.expand_coordinates:
        y_ml = lattice_generative_conv_transpose3d(
            x_ml,
            weight_mx,
            bias_mx,
            kernel_size=kernel_size,
            stride=stride,
            weight_layout="flat",
        )
        _LATTICE_STATS["generative_transpose"] += 1
    elif module.is_transpose:
        y_ml = lattice_conv_transpose3d(
            x_ml,
            weight_mx,
            bias_mx,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            weight_layout="flat",
        )
        _LATTICE_STATS["transpose"] += 1
    else:
        y_ml = lattice_conv3d(
            x_ml,
            weight_mx,
            bias_mx,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            weight_layout="flat",
        )
        _LATTICE_STATS["conv"] += 1

    if coordinates is not None:
        y_ml = _restrict_lattice_output(y_ml, sparse, coordinates)

    return _me_sparse_from_lattice(
        y_ml,
        sparse,
        sparse._manager,
        output_coordinate_map_key=output_key,
    )


def _patched_conv_forward(module, sparse, coordinates=None):
    assert _ORIG_CONV_FORWARD is not None
    if module.use_mm or not _can_accelerate_module(module, sparse):
        _LATTICE_STATS["fallback"] += 1
        return _ORIG_CONV_FORWARD(module, sparse, coordinates)
    try:
        return _lattice_conv_forward(module, sparse, coordinates)
    except Exception:
        if _lattice_strict():
            raise
        _LATTICE_STATS["fallback"] += 1
        return _ORIG_CONV_FORWARD(module, sparse, coordinates)


def _lattice_pool_forward(module, sparse, coordinates=None):
    from MinkowskiEngineBackend._C import PoolingMode
    from mlx_lattice.ops import pool3d as lattice_pool3d

    kg = module.kernel_generator
    kernel_size = _triple(kg.kernel_size)
    stride = _triple(kg.kernel_stride)
    x_ml = _lattice_input_from_me(sparse)
    _, output_key = _me_target_coords(sparse, coordinates)
    if module.pooling_mode == PoolingMode.LOCAL_AVG_POOLING:
        mode = "avg"
    elif module.pooling_mode == PoolingMode.LOCAL_MAX_POOLING:
        mode = "max"
    else:
        raise ValueError(f"unsupported pooling mode: {module.pooling_mode}")
    y_ml = lattice_pool3d(x_ml, kernel_size=kernel_size, stride=stride, mode=mode)
    if coordinates is not None:
        y_ml = _restrict_lattice_output(y_ml, sparse, coordinates)
    _LATTICE_STATS["pool"] += 1
    return _me_sparse_from_lattice(
        y_ml,
        sparse,
        sparse._manager,
        output_coordinate_map_key=output_key,
    )


def _lattice_pool_transpose_forward(module, sparse, coordinates=None):
    from MinkowskiEngineBackend._C import PoolingMode
    from mlx_lattice.ops import generative_conv_transpose3d as lattice_generative_conv_transpose3d

    if module.pooling_mode != PoolingMode.LOCAL_AVG_POOLING:
        raise ValueError(f"unsupported transpose pooling mode: {module.pooling_mode}")

    kg = module.kernel_generator
    kernel_size = _triple(kg.kernel_size)
    stride = _triple(kg.kernel_stride)
    x_ml = _lattice_input_from_me(sparse)
    _, output_key = _me_target_coords(sparse, coordinates)
    channels = x_ml.channels

    sum_weight = _sum_unpool_weight(channels, kernel_size)
    summed = lattice_generative_conv_transpose3d(
        x_ml,
        sum_weight,
        None,
        kernel_size=kernel_size,
        stride=stride,
        weight_layout="flat",
    )
    ones = mx.ones((x_ml.n_points, 1), dtype=mx.float32)
    counts = lattice_generative_conv_transpose3d(
        x_ml.replace(feats=ones),
        _count_unpool_weight(kernel_size),
        None,
        kernel_size=kernel_size,
        stride=stride,
        weight_layout="flat",
    )
    avg_feats = summed.feats / mx.maximum(counts.feats, 1.0)
    y_ml = summed.replace(feats=avg_feats)
    if coordinates is not None:
        y_ml = _restrict_lattice_output(y_ml, sparse, coordinates)
    _LATTICE_STATS["pool_transpose"] += 1
    return _me_sparse_from_lattice(
        y_ml,
        sparse,
        sparse._manager,
        output_coordinate_map_key=output_key,
    )


def _patched_pool_forward(module, sparse, coordinates=None):
    assert _ORIG_POOL_FORWARD is not None
    if not _can_accelerate_module(module, sparse):
        _LATTICE_STATS["fallback"] += 1
        return _ORIG_POOL_FORWARD(module, sparse, coordinates)
    try:
        if module.is_transpose:
            if not module.kernel_generator.expand_coordinates:
                raise ValueError("transpose pool requires expand_coordinates=True")
            return _lattice_pool_transpose_forward(module, sparse, coordinates)
        return _lattice_pool_forward(module, sparse, coordinates)
    except Exception:
        if _lattice_strict():
            raise
        _LATTICE_STATS["fallback"] += 1
        return _ORIG_POOL_FORWARD(module, sparse, coordinates)


def _lattice_nonlinearity_forward(module, sparse):
    from MinkowskiEngine import MinkowskiReLU, MinkowskiSigmoid
    from mlx_lattice.ops import relu as lattice_relu
    from mlx_lattice.ops import sigmoid as lattice_sigmoid

    x_ml = _lattice_input_from_me(sparse)
    if isinstance(module, MinkowskiReLU):
        y_ml = lattice_relu(x_ml)
        _LATTICE_STATS["relu"] += 1
    elif isinstance(module, MinkowskiSigmoid):
        y_ml = lattice_sigmoid(x_ml)
        _LATTICE_STATS["sigmoid"] += 1
    else:
        raise ValueError(f"unsupported nonlinearity: {module.__class__.__name__}")

    mx.eval(y_ml.feats)
    out_feats = torch.from_numpy(np.asarray(y_ml.feats).astype(np.float32, copy=False)).to(
        device=sparse.device,
        dtype=torch.float32,
    )
    import MinkowskiEngine as ME

    return ME.SparseTensor(
        features=out_feats,
        coordinate_map_key=sparse.coordinate_map_key,
        coordinate_manager=sparse._manager,
    )


def _patched_nonlinearity_forward(module, sparse):
    assert _ORIG_NLN_FORWARD is not None
    from MinkowskiEngine import MinkowskiReLU, MinkowskiSigmoid, SparseTensor

    if not isinstance(sparse, SparseTensor) or not _can_accelerate_sparse(sparse):
        _LATTICE_STATS["fallback"] += 1
        return _ORIG_NLN_FORWARD(module, sparse)

    if not isinstance(module, (MinkowskiReLU, MinkowskiSigmoid)):
        _LATTICE_STATS["fallback"] += 1
        return _ORIG_NLN_FORWARD(module, sparse)

    try:
        return _lattice_nonlinearity_forward(module, sparse)
    except Exception:
        if _lattice_strict():
            raise
        _LATTICE_STATS["fallback"] += 1
        return _ORIG_NLN_FORWARD(module, sparse)


def install_me_normalized_aliases() -> None:
    """Map Normalized* ME layers to standard conv on builds without them (e.g. ME 0.5.4)."""
    import MinkowskiEngine as ME

    if not hasattr(ME, "MinkowskiNormalizedConvolution"):
        ME.MinkowskiNormalizedConvolution = ME.MinkowskiConvolution
    if not hasattr(ME, "MinkowskiNormalizedConvolutionTranspose"):
        ME.MinkowskiNormalizedConvolutionTranspose = ME.MinkowskiConvolutionTranspose
    if not hasattr(ME, "MinkowskiGenerativeNormalizedConvolutionTranspose"):
        ME.MinkowskiGenerativeNormalizedConvolutionTranspose = (
            ME.MinkowskiGenerativeConvolutionTranspose
        )


def install_me_lattice_acceleration(*, force: bool = False) -> None:
    """Monkey-patch ME conv/pool/nonlinearity forwards through mlx-lattice."""
    global _ME_LATTICE_INSTALLED, _ORIG_CONV_FORWARD, _ORIG_POOL_FORWARD, _ORIG_NLN_FORWARD

    if _ME_LATTICE_INSTALLED and not force:
        return

    try:
        import mlx_lattice  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mlx-lattice is required for ME lattice acceleration. "
            "Install with: pip install 'mlx-lattice>=0.1.11'"
        ) from exc

    import MinkowskiEngine as ME

    install_me_normalized_aliases()

    conv_cls = ME.MinkowskiConvolution
    conv_t_cls = ME.MinkowskiConvolutionTranspose
    gen_conv_t_cls = ME.MinkowskiGenerativeConvolutionTranspose
    from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolutionBase
    from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiNonlinearityBase
    from MinkowskiEngine.MinkowskiPooling import MinkowskiPoolingBase

    ME.MinkowskiConvolution = conv_cls
    ME.MinkowskiConvolutionTranspose = conv_t_cls
    ME.MinkowskiGenerativeConvolutionTranspose = gen_conv_t_cls
    install_me_normalized_aliases()

    if _ORIG_CONV_FORWARD is None:
        _ORIG_CONV_FORWARD = MinkowskiConvolutionBase.forward
    if _ORIG_POOL_FORWARD is None:
        _ORIG_POOL_FORWARD = MinkowskiPoolingBase.forward
    if _ORIG_NLN_FORWARD is None:
        _ORIG_NLN_FORWARD = MinkowskiNonlinearityBase.forward

    MinkowskiConvolutionBase.forward = _patched_conv_forward
    MinkowskiPoolingBase.forward = _patched_pool_forward
    MinkowskiNonlinearityBase.forward = _patched_nonlinearity_forward
    _ME_LATTICE_INSTALLED = True
