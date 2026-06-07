"""mlx-lattice sparse backend for Gameleon geometry-meta encode/decode."""

from __future__ import annotations

import math
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn

_GEOMETRY_BACKEND_INSTALLED = False
_SPARSE_IMPORT_NAME = "torchsparse"


def make_ntuple(value: Union[int, Tuple[int, ...], List[int]], ndim: int = 3) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, (tuple, list)):
        if len(value) == ndim:
            return tuple(int(v) for v in value)
        if len(value) == 1:
            return (int(value[0]),) * ndim
    raise ValueError(f"Cannot create {ndim}-tuple from {value!r}")


def _to_mx_coords(coords: torch.Tensor) -> mx.array:
    return mx.array(coords.detach().cpu().numpy().astype(np.int32, copy=False))


def _to_mx_feats(feats: torch.Tensor) -> mx.array:
    return mx.array(feats.detach().cpu().numpy().astype(np.float32, copy=False))


def _torch_from_mx(array, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).to(device=device, dtype=dtype)


class SparseTensor:
    def __init__(
        self,
        feats: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, ...]] = 1,
        *,
        features: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
    ) -> None:
        feats = feats if feats is not None else features
        coords = coords if coords is not None else coordinates
        if feats is None or coords is None:
            raise ValueError("SparseTensor requires feats and coords")
        self.feats = feats
        self.coords = coords
        self.stride = make_ntuple(stride, ndim=3)
        self.cmaps: Dict[Tuple[int, ...], torch.Tensor] = {}
        self.kmaps: Dict[Tuple[Any, ...], Any] = {}
        self._lattice = None

    @property
    def F(self) -> torch.Tensor:
        return self.feats

    @F.setter
    def F(self, feats: torch.Tensor) -> None:
        self.feats = feats
        self._lattice = None

    @property
    def C(self) -> torch.Tensor:
        return self.coords

    @C.setter
    def C(self, coords: torch.Tensor) -> None:
        self.coords = coords
        self._lattice = None

    @property
    def s(self) -> Tuple[int, ...]:
        return self.stride

    @s.setter
    def s(self, stride: Union[int, Tuple[int, ...]]) -> None:
        self.stride = make_ntuple(stride, ndim=3)
        self._lattice = None

    def _to_lattice(self):
        from mlx_lattice import SparseTensor as LatticeSparseTensor

        if self._lattice is not None:
            return self._lattice.replace(feats=_to_mx_feats(self.feats))
        lattice = LatticeSparseTensor(
            _to_mx_coords(self.coords),
            _to_mx_feats(self.feats),
            stride=self.stride,
        )
        self._lattice = lattice
        return lattice

    @classmethod
    def _from_lattice(cls, lattice_out, template: SparseTensor) -> SparseTensor:
        coords = _torch_from_mx(
            lattice_out.coords,
            device=template.coords.device,
            dtype=template.coords.dtype,
        )
        feats = _torch_from_mx(
            lattice_out.feats,
            device=template.feats.device,
            dtype=template.feats.dtype,
        )
        output = cls(coords=coords, feats=feats, stride=tuple(lattice_out.stride))
        output.cmaps = dict(template.cmaps)
        output.cmaps.setdefault(output.stride, output.coords)
        output.kmaps = dict(template.kmaps)
        output._lattice = lattice_out
        return output

    def cpu(self) -> SparseTensor:
        self.coords = self.coords.cpu()
        self.feats = self.feats.cpu()
        self._lattice = None
        return self

    def cuda(self) -> SparseTensor:
        self.coords = self.coords.cuda()
        self.feats = self.feats.cuda()
        self._lattice = None
        return self

    def half(self) -> SparseTensor:
        self.feats = self.feats.half()
        self._lattice = None
        return self

    def detach(self) -> SparseTensor:
        self.coords = self.coords.detach()
        self.feats = self.feats.detach()
        return self

    def to(self, device, non_blocking: bool = True) -> SparseTensor:
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        self._lattice = None
        return self

    def __add__(self, other: SparseTensor) -> SparseTensor:
        output = SparseTensor(coords=self.coords, feats=self.feats + other.feats, stride=self.stride)
        output.cmaps = self.cmaps
        output.kmaps = self.kmaps
        return output


class PointTensor:
    def __init__(self, feats, coords, idx_query=None, weights=None):
        self.F = feats
        self.C = coords
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_features = {"idx_query": {}, "counts": {}}

    def to(self, device, non_blocking: bool = True):
        self.F = self.F.to(device, non_blocking=non_blocking)
        self.C = self.C.to(device, non_blocking=non_blocking)
        return self


def _prepare_weight(weight: torch.Tensor, kernel_size: Tuple[int, int, int]) -> mx.array:
    w = weight.detach().cpu().numpy().astype(np.float32, copy=False)
    if w.ndim == 2:
        w = w.reshape(1, w.shape[0], w.shape[1])
    return mx.array(w)


def conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    transposed: bool = False,
    epsilon: float = 0.0,
    mm_thresh: int = 0,
    kmap_mode: str = "hashmap",
) -> SparseTensor:
    from mlx_lattice.ops import conv3d as lattice_conv3d
    from mlx_lattice.ops import conv_transpose3d as lattice_conv_transpose3d

    del epsilon, mm_thresh, kmap_mode
    kernel_size_nt = make_ntuple(kernel_size, ndim=3)
    stride_nt = make_ntuple(stride, ndim=3)
    dilation_nt = make_ntuple(dilation, ndim=3)

    x_ml = input._to_lattice()
    weight_mx = _prepare_weight(weight, kernel_size_nt)
    bias_mx = None
    if bias is not None:
        bias_mx = mx.array(bias.detach().cpu().numpy().astype(np.float32, copy=False))

    if transposed:
        y_ml = lattice_conv_transpose3d(
            x_ml,
            weight_mx,
            bias_mx,
            kernel_size=kernel_size_nt,
            stride=stride_nt,
            padding=0,
            dilation=dilation_nt,
            weight_layout="flat",
        )
    else:
        y_ml = lattice_conv3d(
            x_ml,
            weight_mx,
            bias_mx,
            kernel_size=kernel_size_nt,
            stride=stride_nt,
            padding=0,
            dilation=dilation_nt,
            transposed=False,
            weight_layout="flat",
        )
    mx.eval(y_ml.coords, y_ml.feats)
    return SparseTensor._from_lattice(y_ml, input)


def spdownsample(
    coords: torch.Tensor,
    stride: Union[int, Tuple[int, ...]] = 2,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    tensor_stride: Union[int, Tuple[int, ...]] = 1,
) -> torch.Tensor:
    import mlx_lattice as ml

    stride_nt = make_ntuple(stride, ndim=3)
    kernel_size_nt = make_ntuple(kernel_size, ndim=3)
    tensor_stride_nt = make_ntuple(tensor_stride, ndim=3)
    if not all(stride_nt[k] in [1, kernel_size_nt[k]] for k in range(3)):
        raise NotImplementedError

    effective = tuple(stride_nt[k] * tensor_stride_nt[k] for k in range(3))
    coords_mx = _to_mx_coords(coords)
    if len(set(effective)) == 1:
        out_mx = ml.spdownsample(coords_mx, stride=effective[0])
    else:
        out_mx = ml.spdownsample(coords_mx, stride=effective)
    mx.eval(out_mx)
    return _torch_from_mx(out_mx, device=coords.device, dtype=coords.dtype)


def build_kernel_map(
    _coords: torch.Tensor,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    stride: Union[int, Tuple[int, ...]] = 2,
    tensor_stride: Union[int, Tuple[int, ...]] = 1,
    mode: str = "hashmap",
):
    import mlx_lattice as ml

    if mode == "grid":
        raise NotImplementedError("grid kmap mode is not supported by mlx-lattice")

    kernel_size_nt = make_ntuple(kernel_size, ndim=3)
    stride_nt = make_ntuple(stride, ndim=3)
    tensor_stride_nt = make_ntuple(tensor_stride, ndim=3)
    effective_stride = tuple(stride_nt[k] * tensor_stride_nt[k] for k in range(3))

    mapping = ml.build_kernel_map(
        _to_mx_coords(_coords),
        kernel_size=kernel_size_nt,
        stride=effective_stride,
        padding=0,
        dilation=1,
    )
    mx.eval(mapping.maps, mapping.sizes, mapping.out_coords)

    nbmaps = _torch_from_mx(mapping.maps, device=_coords.device, dtype=torch.int64)
    nbsizes = _torch_from_mx(mapping.sizes, device=_coords.device, dtype=torch.int64)

    if any(s > 1 for s in effective_stride):
        out_coords = _torch_from_mx(mapping.out_coords, device=_coords.device, dtype=_coords.dtype)
        return nbmaps, nbsizes, out_coords, None, None
    return nbmaps, nbsizes, None, None


class _ConvConfig:
    @staticmethod
    def get_default_conv_config():
        return types.SimpleNamespace(kmap_mode="hashmap")

    @staticmethod
    def set_global_conv_config(_config) -> None:
        return None


def fapply(input: SparseTensor, fn) -> SparseTensor:
    return SparseTensor(coords=input.coords, feats=fn(input.feats), stride=input.stride)


class ReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
        bias: bool = False,
        transposed: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed
        self.config = config or {}
        self.kernel_volume = int(math.prod(self.kernel_size))
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(self.in_channels * max(self.kernel_volume, 1))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return conv3d(
            input,
            self.kernel,
            kernel_size=self.kernel_size,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            transposed=self.transposed,
            kmap_mode=self.config.get("kmap_mode", "hashmap"),
        )


def install_mlx_lattice_geometry_backend(*, force: bool = False) -> None:
    """Register mlx-lattice sparse ops before importing Gameleon geometry-meta modules."""
    global _GEOMETRY_BACKEND_INSTALLED
    if _GEOMETRY_BACKEND_INSTALLED and not force:
        return

    name = _SPARSE_IMPORT_NAME
    functional = types.ModuleType(f"{name}.nn.functional")
    functional.conv3d = conv3d
    functional.spdownsample = spdownsample
    functional.build_kernel_map = build_kernel_map
    functional.conv_config = _ConvConfig

    nn_modules = types.ModuleType(f"{name}.nn.modules")
    nn_modules.Conv3d = Conv3d
    nn_modules.ReLU = ReLU

    nn_pkg = types.ModuleType(f"{name}.nn")
    nn_pkg.Conv3d = Conv3d
    nn_pkg.ReLU = ReLU
    nn_pkg.functional = functional
    nn_pkg.modules = nn_modules

    utils_pkg = types.ModuleType(f"{name}.utils")
    utils_pkg.make_ntuple = make_ntuple

    backends_pkg = types.ModuleType(f"{name}.backends")
    backends_pkg.benchmark = False

    root = types.ModuleType(name)
    root.SparseTensor = SparseTensor
    root.PointTensor = PointTensor
    root.nn = nn_pkg
    root.utils = utils_pkg
    root.backends = backends_pkg
    root.backend = types.ModuleType(f"{name}.backend")

    sys.modules[name] = root
    sys.modules[f"{name}.nn"] = nn_pkg
    sys.modules[f"{name}.nn.functional"] = functional
    sys.modules[f"{name}.nn.modules"] = nn_modules
    sys.modules[f"{name}.utils"] = utils_pkg
    sys.modules[f"{name}.backends"] = backends_pkg
    sys.modules[f"{name}.backend"] = root.backend

    _GEOMETRY_BACKEND_INSTALLED = True


def geometry_backend_installed() -> bool:
    return _GEOMETRY_BACKEND_INSTALLED
