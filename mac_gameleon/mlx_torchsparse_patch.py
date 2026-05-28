"""Runtime patch helpers for TorchSparse MLX experiments."""

from __future__ import annotations

from typing import Tuple, Union

import mlx.core as mx
import numpy as np
import torch
from torchsparse.utils import make_ntuple

_ORIG_BUILD_KERNEL_MAP = None
_ORIG_FOG_FORWARD = None
_ORIG_CONV3D = None


def _spdownsample_mlx(
    coords: torch.Tensor,
    stride: Union[int, Tuple[int, ...]] = 2,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    tensor_stride: Union[int, Tuple[int, ...]] = 1,
) -> torch.Tensor:
    """Drop-in replacement for torchsparse.nn.functional.spdownsample.

    This keeps the original behavior for CPU path but performs the core
    quantization arithmetic via MLX on Apple GPU.
    """
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    tensor_stride = make_ntuple(tensor_stride, ndim=3)

    if not all(stride[k] in [1, kernel_size[k]] for k in range(3)):
        if coords.device.type == "cuda":
            # Keep original semantics for CUDA-only branch.
            import torchsparse.backend  # local import to avoid import cycles

            cuda_coords = coords[:, [3, 0, 1, 2]]
            out = torchsparse.backend.downsample_cuda(
                cuda_coords,
                cuda_coords.max(0).values,
                cuda_coords.min(0).values,
                kernel_size,
                stride,
                tensor_stride,
            )[:, [1, 2, 3, 0]]
            return out
        raise NotImplementedError

    sample_stride = np.asarray(
        [stride[k] * tensor_stride[k] for k in range(3)],
        dtype=np.int32,
    )[None, :]

    # MLX arithmetic on xyz, then return to torch for downstream unique/indexing.
    xyz = coords[:, :3].detach().cpu().numpy().astype(np.int32, copy=False)
    xyz_mx = mx.array(xyz)
    stride_mx = mx.array(sample_stride)
    q_xyz_mx = mx.floor(xyz_mx / stride_mx) * stride_mx
    mx.eval(q_xyz_mx)
    q_xyz = np.asarray(q_xyz_mx, dtype=np.int32)

    out = coords.clone()
    out[:, :3] = torch.from_numpy(q_xyz).to(device=coords.device, dtype=coords.dtype)
    out = out[:, [3, 0, 1, 2]]
    out = torch.unique(out, dim=0)
    out = out[:, [1, 2, 3, 0]]
    return out


def patch_torchsparse_spdownsample_with_mlx() -> None:
    """Monkey-patch TorchSparse F.spdownsample to MLX-backed implementation."""
    import torchsparse.nn.functional as F
    import torchsparse.nn.functional.downsample as downsample_mod

    F.spdownsample = _spdownsample_mlx
    downsample_mod.spdownsample = _spdownsample_mlx


def _kernel_offsets_mlx(
    kernel_size: Union[int, Tuple[int, ...]],
    tensor_stride: Union[int, Tuple[int, ...]],
    device: torch.device,
) -> torch.Tensor:
    """MLX-assisted kernel offset generation compatible with TorchSparse hashing."""
    kernel_size = make_ntuple(kernel_size, ndim=3)
    tensor_stride = make_ntuple(tensor_stride, ndim=3)

    # Build ranges on MLX so this stage can run on Apple GPU.
    axes = []
    for k in range(3):
        start = -kernel_size[k] // 2 + 1
        stop = kernel_size[k] // 2 + 1
        axis = mx.arange(start, stop, dtype=mx.int32) * int(tensor_stride[k])
        axes.append(axis)
    x, y, z = mx.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    offsets_mx = mx.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
    mx.eval(offsets_mx)
    offsets_np = np.asarray(offsets_mx, dtype=np.int32)

    # Weight layout compatibility with original torchsparse.nn.utils.get_kernel_offsets.
    if int(np.prod(kernel_size)) % 2 == 1:
        order = np.lexsort((offsets_np[:, 0], offsets_np[:, 1], offsets_np[:, 2]))
    else:
        order = np.lexsort((offsets_np[:, 2], offsets_np[:, 1], offsets_np[:, 0]))
    offsets_np = offsets_np[order]
    return torch.from_numpy(offsets_np).to(device=device, dtype=torch.int32)


def _build_kernel_map_mlx(
    _coords: torch.Tensor,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    stride: Union[int, Tuple[int, ...]] = 2,
    tensor_stride: Union[int, Tuple[int, ...]] = 1,
    mode: str = "hashmap",
):
    """MLX-hybrid drop-in replacement for TorchSparse build_kernel_map."""
    if mode == "grid":
        return _ORIG_BUILD_KERNEL_MAP(  # type: ignore[misc]
            _coords,
            kernel_size=kernel_size,
            stride=stride,
            tensor_stride=tensor_stride,
            mode=mode,
        )

    import torchsparse.backend
    from torchsparse.nn import functional as F

    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    tensor_stride = make_ntuple(tensor_stride, ndim=3)

    # Use MLX for offsets generation; keep hash/query backend behavior unchanged.
    offsets = _kernel_offsets_mlx(
        kernel_size=kernel_size,
        tensor_stride=tensor_stride,
        device=_coords.device,
    )
    references = F.sphash(_coords)
    if any(s > 1 for s in stride):
        coords = F.spdownsample(_coords, stride, kernel_size, tensor_stride)
    else:
        coords = _coords
    queries = F.sphash(coords, offsets)
    results = F.sphashquery(queries, references)
    nbsizes = torch.sum(results != -1, dim=1)
    nbmaps = torch.nonzero(results != -1)
    nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]
    nbmaps = nbmaps.contiguous()

    if hasattr(torchsparse.backend, "build_mask_from_kmap"):
        input_mask, output_mask = torchsparse.backend.build_mask_from_kmap(
            _coords.shape[0], coords.shape[0], nbmaps.int(), nbsizes.int()
        )
    else:
        input_mask, output_mask = None, None

    if any(s > 1 for s in stride):
        return nbmaps, nbsizes, coords, input_mask, output_mask
    return nbmaps, nbsizes, input_mask, output_mask


def patch_torchsparse_build_kmap_with_mlx() -> None:
    """Monkey-patch TorchSparse F.build_kernel_map to MLX-hybrid implementation."""
    global _ORIG_BUILD_KERNEL_MAP
    import torchsparse.nn.functional as F
    import torchsparse.nn.functional.build_kmap as build_kmap_mod

    if _ORIG_BUILD_KERNEL_MAP is None:
        _ORIG_BUILD_KERNEL_MAP = build_kmap_mod.build_kernel_map

    F.build_kernel_map = _build_kernel_map_mlx
    build_kmap_mod.build_kernel_map = _build_kernel_map_mlx


def _fog_forward_mlx(self, x):
    """MLX-hybrid replacement for Gameleon FOG.forward.

    It avoids torchsparse Conv3d(k=2,s=2) by grouping child voxels into parent
    voxels directly and summing occupancy codes.
    """
    from torchsparse import SparseTensor

    coords_t = x.coords
    # Keep existing occupancy definition from model code.
    occ_t = self.pos(coords_t).reshape(-1).detach().cpu().numpy().astype(np.float32)
    coords_np = coords_t.detach().cpu().numpy().astype(np.int32, copy=False)

    if coords_np.shape[1] != 4:
        # Unexpected layout; fail safe to original forward if available.
        if _ORIG_FOG_FORWARD is not None:
            return _ORIG_FOG_FORWARD(self, x)
        raise RuntimeError(f"Unexpected coords shape for FOG MLX patch: {coords_np.shape}")

    batch = coords_np[:, :1]
    xyz = coords_np[:, 1:]

    xyz_mx = mx.array(xyz)
    parent_xyz_mx = mx.floor(xyz_mx / 2).astype(mx.int32)
    mx.eval(parent_xyz_mx)
    parent_xyz = np.asarray(parent_xyz_mx, dtype=np.int32)
    parent_coords = np.concatenate([batch, parent_xyz], axis=1)

    unique_parents, inverse = np.unique(parent_coords, axis=0, return_inverse=True)
    # IMPORTANT: avoid dense (M x N) assignment matrix; it explodes on full clouds.
    # Aggregate occupancies with indexed accumulation (O(N) memory).
    summed_occ = np.zeros((unique_parents.shape[0], 1), dtype=np.float32)
    np.add.at(summed_occ[:, 0], inverse, occ_t)

    out_coords = torch.from_numpy(unique_parents).to(device=coords_t.device, dtype=coords_t.dtype)
    out_feats = torch.from_numpy(summed_occ).to(device=x.feats.device, dtype=x.feats.dtype)
    return SparseTensor(coords=out_coords, feats=out_feats, stride=(1, 1, 1))


def patch_gameleon_fog_with_mlx() -> None:
    """Monkey-patch Gameleon FOG.forward to MLX-hybrid implementation."""
    global _ORIG_FOG_FORWARD

    from lossless_torchsparse.src.kit import nn as kit_nn

    if _ORIG_FOG_FORWARD is None:
        _ORIG_FOG_FORWARD = kit_nn.FOG.forward
    kit_nn.FOG.forward = _fog_forward_mlx


def _conv3d_mlx(
    input,
    weight: torch.Tensor,
    kernel_size,
    bias=None,
    stride=1,
    dilation=1,
    transposed: bool = False,
    epsilon: float = 0.0,
    mm_thresh: int = 0,
    kmap_mode: str = "hashmap",
):
    """MLX-hybrid replacement for torchsparse.nn.functional.conv3d.

    Fast-path coverage:
    - non-transposed
    - kernel_size = (3,3,3)
    - stride = (1,1,1)
    - dilation = (1,1,1)
    - CPU input feats
    Otherwise it falls back to original torchsparse conv3d.
    """
    import torchsparse
    from torchsparse import SparseTensor
    from torchsparse.nn import functional as F

    kernel_size_nt = make_ntuple(kernel_size, ndim=3)
    stride_nt = make_ntuple(stride, ndim=3)
    dilation_nt = make_ntuple(dilation, ndim=3)

    if not (
        not transposed
        and kernel_size_nt == (3, 3, 3)
        and stride_nt == (1, 1, 1)
        and dilation_nt == (1, 1, 1)
        and input.feats.device.type == "cpu"
        and weight.ndim == 3
    ):
        return _ORIG_CONV3D(  # type: ignore[misc]
            input,
            weight,
            kernel_size=kernel_size,
            bias=bias,
            stride=stride,
            dilation=dilation,
            transposed=transposed,
            epsilon=epsilon,
            mm_thresh=mm_thresh,
            kmap_mode=kmap_mode,
        )

    feats = input.feats
    coords = input.coords
    kmap = input.kmaps.get((input.stride, kernel_size_nt, stride_nt, dilation_nt))
    if kmap is None:
        kmap_out = F.build_kernel_map(coords, kernel_size_nt, stride_nt, input.stride, kmap_mode)
        if len(kmap_out) == 2:
            nbmaps, nbsizes = kmap_out
            input_mask, output_mask = None, None
        elif len(kmap_out) == 4:
            nbmaps, nbsizes, input_mask, output_mask = kmap_out
        else:
            return _ORIG_CONV3D(  # type: ignore[misc]
                input,
                weight,
                kernel_size=kernel_size,
                bias=bias,
                stride=stride,
                dilation=dilation,
                transposed=transposed,
                epsilon=epsilon,
                mm_thresh=mm_thresh,
                kmap_mode=kmap_mode,
            )
        nbsizes = nbsizes.cpu()
        kmap = [
            nbmaps,
            nbsizes,
            (feats.shape[0], coords.shape[0]),
            input_mask,
            output_mask,
        ]
        input.kmaps[(input.stride, kernel_size_nt, stride_nt, dilation_nt)] = kmap

    nbmaps, nbsizes = kmap[0], kmap[1]
    out = torch.zeros(
        (coords.shape[0], weight.shape[-1]),
        dtype=feats.dtype,
        device=feats.device,
    )

    feats_np = feats.detach().cpu().numpy().astype(np.float32, copy=False)
    weight_np = weight.detach().cpu().numpy().astype(np.float32, copy=False)
    cur_st = 0
    for kernel_idx in range(weight_np.shape[0]):
        cur_ed = cur_st + int(nbsizes[kernel_idx])
        if cur_ed <= cur_st:
            continue
        maps = nbmaps[cur_st:cur_ed]
        in_map = maps[:, 0].long()
        out_map = maps[:, 1].long()
        cur_st = cur_ed

        in_np = feats_np[in_map.cpu().numpy()]
        prod_mx = mx.array(in_np) @ mx.array(weight_np[kernel_idx])
        mx.eval(prod_mx)
        prod_t = torch.from_numpy(np.asarray(prod_mx, dtype=np.float32)).to(
            device=out.device, dtype=out.dtype
        )
        out.index_add_(0, out_map, prod_t)

    if bias is not None:
        out += bias

    output = SparseTensor(
        coords=coords,
        feats=out,
        stride=tuple(input.stride[k] * stride_nt[k] for k in range(3)),
    )
    output.cmaps = input.cmaps
    output.cmaps.setdefault(output.stride, output.coords)
    output.kmaps = input.kmaps
    return output


def patch_torchsparse_conv3d_k3s1_with_mlx() -> None:
    """Monkey-patch torchsparse F.conv3d for k=3,s=1 non-transposed path."""
    global _ORIG_CONV3D
    import torchsparse.nn.functional as F
    import torchsparse.nn.functional.conv as conv_mod

    if _ORIG_CONV3D is None:
        _ORIG_CONV3D = conv_mod.conv3d
    F.conv3d = _conv3d_mlx
    conv_mod.conv3d = _conv3d_mlx

