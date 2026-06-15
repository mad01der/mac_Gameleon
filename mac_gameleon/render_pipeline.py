"""Post-decode render + PSNR using gsplat-mlx (Mac port of Gameleon Step 3 tail)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx
import numpy as np
import torch
from gsplat_mlx.rendering import rasterization
from PIL import Image

from mac_gameleon.camera import build_cardinal_cameras
from mac_gameleon.gameleon_render import render_mesh_gt_gameleon
from mac_gameleon.paths import DEFAULT_MESH_GT, render_output_paths

DEFAULT_FOV = 45
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_SUPER_SAMPLE_RATE = 1
DEFAULT_NUM_VIEWS = 4
DEFAULT_BACKGROUND_COLOR = 1.0
_SH_DC_SCALE = 0.28209479177387814


def pcgc_rescale_np(
    xyz: np.ndarray,
    offset: float | int,
    scale_factor: float | int,
) -> np.ndarray:
    return (xyz.astype(np.float32) - float(offset)) / float(scale_factor)


def prune_decoded_gaussian_dict(
    decoded_gaussian_dict: dict[str, object],
    *,
    prune_opacity: bool = True,
) -> dict[str, object]:
    decoded_primitives = list(decoded_gaussian_dict["decoded_primitives"])
    decoded_dc = list(decoded_gaussian_dict["decoded_dc"])
    decoded_r = list(decoded_gaussian_dict["decoded_r"])
    decoded_s = list(decoded_gaussian_dict["decoded_s"])
    decoded_o = list(decoded_gaussian_dict["decoded_o"])

    if prune_opacity:
        non_zero_indices = [torch.nonzero(opacity > 0)[:, 0] for opacity in decoded_o]
        decoded_primitives = [decoded_primitives[i][non_zero_indices[i]] for i in range(len(decoded_primitives))]
        decoded_dc = [decoded_dc[i][non_zero_indices[i]] for i in range(len(decoded_dc))]
        decoded_r = [decoded_r[i][non_zero_indices[i]] for i in range(len(decoded_r))]
        decoded_s = [decoded_s[i][non_zero_indices[i]] for i in range(len(decoded_s))]
        decoded_o = [decoded_o[i][non_zero_indices[i]] for i in range(len(decoded_o))]

    return {
        "decoded_primitives": decoded_primitives,
        "decoded_dc": decoded_dc,
        "decoded_r": decoded_r,
        "decoded_s": decoded_s,
        "decoded_o": decoded_o,
        "decoded_texture_feature": decoded_gaussian_dict.get("decoded_texture_feature"),
    }


def _normalize_quaternion_np(quaternion: np.ndarray) -> np.ndarray:
    quat = quaternion.astype(np.float32, copy=False)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.clip(norms, 1e-8, None)


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) - 0.5) / _SH_DC_SCALE


def _inverse_sigmoid_np(opacity: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    opacity = np.clip(opacity.astype(np.float32), eps, 1.0 - eps)
    return np.log(opacity / (1.0 - opacity))


def export_decoded_ply(
    decoded_gaussian_dict: dict[str, object],
    output_path: str | Path,
    *,
    frame: int = 0,
    num_f_rest: int = 45,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = decoded_gaussian_dict["decoded_primitives"][frame].detach().cpu().numpy().astype(np.float32)
    rgb = decoded_gaussian_dict["decoded_dc"][frame].detach().cpu().numpy().astype(np.float32)
    rotation = _normalize_quaternion_np(decoded_gaussian_dict["decoded_r"][frame].detach().cpu().numpy())
    scale = np.log(np.clip(decoded_gaussian_dict["decoded_s"][frame].detach().cpu().numpy().astype(np.float32), 1e-8, None))
    opacity = _inverse_sigmoid_np(
        decoded_gaussian_dict["decoded_o"][frame].detach().cpu().numpy().reshape(-1, 1)
    )
    f_dc = _rgb_to_sh_dc(rgb)

    dtype_fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
    ]
    dtype_fields.extend((f"f_rest_{idx}", "<f4") for idx in range(num_f_rest))
    dtype_fields.extend(
        [
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )

    vertex = np.empty(xyz.shape[0], dtype=np.dtype(dtype_fields))
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["f_dc_0"] = f_dc[:, 0]
    vertex["f_dc_1"] = f_dc[:, 1]
    vertex["f_dc_2"] = f_dc[:, 2]
    for idx in range(num_f_rest):
        vertex[f"f_rest_{idx}"] = 0.0
    vertex["opacity"] = opacity[:, 0]
    vertex["scale_0"] = scale[:, 0]
    vertex["scale_1"] = scale[:, 1]
    vertex["scale_2"] = scale[:, 2]
    vertex["rot_0"] = rotation[:, 0]
    vertex["rot_1"] = rotation[:, 1]
    vertex["rot_2"] = rotation[:, 2]
    vertex["rot_3"] = rotation[:, 3]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    header_lines.extend(f"property float f_rest_{idx}" for idx in range(num_f_rest))
    header_lines.extend(
        [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )

    with output_path.open("wb") as handle:
        handle.write(("\n".join(header_lines) + "\n").encode("ascii"))
        vertex.tofile(handle)
    return output_path


def compute_psnr_from_arrays(
    gt_images: Sequence[np.ndarray],
    dec_images: Sequence[np.ndarray],
) -> list[float]:
    if len(gt_images) != len(dec_images):
        raise ValueError(f"GT and decoded image counts differ: {len(gt_images)} vs {len(dec_images)}")
    psnrs: list[float] = []
    for gt, dec in zip(gt_images, dec_images):
        mse = float(np.mean((gt.astype(np.float32) - dec.astype(np.float32)) ** 2))
        mse = max(mse, 1e-12)
        psnrs.append(10.0 * math.log10(1.0 / mse))
    return psnrs


def render_dir_ready(render_dir: Path, num_views: int) -> bool:
    for idx in range(num_views):
        if not (render_dir / f"rgb_{idx}.png").is_file():
            return False
    # Ignore cache from an older run with more views (e.g. 12 -> 4).
    if (render_dir / f"rgb_{num_views}.png").is_file():
        return False
    return True


def _scale_intrinsics(K: np.ndarray, super_sample_rate: int) -> np.ndarray:
    if super_sample_rate <= 1:
        return K.astype(np.float32, copy=True)
    scaled = K.astype(np.float32, copy=True)
    scaled[..., 0, 0] *= super_sample_rate
    scaled[..., 1, 1] *= super_sample_rate
    scaled[..., 0, 2] = (scaled[..., 0, 2] + 0.5) * super_sample_rate - 0.5
    scaled[..., 1, 2] = (scaled[..., 1, 2] + 0.5) * super_sample_rate - 0.5
    return scaled


def _downsample_rgb(image: np.ndarray, super_sample_rate: int) -> np.ndarray:
    if super_sample_rate <= 1:
        return image
    height, width = image.shape[:2]
    out_h = height // super_sample_rate
    out_w = width // super_sample_rate
    trimmed = image[: out_h * super_sample_rate, : out_w * super_sample_rate]
    blocks = trimmed.reshape(out_h, super_sample_rate, out_w, super_sample_rate, -1)
    return blocks.mean(axis=(1, 3))


def _write_png(path: Path, rgb_float: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_uint8 = np.clip(rgb_float * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(path, compress_level=0, optimize=False)


def _read_png_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def _build_render_cameras(
    *,
    fov: int,
    width: int,
    height: int,
    num_views: int = DEFAULT_NUM_VIEWS,
) -> tuple[np.ndarray, np.ndarray]:
    return build_cardinal_cameras(
        num_views=num_views,
        width=width,
        height=height,
        fov_deg=float(fov),
    )


def _prepare_render_arrays(
    decoded_gaussian_dict: dict[str, object],
    *,
    offset: float | int,
    scale_factor: float | int,
    prune_opacity: bool,
    frame: int = 0,
) -> dict[str, np.ndarray]:
    pruned = prune_decoded_gaussian_dict(decoded_gaussian_dict, prune_opacity=prune_opacity)
    radius = math.sqrt(3.0) / float(scale_factor) * 6.0
    means = pcgc_rescale_np(
        pruned["decoded_primitives"][frame].detach().cpu().numpy(),
        offset,
        scale_factor,
    )
    quats = _normalize_quaternion_np(pruned["decoded_r"][frame].detach().cpu().numpy())
    scales = pruned["decoded_s"][frame].detach().cpu().numpy().astype(np.float32) * radius
    opacities = pruned["decoded_o"][frame].detach().cpu().numpy().reshape(-1).astype(np.float32)
    colors = np.clip(pruned["decoded_dc"][frame].detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    if means.shape[0] == 0:
        raise ValueError("No Gaussians left after opacity pruning")
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "num_points": int(means.shape[0]),
    }


def _render_gaussians_to_pngs(
    arrays: dict[str, np.ndarray],
    *,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: Path,
    width: int,
    height: int,
    super_sample_rate: int,
    background_color: float,
    log: Optional[Callable[[str], None]] = None,
    label: str = "render",
) -> tuple[list[np.ndarray], float]:
    """Render views one at a time so progress is visible and PNGs can resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    render_w = int(width * super_sample_rate)
    render_h = int(height * super_sample_rate)
    K = _scale_intrinsics(intrinsics, super_sample_rate)
    num_views = int(viewmats.shape[0])
    num_points = int(arrays["num_points"])

    means_mx = mx.array(arrays["means"])
    quats_mx = mx.array(arrays["quats"])
    scales_mx = mx.array(arrays["scales"])
    opacities_mx = mx.array(arrays["opacities"])
    colors_mx = mx.array(arrays["colors"][None, :, :])

    rgb_stack: list[np.ndarray] = []
    rasterize_sec = 0.0
    for view_idx in range(num_views):
        png_path = output_dir / f"rgb_{view_idx}.png"
        if png_path.is_file():
            rgb_stack.append(_read_png_float(png_path))
            continue

        if log is not None:
            log(
                f"{label}: view {view_idx + 1}/{num_views} "
                f"({num_points} splats, {render_w}x{render_h})..."
            )

        view_t0 = time.perf_counter()
        bg = np.full((1, 3), float(background_color), dtype=np.float32)
        render_colors, _render_alphas, _info = rasterization(
            means=means_mx,
            quats=quats_mx,
            scales=scales_mx,
            opacities=opacities_mx,
            colors=colors_mx,
            viewmats=mx.array(viewmats[view_idx : view_idx + 1]),
            Ks=mx.array(K[view_idx : view_idx + 1]),
            width=render_w,
            height=render_h,
            backgrounds=mx.array(bg),
            render_mode="RGB",
            sh_degree=None,
            differentiable=False,
        )
        mx.eval(render_colors)
        rasterize_sec += time.perf_counter() - view_t0

        frame = np.array(render_colors[0], dtype=np.float32)
        frame = _downsample_rgb(frame, super_sample_rate)
        rgb_stack.append(frame)
        _write_png(png_path, frame)
        if log is not None:
            log(f"{label}: view {view_idx + 1}/{num_views} done in {time.perf_counter() - view_t0:.1f}s")

    return rgb_stack, rasterize_sec


def export_decoded_gaussian_ply(
    decoded_gaussian_dict: dict[str, object],
    outdir: Path,
    *,
    level: int,
) -> Path:
    """Write standard 3DGS PLY from decode output (no rendering)."""
    paths = render_output_paths(outdir, level=level)
    paths["render_root"].mkdir(parents=True, exist_ok=True)
    return export_decoded_ply(decoded_gaussian_dict, paths["decoded_gaussian_ply"])


def write_decode_summary(
    outdir: Path,
    *,
    level: int,
    rate_metrics: dict[str, float | int],
    decode_sec: float,
    decode_timing: Optional[dict[str, Any]],
    decoded_pcd_path: str,
    gaussian_points: int,
    render_info: Optional[dict[str, Any]] = None,
    codec_timing: Optional[dict[str, float]] = None,
) -> Path:
    """Write Gameleon-aligned summary.json with bpp (+ optional render metrics)."""
    paths = render_output_paths(outdir, level=level)
    summary: dict[str, Any] = {
        "level": int(level),
        "gaussian_points": int(gaussian_points),
        "decoded_pcd_path": decoded_pcd_path,
        "attribute_decode_sec": float(decode_sec),
        "decode_timing": dict(decode_timing or {}),
        **rate_metrics,
    }
    if codec_timing:
        summary.update(codec_timing)
    if render_info is not None:
        summary.update(
            {
                "render_psnr": render_info.get("render_psnr"),
                "per_view_psnr": render_info.get("per_view_psnr"),
                "num_rendered_points": render_info.get("num_rendered_points"),
                "decoded_render_dir": render_info.get("render_dir"),
                "gt_render_dir": render_info.get("gt_render_dir"),
                "render_timing": render_info.get("render_timing"),
                "render_backend_decoded": "gsplat-mlx",
                "render_backend_gt": render_info.get("render_backend_gt", "gameleon_mesh"),
                "gt_mode": render_info.get("gt_mode", "mesh"),
                "camera_backend": "mac_gameleon.cardinal",
            }
        )
    summary_path = paths["summary_json"]
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def run_post_decode_render(
    adapter,
    decoded_gaussian_dict: dict[str, object],
    *,
    outdir: Path,
    level: int,
    mesh_gt: Optional[Path] = None,
    fov: int = DEFAULT_FOV,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    super_sample_rate: int = DEFAULT_SUPER_SAMPLE_RATE,
    background_color: float = DEFAULT_BACKGROUND_COLOR,
    manual_scale: bool = False,
    mesh_input_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    prune_opacity: bool = True,
    write_render_pngs: bool = True,
    write_decoded_ply: bool = True,
    write_summary: bool = True,
    decode_sec: float = 0.0,
    decode_timing: Optional[dict[str, Any]] = None,
    rate_metrics: Optional[dict[str, float | int]] = None,
    codec_timing: Optional[dict[str, float]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    paths = render_output_paths(outdir, level=level)
    render_root = paths["render_root"]
    decoded_render_dir = paths["decoded_render_dir"]
    gt_render_dir = paths["gt_render_dir"]
    render_root.mkdir(parents=True, exist_ok=True)
    decoded_render_dir.mkdir(parents=True, exist_ok=True)
    gt_render_dir.mkdir(parents=True, exist_ok=True)

    viewmats, intrinsics = _build_render_cameras(fov=fov, width=width, height=height)
    num_views = int(viewmats.shape[0])

    gt_rgb_stack: list[np.ndarray] = []
    gt_render_sec = 0.0
    if mesh_gt is None:
        mesh_gt = DEFAULT_MESH_GT
    if mesh_gt.is_file():
        if render_dir_ready(gt_render_dir, num_views):
            if log is not None:
                log(f"GT render: reusing cached PNGs in {gt_render_dir.name}/")
            gt_rgb_stack = [_read_png_float(gt_render_dir / f"rgb_{idx}.png") for idx in range(num_views)]
        else:
            gt_render_sec = render_mesh_gt_gameleon(
                mesh_gt,
                gt_render_dir,
                fov=fov,
                width=width,
                height=height,
                num_views=num_views,
                background_color=background_color,
                manual_scale=manual_scale,
                mesh_input_offset=mesh_input_offset,
                log=log,
            )
            gt_rgb_stack = [_read_png_float(gt_render_dir / f"rgb_{idx}.png") for idx in range(num_views)]
    elif log is not None:
        log(f"GT mesh missing, skipping PSNR: {mesh_gt}")

    decoded_pcd_path = None
    if write_decoded_ply:
        decoded_pcd_path = export_decoded_ply(
            decoded_gaussian_dict,
            paths["decoded_gaussian_ply"],
        )

    render_arrays = _prepare_render_arrays(
        decoded_gaussian_dict,
        offset=adapter.offset,
        scale_factor=adapter.scale_factor,
        prune_opacity=prune_opacity,
    )
    decoded_rgb_stack: list[np.ndarray] = []
    rasterize_sec = 0.0
    if write_render_pngs:
        if render_dir_ready(decoded_render_dir, num_views):
            if log is not None:
                log(f"Decoded render: reusing cached PNGs in {decoded_render_dir.name}/")
            decoded_rgb_stack = [
                _read_png_float(decoded_render_dir / f"rgb_{idx}.png") for idx in range(num_views)
            ]
        else:
            if log is not None:
                log(
                    f"Decoded render: {render_arrays['num_points']} splats, "
                    f"{num_views} views, {width}x{height}..."
                )
            decoded_rgb_stack, rasterize_sec = _render_gaussians_to_pngs(
                render_arrays,
                viewmats=viewmats,
                intrinsics=intrinsics,
                output_dir=decoded_render_dir,
                width=width,
                height=height,
                super_sample_rate=super_sample_rate,
                background_color=background_color,
                log=log,
                label="Decoded render",
            )

    per_view_psnr: list[float] = []
    render_psnr: Optional[float] = None
    if gt_rgb_stack and decoded_rgb_stack and len(gt_rgb_stack) == len(decoded_rgb_stack):
        per_view_psnr = compute_psnr_from_arrays(gt_rgb_stack, decoded_rgb_stack)
        render_psnr = float(sum(per_view_psnr) / len(per_view_psnr))

    render_info = {
        "render_root": str(render_root),
        "render_dir": str(decoded_render_dir),
        "gt_render_dir": str(gt_render_dir),
        "decoded_pcd_path": None if decoded_pcd_path is None else str(decoded_pcd_path),
        "num_rendered_points": int(render_arrays["num_points"]),
        "num_views": num_views,
        "render_psnr": render_psnr,
        "per_view_psnr": per_view_psnr,
        "render_timing": {
            "render_rasterize": float(rasterize_sec),
            "gt_render": float(gt_render_sec),
            "render_total": float(rasterize_sec + gt_render_sec),
        },
        "render_backend_decoded": "gsplat-mlx",
        "render_backend_gt": "gameleon_mesh",
        "gt_mode": "mesh",
    }

    if write_summary:
        summary: dict[str, Any] = {
            "level": int(level),
            "render_psnr": render_psnr,
            "per_view_psnr": per_view_psnr,
            "num_rendered_points": int(render_arrays["num_points"]),
            "decoded_render_dir": str(decoded_render_dir),
            "gt_render_dir": str(gt_render_dir),
            "decoded_pcd_path": render_info["decoded_pcd_path"],
            "attribute_decode_sec": float(decode_sec),
            "decode_timing": dict(decode_timing or {}),
            "render_timing": render_info["render_timing"],
            "render_backend_decoded": "gsplat-mlx",
            "render_backend_gt": "gameleon_mesh",
            "gt_mode": "mesh",
            "camera_backend": "mac_gameleon.cardinal",
        }
        if rate_metrics is not None:
            summary.update(rate_metrics)
        if codec_timing:
            summary.update(codec_timing)
        summary_path = paths["summary_json"]
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        render_info["summary_json"] = str(summary_path)

    return render_info
