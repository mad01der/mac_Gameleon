"""Load standard 3D Gaussian Splatting PLY (Gameleon export compatible)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class GaussianPlyData:
    means: np.ndarray  # [N, 3] float32
    quats: np.ndarray  # [N, 4] float32 (w, x, y, z)
    scales: np.ndarray  # [N, 3] float32 (activated, positive)
    opacities: np.ndarray  # [N] float32 in [0, 1]
    sh_coeffs: np.ndarray  # [N, K, 3] float32
    sh_degree: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _parse_ply_header(path: Path) -> Tuple[str, int, List[str], int]:
    properties: List[str] = []
    fmt = ""
    vertex_count = 0
    header_bytes = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            header_bytes += len(line)
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property "):
                properties.append(text.split()[-1])
            elif text == "end_header":
                break
    return fmt, vertex_count, properties, header_bytes


def _infer_sh_degree(num_f_rest: int) -> int:
    # 3 f_dc + num_f_rest = 3 * (degree+1)^2  =>  (degree+1)^2 = 1 + num_f_rest/3
    total_per_channel = 1 + num_f_rest // 3
    degree = int(round(np.sqrt(total_per_channel) - 1.0))
    expected_rest = 3 * ((degree + 1) ** 2 - 1)
    if expected_rest != num_f_rest:
        raise ValueError(
            f"Cannot infer SH degree from f_rest count {num_f_rest} (expected {expected_rest} for degree {degree})"
        )
    return degree


def _build_sh_coeffs(
    f_dc: np.ndarray,
    f_rest_flat: np.ndarray,
    sh_degree: int,
) -> np.ndarray:
    """Pack f_dc + f_rest_* into [N, K, 3] (Inria 3DGS layout)."""
    n = f_dc.shape[0]
    k = (sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), dtype=np.float32)
    sh[:, 0, :] = f_dc.astype(np.float32)
    if k > 1:
        rest = f_rest_flat.reshape(n, 3, k - 1).transpose(0, 2, 1).astype(np.float32)
        sh[:, 1:, :] = rest
    return sh


def load_gaussian_ply(path: str | Path, max_points: int | None = None) -> GaussianPlyData:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    fmt, vertex_count, properties, header_bytes = _parse_ply_header(path)
    if vertex_count <= 0:
        raise ValueError(f"PLY has no vertices: {path}")

    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = required - set(properties)
    if missing:
        raise ValueError(f"PLY missing properties {sorted(missing)}: {path}")

    f_dc_names = [p for p in properties if p in ("f_dc_0", "f_dc_1", "f_dc_2")]
    if len(f_dc_names) != 3:
        raise ValueError(f"PLY must contain f_dc_0..2, got {f_dc_names}")

    f_rest_names = sorted(
        [p for p in properties if p.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    sh_degree = _infer_sh_degree(len(f_rest_names)) if f_rest_names else 0

    dtype_fields = [(name, "<f4") for name in properties]
    vertex_dtype = np.dtype(dtype_fields)

    if fmt == "binary_little_endian":
        with path.open("rb") as f:
            f.seek(header_bytes)
            blob = f.read(vertex_dtype.itemsize * vertex_count)
        if len(blob) != vertex_dtype.itemsize * vertex_count:
            raise ValueError(f"Truncated binary PLY body: {path}")
        vertices = np.frombuffer(blob, dtype=vertex_dtype, count=vertex_count)
    elif fmt == "ascii":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(len(properties) + 4):
                f.readline()
            rows = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    raise ValueError(f"Truncated ascii PLY body: {path}")
                rows.append([float(x) for x in line.split()])
        vertices = np.array(rows, dtype=np.float32)
        # rebuild structured array
        structured = np.empty(vertex_count, dtype=vertex_dtype)
        for idx, name in enumerate(properties):
            structured[name] = vertices[:, idx]
        vertices = structured
    else:
        raise ValueError(f"Unsupported PLY format {fmt!r} in {path}")

    if max_points is not None and vertex_count > max_points:
        vertices = vertices[:max_points]

    means = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)
    quats = np.stack(
        [vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]],
        axis=1,
    ).astype(np.float32)
    quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(quat_norm, 1e-8, None)

    scales = np.exp(
        np.stack(
            [vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]],
            axis=1,
        ).astype(np.float32)
    )
    opacities = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float32))

    f_dc = np.stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
        axis=1,
    ).astype(np.float32)
    if f_rest_names:
        f_rest_flat = np.stack([vertices[name] for name in f_rest_names], axis=1).astype(np.float32)
    else:
        f_rest_flat = np.zeros((means.shape[0], 0), dtype=np.float32)

    sh_coeffs = _build_sh_coeffs(f_dc, f_rest_flat, sh_degree)

    return GaussianPlyData(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh_coeffs=sh_coeffs,
        sh_degree=sh_degree,
    )
