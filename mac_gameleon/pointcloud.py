"""Minimal PointCloud helper for attribute-meta encode (no render deps)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d
import torch


class PointCloud:
    def __init__(
        self,
        xyz_w: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        normal_w: Optional[torch.Tensor] = None,
    ) -> None:
        self.xyz_w = xyz_w
        self.rgb = rgb
        self.normal_w = normal_w

    @staticmethod
    def from_o3d_pcd(o3d_pcd: o3d.geometry.PointCloud) -> PointCloud:
        xyz_w = torch.from_numpy(np.asarray(o3d_pcd.points, dtype=np.float32)).unsqueeze(0)
        if o3d_pcd.has_colors():
            rgb = torch.from_numpy(np.asarray(o3d_pcd.colors, dtype=np.float32)).unsqueeze(0)
        else:
            rgb = None
        if o3d_pcd.has_normals():
            normal_w = torch.from_numpy(np.asarray(o3d_pcd.normals, dtype=np.float32)).unsqueeze(0)
        else:
            normal_w = None
        return PointCloud(xyz_w=xyz_w, rgb=rgb, normal_w=normal_w)
