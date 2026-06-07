"""Resolve compute device for Mac CPU Gameleon runs."""

from __future__ import annotations

import os

import torch


def resolve_gameleon_device(explicit: str | None = None) -> str:
    """Return a torch device string such as ``cpu`` or ``cuda:0``."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ.get("GAMELEON_DEVICE", "").strip().lower()
    if env:
        return env
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_torch_device(explicit: str | None = None) -> torch.device:
    return torch.device(resolve_gameleon_device(explicit))
