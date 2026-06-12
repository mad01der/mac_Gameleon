"""Lightweight PCML checkpoint loader for attribute-meta (no render deps)."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml


def load_pcml(ckpt: str, model_class, debug: bool = False):
    ckpt_path = Path(ckpt)
    opt_path = ckpt_path.parents[1] / "option" / "options.yaml"

    with opt_path.open("r", encoding="utf-8") as opt_file:
        data = yaml.load(opt_file, Loader=yaml.FullLoader)

    info = data["pcml_info"]
    info["rounding_mode"] = "round"
    info["profiling"] = bool(debug)
    info["debug"] = debug
    model = model_class(info)
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    pcml_ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    model.load_state_dict(pcml_ckpt["pcml_model"], strict=False)
    return model, info
