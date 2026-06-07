"""Apply geometry-meta patches to sibling Gameleon checkout."""

from __future__ import annotations

from pathlib import Path

from mac_gameleon.paths import GAMELEON_ROOT


def _log_patched(path: Path, *, quiet: bool) -> None:
    if not quiet:
        print(f"patched {path}")


def _patch_coder_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    helper = (
        "\n\n"
        "def _resolve_gameleon_device():\n"
        "    env = os.environ.get('GAMELEON_DEVICE', '').strip().lower()\n"
        "    if env:\n"
        "        return env\n"
        "    return 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    )
    if "_resolve_gameleon_device" not in text:
        anchor = "from basic_models.default import encode\nimport random\n"
        if anchor not in text:
            raise SystemExit(f"unexpected {path}: missing anchor for helper")
        text = text.replace(anchor, anchor + helper)

    old = "                device='cuda',"
    new = "                device=_resolve_gameleon_device(),"
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit(f"unexpected {path}: CoderIntra device line not found")

    old_ckpt = "        ckpt = torch.load(ckptdir, map_location=device)"
    new_ckpt = "        ckpt = torch.load(ckptdir, map_location=device, weights_only=False)"
    if old_ckpt in text:
        text = text.replace(old_ckpt, new_ckpt, 1)
    elif "weights_only=False" not in text:
        raise SystemExit(f"unexpected {path}: load_ckpt torch.load line not found")

    path.write_text(text)
    _log_patched(path, quiet=quiet)


def _patch_coder_intra_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    old_sig = "    def __init__(self, model_path, device='cuda:0', channels=32, kernel_size=3,"
    new_sig = "    def __init__(self, model_path, device=None, channels=32, kernel_size=3,"
    if old_sig in text:
        text = text.replace(old_sig, new_sig, 1)
    elif new_sig not in text:
        raise SystemExit(f"unexpected {path}: __init__ signature not found")

    anchor = "        super(CoderIntra, self).__init__()\n        self.device = device\n"
    replacement = (
        "        super(CoderIntra, self).__init__()\n"
        "        if device is None:\n"
        "            env = os.environ.get('GAMELEON_DEVICE', '').strip().lower()\n"
        "            device = env if env else ('cuda:0' if torch.cuda.is_available() else 'cpu')\n"
        "        self.device = device\n"
    )
    if anchor in text:
        text = text.replace(anchor, replacement, 1)
    elif "if device is None:" not in text:
        raise SystemExit(f"unexpected {path}: device assignment anchor not found")

    old_block = (
        "        # Set torchsparse config\n"
        "        conv_config = F.conv_config.get_default_conv_config()\n"
        "        conv_config.kmap_mode = \"hashmap\"\n"
        "        F.conv_config.set_global_conv_config(conv_config)\n"
    )
    new_block = (
        "        # Set torchsparse config (v2.1+); v2.0.0 has no global conv_config API.\n"
        "        if hasattr(F, \"conv_config\"):\n"
        "            conv_config = F.conv_config.get_default_conv_config()\n"
        "            conv_config.kmap_mode = \"hashmap\"\n"
        "            F.conv_config.set_global_conv_config(conv_config)\n"
    )
    if old_block in text:
        text = text.replace(old_block, new_block, 1)
    elif "if hasattr(F, \"conv_config\"):" not in text:
        raise SystemExit(f"unexpected {path}: conv config block not found")

    old_warmup = (
        "        # Warm-up\n"
        "        random_coords = torch.randint(low=0, high=2048, size=(2048, 3)).int().to(device)\n"
        "        self.model(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),\n"
        "                    feats=torch.ones((2048, 1))).to(device))\n"
    )
    meta_skip_warmup = (
        "        # Warm-up (CUDA only; skip on CPU — FOG multi-scale loop is slow and optional)\n"
        "        if not str(device).startswith('cpu'):\n"
        "            random_coords = torch.randint(low=0, high=2048, size=(2048, 3)).int().to(device)\n"
        "            self.model(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),\n"
        "                        feats=torch.ones((2048, 1))).to(device))\n"
    )
    old_cpu_warmup = (
        "        # Warm-up (smaller grid on CPU to avoid torchsparse int32 overflow in FOG)\n"
        "        warm_n, warm_high = (512, 64) if str(device).startswith('cpu') else (2048, 2048)\n"
        "        random_coords = torch.randint(low=0, high=warm_high, size=(warm_n, 3)).int().to(device)\n"
        "        self.model(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),\n"
        "                    feats=torch.ones((warm_n, 1))).to(device))\n"
    )
    if old_warmup in text:
        text = text.replace(old_warmup, meta_skip_warmup, 1)
    elif old_cpu_warmup in text:
        text = text.replace(old_cpu_warmup, meta_skip_warmup, 1)
    elif not ("if not str(device).startswith('cpu'):" in text and "Warm-up (CUDA only" in text):
        raise SystemExit(f"unexpected {path}: warm-up block not found")

    old_lossy = "            lossy_ckpt = torch.load(lossy_model_path, map_location=device)"
    new_lossy = (
        "            lossy_ckpt = torch.load(lossy_model_path, map_location=device, weights_only=False)"
    )
    if old_lossy in text:
        text = text.replace(old_lossy, new_lossy, 1)
    elif "lossy_model_path, map_location=device, weights_only=False" not in text:
        raise SystemExit(f"unexpected {path}: lossy torch.load line not found")

    path.write_text(text)
    _log_patched(path, quiet=quiet)


def _patch_kit_nn_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    old_fcg = (
        "        self.expand_coords_base = torch.tensor([\n"
        "            [0, 0, 0], # -> 1 (occupancy adder)\n"
        "            [1, 0, 0], # -> 2 (occupancy adder)\n"
        "            [0, 1, 0], # -> 4 (occupancy adder)\n"
        "            [1, 1, 0], # -> 8 (occupancy adder)\n"
        "            [0, 0, 1], # -> 16 (occupancy adder)\n"
        "            [1, 0, 1], # -> 32 (occupancy adder)\n"
        "            [0, 1, 1], # -> 64 (occupancy adder)\n"
        "            [1, 1, 1], # -> 128 (occupancy adder)\n"
        "        ], device='cuda')\n\n"
        "        self.pos = torch.arange(0, 8, device='cuda').view(1, 8)\n"
    )
    new_fcg = (
        "        expand_coords_base = torch.tensor([\n"
        "            [0, 0, 0], # -> 1 (occupancy adder)\n"
        "            [1, 0, 0], # -> 2 (occupancy adder)\n"
        "            [0, 1, 0], # -> 4 (occupancy adder)\n"
        "            [1, 1, 0], # -> 8 (occupancy adder)\n"
        "            [0, 0, 1], # -> 16 (occupancy adder)\n"
        "            [1, 0, 1], # -> 32 (occupancy adder)\n"
        "            [0, 1, 1], # -> 64 (occupancy adder)\n"
        "            [1, 1, 1], # -> 128 (occupancy adder)\n"
        "        ])\n"
        "        self.register_buffer('expand_coords_base', expand_coords_base)\n"
        "        self.register_buffer('pos', torch.arange(0, 8).view(1, 8))\n"
    )
    if old_fcg in text:
        text = text.replace(old_fcg, new_fcg, 1)
    elif "register_buffer('expand_coords_base'" not in text:
        raise SystemExit(f"unexpected {path}: FCG cuda block not found")

    old_fog = "        self.pos_multiplier = torch.tensor([[1, 2, 4]], device='cuda')\n"
    new_fog = "        self.register_buffer('pos_multiplier', torch.tensor([[1, 2, 4]]))\n"
    if old_fog in text:
        text = text.replace(old_fog, new_fog, 1)
    elif "register_buffer('pos_multiplier'" not in text:
        raise SystemExit(f"unexpected {path}: FOG cuda line not found")

    old_fog_fwd = (
        "        ds_x = self.conv(x) # coordinate = ds_x.C and occupancy = ds_x.F\n"
        "        return ds_x\n"
    )
    new_fog_fwd = (
        "        ds_x = self.conv(x) # coordinate = ds_x.C and occupancy = ds_x.F\n"
        "        # Coords are already downsampled; reset stride so FOG loops do not\n"
        "        # accumulate past int32 limits in sparse hash paths on Metal.\n"
        "        return SparseTensor(coords=ds_x.coords, feats=ds_x.feats, stride=(1, 1, 1))\n"
    )
    if old_fog_fwd in text:
        text = text.replace(old_fog_fwd, new_fog_fwd, 1)
    elif "reset stride so FOG loops" not in text:
        raise SystemExit(f"unexpected {path}: FOG forward return not found")

    path.write_text(text)
    _log_patched(path, quiet=quiet)


def _patch_checkpoint_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    old = "    ckpt = torch.load(checkpoint_path, map_location=map_location)"
    new = "    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)"
    if old in text:
        text = text.replace(old, new, 1)
    elif "weights_only=False" not in text:
        raise SystemExit(f"unexpected {path}: torch.load line not found")
    path.write_text(text)
    _log_patched(path, quiet=quiet)


def _patch_sparse_tensor_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    old = "from pytorch3d.ops.knn import knn_points, knn_gather\n"
    new = (
        "try:\n"
        "    from pytorch3d.ops.knn import knn_points, knn_gather\n"
        "except ImportError:\n"
        "    knn_points = None\n"
        "    knn_gather = None\n"
    )
    if old in text:
        text = text.replace(old, new, 1)
    elif not ("except ImportError:" in text and "knn_points = None" in text):
        raise SystemExit(f"unexpected {path}: pytorch3d import not found")
    path.write_text(text)
    _log_patched(path, quiet=quiet)


def apply_geometry_meta_patches(
    gameleon_root: Path | None = None,
    *,
    quiet: bool = True,
) -> None:
    root = gameleon_root or GAMELEON_ROOT
    pkg = root / "gameleon"
    if not pkg.is_dir():
        raise SystemExit(f"Gameleon package not found: {pkg}")

    _patch_coder_py(pkg / "geometry" / "coder.py", quiet=quiet)
    _patch_coder_intra_py(pkg / "lossless_torchsparse" / "src" / "coder" / "coder_intra.py", quiet=quiet)
    _patch_checkpoint_py(pkg / "lossless_torchsparse" / "src" / "kit" / "checkpoint.py", quiet=quiet)
    _patch_kit_nn_py(pkg / "lossless_torchsparse" / "src" / "kit" / "nn.py", quiet=quiet)
    _patch_sparse_tensor_py(pkg / "data_utils" / "sparse_tensor.py", quiet=quiet)
