"""Apply attribute-meta patches to sibling Gameleon checkout."""

from __future__ import annotations

from pathlib import Path

from mac_gameleon.paths import GAMELEON_ROOT


def _log_patched(path: Path, *, quiet: bool) -> None:
    if not quiet:
        print(f"patched {path}")


def _patch_attribute_adapter_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    orig = text

    helper = (
        "\n\n"
        "def _resolve_gameleon_device():\n"
        "    env = os.environ.get('GAMELEON_DEVICE', '').strip().lower()\n"
        "    if env:\n"
        "        return env\n"
        "    return 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    )
    anchor = "from scipy.spatial import cKDTree\n"
    if "_resolve_gameleon_device" not in text:
        if anchor not in text:
            raise SystemExit(f"unexpected {path}: missing scipy import anchor")
        text = text.replace(anchor, anchor + helper, 1)

    old_device = '        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    new_device = "        self.device = torch.device(_resolve_gameleon_device())"
    if old_device in text:
        text = text.replace(old_device, new_device, 1)
    elif new_device not in text:
        raise SystemExit(f"unexpected {path}: device assignment not found")

    if text != orig:
        path.write_text(text)
        _log_patched(path, quiet=quiet)


def _patch_simple_raw_render_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    orig = text

    patched_import = (
        "try:\n"
        "    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer\n"
        "except ImportError:\n"
        "    GaussianRasterizationSettings = None\n"
        "    GaussianRasterizer = None\n"
    )
    if patched_import not in text:
        old = (
            "from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer\n"
        )
        if old in text:
            text = text.replace(old, patched_import, 1)
        elif "GaussianRasterizer = None" not in text:
            raise SystemExit(f"unexpected {path}: diff_gaussian import not found")

    old_load = "    pcml_ckpt = torch.load(ckpt, map_location=map_location)"
    new_load = "    pcml_ckpt = torch.load(ckpt, map_location=map_location, weights_only=False)"
    if old_load in text:
        text = text.replace(old_load, new_load, 1)
    elif "weights_only=False" not in text:
        raise SystemExit(f"unexpected {path}: load_pcml torch.load line not found")

    old_ret = "    print(ret)\n    print('Loaded weights for pcml.')"
    new_ret = (
        "    missing = len(ret.missing_keys)\n"
        "    unexpected = len(ret.unexpected_keys)\n"
        "    if missing or unexpected:\n"
        "        print(f'PCML checkpoint loaded (missing={missing}, unexpected={unexpected})')\n"
    )
    if old_ret in text:
        text = text.replace(old_ret, new_ret, 1)
    elif "print(ret)" in text and "PCML checkpoint loaded" not in text:
        text = text.replace("    print(ret)\n", "", 1)
        text = text.replace("    print('Loaded weights for pcml.')\n", "", 1)

    old_opt = "    print(opt_pth)\n\n"
    if old_opt in text:
        text = text.replace(old_opt, "", 1)

    if text != orig:
        path.write_text(text)
        _log_patched(path, quiet=quiet)


def _patch_pcml_model_quiet(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    orig = text
    marker = "# mac_gameleon: quiet init prints unless GAMELEON_VERBOSE=1"
    if marker in text:
        return

    import_anchor = "import time\n"
    if import_anchor in text and "\nimport os\n" not in text[: text.find(import_anchor) + 1]:
        text = text.replace(import_anchor, "import os\n" + import_anchor, 1)

    old_block = (
        "        print(\n"
        "            'Attribute sparse kernel sizes: '\n"
        "            f'spatial={self.attr_spatial_kernel_size}, '\n"
        "            f'entropy={self.attr_entropy_kernel_size}, '\n"
        "            f'fusion={self.attr_fusion_kernel_size}, '\n"
        "            f'generator={self.attr_generator_kernel_size}'\n"
        "        )\n"
        "        print(\n"
        "            'Attribute block layers: '\n"
        "            f'extractor={self.attr_extractor_block_layers}, '\n"
        "            f'entropy={self.attr_entropy_block_layers}, '\n"
        "            f'fusion={self.attr_fusion_block_layers}, '\n"
        "            f'generator={self.attr_generator_block_layers}'\n"
        "        )"
    )
    new_block = (
        f"        {marker}\n"
        "        if os.environ.get('GAMELEON_VERBOSE', '').strip().lower() in {'1', 'true', 'yes'}:\n"
        "            print(\n"
        "                'Attribute sparse kernel sizes: '\n"
        "                f'spatial={self.attr_spatial_kernel_size}, '\n"
        "                f'entropy={self.attr_entropy_kernel_size}, '\n"
        "                f'fusion={self.attr_fusion_kernel_size}, '\n"
        "                f'generator={self.attr_generator_kernel_size}'\n"
        "            )\n"
        "            print(\n"
        "                'Attribute block layers: '\n"
        "                f'extractor={self.attr_extractor_block_layers}, '\n"
        "                f'entropy={self.attr_entropy_block_layers}, '\n"
        "                f'fusion={self.attr_fusion_block_layers}, '\n"
        "                f'generator={self.attr_generator_block_layers}'\n"
        "            )"
    )
    if old_block not in text:
        raise SystemExit(f"unexpected {path}: missing PCML init print blocks")
    text = text.replace(old_block, new_block, 1)

    if text != orig:
        path.write_text(text)
        _log_patched(path, quiet=quiet)


def _patch_attribute_adapter_pointcloud_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    orig = text
    old = "from gameleon_attribute.structures import PointCloud\n"
    new = (
        "try:\n"
        "    from mac_gameleon.pointcloud import PointCloud\n"
        "except ImportError:\n"
        "    from gameleon_attribute.structures import PointCloud\n"
    )
    if "from mac_gameleon.pointcloud import PointCloud" in text:
        return
    if old in text:
        text = text.replace(old, new, 1)
    else:
        raise SystemExit(f"unexpected {path}: PointCloud import not found")
    if text != orig:
        path.write_text(text)
        _log_patched(path, quiet=quiet)


def _patch_attribute_adapter_imports_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    orig = text
    old = "from gameleon_attribute.simple_raw_render import generate_cam, load_pcml\n"
    new = (
        "try:\n"
        "    from mac_gameleon.pcml_loader import load_pcml\n"
        "except ImportError:\n"
        "    from gameleon_attribute.simple_raw_render import load_pcml\n"
        "\n"
        "def _lazy_generate_cam(camera_info, save_temp_state_dict=False):\n"
        "    from gameleon_attribute.simple_raw_render import generate_cam\n"
        "    return generate_cam(camera_info, save_temp_state_dict=save_temp_state_dict)\n"
    )
    if "from mac_gameleon.pcml_loader import load_pcml" in text:
        pass
    elif old in text:
        text = text.replace(old, new, 1)
    else:
        raise SystemExit(f"unexpected {path}: simple_raw_render import not found")

    old_call = "        return generate_cam(camera_info, save_temp_state_dict=False)"
    new_call = "        return _lazy_generate_cam(camera_info, save_temp_state_dict=False)"
    if old_call in text:
        text = text.replace(old_call, new_call, 1)
    elif new_call not in text:
        raise SystemExit(f"unexpected {path}: generate_cam call site not found")

    if text != orig:
        path.write_text(text)
        _log_patched(path, quiet=quiet)


def _patch_plib_render_py(path: Path, *, quiet: bool) -> None:
    text = path.read_text()
    orig = text

    replacements = [
        (
            "import cv2\n",
            "try:\n    import cv2\nexcept ImportError:\n    cv2 = None\n",
        ),
        (
            "import xatlas\n",
            "try:\n    import xatlas\nexcept ImportError:\n    xatlas = None\n",
        ),
    ]
    for old, new in replacements:
        if new in text:
            continue
        if old in text:
            text = text.replace(old, new, 1)
        elif old.strip().split()[1] + " = None" not in text:
            raise SystemExit(f"unexpected {path}: missing {old.strip()}")

    if text != orig:
        path.write_text(text)
        _log_patched(path, quiet=quiet)


def apply_attribute_meta_patches(
    gameleon_root: Path | None = None,
    *,
    quiet: bool = True,
) -> None:
    root = gameleon_root or GAMELEON_ROOT
    pkg = root / "gameleon"
    if not pkg.is_dir():
        raise SystemExit(f"Gameleon package not found: {pkg}")

    _patch_attribute_adapter_py(pkg / "core" / "attribute_adapter.py", quiet=quiet)
    _patch_attribute_adapter_imports_py(pkg / "core" / "attribute_adapter.py", quiet=quiet)
    _patch_attribute_adapter_pointcloud_py(pkg / "core" / "attribute_adapter.py", quiet=quiet)
    _patch_simple_raw_render_py(pkg / "gameleon_attribute" / "simple_raw_render.py", quiet=quiet)
    _patch_pcml_model_quiet(pkg / "gameleon_attribute" / "models" / "pcml_model.py", quiet=quiet)
    _patch_plib_render_py(pkg / "gameleon_attribute" / "plib" / "render.py", quiet=quiet)
