"""Shared bootstrap for attribute-meta / encode smoke scripts."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]


def setup_attribute_env(*, no_lattice: bool = False) -> None:
    os.environ.setdefault("GAMELEON_DEVICE", "cpu")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(ROOT / ".cache" / "torch_extensions"),
    )
    if no_lattice:
        os.environ["GAMELEON_ME_LATTICE"] = "0"
    else:
        os.environ.setdefault("GAMELEON_ME_LATTICE", "1")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from mac_gameleon.paths import GAMELEON_ATTRIBUTE_ROOT, GAMELEON_PACKAGE_ROOT

    for path in (str(GAMELEON_PACKAGE_ROOT), str(GAMELEON_ATTRIBUTE_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def install_attribute_backend(*, no_lattice: bool, log: Optional[Callable[[str], None]] = None) -> None:
    from mac_gameleon.attribute_meta_patches import apply_attribute_meta_patches
    from mac_gameleon.me_lattice_patch import install_me_lattice_acceleration, install_me_normalized_aliases

    install_me_normalized_aliases()
    if not no_lattice:
        import mlx_lattice  # noqa: F401

        install_me_lattice_acceleration(force=True)
        if log:
            log("mlx-lattice ME patch installed")
    elif log:
        log("ME lattice patch disabled (native ME CPU)")

    apply_attribute_meta_patches(quiet=True)
    if log:
        log("Gameleon attribute-meta patches applied")


def prewarm_attribute_extensions(*, log: Optional[Callable[[str], None]] = None) -> None:
    from mac_gameleon.prewarm_torchac import prewarm_torchac

    if log:
        log("Pre-warming torchac...")
    prewarm_torchac(log=log)

    if log:
        log("Pre-warming arithmeticcoding_ext (first run may compile)...")
    t0 = time.perf_counter()
    from gameleon_attribute.models.pcml_model import _load_arithmeticcoding_ext_module

    _load_arithmeticcoding_ext_module()
    if log:
        log(f"arithmeticcoding_ext ready ({time.perf_counter() - t0:.2f}s)")


def load_attribute_adapter(ckpt: str, *, log: Optional[Callable[[str], None]] = None):
    if log:
        log("Loading GameleonAttributeAdapter...")
    from core.attribute_adapter import GameleonAttributeAdapter

    adapter = GameleonAttributeAdapter(
        ckpt=ckpt,
        runtime_precision="fp32",
        debug=False,
    )
    if log:
        log("PCML model loaded")
    return adapter


@contextlib.contextmanager
def gameleon_attribute_workdir() -> Iterator[None]:
    from mac_gameleon.paths import GAMELEON_ATTRIBUTE_ROOT

    prev_cwd = os.getcwd()
    os.chdir(GAMELEON_ATTRIBUTE_ROOT)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
