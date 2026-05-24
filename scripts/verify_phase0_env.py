#!/usr/bin/env python3
"""Phase 0 checklist: Mac CPU env + Gameleon import + weights/data paths."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mac_gameleon.paths import required_paths  # noqa: E402


def _check(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        raise SystemExit(1)


def main() -> int:
    print("Phase 0 verification")
    print(f"  MAC_GAMELEON_ROOT={os.environ.get('MAC_GAMELEON_ROOT', ROOT)}")
    print(f"  GAMELEON_ROOT={os.environ.get('GAMELEON_ROOT', '(unset)')}")
    print(f"  GAMELEON_DEVICE={os.environ.get('GAMELEON_DEVICE', '(unset)')}")

    import torch

    _check("torch", True, f"{torch.__version__} cuda={torch.cuda.is_available()}")
    _check("cpu device expected", not torch.cuda.is_available() or os.environ.get("GAMELEON_DEVICE") == "cpu")

    for mod in ("mlx", "torchsparse", "MinkowskiEngine", "open3d", "gsplat_mlx"):
        spec = importlib.util.find_spec(mod.replace("-", "_").split(".")[0])
        _check(f"import {mod}", spec is not None)

    import MinkowskiEngine as ME  # noqa: F401

    _check("MinkowskiEngine version", True, getattr(ME, "__version__", "?"))

    import gameleon  # noqa: F401

    pkg_root = Path(gameleon.__file__).resolve().parent
    _check("gameleon package", pkg_root.is_dir(), str(pkg_root))

    exe = ROOT / ".venv" / "bin" / "gameleon-test"
    _check("gameleon-test entrypoint", exe.is_file(), str(exe))

    help_out = subprocess.run(
        [str(exe), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    _check("gameleon-test --help", help_out.returncode == 0)

    for label, path in required_paths().items():
        _check(label, path.exists(), str(path))

    print("\nPhase 0 OK — environment ready for geometry/attribute integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
