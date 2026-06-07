"""Pre-compile torchac C++ backend so geometry-meta imports do not appear to hang."""

from __future__ import annotations

import os
import time
from pathlib import Path


def _extension_cache_dir() -> Path:
    root = Path(os.environ.get("MAC_GAMELEON_ROOT", Path(__file__).resolve().parents[1]))
    return Path(os.environ.get("TORCH_EXTENSIONS_DIR", root / ".cache" / "torch_extensions"))


def _stale_lock(lock_path: Path, *, max_age_s: float = 600.0) -> bool:
    if not lock_path.is_file():
        return False
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return True
    return age > max_age_s


def prewarm_torchac(*, log=print) -> None:
    """Import torchac once; compile backend if needed. Safe to call repeatedly."""
    cache_dir = _extension_cache_dir()
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    backend_dir = cache_dir / "torchac_backend"
    lock_path = backend_dir / "lock"
    if _stale_lock(lock_path):
        log(f"Removing stale torchac build lock: {lock_path}")
        lock_path.unlink(missing_ok=True)

    so_candidates = list(backend_dir.glob("torchac_backend*.so"))
    if so_candidates:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import torchac  # noqa: F401

        return

    log(
        "Compiling torchac C++ backend (first run only, ~5–30s). "
        "Do not start multiple geometry_meta.py processes in parallel."
    )
    t0 = time.perf_counter()
    import torchac  # noqa: F401

    log(f"torchac ready ({time.perf_counter() - t0:.1f}s).")
