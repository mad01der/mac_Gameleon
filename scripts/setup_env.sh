#!/usr/bin/env bash
# One-time / idempotent Mac CPU environment for Gameleon integration (Phase 0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

SKIP_NATIVE="${SKIP_NATIVE:-0}"
SKIP_GAMELEON="${SKIP_GAMELEON:-0}"

echo "==> mac_Gameleon setup (ROOT=$ROOT)"
echo "    PYTHON=$PYTHON  SKIP_NATIVE=$SKIP_NATIVE  SKIP_GAMELEON=$SKIP_GAMELEON"

if command -v brew >/dev/null 2>&1; then
  echo "==> Homebrew deps (openblas, libomp, google-sparsehash)"
  brew list openblas >/dev/null 2>&1 || brew install openblas
  brew list libomp >/dev/null 2>&1 || brew install libomp
  brew list google-sparsehash >/dev/null 2>&1 || brew install google-sparsehash
else
  echo "WARN: Homebrew not found; install openblas, libomp, google-sparsehash manually." >&2
fi

if [[ ! -d "$ROOT/../Gameleon/gameleon" ]]; then
  echo "ERROR: Expected Gameleon repo at $ROOT/../Gameleon" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating .venv"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source "$ROOT/scripts/env_mac_cpu.sh"

echo "==> Python: $(python -V) ($(which python))"

pip install -U pip wheel "setuptools>=68,<82" ninja

echo "==> PyTorch (Mac)"
pip install -U torch torchvision

echo "==> Python deps (requirements-mac-cpu.txt)"
pip install -r "$ROOT/requirements-mac-cpu.txt"

echo "==> gsplat-mlx (editable)"
pip install -e "$ROOT/vendor/gsplat-mlx"

if [[ "$SKIP_GAMELEON" != "1" ]]; then
  echo "==> Gameleon (editable, --no-deps; native ME/TorchSparse installed separately)"
  pip install -e "$ROOT/../Gameleon" --no-deps
  echo "==> Gameleon geometry CPU patches"
  python "$ROOT/patches/gameleon_geometry_cpu/apply_patches.py" "${GAMELEON_ROOT:-$ROOT/../Gameleon}"
fi

if [[ "$SKIP_NATIVE" != "1" ]]; then
  echo "==> TorchSparse CPU"
  "$ROOT/scripts/install_torchsparse_cpu.sh"
  echo "==> MinkowskiEngine CPU"
  "$ROOT/scripts/install_minkowski_cpu.sh"
else
  echo "==> SKIP_NATIVE=1: skipping TorchSparse / MinkowskiEngine rebuild"
fi

echo "==> Phase 0 verification"
python "$ROOT/scripts/verify_phase0_env.py"

echo ""
echo "Setup complete. For new shells:"
echo "  cd $ROOT"
echo "  source scripts/env_mac_cpu.sh"
