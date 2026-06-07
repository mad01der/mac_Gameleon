#!/usr/bin/env bash
# One-time / idempotent Mac environment for Gameleon integration.
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

if command -v brew >/dev/null 2>&1; then
  brew list openblas >/dev/null 2>&1 || brew install openblas
  brew list libomp >/dev/null 2>&1 || brew install libomp
  brew list google-sparsehash >/dev/null 2>&1 || brew install google-sparsehash
fi

if [[ ! -d "$ROOT/../Gameleon/gameleon" ]]; then
  echo "ERROR: Expected Gameleon repo at $ROOT/../Gameleon" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source "$ROOT/scripts/env_mac_cpu.sh"

pip install -U pip wheel "setuptools>=68,<82" ninja
pip install -U torch torchvision
pip install -r "$ROOT/requirements-mac-cpu.txt"
pip uninstall -y torchsparse 2>/dev/null || true
pip install -e "$ROOT/vendor/gsplat-mlx"

if [[ "$SKIP_GAMELEON" != "1" ]]; then
  pip install -e "$ROOT/../Gameleon" --no-deps
  python -c "from mac_gameleon.geometry_meta_patches import apply_geometry_meta_patches; apply_geometry_meta_patches(quiet=True)"
fi

if [[ "$SKIP_NATIVE" != "1" ]]; then
  "$ROOT/scripts/install_minkowski_cpu.sh"
fi

python -c "from mac_gameleon.prewarm_torchac import prewarm_torchac; prewarm_torchac()"
python -c "import mlx_lattice, gameleon; from mac_gameleon.paths import required_paths; assert all(p.exists() for p in required_paths().values())"

echo ""
echo "Setup complete:"
echo "  source scripts/env_mac_cpu.sh"
echo "  python scripts/geometry_meta.py"
