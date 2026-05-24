#!/usr/bin/env bash
# Install MinkowskiEngine v0.5.4 CPU backend on Apple Silicon.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/minkowskiengine"
PATCHES="$ROOT/patches/minkowskiengine_v0.5.4"
TAG="${MINKOWSKI_TAG:-v0.5.4}"

cd "$ROOT"
if [[ ! -d .venv ]]; then
  echo "Missing .venv. Create one with Python 3.10+ and install torch first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -d "$VENDOR" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/NVIDIA/MinkowskiEngine.git "$VENDOR"
fi

cd "$VENDOR"
if [[ -d .git ]]; then
  git fetch --tags origin 2>/dev/null || true
  git checkout "$TAG"
  git clean -fd >/dev/null 2>&1 || true
fi
rm -rf build dist MinkowskiEngine.egg-info 2>/dev/null || true

cp "$PATCHES/setup.py" setup.py
chmod +x "$PATCHES/apply_source_patches.sh"
"$PATCHES/apply_source_patches.sh" "$VENDOR"

brew list openblas >/dev/null 2>&1 || brew install openblas
brew list libomp >/dev/null 2>&1 || brew install libomp

OPENBLAS_PREFIX="$(brew --prefix openblas)"
export MAX_JOBS="${MAX_JOBS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
pip install "setuptools>=68,<82" ninja

pip uninstall -y MinkowskiEngine 2>/dev/null || true
python setup.py install \
  --cpu_only \
  --blas=openblas \
  --blas_include_dirs="${OPENBLAS_PREFIX}/include" \
  --blas_library_dirs="${OPENBLAS_PREFIX}/lib"

python "$ROOT/scripts/test_minkowski_cpu.py"
echo "MinkowskiEngine CPU install OK (tag=$TAG)."
