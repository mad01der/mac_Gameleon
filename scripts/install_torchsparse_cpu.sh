#!/usr/bin/env bash
# Install vendored TorchSparse v2.0.0 with CPU backend on Apple Silicon.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/torchsparse"
PATCHES="$ROOT/patches/torchsparse_v2.0.0"
TAG="${TORCHSPARSE_TAG:-v2.0.0}"

cd "$ROOT"
if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./scripts/setup_env.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -d "$VENDOR/.git" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/mit-han-lab/torchsparse.git "$VENDOR"
fi

cd "$VENDOR"
git fetch --tags origin 2>/dev/null || true
git checkout "$TAG"
git clean -fd >/dev/null 2>&1 || true
rm -rf build dist torchsparse.egg-info 2>/dev/null || true
find torchsparse -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

cp "$PATCHES/setup.py" setup.py
cp "$PATCHES/build_kmap.py" torchsparse/nn/functional/build_kmap.py
cp "$PATCHES/downsample.py" torchsparse/nn/functional/downsample.py

brew list google-sparsehash >/dev/null 2>&1 || brew install google-sparsehash
brew list libomp >/dev/null 2>&1 || brew install libomp

export CPLUS_INCLUDE_PATH="$(brew --prefix google-sparsehash)/include:${CPLUS_INCLUDE_PATH:-}"
pip install "setuptools>=68,<82" ninja
pip uninstall -y torchsparse 2>/dev/null || true
pip install --no-build-isolation -v .

python "$ROOT/scripts/test_torchsparse_cpu.py"
echo "TorchSparse CPU install OK (tag=$TAG)."
