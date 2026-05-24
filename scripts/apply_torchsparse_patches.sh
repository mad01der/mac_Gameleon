#!/usr/bin/env bash
# Copy TorchSparse Mac CPU patches into the active venv (no rebuild).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCHES="$ROOT/patches/torchsparse_v2.0.0"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Missing .venv. Run ./scripts/setup_env.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

SP="$(python -c "import torchsparse, pathlib; print(pathlib.Path(torchsparse.__file__).parent)")"
cp "$PATCHES/build_kmap.py" "$SP/nn/functional/build_kmap.py"
cp "$PATCHES/downsample.py" "$SP/nn/functional/downsample.py"
echo "TorchSparse runtime patches applied under $SP"
