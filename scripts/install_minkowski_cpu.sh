#!/usr/bin/env bash
# Install MinkowskiEngine v0.5.4 CPU backend on Apple Silicon.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/minkowskiengine"
SETUP_TEMPLATE="$ROOT/mac_gameleon/minkowski_setup.py"
TAG="${MINKOWSKI_TAG:-v0.5.4}"

cd "$ROOT"
if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./scripts/setup_env.sh first." >&2
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

cp "$SETUP_TEMPLATE" setup.py

python3 - <<'PY' "$VENDOR"
import pathlib, sys
vendor = pathlib.Path(sys.argv[1])
utils = vendor / "src/utils.hpp"
text = utils.read_text()
old = "  std::chrono::system_clock::time_point m_start;"
new = "  std::chrono::high_resolution_clock::time_point m_start;"
if old in text:
    utils.write_text(text.replace(old, new, 1))
elif new not in text:
    raise SystemExit(f"unexpected {utils}")

coord = vendor / "src/coordinate_map_cpu.hpp"
text = coord.read_text()
for old, repl in [
    ("std::min((n + 1) * stride, uint64_t(num_tfield))", "std::min<uint64_t>((n + 1) * stride, uint64_t(num_tfield))"),
    ("std::min((n + 1) * stride, uint64_t(size()))", "std::min<uint64_t>((n + 1) * stride, uint64_t(size()))"),
]:
    if old in text:
        text = text.replace(old, repl)
    elif repl not in text:
        raise SystemExit(f"missing pattern in {coord}: {old!r}")
coord.write_text(text)

for path in (vendor / "MinkowskiEngine").rglob("*.py"):
    text = path.read_text()
    orig = text
    text = text.replace(
        "from collections import Sequence, namedtuple",
        "from collections import namedtuple\nfrom collections.abc import Sequence",
    )
    text = text.replace("from collections import Sequence", "from collections.abc import Sequence")
    if text != orig:
        path.write_text(text)
PY

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

python - <<'PY'
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import MinkowskiEngine as ME

coords = torch.randint(0, 4, (100, 3), dtype=torch.int32)
feats = torch.randn(100, 16)
coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
x = ME.SparseTensor(features=feats_batch, coordinates=coords_batch, tensor_stride=1, device="cpu")
conv_cls = getattr(ME, "MinkowskiNormalizedConvolution", ME.MinkowskiConvolution)
y = conv_cls(16, 16, kernel_size=3, dimension=3)(x)
print(f"MinkowskiEngine OK: {ME.__version__} conv {tuple(y.F.shape)}")
PY

echo "MinkowskiEngine install OK (tag=$TAG)."
