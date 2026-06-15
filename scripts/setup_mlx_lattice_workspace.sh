#!/usr/bin/env bash
# Clone mlx-lattice, install mlx_gameleon as a uv workspace member, and build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLX_LATTICE_DIR="$ROOT/vendor/mlx-lattice"
MLX_GAMELEON_SRC="$ROOT/vendor/mlx_gameleon"
MLX_GAMELEON_DST="$MLX_LATTICE_DIR/mlx_gameleon"
MLX_LATTICE_REPO="${MLX_LATTICE_REPO:-https://github.com/utakotoba/mlx-lattice.git}"
MLX_LATTICE_BRANCH="${MLX_LATTICE_BRANCH:-main}"

echo "==> mlx-lattice workspace setup (ROOT=$ROOT)"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: mlx-lattice requires Apple Silicon (arm64)." >&2
  exit 1
fi

if ! xcrun -sdk macosx -find metal >/dev/null 2>&1; then
  echo "ERROR: Metal compiler not found (xcrun metal)." >&2
  echo "Install full Xcode from the App Store (Command Line Tools alone are not enough)." >&2
  echo "Then run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
  echo "Open Xcode once to accept the license and finish toolchain setup." >&2
  exit 1
fi

if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  echo "ERROR: Metal Toolchain is not installed (Xcode 16+ separates this component)." >&2
  echo "Run one of:" >&2
  echo "  xcodebuild -downloadComponent MetalToolchain" >&2
  echo "  Xcode → Settings → Components → download Metal Toolchain" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -d "$MLX_LATTICE_DIR/.git" ]]; then
  echo "==> Cloning mlx-lattice from $MLX_LATTICE_REPO ..."
  git clone --depth 1 --branch "$MLX_LATTICE_BRANCH" "$MLX_LATTICE_REPO" "$MLX_LATTICE_DIR"
elif [[ "${MLX_LATTICE_SKIP_UPDATE:-0}" == "1" ]]; then
  echo "==> mlx-lattice already cloned; skipping remote update (MLX_LATTICE_SKIP_UPDATE=1)."
else
  echo "==> mlx-lattice already cloned; trying to fetch latest (continues on network failure)..."
  if git -C "$MLX_LATTICE_DIR" fetch --depth 1 origin "$MLX_LATTICE_BRANCH"; then
    git -C "$MLX_LATTICE_DIR" checkout "$MLX_LATTICE_BRANCH"
    git -C "$MLX_LATTICE_DIR" pull --ff-only origin "$MLX_LATTICE_BRANCH" || true
  else
    echo "WARN: could not reach GitHub; using existing mlx-lattice at $MLX_LATTICE_DIR" >&2
  fi
fi

echo "==> Syncing mlx_gameleon into mlx-lattice workspace..."
rm -rf "$MLX_GAMELEON_DST"
mkdir -p "$MLX_GAMELEON_DST"
rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$MLX_GAMELEON_SRC/" "$MLX_GAMELEON_DST/"

python3 - <<'PY' "$MLX_LATTICE_DIR/pyproject.toml"
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
member = "mlx_gameleon"

match = re.search(r"members\s*=\s*\[(.*?)\]", text, flags=re.S)
if match is None:
    insert = "\n[tool.uv.workspace]\nmembers = [\"benchmarks\", \"mlx_gameleon\"]\n"
    if "[tool.uv.workspace]" not in text:
        text += insert
else:
    body = match.group(1)
    if member not in body:
        items = [item.strip().strip('"').strip("'") for item in body.split(",") if item.strip()]
        items.append(member)
        new_body = ", ".join(f'"{item}"' for item in items)
        text = text[: match.start(1)] + new_body + text[match.end(1) :]

path.write_text(text)
print(f"Patched workspace members in {path}")
PY

cd "$MLX_LATTICE_DIR"
echo "==> uv sync (build mlx-lattice + mlx-gameleon)..."
rm -rf build
uv sync --all-packages --reinstall-package mlx-lattice

echo ""
echo "Setup complete."
echo "  cd $MLX_LATTICE_DIR"
echo "  uv run mlx-gameleon-bench --points 100000 --warmup 1 --iters 3"
echo ""
echo "Verify mlx-lattice entropy ops (new builds):"
echo "  uv run python -c \"from mlx_lattice.ops import normalized_cdf, range_decode, range_encode; print('entropy ops OK')\""
