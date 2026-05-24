#!/usr/bin/env bash
# Source this before running Gameleon on Mac CPU:
#   source scripts/env_mac_cpu.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

export MAC_GAMELEON_ROOT="$ROOT"
export GAMELEON_ROOT="$(cd "$ROOT/../Gameleon" && pwd)"
export GAMELEON_PACKAGE_ROOT="$GAMELEON_ROOT/gameleon"
export GAMELEON_ATTRIBUTE_ROOT="$GAMELEON_PACKAGE_ROOT/gameleon_attribute"

export GAMELEON_DEVICE="${GAMELEON_DEVICE:-cpu}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_mac_gameleon}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# mac_gameleon imports + Gameleon attribute modules (before pip -e covers all cases).
export PYTHONPATH="${ROOT}:${GAMELEON_PACKAGE_ROOT}:${GAMELEON_ATTRIBUTE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
