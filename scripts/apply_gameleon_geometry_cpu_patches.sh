#!/usr/bin/env bash
# Apply Phase 1 patches to ../Gameleon (geometry CPU device routing).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GAMELEON_ROOT="${GAMELEON_ROOT:-$ROOT/../Gameleon}"

python3 "$ROOT/patches/gameleon_geometry_cpu/apply_patches.py" "$GAMELEON_ROOT"
echo "Gameleon geometry CPU patches applied under $GAMELEON_ROOT"
