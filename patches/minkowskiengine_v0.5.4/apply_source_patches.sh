#!/usr/bin/env bash
# Apply Mac / modern-clang source fixes on top of MinkowskiEngine v0.5.4.
set -euo pipefail

VENDOR="${1:?vendor dir required}"

UTILS="$VENDOR/src/utils.hpp"
COORD="$VENDOR/src/coordinate_map_cpu.hpp"

python3 - <<'PY' "$UTILS"
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
old = "  std::chrono::system_clock::time_point m_start;"
new = "  std::chrono::high_resolution_clock::time_point m_start;"
if old not in text:
    if new in text:
        raise SystemExit(0)
    raise SystemExit(f"unexpected {path}")
path.write_text(text.replace(old, new, 1))
PY

python3 - <<'PY' "$COORD"
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
replacements = [
    (
        "std::min((n + 1) * stride, uint64_t(num_tfield))",
        "std::min<uint64_t>((n + 1) * stride, uint64_t(num_tfield))",
    ),
    (
        "std::min((n + 1) * stride, uint64_t(size()))",
        "std::min<uint64_t>((n + 1) * stride, uint64_t(size()))",
    ),
]
for old, new in replacements:
    count = text.count(old)
    if count == 0 and new in text:
        continue
    if count == 0:
        raise SystemExit(f"missing pattern in {path}: {old!r}")
    text = text.replace(old, new)
path.write_text(text)
PY

python3 - <<'PY' "$VENDOR/MinkowskiEngine"
import pathlib, sys

root = pathlib.Path(sys.argv[1])
for path in root.rglob("*.py"):
    text = path.read_text()
    orig = text
    text = text.replace(
        "from collections import Sequence, namedtuple",
        "from collections import namedtuple\nfrom collections.abc import Sequence",
    )
    text = text.replace(
        "from collections import Sequence",
        "from collections.abc import Sequence",
    )
    if text != orig:
        path.write_text(text)
        print(f"patched {path}")
PY
