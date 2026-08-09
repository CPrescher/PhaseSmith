#!/bin/sh
set -eu

workspace_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$workspace_root"

dependency_tree=$(cargo tree -p phasesmith-tauri --edges normal,build)
if printf '%s\n' "$dependency_tree" | grep -Eiq 'phasesmith-py|pyo3|numpy|python'; then
  printf '%s\n' "desktop dependency tree contains a forbidden Python dependency" >&2
  exit 1
fi

cargo build -p phasesmith-tauri --release
binary="$workspace_root/target/release/phasesmith"
test -x "$binary"

if command -v otool >/dev/null 2>&1; then
  if otool -L "$binary" | grep -Eiq 'libpython|python\.framework'; then
    printf '%s\n' "desktop binary links a forbidden Python runtime" >&2
    exit 1
  fi
elif command -v ldd >/dev/null 2>&1; then
  if ldd "$binary" | grep -Eiq 'libpython'; then
    printf '%s\n' "desktop binary links a forbidden Python runtime" >&2
    exit 1
  fi
fi

bundle_root="$workspace_root/target/release/bundle"
if test -d "$bundle_root"; then
  if find "$bundle_root" -type f \
    \( -iname 'python*' -o -iname 'libpython*' -o -iname '*.whl' -o -iname '_core*.so' \) \
    | grep -q .; then
    printf '%s\n' "desktop bundle contains a forbidden Python artifact" >&2
    exit 1
  fi
fi

printf '%s\n' "desktop distribution audit passed: no Python runtime or sidecar artifacts"
