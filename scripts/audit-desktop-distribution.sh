#!/bin/sh
set -eu

workspace_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$workspace_root"
app_manifest="$workspace_root/apps/phasesmith-desktop/src-tauri/Cargo.toml"
target_dir="$workspace_root/target/phasesmith-tauri"

dependency_tree=$(cargo tree --manifest-path "$app_manifest" --edges normal,build)
if printf '%s\n' "$dependency_tree" | grep -Eiq 'phasesmith-py|pyo3|numpy|python'; then
  printf '%s\n' "desktop dependency tree contains a forbidden Python dependency" >&2
  exit 1
fi

CARGO_TARGET_DIR="$target_dir" cargo build --manifest-path "$app_manifest" --release
binary="$target_dir/release/phasesmith"
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

bundle_root="$target_dir/release/bundle"
if test -d "$bundle_root"; then
  if find "$bundle_root" -type f \
    \( -iname 'python*' -o -iname 'libpython*' -o -iname '*.whl' -o -iname '_core*.so' \) \
    | grep -q .; then
    printf '%s\n' "desktop bundle contains a forbidden Python artifact" >&2
    exit 1
  fi
fi

printf '%s\n' "desktop distribution audit passed: no Python runtime or sidecar artifacts"
