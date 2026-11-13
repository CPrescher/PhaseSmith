#!/usr/bin/env python3
"""List public PhaseSmith crates that already have a version on crates.io.

The release workflow excludes these crates from ``cargo publish --workspace`` so
an interrupted publication can be resumed by re-running the job. Crate versions
are immutable: a listed version, yanked or not, can never be uploaded again.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_URL = "https://index.crates.io"
USER_AGENT = "phasesmith-release-workflow (https://github.com/CPrescher/PhaseSmith)"

Fetch = Callable[[str], str | None]


def _public_crates() -> set[str]:
    specification = importlib.util.spec_from_file_location(
        "release_version", ROOT / "scripts" / "release_version.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return set(module.PUBLIC_CRATES)


def index_path(name: str) -> str:
    """Return the sparse-index path of crate *name*."""

    lowered = name.lower()
    if len(lowered) <= 2:
        return f"{len(lowered)}/{lowered}"
    if len(lowered) == 3:
        return f"3/{lowered[0]}/{lowered}"
    return f"{lowered[:2]}/{lowered[2:4]}/{lowered}"


def fetch_index(path: str) -> str | None:
    """Fetch one sparse-index file; ``None`` means the crate does not exist."""

    request = urllib.request.Request(
        f"{INDEX_URL}/{path}",
        headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code in (403, 404, 410):
            return None
        raise


def published_versions(name: str, fetch: Fetch = fetch_index) -> set[str]:
    """Return every version of *name* recorded in the registry index."""

    body = fetch(index_path(name))
    if body is None:
        return set()
    return {json.loads(line)["vers"] for line in body.splitlines() if line.strip()}


def already_published(version: str, crates: Iterable[str], fetch: Fetch = fetch_index) -> list[str]:
    """Return the sorted crates whose *version* is already on the registry."""

    return sorted(name for name in crates if version in published_versions(name, fetch))


def main() -> int:
    """Print one already-published crate name per line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Release version to look up")
    args = parser.parse_args()
    try:
        names = already_published(args.version, _public_crates())
    except (OSError, ValueError, KeyError) as error:
        print(f"crates.io index lookup failed: {error}", file=sys.stderr)
        return 1
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
