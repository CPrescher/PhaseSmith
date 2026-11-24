"""Repository hygiene guards for docs, notices, and portable script defaults."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONAL_PATH_RE = re.compile(r"/Users/[A-Za-z0-9._-]+/")


def test_no_personal_home_paths_in_code_scripts_or_active_docs() -> None:
    offenders: list[str] = []
    scan_roots = (
        ("python", ("*.py",)),
        ("tests", ("*.py",)),
        ("scripts", ("*.py",)),
        ("tools", ("*.py",)),
        ("benchmarks", ("*.py", "*.md")),
        ("docs", ("*.md",)),
    )
    for folder, patterns in scan_roots:
        base = ROOT / folder
        for pattern in patterns:
            for path in sorted(base.rglob(pattern)):
                if "archive" in path.parts:
                    continue
                if path == Path(__file__).resolve():
                    continue
                text = path.read_text(encoding="utf-8")
                if PERSONAL_PATH_RE.search(text):
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"personal home paths found in: {', '.join(offenders)}"


def test_mkdocs_nav_has_no_duplicate_or_missing_pages() -> None:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    in_nav = False
    seen: list[str] = []
    for line in text.splitlines():
        if line == "nav:":
            in_nav = True
            continue
        if in_nav and line and not line.startswith(" "):
            break
        if not in_nav:
            continue
        match = re.search(r":\s+([A-Za-z0-9_./-]+\.md)\s*$", line)
        if match is not None:
            seen.append(match.group(1))
    duplicates = sorted({item for item in seen if seen.count(item) > 1})
    assert not duplicates, f"duplicate mkdocs.yml nav entries: {duplicates}"
    missing = [item for item in seen if not (ROOT / "docs" / item).is_file()]
    assert not missing, f"missing mkdocs.yml nav pages: {missing}"


def test_third_party_notices_match_locked_persistence_versions_and_live_paths() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "phasesmith.experimental." not in notices

    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    locked_versions = {
        pkg["name"]: pkg["version"]
        for pkg in lock.get("package", [])
        if pkg.get("name") in {"zip", "flate2", "sha2", "moyo", "rustfft"}
    }
    for crate_name in ("zip", "flate2", "sha2"):
        expected = f"`{crate_name}` {locked_versions[crate_name]}"
        assert expected in notices, f"THIRD_PARTY_NOTICES.md missing {expected!r}"
