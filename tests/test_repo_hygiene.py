"""Repository hygiene guards for docs, notices, crash logs, and portable paths."""

from __future__ import annotations

import re
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONAL_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")


def test_no_personal_home_paths_in_code_scripts_or_active_docs() -> None:
    offenders: list[str] = []
    scan_roots = (
        ("python", ("*.py",)),
        ("crates", ("*.rs", "*.toml")),
        ("tests", ("*.py",)),
        ("scripts", ("*.py", "*.sh")),
        ("tools", ("*.py", "*.sh")),
        ("oracle/scripts", ("*.py",)),
        (".github", ("*.yml", "*.yaml")),
        ("benchmarks", ("*.py", "*.md")),
        ("docs", ("*.md",)),
    )
    for folder, patterns in scan_roots:
        base = ROOT / folder
        for pattern in patterns:
            for path in sorted(base.rglob(pattern)):
                if "archive" in path.parts or "__pycache__" in path.parts:
                    continue
                if path == Path(__file__).resolve():
                    continue
                text = path.read_text(encoding="utf-8")
                if PERSONAL_PATH_RE.search(text):
                    offenders.append(str(path.relative_to(ROOT)))
    for root_file in (
        "README.md",
        "PROJECT_BRIEF.md",
        "CHANGELOG.md",
        "Cargo.toml",
        "pyproject.toml",
        "mkdocs.yml",
    ):
        path = ROOT / root_file
        if path.is_file() and PERSONAL_PATH_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(root_file)
    assert not offenders, f"personal home paths found in: {', '.join(offenders)}"


def test_no_committed_python_tracebacks_in_validation_results() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "validation/results").rglob("*.txt"))
        if "Traceback (most recent call last):" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"committed traceback logs found: {', '.join(offenders)}"


def test_mkdocs_nav_has_no_duplicate_missing_or_excluded_pages() -> None:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav_pages: list[str] = []
    excluded_pages: set[str] = set()
    section: str | None = None
    for line in text.splitlines():
        if line in ("nav:", "exclude_docs:", "not_in_nav:"):
            section = line[:-1]
            continue
        if line and not line.startswith(" ") and not line.startswith("#"):
            section = None
            continue
        if section == "nav":
            match = re.search(r"(?::|^\s*-)\s+[\"']?([A-Za-z0-9_./-]+\.md)[\"']?\s*$", line)
            if match is not None:
                nav_pages.append(match.group(1))
        elif section == "exclude_docs":
            entry = line.strip()
            if entry and not entry.startswith("#"):
                excluded_pages.add(entry)
    duplicates = sorted({item for item in nav_pages if nav_pages.count(item) > 1})
    assert not duplicates, f"duplicate mkdocs.yml nav entries: {duplicates}"
    missing = [item for item in nav_pages if not (ROOT / "docs" / item).is_file()]
    assert not missing, f"missing mkdocs.yml nav pages: {missing}"
    overlap = sorted(excluded_pages.intersection(nav_pages))
    assert not overlap, f"mkdocs.yml nav entries also excluded: {overlap}"


def test_third_party_notices_match_locked_versions_and_live_paths() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "phasesmith.experimental." not in notices

    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    versions_by_crate: dict[str, set[str]] = defaultdict(set)
    for pkg in lock.get("package", []):
        name = pkg.get("name")
        if name in {"zip", "flate2", "sha2", "moyo", "rustfft"}:
            versions_by_crate[name].add(pkg["version"])
    for crate_name in ("zip", "flate2", "sha2", "moyo", "rustfft"):
        versions = versions_by_crate[crate_name]
        assert len(versions) == 1, (
            f"expected single locked version for {crate_name}, got {versions}"
        )
    for crate_name in ("zip", "flate2", "sha2"):
        ver = next(iter(versions_by_crate[crate_name]))
        expected = f"`{crate_name}` {ver}"
        assert expected in notices, f"THIRD_PARTY_NOTICES.md missing {expected!r}"
    assert f"Moyo {next(iter(versions_by_crate['moyo']))}" in notices
    assert f"RustFFT {next(iter(versions_by_crate['rustfft']))}" in notices
