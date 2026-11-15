from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/published_crates.py"


def _module() -> object:
    specification = importlib.util.spec_from_file_location("published_crates", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _index(*versions: str, yanked: str | None = None) -> str:
    return "\n".join(
        json.dumps({"name": "x", "vers": version, "yanked": version == yanked})
        for version in versions
    )


def test_index_paths_follow_the_sparse_registry_layout() -> None:
    module = _module()
    assert module.index_path("a") == "1/a"
    assert module.index_path("ab") == "2/ab"
    assert module.index_path("abc") == "3/a/abc"
    assert module.index_path("PhaseSmith-Core") == "ph/as/phasesmith-core"


def test_only_crates_with_the_release_version_are_reported() -> None:
    module = _module()
    index = {
        "ph/as/phasesmith": _index("0.8.1"),
        "ph/as/phasesmith-core": _index("0.8.1", "0.8.2"),
        "ph/as/phasesmith-hdf5": _index("0.8.2", yanked="0.8.2"),
    }
    published = module.already_published(
        "0.8.2",
        ["phasesmith", "phasesmith-core", "phasesmith-hdf5", "phasesmith-new"],
        index.get,
    )
    # A yanked version still occupies its number; a missing crate is unpublished.
    assert published == ["phasesmith-core", "phasesmith-hdf5"]


def test_public_crates_come_from_the_release_version_script() -> None:
    module = _module()
    crates = module._public_crates()
    assert "phasesmith" in crates
    assert "phasesmith-py" not in crates
    assert "phasesmith-validation" not in crates
