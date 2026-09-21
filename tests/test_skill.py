"""Distribution and behavior contracts for the portable agent instructions."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from phasesmith import _skill
from phasesmith.cli import main

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "skills" / "phasesmith-ai-workflows"


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _generator():
    spec = importlib.util.spec_from_file_location("sync_skill", ROOT / "scripts/sync_skill.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_skill_is_complete_and_matches_canonical_source():
    # Release jobs run this against an installed wheel or an sdist-built wheel.
    installed = _tree(_skill.skill_path())
    assert installed == _tree(SOURCE)
    assert "SKILL.md" in installed
    assert "agents/openai.yaml" in installed
    assert {name for name in installed if name.startswith("references/")} == {
        f"references/{name}.md" for name in _skill.REFERENCES
    }
    for relative, data in installed.items():
        if not relative.endswith(".md"):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", data.decode("utf-8")):
            if "://" not in target and not target.startswith("#"):
                linked = (_skill.skill_path() / relative).parent / target.split("#")[0]
                assert linked.is_file(), (relative, target)


def test_skill_discovery_is_independent_of_working_directory(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "phasesmith.cli", "skill", "--path"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    path = Path(completed.stdout.strip())
    assert path.is_absolute()
    assert path == _skill.skill_path()
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["skill", *_skill.REFERENCES, "all"])
def test_skill_prints_requested_text(name, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    arguments = ["skill", "--print"] + ([] if name == "skill" else [name])
    assert main(arguments) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if name == "all":
        expected_paths = ["SKILL.md", *(f"references/{ref}.md" for ref in _skill.REFERENCES)]
        for path in expected_paths:
            assert (_skill.skill_path() / path).read_text(encoding="utf-8") in output.out
    else:
        relative = "SKILL.md" if name == "skill" else f"references/{name}.md"
        assert output.out == (_skill.skill_path() / relative).read_text(encoding="utf-8")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["skill"],
        ["skill", "--path", "--print"],
        ["skill", "--print", "unknown"],
        ["skill", "--print", "../../pyproject.toml"],
        ["skill", "--install"],
    ],
)
def test_invalid_skill_arguments_keep_json_error_contract(arguments, capsys):
    assert main(arguments) == 2
    output = capsys.readouterr()
    assert output.out == ""
    error = json.loads(output.err)
    assert error["code"] == "cli.arguments"


def test_missing_installed_reference_does_not_fall_back_to_checkout(tmp_path, monkeypatch, capsys):
    package = tmp_path / "phasesmith"
    copied = package / "_skills" / _skill.SKILL_NAME
    shutil.copytree(SOURCE, copied)
    (copied / "references" / "workflow.md").unlink()
    monkeypatch.setattr(_skill, "__file__", str(package / "_skill.py"))
    assert main(["skill", "--path"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["code"] == "skill.resources"


def test_generated_skill_is_current():
    assert _generator().sync(ROOT, check=True)


def test_offline_advisor_example_uses_installed_contracts(tmp_path):
    destination = tmp_path / "advisor-cycle"
    subprocess.run(
        [sys.executable, ROOT / "examples/automation/run_advisor_cycle.py", destination],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["first_plan_id"] != summary["next_plan_id"]
    review = json.loads((destination / "review-one.json").read_text(encoding="utf-8"))
    assert review["status"] == "review_required"
    lint = json.loads((destination / "lint-one.json").read_text(encoding="utf-8"))
    assert lint["valid_contract"] is True
    assert lint["provenance"]["kind"] == "software"
    # Preparing the child plan must not accidentally execute a second refinement.
    assert not (destination / "run-two").exists()


def test_generation_detects_drift_without_writing_and_removes_obsolete_outputs(tmp_path):
    source = tmp_path / "skills" / _skill.SKILL_NAME
    shutil.copytree(SOURCE, source)
    generator = _generator()
    assert generator.sync(tmp_path, check=False)
    assert generator.sync(tmp_path, check=True)
    expected = _tree(tmp_path)

    (source / "references" / "workflow.md").write_text("# Changed protocol\n", encoding="utf-8")
    obsolete = tmp_path / "docs" / "agent-skill" / "references" / "obsolete.md"
    obsolete.write_text("obsolete\n", encoding="utf-8")
    before = _tree(tmp_path)
    assert not generator.sync(tmp_path, check=True)
    assert _tree(tmp_path) == before

    assert generator.sync(tmp_path, check=False)
    assert not obsolete.exists()
    assert generator.sync(tmp_path, check=True)
    assert _tree(tmp_path) != expected
    packaged = tmp_path / "python" / "phasesmith" / "_skills" / _skill.SKILL_NAME
    assert _tree(packaged) == _tree(source)


def test_crlf_checkout_respects_skill_lf_attributes(tmp_path):
    # Simulate Windows core.autocrlf=true rather than changing developer Git config.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copy(ROOT / ".gitattributes", checkout / ".gitattributes")
    for base in (
        Path("skills") / _skill.SKILL_NAME,
        Path("python/phasesmith/_skills") / _skill.SKILL_NAME,
        Path("docs/agent-skill"),
    ):
        shutil.copytree(ROOT / base, checkout / base)
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    git = ["git", "-C", str(checkout), "-c", "core.autocrlf=true"]
    subprocess.run([*git, "add", "."], check=True, capture_output=True)
    # Remove only temporary copied files and let Git restore them using its attributes.
    for base in ("skills", "python", "docs"):
        shutil.rmtree(checkout / base)
    subprocess.run([*git, "checkout-index", "--all"], check=True)
    assert _generator().sync(checkout, check=True)
