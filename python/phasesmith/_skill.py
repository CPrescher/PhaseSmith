"""Read the version-matched agent instructions shipped with the distribution."""

from __future__ import annotations

from pathlib import Path

from .automation import AutomationError

SKILL_NAME = "phasesmith-ai-workflows"
REFERENCES = ("experiment", "workflow", "interpretation")


def skill_path() -> Path:
    """Return the installed resource directory, never a source-checkout fallback."""
    root = Path(__file__).resolve().parent / "_skills" / SKILL_NAME
    required = ("SKILL.md", *(f"references/{name}.md" for name in REFERENCES))
    if not all((root / name).is_file() for name in required):
        raise AutomationError(
            "skill.resources",
            "the installed PhaseSmith skill is incomplete; reinstall this distribution",
        )
    return root


def skill_text(reference: str) -> str:
    """Read the entrypoint, a named reference, or the complete instruction tree."""
    root = skill_path()
    names = {"skill": "SKILL.md", **{name: f"references/{name}.md" for name in REFERENCES}}
    if reference == "all":
        return "\n".join(
            f"<!-- {filename} -->\n{(root / filename).read_text(encoding='utf-8')}"
            for filename in names.values()
        )
    return (root / names[reference]).read_text(encoding="utf-8")
