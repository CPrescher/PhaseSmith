"""Constrained task API for human- or AI-orchestrated PhaseSmith workflows.

This module deliberately accepts only persisted, already typed Rietveld
projects.  It exposes inspection, planning, proposal validation, approval, and
execution boundaries without allowing an AI-generated record to invent new
physical models or parameter authorization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from .io.cif import read_cif
from .io.powder import PowderFormat, read_powder_data
from .project import RietveldProject
from .radiation import BraggBrentanoGeometry, ComponentRadiation, DebyeScherrerGeometry
from .refinement.core import AffineConstraint, FixedConstraint, LinearConstraint
from .refinement.readiness import RietveldReadinessReport
from .refinement.rietveld import RietveldParameterSelection, RietveldResult
from .refinement.runtime import RefinementLimits, RefinementLogger
from .refinement.workflow import (
    RietveldRecipe,
    RietveldStage,
    RietveldWorkflowResult,
    intelligent_rietveld_recipe,
    validate_rietveld_recipe,
)
from .reporting import rietveld_result_record

WORKFLOW_SPEC_SCHEMA = "phasesmith.workflow-spec.v1"
WORKFLOW_PLAN_SCHEMA = "phasesmith.workflow-plan.v1"
RECIPE_PROPOSAL_SCHEMA = "phasesmith.recipe-proposal.v1"
WORKFLOW_RESULT_SCHEMA = "phasesmith.workflow-result.v1"
RESUME_RESULT_SCHEMA = "phasesmith.resume-result.v1"
AUTOMATION_ERROR_SCHEMA = "phasesmith.automation-error.v1"
ADVISOR_CONTEXT_SCHEMA = "phasesmith.advisor-context.v1"
RECIPE_LINT_SCHEMA = "phasesmith.recipe-lint.v1"
WORKFLOW_REVIEW_SCHEMA = "phasesmith.workflow-review.v1"

__all__ = [
    "ADVISOR_CONTEXT_SCHEMA",
    "AUTOMATION_ERROR_SCHEMA",
    "RECIPE_LINT_SCHEMA",
    "RECIPE_PROPOSAL_SCHEMA",
    "RESUME_RESULT_SCHEMA",
    "WORKFLOW_PLAN_SCHEMA",
    "WORKFLOW_RESULT_SCHEMA",
    "WORKFLOW_REVIEW_SCHEMA",
    "WORKFLOW_SPEC_SCHEMA",
    "AdvisorContext",
    "AutomationError",
    "RecipeProposal",
    "ResumeRunResult",
    "WorkflowLineage",
    "WorkflowOutputs",
    "WorkflowPlan",
    "WorkflowRunResult",
    "WorkflowSpec",
    "inspect_cif_file",
    "inspect_powder_file",
    "lint_recipe_proposal",
    "lint_recipe_proposal_file",
    "load_recipe_proposal",
    "load_workflow_spec",
    "parse_recipe_proposal",
    "plan_workflow",
    "recipe_proposal_schema",
    "replan_workflow",
    "resume_workflow",
    "review_workflow_output",
    "run_workflow",
    "workflow_spec_schema",
    "write_workflow_plan",
]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_PROJECT_FILES = 256
_MAX_PROJECT_BYTES = 4 * 1024 * 1024 * 1024
_MAX_RECIPE_STAGES = 12

_GUIDANCE = (
    "Read every readiness diagnostic before proposing stages; do not conceal warnings.",
    "Use only parameter families authorized by the plan and never infer missing physics.",
    "Keep stages cumulative so an accepted physical state is never narrowed silently.",
    "Establish scale and refinable background before adding strongly correlated terms.",
    "Align lattice and explicit position nuisances before releasing profile widths.",
    "Delay occupancy until scale is stable and review their correlation diagnostics.",
    "Do not refine wavelength with lattice, or zero shift with displacement, without a "
    "documented independent constraint.",
    "Release sample/profile terms only after position and intensity models are credible.",
    "Treat convergence, rank, correlations, bounds, residual shape, and provenance as "
    "joint acceptance evidence; a lower Rwp alone is insufficient.",
)


class AutomationError(RuntimeError):
    """Stable structured failure at the automation boundary."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        if not _IDENTIFIER.fullmatch(code):
            raise ValueError("automation error code must be a stable identifier")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("automation error message must be non-empty")
        super().__init__(message)
        self.code = code
        self.details = dict(details)
        json.dumps(self.to_record(), allow_nan=False)

    def to_record(self) -> dict[str, object]:
        return {
            "schema": AUTOMATION_ERROR_SCHEMA,
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class WorkflowOutputs:
    """Caller-selected finite output products."""

    write_csv: bool = True
    save_project: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.write_csv, bool) or not isinstance(self.save_project, bool):
            raise TypeError("workflow output selections must be boolean")

    def to_record(self) -> dict[str, bool]:
        return {"write_csv": self.write_csv, "save_project": self.save_project}


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Version-1 task specification for one persisted Rietveld project."""

    workflow_id: str
    project_path: Path
    output_directory: Path
    limits: RefinementLimits
    outputs: WorkflowOutputs = WorkflowOutputs()

    def __post_init__(self) -> None:
        if not isinstance(self.workflow_id, str) or _IDENTIFIER.fullmatch(self.workflow_id) is None:
            raise ValueError("workflow_id must be a stable 1-100 character identifier")
        project = Path(self.project_path).resolve()
        output = Path(self.output_directory).resolve()
        if not isinstance(self.limits, RefinementLimits):
            raise TypeError("limits must be RefinementLimits")
        if not isinstance(self.outputs, WorkflowOutputs):
            raise TypeError("outputs must be WorkflowOutputs")
        if output == project or output.is_relative_to(project):
            raise ValueError("output_directory must not be the project or lie inside it")
        object.__setattr__(self, "project_path", project)
        object.__setattr__(self, "output_directory", output)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        base_directory: str | Path | None = None,
    ) -> WorkflowSpec:
        """Parse a strict finite record and resolve paths relative to its source."""

        values = _mapping(record, "workflow specification")
        _exact_keys(
            values,
            {
                "schema",
                "workflow_id",
                "project_path",
                "output_directory",
                "limits",
                "outputs",
            },
            "workflow specification",
        )
        if values["schema"] != WORKFLOW_SPEC_SCHEMA:
            raise AutomationError("spec.schema", "unsupported workflow specification schema")
        base = Path.cwd() if base_directory is None else Path(base_directory)
        project_path = _resolved_path(values["project_path"], base, "project_path")
        output_directory = _resolved_path(values["output_directory"], base, "output_directory")
        limits = _limits_from_record(values["limits"])
        outputs_record = _mapping(values["outputs"], "outputs")
        _exact_keys(outputs_record, {"write_csv", "save_project"}, "outputs")
        return cls(
            _trimmed(values["workflow_id"], "workflow_id"),
            project_path,
            output_directory,
            limits,
            WorkflowOutputs(
                _boolean(outputs_record["write_csv"], "outputs.write_csv"),
                _boolean(outputs_record["save_project"], "outputs.save_project"),
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema": WORKFLOW_SPEC_SCHEMA,
            "workflow_id": self.workflow_id,
            "project_path": str(self.project_path),
            "output_directory": str(self.output_directory),
            "limits": _limits_record(self.limits),
            "outputs": self.outputs.to_record(),
        }


@dataclass(frozen=True, slots=True)
class AdvisorContext:
    """Immutable sanitized scientific facts intended for an external advisor."""

    _record: dict[str, object]

    def __post_init__(self) -> None:
        record = deepcopy(_mapping(self._record, "advisor context"))
        if record.get("schema") != ADVISOR_CONTEXT_SCHEMA:
            raise ValueError("advisor context must use the supported schema")
        encoded = _canonical_json(record)
        object.__setattr__(self, "_record", json.loads(encoded))

    def to_record(self) -> dict[str, object]:
        return deepcopy(self._record)


@dataclass(frozen=True, slots=True)
class WorkflowLineage:
    """Digest-bound parentage for an explicitly replanned workflow."""

    parent_plan_id: str
    source_result_sha256: str
    source_review_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("parent_plan_id", self.parent_plan_id),
            ("source_result_sha256", self.source_result_sha256),
            ("source_review_sha256", self.source_review_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_record(self) -> dict[str, str]:
        return {
            "parent_plan_id": self.parent_plan_id,
            "source_result_sha256": self.source_result_sha256,
            "source_review_sha256": self.source_review_sha256,
        }


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    """Immutable non-numerical review bound to exact project bytes."""

    spec: WorkflowSpec
    project_fingerprint: str
    readiness: RietveldReadinessReport
    default_recipe: RietveldRecipe
    authorized_selection: RietveldParameterSelection
    parameter_labels: tuple[str, ...]
    advisor_context: AdvisorContext
    can_resume: bool
    plan_id: str
    lineage: WorkflowLineage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, WorkflowSpec):
            raise TypeError("spec must be WorkflowSpec")
        if not re.fullmatch(r"[0-9a-f]{64}", self.project_fingerprint):
            raise ValueError("project_fingerprint must be a lowercase SHA-256 digest")
        if not isinstance(self.readiness, RietveldReadinessReport):
            raise TypeError("readiness must be RietveldReadinessReport")
        if not isinstance(self.default_recipe, RietveldRecipe):
            raise TypeError("default_recipe must be RietveldRecipe")
        if not isinstance(self.authorized_selection, RietveldParameterSelection):
            raise TypeError("authorized_selection must be RietveldParameterSelection")
        labels = tuple(self.parameter_labels)
        if any(not isinstance(label, str) or not label for label in labels):
            raise TypeError("parameter_labels must contain non-empty strings")
        object.__setattr__(self, "parameter_labels", labels)
        if not isinstance(self.advisor_context, AdvisorContext):
            raise TypeError("advisor_context must be AdvisorContext")
        if not isinstance(self.can_resume, bool):
            raise TypeError("can_resume must be a bool")
        if not re.fullmatch(r"[0-9a-f]{64}", self.plan_id):
            raise ValueError("plan_id must be a lowercase SHA-256 digest")
        if self.lineage is not None and not isinstance(self.lineage, WorkflowLineage):
            raise TypeError("lineage must be WorkflowLineage or None")

    @property
    def blocked(self) -> bool:
        return self.readiness.has_errors

    def to_record(self) -> dict[str, object]:
        return _plan_record(
            self.spec,
            self.project_fingerprint,
            self.readiness,
            self.default_recipe,
            self.authorized_selection,
            self.parameter_labels,
            self.advisor_context,
            self.can_resume,
            self.plan_id,
            self.lineage,
        )


@dataclass(frozen=True, slots=True)
class RecipeProposal:
    """An untrusted external staged proposal after PhaseSmith validation."""

    plan_id: str
    proposal_id: str
    generated_by: str
    assumptions: tuple[str, ...]
    recipe: RietveldRecipe

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.plan_id):
            raise ValueError("plan_id must be a lowercase SHA-256 digest")
        if _IDENTIFIER.fullmatch(self.proposal_id) is None:
            raise ValueError("proposal_id must be a stable identifier")
        if (
            not isinstance(self.generated_by, str)
            or not self.generated_by
            or self.generated_by != self.generated_by.strip()
        ):
            raise ValueError("generated_by must be a non-empty trimmed string")
        assumptions = tuple(self.assumptions)
        if any(
            not isinstance(item, str) or not item or item != item.strip() for item in assumptions
        ):
            raise TypeError("assumptions must contain non-empty trimmed strings")
        object.__setattr__(self, "assumptions", assumptions)
        if not isinstance(self.recipe, RietveldRecipe):
            raise TypeError("recipe must be RietveldRecipe")

    def to_record(self) -> dict[str, object]:
        return {
            "schema": RECIPE_PROPOSAL_SCHEMA,
            "plan_id": self.plan_id,
            "proposal_id": self.proposal_id,
            "generated_by": self.generated_by,
            "assumptions": list(self.assumptions),
            "recipe": {
                "name": self.recipe.name,
                "stages": [
                    {
                        "name": stage.name,
                        "selection": _selection_record(stage.selection),
                        "rationale": list(stage.rationale),
                    }
                    for stage in self.recipe.stages
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class WorkflowRunResult:
    """Auditable automation outcome and owned output paths."""

    plan: WorkflowPlan
    recipe_source: Literal["deterministic", "external_proposal"]
    recipe: RietveldRecipe
    workflow: RietveldWorkflowResult
    output_directory: Path
    result_json: Path
    pattern_csv: Path | None
    saved_project: Path | None

    def to_record(self) -> dict[str, object]:
        return {
            "schema": WORKFLOW_RESULT_SCHEMA,
            "plan_id": self.plan.plan_id,
            "workflow_id": self.plan.spec.workflow_id,
            "recipe_source": self.recipe_source,
            "recipe": self.recipe.to_record(),
            "completed": self.workflow.completed,
            "workflow": self.workflow.to_record(),
            "outputs": {
                "directory": str(self.output_directory),
                "result_json": str(self.result_json),
                "pattern_csv": None if self.pattern_csv is None else str(self.pattern_csv),
                "project": None if self.saved_project is None else str(self.saved_project),
            },
        }


@dataclass(frozen=True, slots=True)
class ResumeRunResult:
    """Auditable continuation outcome for an existing project checkpoint."""

    plan: WorkflowPlan
    result: RietveldResult
    output_directory: Path
    result_json: Path
    pattern_csv: Path | None
    saved_project: Path | None

    def to_record(self) -> dict[str, object]:
        return {
            "schema": RESUME_RESULT_SCHEMA,
            "plan_id": self.plan.plan_id,
            "workflow_id": self.plan.spec.workflow_id,
            "action": "resume",
            "termination": self.result.termination_reason.value,
            "result": rietveld_result_record(self.result),
            "outputs": {
                "directory": str(self.output_directory),
                "result_json": str(self.result_json),
                "pattern_csv": None if self.pattern_csv is None else str(self.pattern_csv),
                "project": None if self.saved_project is None else str(self.saved_project),
            },
        }


def load_workflow_spec(path: str | Path) -> WorkflowSpec:
    """Load one size-limited duplicate-key-free workflow specification."""

    source = Path(path).resolve()
    record = _read_json(source)
    return WorkflowSpec.from_record(record, base_directory=source.parent)


def plan_workflow(spec: WorkflowSpec) -> WorkflowPlan:
    """Load and inspect a project without evaluating or refining it."""

    return _build_workflow_plan(spec, lineage=None)


def _build_workflow_plan(
    spec: WorkflowSpec,
    *,
    lineage: WorkflowLineage | None,
) -> WorkflowPlan:
    if not isinstance(spec, WorkflowSpec):
        raise TypeError("spec must be WorkflowSpec")
    fingerprint = _project_fingerprint(spec.project_path)
    project = _load_project(spec.project_path)
    verified_fingerprint = _project_fingerprint(spec.project_path)
    if verified_fingerprint != fingerprint:
        raise AutomationError(
            "project.changed_during_plan",
            "project bytes changed while the read-only plan was being created",
            initial_sha256=fingerprint,
            final_sha256=verified_fingerprint,
        )
    readiness = project.review_readiness()
    recipe = intelligent_rietveld_recipe(project.input)
    labels = tuple(item.label for item in project.input.parameters.keys)
    advisor_context = _advisor_context(project)
    base = _plan_record(
        spec,
        fingerprint,
        readiness,
        recipe,
        project.input.selection,
        labels,
        advisor_context,
        project.checkpoint is not None,
        None,
        lineage,
    )
    plan_id = hashlib.sha256(_canonical_json(base)).hexdigest()
    return WorkflowPlan(
        spec,
        fingerprint,
        readiness,
        recipe,
        project.input.selection,
        labels,
        advisor_context,
        project.checkpoint is not None,
        plan_id,
        lineage,
    )


def parse_recipe_proposal(
    record: Mapping[str, object],
    plan: WorkflowPlan,
) -> RecipeProposal:
    """Validate an AI/human proposal without trusting its scientific authority."""

    if not isinstance(plan, WorkflowPlan):
        raise TypeError("plan must be WorkflowPlan")
    values = _mapping(record, "recipe proposal")
    _exact_keys(
        values,
        {"schema", "plan_id", "proposal_id", "generated_by", "assumptions", "recipe"},
        "recipe proposal",
    )
    if values["schema"] != RECIPE_PROPOSAL_SCHEMA:
        raise AutomationError("proposal.schema", "unsupported recipe proposal schema")
    if values["plan_id"] != plan.plan_id:
        raise AutomationError(
            "proposal.plan_mismatch",
            "recipe proposal is not bound to this exact workflow plan",
        )
    proposal_id = _trimmed(values["proposal_id"], "proposal_id")
    if _IDENTIFIER.fullmatch(proposal_id) is None:
        raise AutomationError("proposal.id", "proposal_id must be a stable identifier")
    generated_by = _trimmed(values["generated_by"], "generated_by")
    assumptions = _string_tuple(values["assumptions"], "assumptions", maximum=32)
    recipe_record = _mapping(values["recipe"], "recipe")
    _exact_keys(recipe_record, {"name", "stages"}, "recipe")
    stage_records = recipe_record["stages"]
    if not isinstance(stage_records, list) or not 1 <= len(stage_records) <= _MAX_RECIPE_STAGES:
        raise AutomationError(
            "proposal.stage_count",
            f"recipe must contain 1-{_MAX_RECIPE_STAGES} stages",
        )
    stages = tuple(_stage_from_record(item) for item in stage_records)
    recipe = RietveldRecipe(
        _trimmed(recipe_record["name"], "recipe.name"),
        stages,
        "explicit",
        (
            f"External proposal {proposal_id} generated by {generated_by}.",
            *(f"Declared assumption: {item}" for item in assumptions),
        ),
    )
    project = _load_project(plan.spec.project_path)
    _verify_project_fingerprint(plan.spec.project_path, plan.project_fingerprint)
    _validate_external_recipe(recipe, plan, project)
    return RecipeProposal(plan.plan_id, proposal_id, generated_by, assumptions, recipe)


def load_recipe_proposal(path: str | Path, plan: WorkflowPlan) -> RecipeProposal:
    """Load and validate one size-limited recipe proposal."""

    return parse_recipe_proposal(_read_json(Path(path).resolve()), plan)


def lint_recipe_proposal_file(
    path: str | Path,
    plan: WorkflowPlan,
) -> dict[str, object]:
    """Load and deterministically lint one size-limited recipe proposal."""

    return lint_recipe_proposal(_read_json(Path(path).resolve()), plan)


def lint_recipe_proposal(
    record: Mapping[str, object],
    plan: WorkflowPlan,
) -> dict[str, object]:
    """Return deterministic contract and scientific-risk findings for a proposal."""

    if not isinstance(plan, WorkflowPlan):
        raise TypeError("plan must be WorkflowPlan")
    findings: list[dict[str, object]] = []
    try:
        proposal = parse_recipe_proposal(record, plan)
    except AutomationError as error:
        findings.append(
            _lint_finding(
                "error",
                error.code,
                str(error),
                evidence=error.details,
            )
        )
        return _lint_record(plan.plan_id, None, findings, valid_contract=False)

    stages = proposal.recipe.stages
    maximum = plan.authorized_selection
    first = stages[0].selection
    if maximum.phase_scale and not first.phase_scale:
        findings.append(
            _lint_finding(
                "warning",
                "recipe.scale_delayed",
                "Phase scale is authorized but absent from the first stage.",
                stage=stages[0].name,
            )
        )
    if maximum.background and not first.background:
        findings.append(
            _lint_finding(
                "warning",
                "recipe.background_delayed",
                "A refinable background is authorized but absent from the first stage.",
                stage=stages[0].name,
            )
        )
    for family in ("occupancy", "sample_physics", "u_iso"):
        if getattr(first, family):
            findings.append(
                _lint_finding(
                    "warning",
                    f"recipe.early_{family}",
                    f"The first stage releases {family.replace('_', ' ')} before a staged "
                    "scale/position foundation has been demonstrated.",
                    stage=stages[0].name,
                )
            )

    previous: frozenset[str] = frozenset()
    position_terms = {
        "lattice",
        "instrument:wavelength_angstrom",
        "instrument:zero_shift_deg",
        "instrument:sample_displacement_mm",
        "instrument:displace_x_micrometre",
        "instrument:displace_y_micrometre",
    }
    width_terms = {
        "instrument:u_deg2",
        "instrument:v_deg2",
        "instrument:w_deg2",
        "instrument:x_deg",
        "instrument:y_deg",
    }
    for stage in stages:
        current = _selection_keys(stage.selection)
        added = current - previous
        if current == previous:
            findings.append(
                _lint_finding(
                    "warning",
                    "recipe.redundant_stage",
                    "This stage activates no parameter family beyond the preceding stage.",
                    stage=stage.name,
                )
            )
        if len(added) > 4:
            findings.append(
                _lint_finding(
                    "warning",
                    "recipe.broad_release",
                    "This stage releases more than four parameter families at once.",
                    stage=stage.name,
                    evidence={"added": sorted(added)},
                )
            )
        if stage.selection.phase_scale and stage.selection.occupancy:
            findings.append(
                _lint_finding(
                    "warning",
                    "correlation.scale_occupancy",
                    "Phase scale and occupancy are active together; inspect rank and correlation.",
                    stage=stage.name,
                )
            )
        instruments = set(stage.selection.instrument_parameters)
        if stage.selection.lattice and "wavelength_angstrom" in instruments:
            findings.append(
                _lint_finding(
                    "warning",
                    "correlation.lattice_wavelength",
                    "Lattice and wavelength are active together without evidence of an anchor.",
                    stage=stage.name,
                )
            )
        displacement = {
            "sample_displacement_mm",
            "displace_x_micrometre",
            "displace_y_micrometre",
        }
        if "zero_shift_deg" in instruments and instruments.intersection(displacement):
            findings.append(
                _lint_finding(
                    "warning",
                    "correlation.zero_displacement",
                    "Zero shift and specimen displacement are active together.",
                    stage=stage.name,
                )
            )
        if added.intersection(width_terms) and (
            added.intersection(position_terms) or not current.intersection(position_terms)
        ):
            findings.append(
                _lint_finding(
                    "warning",
                    "recipe.width_before_position_stability",
                    "Profile widths are released without an earlier position-only stage.",
                    stage=stage.name,
                )
            )
        previous = current
    if plan.readiness.has_warnings and not proposal.assumptions:
        findings.append(
            _lint_finding(
                "info",
                "recipe.readiness_assumptions_absent",
                "The plan has readiness warnings but the proposal declares no assumptions.",
            )
        )
    findings.append(
        _lint_finding(
            "info",
            "recipe.contract_valid",
            "The proposal satisfies the exact plan, authorization, cumulative-stage, "
            "and policy contract.",
        )
    )
    return _lint_record(
        plan.plan_id,
        proposal.proposal_id,
        findings,
        valid_contract=True,
    )


def _lint_finding(
    severity: Literal["info", "warning", "error"],
    code: str,
    message: str,
    *,
    stage: str | None = None,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "stage": stage,
        "evidence": {} if evidence is None else dict(evidence),
    }


def _lint_record(
    plan_id: str,
    proposal_id: str | None,
    findings: list[dict[str, object]],
    *,
    valid_contract: bool,
) -> dict[str, object]:
    return {
        "schema": RECIPE_LINT_SCHEMA,
        "plan_id": plan_id,
        "proposal_id": proposal_id,
        "valid_contract": valid_contract,
        "error_count": sum(item["severity"] == "error" for item in findings),
        "warning_count": sum(item["severity"] == "warning" for item in findings),
        "findings": findings,
    }


def run_workflow(
    plan: WorkflowPlan,
    *,
    approval_plan_id: str,
    proposal: RecipeProposal | None = None,
    overwrite: bool = False,
    logger: RefinementLogger | None = None,
) -> WorkflowRunResult:
    """Execute one approved, byte-bound plan and write only owned products."""

    if not isinstance(plan, WorkflowPlan):
        raise TypeError("plan must be WorkflowPlan")
    if not isinstance(approval_plan_id, str) or approval_plan_id != plan.plan_id:
        raise AutomationError(
            "approval.required",
            "execution requires the exact plan_id printed by the planning step",
        )
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    refreshed = _build_workflow_plan(plan.spec, lineage=plan.lineage)
    if refreshed.plan_id != plan.plan_id:
        raise AutomationError(
            "plan.stale",
            "project bytes or planning inputs changed after approval; create a new plan",
            approved_plan_id=plan.plan_id,
            current_plan_id=refreshed.plan_id,
        )
    if refreshed.blocked:
        raise AutomationError(
            "plan.blocked",
            "readiness errors block numerical execution",
            readiness=refreshed.readiness.to_record(),
        )
    if proposal is not None and (
        not isinstance(proposal, RecipeProposal) or proposal.plan_id != plan.plan_id
    ):
        raise AutomationError(
            "proposal.plan_mismatch",
            "validated recipe proposal does not belong to the approved plan",
        )
    recipe = refreshed.default_recipe if proposal is None else proposal.recipe
    source: Literal["deterministic", "external_proposal"] = (
        "deterministic" if proposal is None else "external_proposal"
    )
    output = plan.spec.output_directory
    result_path = output / "result.json"
    plan_path = output / "plan.json"
    recipe_path = output / "recipe.json"
    workflow_path = output / "workflow.json"
    workflow_result_path = output / "workflow-result.json"
    csv_path = output / "pattern.csv" if plan.spec.outputs.write_csv else None
    project_path = output / "project" if plan.spec.outputs.save_project else None
    owned = [plan_path, recipe_path, workflow_path, result_path, workflow_result_path]
    if csv_path is not None:
        owned.append(csv_path)
    if project_path is not None:
        owned.append(project_path)
    existing = [str(path) for path in owned if path.exists()]
    if existing and not overwrite:
        raise AutomationError(
            "output.exists",
            "refusing to overwrite workflow-owned outputs",
            paths=existing,
        )

    project = _load_project(plan.spec.project_path)
    _verify_project_fingerprint(plan.spec.project_path, refreshed.project_fingerprint)
    if proposal is not None:
        _validate_external_recipe(proposal.recipe, refreshed, project)
    project.options = replace(project.options, limits=plan.spec.limits)
    try:
        workflow = project.refine_recipe(recipe, logger=logger)
    except Exception as error:
        if isinstance(error, AutomationError):  # pragma: no cover - defensive
            raise
        raise AutomationError(
            "workflow.execution_failed",
            "PhaseSmith could not execute the approved workflow",
            exception_type=type(error).__name__,
            reason=str(error),
        ) from error
    output.mkdir(parents=True, exist_ok=True)
    _write_json(plan_path, refreshed.to_record(), overwrite=overwrite)
    _write_json(recipe_path, recipe.to_record(), overwrite=overwrite)
    _write_json(workflow_path, workflow.to_record(), overwrite=overwrite)
    _write_json(result_path, rietveld_result_record(workflow.final_result), overwrite=overwrite)
    if csv_path is not None:
        project.write_reports(csv_path=csv_path)
    if project_path is not None:
        project.save(project_path, overwrite=overwrite)
    result = WorkflowRunResult(
        refreshed,
        source,
        recipe,
        workflow,
        output,
        result_path,
        csv_path,
        project_path,
    )
    _write_json(workflow_result_path, result.to_record(), overwrite=overwrite)
    return result


def resume_workflow(
    plan: WorkflowPlan,
    *,
    approval_plan_id: str,
    overwrite: bool = False,
    logger: RefinementLogger | None = None,
) -> ResumeRunResult:
    """Continue one approved persisted checkpoint using its exact active model."""

    if not isinstance(plan, WorkflowPlan):
        raise TypeError("plan must be WorkflowPlan")
    if not isinstance(approval_plan_id, str) or approval_plan_id != plan.plan_id:
        raise AutomationError(
            "approval.required",
            "resume requires the exact plan_id printed by the planning step",
        )
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    refreshed = _build_workflow_plan(plan.spec, lineage=plan.lineage)
    if refreshed.plan_id != plan.plan_id:
        raise AutomationError(
            "plan.stale",
            "project bytes or planning inputs changed after approval; create a new plan",
            approved_plan_id=plan.plan_id,
            current_plan_id=refreshed.plan_id,
        )
    if refreshed.blocked:
        raise AutomationError(
            "plan.blocked",
            "readiness errors block numerical execution",
            readiness=refreshed.readiness.to_record(),
        )
    if not refreshed.can_resume:
        raise AutomationError(
            "resume.unavailable",
            "the persisted project has no continuation checkpoint",
        )

    output = plan.spec.output_directory
    result_path = output / "result.json"
    plan_path = output / "plan.json"
    action_path = output / "action.json"
    resume_result_path = output / "resume-result.json"
    csv_path = output / "pattern.csv" if plan.spec.outputs.write_csv else None
    project_path = output / "project" if plan.spec.outputs.save_project else None
    owned = [plan_path, action_path, result_path, resume_result_path]
    if csv_path is not None:
        owned.append(csv_path)
    if project_path is not None:
        owned.append(project_path)
    existing = [str(path) for path in owned if path.exists()]
    if existing and not overwrite:
        raise AutomationError(
            "output.exists",
            "refusing to overwrite resume-owned outputs",
            paths=existing,
        )

    project = _load_project(plan.spec.project_path)
    _verify_project_fingerprint(plan.spec.project_path, refreshed.project_fingerprint)
    project.options = replace(project.options, limits=plan.spec.limits)
    try:
        refinement_result = project.refine(logger=logger)
    except Exception as error:
        raise AutomationError(
            "resume.execution_failed",
            "PhaseSmith could not continue the approved checkpoint",
            exception_type=type(error).__name__,
            reason=str(error),
        ) from error
    output.mkdir(parents=True, exist_ok=True)
    _write_json(plan_path, refreshed.to_record(), overwrite=overwrite)
    _write_json(
        action_path,
        {
            "schema": "phasesmith.automation-action.v1",
            "action": "resume",
            "plan_id": refreshed.plan_id,
        },
        overwrite=overwrite,
    )
    _write_json(result_path, rietveld_result_record(refinement_result), overwrite=overwrite)
    if csv_path is not None:
        project.write_reports(csv_path=csv_path)
    if project_path is not None:
        project.save(project_path, overwrite=overwrite)
    result = ResumeRunResult(
        refreshed,
        refinement_result,
        output,
        result_path,
        csv_path,
        project_path,
    )
    _write_json(resume_result_path, result.to_record(), overwrite=overwrite)
    return result


def review_workflow_output(path: str | Path) -> dict[str, object]:
    """Build a deterministic scientific review from one workflow-run directory."""

    output = Path(path).resolve()
    terminal_path = output / "workflow-result.json"
    plan_path = output / "plan.json"
    workflow_path = output / "workflow.json"
    result_path = output / "result.json"
    terminal = _read_json(terminal_path, max_bytes=16 * 1024 * 1024)
    plan = _read_json(plan_path, max_bytes=16 * 1024 * 1024)
    workflow = _read_json(workflow_path, max_bytes=16 * 1024 * 1024)
    result = _read_json(result_path, max_bytes=32 * 1024 * 1024)
    if terminal.get("schema") != WORKFLOW_RESULT_SCHEMA:
        raise AutomationError("review.schema", "terminal workflow result has an unsupported schema")
    if plan.get("schema") != WORKFLOW_PLAN_SCHEMA:
        raise AutomationError("review.schema", "stored workflow plan has an unsupported schema")
    if result.get("schema") != "phasesmith.result-report.v1":
        raise AutomationError("review.schema", "stored numerical result has an unsupported schema")
    plan_id = _trimmed(terminal.get("plan_id"), "workflow-result.plan_id")
    if plan.get("plan_id") != plan_id:
        raise AutomationError("review.plan_mismatch", "stored plan and result plan IDs differ")
    unsigned_plan = deepcopy(plan)
    unsigned_plan["plan_id"] = None
    if hashlib.sha256(_canonical_json(unsigned_plan)).hexdigest() != plan_id:
        raise AutomationError(
            "review.plan_digest",
            "stored workflow plan does not reproduce its declared plan ID",
        )
    if terminal.get("workflow") != workflow:
        raise AutomationError(
            "review.workflow_mismatch",
            "terminal and standalone workflow records differ",
        )
    terminal_outputs = _mapping(terminal.get("outputs"), "workflow-result.outputs")
    declared_result = Path(
        _trimmed(terminal_outputs.get("result_json"), "workflow-result.outputs.result_json")
    ).resolve()
    if declared_result != result_path:
        raise AutomationError(
            "review.output_mismatch",
            "terminal result path does not identify the reviewed numerical result",
        )
    declared_csv = terminal_outputs.get("pattern_csv")
    if declared_csv is None:
        csv_path = None
    else:
        csv_path = Path(_trimmed(declared_csv, "workflow-result.outputs.pattern_csv")).resolve()
        if csv_path != output / "pattern.csv" or not csv_path.is_file():
            raise AutomationError(
                "review.output_mismatch",
                "terminal pattern path does not identify the reviewed CSV",
            )

    findings: list[dict[str, object]] = []
    completed = _boolean(terminal.get("completed"), "workflow-result.completed")
    if _boolean(workflow.get("completed"), "workflow.completed") != completed:
        raise AutomationError(
            "review.workflow_mismatch",
            "terminal and workflow completion states differ",
        )
    termination_record = _mapping(result.get("termination"), "result.termination")
    termination = _trimmed(termination_record.get("reason"), "result.termination.reason")
    if not completed:
        findings.append(
            _review_finding(
                "warning",
                "workflow.incomplete",
                "The staged workflow did not reach an accepted final stage.",
                recommendation="Inspect the stopping stage before resuming or replanning.",
            )
        )
    if termination not in {"converged", "stagnated"}:
        findings.append(
            _review_finding(
                "warning",
                "termination.requires_review",
                f"The final numerical termination was {termination!r}.",
                recommendation="Review budgets, rejection history, and the last accepted state.",
            )
        )

    stage_records = _workflow_stage_records(workflow)
    stages = []
    for stage in stage_records:
        start = _finite_number(stage.get("starting_rwp"), "stage.starting_rwp")
        final = _finite_number(stage.get("final_rwp"), "stage.final_rwp")
        improvement = start - final
        relative = None if start == 0.0 else improvement / start
        name = _trimmed(
            _mapping(stage.get("stage"), "stage metadata").get("name"),
            "stage.name",
        )
        stages.append(
            {
                "name": name,
                "accepted": _boolean(stage.get("accepted"), f"stage {name}.accepted"),
                "termination": _trimmed(
                    stage.get("termination"),
                    f"stage {name}.termination",
                ),
                "starting_rwp": start,
                "final_rwp": final,
                "absolute_rwp_improvement": improvement,
                "relative_rwp_improvement": relative,
                "iterations": stage.get("iterations"),
                "evaluations": stage.get("evaluations"),
            }
        )
        if improvement < 0.0:
            findings.append(
                _review_finding(
                    "warning",
                    "stage.rwp_regression",
                    f"Stage {name!r} ended with a higher Rwp than it started.",
                    recommendation=(
                        "Do not promote this stage without examining its accepted-state policy."
                    ),
                    evidence={"starting_rwp": start, "final_rwp": final},
                )
            )

    result_parameters = _record_list(result.get("parameters"), "result.parameters")
    advisor = _mapping(plan.get("advisor_context"), "plan.advisor_context")
    initial_parameters = {
        item["label"]: item
        for item in _record_list(advisor.get("parameters"), "advisor parameters")
    }
    movements = []
    bound_contacts = []
    for parameter in result_parameters:
        label = _trimmed(parameter.get("label"), "result parameter label")
        value = _finite_number(parameter.get("value"), f"result parameter {label}")
        scale = _finite_number(parameter.get("scale"), f"result parameter scale {label}")
        if scale <= 0.0:
            raise AutomationError(
                "review.record",
                f"result parameter scale {label} must be positive",
            )
        initial = initial_parameters.get(label)
        if initial is not None:
            before = _finite_number(initial.get("value"), f"initial parameter {label}")
            movements.append(
                {
                    "label": label,
                    "before": before,
                    "after": value,
                    "change": value - before,
                    "scaled_change": (value - before) / scale,
                }
            )
        for side in ("lower", "upper"):
            bound = parameter.get(side)
            if bound is None:
                continue
            bound_value = _finite_number(bound, f"result parameter {side} {label}")
            tolerance = max(1.0e-10, abs(scale) * 1.0e-6)
            if abs(value - bound_value) <= tolerance:
                bound_contacts.append({"label": label, "side": side, "value": value})
    movements.sort(key=lambda item: (-abs(item["scaled_change"]), item["label"]))
    if bound_contacts:
        findings.append(
            _review_finding(
                "warning",
                "parameters.at_bounds",
                "One or more final parameters contact an active bound.",
                recommendation=(
                    "Check physical plausibility and whether another term is compensating."
                ),
                evidence={"contacts": bound_contacts},
            )
        )
    large_movements = [item for item in movements if abs(item["scaled_change"]) >= 5.0]
    if large_movements:
        findings.append(
            _review_finding(
                "warning",
                "parameters.large_motion",
                "One or more parameters moved by at least five configured scales.",
                recommendation="Verify starting values, bounds, and physical interpretation.",
                evidence={"parameters": large_movements[:10]},
            )
        )

    diagnostics = _mapping(result.get("diagnostics"), "result.diagnostics")
    rank = diagnostics.get("jacobian_rank")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int)):
        raise AutomationError("review.record", "jacobian_rank must be an integer or null")
    correlations = _record_list(
        diagnostics.get("unresolved_correlations"),
        "result unresolved correlations",
    )
    active_labels = {
        _trimmed(item.get("label"), "result parameter label") for item in result_parameters
    }
    constrained_targets = {
        _trimmed(item.get("target"), "advisor constraint target")
        for item in _record_list(advisor.get("constraints"), "advisor constraints")
        if item.get("target") in active_labels
    }
    free_parameter_count = len(result_parameters) - len(constrained_targets)
    if rank is not None and rank < free_parameter_count:
        findings.append(
            _review_finding(
                "warning",
                "identifiability.rank_deficient",
                f"Jacobian rank {rank} is below the {free_parameter_count} free parameters.",
                recommendation="Narrow or constrain the correlated parameter selection.",
            )
        )
    if correlations:
        findings.append(
            _review_finding(
                "warning",
                "identifiability.correlations",
                "The final solve reports unresolved parameter correlations.",
                recommendation=(
                    "Constrain or remove one member of each scientifically redundant pair."
                ),
                evidence={"pairs": correlations},
            )
        )

    residual = None
    if csv_path is not None:
        residual = _review_pattern_csv(csv_path, advisor)
        lag = residual.get("lag1_residual_correlation")
        if isinstance(lag, (int, float)) and abs(lag) >= 0.5:
            findings.append(
                _review_finding(
                    "warning",
                    "residual.serial_structure",
                    "Residuals have substantial lag-1 correlation.",
                    recommendation=(
                        "Inspect background, position, profile, and omitted-phase structure."
                    ),
                    evidence={"lag1_residual_correlation": lag},
                )
            )
        regions = residual.get("regions", [])
        populated = [item for item in regions if item["count"]]
        rms_values = sorted(item["weighted_rms"] for item in populated)
        if rms_values and rms_values[len(rms_values) // 2] > 0.0:
            worst = max(populated, key=lambda item: item["weighted_rms"])
            ratio = worst["weighted_rms"] / rms_values[len(rms_values) // 2]
            if ratio >= 2.0:
                findings.append(
                    _review_finding(
                        "warning",
                        "residual.localized_mismatch",
                        "One coordinate region has at least twice the median weighted "
                        "residual RMS.",
                        recommendation=(
                            "Inspect that region for missing phases or local model mismatch."
                        ),
                        evidence={"region": worst, "ratio_to_median": ratio},
                    )
                )
    else:
        findings.append(
            _review_finding(
                "info",
                "residual.csv_unavailable",
                "No pattern CSV was requested, so residual-shape diagnostics are unavailable.",
                recommendation="Enable write_csv in a future workflow for spatial residual review.",
            )
        )

    metrics = _mapping(result.get("metrics"), "result.metrics")
    if not any(item["severity"] == "warning" for item in findings):
        findings.append(
            _review_finding(
                "info",
                "workflow.no_deterministic_warning",
                "No deterministic review warning was triggered.",
                recommendation=(
                    "Still inspect plots, provenance, and physical plausibility before acceptance."
                ),
            )
        )
    base = {
        "schema": WORKFLOW_REVIEW_SCHEMA,
        "review_id": None,
        "plan_id": plan_id,
        "workflow_id": terminal.get("workflow_id"),
        "status": "review_required",
        "completed": completed,
        "termination": termination,
        "metrics": metrics,
        "stages": stages,
        "parameter_movements": movements,
        "bound_contacts": bound_contacts,
        "jacobian_rank": rank,
        "parameter_count": len(result_parameters),
        "free_parameter_count": free_parameter_count,
        "unresolved_correlations": correlations,
        "residual": residual,
        "findings": findings,
        "sources": {
            "workflow_result": _file_digest_record(terminal_path),
            "plan": _file_digest_record(plan_path),
            "workflow": _file_digest_record(workflow_path),
            "result": _file_digest_record(result_path),
            "pattern_csv": None if csv_path is None else _file_digest_record(csv_path),
        },
    }
    review_id = hashlib.sha256(_canonical_json(base)).hexdigest()
    base["review_id"] = review_id
    return base


def replan_workflow(
    result_directory: str | Path,
    *,
    output_directory: str | Path,
    workflow_id: str | None = None,
    limits: RefinementLimits | None = None,
    outputs: WorkflowOutputs | None = None,
) -> WorkflowPlan:
    """Create a new non-executing plan from the last saved accepted project."""

    source = Path(result_directory).resolve()
    next_output = Path(output_directory).resolve()
    if next_output == source or next_output.is_relative_to(source):
        raise AutomationError(
            "replan.output_reuse",
            "a replanned workflow output must lie outside the source audit directory",
        )
    terminal_path = source / "workflow-result.json"
    terminal = _read_json(terminal_path, max_bytes=16 * 1024 * 1024)
    if terminal.get("schema") != WORKFLOW_RESULT_SCHEMA:
        raise AutomationError("replan.schema", "source is not a workflow run output")
    parent_plan_id = _trimmed(terminal.get("plan_id"), "workflow-result.plan_id")
    plan_record = _read_json(source / "plan.json", max_bytes=16 * 1024 * 1024)
    if plan_record.get("plan_id") != parent_plan_id:
        raise AutomationError("replan.plan_mismatch", "source plan and result plan IDs differ")
    previous_spec = WorkflowSpec.from_record(
        _mapping(plan_record.get("workflow_spec"), "stored workflow specification")
    )
    saved_project = source / "project"
    declared_project = _mapping(terminal.get("outputs"), "workflow result outputs").get("project")
    if (
        declared_project is None
        or Path(_trimmed(declared_project, "workflow-result.outputs.project")).resolve()
        != saved_project
    ):
        raise AutomationError(
            "replan.project_missing",
            "source workflow did not retain the expected accepted-state project",
        )
    review = review_workflow_output(source)
    selected_id = (
        _next_workflow_id(previous_spec.workflow_id) if workflow_id is None else workflow_id
    )
    spec = WorkflowSpec(
        selected_id,
        saved_project,
        next_output,
        previous_spec.limits if limits is None else limits,
        previous_spec.outputs if outputs is None else outputs,
    )
    lineage = WorkflowLineage(
        parent_plan_id,
        _trimmed(
            _file_digest_record(terminal_path)["sha256"],
            "source workflow result digest",
        ),
        _trimmed(review["review_id"], "source review digest"),
    )
    return _build_workflow_plan(spec, lineage=lineage)


def _next_workflow_id(previous: str) -> str:
    suffix = ".next"
    return f"{previous[: 100 - len(suffix)]}{suffix}"


def _review_finding(
    severity: Literal["info", "warning", "error"],
    code: str,
    message: str,
    *,
    recommendation: str,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "recommendation": recommendation,
        "evidence": {} if evidence is None else dict(evidence),
    }


def _workflow_stage_records(workflow: Mapping[str, object]) -> list[dict[str, object]]:
    values = workflow.get("stages")
    return _record_list(values, "workflow stages")


def _record_list(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise AutomationError("review.record", f"{name} must be a JSON array")
    return [_mapping(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise AutomationError("review.record", f"{name} must be a finite number")
    return float(value)


def _review_pattern_csv(path: Path, advisor: Mapping[str, object]) -> dict[str, object]:
    size = path.stat().st_size
    if size > 512 * 1024 * 1024:
        raise AutomationError(
            "review.csv_too_large",
            "pattern CSV exceeds the 512 MiB review limit",
            size_bytes=size,
        )
    pattern = _mapping(advisor.get("pattern"), "advisor pattern")
    minimum = _finite_number(pattern.get("minimum"), "advisor pattern minimum")
    maximum = _finite_number(pattern.get("maximum"), "advisor pattern maximum")
    bins = [{"count": 0, "sum_squared": 0.0, "weighted_sum_squared": 0.0} for _ in range(10)]
    count = 0
    total = 0.0
    total_squared = 0.0
    weighted_squared = 0.0
    maximum_absolute = 0.0
    pair_sum = 0.0
    previous = None
    first = None
    last = None
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"x_deg", "residual_calculated_minus_observed", "weight", "included"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AutomationError("review.csv_schema", "pattern CSV has unexpected columns")
        for row_number, row in enumerate(reader, start=2):
            if row_number > 5_000_001:
                raise AutomationError(
                    "review.csv_too_large", "pattern CSV exceeds five million rows"
                )
            if row["included"].lower() not in {"true", "1"}:
                continue
            try:
                coordinate = float(row["x_deg"])
                residual = float(row["residual_calculated_minus_observed"])
                weight = float(row["weight"])
            except (TypeError, ValueError) as error:
                raise AutomationError(
                    "review.csv_value",
                    "pattern CSV contains a non-numeric included row",
                    row=row_number,
                ) from error
            if not np.isfinite((coordinate, residual, weight)).all() or weight < 0.0:
                raise AutomationError(
                    "review.csv_value",
                    "pattern CSV contains invalid included values",
                    row=row_number,
                )
            count += 1
            total += residual
            total_squared += residual * residual
            weighted_squared += residual * residual * weight
            maximum_absolute = max(maximum_absolute, abs(residual))
            if first is None:
                first = residual
            if previous is not None:
                pair_sum += previous * residual
            previous = residual
            last = residual
            fraction = 0.0 if maximum == minimum else (coordinate - minimum) / (maximum - minimum)
            index = min(9, max(0, int(fraction * 10.0)))
            bins[index]["count"] += 1
            bins[index]["sum_squared"] += residual * residual
            bins[index]["weighted_sum_squared"] += residual * residual * weight
    if count == 0:
        raise AutomationError("review.csv_empty", "pattern CSV contains no included samples")
    mean = total / count
    variance_sum = max(0.0, total_squared - count * mean * mean)
    if count < 2 or variance_sum == 0.0:
        lag = None
    else:
        covariance_sum = pair_sum - mean * (2.0 * total - first - last) + (count - 1) * mean * mean
        lag = float(np.clip(covariance_sum / variance_sum, -1.0, 1.0))
    regions = []
    span = maximum - minimum
    for index, item in enumerate(bins):
        region_count = item["count"]
        regions.append(
            {
                "minimum": minimum + span * index / 10.0,
                "maximum": minimum + span * (index + 1) / 10.0,
                "count": region_count,
                "rms": None
                if region_count == 0
                else float(np.sqrt(item["sum_squared"] / region_count)),
                "weighted_rms": None
                if region_count == 0
                else float(np.sqrt(item["weighted_sum_squared"] / region_count)),
            }
        )
    return {
        "included_sample_count": count,
        "mean": mean,
        "rms": float(np.sqrt(total_squared / count)),
        "weighted_rms": float(np.sqrt(weighted_squared / count)),
        "maximum_absolute": maximum_absolute,
        "lag1_residual_correlation": lag,
        "regions": regions,
    }


def write_workflow_plan(
    plan: WorkflowPlan,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a finite plan record without silently replacing one."""

    if not isinstance(plan, WorkflowPlan):
        raise TypeError("plan must be WorkflowPlan")
    destination = Path(path).resolve()
    _write_json(destination, plan.to_record(), overwrite=overwrite)
    return destination


def inspect_powder_file(
    path: str | Path,
    *,
    format: PowderFormat = "auto",
    bank: int = 1,
) -> dict[str, object]:
    """Return bounded machine-readable facts without guessing experiment physics."""

    source = Path(path).resolve()
    data = read_powder_data(source, format=format, bank=bank)
    differences = np.diff(data.x)
    coordinate_unit = "unknown" if data.format == "columns" else "degree_2theta"
    unknown_physics = [
        "radiation and wavelength",
        "instrument profile",
        "specimen geometry",
        "integrated-intensity correction",
        "background model",
    ]
    if data.format == "columns":
        unknown_physics.insert(0, "coordinate unit")
    return {
        "schema": "phasesmith.powder-inspection.v1",
        "source": _file_digest_record(source),
        "format": data.format,
        "bank": data.bank,
        "coordinate": {
            "unit": coordinate_unit,
            "minimum": float(data.x[0]),
            "maximum": float(data.x[-1]),
            "sample_count": int(data.x.size),
            "minimum_step": None if differences.size == 0 else float(np.min(differences)),
            "maximum_step": None if differences.size == 0 else float(np.max(differences)),
        },
        "intensity": {
            "minimum": float(np.min(data.observed_y)),
            "maximum": float(np.max(data.observed_y)),
            "uncertainty_present": data.uncertainty is not None,
            "masked_samples": 0 if data.mask is None else int(np.count_nonzero(~data.mask)),
        },
        "unknown_physics": unknown_physics,
    }


def inspect_cif_file(
    path: str | Path,
    *,
    block: str | None = None,
    strict: bool = True,
) -> dict[str, object]:
    """Return structure and provenance facts without constructing a workflow."""

    source = Path(path).resolve()
    imported = read_cif(source, block=block, strict=strict)
    structure = imported.structure
    return {
        "schema": "phasesmith.cif-inspection.v1",
        "source": _file_digest_record(source),
        "selected_block": imported.selected_block,
        "available_blocks": list(imported.available_blocks),
        "structure": {
            "structure_id": structure.structure_id,
            "name": structure.name,
            "cell": list(structure.cell.as_tuple()),
            "site_count": len(structure.sites),
            "site_ids": [site.site_id for site in structure.sites],
            "species": sorted({site.type_symbol for site in structure.sites}),
            "symmetry_operation_count": len(structure.space_group.operations),
            "source": None if structure.source is None else asdict(structure.source),
        },
        "diagnostics": [asdict(item) for item in imported.diagnostics],
    }


def workflow_spec_schema() -> dict[str, object]:
    """Return the JSON Schema for the version-1 persisted-project task."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://phasesmith.org/schema/workflow-spec-v1.json",
        "title": WORKFLOW_SPEC_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "workflow_id",
            "project_path",
            "output_directory",
            "limits",
            "outputs",
        ],
        "properties": {
            "schema": {"const": WORKFLOW_SPEC_SCHEMA},
            "workflow_id": {"type": "string", "pattern": _IDENTIFIER.pattern},
            "project_path": {"type": "string", "minLength": 1},
            "output_directory": {"type": "string", "minLength": 1},
            "limits": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "max_iterations",
                    "max_evaluations",
                    "max_runtime_seconds",
                    "max_consecutive_rejections",
                ],
                "properties": {
                    "max_iterations": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10_000,
                    },
                    "max_evaluations": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1_000_000,
                    },
                    "max_runtime_seconds": {
                        "type": ["number", "null"],
                        "exclusiveMinimum": 0,
                        "maximum": 86_400,
                    },
                    "max_consecutive_rejections": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1_000,
                    },
                },
            },
            "outputs": {
                "type": "object",
                "additionalProperties": False,
                "required": ["write_csv", "save_project"],
                "properties": {
                    "write_csv": {"type": "boolean"},
                    "save_project": {"type": "boolean"},
                },
            },
        },
    }


def recipe_proposal_schema() -> dict[str, object]:
    """Return the JSON Schema advertised to an external recipe advisor."""

    selection = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "phase_scale",
            "lattice",
            "coordinates",
            "occupancy",
            "u_iso",
            "sample_physics",
            "instrument_parameters",
            "background",
        ],
        "properties": {
            **{
                name: {"type": "boolean"}
                for name in (
                    "phase_scale",
                    "lattice",
                    "coordinates",
                    "occupancy",
                    "u_iso",
                    "sample_physics",
                    "background",
                )
            },
            "instrument_parameters": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string"},
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://phasesmith.org/schema/recipe-proposal-v1.json",
        "title": RECIPE_PROPOSAL_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "plan_id",
            "proposal_id",
            "generated_by",
            "assumptions",
            "recipe",
        ],
        "properties": {
            "schema": {"const": RECIPE_PROPOSAL_SCHEMA},
            "plan_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "proposal_id": {"type": "string", "pattern": _IDENTIFIER.pattern},
            "generated_by": {"type": "string", "minLength": 1},
            "assumptions": {
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string", "minLength": 1},
            },
            "recipe": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "stages"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "stages": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_RECIPE_STAGES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "selection", "rationale"],
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "selection": selection,
                                "rationale": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 16,
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AutomationError("record.type", f"{name} must be a JSON object")
    return dict(value)


def _exact_keys(values: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(values)
    if actual != expected:
        raise AutomationError(
            "record.keys",
            f"{name} fields do not match the versioned contract",
            missing=sorted(expected - actual),
            unknown=sorted(actual - expected),
        )


def _trimmed(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AutomationError("record.string", f"{name} must be a non-empty trimmed string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise AutomationError("record.boolean", f"{name} must be a boolean")
    return value


def _positive_integer(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise AutomationError(
            "record.integer",
            f"{name} must be an integer in [1, {maximum}]",
        )
    return value


def _string_tuple(value: object, name: str, *, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise AutomationError("record.array", f"{name} must contain at most {maximum} strings")
    return tuple(_trimmed(item, f"{name}[{index}]") for index, item in enumerate(value))


def _resolved_path(value: object, base: Path, name: str) -> Path:
    text = _trimmed(value, name)
    path = Path(text)
    return (path if path.is_absolute() else base / path).resolve()


def _limits_from_record(value: object) -> RefinementLimits:
    record = _mapping(value, "limits")
    _exact_keys(
        record,
        {
            "max_iterations",
            "max_evaluations",
            "max_runtime_seconds",
            "max_consecutive_rejections",
        },
        "limits",
    )
    runtime = record["max_runtime_seconds"]
    if runtime is not None and (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not np.isfinite(runtime)
        or not 0.0 < float(runtime) <= 86_400.0
    ):
        raise AutomationError(
            "record.runtime",
            "limits.max_runtime_seconds must be null or lie in (0, 86400]",
        )
    return RefinementLimits(
        _positive_integer(record["max_iterations"], "limits.max_iterations", 10_000),
        _positive_integer(record["max_evaluations"], "limits.max_evaluations", 1_000_000),
        None if runtime is None else float(runtime),
        _positive_integer(
            record["max_consecutive_rejections"],
            "limits.max_consecutive_rejections",
            1_000,
        ),
    )


def _limits_record(limits: RefinementLimits) -> dict[str, object]:
    return {
        "max_iterations": limits.max_iterations,
        "max_evaluations": limits.max_evaluations,
        "max_runtime_seconds": limits.max_runtime_seconds,
        "max_consecutive_rejections": limits.max_consecutive_rejections,
    }


def _selection_record(selection: RietveldParameterSelection) -> dict[str, object]:
    return {
        "phase_scale": selection.phase_scale,
        "lattice": selection.lattice,
        "coordinates": selection.coordinates,
        "occupancy": selection.occupancy,
        "u_iso": selection.u_iso,
        "sample_physics": selection.sample_physics,
        "instrument_parameters": list(selection.instrument_parameters),
        "background": selection.background,
    }


def _selection_from_record(value: object) -> RietveldParameterSelection:
    record = _mapping(value, "stage.selection")
    expected = {
        "phase_scale",
        "lattice",
        "coordinates",
        "occupancy",
        "u_iso",
        "sample_physics",
        "instrument_parameters",
        "background",
    }
    _exact_keys(record, expected, "stage.selection")
    instruments = _string_tuple(
        record["instrument_parameters"],
        "stage.selection.instrument_parameters",
        maximum=16,
    )
    try:
        return RietveldParameterSelection(
            phase_scale=_boolean(record["phase_scale"], "stage.selection.phase_scale"),
            lattice=_boolean(record["lattice"], "stage.selection.lattice"),
            coordinates=_boolean(record["coordinates"], "stage.selection.coordinates"),
            occupancy=_boolean(record["occupancy"], "stage.selection.occupancy"),
            u_iso=_boolean(record["u_iso"], "stage.selection.u_iso"),
            sample_physics=_boolean(record["sample_physics"], "stage.selection.sample_physics"),
            instrument_parameters=instruments,
            background=_boolean(record["background"], "stage.selection.background"),
        )
    except (TypeError, ValueError) as error:
        raise AutomationError(
            "proposal.selection",
            "stage selection contains an unsupported parameter family",
            reason=str(error),
        ) from error


def _stage_from_record(value: object) -> RietveldStage:
    record = _mapping(value, "recipe stage")
    _exact_keys(record, {"name", "selection", "rationale"}, "recipe stage")
    rationale = _string_tuple(record["rationale"], "stage.rationale", maximum=16)
    if not rationale:
        raise AutomationError("proposal.rationale", "every proposal stage requires rationale")
    return RietveldStage(
        _trimmed(record["name"], "stage.name"),
        _selection_from_record(record["selection"]),
        rationale,
    )


def _selection_keys(selection: RietveldParameterSelection) -> frozenset[str]:
    keys = {
        name
        for name in (
            "phase_scale",
            "lattice",
            "coordinates",
            "occupancy",
            "u_iso",
            "sample_physics",
            "background",
        )
        if getattr(selection, name)
    }
    keys.update(f"instrument:{name}" for name in selection.instrument_parameters)
    return frozenset(keys)


def _validate_cumulative(stages: tuple[RietveldStage, ...]) -> None:
    previous: frozenset[str] = frozenset()
    for stage in stages:
        current = _selection_keys(stage.selection)
        if not previous.issubset(current):
            raise AutomationError(
                "proposal.non_cumulative",
                "external recipe stages must be cumulative",
                stage=stage.name,
                removed=sorted(previous - current),
            )
        previous = current


def _validate_external_recipe(
    recipe: RietveldRecipe,
    plan: WorkflowPlan,
    project: RietveldProject,
) -> None:
    if not 1 <= len(recipe.stages) <= _MAX_RECIPE_STAGES:
        raise AutomationError(
            "proposal.stage_count",
            f"recipe must contain 1-{_MAX_RECIPE_STAGES} stages",
        )
    if recipe.mode != "explicit":
        raise AutomationError(
            "proposal.mode",
            "external recipe proposals must use explicit mode",
        )
    _validate_cumulative(recipe.stages)
    if recipe.stages[-1].selection != plan.authorized_selection:
        raise AutomationError(
            "proposal.final_authorization",
            "the final proposal stage must activate the complete authorized selection",
        )
    for stage in recipe.stages:
        if stage.options is not None:
            raise AutomationError(
                "proposal.custom_options",
                "external recipe stages cannot replace approved solver options",
                stage=stage.name,
            )
        if tuple(reason.value for reason in stage.accepted_terminations) != (
            "converged",
            "stagnated",
        ):
            raise AutomationError(
                "proposal.termination_policy",
                "external recipe stages cannot replace the safe termination policy",
                stage=stage.name,
            )
    try:
        validate_rietveld_recipe(project.input, recipe)
    except (TypeError, ValueError) as error:
        raise AutomationError(
            "proposal.invalid_recipe",
            "recipe proposal violates the PhaseSmith parameter or constraint contract",
            reason=str(error),
        ) from error


def _advisor_context(project: RietveldProject) -> AdvisorContext:
    input_data = project.input
    experiment = input_data.experiment
    radiation = experiment.radiation
    if isinstance(radiation, ComponentRadiation):
        radiation_record: dict[str, object] = {
            "kind": type(radiation).__name__,
            "probe": radiation.probe.value,
            "reference_wavelength_angstrom": radiation.wavelength_angstrom,
            "components": [
                {
                    "wavelength_angstrom": float(wavelength),
                    "normalized_intensity": float(intensity),
                }
                for wavelength, intensity in zip(
                    radiation.components.wavelengths_angstrom,
                    radiation.components.normalized_intensities,
                    strict=True,
                )
            ],
        }
    else:
        radiation_record = {
            "kind": type(radiation).__name__,
            "probe": radiation.probe.value,
            "wavelength_angstrom": radiation.wavelength_angstrom,
        }
    geometry = experiment.geometry
    if isinstance(geometry, BraggBrentanoGeometry):
        geometry_record: dict[str, object] | None = {
            "kind": type(geometry).__name__,
            "goniometer_radius_mm": geometry.goniometer_radius_mm,
            "sample_displacement_mm": geometry.sample_displacement_mm,
        }
    elif isinstance(geometry, DebyeScherrerGeometry):
        geometry_record = {
            "kind": type(geometry).__name__,
            "goniometer_radius_mm": geometry.goniometer_radius_mm,
            "displace_x_micrometre": geometry.displace_x_micrometre,
            "displace_y_micrometre": geometry.displace_y_micrometre,
        }
    else:
        geometry_record = None
    axial = experiment.axial_geometry
    axial_record = (
        None
        if axial is None
        else {
            "kind": type(axial).__name__,
            "sample_over_radius": axial.sample_over_radius,
            "detector_over_radius": axial.detector_over_radius,
        }
    )
    observed = input_data.pattern.observed_y
    included = (
        input_data.pattern.x.size
        if input_data.pattern.mask is None
        else int(np.count_nonzero(input_data.pattern.mask))
    )
    background = input_data.background
    background_record = (
        None
        if background is None
        else {
            "kind": type(background).__name__,
            "background_id": background.background_id,
            "parameter_names": list(background.parameter_names()),
            "coefficient_count": len(background.coefficients),
        }
    )
    record = {
        "schema": ADVISOR_CONTEXT_SCHEMA,
        "privacy": {
            "contains_paths": False,
            "contains_raw_pattern": False,
            "contains_raw_cif": False,
            "note": "Identifiers and numerical scientific metadata remain visible.",
        },
        "pattern": {
            "coordinate_unit": "degree_2theta",
            "minimum": float(input_data.pattern.x[0]),
            "maximum": float(input_data.pattern.x[-1]),
            "sample_count": int(input_data.pattern.x.size),
            "included_sample_count": included,
            "uncertainty_present": input_data.pattern.uncertainty is not None,
            "fixed_background_present": bool(np.any(input_data.pattern.background != 0.0)),
            "observed_minimum": None if observed is None else float(np.min(observed)),
            "observed_maximum": None if observed is None else float(np.max(observed)),
        },
        "experiment": {
            "radiation": radiation_record,
            "instrument": {
                "kind": type(experiment.instrument).__name__,
                "wavelength_angstrom": experiment.instrument.wavelength_angstrom,
                "u_deg2": experiment.instrument.u_deg2,
                "v_deg2": experiment.instrument.v_deg2,
                "w_deg2": experiment.instrument.w_deg2,
                "x_deg": experiment.instrument.x_deg,
                "y_deg": experiment.instrument.y_deg,
                "zero_shift_deg": experiment.zero_shift_deg,
            },
            "geometry": geometry_record,
            "axial_geometry": axial_record,
        },
        "phases": [_advisor_phase_record(phase) for phase in input_data.phases],
        "background": background_record,
        "parameters": [_advisor_parameter_record(spec) for spec in input_data.parameters.specs],
        "constraints": [_advisor_constraint_record(item) for item in input_data.constraints],
        "limitations": [
            "No residual or objective was evaluated during planning.",
            "Phase completeness and omitted physical effects cannot be inferred from the project.",
            "Model identities and values are evidence, not permission to expand authorization.",
        ],
    }
    return AdvisorContext(record)


def _advisor_phase_record(phase: object) -> dict[str, object]:
    structure = phase.structure
    descriptor = phase.scattering.descriptor
    physics = phase.physics
    providers = (
        physics.providers
        if hasattr(physics, "providers")
        else (() if physics is None else (physics,))
    )
    return {
        "phase_id": phase.phase_id,
        "name": phase.name,
        "scale": phase.scale,
        "cell": list(structure.cell.as_tuple()),
        "crystal_system": structure.space_group.crystal_system,
        "symmetry_operation_count": len(structure.space_group.operations),
        "independent_site_count": len(structure.sites),
        "species": sorted({site.type_symbol for site in structure.sites}),
        "reflection_family_count": phase.reflections.reflection_count,
        "source_provenance_present": structure.source is not None,
        "scattering": {
            "provider_id": descriptor.provider_id,
            "provider_version": descriptor.provider_version,
            "probe": descriptor.probe,
            "amplitude_unit": descriptor.amplitude_unit,
        },
        "intensity_correction": {"kind": type(phase.intensity_correction).__name__},
        "sample_physics": [
            {
                "kind": type(provider).__name__,
                "provider_id": provider.descriptor.provider_id,
                "provider_version": provider.descriptor.provider_version,
            }
            for provider in providers
        ],
    }


def _advisor_parameter_record(spec: object) -> dict[str, object]:
    lower = spec.bounds.lower
    upper = spec.bounds.upper
    return {
        "label": spec.key.label,
        "module": spec.key.module,
        "owner_id": spec.key.owner_id,
        "name": spec.key.name,
        "value": spec.value,
        "unit": spec.unit,
        "lower": None if not np.isfinite(lower) else lower,
        "upper": None if not np.isfinite(upper) else upper,
        "scale": spec.scale,
        "refine": spec.refine,
    }


def _advisor_constraint_record(constraint: object) -> dict[str, object]:
    if isinstance(constraint, FixedConstraint):
        return {
            "kind": "fixed",
            "target": constraint.target.label,
            "value": constraint.value,
        }
    if isinstance(constraint, AffineConstraint):
        return {
            "kind": "affine",
            "target": constraint.target.label,
            "source": constraint.source.label,
            "multiplier": constraint.multiplier,
            "offset": constraint.offset,
        }
    if isinstance(constraint, LinearConstraint):
        return {
            "kind": "linear",
            "target": constraint.target.label,
            "terms": [
                {"source": key.label, "coefficient": coefficient}
                for key, coefficient in constraint.terms
            ],
            "offset": constraint.offset,
        }
    raise TypeError("unsupported Rietveld constraint")


def _plan_record(
    spec: WorkflowSpec,
    fingerprint: str,
    readiness: RietveldReadinessReport,
    recipe: RietveldRecipe,
    authorization: RietveldParameterSelection,
    parameter_labels: tuple[str, ...],
    advisor_context: AdvisorContext,
    can_resume: bool,
    plan_id: str | None,
    lineage: WorkflowLineage | None,
) -> dict[str, object]:
    return {
        "schema": WORKFLOW_PLAN_SCHEMA,
        "plan_id": plan_id,
        "workflow_spec": spec.to_record(),
        "project": {
            "sha256": fingerprint,
            "can_resume": can_resume,
        },
        "lineage": None if lineage is None else lineage.to_record(),
        "status": "blocked" if readiness.has_errors else "ready_for_approval",
        "readiness": readiness.to_record(),
        "authorization": {
            "maximum_selection": _selection_record(authorization),
            "parameter_labels": list(parameter_labels),
        },
        "advisor_context": advisor_context.to_record(),
        "default_recipe": recipe.to_record(),
        "external_recipe_proposal": {
            "schema": RECIPE_PROPOSAL_SCHEMA,
            "maximum_stages": _MAX_RECIPE_STAGES,
            "must_be_cumulative": True,
            "final_stage_must_equal_maximum_selection": True,
            "custom_stage_options_allowed": False,
            "safe_accepted_terminations": ["converged", "stagnated"],
        },
        "guidance": list(_GUIDANCE),
        "approval": {
            "required": True,
            "instruction": "Pass this exact plan_id to run only after reviewing the plan.",
        },
    }


def _canonical_json(record: object) -> bytes:
    return json.dumps(
        record,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AutomationError("json.duplicate_key", "JSON contains a duplicate key", key=key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise AutomationError(
        "json.nonfinite",
        "JSON numeric values must be finite",
        value=value,
    )


def _read_json(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> dict[str, object]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise AutomationError(
                "json.too_large",
                "JSON input exceeds the automation size limit",
                size_bytes=size,
                max_bytes=max_bytes,
            )
        record = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except AutomationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AutomationError(
            "json.invalid",
            "could not read a finite UTF-8 JSON object",
            path=str(path),
            reason=str(error),
        ) from error
    return _mapping(record, "JSON document")


def _load_project(path: Path) -> RietveldProject:
    try:
        return RietveldProject.load(path)
    except Exception as error:
        raise AutomationError(
            "project.load_failed",
            "could not load the persisted Rietveld project",
            path=str(path),
            exception_type=type(error).__name__,
            reason=str(error),
        ) from error


def _project_fingerprint(root: Path) -> str:
    if not root.is_dir():
        raise AutomationError(
            "project.not_directory",
            "project_path must be an existing directory",
            path=str(root),
        )
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AutomationError(
                "project.symlink",
                "project fingerprints reject symbolic links",
                path=str(path),
            )
        if not path.is_file():
            continue
        files.append(path)
        total += path.stat().st_size
        if len(files) > _MAX_PROJECT_FILES or total > _MAX_PROJECT_BYTES:
            raise AutomationError(
                "project.too_large",
                "project exceeds automation fingerprint limits",
                file_count=len(files),
                total_bytes=total,
            )
    if not files:
        raise AutomationError("project.empty", "project directory contains no files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _verify_project_fingerprint(root: Path, expected: str) -> None:
    actual = _project_fingerprint(root)
    if actual != expected:
        raise AutomationError(
            "plan.stale",
            "project bytes changed while the approved project was being loaded",
            approved_sha256=expected,
            current_sha256=actual,
        )


def _file_digest_record(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise AutomationError(
            "input.not_file",
            "input path must be one regular file",
            path=str(path),
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _write_json(path: Path, record: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise AutomationError("output.exists", "refusing to overwrite output", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        if overwrite:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise AutomationError(
                    "output.exists",
                    "refusing to overwrite output",
                    path=str(path),
                ) from error
            temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
