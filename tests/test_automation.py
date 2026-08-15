from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith import automation
from phasesmith.cli import main as cli_main
from phasesmith.refinement import (
    RefinementLimits,
    RietveldOptions,
    RietveldParameterSelection,
    rietveld,
)

P1_CIF = """
data_automation
_cell_length_a 4.1
_cell_length_b 5.2
_cell_length_c 6.3
_cell_angle_alpha 77
_cell_angle_beta 83
_cell_angle_gamma 72
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0.11 0.22 0.33 1.0 0.012
O1 O 0.41 0.52 0.63 1.0 0.018
"""


def _request() -> rietveld.RietveldInput:
    x = np.linspace(15.0, 100.0, 3_401)
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument)
    selection = RietveldParameterSelection(
        phase_scale=True,
        instrument_parameters=("zero_shift_deg",),
    )
    initial = rietveld.RietveldInput.from_cif(
        phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment,
        P1_CIF,
        phase_id="automation",
        selection=selection,
        scale=0.72,
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(1.5406),
    )
    truth = rietveld.calculate(
        initial.pattern,
        replace(experiment, zero_shift_deg=0.035),
        (replace(initial.phases[0], scale=1.0),),
    )
    return replace(initial, pattern=phasesmith.PowderPattern(x, observed_y=truth.y))


def _saved_project(tmp_path):
    project = phasesmith.RietveldProject(
        _request(),
        RietveldOptions(
            limits=RefinementLimits(max_iterations=50, max_evaluations=500),
            estimate_covariance=False,
        ),
    )
    return project.save(tmp_path / "source-project")


def _spec(tmp_path) -> automation.WorkflowSpec:
    return automation.WorkflowSpec(
        "automation-test",
        _saved_project(tmp_path),
        tmp_path / "output",
        RefinementLimits(50, 500, None, 20),
        automation.WorkflowOutputs(write_csv=True, save_project=True),
    )


def _proposal_record(plan: automation.WorkflowPlan) -> dict[str, object]:
    maximum = plan.to_record()["authorization"]["maximum_selection"]
    scale_only = dict(maximum)
    scale_only["instrument_parameters"] = []
    return {
        "schema": automation.RECIPE_PROPOSAL_SCHEMA,
        "plan_id": plan.plan_id,
        "proposal_id": "gpt-guided-v1",
        "generated_by": "gpt-5.6-sol",
        "assumptions": ["The persisted project already encodes the physical model."],
        "recipe": {
            "name": "scale-then-position",
            "stages": [
                {
                    "name": "scale",
                    "selection": scale_only,
                    "rationale": ["Establish intensity scale before position terms."],
                },
                {
                    "name": "positions",
                    "selection": maximum,
                    "rationale": ["Release the authorized zero shift after scale."],
                },
            ],
        },
    }


def test_plan_is_read_only_stable_and_discloses_ai_guidance(tmp_path) -> None:
    spec = _spec(tmp_path)

    first = automation.plan_workflow(spec)
    second = automation.plan_workflow(spec)

    assert first.plan_id == second.plan_id
    assert first.blocked is False
    assert first.default_recipe.mode == "intelligent"
    record = first.to_record()
    assert record["approval"]["required"] is True
    context = record["advisor_context"]
    assert context["schema"] == automation.ADVISOR_CONTEXT_SCHEMA
    assert context["privacy"]["contains_paths"] is False
    assert context["pattern"]["sample_count"] == 3_401
    assert context["experiment"]["radiation"]["probe"] == "x-ray"
    assert context["phases"][0]["independent_site_count"] == 2
    assert {item["name"] for item in context["parameters"]} >= {
        "scale",
        "zero_shift_deg",
    }
    assert str(tmp_path) not in json.dumps(context)
    guidance = record["guidance"]
    assert any("lower Rwp alone" in item for item in guidance)
    assert not spec.output_directory.exists()


def test_external_recipe_is_plan_bound_cumulative_and_fully_authorized(tmp_path) -> None:
    plan = automation.plan_workflow(_spec(tmp_path))
    record = _proposal_record(plan)

    proposal = automation.parse_recipe_proposal(record, plan)

    assert proposal.generated_by == "gpt-5.6-sol"
    assert proposal.recipe.stages[-1].selection == plan.authorized_selection
    assert automation.parse_recipe_proposal(proposal.to_record(), plan) == proposal

    wrong_plan = dict(record)
    wrong_plan["plan_id"] = "0" * 64
    with pytest.raises(automation.AutomationError, match="not bound") as error:
        automation.parse_recipe_proposal(wrong_plan, plan)
    assert error.value.code == "proposal.plan_mismatch"

    non_cumulative = _proposal_record(plan)
    stages = non_cumulative["recipe"]["stages"]
    stages.reverse()
    with pytest.raises(automation.AutomationError, match="cumulative"):
        automation.parse_recipe_proposal(non_cumulative, plan)


def test_recipe_lint_is_structured_and_preserves_strict_validation(tmp_path) -> None:
    plan = automation.plan_workflow(_spec(tmp_path))
    record = _proposal_record(plan)

    lint = automation.lint_recipe_proposal(record, plan)

    assert lint["schema"] == automation.RECIPE_LINT_SCHEMA
    assert lint["valid_contract"] is True
    assert lint["error_count"] == 0
    assert any(item["code"] == "recipe.contract_valid" for item in lint["findings"])

    invalid = dict(record)
    invalid["plan_id"] = "0" * 64
    rejected = automation.lint_recipe_proposal(invalid, plan)
    assert rejected["valid_contract"] is False
    assert rejected["error_count"] == 1
    assert rejected["findings"][0]["code"] == "proposal.plan_mismatch"


def test_run_revalidates_directly_constructed_external_proposals(tmp_path) -> None:
    plan = automation.plan_workflow(_spec(tmp_path))
    bypass_recipe = phasesmith.refinement.RietveldRecipe(
        "attempted-bypass",
        tuple(reversed(plan.default_recipe.stages)),
    )
    bypass = automation.RecipeProposal(
        plan.plan_id,
        "attempted-bypass",
        "direct-constructor",
        (),
        bypass_recipe,
    )

    with pytest.raises(automation.AutomationError) as error:
        automation.run_workflow(
            plan,
            approval_plan_id=plan.plan_id,
            proposal=bypass,
        )

    assert error.value.code == "proposal.non_cumulative"
    assert not plan.spec.output_directory.exists()


def test_execution_requires_exact_approval_and_rejects_stale_project(tmp_path) -> None:
    spec = _spec(tmp_path)
    plan = automation.plan_workflow(spec)

    with pytest.raises(automation.AutomationError) as approval:
        automation.run_workflow(plan, approval_plan_id="not-approved")
    assert approval.value.code == "approval.required"

    (spec.project_path / "after-plan.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(automation.AutomationError) as stale:
        automation.run_workflow(plan, approval_plan_id=plan.plan_id)
    assert stale.value.code == "plan.stale"


def test_output_collision_is_rejected_before_numerical_execution(tmp_path, monkeypatch) -> None:
    spec = _spec(tmp_path)
    plan = automation.plan_workflow(spec)
    spec.output_directory.mkdir()
    (spec.output_directory / "workflow-result.json").write_text("owned", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("refinement must not start")

    monkeypatch.setattr(phasesmith.RietveldProject, "refine_recipe", forbidden)
    with pytest.raises(automation.AutomationError) as error:
        automation.run_workflow(plan, approval_plan_id=plan.plan_id)
    assert error.value.code == "output.exists"


def test_validated_ai_recipe_runs_and_writes_auditable_outputs(tmp_path, capsys) -> None:
    spec = _spec(tmp_path)
    plan = automation.plan_workflow(spec)
    proposal = automation.parse_recipe_proposal(_proposal_record(plan), plan)

    completed = automation.run_workflow(
        plan,
        approval_plan_id=plan.plan_id,
        proposal=proposal,
    )

    assert completed.recipe_source == "external_proposal"
    assert completed.workflow.completed is True
    assert completed.workflow.final_result.metrics.rwp < 1.0e-5
    assert completed.result_json.is_file()
    assert completed.pattern_csv is not None and completed.pattern_csv.is_file()
    assert completed.saved_project is not None and completed.saved_project.is_dir()
    terminal = json.loads((spec.output_directory / "workflow-result.json").read_text())
    assert terminal["plan_id"] == plan.plan_id
    assert terminal["recipe"]["planner_notes"][0].startswith("External proposal")

    review = automation.review_workflow_output(spec.output_directory)
    assert review["schema"] == automation.WORKFLOW_REVIEW_SCHEMA
    assert review["plan_id"] == plan.plan_id
    assert review["status"] == "review_required"
    assert review["residual"]["included_sample_count"] == 3_401
    assert len(review["review_id"]) == 64
    assert automation.review_workflow_output(spec.output_directory) == review

    assert cli_main(["review", str(spec.output_directory)]) == 0
    cli_review = json.loads(capsys.readouterr().out)
    assert cli_review["review_id"] == review["review_id"]

    cli_plan_path = tmp_path / "cli-next-plan.json"
    assert (
        cli_main(
            [
                "replan",
                str(spec.output_directory),
                "--output-directory",
                str(tmp_path / "cli-next-output"),
                "--plan-output",
                str(cli_plan_path),
            ]
        )
        == 0
    )
    cli_replan = json.loads(capsys.readouterr().out)
    assert cli_replan["lineage"]["parent_plan_id"] == plan.plan_id
    assert json.loads(cli_plan_path.read_text())["plan_id"] == cli_replan["plan_id"]

    replanned = automation.replan_workflow(
        spec.output_directory,
        output_directory=tmp_path / "next-output",
    )
    assert replanned.lineage is not None
    assert replanned.lineage.parent_plan_id == plan.plan_id
    assert replanned.lineage.source_review_sha256 == review["review_id"]
    assert replanned.spec.project_path == spec.output_directory / "project"
    assert replanned.plan_id != plan.plan_id

    with pytest.raises(automation.AutomationError) as nested_output:
        automation.replan_workflow(
            spec.output_directory,
            output_directory=spec.output_directory / "nested",
        )
    assert nested_output.value.code == "replan.output_reuse"

    next_result = automation.run_workflow(
        replanned,
        approval_plan_id=replanned.plan_id,
    )
    assert next_result.plan.lineage == replanned.lineage
    stored_next_plan = json.loads((tmp_path / "next-output" / "plan.json").read_text())
    assert stored_next_plan["lineage"]["parent_plan_id"] == plan.plan_id

    stored_plan_path = spec.output_directory / "plan.json"
    tampered_plan = json.loads(stored_plan_path.read_text())
    tampered_plan["guidance"].append("unapproved addition")
    stored_plan_path.write_text(json.dumps(tampered_plan), encoding="utf-8")
    with pytest.raises(automation.AutomationError) as tampered:
        automation.review_workflow_output(spec.output_directory)
    assert tampered.value.code == "review.plan_digest"


def test_resume_requires_a_real_checkpoint(tmp_path) -> None:
    plan = automation.plan_workflow(_spec(tmp_path))

    with pytest.raises(automation.AutomationError) as error:
        automation.resume_workflow(plan, approval_plan_id=plan.plan_id)

    assert error.value.code == "resume.unavailable"


def test_resume_continues_a_persisted_checkpoint(tmp_path) -> None:
    project = phasesmith.RietveldProject(
        _request(),
        RietveldOptions(
            limits=RefinementLimits(max_iterations=1, max_evaluations=100),
            estimate_covariance=False,
        ),
    )
    project.refine()
    assert project.checkpoint is not None
    source = project.save(tmp_path / "checkpoint-project")
    spec = automation.WorkflowSpec(
        "resume-test",
        source,
        tmp_path / "resumed",
        RefinementLimits(50, 500, None, 20),
        automation.WorkflowOutputs(write_csv=False, save_project=True),
    )
    plan = automation.plan_workflow(spec)

    resumed = automation.resume_workflow(plan, approval_plan_id=plan.plan_id)

    assert plan.can_resume is True
    assert resumed.result.metrics.rwp < 1.0e-5
    assert (spec.output_directory / "resume-result.json").is_file()
    assert resumed.saved_project is not None and resumed.saved_project.is_dir()


def test_raw_inspection_reports_facts_and_unknown_physics(tmp_path) -> None:
    pattern_path = tmp_path / "pattern.xy"
    pattern_path.write_text("10 2\n11 3\n12 5\n", encoding="utf-8")
    cif_path = tmp_path / "phase.cif"
    cif_path.write_text(P1_CIF, encoding="utf-8")

    pattern = automation.inspect_powder_file(pattern_path)
    structure = automation.inspect_cif_file(cif_path)

    assert pattern["coordinate"]["sample_count"] == 3
    assert pattern["coordinate"]["unit"] == "unknown"
    assert "coordinate unit" in pattern["unknown_physics"]
    assert "radiation and wavelength" in pattern["unknown_physics"]
    assert structure["structure"]["site_count"] == 2
    assert structure["source"]["sha256"]


def test_strict_json_loaders_reject_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "spec.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")

    with pytest.raises(automation.AutomationError) as error:
        automation.load_workflow_spec(path)

    assert error.value.code == "json.duplicate_key"

    path.write_text('{"schema":"a","value":NaN}', encoding="utf-8")
    with pytest.raises(automation.AutomationError) as nonfinite:
        automation.load_workflow_spec(path)
    assert nonfinite.value.code == "json.nonfinite"


def test_cli_plans_json_and_returns_structured_approval_errors(tmp_path, capsys) -> None:
    project_path = _saved_project(tmp_path)
    spec_path = tmp_path / "workflow.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema": automation.WORKFLOW_SPEC_SCHEMA,
                "workflow_id": "cli-test",
                "project_path": str(project_path),
                "output_directory": str(tmp_path / "cli-output"),
                "limits": {
                    "max_iterations": 50,
                    "max_evaluations": 500,
                    "max_runtime_seconds": None,
                    "max_consecutive_rejections": 20,
                },
                "outputs": {"write_csv": False, "save_project": False},
            }
        ),
        encoding="utf-8",
    )

    assert cli_main(["plan", str(spec_path)]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["status"] == "ready_for_approval"
    assert len(planned["plan_id"]) == 64

    proposal_path = tmp_path / "proposal.json"
    proposal = _proposal_record(automation.plan_workflow(automation.load_workflow_spec(spec_path)))
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    assert cli_main(["lint-recipe", str(spec_path), str(proposal_path)]) == 0
    lint = json.loads(capsys.readouterr().out)
    assert lint["valid_contract"] is True

    assert cli_main(["run", str(spec_path), "--approve", "wrong"]) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["schema"] == automation.AUTOMATION_ERROR_SCHEMA
    assert failure["code"] == "approval.required"

    assert cli_main(["run", str(spec_path)]) == 2
    arguments = json.loads(capsys.readouterr().err)
    assert arguments["code"] == "cli.arguments"
