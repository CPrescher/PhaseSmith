"""Run a complete offline PhaseSmith advisor/review cycle on synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import phasesmith
from phasesmith import automation
from phasesmith.refinement import RefinementLimits, rietveld
from phasesmith.refinement.rietveld import (
    RietveldOptions,
    RietveldParameterSelection,
)

P1_CIF = """
data_advisor_demo
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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _project(root: Path) -> Path:
    x = np.linspace(15.0, 100.0, 1_701)
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406,
        2.0e-4,
        -1.0e-4,
        2.0e-4,
        1.5e-3,
        3.0e-3,
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
        phase_id="advisor-demo",
        selection=selection,
        scale=0.75,
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(1.5406),
    )
    truth = rietveld.calculate(
        initial.pattern,
        replace(experiment, zero_shift_deg=0.035),
        (replace(initial.phases[0], scale=1.0),),
    )
    request = replace(initial, pattern=phasesmith.PowderPattern(x, observed_y=truth.y))
    project = phasesmith.RietveldProject(
        request,
        RietveldOptions(
            limits=RefinementLimits(max_iterations=50, max_evaluations=500),
            estimate_covariance=False,
        ),
    )
    return project.save(root / "project")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="new directory for the complete demo")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise SystemExit(f"refusing to replace existing demo directory: {root}")
    root.mkdir(parents=True)

    spec = automation.WorkflowSpec(
        "advisor-demo",
        _project(root),
        root / "run-one",
        RefinementLimits(50, 500, None, 20),
        automation.WorkflowOutputs(write_csv=True, save_project=True),
    )
    plan = automation.plan_workflow(spec)
    automation.write_workflow_plan(plan, root / "plan-one.json")
    packet = automation.advisor_packet(plan)
    _write_json(root / "advisor-packet-one.json", packet)

    maximum = plan.to_record()["authorization"]["maximum_selection"]
    scale_only = dict(maximum)
    scale_only["instrument_parameters"] = []
    prompt = "Establish scale before releasing the authorized position nuisance."
    proposal_record = {
        "schema": automation.RECIPE_PROPOSAL_SCHEMA,
        "plan_id": plan.plan_id,
        "proposal_id": "offline-example-v1",
        "provenance": {
            "kind": "software",
            "name": "PhaseSmith offline example",
            "provider": None,
            "version": phasesmith.__version__,
            "client": "examples/automation/run_advisor_cycle.py",
            "advisor_packet_id": packet["packet_id"],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request_id": None,
        },
        "assumptions": ["The synthetic project contains the complete physical model."],
        "recipe": {
            "name": "scale-then-position",
            "stages": [
                {
                    "name": "scale",
                    "selection": scale_only,
                    "rationale": ["Establish scale before a correlated position term."],
                },
                {
                    "name": "position",
                    "selection": maximum,
                    "rationale": ["Release the caller-authorized zero shift."],
                },
            ],
        },
    }
    _write_json(root / "proposal-one.json", proposal_record)
    lint = automation.lint_recipe_proposal(proposal_record, plan)
    _write_json(root / "lint-one.json", lint)
    proposal = automation.parse_recipe_proposal(proposal_record, plan)
    result = automation.run_workflow(
        plan,
        approval_plan_id=plan.plan_id,
        proposal=proposal,
    )
    review = automation.review_workflow_output(result.output_directory)
    _write_json(root / "review-one.json", review)

    next_plan, review_packet = automation.prepare_review_packet(
        result.output_directory,
        output_directory=root / "run-two",
    )
    automation.write_workflow_plan(next_plan, root / "plan-two.json")
    _write_json(root / "review-packet-two.json", review_packet)
    _write_json(
        root / "summary.json",
        {
            "first_plan_id": plan.plan_id,
            "first_review_id": review["review_id"],
            "next_plan_id": next_plan.plan_id,
            "next_advisor_packet_id": review_packet["next_advisor_packet"]["packet_id"],
            "completed": result.workflow.completed,
        },
    )
    print(root / "summary.json")


if __name__ == "__main__":
    main()
