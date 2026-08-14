from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith.refinement import (
    AffineConstraint,
    FixedConstraint,
    RefinementLimits,
    RietveldOptions,
    RietveldParameterSelection,
    RietveldRecipe,
    RietveldStage,
    TerminationReason,
    intelligent_rietveld_recipe,
    review_rietveld_input,
    rietveld,
    run_rietveld_recipe,
)

P1_CIF = """
data_workflow
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


def shifted_request() -> rietveld.RietveldInput:
    x = np.linspace(15.0, 100.0, 3_401)
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    starting_experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument)
    selection = RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
        instrument_parameters=("zero_shift_deg",),
    )
    initial = rietveld.RietveldInput.from_cif(
        phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
        starting_experiment,
        P1_CIF,
        phase_id="workflow",
        selection=selection,
        scale=0.72,
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(1.5406),
    )
    truth_experiment = replace(starting_experiment, zero_shift_deg=0.035)
    truth_phase = replace(initial.phases[0], scale=1.0)
    truth = rietveld.calculate(initial.pattern, truth_experiment, (truth_phase,))
    return replace(
        initial,
        pattern=phasesmith.PowderPattern(x, observed_y=truth.y),
    )


def test_readiness_report_discloses_active_conversion_and_model_choices() -> None:
    request = shifted_request()

    report = review_rietveld_input(request)

    codes = [item.code for item in report.diagnostics]
    assert "structure.source" in codes
    assert "structure.symmetry" in codes
    assert "experiment.radiation" in codes
    assert "experiment.geometry_missing" in codes
    assert "phase.scattering" in codes
    assert "phase.intensity_correction_geometry_unconfirmed" in codes
    assert report.has_warnings is True
    assert report.has_errors is False
    assert report.to_record()["diagnostics"][0]["phase_id"] == "workflow"
    assert phasesmith.RietveldProject(request).review_readiness() == report


def test_readiness_report_flags_neutral_correction_and_missing_provenance() -> None:
    request = shifted_request()
    structure = replace(request.phases[0].structure, source=None)
    phase = replace(
        request.phases[0],
        structure=structure,
        intensity_correction=phasesmith.NeutralIntegratedIntensityCorrection(),
    )
    request = replace(request, phases=(phase,))

    diagnostics = review_rietveld_input(request).diagnostics

    warnings = {item.code for item in diagnostics if item.severity == "warning"}
    assert "structure.source_missing" in warnings
    assert "phase.neutral_intensity_correction" in warnings


def test_readiness_report_flags_probe_and_experiment_mismatches() -> None:
    request = shifted_request()
    phase = replace(
        request.phases[0],
        scattering=phasesmith.NeutronNuclear(),
        intensity_correction=phasesmith.TimeOfFlightNeutronLorentz(90.0),
    )
    request = replace(request, phases=(phase,))

    codes = [item.code for item in review_rietveld_input(request).diagnostics]

    assert "phase.scattering_probe_mismatch" in codes
    assert "phase.intensity_correction_probe_mismatch" in codes
    assert "phase.intensity_correction_experiment_mismatch" in codes

    phase = replace(
        request.phases[0],
        scattering=phasesmith.XrayNonResonant(),
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(1.0),
    )
    request = replace(request, phases=(phase,))
    codes = [item.code for item in review_rietveld_input(request).diagnostics]
    assert "phase.intensity_correction_wavelength_mismatch" in codes

    experiment = replace(
        request.experiment,
        geometry=phasesmith.DebyeScherrerGeometry(240.0),
    )
    request = replace(request, experiment=experiment)
    codes = [item.code for item in review_rietveld_input(request).diagnostics]
    assert "phase.intensity_correction_geometry_mismatch" in codes


def test_readiness_report_flags_risky_joint_parameter_selection() -> None:
    request = shifted_request()
    geometry = phasesmith.BraggBrentanoGeometry(240.0)
    experiment = replace(request.experiment, geometry=geometry)
    selection = replace(
        request.selection,
        occupancy=True,
        instrument_parameters=("zero_shift_deg", "sample_displacement_mm"),
    )
    parameters = rietveld.build_parameter_set(
        request.phases,
        request.lattice_domains,
        selection,
        experiment=experiment,
    )
    request = replace(
        request,
        experiment=experiment,
        selection=selection,
        parameters=parameters,
    )

    codes = [item.code for item in review_rietveld_input(request).diagnostics]

    assert "selection.scale_occupancy_correlation" in codes
    assert "selection.zero_displacement_correlation" in codes


def test_intelligent_recipe_is_cumulative_advice_outside_solver() -> None:
    request = shifted_request()

    recipe = intelligent_rietveld_recipe(request)

    assert recipe.mode == "intelligent"
    assert [stage.name for stage in recipe.stages] == ["scale_background", "positions"]
    assert recipe.stages[0].selection.instrument_parameters == ()
    assert recipe.stages[1].selection.instrument_parameters == ("zero_shift_deg",)
    assert "advisory workflow orchestration" in recipe.planner_notes[0]
    assert recipe.to_record()["stages"][1]["rationale"]


def test_intelligent_recipe_places_debye_scherrer_displacements_in_position_stage() -> None:
    request = shifted_request()
    experiment = replace(
        request.experiment,
        geometry=phasesmith.DebyeScherrerGeometry(650.0),
    )
    selected = replace(
        request.selection,
        instrument_parameters=("displace_x_micrometre", "displace_y_micrometre"),
    )
    request = replace(
        request,
        experiment=experiment,
        selection=selected,
        parameters=rietveld.build_parameter_set(
            request.phases,
            request.lattice_domains,
            selected,
            experiment=experiment,
        ),
    )

    recipe = intelligent_rietveld_recipe(request)

    positions = next(stage for stage in recipe.stages if stage.name == "positions")
    assert positions.selection.instrument_parameters == (
        "displace_x_micrometre",
        "displace_y_micrometre",
    )
    assert any("Debye-Scherrer X/Y" in note for note in recipe.planner_notes)


def test_intelligent_recipe_skips_empty_preparatory_stages() -> None:
    request = shifted_request()
    coordinates_only = replace(
        request.selection,
        phase_scale=False,
        coordinates=True,
        instrument_parameters=(),
    )
    request = replace(
        request,
        parameters=rietveld.build_parameter_set(
            request.phases,
            request.lattice_domains,
            coordinates_only,
            experiment=request.experiment,
        ),
        selection=coordinates_only,
    )

    recipe = intelligent_rietveld_recipe(request)

    assert [stage.name for stage in recipe.stages] == ["structure"]
    assert recipe.stages[0].selection.coordinates is True


def test_intelligent_recipe_refines_scale_then_shift_and_reports_stages() -> None:
    request = shifted_request()
    options = RietveldOptions(
        limits=RefinementLimits(max_iterations=50, max_evaluations=500),
        estimate_covariance=False,
    )

    workflow = run_rietveld_recipe(
        request,
        intelligent_rietveld_recipe(request),
        options=options,
    )

    assert workflow.completed is True
    assert len(workflow.stages) == 2
    assert workflow.stages[1].starting_rwp == workflow.stages[0].result.metrics.rwp
    assert workflow.final_result.metrics.rwp < 1.0e-6
    assert workflow.final_result.experiment.zero_shift_deg == pytest.approx(0.035, abs=2.0e-7)
    record = workflow.to_record()
    assert record["completed"] is True
    assert record["stages"][1]["final_rwp"] < record["stages"][1]["starting_rwp"]


def test_recipe_rejects_parameters_outside_authorized_maximum() -> None:
    request = shifted_request()
    unauthorized = replace(request.selection, coordinates=True)
    recipe = RietveldRecipe(
        "unauthorized",
        (RietveldStage("coordinates", unauthorized, ("Deliberately invalid test stage.",)),),
    )

    with pytest.raises(ValueError, match="outside the input's maximum selection"):
        run_rietveld_recipe(request, recipe)


def test_recipe_restores_constraints_that_first_become_active_in_a_later_stage() -> None:
    request = shifted_request()
    zero_key = next(
        spec.key for spec in request.parameters.specs if spec.key.name == "zero_shift_deg"
    )
    request = replace(
        request,
        constraints=(FixedConstraint(zero_key, 0.0),),
    )

    workflow = run_rietveld_recipe(
        request,
        intelligent_rietveld_recipe(request),
        options=RietveldOptions(
            limits=RefinementLimits(max_iterations=50, max_evaluations=500),
            estimate_covariance=False,
        ),
    )

    assert workflow.completed is True
    assert workflow.final_result.experiment.zero_shift_deg == 0.0


def test_recipe_validates_later_stage_constraint_dependencies_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = shifted_request()
    zero_key = next(
        spec.key for spec in request.parameters.specs if spec.key.name == "zero_shift_deg"
    )
    scale_key = next(spec.key for spec in request.parameters.specs if spec.key.name == "scale")
    request = replace(
        request,
        constraints=(AffineConstraint(zero_key, scale_key, 0.0, 0.0),),
    )
    scale_only = replace(request.selection, instrument_parameters=())
    zero_only = replace(request.selection, phase_scale=False)
    recipe = RietveldRecipe(
        "invalid-later-dependency",
        (
            RietveldStage("scale", scale_only, ("Establish scale.",)),
            RietveldStage("zero", zero_only, ("Deliberately omit the source.",)),
        ),
    )

    def unexpected_calculation(*args: object, **kwargs: object) -> None:
        raise AssertionError("numerical work started before recipe validation")

    monkeypatch.setattr(rietveld, "calculate", unexpected_calculation)

    with pytest.raises(ValueError, match="without dependencies"):
        run_rietveld_recipe(request, recipe)


def test_zero_shift_uses_a_physical_optimization_scale() -> None:
    request = shifted_request()
    zero = next(spec for spec in request.parameters.specs if spec.key.name == "zero_shift_deg")

    assert zero.value == 0.0
    assert zero.scale == 0.05


def test_project_exposes_advice_and_optional_intelligent_execution() -> None:
    request = shifted_request()
    project = phasesmith.RietveldProject(
        request,
        RietveldOptions(
            limits=RefinementLimits(max_iterations=50, max_evaluations=500),
            estimate_covariance=False,
        ),
    )

    proposal = project.propose_intelligent_recipe()
    assert project.last_result is None

    workflow = project.refine_intelligently()

    assert workflow.recipe == proposal
    assert workflow.completed is True
    assert project.last_workflow == workflow
    assert project.last_result == workflow.final_result
    assert project.input.experiment.zero_shift_deg == pytest.approx(0.035, abs=2.0e-7)


def test_project_does_not_promote_a_stage_rejected_by_recipe_policy() -> None:
    request = shifted_request()
    original_experiment = request.experiment
    recipe = RietveldRecipe(
        "reject-convergence",
        (
            RietveldStage(
                "scale",
                replace(request.selection, instrument_parameters=()),
                ("Exercise an explicit rejection policy.",),
                accepted_terminations=(TerminationReason.CANCELLED,),
            ),
        ),
    )
    project = phasesmith.RietveldProject(
        request,
        RietveldOptions(
            limits=RefinementLimits(max_iterations=50, max_evaluations=500),
            estimate_covariance=False,
        ),
    )

    workflow = project.refine_recipe(recipe)

    assert workflow.completed is False
    assert workflow.last_accepted_stage is None
    assert project.input.experiment == original_experiment
    assert project.last_result == workflow.final_result


def test_project_retains_maximum_authorization_after_a_later_stage_is_rejected() -> None:
    request = shifted_request()
    original_selection = request.selection
    zero_key = next(
        spec.key for spec in request.parameters.specs if spec.key.name == "zero_shift_deg"
    )
    constraint = FixedConstraint(zero_key, 0.0)
    request = replace(
        request,
        constraints=(constraint,),
    )
    scale_only = replace(original_selection, instrument_parameters=())
    recipe = RietveldRecipe(
        "reject-second-stage",
        (
            RietveldStage("scale", scale_only, ("Establish the phase scale.",)),
            RietveldStage(
                "positions",
                original_selection,
                ("Exercise later-stage rejection.",),
                accepted_terminations=(TerminationReason.CANCELLED,),
            ),
        ),
    )
    project = phasesmith.RietveldProject(
        request,
        RietveldOptions(
            limits=RefinementLimits(max_iterations=50, max_evaluations=500),
            estimate_covariance=False,
        ),
    )

    workflow = project.refine_recipe(recipe)

    assert workflow.completed is False
    assert workflow.last_accepted_stage is workflow.stages[0]
    assert project.input.selection == original_selection
    assert project.input.parameters.keys == request.parameters.keys
    assert project.input.constraints == (constraint,)
    assert [stage.name for stage in project.propose_intelligent_recipe().stages] == [
        "scale_background",
        "positions",
    ]

    project.accept_result()

    assert project.input.selection == original_selection
    assert project.input.parameters.keys == request.parameters.keys
    assert project.input.constraints == (constraint,)
