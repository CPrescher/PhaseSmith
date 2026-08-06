from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import rietveld
from rietveld.refinement import (
    AffineConstraint,
    CheckpointCallbackError,
    PolynomialBackground,
)
from rietveld.refinement import rietveld as structural_refinement

P1_CIF = """
data_p1
_chemical_name_common 'P1 structural test'
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
Si1 Si 0.11 0.22 0.33 0.9 0.012
O1 O 0.41 0.52 0.63 1.0 0.018
"""


def experiment() -> rietveld.ConstantWavelengthExperiment:
    instrument = rietveld.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    return rietveld.ConstantWavelengthExperiment.x_ray(instrument)


def selection(**changes: bool) -> structural_refinement.RietveldParameterSelection:
    return replace(
        structural_refinement.RietveldParameterSelection(
            phase_scale=False,
            lattice=False,
            coordinates=False,
            occupancy=False,
            u_iso=False,
        ),
        **changes,
    )


def request_from_cif(
    selected: structural_refinement.RietveldParameterSelection,
) -> structural_refinement.RietveldInput:
    x = np.linspace(15.0, 100.0, 8_501)
    initial = structural_refinement.RietveldInput.from_cif(
        rietveld.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selected,
    )
    calculated = structural_refinement.calculate(
        initial.pattern, initial.experiment, initial.phases
    )
    return replace(
        initial,
        pattern=rietveld.PowderPattern(x, observed_y=calculated.y),
    )


def test_cif_request_builds_guarded_structural_phase_and_typed_parameters() -> None:
    request = request_from_cif(
        selection(phase_scale=True, lattice=True, coordinates=True, occupancy=True, u_iso=True)
    )
    phase = request.phases[0]
    domain = request.lattice_domains[0]
    assert domain is not None
    assert phase.structure.source is not None
    assert phase.structure.source.backend == "gemmi"
    assert phase.reflections.reflection_count > 0
    assert len(request.parameters.specs) == 1 + 6 + 2 * 3 + 2 + 2
    labels = tuple(spec.key.label for spec in request.parameters.specs)
    assert labels[0] == "phase[alpha].scale"
    assert "site[alpha/Si1].occupancy" in labels
    assert "site[alpha/O1].u_iso_angstrom2" in labels


def test_combined_structural_calculation_sums_profiles_and_background_once() -> None:
    request = request_from_cif(selection())
    first = request.phases[0]
    second = replace(first, phase_id="beta", scale=0.4)
    background = np.full(request.pattern.x.size, 0.3)
    pattern = rietveld.PowderPattern(
        request.pattern.x, observed_y=request.pattern.observed_y, background=background
    )
    combined = structural_refinement.calculate(pattern, request.experiment, (first, second))
    individual = tuple(
        rietveld.calculate_structural_pattern(pattern, request.experiment, phase)
        for phase in (first, second)
    )
    np.testing.assert_allclose(
        combined.profile_y,
        individual[0].profile_y + individual[1].profile_y,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(combined.y, combined.profile_y + background, atol=0.0)


def test_special_position_coordinate_selection_uses_only_allowed_tangent_space() -> None:
    inversion = rietveld.SymmetryOperation(-np.eye(3, dtype=np.int64), (0, 0, 0))
    group = rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), inversion])
    structure = rietveld.CrystalStructure(
        "special",
        "Special position",
        rietveld.UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0),
        group,
        (
            rietveld.AtomSite("origin", "X1", "Si", "Si", (0.0, 0.0, 0.0)),
            rietveld.AtomSite(
                "general",
                "X2",
                "O",
                "O",
                (0.13, 0.24, 0.35),
            ),
        ),
    )
    generated = rietveld.PreparedReflectionGenerator(group).generate(
        structure.cell,
        rietveld.CwTwoThetaRange(20.0, 80.0, experiment().radiation.wavelength_angstrom),
    )
    phase = rietveld.RietveldPhase(
        "alpha",
        "Special",
        structure,
        rietveld.StructuralReflectionBatch.from_generated(generated),
        rietveld.XrayNonResonant(),
        rietveld.NeutralIntegratedIntensityCorrection(),
    )
    parameters = structural_refinement.build_parameter_set(
        (phase,),
        (None,),
        selection(coordinates=True),
    )
    coordinate_keys = tuple(spec.key for spec in parameters.specs if spec.key.module == "site")
    assert all(key.owner_id != "alpha/origin" for key in coordinate_keys)
    assert tuple(key.name for key in coordinate_keys) == ("x", "y", "z")


def test_site_coordinate_model_handles_non_origin_fixed_point_exactly() -> None:
    operation = rietveld.SymmetryOperation(
        -np.eye(3, dtype=np.int64),
        (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)),
    )
    group = rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), operation])
    structure = rietveld.CrystalStructure(
        "fixed",
        "Fixed point",
        rietveld.UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0),
        group,
        (rietveld.AtomSite("center", "X1", "Si", "Si", (0.5, 0.5, 0.5)),),
    )
    model = structural_refinement._site_coordinate_model(
        "alpha", structure, structure.sites[0], 1.0e-10
    )
    assert model.special_position
    assert model.parameter_names == ()
    assert model.basis.shape == (3, 0)


def test_matrix_free_refinement_recovers_phase_scale_and_emits_checkpoints() -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.65)
    starting_parameters = structural_refinement.build_parameter_set(
        (starting_phase,), (None,), truth.selection
    )
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=starting_parameters,
    )
    events = []
    checkpoints = []
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            limits=structural_refinement.RefinementLimits(
                max_iterations=10,
                max_evaluations=200,
            ),
            min_iterations=1,
        ),
        logger=events.append,
        checkpoint_callback=checkpoints.append,
    )
    assert result.termination_reason is structural_refinement.TerminationReason.CONVERGED
    assert result.phases[0].scale == pytest.approx(1.0, rel=2.0e-8)
    assert result.metrics.rwp < 1.0e-8
    assert result.jacobian_rank == 1
    assert result.covariance is not None
    assert result.covariance.shape == (1, 1)
    assert checkpoints
    assert checkpoints[-1] == result.checkpoint
    assert events[0].kind is structural_refinement.RefinementEventKind.START
    assert events[-1].kind is structural_refinement.RefinementEventKind.TERMINATION


def test_combined_structural_products_match_finite_difference_and_adjoint() -> None:
    request = request_from_cif(
        selection(phase_scale=True, lattice=True, coordinates=True, occupancy=True, u_iso=True)
    )
    options = structural_refinement.RietveldOptions(
        limits=structural_refinement.RefinementLimits(max_evaluations=100)
    )
    runtime = structural_refinement.RefinementRuntime(options.limits)
    linearization = structural_refinement._RietveldLinearization.prepare(
        request,
        request.experiment,
        request.background,
        request.phases,
        request.parameters,
        options,
        runtime,
    )
    transform = structural_refinement.ConstraintTransform(request.parameters)
    rng = np.random.default_rng(20260807)
    direction = rng.normal(size=len(transform.free_keys))
    direction /= np.linalg.norm(direction)
    analytical = linearization.jvp(direction)
    step = 5.0e-7
    packed = transform.pack()
    calculations = []
    for sign in (-1.0, 1.0):
        values = transform.unpack(packed + sign * step * direction)
        phases, _ = structural_refinement._apply_parameter_values(
            request.phases,
            request.lattice_domains,
            request.parameters,
            values,
        )
        calculations.append(
            structural_refinement.calculate(request.pattern, request.experiment, phases).y
        )
    finite_difference = (calculations[1] - calculations[0]) / (2.0 * step)
    relative_l2_error = np.linalg.norm(analytical - finite_difference) / np.linalg.norm(
        finite_difference
    )
    # A few samples can cross an exact finite-support boundary; the full
    # directional product remains tightly converged in norm.
    assert relative_l2_error < 2.0e-5
    samples = rng.normal(size=request.pattern.x.size)
    left = float(analytical @ samples)
    right = float(direction @ linearization.vjp(samples))
    assert left == pytest.approx(right, rel=3.0e-12, abs=3.0e-9)


def test_pre_requested_cancellation_returns_the_unmodified_safe_state() -> None:
    request = request_from_cif(selection(phase_scale=True))
    token = rietveld.CancellationToken()
    token.request("test_stop")
    result = structural_refinement.refine(request, cancellation=token)
    assert result.termination_reason is structural_refinement.TerminationReason.CANCELLED
    assert result.termination_message == "test_stop"
    assert result.phases == request.phases
    assert result.parameters == request.parameters
    assert result.history == ()


@pytest.mark.parametrize("family", ["lattice", "coordinates", "occupancy", "u_iso"])
def test_matrix_free_refinement_improves_each_structural_parameter_family(
    family: str,
) -> None:
    selected = selection(**{family: True})
    truth = request_from_cif(selected)
    phase = truth.phases[0]
    structure = phase.structure
    domain = truth.lattice_domains[0]
    if family == "lattice":
        cell_values = list(structure.cell.as_tuple())
        cell_values[0] += 0.004
        structure = replace(structure, cell=rietveld.UnitCell(*cell_values))
        assert domain is not None
        phase = replace(
            phase,
            structure=structure,
            reflections=domain.generate(structure.cell).reflections,
        )
    else:
        sites = list(structure.sites)
        first = sites[0]
        if family == "coordinates":
            xyz = list(first.fractional_xyz)
            xyz[0] += 8.0e-4
            first = replace(first, fractional_xyz=tuple(xyz))
        elif family == "occupancy":
            first = replace(first, occupancy=first.occupancy - 0.015)
        else:
            first = replace(first, u_iso_angstrom2=first.u_iso_angstrom2 + 5.0e-4)
        sites[0] = first
        phase = replace(phase, structure=replace(structure, sites=tuple(sites)))
    starting_parameters = structural_refinement.build_parameter_set(
        (phase,), truth.lattice_domains, selected
    )
    request = replace(truth, phases=(phase,), parameters=starting_parameters)
    initial = structural_refinement.calculate(request.pattern, request.experiment, request.phases)
    initial_metrics = structural_refinement.evaluate_residuals(request.pattern, initial.y)
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            limits=structural_refinement.RefinementLimits(
                max_iterations=12,
                max_evaluations=800,
            ),
            max_cg_iterations=20,
        ),
    )
    assert result.history
    assert result.metrics.chi_square < initial_metrics.chi_square * 1.0e-4


def test_rank_deficient_multiphase_scales_are_reported_without_fake_covariance() -> None:
    single = request_from_cif(selection(phase_scale=True))
    first = replace(single.phases[0], phase_id="alpha", scale=0.6)
    second = replace(single.phases[0], phase_id="beta", scale=0.4)
    phases = (first, second)
    parameters = structural_refinement.build_parameter_set(
        phases,
        (None, None),
        single.selection,
    )
    calculated = structural_refinement.calculate(single.pattern, single.experiment, phases)
    request = structural_refinement.RietveldInput(
        rietveld.PowderPattern(single.pattern.x, observed_y=calculated.y),
        single.experiment,
        phases,
        (None, None),
        parameters,
        selection=single.selection,
    )
    result = structural_refinement.refine(request)
    assert result.termination_reason is structural_refinement.TerminationReason.CONVERGED
    assert result.jacobian_rank == 1
    assert result.covariance is None
    assert len(result.unresolved_correlations) == 1
    assert abs(result.unresolved_correlations[0].correlation) == pytest.approx(1.0)


def test_affine_multiphase_scale_constraint_is_applied_in_native_products() -> None:
    single = request_from_cif(selection(phase_scale=True))
    truth_phases = (
        replace(single.phases[0], phase_id="alpha", scale=0.6),
        replace(single.phases[0], phase_id="beta", scale=0.4),
    )
    observed = structural_refinement.calculate(single.pattern, single.experiment, truth_phases).y
    starting_phases = (
        replace(truth_phases[0], scale=0.48),
        replace(truth_phases[1], scale=0.32),
    )
    parameters = structural_refinement.build_parameter_set(
        starting_phases, (None, None), single.selection
    )
    constraint = AffineConstraint(
        structural_refinement.phase_scale_key("beta"),
        structural_refinement.phase_scale_key("alpha"),
        multiplier=2.0 / 3.0,
    )
    request = structural_refinement.RietveldInput(
        rietveld.PowderPattern(single.pattern.x, observed_y=observed),
        single.experiment,
        starting_phases,
        (None, None),
        parameters,
        (constraint,),
        single.selection,
    )
    result = structural_refinement.refine(request)
    assert result.phases[0].scale == pytest.approx(0.6, rel=2.0e-8)
    assert result.phases[1].scale == pytest.approx(0.4, rel=2.0e-8)
    assert result.covariance is not None
    assert result.covariance[0, 1] != 0.0


def test_distinct_multiphase_scales_recover_independently() -> None:
    single = request_from_cif(selection(phase_scale=True))
    first = replace(single.phases[0], phase_id="alpha", scale=0.7)
    second_structure = single.phases[0].structure
    second_sites = list(second_structure.sites)
    xyz = list(second_sites[0].fractional_xyz)
    xyz[0] += 0.07
    second_sites[0] = replace(second_sites[0], fractional_xyz=tuple(xyz))
    second = replace(
        single.phases[0],
        phase_id="beta",
        structure=replace(second_structure, sites=tuple(second_sites)),
        scale=0.3,
    )
    truth_phases = (first, second)
    observed = structural_refinement.calculate(single.pattern, single.experiment, truth_phases).y
    starting = (replace(first, scale=0.58), replace(second, scale=0.42))
    parameters = structural_refinement.build_parameter_set(starting, (None, None), single.selection)
    request = structural_refinement.RietveldInput(
        rietveld.PowderPattern(single.pattern.x, observed_y=observed),
        single.experiment,
        starting,
        (None, None),
        parameters,
        selection=single.selection,
    )
    result = structural_refinement.refine(request)
    assert result.phases[0].scale == pytest.approx(0.7, rel=2.0e-8)
    assert result.phases[1].scale == pytest.approx(0.3, rel=2.0e-8)
    assert result.jacobian_rank == 2


def test_monochromatic_neutron_cif_request_refines_through_same_runtime() -> None:
    neutron_experiment = rietveld.ConstantWavelengthExperiment.neutron(experiment().instrument)
    x = np.linspace(15.0, 100.0, 8_501)
    selected = selection(phase_scale=True)
    initial = structural_refinement.RietveldInput.from_cif(
        rietveld.PowderPattern(x, observed_y=np.zeros_like(x)),
        neutron_experiment,
        P1_CIF,
        phase_id="neutron-alpha",
        selection=selected,
    )
    assert type(initial.phases[0].scattering) is rietveld.NeutronNuclear
    truth = structural_refinement.calculate(initial.pattern, neutron_experiment, initial.phases)
    starting_phase = replace(initial.phases[0], scale=0.72)
    request = replace(
        initial,
        pattern=rietveld.PowderPattern(x, observed_y=truth.y),
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set((starting_phase,), (None,), selected),
    )
    result = structural_refinement.refine(request)
    assert result.phases[0].scale == pytest.approx(1.0, rel=2.0e-8)
    assert result.metrics.rwp < 1.0e-8


def test_checkpoint_resume_reproduces_continuous_accepted_history() -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.41)
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,), (None,), truth.selection
        ),
    )
    common = dict(
        min_iterations=1,
        max_scaled_parameter_step=0.15,
        estimate_covariance=False,
    )
    first = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            limits=structural_refinement.RefinementLimits(max_iterations=1, max_evaluations=100),
            **common,
        ),
    )
    resumed = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            limits=structural_refinement.RefinementLimits(max_iterations=10, max_evaluations=300),
            **common,
        ),
        checkpoint=first.checkpoint,
    )
    continuous = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            limits=structural_refinement.RefinementLimits(max_iterations=10, max_evaluations=300),
            **common,
        ),
    )
    assert resumed.history == continuous.history
    assert resumed.parameters == continuous.parameters
    assert resumed.phases[0].scale == continuous.phases[0].scale


def test_cancellation_requested_by_checkpoint_sink_returns_that_accepted_state() -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.45)
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,), (None,), truth.selection
        ),
    )
    token = rietveld.CancellationToken()
    accepted = []

    def stop_after_checkpoint(checkpoint: object) -> None:
        accepted.append(checkpoint)
        token.request("checkpoint_stop")

    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(estimate_covariance=False),
        cancellation=token,
        checkpoint_callback=stop_after_checkpoint,
    )
    assert result.termination_reason is structural_refinement.TerminationReason.CANCELLED
    assert result.termination_message == "checkpoint_stop"
    assert len(accepted) == 1
    assert result.checkpoint == accepted[0]


def test_model_evaluation_budget_returns_last_calculated_state() -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.7)
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,), (None,), truth.selection
        ),
    )
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            limits=structural_refinement.RefinementLimits(
                max_iterations=10,
                max_evaluations=1,
            )
        ),
    )
    assert result.termination_reason is structural_refinement.TerminationReason.MAX_EVALUATIONS
    assert result.evaluations == 1
    assert result.history == ()
    assert result.phases == request.phases


def test_cw_profile_parameter_refines_through_accumulation_derivative_rows() -> None:
    selected = replace(selection(), instrument_parameters=("w_deg2",))
    truth = request_from_cif(selection())
    starting_instrument = replace(
        truth.experiment.instrument,
        w_deg2=truth.experiment.instrument.w_deg2 + 4.0e-5,
    )
    starting_experiment = replace(truth.experiment, instrument=starting_instrument)
    parameters = structural_refinement.build_parameter_set(
        truth.phases,
        (None,),
        selected,
        experiment=starting_experiment,
    )
    request = structural_refinement.RietveldInput(
        truth.pattern,
        starting_experiment,
        truth.phases,
        (None,),
        parameters,
        selection=selected,
    )
    result = structural_refinement.refine(request)
    assert result.experiment.instrument.w_deg2 == pytest.approx(
        truth.experiment.instrument.w_deg2, rel=2.0e-7
    )
    assert result.metrics.rwp < 1.0e-8


def test_polynomial_background_refines_as_a_separate_typed_domain() -> None:
    truth = request_from_cif(selection())
    expected_background = PolynomialBackground("main", (2.0, 0.3, -0.2))
    calculated = structural_refinement.calculate(
        truth.pattern,
        truth.experiment,
        truth.phases,
        background=expected_background,
    )
    starting_background = PolynomialBackground("main", (1.6, 0.15, -0.05))
    selected = selection(background=True)
    parameters = structural_refinement.build_parameter_set(
        truth.phases,
        (None,),
        selected,
        background=starting_background,
    )
    request = structural_refinement.RietveldInput(
        rietveld.PowderPattern(truth.pattern.x, observed_y=calculated.y),
        truth.experiment,
        truth.phases,
        (None,),
        parameters,
        selection=selected,
        background=starting_background,
    )
    result = structural_refinement.refine(request)
    assert result.background is not None
    np.testing.assert_allclose(
        result.background.coefficients,
        expected_background.coefficients,
        rtol=0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        result.calculation.background,
        expected_background.calculate(truth.pattern.x),
        rtol=0.0,
        atol=2.0e-8,
    )


def test_logger_failure_is_isolated_from_structural_refinement() -> None:
    request = request_from_cif(selection(phase_scale=True))

    def broken_logger(event: object) -> None:
        raise OSError(f"cannot write {event!r}")

    result = structural_refinement.refine(request, logger=broken_logger)
    assert isinstance(result.logger_error, OSError)
    assert result.termination_reason is structural_refinement.TerminationReason.CONVERGED


def test_unexpected_model_failure_emits_last_safe_checkpoint_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_from_cif(selection(phase_scale=True))
    emergency = []

    def fail_calculation(self: object) -> object:
        raise RuntimeError(f"injected failure in {type(self).__name__}")

    monkeypatch.setattr(
        structural_refinement._RietveldLinearization,
        "calculate",
        fail_calculation,
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        structural_refinement.refine(request, checkpoint_callback=emergency.append)
    assert len(emergency) == 1
    assert emergency[0].completed_iterations == 0
    assert emergency[0].phases == request.phases


def test_non_improving_trials_terminate_as_stagnated_without_installing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.7)
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,), (None,), truth.selection
        ),
    )
    original = structural_refinement._RietveldLinearization.calculate
    cached = []

    def unchanged_trial(
        self: structural_refinement._RietveldLinearization,
    ) -> structural_refinement.RietveldCalculationResult:
        if not cached:
            cached.append(original(self))
        else:
            self.runtime.begin_evaluation()
        return cached[0]

    monkeypatch.setattr(
        structural_refinement._RietveldLinearization,
        "calculate",
        unchanged_trial,
    )
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            max_backtracks=2,
            estimate_covariance=False,
        ),
    )
    assert result.termination_reason is structural_refinement.TerminationReason.STAGNATED
    assert result.history == ()
    assert result.phases == request.phases


def test_accepted_checkpoint_sink_failure_is_a_typed_error() -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.7)
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,), (None,), truth.selection
        ),
    )

    def broken_checkpoint(checkpoint: object) -> None:
        raise OSError(f"cannot persist {checkpoint!r}")

    with pytest.raises(CheckpointCallbackError):
        structural_refinement.refine(
            request,
            checkpoint_callback=broken_checkpoint,
        )
