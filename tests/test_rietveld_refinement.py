from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import phasesmith
import pytest
from phasesmith.refinement import (
    AffineConstraint,
    AmorphousBackground,
    AmorphousPeak,
    ChebyshevBackground,
    CheckpointCallbackError,
    PointBackground,
    PolynomialBackground,
)
from phasesmith.refinement import rietveld as structural_refinement

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


def experiment() -> phasesmith.ConstantWavelengthExperiment:
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    return phasesmith.ConstantWavelengthExperiment.x_ray(instrument)


def component_experiment() -> phasesmith.ConstantWavelengthExperiment:
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.54056, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    return phasesmith.ConstantWavelengthExperiment.x_ray_components(
        instrument,
        phasesmith.WavelengthComponents.doublet(1.54056, 1.54439, 0.5),
    )


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
        phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
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
        pattern=phasesmith.PowderPattern(x, observed_y=calculated.y),
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


def test_u_iso_selection_skips_fixed_anisotropic_sites() -> None:
    request = request_from_cif(selection())
    phase = request.phases[0]
    first = replace(
        phase.structure.sites[0],
        u_iso_angstrom2=None,
        anisotropic_displacement=phasesmith.AnisotropicDisplacement(
            (0.020, 0.013, 0.027, 0.002, 0.001, 0.003), "U_cif"
        ),
    )
    phase = replace(
        phase,
        structure=replace(phase.structure, sites=(first, phase.structure.sites[1])),
    )
    parameters = structural_refinement.build_parameter_set(
        (phase,), request.lattice_domains, selection(u_iso=True)
    )
    labels = {spec.key.label for spec in parameters.specs}
    assert "site[alpha/Si1].u_iso_angstrom2" not in labels
    assert "site[alpha/O1].u_iso_angstrom2" in labels
    result = structural_refinement.calculate(request.pattern, request.experiment, (phase,))
    assert np.isfinite(result.y).all()


def test_combined_structural_calculation_sums_profiles_and_background_once() -> None:
    request = request_from_cif(selection())
    first = request.phases[0]
    second = replace(first, phase_id="beta", scale=0.4)
    background = np.full(request.pattern.x.size, 0.3)
    pattern = phasesmith.PowderPattern(
        request.pattern.x, observed_y=request.pattern.observed_y, background=background
    )
    combined = structural_refinement.calculate(pattern, request.experiment, (first, second))
    individual = tuple(
        phasesmith.calculate_structural_pattern(pattern, request.experiment, phase)
        for phase in (first, second)
    )
    np.testing.assert_allclose(
        combined.profile_y,
        individual[0].profile_y + individual[1].profile_y,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(combined.y, combined.profile_y + background, atol=0.0)


def test_parallel_structural_calculation_is_bitwise_deterministic() -> None:
    request = request_from_cif(selection())
    first = request.phases[0]
    second = replace(first, phase_id="beta", scale=0.4)
    phases = (first, second)
    serial = structural_refinement.calculate(
        request.pattern,
        request.experiment,
        phases,
        execution=phasesmith.ExecutionPolicy(threads=1),
    )
    parallel = structural_refinement.calculate(
        request.pattern,
        request.experiment,
        phases,
        execution=phasesmith.ExecutionPolicy(threads=2),
    )
    np.testing.assert_array_equal(parallel.y, serial.y)
    np.testing.assert_array_equal(parallel.profile_y, serial.profile_y)
    for parallel_phase, serial_phase in zip(
        parallel.phase_calculations,
        serial.phase_calculations,
        strict=True,
    ):
        np.testing.assert_array_equal(parallel_phase.y, serial_phase.y)
        np.testing.assert_array_equal(
            parallel_phase.derivatives.global_jacobian,
            serial_phase.derivatives.global_jacobian,
        )


def test_fixed_components_from_cif_generates_exact_visible_union() -> None:
    x = np.linspace(15.0, 100.0, 8_501)
    selected = selection(phase_scale=True)
    request = structural_refinement.RietveldInput.from_cif(
        phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
        component_experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selected,
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(1.54056),
    )
    generator = phasesmith.PreparedReflectionGenerator(request.phases[0].structure.space_group)
    expected = set()
    for wavelength in (1.54056, 1.54439):
        generated = generator.generate(
            request.phases[0].structure.cell,
            phasesmith.CwTwoThetaRange(float(x[0]), float(x[-1]), wavelength),
        )
        expected.update(generated.reflection_ids)
    assert set(request.phases[0].reflections.reflection_ids) == expected
    prepared = phasesmith.PreparedStructuralPattern(
        request.pattern, request.experiment, request.phases[0]
    )
    assert prepared.uses_native_fused_path
    calculation = structural_refinement.calculate(
        request.pattern, request.experiment, request.phases
    )
    assert np.isfinite(calculation.y).all()


def test_fixed_component_phase_scale_refines_and_reports_component_rows() -> None:
    x = np.linspace(15.0, 100.0, 8_501)
    selected = selection(phase_scale=True)
    base = structural_refinement.RietveldInput.from_cif(
        phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
        component_experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selected,
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(1.54056),
    )
    truth = structural_refinement.calculate(base.pattern, base.experiment, base.phases)
    starting_phase = replace(base.phases[0], scale=0.55)
    request = replace(
        base,
        pattern=phasesmith.PowderPattern(x, observed_y=truth.y),
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,), (None,), selected, experiment=base.experiment
        ),
    )
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(estimate_covariance=False),
    )
    assert result.phases[0].scale == pytest.approx(1.0, rel=2.0e-8)
    assert result.metrics.rwp < 1.0e-8
    report = phasesmith.rietveld_result_record(result)
    phase_record = report["phases"][0]
    count = result.phases[0].reflections.reflection_count
    assert phase_record["reflection_count"] == count
    assert phase_record["component_reflection_count"] == 2 * count
    assert {item["component_index"] for item in phase_record["reflections"]} == {0, 1}
    assert phase_record["reflections"][0]["hkl"] == phase_record["reflections"][count]["hkl"]


def test_fixed_component_refinement_rejects_unsupported_parameter_families() -> None:
    x = np.linspace(15.0, 100.0, 8_501)
    pattern = phasesmith.PowderPattern(x, observed_y=np.zeros_like(x))
    with pytest.raises(ValueError, match="do not yet support lattice refinement"):
        structural_refinement.RietveldInput.from_cif(
            pattern,
            component_experiment(),
            P1_CIF,
            phase_id="alpha",
            selection=selection(lattice=True),
        )
    fixed = structural_refinement.RietveldInput.from_cif(
        pattern,
        component_experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selection(),
    )
    with pytest.raises(ValueError, match="do not support wavelength refinement"):
        structural_refinement.build_parameter_set(
            fixed.phases,
            fixed.lattice_domains,
            replace(selection(), instrument_parameters=("wavelength_angstrom",)),
            experiment=fixed.experiment,
        )


def test_special_position_coordinate_selection_uses_only_allowed_tangent_space() -> None:
    inversion = phasesmith.SymmetryOperation(-np.eye(3, dtype=np.int64), (0, 0, 0))
    group = phasesmith.SpaceGroup([phasesmith.SymmetryOperation.identity(), inversion])
    structure = phasesmith.CrystalStructure(
        "special",
        "Special position",
        phasesmith.UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0),
        group,
        (
            phasesmith.AtomSite("origin", "X1", "Si", "Si", (0.0, 0.0, 0.0)),
            phasesmith.AtomSite(
                "general",
                "X2",
                "O",
                "O",
                (0.13, 0.24, 0.35),
            ),
        ),
    )
    generated = phasesmith.PreparedReflectionGenerator(group).generate(
        structure.cell,
        phasesmith.CwTwoThetaRange(20.0, 80.0, experiment().radiation.wavelength_angstrom),
    )
    phase = phasesmith.RietveldPhase(
        "alpha",
        "Special",
        structure,
        phasesmith.StructuralReflectionBatch.from_generated(generated),
        phasesmith.XrayNonResonant(),
        phasesmith.NeutralIntegratedIntensityCorrection(),
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
    operation = phasesmith.SymmetryOperation(
        -np.eye(3, dtype=np.int64),
        (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)),
    )
    group = phasesmith.SpaceGroup([phasesmith.SymmetryOperation.identity(), operation])
    structure = phasesmith.CrystalStructure(
        "fixed",
        "Fixed point",
        phasesmith.UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0),
        group,
        (phasesmith.AtomSite("center", "X1", "Si", "Si", (0.5, 0.5, 0.5)),),
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


def test_matrix_free_refinement_conditions_small_phase_scales_relatively() -> None:
    base = request_from_cif(selection(phase_scale=True))
    truth_phase = replace(base.phases[0], scale=6.0e-4)
    truth = structural_refinement.calculate(
        base.pattern,
        base.experiment,
        (truth_phase,),
    )
    starting_phase = replace(truth_phase, scale=3.0e-4)
    parameters = structural_refinement.build_parameter_set(
        (starting_phase,),
        (None,),
        base.selection,
        experiment=base.experiment,
    )
    request = replace(
        base,
        pattern=phasesmith.PowderPattern(base.pattern.x, observed_y=truth.y),
        phases=(starting_phase,),
        parameters=parameters,
    )

    assert parameters.spec(structural_refinement.phase_scale_key("alpha")).scale == 3.0e-4
    zero_parameters = structural_refinement.build_parameter_set(
        (replace(starting_phase, scale=0.0),),
        (None,),
        base.selection,
        experiment=base.experiment,
    )
    assert zero_parameters.spec(structural_refinement.phase_scale_key("alpha")).scale == 1.0
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(estimate_covariance=False),
    )
    assert result.termination_reason is structural_refinement.TerminationReason.CONVERGED
    assert result.phases[0].scale == pytest.approx(6.0e-4, rel=2.0e-8)
    assert result.metrics.rwp < 1.0e-8


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
        request.lattice_domains,
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


def test_march_dollase_lattice_chain_matches_finite_difference() -> None:
    selected = selection(lattice=True)
    base = request_from_cif(selected)
    metric = phasesmith.ReciprocalMetric(base.phases[0].structure.cell.geometry().reciprocal_metric)
    phase = replace(
        base.phases[0],
        physics=phasesmith.MarchDollasePreferredOrientation(0.78, (1.0, 2.0, 1.0), metric),
    )
    parameters = structural_refinement.build_parameter_set(
        (phase,), base.lattice_domains, selected, experiment=base.experiment
    )
    request = replace(base, phases=(phase,), parameters=parameters, selection=selected)
    options = structural_refinement.RietveldOptions()
    linearization = structural_refinement._RietveldLinearization.prepare(
        request,
        request.experiment,
        request.background,
        request.phases,
        request.lattice_domains,
        request.parameters,
        options,
        structural_refinement.RefinementRuntime(options.limits),
    )
    direction = np.zeros(len(parameters.specs))
    direction[0] = 1.0
    analytical = linearization.jvp(direction)
    transform = structural_refinement.ConstraintTransform(parameters)
    packed = transform.pack()
    step = 5.0e-7
    calculated = []
    for sign in (-1.0, 1.0):
        values = transform.unpack(packed + sign * step * direction)
        phases, _ = structural_refinement._apply_parameter_values(
            request.phases,
            request.lattice_domains,
            request.parameters,
            values,
            wavelength_angstrom=request.experiment.radiation.wavelength_angstrom,
        )
        calculated.append(
            structural_refinement.calculate(request.pattern, request.experiment, phases).y
        )
    finite_difference = (calculated[1] - calculated[0]) / (2.0 * step)
    relative_error = np.linalg.norm(analytical - finite_difference) / np.linalg.norm(
        finite_difference
    )
    assert relative_error < 3.0e-5


def test_pre_requested_cancellation_returns_the_unmodified_safe_state() -> None:
    request = request_from_cif(selection(phase_scale=True))
    token = phasesmith.CancellationToken()
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
        structure = replace(structure, cell=phasesmith.UnitCell(*cell_values))
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
        phasesmith.PowderPattern(single.pattern.x, observed_y=calculated.y),
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
        phasesmith.PowderPattern(single.pattern.x, observed_y=observed),
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
        phasesmith.PowderPattern(single.pattern.x, observed_y=observed),
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


def test_refinement_reuses_guarded_coordinate_models_across_trial_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.45)
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,),
            (None,),
            truth.selection,
        ),
    )
    original = structural_refinement._site_coordinate_model
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(structural_refinement, "_site_coordinate_model", counted)
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(estimate_covariance=False),
    )
    assert result.history
    assert calls == len(starting_phase.structure.sites)


def test_parallel_multiphase_refinement_matches_serial_history_and_derivatives() -> None:
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
    observed = structural_refinement.calculate(
        single.pattern,
        single.experiment,
        (first, second),
    ).y
    starting = (replace(first, scale=0.58), replace(second, scale=0.42))
    parameters = structural_refinement.build_parameter_set(
        starting,
        (None, None),
        single.selection,
    )
    request = structural_refinement.RietveldInput(
        phasesmith.PowderPattern(single.pattern.x, observed_y=observed),
        single.experiment,
        starting,
        (None, None),
        parameters,
        selection=single.selection,
    )
    common = dict(estimate_covariance=False)
    serial = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            execution=phasesmith.ExecutionPolicy(threads=1),
            **common,
        ),
    )
    parallel = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            execution=phasesmith.ExecutionPolicy(threads=2),
            **common,
        ),
    )
    assert parallel.history == serial.history
    assert parallel.parameters == serial.parameters
    np.testing.assert_array_equal(parallel.calculation.y, serial.calculation.y)
    np.testing.assert_array_equal(parallel.metrics.residual, serial.metrics.residual)
    np.testing.assert_array_equal(
        parallel.metrics.weighted_residual,
        serial.metrics.weighted_residual,
    )
    assert parallel.metrics.chi_square == serial.metrics.chi_square
    assert parallel.metrics.rwp == serial.metrics.rwp


def test_monochromatic_neutron_cif_request_refines_through_same_runtime() -> None:
    neutron_experiment = phasesmith.ConstantWavelengthExperiment.neutron(experiment().instrument)
    x = np.linspace(15.0, 100.0, 8_501)
    selected = selection(phase_scale=True)
    initial = structural_refinement.RietveldInput.from_cif(
        phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
        neutron_experiment,
        P1_CIF,
        phase_id="neutron-alpha",
        selection=selected,
    )
    assert type(initial.phases[0].scattering) is phasesmith.NeutronNuclear
    truth = structural_refinement.calculate(initial.pattern, neutron_experiment, initial.phases)
    starting_phase = replace(initial.phases[0], scale=0.72)
    request = replace(
        initial,
        pattern=phasesmith.PowderPattern(x, observed_y=truth.y),
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
    token = phasesmith.CancellationToken()
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


def test_project_facade_refines_stops_reports_and_resumes(tmp_path) -> None:
    truth = request_from_cif(selection(phase_scale=True))
    starting_phase = replace(truth.phases[0], scale=0.55)
    request = replace(
        truth,
        phases=(starting_phase,),
        parameters=structural_refinement.build_parameter_set(
            (starting_phase,), (None,), truth.selection
        ),
    )
    project = phasesmith.RietveldProject(request)
    result = project.refine()
    assert result.phases[0].scale == pytest.approx(1.0, rel=2.0e-8)
    json_path, csv_path = project.write_reports(
        json_path=tmp_path / "result.json",
        csv_path=tmp_path / "pattern.csv",
    )
    assert json_path is not None and csv_path is not None
    report = json.loads(json_path.read_text())
    assert report["schema"] == "phasesmith.result-report.v1"
    assert report["termination"]["reason"] == result.termination_reason.value
    assert report["phases"][0]["reflection_count"] == truth.phases[0].reflections.reflection_count
    csv_lines = csv_path.read_text().splitlines()
    assert len(csv_lines) == truth.pattern.x.size + 1
    assert "weight" in csv_lines[0].split(",")

    saved = project.save(tmp_path / "project")
    restored = phasesmith.RietveldProject.load(saved)
    assert restored.checkpoint is not None and project.checkpoint is not None
    assert restored.checkpoint.parameters == project.checkpoint.parameters
    assert restored.checkpoint.history == project.checkpoint.history
    np.testing.assert_allclose(restored.calculate().y, result.calculation.y)

    stopped = phasesmith.RietveldProject(request)

    def stop_on_start(event: object) -> None:
        if event.kind is structural_refinement.RefinementEventKind.START:
            stopped.stop("test_stop")

    cancelled = stopped.refine(logger=stop_on_start)
    assert cancelled.termination_reason is structural_refinement.TerminationReason.CANCELLED
    assert cancelled.history == ()


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


def test_default_rietveld_budget_can_cover_worst_case_iterations() -> None:
    options = structural_refinement.RietveldOptions()
    maximum_evaluations_per_iteration = (
        1 + 2 * options.max_cg_iterations + (options.max_backtracks + 1)
    )
    assert options.limits.max_evaluations >= (
        1 + options.limits.max_iterations * maximum_evaluations_per_iteration
    )


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


@pytest.mark.parametrize(
    ("name", "truth_value", "starting_value"),
    (
        ("wavelength_angstrom", 1.5406, 1.5410),
        ("zero_shift_deg", 0.025, 0.04),
        ("sample_displacement_mm", 0.18, 0.27),
    ),
)
def test_monochromatic_calibration_parameters_refine_analytically(
    name: str,
    truth_value: float,
    starting_value: float,
) -> None:
    base = request_from_cif(selection(lattice=name == "wavelength_angstrom"))

    def configured(value: float) -> phasesmith.ConstantWavelengthExperiment:
        wavelength = value if name == "wavelength_angstrom" else 1.5406
        selected_instrument = replace(base.experiment.instrument, wavelength_angstrom=wavelength)
        return phasesmith.ConstantWavelengthExperiment(
            phasesmith.MonochromaticRadiation.x_ray(wavelength),
            selected_instrument,
            zero_shift_deg=value if name == "zero_shift_deg" else 0.0,
            geometry=phasesmith.BraggBrentanoGeometry(
                240.0,
                value if name == "sample_displacement_mm" else 0.0,
            ),
        )

    truth_experiment = configured(truth_value)
    calculated = structural_refinement.calculate(
        base.pattern,
        truth_experiment,
        base.phases,
    )
    observed = replace(base.pattern, observed_y=calculated.y)
    starting_experiment = configured(starting_value)
    starting_domains = tuple(
        None
        if domain is None
        else replace(domain, wavelength_angstrom=starting_experiment.radiation.wavelength_angstrom)
        for domain in base.lattice_domains
    )
    selected = replace(selection(), instrument_parameters=(name,))
    parameters = structural_refinement.build_parameter_set(
        base.phases,
        starting_domains,
        selected,
        experiment=starting_experiment,
    )
    request = structural_refinement.RietveldInput(
        observed,
        starting_experiment,
        base.phases,
        starting_domains,
        parameters,
        selection=selected,
    )
    result = structural_refinement.refine(request)
    actual = (
        result.experiment.geometry.sample_displacement_mm
        if name == "sample_displacement_mm"
        else getattr(result.experiment, name)
        if name == "zero_shift_deg"
        else result.experiment.radiation.wavelength_angstrom
    )
    assert actual == pytest.approx(truth_value, rel=2.0e-6, abs=2.0e-8)
    assert result.metrics.rwp < 2.0e-7
    if name == "wavelength_angstrom":
        domain = result.checkpoint.lattice_domains[0]
        assert domain is not None
        assert domain.wavelength_angstrom == result.experiment.radiation.wavelength_angstrom


@pytest.mark.parametrize("kind", ("size", "microstrain", "march"))
def test_builtin_sample_physics_parameters_are_refinable(kind: str) -> None:
    base = request_from_cif(selection())
    metric = phasesmith.ReciprocalMetric(base.phases[0].structure.cell.geometry().reciprocal_metric)
    if kind == "size":
        truth_model = phasesmith.IsotropicSizeBroadening(75.0)
        starting_model = phasesmith.IsotropicSizeBroadening(62.0)
        expected = 75.0
    elif kind == "microstrain":
        truth_model = phasesmith.IsotropicMicrostrainBroadening(7.0e-4)
        starting_model = phasesmith.IsotropicMicrostrainBroadening(9.0e-4)
        expected = 7.0e-4
    else:
        truth_model = phasesmith.MarchDollasePreferredOrientation(0.82, (0, 0, 1), metric)
        starting_model = phasesmith.MarchDollasePreferredOrientation(0.9, (0, 0, 1), metric)
        expected = 0.82
    truth_phase = replace(base.phases[0], physics=truth_model)
    starting_phase = replace(base.phases[0], physics=starting_model)
    calculated = structural_refinement.calculate(base.pattern, base.experiment, (truth_phase,))
    observed = replace(base.pattern, observed_y=calculated.y)
    selected = selection(sample_physics=True)
    parameters = structural_refinement.build_parameter_set(
        (starting_phase,),
        (None,),
        selected,
        experiment=base.experiment,
    )
    request = structural_refinement.RietveldInput(
        observed,
        base.experiment,
        (starting_phase,),
        (None,),
        parameters,
        selection=selected,
    )
    result = structural_refinement.refine(request)
    model = result.phases[0].physics
    actual = (
        model.crystallite_size_nm
        if kind == "size"
        else model.rms_microstrain
        if kind == "microstrain"
        else model.march_ratio
    )
    assert actual == pytest.approx(expected, rel=3.0e-5)
    assert result.metrics.rwp < 2.0e-7


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
        phasesmith.PowderPattern(truth.pattern.x, observed_y=calculated.y),
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


@pytest.mark.parametrize(
    ("expected_background", "starting_background"),
    (
        (
            ChebyshevBackground("cheb", (2.0, 0.3, -0.2), (15.0, 100.0)),
            ChebyshevBackground("cheb", (1.7, 0.15, -0.05), (15.0, 100.0)),
        ),
        (
            PointBackground("points", (15.0, 45.0, 75.0, 100.0), (1.0, 2.0, 1.4, 2.2)),
            PointBackground("points", (15.0, 45.0, 75.0, 100.0), (0.8, 1.7, 1.2, 1.9)),
        ),
        (
            AmorphousBackground("glass", (AmorphousPeak(18.0, 56.0, 14.0),)),
            AmorphousBackground("glass", (AmorphousPeak(16.0, 55.0, 13.0),)),
        ),
    ),
)
def test_richer_background_models_refine_through_the_common_contract(
    expected_background: object,
    starting_background: object,
) -> None:
    truth = request_from_cif(selection())
    calculated = structural_refinement.calculate(
        truth.pattern,
        truth.experiment,
        truth.phases,
        background=expected_background,
    )
    selected = selection(background=True)
    parameters = structural_refinement.build_parameter_set(
        truth.phases,
        (None,),
        selected,
        experiment=truth.experiment,
        background=starting_background,
    )
    request = structural_refinement.RietveldInput(
        replace(truth.pattern, observed_y=calculated.y),
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
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    assert result.metrics.rwp < 2.0e-7


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


def test_out_of_domain_trials_are_rejected_without_losing_the_safe_state(
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
    calls = 0
    events = []

    def reject_trial(
        self: structural_refinement._RietveldLinearization,
    ) -> structural_refinement.RietveldCalculationResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(self)
        self.runtime.begin_evaluation()
        raise ValueError("derived Gaussian variance must be positive and finite")

    monkeypatch.setattr(
        structural_refinement._RietveldLinearization,
        "calculate",
        reject_trial,
    )
    result = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            max_backtracks=2,
            estimate_covariance=False,
        ),
        logger=events.append,
    )
    rejected = [
        event
        for event in events
        if event.kind is structural_refinement.RefinementEventKind.STEP_REJECTED
    ]
    assert result.termination_reason is structural_refinement.TerminationReason.STAGNATED
    assert result.history == ()
    assert result.phases == request.phases
    assert calls == 4
    assert len(rejected) == 3
    assert all("outside the numerical model domain" in event.message for event in rejected)
    assert all(
        dict(event.diagnostics)["reason"] == "derived Gaussian variance must be positive and finite"
        for event in rejected
    )


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
