from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import phasesmith
import pytest
from phasesmith import persistence
from phasesmith.refinement import (
    AffineConstraint,
    AmorphousBackground,
    AmorphousPeak,
    ChebyshevBackground,
    CompositeBackground,
    LatticeParameterBounds,
    LatticeParameterization,
    LinearConstraint,
    PointBackground,
    PolynomialBackground,
    lebail,
)
from phasesmith.refinement import rietveld as structural_refinement


def test_current_persistence_schema_matches_the_writer_version() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / f"persistence-v{persistence.FORMAT_VERSION}.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    assert schema["properties"]["format_version"]["const"] == persistence.FORMAT_VERSION


def instrument() -> phasesmith.ConstantWavelengthInstrument:
    return phasesmith.ConstantWavelengthInstrument(
        1.5406,
        2.0e-4,
        -1.0e-4,
        2.0e-4,
        1.5e-3,
        3.0e-3,
    )


def phase(intensities: np.ndarray) -> phasesmith.Phase:
    positions = np.array([39.9, 40.1])
    d_spacing = instrument().wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    physics = phasesmith.CompositePhysicsProvider(
        (
            phasesmith.IsotropicSizeBroadening(80.0),
            phasesmith.IsotropicMicrostrainBroadening(3.0e-4),
            phasesmith.IsotropicLorentzianMicrostrainBroadening(2.0e-4),
            phasesmith.MarchDollasePreferredOrientation(
                0.8,
                (0.0, 0.0, 1.0),
                phasesmith.ReciprocalMetric.orthogonal(4.0, 4.0, 4.0),
            ),
        )
    )
    return phasesmith.Phase(
        "alpha",
        "Alpha",
        phasesmith.ReflectionBatch(
            ["alpha-100", "alpha-110"],
            [[1, 0, 0], [1, 1, 0]],
            d_spacing,
            positions,
            intensities,
        ),
        scale=0.9,
        physics=physics,
    )


def structural_phase() -> phasesmith.RietveldPhase:
    structure = phasesmith.CrystalStructure(
        "structure-alpha",
        "Structural alpha",
        phasesmith.UnitCell(4.1, 4.1, 4.1, 90.0, 90.0, 90.0),
        phasesmith.SpaceGroup.p1(),
        (
            phasesmith.AtomSite("si", "Si1", "Si", "Si", (0.1, 0.2, 0.3), 0.9, 0.012),
            phasesmith.AtomSite("o", "O1", "O", "O", (0.4, 0.5, 0.6), 1.0, 0.018),
        ),
    )
    return phasesmith.RietveldPhase(
        "structural-alpha",
        "Structural alpha",
        structure,
        phasesmith.StructuralReflectionBatch(
            ("1,0,0", "1,1,0"),
            [[1, 0, 0], [1, 1, 0]],
            [2, 4],
        ),
        phasesmith.XrayNonResonant(),
        phasesmith.BraggBrentanoUnpolarizedLp(instrument().wavelength_angstrom),
        scale=1.25,
        physics=phasesmith.IsotropicSizeBroadening(75.0),
        coordinate_tolerance=2.0e-10,
    )


def refinement_models() -> tuple[phasesmith.PowderPattern, phasesmith.Phase, lebail.LeBailResult]:
    x = np.linspace(38.0, 42.0, 2001)
    truth = phase(np.array([7.0, 4.0]))
    background = 0.2 + 0.01 * (x - x[0])
    calculated = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x, background=background),
        instrument(),
        (truth,),
        options=phasesmith.CalculationOptions(return_phase_components=True),
    )
    pattern = phasesmith.PowderPattern(
        x,
        observed_y=calculated.y,
        background=background,
        uncertainty=np.sqrt(np.maximum(calculated.y, 1.0)),
        mask=(x < 41.5),
    )
    starting = phase(np.ones(2))
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument(), (starting,)),
        lebail.LeBailOptions(max_iterations=20),
    )
    return pattern, starting, result


def dynamic_lebail_phase() -> lebail.LeBailPhase:
    structure = phasesmith.CrystalStructure(
        "dynamic-alpha",
        "Dynamic alpha",
        phasesmith.UnitCell(4.0, 5.0, 6.0, 78.0, 82.0, 73.0),
        phasesmith.SpaceGroup.p1(),
    )
    parameterization = LatticeParameterization(structure.space_group, structure.cell)
    bounds = LatticeParameterBounds.around(
        parameterization, relative_length=0.02, angle_delta_deg=2.0
    )
    return lebail.LeBailPhase.from_structure(
        structure,
        phase_id="dynamic-alpha",
        wavelength_angstrom=instrument().wavelength_angstrom,
        two_theta_min_deg=20.0,
        two_theta_max_deg=80.0,
        lattice_bounds=bounds,
    )


def test_full_bundle_round_trips_models_results_arrays_and_resume(tmp_path) -> None:
    pattern, starting, result = refinement_models()
    parameters = lebail.build_parameter_set(instrument(), (starting,), reflection_positions=True)
    constraints = (AffineConstraint(parameters.keys[1], parameters.keys[0], 1.0, 0.2),)
    bundle = persistence.PersistenceBundle(
        pattern=pattern,
        instrument=instrument(),
        experiment=phasesmith.ConstantWavelengthExperiment.neutron(instrument()),
        fcj_geometry=phasesmith.FcjGeometry(0.01, 0.02),
        wavelength_components=phasesmith.WavelengthComponents.doublet(1.5406, 1.5444, 0.5),
        phases=(starting,),
        calculation_options=phasesmith.CalculationOptions(
            support_fwhm=15.0,
            return_phase_components=True,
            execution=phasesmith.ExecutionPolicy(threads=2),
        ),
        calculation_result=result.calculation,
        parameters=parameters,
        lebail_options=lebail.LeBailOptions(
            max_iterations=20,
            execution=phasesmith.ExecutionPolicy(threads=3),
        ),
        lebail_checkpoint=result.checkpoint,
        lebail_result=result,
        constraints=constraints,
        metadata={"sample": "synthetic", "temperature_k": 300.0},
    )
    destination = persistence.save_bundle(tmp_path / "project", bundle)
    restored = persistence.load_bundle(destination)

    assert restored.metadata == bundle.metadata
    assert restored.calculation_options == bundle.calculation_options
    assert restored.lebail_options == bundle.lebail_options
    assert restored.parameters == parameters
    assert restored.constraints == constraints
    assert isinstance(restored.instrument, phasesmith.ConstantWavelengthInstrument)
    assert restored.instrument == instrument()
    assert restored.experiment == phasesmith.ConstantWavelengthExperiment.neutron(instrument())
    assert restored.fcj_geometry == phasesmith.FcjGeometry(0.01, 0.02)
    assert restored.wavelength_components is not None
    np.testing.assert_array_equal(
        restored.wavelength_components.wavelengths_angstrom,
        [1.5406, 1.5444],
    )
    assert restored.pattern is not None
    np.testing.assert_array_equal(restored.pattern.x, pattern.x)
    np.testing.assert_array_equal(restored.pattern.mask, pattern.mask)
    assert isinstance(restored.phases[0].physics, phasesmith.CompositePhysicsProvider)
    assert len(restored.phases[0].physics.providers) == 4
    assert restored.lebail_result is not None
    assert restored.calculation_result is not None
    np.testing.assert_array_equal(restored.calculation_result.y, result.calculation.y)
    np.testing.assert_array_equal(restored.lebail_result.calculation.y, result.calculation.y)
    np.testing.assert_array_equal(
        restored.lebail_result.calculation.derivatives.local.values,
        result.calculation.derivatives.local.values,
    )
    assert restored.lebail_result.history == result.history
    assert restored.lebail_result.intensities == result.intensities
    assert restored.lebail_checkpoint is not None
    reconstructed_input = restored.to_lebail_input()
    assert reconstructed_input.parameters == parameters
    assert reconstructed_input.constraints == constraints
    np.testing.assert_array_equal(reconstructed_input.pattern.x, pattern.x)
    resumed = lebail.refine(
        lebail.LeBailInput(pattern, instrument(), (starting,)),
        lebail.LeBailOptions(max_iterations=result.checkpoint.completed_iterations + 1),
        checkpoint=restored.lebail_checkpoint,
    )
    assert resumed.history[: len(result.history)] == result.history

    manifest = json.loads((destination / persistence.MANIFEST_NAME).read_text())
    assert manifest["format_version"] == persistence.FORMAT_VERSION
    with np.load(destination / persistence.ARCHIVE_NAME, allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[name].dtype != object for name in archive.files)


def test_parameter_change_history_round_trips(tmp_path) -> None:
    x = np.linspace(39.0, 41.0, 4001)
    truth = phase(np.array([8.0, 0.0]))
    truth = phasesmith.Phase(
        truth.phase_id,
        truth.name,
        phasesmith.ReflectionBatch(
            ["alpha-100"],
            [[1, 0, 0]],
            truth.reflections.d_spacing_angstrom[:1],
            [40.0],
            [8.0],
        ),
    )
    calculated = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), (truth,))
    starting = phasesmith.Phase(
        truth.phase_id,
        truth.name,
        phasesmith.ReflectionBatch(
            ["alpha-100"],
            [[1, 0, 0]],
            truth.reflections.d_spacing_angstrom,
            [39.985],
            [8.0],
        ),
    )
    pattern = phasesmith.PowderPattern(x, observed_y=calculated.y)
    parameters = lebail.build_parameter_set(instrument(), (starting,), reflection_positions=True)
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument(), (starting,), parameters),
        lebail.LeBailOptions(max_iterations=10, max_scaled_parameter_step=1.0),
    )
    assert any(record.parameter_changes for record in result.history)

    destination = persistence.save_bundle(
        tmp_path / "changes",
        persistence.PersistenceBundle(lebail_result=result),
    )
    restored = persistence.load_bundle(destination)
    assert restored.lebail_result is not None
    assert restored.lebail_result.history == result.history


def test_lebail_residual_background_round_trips_with_checkpoint(tmp_path) -> None:
    x = np.linspace(39.0, 41.0, 2_001)
    truth = (phase(np.array([8.0, 5.0])),)
    fixed = np.full_like(x, 0.4)
    residual = ChebyshevBackground("lebail-residual", (0.2, -0.05), (39.0, 41.0))
    blank = phasesmith.PowderPattern(x, background=fixed)
    calculated = phasesmith.calculate_pattern(blank, instrument(), truth)
    pattern = phasesmith.PowderPattern(
        x,
        observed_y=calculated.y + residual.calculate(x),
        background=fixed,
    )
    starting = (phase(np.ones(2)),)
    request = lebail.LeBailInput(pattern, instrument(), starting).with_refinable_background(
        ChebyshevBackground("lebail-residual", (0.0, 0.0), (39.0, 41.0))
    )
    result = lebail.refine(request, lebail.LeBailOptions(max_iterations=5))
    assert result.background is not None

    destination = persistence.save_bundle(
        tmp_path / "lebail-background",
        persistence.PersistenceBundle(
            pattern=pattern,
            instrument=instrument(),
            phases=starting,
            parameters=request.parameters,
            lebail_background=request.background,
            lebail_checkpoint=result.checkpoint,
            lebail_result=result,
        ),
    )
    restored = persistence.load_bundle(destination)
    assert restored.lebail_background == request.background
    assert restored.to_lebail_input().background == request.background
    assert restored.lebail_checkpoint is not None
    assert restored.lebail_checkpoint.background == result.background
    assert restored.lebail_result is not None
    assert restored.lebail_result.background == result.background


def test_multi_source_linear_constraint_round_trips(tmp_path) -> None:
    parameters = lebail.build_parameter_set(
        instrument(),
        (phase(np.ones(2)),),
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2"),
    )
    constraint = LinearConstraint(
        parameters.keys[2],
        ((parameters.keys[0], 0.5), (parameters.keys[1], 0.2)),
        4.0e-5,
    )
    path = persistence.save_bundle(
        tmp_path / "linear-constraint",
        persistence.PersistenceBundle(
            parameters=parameters,
            constraints=(constraint,),
        ),
    )
    restored = persistence.load_bundle(path)
    assert restored.constraints == (constraint,)


def test_dynamic_lebail_phase_domain_round_trips_and_resumes(tmp_path) -> None:
    starting = dynamic_lebail_phase()
    x = np.linspace(20.0, 80.0, 6_001)
    calculated = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x), instrument(), (starting,)
    )
    pattern = phasesmith.PowderPattern(x, observed_y=calculated.y)
    parameters = lebail.build_parameter_set(instrument(), (starting,), lattice_parameters=True)
    request = lebail.LeBailInput(pattern, instrument(), (starting,), parameters)
    first = lebail.iterate_once(request)
    path = persistence.save_bundle(
        tmp_path / "dynamic",
        persistence.PersistenceBundle(
            pattern=pattern,
            instrument=instrument(),
            phases=(starting,),
            parameters=parameters,
            lebail_checkpoint=first.checkpoint,
            lebail_result=first,
        ),
    )

    restored = persistence.load_bundle(path)
    assert isinstance(restored.phases[0], lebail.LeBailPhase)
    restored_phase = restored.phases[0]
    assert restored_phase.structure == starting.structure
    assert restored_phase.reflections_generated
    assert restored_phase.reflection_domain is not None
    assert starting.reflection_domain is not None
    np.testing.assert_array_equal(
        restored_phase.reflection_domain.bounds.lower,
        starting.reflection_domain.bounds.lower,
    )
    np.testing.assert_array_equal(
        restored_phase.reflection_domain.bounds.upper,
        starting.reflection_domain.bounds.upper,
    )
    assert restored.lebail_checkpoint is not None
    assert isinstance(restored.lebail_checkpoint.phases[0], lebail.LeBailPhase)
    resumed = lebail.iterate_once(restored.to_lebail_input(), checkpoint=restored.lebail_checkpoint)
    assert resumed.checkpoint.completed_iterations == 2


def test_structural_phase_round_trips_separately_from_lebail_phases(tmp_path) -> None:
    structural = structural_phase()
    path = persistence.save_bundle(
        tmp_path / "structural",
        persistence.PersistenceBundle(
            phases=(phase(np.ones(2)),),
            rietveld_phases=(structural,),
        ),
    )
    restored = persistence.load_bundle(path)
    assert len(restored.phases) == 1
    assert len(restored.rietveld_phases) == 1
    actual = restored.rietveld_phases[0]
    assert actual.phase_id == structural.phase_id
    assert actual.structure == structural.structure
    assert actual.reflections.reflection_ids == structural.reflections.reflection_ids
    np.testing.assert_array_equal(actual.reflections.hkl, structural.reflections.hkl)
    np.testing.assert_array_equal(
        actual.reflections.multiplicity, structural.reflections.multiplicity
    )
    assert type(actual.scattering) is phasesmith.XrayNonResonant
    assert actual.intensity_correction == structural.intensity_correction
    assert actual.scale == structural.scale
    assert actual.coordinate_tolerance == structural.coordinate_tolerance
    assert actual.physics == structural.physics

    calculation_pattern = phasesmith.PowderPattern(np.linspace(10.0, 80.0, 7_001))
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument())
    expected = phasesmith.calculate_structural_pattern(calculation_pattern, experiment, structural)
    observed = phasesmith.calculate_structural_pattern(calculation_pattern, experiment, actual)
    np.testing.assert_array_equal(observed.y, expected.y)
    np.testing.assert_array_equal(
        observed.reflections.integrated_intensity,
        expected.reflections.integrated_intensity,
    )


def test_fixed_dispersion_and_polarized_lp_round_trip(tmp_path) -> None:
    structural = replace(
        structural_phase(),
        scattering=phasesmith.XrayFixedDispersion(
            {"Si": 0.212567 + 0.245496j, "O": 0.049384 + 0.032232j}
        ),
        intensity_correction=phasesmith.BraggBrentanoPolarizedLp(1.5406, 0.7),
    )
    destination = persistence.save_bundle(
        tmp_path / "dispersion-polarization",
        persistence.PersistenceBundle(rietveld_phases=(structural,)),
    )
    restored = persistence.load_bundle(destination).rietveld_phases[0]
    assert restored.scattering == structural.scattering
    assert restored.intensity_correction == structural.intensity_correction
    manifest = json.loads((destination / persistence.MANIFEST_NAME).read_text())
    assert manifest["format_version"] == persistence.FORMAT_VERSION


def test_neutron_lorentz_round_trip(tmp_path) -> None:
    structural = replace(
        structural_phase(),
        scattering=phasesmith.NeutronNuclear(),
        intensity_correction=phasesmith.ConstantWavelengthNeutronLorentz(1.5406),
    )
    destination = persistence.save_bundle(
        tmp_path / "neutron-lorentz",
        persistence.PersistenceBundle(rietveld_phases=(structural,)),
    )

    restored = persistence.load_bundle(destination).rietveld_phases[0]

    assert restored.scattering == structural.scattering
    assert restored.intensity_correction == structural.intensity_correction


def test_rietveld_checkpoint_domain_and_options_round_trip_and_resume(tmp_path) -> None:
    x = np.linspace(10.0, 80.0, 7_001)
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument())
    truth_phase = structural_phase()
    calculated = structural_refinement.calculate(
        phasesmith.PowderPattern(x), experiment, (truth_phase,)
    )
    starting_phase = replace(truth_phase, scale=0.55)
    selection = structural_refinement.RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
    )
    parameters = structural_refinement.build_parameter_set((starting_phase,), (None,), selection)
    pattern = phasesmith.PowderPattern(x, observed_y=calculated.y)
    request = structural_refinement.RietveldInput(
        pattern,
        experiment,
        (starting_phase,),
        (None,),
        parameters,
        selection=selection,
    )
    first = structural_refinement.refine(
        request,
        structural_refinement.RietveldOptions(
            limits=structural_refinement.RefinementLimits(
                max_iterations=1,
                max_evaluations=100,
            ),
            max_scaled_parameter_step=0.1,
            estimate_covariance=False,
        ),
    )
    resume_options = structural_refinement.RietveldOptions(
        limits=structural_refinement.RefinementLimits(
            max_iterations=20,
            max_evaluations=500,
        ),
        max_scaled_parameter_step=0.1,
        estimate_covariance=False,
        execution=phasesmith.ExecutionPolicy(threads=2),
    )
    destination = persistence.save_bundle(
        tmp_path / "rietveld-restart",
        persistence.PersistenceBundle(
            pattern=pattern,
            experiment=experiment,
            rietveld_phases=(starting_phase,),
            rietveld_domains=(None,),
            rietveld_selection=selection,
            rietveld_options=resume_options,
            rietveld_checkpoint=first.checkpoint,
            rietveld_background=PolynomialBackground("restart", (0.0,)),
            parameters=parameters,
        ),
    )
    restored = persistence.load_bundle(destination)
    assert restored.rietveld_selection == selection
    assert restored.rietveld_options == resume_options
    assert restored.rietveld_background == PolynomialBackground("restart", (0.0,))
    assert restored.rietveld_checkpoint is not None
    assert (
        restored.rietveld_checkpoint.completed_iterations == first.checkpoint.completed_iterations
    )
    assert restored.rietveld_checkpoint.parameters == first.checkpoint.parameters
    assert restored.rietveld_checkpoint.history == first.checkpoint.history
    assert restored.rietveld_checkpoint.phases[0].scale == first.checkpoint.phases[0].scale
    assert restored.rietveld_checkpoint.experiment == experiment
    resumed = structural_refinement.refine(
        restored.to_rietveld_input(),
        restored.rietveld_options,
        checkpoint=restored.rietveld_checkpoint,
    )
    assert resumed.phases[0].scale == pytest.approx(truth_phase.scale, rel=2.0e-8)
    assert resumed.metrics.rwp < 1.0e-8


def test_guarded_structural_reflection_domain_round_trips(tmp_path) -> None:
    phase = structural_phase()
    parameterization = LatticeParameterization(phase.structure.space_group, phase.structure.cell)
    bounds = LatticeParameterBounds.around(
        parameterization, relative_length=0.03, angle_delta_deg=3.0
    )
    domain = structural_refinement.CwStructuralReflectionDomain(
        phase.structure.space_group,
        parameterization,
        bounds,
        instrument().wavelength_angstrom,
        10.0,
        80.0,
    )
    path = persistence.save_bundle(
        tmp_path / "structural-domain",
        persistence.PersistenceBundle(
            rietveld_phases=(phase,),
            rietveld_domains=(domain,),
        ),
    )
    restored = persistence.load_bundle(path)
    actual = restored.rietveld_domains[0]
    assert actual is not None
    assert actual.parameterization.parameter_names == parameterization.parameter_names
    np.testing.assert_array_equal(actual.bounds.lower, bounds.lower)
    np.testing.assert_array_equal(actual.bounds.upper, bounds.upper)
    assert actual.wavelength_angstrom == domain.wavelength_angstrom
    assert actual.guard_scale == domain.guard_scale


def test_version_one_bundle_migrates_with_no_structural_phases(tmp_path) -> None:
    path = persistence.save_bundle(
        tmp_path / "version-one",
        persistence.PersistenceBundle(phases=(phase(np.ones(2)),)),
    )
    manifest_path = path / persistence.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 1
    manifest["bundle"].pop("rietveld_phases")
    manifest_path.write_text(json.dumps(manifest))

    restored = persistence.load_bundle(path)
    assert len(restored.phases) == 1
    assert restored.rietveld_phases == ()


def test_version_two_generic_bundle_remains_loadable(tmp_path) -> None:
    path = persistence.save_bundle(
        tmp_path / "version-two",
        persistence.PersistenceBundle(phases=(phase(np.ones(2)),)),
    )
    manifest_path = path / persistence.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 2
    manifest_path.write_text(json.dumps(manifest))

    restored = persistence.load_bundle(path)
    assert len(restored.phases) == 1
    assert type(restored.phases[0]) is phasesmith.Phase


def test_version_eight_rietveld_options_gain_serial_execution_default(tmp_path) -> None:
    path = persistence.save_bundle(
        tmp_path / "version-eight",
        persistence.PersistenceBundle(
            rietveld_options=structural_refinement.RietveldOptions(
                estimate_covariance=False,
                execution=phasesmith.ExecutionPolicy(threads=3),
            )
        ),
    )
    manifest_path = path / persistence.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 8
    del manifest["bundle"]["rietveld_options"]["execution"]
    manifest_path.write_text(json.dumps(manifest))

    restored = persistence.load_bundle(path)
    assert restored.rietveld_options is not None
    assert restored.rietveld_options.execution == phasesmith.ExecutionPolicy(threads=1)


def test_version_nine_calculation_and_lebail_gain_serial_execution_default(tmp_path) -> None:
    path = persistence.save_bundle(
        tmp_path / "version-nine",
        persistence.PersistenceBundle(
            calculation_options=phasesmith.CalculationOptions(
                execution=phasesmith.ExecutionPolicy(threads=2)
            ),
            lebail_options=lebail.LeBailOptions(execution=phasesmith.ExecutionPolicy(threads=3)),
        ),
    )
    manifest_path = path / persistence.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 9
    del manifest["bundle"]["calculation_options"]["execution"]
    del manifest["bundle"]["lebail_options"]["execution"]
    manifest_path.write_text(json.dumps(manifest))

    restored = persistence.load_bundle(path)
    assert restored.calculation_options is not None
    assert restored.lebail_options is not None
    assert restored.calculation_options.execution == phasesmith.ExecutionPolicy(threads=1)
    assert restored.lebail_options.execution == phasesmith.ExecutionPolicy(threads=1)


def test_undefined_zero_pattern_ratios_round_trip_as_explicit_nulls(tmp_path) -> None:
    x = np.linspace(38.0, 42.0, 101)
    pattern = phasesmith.PowderPattern(x, observed_y=np.zeros_like(x))
    result = lebail.iterate_once(lebail.LeBailInput(pattern, instrument(), (phase(np.ones(2)),)))
    assert np.isposinf(result.metrics.rwp)

    destination = persistence.save_bundle(
        tmp_path / "undefined-ratios",
        persistence.PersistenceBundle(lebail_result=result),
    )
    manifest = json.loads((destination / persistence.MANIFEST_NAME).read_text())
    assert manifest["bundle"]["lebail_result"]["metrics"]["rwp"] is None
    restored = persistence.load_bundle(destination)
    assert restored.lebail_result is not None
    assert np.isposinf(restored.lebail_result.metrics.rwp)
    assert np.isposinf(restored.lebail_result.checkpoint.previous_rwp)


def test_tof_instrument_and_disabled_infinite_size_round_trip(tmp_path) -> None:
    tof = phasesmith.TofInstrument(
        -0.7,
        5084.0,
        -2.6,
        0.0,
        5.0,
        0.03,
        0.001,
        0.0,
        1.0,
        15.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    disabled = phase(np.ones(2))
    disabled = phasesmith.Phase(
        disabled.phase_id,
        disabled.name,
        disabled.reflections,
        physics=phasesmith.IsotropicSizeBroadening(np.inf),
    )
    destination = persistence.save_bundle(
        tmp_path / "tof",
        persistence.PersistenceBundle(instrument=tof, phases=(disabled,)),
    )
    restored = persistence.load_bundle(destination)
    assert restored.instrument == tof
    assert isinstance(restored.phases[0].physics, phasesmith.IsotropicSizeBroadening)
    assert np.isinf(restored.phases[0].physics.crystallite_size_nm)


@dataclass(frozen=True)
class CustomProvider:
    amplitude: float
    descriptor = phasesmith.ProviderDescriptor("example.custom", "2")

    def evaluate(self, context: phasesmith.PhysicsContext) -> phasesmith.PhysicsContribution:
        count = context.reflections.reflection_count
        zeros = np.zeros(count)
        return phasesmith.PhysicsContribution(
            gaussian_variance_deg2=zeros,
            lorentzian_fwhm_deg=np.full(count, self.amplitude),
            intensity_multiplier=np.ones(count),
            d_gaussian_variance_d_position=zeros,
            d_lorentzian_fwhm_d_position=zeros,
            d_intensity_multiplier_d_position=zeros,
            parameter_names=("amplitude",),
            d_gaussian_variance_d_parameters=zeros[None, :],
            d_lorentzian_fwhm_d_parameters=np.ones((1, count)),
            d_intensity_multiplier_d_parameters=zeros[None, :],
        )


class CustomCodec:
    provider_id = "example.custom"

    def encode(self, provider: object) -> dict[str, object] | None:
        if isinstance(provider, CustomProvider):
            return {"amplitude": provider.amplitude}
        return None

    def decode(self, configuration: dict[str, object], provider_version: str) -> CustomProvider:
        assert provider_version == "2"
        return CustomProvider(float(configuration["amplitude"]))


def test_custom_provider_requires_and_round_trips_through_explicit_codec(tmp_path) -> None:
    original = phase(np.ones(2))
    custom = phasesmith.Phase(
        original.phase_id,
        original.name,
        original.reflections,
        physics=CustomProvider(0.003),
    )
    bundle = persistence.PersistenceBundle(phases=(custom,))
    with pytest.raises(TypeError, match="PhysicsProviderCodec"):
        persistence.save_bundle(tmp_path / "missing", bundle)
    path = persistence.save_bundle(tmp_path / "custom", bundle, provider_codecs=(CustomCodec(),))
    with pytest.raises(persistence.PersistenceError, match="no PhysicsProviderCodec"):
        persistence.load_bundle(path)
    restored = persistence.load_bundle(path, provider_codecs=(CustomCodec(),))
    assert restored.phases[0].physics == CustomProvider(0.003)


def test_hash_version_and_overwrite_guards_are_enforced(tmp_path) -> None:
    path = persistence.save_bundle(
        tmp_path / "guarded", persistence.PersistenceBundle(instrument=instrument())
    )
    with pytest.raises(FileExistsError):
        persistence.save_bundle(path, persistence.PersistenceBundle())

    manifest_path = path / persistence.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(persistence.PersistenceError, match="unsupported"):
        persistence.load_bundle(path)

    manifest["format_version"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(persistence.PersistenceError, match="unsupported"):
        persistence.load_bundle(path)

    manifest["format_version"] = persistence.FORMAT_VERSION
    manifest["archive"]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(persistence.PersistenceError, match="SHA-256"):
        persistence.load_bundle(path)


def test_format_five_experiment_and_background_models_round_trip(tmp_path) -> None:
    experiment = phasesmith.ConstantWavelengthExperiment(
        phasesmith.MonochromaticRadiation.x_ray(instrument().wavelength_angstrom),
        instrument(),
        zero_shift_deg=0.035,
        geometry=phasesmith.BraggBrentanoGeometry(240.0, 0.18),
        axial_geometry=phasesmith.FcjGeometry(0.013, 0.009),
    )
    backgrounds = (
        PolynomialBackground("power", (1.0, 0.2)),
        ChebyshevBackground("cheb", (1.0, -0.2), (10.0, 90.0)),
        PointBackground("points", (10.0, 40.0, 90.0), (1.0, 2.0, 1.5)),
        AmorphousBackground("glass", (AmorphousPeak(12.0, 45.0, 13.0),)),
    )
    backgrounds += (CompositeBackground("combined", backgrounds[:2]),)
    for background in backgrounds:
        destination = persistence.save_bundle(
            tmp_path / background.background_id,
            persistence.PersistenceBundle(
                experiment=experiment,
                rietveld_background=background,
            ),
        )
        restored = persistence.load_bundle(destination)
        assert restored.experiment == experiment
        assert restored.rietveld_background == background


def test_format_six_fixed_component_experiment_round_trips(tmp_path) -> None:
    components = phasesmith.WavelengthComponents.doublet(1.5406, 1.5444, 0.5)
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray_components(instrument(), components)
    destination = persistence.save_bundle(
        tmp_path / "component-experiment",
        persistence.PersistenceBundle(experiment=experiment),
    )
    restored = persistence.load_bundle(destination)
    assert restored.experiment == experiment
    manifest = json.loads((destination / persistence.MANIFEST_NAME).read_text())
    radiation = manifest["bundle"]["experiment"]["radiation"]
    assert radiation == {
        "type": "components",
        "probe": "x-ray",
        "wavelengths_angstrom": [1.5406, 1.5444],
        "relative_intensities": [1.0, 0.5],
    }


def test_format_twelve_debye_scherrer_geometry_round_trips(tmp_path) -> None:
    experiment = phasesmith.ConstantWavelengthExperiment(
        phasesmith.MonochromaticRadiation.neutron(instrument().wavelength_angstrom),
        instrument(),
        zero_shift_deg=-0.1,
        geometry=phasesmith.DebyeScherrerGeometry(650.0, 1578.8, 49.9),
    )
    destination = persistence.save_bundle(
        tmp_path / "debye-scherrer",
        persistence.PersistenceBundle(experiment=experiment),
    )
    restored = persistence.load_bundle(destination)
    assert restored.experiment == experiment
    manifest = json.loads((destination / persistence.MANIFEST_NAME).read_text())
    assert manifest["bundle"]["experiment"]["geometry"] == {
        "type": "debye_scherrer",
        "goniometer_radius_mm": 650.0,
        "displace_x_micrometre": 1578.8,
        "displace_y_micrometre": 49.9,
    }


def test_format_five_flat_monochromatic_experiment_still_loads(tmp_path) -> None:
    experiment = phasesmith.ConstantWavelengthExperiment.neutron(instrument())
    destination = persistence.save_bundle(
        tmp_path / "format-five",
        persistence.PersistenceBundle(experiment=experiment),
    )
    manifest_path = destination / persistence.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    record = manifest["bundle"]["experiment"]
    radiation = record.pop("radiation")
    record["probe"] = radiation["probe"]
    record["wavelength_angstrom"] = radiation["wavelength_angstrom"]
    manifest["format_version"] = 5
    manifest_path.write_text(json.dumps(manifest))
    restored = persistence.load_bundle(destination)
    assert restored.experiment == experiment
