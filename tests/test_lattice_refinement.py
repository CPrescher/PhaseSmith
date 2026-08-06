from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest
import rietveld
from rietveld.refinement import (
    CwLatticeReflectionDomain,
    CwStructuralReflectionDomain,
    FixedConstraint,
    LatticeParameterBounds,
    LatticeParameterization,
    cw_lattice_geometry,
    lebail,
    tof_lattice_geometry,
)


def cyclic_group(generator: np.ndarray, order: int) -> rietveld.SpaceGroup:
    operations = []
    current = np.eye(3, dtype=np.int64)
    for _ in range(order):
        operations.append(rietveld.SymmetryOperation(current, (0, 0, 0)))
        current = generator @ current
    return rietveld.SpaceGroup(operations)


def orthorhombic_group() -> rietveld.SpaceGroup:
    return rietveld.SpaceGroup(
        [
            rietveld.SymmetryOperation(np.diag(signs), (0, 0, 0))
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
    )


def cubic_group() -> rietveld.SpaceGroup:
    operations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if round(np.linalg.det(matrix)) == 1:
                operations.append(rietveld.SymmetryOperation(matrix, (0, 0, 0)))
    return rietveld.SpaceGroup(operations)


def lattice_cases() -> tuple[tuple[rietveld.SpaceGroup, rietveld.UnitCell, tuple[str, ...]], ...]:
    monoclinic_a = cyclic_group(np.diag([1, -1, -1]), 2)
    tetragonal = cyclic_group(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64), 4)
    trigonal_hexagonal = cyclic_group(
        np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64), 3
    )
    hexagonal = cyclic_group(np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64), 6)
    trigonal_rhombohedral = cyclic_group(
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int64), 3
    )
    return (
        (
            rietveld.SpaceGroup.p1(),
            rietveld.UnitCell(4.1, 5.2, 6.3, 77.0, 83.0, 72.0),
            ("a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg"),
        ),
        (
            monoclinic_a,
            rietveld.UnitCell(4.0, 5.0, 6.0, 103.0, 90.0, 90.0),
            ("a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg"),
        ),
        (
            orthorhombic_group(),
            rietveld.UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0),
            ("a_angstrom", "b_angstrom", "c_angstrom"),
        ),
        (
            tetragonal,
            rietveld.UnitCell(4.0, 4.0, 6.0, 90.0, 90.0, 90.0),
            ("a_angstrom", "c_angstrom"),
        ),
        (
            trigonal_hexagonal,
            rietveld.UnitCell(4.0, 4.0, 6.0, 90.0, 90.0, 120.0),
            ("a_angstrom", "c_angstrom"),
        ),
        (
            trigonal_rhombohedral,
            rietveld.UnitCell(5.0, 5.0, 5.0, 75.0, 75.0, 75.0),
            ("a_angstrom", "alpha_deg"),
        ),
        (
            hexagonal,
            rietveld.UnitCell(4.0, 4.0, 6.0, 90.0, 90.0, 120.0),
            ("a_angstrom", "c_angstrom"),
        ),
        (
            cubic_group(),
            rietveld.UnitCell(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
            ("a_angstrom",),
        ),
    )


@pytest.mark.parametrize(("group", "cell", "names"), lattice_cases())
def test_parameterization_matches_crystal_system_and_analytical_jacobian(
    group: rietveld.SpaceGroup,
    cell: rietveld.UnitCell,
    names: tuple[str, ...],
) -> None:
    parameterization = LatticeParameterization(group, cell)
    assert parameterization.parameter_names == names
    assert parameterization.parameter_count == group.metric_constraints.independent_parameter_count
    assert parameterization.to_cell(parameterization.reference_values) == cell
    jacobian = parameterization.cell_parameter_jacobian(parameterization.reference_values)
    for parameter in range(parameterization.parameter_count):
        step = 1.0e-6
        direction = np.zeros(parameterization.parameter_count)
        direction[parameter] = step
        plus = np.asarray(
            parameterization.to_cell(parameterization.reference_values + direction).as_tuple()
        )
        minus = np.asarray(
            parameterization.to_cell(parameterization.reference_values - direction).as_tuple()
        )
        np.testing.assert_allclose(
            jacobian[:, parameter], (plus - minus) / (2.0 * step), rtol=0.0, atol=3.0e-9
        )


def test_guarded_domain_preserves_intensities_by_stable_family_id() -> None:
    group, cell, _ = lattice_cases()[3]
    parameterization = LatticeParameterization(group, cell)
    bounds = LatticeParameterBounds.around(parameterization, relative_length=0.1)
    domain = CwLatticeReflectionDomain(
        group,
        parameterization,
        bounds,
        wavelength_angstrom=1.5406,
        visible_two_theta_min_deg=20.0,
        visible_two_theta_max_deg=70.0,
        initial_intensity=0.25,
    )
    initial = domain.generate(cell)
    assert initial.reflections.reflection_count > int(np.count_nonzero(initial.visible))
    assert initial.added_reflection_ids == initial.reflections.reflection_ids
    assert initial.preserved_reflection_count == 0
    seeded = rietveld.ReflectionBatch(
        initial.reflections.reflection_ids,
        initial.reflections.hkl,
        initial.reflections.d_spacing_angstrom,
        initial.reflections.two_theta_deg,
        np.arange(1, initial.reflections.reflection_count + 1, dtype=np.float64),
    )
    moved_values = parameterization.reference_values * np.array([1.08, 0.92])
    moved = domain.generate(parameterization.to_cell(moved_values), seeded)
    previous = dict(zip(seeded.reflection_ids, seeded.integrated_intensity, strict=True))
    for reflection_id, intensity in zip(
        moved.reflections.reflection_ids, moved.reflections.integrated_intensity, strict=True
    ):
        assert intensity == previous.get(reflection_id, domain.initial_intensity)
    assert moved.preserved_reflection_count == len(
        set(seeded.reflection_ids) & set(moved.reflections.reflection_ids)
    )
    assert not moved.visible.flags.writeable


def test_guarded_domain_contains_every_visible_family_across_bounded_cells() -> None:
    group, cell, _ = lattice_cases()[0]
    parameterization = LatticeParameterization(group, cell)
    bounds = LatticeParameterBounds.around(
        parameterization, relative_length=0.04, angle_delta_deg=4.0
    )
    domain = CwLatticeReflectionDomain(
        group,
        parameterization,
        bounds,
        wavelength_angstrom=1.5406,
        visible_two_theta_min_deg=20.0,
        visible_two_theta_max_deg=90.0,
    )
    guarded_ids = set(domain.generate(cell).reflections.reflection_ids)
    generator = rietveld.PreparedReflectionGenerator(group)
    rng = np.random.default_rng(20260806)
    samples = [bounds.lower, bounds.upper]
    samples.extend(rng.uniform(bounds.lower, bounds.upper) for _ in range(24))
    for values in samples:
        generated = generator.generate(
            parameterization.to_cell(values),
            rietveld.CwTwoThetaRange(20.0, 90.0, 1.5406),
        )
        assert set(generated.reflection_ids) <= guarded_ids


def test_structural_guarded_domain_preserves_family_ids_and_multiplicities() -> None:
    group, cell, _ = lattice_cases()[3]
    parameterization = LatticeParameterization(group, cell)
    bounds = LatticeParameterBounds.around(parameterization, relative_length=0.05)
    domain = CwStructuralReflectionDomain(
        group,
        parameterization,
        bounds,
        1.5406,
        20.0,
        90.0,
    )
    initial = domain.generate(cell)
    assert initial.reflections.reflection_count > int(np.count_nonzero(initial.visible))
    assert np.all(initial.reflections.multiplicity > 0)
    assert initial.added_reflection_ids == initial.reflections.reflection_ids
    moved = domain.generate(parameterization.to_cell([4.1, 5.9]), initial.reflections)
    assert moved.preserved_reflection_count == len(
        set(initial.reflections.reflection_ids) & set(moved.reflections.reflection_ids)
    )
    assert moved.d_spacing_angstrom.shape == (moved.reflections.reflection_count,)
    assert moved.two_theta_deg.shape == (moved.reflections.reflection_count,)
    assert not moved.visible.flags.writeable


def test_lattice_bounds_reject_infinite_or_incompatible_domains() -> None:
    parameterization = LatticeParameterization(
        rietveld.SpaceGroup.p1(),
        rietveld.UnitCell(4.1, 5.2, 6.3, 77.0, 83.0, 72.0),
    )
    with pytest.raises(ValueError, match="finite"):
        LatticeParameterBounds(
            parameterization,
            np.full(parameterization.parameter_count, -np.inf),
            np.full(parameterization.parameter_count, np.inf),
        )
    with pytest.raises(ValueError, match="reference"):
        LatticeParameterBounds(
            parameterization,
            parameterization.reference_values + 1.0,
            parameterization.reference_values + 2.0,
        )


@pytest.mark.parametrize(("group", "cell", "names"), lattice_cases())
def test_cw_and_tof_lattice_derivatives_match_centered_differences(
    group: rietveld.SpaceGroup,
    cell: rietveld.UnitCell,
    names: tuple[str, ...],
) -> None:
    del names
    parameterization = LatticeParameterization(group, cell)
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.int64)
    tof_instrument = rietveld.TofInstrument(
        2.0,
        1000.0,
        0.5,
        0.2,
        1.0,
        0.02,
        0.01,
        0.005,
        10.0,
        1.0,
        0.2,
        0.1,
        2.0,
        0.3,
        1.0,
    )
    cw = cw_lattice_geometry(parameterization, cell, hkl, 1.0)
    tof = tof_lattice_geometry(parameterization, cell, hkl, tof_instrument)
    for parameter in range(parameterization.parameter_count):
        step = 1.0e-6
        direction = np.zeros(parameterization.parameter_count)
        direction[parameter] = step
        plus_cell = parameterization.to_cell(parameterization.reference_values + direction)
        minus_cell = parameterization.to_cell(parameterization.reference_values - direction)
        plus_cw = cw_lattice_geometry(parameterization, plus_cell, hkl, 1.0)
        minus_cw = cw_lattice_geometry(parameterization, minus_cell, hkl, 1.0)
        plus_tof = tof_lattice_geometry(parameterization, plus_cell, hkl, tof_instrument)
        minus_tof = tof_lattice_geometry(parameterization, minus_cell, hkl, tof_instrument)
        np.testing.assert_allclose(
            cw.d_d_spacing_d_parameters[:, parameter],
            (plus_cw.d_spacing_angstrom - minus_cw.d_spacing_angstrom) / (2.0 * step),
            rtol=2.0e-8,
            atol=2.0e-9,
        )
        np.testing.assert_allclose(
            cw.d_coordinate_d_parameters[:, parameter],
            (plus_cw.coordinate - minus_cw.coordinate) / (2.0 * step),
            rtol=3.0e-8,
            atol=3.0e-8,
        )
        np.testing.assert_allclose(
            tof.d_coordinate_d_parameters[:, parameter],
            (plus_tof.coordinate - minus_tof.coordinate) / (2.0 * step),
            rtol=3.0e-8,
            atol=2.0e-5,
        )


def lattice_lebail_phase(cell: rietveld.UnitCell) -> lebail.LeBailPhase:
    group, _, _ = lattice_cases()[3]
    structure = rietveld.CrystalStructure(
        "lattice-test",
        "Lattice test",
        cell,
        group,
    )
    parameterization = LatticeParameterization(group, cell)
    bounds = LatticeParameterBounds.around(parameterization, relative_length=0.04)
    phase = lebail.LeBailPhase.from_structure(
        structure,
        phase_id="alpha",
        wavelength_angstrom=1.5406,
        two_theta_min_deg=20.0,
        two_theta_max_deg=90.0,
        initial_intensity=1.0,
        lattice_bounds=bounds,
    )
    intensities = 2.0 + np.arange(phase.reflections.reflection_count, dtype=np.float64) % 7
    return replace(
        phase,
        reflections=rietveld.ReflectionBatch(
            phase.reflections.reflection_ids,
            phase.reflections.hkl,
            phase.reflections.d_spacing_angstrom,
            phase.reflections.two_theta_deg,
            intensities,
        ),
    )


def test_lebail_lattice_pattern_columns_match_full_centered_differences() -> None:
    phase = lattice_lebail_phase(rietveld.UnitCell(4.0, 4.0, 6.0, 90.0, 90.0, 90.0))
    instrument = rietveld.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    pattern = rietveld.PowderPattern(np.linspace(20.0, 90.0, 7_001))
    baseline = rietveld.calculate_pattern(pattern, instrument, (phase,))
    parameters = lebail.build_parameter_set(instrument, (phase,), lattice_parameters=True)
    analytical = lebail._parameter_columns(baseline, parameters, instrument, (phase,))
    for column, spec in enumerate(parameters.specs):
        step = 1.0e-6
        plus_instrument, plus_phases = lebail._apply_parameter_values(
            instrument, (phase,), {spec.key: spec.value + step}
        )
        minus_instrument, minus_phases = lebail._apply_parameter_values(
            instrument, (phase,), {spec.key: spec.value - step}
        )
        plus = rietveld.calculate_pattern(pattern, plus_instrument, plus_phases).y
        minus = rietveld.calculate_pattern(pattern, minus_instrument, minus_phases).y
        finite_difference = (plus - minus) / (2.0 * step)
        scale = max(float(np.max(np.abs(finite_difference))), 1.0)
        # The local derivative is analytical; this end-to-end difference also
        # contains subtraction error from tall, narrow sampled peaks.
        assert float(np.max(np.abs(analytical[:, column] - finite_difference))) / scale < 2.0e-6


def test_lattice_refinement_recovers_a_cubic_like_tetragonal_axis() -> None:
    starting = lattice_lebail_phase(rietveld.UnitCell(3.995, 3.995, 6.0, 90.0, 90.0, 90.0))
    domain = starting.reflection_domain
    assert domain is not None
    true_values = domain.parameterization.values_from_cell(starting.structure.cell).copy()
    true_values[0] = 4.0
    true_cell = domain.parameterization.to_cell(true_values)
    generated = domain.generate(true_cell, starting.reflections)
    truth = replace(
        starting,
        structure=replace(starting.structure, cell=true_cell),
        reflections=generated.reflections,
    )
    instrument = rietveld.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    x = np.linspace(20.0, 90.0, 7_001)
    calculated = rietveld.calculate_pattern(rietveld.PowderPattern(x), instrument, (truth,))
    pattern = rietveld.PowderPattern(x, observed_y=calculated.y)
    parameters = lebail.build_parameter_set(instrument, (starting,), lattice_parameters=True)
    constraints = (FixedConstraint(parameters.keys[1], parameters.specs[1].value),)
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument, (starting,), parameters, constraints),
        lebail.LeBailOptions(
            max_iterations=30,
            min_iterations=2,
            intensity_tolerance=1.0e-7,
            max_scaled_parameter_step=0.05,
        ),
    )
    assert isinstance(result.phases[0], lebail.LeBailPhase)
    assert result.phases[0].structure.cell.a_angstrom == pytest.approx(4.0, abs=2.0e-6)
    assert result.phases[0].structure.cell.b_angstrom == pytest.approx(4.0, abs=2.0e-6)
    assert result.metrics.rwp < 2.0e-5


def test_guard_only_reflections_keep_their_intensity_when_unobserved() -> None:
    phase = lattice_lebail_phase(rietveld.UnitCell(4.0, 4.0, 6.0, 90.0, 90.0, 90.0))
    guard_only = ~phase.visible_reflection_mask
    assert np.any(guard_only)
    instrument = rietveld.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    x = np.linspace(20.0, 90.0, 7_001)
    calculated = rietveld.calculate_pattern(rietveld.PowderPattern(x), instrument, (phase,))
    result = lebail.iterate_once(
        lebail.LeBailInput(
            rietveld.PowderPattern(x, observed_y=calculated.y),
            instrument,
            (phase,),
        )
    )
    final = np.asarray([item.integrated_intensity for item in result.intensities])
    np.testing.assert_array_equal(
        final[guard_only], phase.reflections.integrated_intensity[guard_only]
    )
