from __future__ import annotations

import itertools

import numpy as np
import pytest
import rietveld
from rietveld.refinement import (
    CwLatticeReflectionDomain,
    LatticeParameterBounds,
    LatticeParameterization,
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
