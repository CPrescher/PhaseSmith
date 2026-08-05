from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import rietveld
from rietveld.symmetry_reference import (
    reference_expand_sites,
    reference_family,
    reference_generate_d_spacing,
    reference_is_systematically_absent,
)


def inversion() -> rietveld.SymmetryOperation:
    return rietveld.SymmetryOperation(-np.eye(3, dtype=np.int64), (0, 0, 0))


def inversion_group() -> rietveld.SpaceGroup:
    return rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), inversion()])


def body_centred_group() -> rietveld.SpaceGroup:
    return rietveld.SpaceGroup(
        [
            rietveld.SymmetryOperation.identity(),
            rietveld.SymmetryOperation(np.eye(3, dtype=np.int64), ("1/2", "1/2", "1/2")),
        ]
    )


def face_centred_group() -> rietveld.SpaceGroup:
    identity = np.eye(3, dtype=np.int64)
    return rietveld.SpaceGroup(
        [
            rietveld.SymmetryOperation.identity(),
            rietveld.SymmetryOperation(identity, (0, "1/2", "1/2")),
            rietveld.SymmetryOperation(identity, ("1/2", 0, "1/2")),
            rietveld.SymmetryOperation(identity, ("1/2", "1/2", 0)),
        ]
    )


def p21_group() -> rietveld.SpaceGroup:
    screw = rietveld.SymmetryOperation([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], (0, "1/2", 0))
    return rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), screw])


def glide_group() -> rietveld.SpaceGroup:
    glide = rietveld.SymmetryOperation([[1, 0, 0], [0, -1, 0], [0, 0, 1]], (0, 0, "1/2"))
    return rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), glide])


def p31_group() -> rietveld.SpaceGroup:
    rotation = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    return rietveld.SpaceGroup(
        [
            rietveld.SymmetryOperation.identity(),
            rietveld.SymmetryOperation(rotation, (0, 0, "1/3")),
            rietveld.SymmetryOperation(rotation @ rotation, (0, 0, "2/3")),
        ]
    )


def cyclic_rotation_group(rotation: np.ndarray, order: int) -> rietveld.SpaceGroup:
    operations = []
    current = np.eye(3, dtype=np.int64)
    for _ in range(order):
        operations.append(rietveld.SymmetryOperation(current, (0, 0, 0)))
        current = current @ rotation
    return rietveld.SpaceGroup(operations)


def test_exact_operations_validate_and_group_order_is_canonical() -> None:
    identity = rietveld.SymmetryOperation.identity()
    assert identity == rietveld.SymmetryOperation(np.eye(3, dtype=int), (0, 0, 0))
    assert hash(identity) == hash(rietveld.SymmetryOperation.identity())
    with pytest.raises(TypeError, match="Fraction"):
        rietveld.SymmetryOperation(np.eye(3, dtype=int), (0.5, 0, 0))
    with pytest.raises(ValueError, match="determinant"):
        rietveld.SymmetryOperation(np.zeros((3, 3), dtype=int), (0, 0, 0))
    with pytest.raises(ValueError, match="closed"):
        rietveld.SpaceGroup(
            [identity, rietveld.SymmetryOperation([[0, -1, 0], [1, 0, 0], [0, 0, 1]], (0, 0, 0))]
        )
    forward = rietveld.SpaceGroup([identity, inversion()])
    reverse = rietveld.SpaceGroup([inversion(), identity])
    assert forward == reverse
    assert forward.operations == reverse.operations


def test_special_position_expansion_matches_independent_reference() -> None:
    group = inversion_group()
    asymmetric = np.array([[0.0, 0.0, 0.0], [0.13, 0.27, 0.41]])
    native = group.expand_sites(asymmetric)
    positions, source = reference_expand_sites(group, asymmetric, tolerance=1e-10)
    np.testing.assert_allclose(native.fractional_xyz, positions, rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(native.source_site, source)
    assert native.fractional_xyz.shape == (3, 3)


@pytest.mark.parametrize(
    ("group", "hkl", "expected"),
    [
        (body_centred_group(), (1, 0, 0), True),
        (body_centred_group(), (1, 1, 0), False),
        (face_centred_group(), (1, 0, 0), True),
        (face_centred_group(), (1, 1, 1), False),
        (face_centred_group(), (2, 0, 0), False),
        (p21_group(), (0, 1, 0), True),
        (p21_group(), (0, 2, 0), False),
        (p21_group(), (1, 1, 0), False),
        (glide_group(), (1, 0, 1), True),
        (glide_group(), (1, 0, 2), False),
        (glide_group(), (1, 1, 1), False),
        (p31_group(), (0, 0, 1), True),
        (p31_group(), (0, 0, 2), True),
        (p31_group(), (0, 0, 3), False),
    ],
)
def test_exact_absences_match_closed_forms_and_independent_reference(
    group: rietveld.SpaceGroup, hkl: tuple[int, int, int], expected: bool
) -> None:
    native = bool(group.systematic_absences([hkl])[0])
    assert native is expected
    assert native is reference_is_systematically_absent(group, hkl)


def test_reflection_family_topology_matches_independent_reference() -> None:
    group = inversion_group()
    indices = np.array([[-1, 2, 3], [2, 0, -1], [0, 0, 4]])
    native = group.reflection_families(indices, merge_friedel=True)
    expected = [reference_family(group, tuple(row), merge_friedel=True) for row in indices]
    np.testing.assert_array_equal(native.canonical_hkl, [item[0] for item in expected])
    np.testing.assert_array_equal(native.multiplicity, [item[1] for item in expected])
    assert native.reflection_ids == tuple(f"hkl:{h},{k},{ell}" for (h, k, ell), _ in expected)


def test_randomized_triclinic_reflections_match_independent_brute_force() -> None:
    rng = np.random.default_rng(271828)
    group = rietveld.SpaceGroup.p1()
    generator = rietveld.PreparedReflectionGenerator(group, max_candidates=2_000_000)
    for _ in range(5):
        cell = rietveld.UnitCell(
            *rng.uniform([3.5, 4.0, 4.5, 70.0, 75.0, 80.0], [5.0, 5.5, 6.0, 100.0, 105.0, 110.0])
        )
        native = generator.generate(cell, rietveld.DSpacingRange(1.2, 4.0))
        reference = reference_generate_d_spacing(cell, group, 1.2, 4.0, merge_friedel=True)
        np.testing.assert_array_equal(native.hkl, [item[0] for item in reference])
        np.testing.assert_array_equal(native.multiplicity, [item[1] for item in reference])
        np.testing.assert_allclose(
            native.d_spacing_angstrom, [item[2] for item in reference], rtol=3e-15, atol=2e-15
        )


def test_nonstandard_monoclinic_setting_and_cell_constraints() -> None:
    twofold_a = rietveld.SymmetryOperation([[1, 0, 0], [0, -1, 0], [0, 0, -1]], (0, 0, 0))
    group = rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), twofold_a])
    assert group.crystal_system == "monoclinic"
    assert group.metric_constraints.independent_parameter_count == 4
    cell = rietveld.UnitCell(4.0, 5.0, 6.0, 103.0, 90.0, 90.0)
    result = rietveld.PreparedReflectionGenerator(group).generate(
        cell, rietveld.DSpacingRange(1.5, 5.0)
    )
    assert len(result.reflection_ids) > 0
    incompatible = replace(cell, gamma_deg=91.0)
    with pytest.raises(ValueError, match="incompatible"):
        rietveld.PreparedReflectionGenerator(group).generate(
            incompatible, rietveld.DSpacingRange(1.5, 5.0)
        )


def test_crystal_systems_and_metric_dimensions_are_derived_from_rotations() -> None:
    identity = rietveld.SpaceGroup.p1()
    monoclinic = cyclic_rotation_group(np.diag([1, -1, -1]), 2)
    orthorhombic = rietveld.SpaceGroup(
        [
            rietveld.SymmetryOperation(np.diag(signs), (0, 0, 0))
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
    )
    tetragonal = cyclic_rotation_group(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64), 4
    )
    trigonal = cyclic_rotation_group(
        np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64), 3
    )
    hexagonal = cyclic_rotation_group(
        np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64), 6
    )
    cubic_operations = []
    for permutation in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        for signs in np.ndindex(2, 2, 2):
            matrix = np.zeros((3, 3), dtype=np.int64)
            for row, column in enumerate(permutation):
                matrix[row, column] = 1 if signs[row] else -1
            if round(np.linalg.det(matrix)) == 1:
                cubic_operations.append(rietveld.SymmetryOperation(matrix, (0, 0, 0)))
    cubic = rietveld.SpaceGroup(cubic_operations)

    assert [
        (group.crystal_system, group.metric_constraints.independent_parameter_count)
        for group in (identity, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic)
    ] == [
        ("triclinic", 6),
        ("monoclinic", 4),
        ("orthorhombic", 3),
        ("tetragonal", 2),
        ("trigonal", 2),
        ("hexagonal", 2),
        ("cubic", 1),
    ]
    for group in (identity, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic):
        constraints = group.metric_constraints
        assert constraints.parameterization_basis.shape == (
            constraints.independent_parameter_count,
            6,
        )
        np.testing.assert_array_equal(
            constraints.equations @ constraints.parameterization_basis.T,
            0,
        )


def test_d_q_cw_and_tof_ranges_are_consistent_for_monochromatic_data() -> None:
    cell = rietveld.UnitCell(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    generator = rietveld.PreparedReflectionGenerator(rietveld.SpaceGroup.p1())
    d_result = generator.generate(cell, rietveld.DSpacingRange(2.0, 4.0))
    q_result = generator.generate(cell, rietveld.ScatteringVectorRange(2 * np.pi / 4.0, np.pi))
    wavelength = 1.0
    cw_result = generator.generate(
        cell,
        rietveld.CwTwoThetaRange(
            2 * np.degrees(np.arcsin(wavelength / 8.0)),
            2 * np.degrees(np.arcsin(wavelength / 4.0)),
            wavelength,
        ),
    )
    tof_result = generator.generate(
        cell,
        rietveld.TofRange(2000.0, 4000.0, 1.5, 4.5, 0.0, 1000.0),
    )
    for result in (q_result, cw_result, tof_result):
        np.testing.assert_array_equal(result.hkl, d_result.hkl)


def test_generated_cell_derivatives_match_centered_differences() -> None:
    cell = rietveld.UnitCell(4.3, 5.1, 6.2, 78.0, 83.0, 71.0)
    generator = rietveld.PreparedReflectionGenerator(rietveld.SpaceGroup.p1())
    result = generator.generate(cell, rietveld.DSpacingRange(1.1, 7.0))
    parameter_names = (
        "a_angstrom",
        "b_angstrom",
        "c_angstrom",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
    )
    for parameter, name in enumerate(parameter_names):
        step = 1e-6 if parameter < 3 else 1e-7
        plus = replace(cell, **{name: getattr(cell, name) + step}).d_spacings(result.hkl)
        minus = replace(cell, **{name: getattr(cell, name) - step}).d_spacings(result.hkl)
        finite = (plus.d_spacing_angstrom - minus.d_spacing_angstrom) / (2 * step)
        np.testing.assert_allclose(
            result.d_spacing_derivatives[:, parameter], finite, rtol=3e-7, atol=3e-8
        )


def test_generator_limits_invalid_ranges_and_read_only_results_are_explicit() -> None:
    cell = rietveld.UnitCell(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    with pytest.raises(ValueError, match="positive"):
        rietveld.PreparedReflectionGenerator(rietveld.SpaceGroup.p1(), max_candidates=0)
    with pytest.raises(ValueError, match="positive"):
        rietveld.PreparedReflectionGenerator(rietveld.SpaceGroup.p1(), max_candidates=-1)
    generator = rietveld.PreparedReflectionGenerator(rietveld.SpaceGroup.p1(), max_candidates=10)
    with pytest.raises(ValueError, match="configured limit"):
        generator.generate(cell, rietveld.DSpacingRange(0.5, 4.0))
    generator = rietveld.PreparedReflectionGenerator(rietveld.SpaceGroup.p1())
    with pytest.raises(ValueError, match="range"):
        generator.generate(cell, rietveld.DSpacingRange(4.0, 1.0))
    result = generator.generate(cell, rietveld.DSpacingRange(2.0, 4.0))
    assert len(result.reflection_ids) == len(set(result.reflection_ids))
    assert len(np.unique(result.d_spacing_angstrom)) < len(result.d_spacing_angstrom)
    assert not result.hkl.flags.writeable
    assert not result.d_spacing_derivatives.flags.writeable
    assert not generator.space_group.metric_constraints.equations.flags.writeable
    assert not generator.space_group.metric_constraints.parameterization_basis.flags.writeable
