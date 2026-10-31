"""Independent difference Fourier, projection and fixed-orbit derivative checks."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant
from phasesmith.single_crystal.reference.solution_completion_reference import (
    difference_reference,
    orbit_reference,
    project_reference,
)
from phasesmith.single_crystal.solution import SolutionOptions

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_CompletionSearch"), reason="requires experimental-solution"
)


def unique_images(group, point):
    images = []
    rotations = []
    translations = []
    for op in group.operations:
        r = np.array(op.rotation)
        t = np.array([float(v) for v in op.translation])
        image = (r @ point + t) % 1
        if any(np.linalg.norm((image - other + 0.5) % 1 - 0.5) < 1e-8 for other in images):
            continue
        images.append(image)
        rotations.append(r)
        translations.append(t)
    return np.array(images), np.array(rotations), np.array(translations)


def setup(symbol="P 1", cell=None, point=(0.31, 0.42, 0.67)):
    cell = cell or UnitCell(6, 7, 8, 79, 96, 107)
    group = space_group_by_symbol(symbol).space_group
    hkl = np.array(
        [h for h in itertools.product(range(-3, 4), repeat=3) if h != (0, 0, 0)], dtype=np.int64
    )
    hkl = hkl[~group.systematic_absences(hkl)]
    fixed, _, _ = unique_images(group, np.array([0.12, 0.23, 0.34]))
    species = ["Si"] * len(fixed)
    observed = np.sqrt(np.sum(hkl**2, axis=1) + 1.0) * 3
    s = np.linalg.norm(hkl @ np.linalg.inv(cell.geometry().direct_basis), axis=1) / 2
    factors, _ = xray_non_resonant([*species, "O"], s)
    native = _core._CompletionSearch(
        list(cell.as_tuple()),
        group._native,
        species,
        fixed.ravel(),
        "O",
        hkl.ravel(),
        observed,
        0.014,
    )
    images, r, t = unique_images(group, np.array(point))
    return native, cell, group, hkl, observed, fixed, factors, s, images, r, t


@pytest.mark.parametrize(
    "symbol,cell,point,tangents",
    [
        ("P 1", UnitCell(6, 7, 8, 79, 96, 107), (0.31, 0.42, 0.67), np.eye(3)),
        ("P -1", UnitCell(6, 7, 8, 79, 96, 107), (0.31, 0.42, 0.67), np.eye(3)),
        ("P m", UnitCell(6, 7, 8, 90, 96, 90), (0.31, 0.0, 0.67), np.array([[1, 0, 0], [0, 0, 1]])),
        (
            "P 31 2 1",
            UnitCell(4.913, 4.913, 5.405, 90, 90, 120),
            (0.4133, 0.2672, 0.1188),
            np.eye(3),
        ),
    ],
)
def test_orbit_values_and_fused_derivatives(symbol, cell, point, tangents):
    native, cell, _group, hkl, observed, fixed, factors, s, _images, r, t = setup(
        symbol, cell, point
    )
    reference = orbit_reference(
        hkl, observed, fixed, factors[:, :-1], factors[:, -1], s, 0.014, point, r, t
    )
    result = native.evaluate(point)
    for key in ["objective", "amplitude_scale", "residual", "r1"]:
        np.testing.assert_allclose(result[key], reference[key], rtol=5e-12, atol=3e-13)
    np.testing.assert_allclose(
        np.array(native.framework_f()).view(),
        np.column_stack([reference["base"].real, reference["base"].imag]),
        rtol=5e-12,
        atol=2e-13,
    )
    jac = np.array(result["jacobian"]).reshape(3, len(hkl))
    step = 1e-6
    for tangent in tangents:
        plus = orbit_reference(
            hkl,
            observed,
            fixed,
            factors[:, :-1],
            factors[:, -1],
            s,
            0.014,
            np.asarray(point) + step * tangent,
            r,
            t,
        )
        minus = orbit_reference(
            hkl,
            observed,
            fixed,
            factors[:, :-1],
            factors[:, -1],
            s,
            0.014,
            np.asarray(point) - step * tangent,
            r,
            t,
        )
        np.testing.assert_allclose(
            tangent @ jac, (plus["residual"] - minus["residual"]) / (2 * step), rtol=2e-5, atol=2e-8
        )
        fd = (plus["objective"] - minus["objective"]) / (2 * step)
        assert abs(tangent @ np.array(result["gradient"]) - fd) < 2e-8


def test_signed_difference_map_scale_and_friedel_boundary():
    native, cell, group, hkl, observed, fixed, _factors, _s, *_ = setup()
    base = np.array(native.framework_f())
    base = base[:, 0] + 1j * base[:, 1]
    ref, k = difference_reference((16, 16, 16), hkl, observed, base)
    result = native.difference_map((16, 16, 16))
    np.testing.assert_allclose(
        np.array(result["density"]).reshape(ref.shape), ref, rtol=2e-11, atol=2e-14
    )
    assert abs(result["amplitude_scale"] - k) < 1e-14
    assert np.min(ref) < 0 and np.max(ref) > 0
    assert abs(np.sum(result["density"])) < 1e-12
    scaled = _core._CompletionSearch(
        list(cell.as_tuple()),
        group._native,
        ["Si"] * len(fixed),
        fixed.ravel(),
        "O",
        hkl.ravel(),
        observed * 17,
        0.014,
    )
    np.testing.assert_allclose(
        result["density"], scaled.difference_map((16, 16, 16))["density"], rtol=3e-12, atol=2e-14
    )
    aliased = _core._CompletionSearch(
        list(cell.as_tuple()),
        group._native,
        ["Si"] * len(fixed),
        fixed.ravel(),
        "O",
        np.array([-4, 0, 0, 4, 0, 0], dtype=np.int64),
        np.ones(2),
        0.014,
    )
    with pytest.raises(ValueError, match="Nyquist"):
        aliased.difference_map((8, 8, 8))
    bad = _core._CompletionSearch(
        list(cell.as_tuple()),
        group._native,
        ["Si"] * len(fixed),
        fixed.ravel(),
        "O",
        hkl[:-1].ravel(),
        observed[:-1],
        0.014,
    )
    with pytest.raises(ValueError, match="Friedel"):
        bad.difference_map((16, 16, 16))


def test_affine_projection_mirror_glide_and_intersection():
    cell = UnitCell(6, 7, 8, 90, 96, 90)
    native, cell, group, *_ = setup("P 2/m", cell)
    for seed in [(0.012, 0.008, 0.017), (0.2, 0.007, 0.4), (0.2, 0.2, 0.4), (0.99, 0.004, 0.99)]:
        for indices in ([0], [1], [2], [0, 1], [1, 2]):
            ops = [group.operations[i] for i in indices]
            r = np.array([op.rotation for op in ops])
            t = np.array([[float(v) for v in op.translation] for op in ops])
            ref = project_reference(cell.geometry().direct_basis, seed, r, t, 0.15)
            actual = native.project(seed, indices, 0.15)
            assert (actual is None) == (ref is None)
            if ref is not None:
                np.testing.assert_allclose(
                    (np.asarray(actual) - ref + 0.5) % 1 - 0.5, 0, atol=3e-13
                )
    glide, _, glide_group, *_ = setup("P c", cell)
    index = next(i for i, op in enumerate(glide_group.operations) if any(op.translation))
    assert glide.project((0.2, 0.0, 0.4), [index], 0.15) is None
    with pytest.raises(ValueError):
        native.project((0.1, 0.2, 0.3), [999], 0.15)


def test_invalid_completion_controls():
    with pytest.raises(TypeError):
        SolutionOptions(completion=1)
    native, *_ = setup()
    with pytest.raises(ValueError):
        native.evaluate((float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError):
        native.project((0.1, 0.2, 0.3), [], 0.0)


def test_partial_framework_retains_inequivalent_origin_settings():
    group = space_group_by_symbol("I 4/m c m").space_group
    points = []
    species = []
    for element, point in [("Fe", (0.5, 0.0, 0.0)), ("Si", (0.5, 0.5, 0.25))]:
        images, _, _ = unique_images(group, np.array(point))
        points.extend(images)
        species.extend([element] * len(images))
    xyz = np.asarray(points) + np.random.default_rng(351).normal(0, 0.001, (len(points), 3))
    models = _core._completion_frameworks(
        [6.0, 6.0, 11.0, 90.0, 90.0, 90.0], group._native, species, xyz.ravel(), 0.15, 8
    )
    assert 1 < len(models) <= 8
    assert all(sorted(m["multiplicities"]) == [4, 4] for m in models)
    with pytest.raises(ValueError):
        _core._completion_frameworks(
            [6.0, 6.0, 11.0, 90.0, 90.0, 90.0], group._native, species, xyz.ravel(), 0.15, 0
        )


def test_complete_corrupted_weak_orbit_and_truthful_stops():
    # Synthetic independent observations; only noisy strong sites and wrong O
    # positions enter the completion call. No reference O coordinate is supplied.
    cell = UnitCell(6, 7, 8, 90, 101, 90)
    group = space_group_by_symbol("P 2/m").space_group
    xyz = []
    species = []
    for element, point in [
        ("Fe", (0.0, 0.0, 0.0)),
        ("Si", (0.5, 0.5, 0.5)),
        ("O", (0.273, 0.0, 0.391)),
    ]:
        images, _, _ = unique_images(group, np.array(point))
        xyz.extend(images)
        species.extend([element] * len(images))
    xyz = np.asarray(xyz)
    hkl = np.array(
        [h for h in itertools.product(range(-6, 7), repeat=3) if h != (0, 0, 0)], dtype=np.int64
    )
    s = np.linalg.norm(hkl @ np.linalg.inv(cell.geometry().direct_basis), axis=1) / 2
    factors, _ = xray_non_resonant(species, s)
    calculated = (
        factors
        * np.exp(-8 * np.pi**2 * 0.01 * s[:, None] ** 2)
        * np.exp(2j * np.pi * (hkl @ xyz.T))
    ).sum(axis=1)
    observed = np.ascontiguousarray(np.abs(calculated))
    initial = xyz.copy()
    initial[:2] += np.array([[0.003, -0.002, 0.001], [-0.002, 0.001, -0.003]])
    initial[2:] = [[0.12, 0.22, 0.34], [0.62, 0.19, 0.83]]
    args = (
        list(cell.as_tuple()),
        group._native,
        species,
        initial.ravel(),
        hkl.ravel(),
        observed,
        (32, 32, 32),
    )
    result = _core._solution_complete_candidate(*args, 20.0, None)
    assert result["completion"]["applied"] and result["completion"]["added_atoms"] == 2
    assert result["candidate"]["r1"] < 1e-5
    assert result["candidate"]["refinement"]["coordinate_parameter_count"] == 2
    refined = np.asarray(result["candidate"]["refinement"]["fractional_xyz"])
    # Check periodic orbit geometry independently, up to inversion/allowed origin.
    o_index = result["candidate"]["refinement"]["species"].index("O")
    best = np.inf
    for shift in itertools.product((0, 0.5), repeat=3):
        model, _, _ = unique_images(group, (refined[o_index] + shift) % 1)
        expected = xyz[np.array(species) == "O"]
        delta = (expected[:, None] - model[None] + 0.5) % 1 - 0.5
        best = min(
            best, np.linalg.norm(delta @ cell.geometry().direct_basis.T, axis=2).min(axis=1).max()
        )
    assert best < 0.001
    timeout = _core._solution_complete_candidate(*args, 1e-12, None)
    assert timeout["completion"]["status"] == "max_runtime"
    from phasesmith.single_crystal.solution import SolutionCancellation

    token = SolutionCancellation()
    token.cancel()
    stopped = _core._solution_complete_candidate(*args, 20.0, token._native)
    assert stopped["completion"]["status"] == "cancelled" and stopped["candidate"] is None


def test_weak_orbit_completion_resolves_ambiguous_tetragonal_framework():
    cell = UnitCell(6.2, 6.2, 10.8, 90, 90, 90)
    group = space_group_by_symbol("I 4/m c m").space_group
    xyz = []
    species = []
    for element, point in [
        ("Fe", (0.5, 0.0, 0.0)),
        ("Si", (0.5, 0.5, 0.25)),
        ("O", (0.637, 0.137, 0.384)),
    ]:
        images, _, _ = unique_images(group, np.array(point))
        xyz.extend(images)
        species.extend([element] * len(images))
    xyz = np.asarray(xyz)
    hkl = np.array(
        [h for h in itertools.product(range(-6, 7), repeat=3) if h != (0, 0, 0)], dtype=np.int64
    )
    hkl = hkl[~group.systematic_absences(hkl)]
    s = np.linalg.norm(hkl @ np.linalg.inv(cell.geometry().direct_basis), axis=1) / 2
    factors, _ = xray_non_resonant(species, s)
    observed = np.abs(
        (
            factors
            * np.exp(-8 * np.pi**2 * 0.01 * s[:, None] ** 2)
            * np.exp(2j * np.pi * (hkl @ xyz.T))
        ).sum(axis=1)
    )
    initial = xyz.copy()
    rng = np.random.default_rng(936)
    initial[:8] += rng.normal(0, 0.001, (8, 3))
    initial[8:] = rng.uniform(0, 1, (16, 3))
    result = _core._solution_complete_candidate(
        list(cell.as_tuple()),
        group._native,
        species,
        initial.ravel(),
        hkl.ravel(),
        np.ascontiguousarray(observed),
        (32, 32, 32),
        20.0,
        None,
    )
    assert result["completion"]["applied"] and result["completion"]["framework_hypotheses"] > 1
    candidate = result["candidate"]
    assert candidate["r1"] < 1e-5
    assert candidate["refinement"]["coordinate_parameter_count"] == 2
    found = np.asarray(candidate["fractional_xyz"])
    found_species = np.array(candidate["species"])
    expected = xyz[np.array(species) == "O"]
    best = np.inf
    for sign in (-1, 1):
        for anchor in xyz[np.array(species) == "Fe"]:
            shifted = (
                sign * found + anchor - sign * found[np.flatnonzero(found_species == "Fe")[0]]
            ) % 1
            delta = (expected[:, None] - shifted[None, found_species == "O"] + 0.5) % 1 - 0.5
            distances = np.linalg.norm(delta @ cell.geometry().direct_basis.T, axis=2)
            best = min(best, distances.min(axis=1).max())
    assert best < 0.001

    import phasesmith as ps
    from phasesmith.io.cif import read_cif
    from phasesmith.scattering import XrayNonResonant
    from phasesmith.single_crystal.solution_symmetry import SymmetryRefinement, candidate_cif
    from phasesmith.symmetry import SpaceGroup

    refined = SymmetryRefinement.from_native(candidate["refinement"], group)
    full_cif = candidate_cif(
        cell,
        tuple(candidate["species"]),
        found,
        refined.full_u_equiv,
        SpaceGroup.p1(),
        refined.full_u_cif,
    )
    structures = [refined.as_structure(cell), read_cif(full_cif).structure]
    amplitudes = []
    for structure in structures:
        values = ps.calculate_structure_factor_values(
            structure, hkl, np.ones(len(hkl), dtype=np.int64), XrayNonResonant()
        )
        prediction = candidate["amplitude_scale"] * np.sqrt(values.f_squared)
        assert abs(np.abs(prediction - observed).sum() / observed.sum() - candidate["r1"]) < 2e-10
        amplitudes.append(prediction)
    np.testing.assert_allclose(amplitudes[0], amplitudes[1], rtol=1e-10, atol=2e-9)


def test_orbit_periodicity_and_hkl_resource_boundary():
    native, cell, group, _hkl, _observed, fixed, *_ = setup()
    point = np.array([0.31, 0.42, 0.67])
    original = native.evaluate(point)
    translated = native.evaluate(point + np.array([3, -7, 12]))
    np.testing.assert_allclose(original["residual"], translated["residual"], rtol=3e-12, atol=2e-13)
    with pytest.raises(ValueError):
        _core._CompletionSearch(
            list(cell.as_tuple()),
            group._native,
            ["Si"] * len(fixed),
            fixed.ravel(),
            "O",
            np.array([-1025, 0, 0, 1025, 0, 0], dtype=np.int64),
            np.ones(2),
            0.01,
        )
