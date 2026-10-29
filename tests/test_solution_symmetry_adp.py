"""Constrained ADP equations, special positions and independent derivatives."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution_reference import symmetric_model_reference
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SymmetricSolutionModel"), reason="requires experimental-solution"
)


def model_case(symbol, cell, sites, *, mode="anisotropic", shift=(0, 0, 0), cache_geometry=True):
    import gemmi

    full, species = [], []
    for element, point in sites:
        unique = []
        for op in gemmi.find_spacegroup_by_name(symbol).operations():
            xyz = np.mod(op.apply_to_xyz(point), 1)
            if not any(np.linalg.norm((xyz - other + 0.5) % 1 - 0.5) < 1e-8 for other in unique):
                unique.append(xyz)
        full.extend(unique)
        species.extend([element] * len(unique))
    hkl = np.array(
        [h for h in itertools.product(range(-3, 4), repeat=3) if h > tuple(-v for v in h)],
        dtype=np.int64,
    )
    observed = np.random.default_rng(29).uniform(1, 20, len(hkl))
    native = _core._SymmetricSolutionModel(
        list(cell.as_tuple()),
        space_group_by_symbol(symbol).space_group._native,
        species,
        np.ascontiguousarray((np.array(full) + shift) % 1).ravel(),
        hkl.ravel(),
        observed,
        mode,
        0.01,
        cache_geometry,
    )
    metadata = native.metadata()
    q = hkl @ np.linalg.inv(cell.geometry().direct_basis)
    scattering, _ = xray_non_resonant(
        [s["species"] for s in metadata["sites"]], np.linalg.norm(q, axis=1) / 2
    )
    return native, metadata, hkl, observed, scattering


CASES = [
    (
        "P 1",
        UnitCell(6, 7, 8, 78, 96, 107),
        [("Si", (0.1, 0.2, 0.3)), ("O", (0.4, 0.6, 0.2))],
        (3, [6, 6], 3),
    ),
    (
        "P -1",
        UnitCell(6, 7, 8, 78, 96, 107),
        [("Si", (0.12, 0.23, 0.31)), ("O", (0, 0, 0))],
        (3, [6, 6], 0),
    ),
    (
        "P m",
        UnitCell(6, 7, 8, 90, 96, 90),
        [("Si", (0.41, 0, 0.12)), ("O", (0.15, 0.26, 0.33))],
        (3, [4, 6], 2),
    ),
    (
        "P 31 2 1",
        UnitCell(4.913, 4.913, 5.405, 90, 90, 120),
        [("Si", (0.4697, 0, 1 / 3)), ("O", (0.4133, 0.2672, 0.1188))],
        (4, [4, 6], 0),
    ),
    (
        "F m -3 m",
        UnitCell(4.2, 4.2, 4.2, 90, 90, 90),
        [("Si", (0, 0, 0)), ("O", (0.5, 0.5, 0.5))],
        (0, [1, 1], 0),
    ),
]


@pytest.mark.parametrize("symbol,cell,sites,counts", CASES)
def test_topology_independent_values_and_centered_derivatives(symbol, cell, sites, counts):
    native, metadata, hkl, observed, factors = model_case(
        symbol, cell, sites, shift=(0.173, 0.287, 0.391)
    )
    nc, nt, gauge = counts
    assert sum(len(s["coordinate_basis"]) for s in metadata["sites"]) == nc
    assert [len(b) for b in metadata["tensor_bases"]] == nt
    assert metadata["origin_gauge_dimension"] == gauge
    assert metadata["maximum_matching_error"] < 1e-9
    p = native.initial_parameters(0.015)
    # Exact repeated eigenvalues first, then general invariant tensors and motions.
    for randomize in (False, True):
        if randomize:
            p += np.random.default_rng(73).normal(0, 0.025, len(p))
        state = native.evaluate(p)
        ref = symmetric_model_reference(
            metadata, cell.geometry().direct_basis, hkl, observed, factors, p
        )
        for key in (
            "residual",
            "objective",
            "amplitude_scale",
            "r1",
            "fractional_xyz",
            "u_cartesian",
        ):
            np.testing.assert_allclose(state[key], ref[key], rtol=5e-11, atol=2e-12)
        jac = np.asarray(state["jacobian"]).reshape(len(p), len(hkl))
        step = 2e-6
        for i in range(len(p)):
            delta = np.zeros(len(p))
            delta[i] = step
            plus = symmetric_model_reference(
                metadata, cell.geometry().direct_basis, hkl, observed, factors, p + delta
            )
            minus = symmetric_model_reference(
                metadata, cell.geometry().direct_basis, hkl, observed, factors, p - delta
            )
            np.testing.assert_allclose(
                jac[i], (plus["residual"] - minus["residual"]) / (2 * step), rtol=3e-5, atol=2e-8
            )
        for site, tensor in zip(metadata["sites"], state["u_cartesian"], strict=True):
            for rotation in np.array(site["stabilizer"]):
                np.testing.assert_allclose(rotation @ tensor @ rotation.T, tensor, atol=2e-13)


def test_isotropic_boundary_and_invalid_inputs():
    symbol, cell, sites, _ = CASES[2]
    native, m, *_ = model_case(symbol, cell, sites, mode="isotropic")
    assert [len(b) for b in m["tensor_bases"]] == [1, 1]
    for u in (1e-5, 0.2):
        value = native.evaluate(native.initial_parameters(u))
        np.testing.assert_allclose(np.linalg.eigvalsh(value["u_cartesian"]), u, atol=3e-15)
    for u in (0, -1, 0.20001, float("nan")):
        with pytest.raises(ValueError):
            native.initial_parameters(u)
    p = native.initial_parameters(0.01)
    for bad in (p[:-1], p * float("nan"), p * 10):
        with pytest.raises(ValueError):
            native.evaluate(bad)
    with pytest.raises(ValueError, match="metric is incompatible"):
        model_case("P 4", UnitCell(5, 6, 7, 90, 90, 90), [("Si", (0, 0, 0))])


def test_refinement_improves_and_has_truthful_stops():
    symbol, cell, sites, _ = CASES[1]
    native, *_ = model_case(symbol, cell, sites)
    p = native.initial_parameters(0.01)
    start = native.evaluate(p)
    result = native.refine(p, 8, 10, None)
    assert result["evaluation"]["objective"] <= start["objective"]
    assert result["termination"] in ("max_iterations", "stagnated", "gradient")
    assert result["accepted_steps"] <= 8
    timeout = native.refine(p, 8, 1e-12, None)
    assert timeout["termination"] == "max_runtime"
    from phasesmith.experimental.solution import SolutionCancellation

    token = SolutionCancellation()
    token.cancel()
    cancelled = native.refine(p, 8, 10, token._native)
    assert cancelled["termination"] == "cancelled"


def test_multiplicity_collapse_is_rejected():
    symbol, cell, sites, _ = CASES[1]
    native, metadata, *_ = model_case(symbol, cell, sites)
    parameters = native.initial_parameters(0.01)
    site = metadata["sites"][0]
    basis = np.asarray(site["coordinate_basis"])
    offset = metadata["offsets"][0][0]
    parameters[offset : offset + 3] = np.linalg.solve(basis.T, -np.asarray(site["fractional_xyz"]))
    with pytest.raises(ValueError, match="multiplicity"):
        native.evaluate(parameters)


def test_f000_counts_every_image_and_has_no_adp_derivative():
    cell = UnitCell(5, 6, 7, 90, 90, 90)
    factors, _ = xray_non_resonant(["Si", "O"], np.array([0.0]))
    observed = np.array([factors.sum()])
    native = _core._SymmetricSolutionModel(
        list(cell.as_tuple()),
        space_group_by_symbol("P 1").space_group._native,
        ["Si", "O"],
        np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8]),
        np.zeros(3, dtype=np.int64),
        observed,
        "anisotropic",
        0.01,
    )
    state = native.evaluate(native.initial_parameters(0.03))
    assert abs(state["amplitude_scale"] - 1) < 1e-14
    np.testing.assert_allclose(state["jacobian"], 0, atol=1e-15)
    assert state["r1"] < 1e-14


@pytest.mark.parametrize(
    "shift", [(-0.125, -0.25, -0.125), (0.137, 0.287, 0.391), (0.493, 0.081, 0.714)]
)
def test_inversion_pairs_recover_pseudosymmetric_framework_origin(shift):
    from phasesmith.experimental.solution_count_free_reference import (
        inversion_origin_seeds_reference,
    )

    cell = UnitCell(8.6967, 4.2205, 10.574, 90, 90, 90)
    # Synthetic rational four-site orbit; no measured/reference coordinates.
    native, metadata, _, _, _ = model_case(
        "Pnma", cell, [("Si", (0.125, 0.25, 0.125))], shift=shift, mode="isotropic"
    )
    assert metadata["maximum_matching_error"] < 1e-12
    full = np.array(
        [
            image
            for image in space_group_by_symbol("Pnma")
            .space_group.expand_sites([[0.125, 0.25, 0.125]])
            .fractional_xyz
        ]
    )
    points = (full + shift) % 1
    seeds = inversion_origin_seeds_reference(points, ["Si"] * len(points), np.zeros(3))
    recovered = np.array(metadata["origin_shift"])
    delta = (seeds - recovered + 0.5) % 1 - 0.5
    assert np.linalg.norm(delta, axis=1).min() < 1e-12
    state = native.evaluate(native.initial_parameters(0.012))
    assert metadata["full_count"] == 4
    expanded = space_group_by_symbol("Pnma").space_group.expand_sites(state["fractional_xyz"])
    assert len(expanded.fractional_xyz) == 4
