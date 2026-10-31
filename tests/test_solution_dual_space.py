"""Independent Fourier/mask equations and experimental strategy contracts."""

import itertools
import json

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import (
    SolutionCancellation,
    SolutionOptions,
    solve_structure,
    write_solution,
)
from phasesmith.experimental.solution_dual_reference import (
    atomic_projection_reference,
    dual_space_reference,
)
from phasesmith.io.hkl import HklData

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_dual_space"), reason="requires experimental-solution"
)


def inputs():
    shape = (16, 16, 16)
    hkl = np.array(
        [h for h in itertools.product(range(-4, 5), repeat=3) if 0 < np.dot(h, h) <= 16],
        dtype=np.int64,
    )
    xyz = np.array([[0.137, 0.259, 0.382], [0.437, 0.628, 0.821], [0.752, 0.128, 0.491]])
    amplitudes = np.abs((np.exp(2j * np.pi * (hkl @ xyz.T)) * [2.3, 1.4, 0.8]).sum(axis=1))
    phases = np.array([0.731 * (h[0] + 3 * h[1] + 11 * h[2]) for h in hkl])
    support = np.zeros(shape, dtype=np.uint8)
    support[0, 0, 0] = 1
    support[tuple((hkl % shape).T)] = 2
    # An unmeasured Hermitian pair inside the support, not a fixed zero.
    keep = np.any(np.abs(hkl) != [1, 0, 0], axis=1)
    for h in hkl[~keep]:
        support[tuple(h % shape)] = 1
    cell = UnitCell(5, 6, 7, 90, 98, 90)
    return shape, cell, hkl[keep], amplitudes[keep], phases[keep], support


def run_native(
    shape,
    cell,
    hkl,
    amplitude,
    phases,
    support,
    *,
    cycles=6,
    count=3,
    sigma=0.25,
    m=3.0,
    omit=0.1,
    seed=987,
    rank=None,
):
    return _core._solution_dual_space(
        shape,
        list(cell.as_tuple()),
        hkl.ravel(),
        amplitude,
        phases,
        support.ravel(),
        cycles,
        count,
        sigma,
        m,
        omit,
        seed,
        rank,
    )


@pytest.mark.parametrize(
    "m,omit,rank,cycles",
    [(1.0, 0.0, None, 6), (3.0, 0.1, None, 6), (2.5, 0.2, None, 9), (3.0, 0.0, 0, 1)],
)
def test_complete_cycles_against_independent_numpy(m, omit, rank, cycles):
    shape, cell, hkl, amplitude, phases, support = inputs()
    native = run_native(
        shape, cell, hkl, amplitude, phases, support, m=m, omit=omit, rank=rank, cycles=cycles
    )
    reference = dual_space_reference(
        shape,
        cell.geometry().direct_basis,
        hkl,
        amplitude,
        phases,
        support,
        cycles=cycles,
        atom_count=3,
        sigma=0.25,
        hybrid_m=m,
        omit_fraction=omit,
        seed=987,
        patterson_rank=rank,
    )
    np.testing.assert_allclose(
        np.array(native["density"]).reshape(shape), reference["density"], rtol=3e-9, atol=2e-11
    )
    np.testing.assert_allclose(native["history"], reference["history"], rtol=2e-10, atol=3e-12)
    assert native["best_cycle"] == reference["best_cycle"]
    if rank is not None:
        assert native["initialization"] == "patterson"
        np.testing.assert_array_equal(
            native["patterson_vector_grid"], reference["patterson_vector_grid"]
        )
    repeated = run_native(
        shape, cell, hkl, amplitude, phases, support, m=m, omit=omit, rank=rank, cycles=cycles
    )
    np.testing.assert_array_equal(native["density"], repeated["density"])
    scaled = run_native(
        shape,
        cell,
        hkl,
        amplitude * 1e20,
        phases,
        support,
        m=m,
        omit=omit,
        rank=rank,
        cycles=cycles,
    )
    np.testing.assert_allclose(native["density"], scaled["density"], rtol=3e-9, atol=2e-11)


@pytest.mark.parametrize("cycle", [0, 2, 49, 50, 53])
def test_cartesian_gaussians_and_omission_against_dense_reference(cycle):
    shape, cell, *_ = inputs()
    density = np.random.default_rng(471).normal(0, 0.5, shape)
    density[0, 0, 0] = 6
    density[8, 9, 7] = 7
    native = _core._solution_atomic_projection(
        density.ravel(), shape, list(cell.as_tuple()), 3, 0.25, 0.2, 123, cycle
    )
    reference = atomic_projection_reference(
        density, cell.geometry().direct_basis, 3, 0.25, cycle=cycle, seed=123, omit_fraction=0.2
    )
    np.testing.assert_allclose(np.array(native).reshape(shape), reference, rtol=2e-12, atol=2e-13)
    assert np.all(np.asarray(native) >= 0)


def test_gaussian_finite_support_includes_boundary():
    # Native and reference use 3 sigma, with an inclusive rounding allowance.
    cell = UnitCell(8, 8, 8, 90, 90, 90)
    rho = np.zeros((16, 16, 16))
    rho[0, 0, 0] = 10
    rho[1, 0, 0] = 1
    rho[2, 0, 0] = 0.5
    native = np.array(
        _core._solution_atomic_projection(
            rho.ravel(), rho.shape, list(cell.as_tuple()), 1, 1 / 6, 0.0, 0, 50
        )
    ).reshape(rho.shape)
    assert native[1, 0, 0] > 0
    assert native[2, 0, 0] == 0


def test_invalid_boundaries_and_zero_map():
    shape, cell, hkl, amplitude, phases, support = inputs()
    for kwargs in (
        {"count": 0},
        {"sigma": 0},
        {"m": 4.1},
        {"omit": 0.31},
        {"rank": 128},
        {"cycles": 0},
        {"sigma": 2.0},
    ):
        with pytest.raises(ValueError):
            run_native(shape, cell, hkl, amplitude, phases, support, **kwargs)
    with pytest.raises(ValueError, match="support"):
        run_native(shape, cell, hkl[:-1], amplitude[:-1], phases[:-1], support)
    invalid = support.copy()
    invalid.flat[1] = 1
    with pytest.raises(ValueError):
        run_native(shape, cell, hkl, amplitude, phases, invalid)
    result = _core._solution_atomic_projection(
        np.zeros(4096), shape, list(cell.as_tuple()), 3, 0.25, 0.1, 0, 0
    )
    assert not np.any(result)


def test_underresolved_mask_is_a_rejected_map():
    shape, cell, hkl, amplitude, phases, support = inputs()
    result = run_native(shape, cell, hkl, amplitude, phases, support, sigma=1e-4)
    assert len(result["history"]) == 1
    assert not np.any(result["density"])


def test_portfolio_metadata_output_and_cancellation(tmp_path):
    _shape, cell, hkl, amplitude, _, _ = inputs()
    data = HklData(hkl, amplitude**2, np.ones(len(hkl)))
    result = solve_structure(
        data,
        cell=cell,
        composition={"Si": 3},
        options=SolutionOptions(
            phasing="portfolio",
            trials=4,
            cycles=8,
            r1_target=0,
            max_seconds=20,
            polish_iterations=2,
            assignment_rounds=0,
            refinement="none",
        ),
    )
    assert [t.method for t in result.phasing_trials] == [
        "charge_flip",
        "dual_space",
        "charge_flip",
        "dual_space",
    ]
    assert len(result.phasing_trials) == len(result.histories)
    assert all(t.completed_cycles == 8 for t in result.phasing_trials)
    out = write_solution(result, tmp_path / "portfolio")
    record = json.loads((out / "result.json").read_text())
    assert record["schema"] == "phasesmith.single-crystal-solution.v1"
    assert record["phasing_trials"][1]["hybrid_m"] == 3
    token = SolutionCancellation()
    token.cancel()
    cancelled = solve_structure(
        data,
        cell=cell,
        composition={"Si": 3},
        cancellation=token,
        options=SolutionOptions(phasing="dual_space"),
    )
    assert cancelled.termination == "cancelled" and not cancelled.phasing_trials
    with pytest.raises(ValueError):
        SolutionOptions(phasing="unknown")


def test_gaussian_volume_and_periodic_moments():
    from math import erf, exp, pi, sqrt

    shape = (64, 64, 64)
    cell = UnitCell(8, 8, 8, 90, 90, 90)
    sigma = 0.4
    projection = np.array(
        _core._solution_atomic_projection(
            np.ones(np.prod(shape)), shape, list(cell.as_tuple()), 1, sigma, 0.0, 0, 50
        )
    ).reshape(shape)
    volume = projection.sum() * (8 / 64) ** 3
    expected_volume = erf(3 / sqrt(2)) - sqrt(2 / pi) * 3 * exp(-4.5)
    assert abs(volume - expected_volume) < 0.004  # finite-grid quadrature at 0.125 Å
    xyz = np.moveaxis(np.indices(shape), 0, -1) * (8 / 64)
    xyz = (xyz + 4) % 8 - 4
    mean = np.einsum("ijk,ijkd->d", projection, xyz) / projection.sum()
    np.testing.assert_allclose(mean, 0, atol=2e-15)
    expected_variance = (
        sigma**2 * (expected_volume - sqrt(2 / pi) * 9 * exp(-4.5)) / expected_volume
    )
    variance = np.einsum("ijk,ijkd->d", projection, xyz**2) / projection.sum()
    np.testing.assert_allclose(variance, expected_variance, atol=0.002)
