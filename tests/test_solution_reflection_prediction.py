"""Prediction provenance, independent equations and complete Fourier-cycle parity."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_prediction_reference import (
    coefficient_reference,
    prediction_update_reference,
    statistics_reference,
)
from phasesmith.experimental.solution_quality_reference import (
    reciprocal_s_squared,
    wilson_reference,
)
from phasesmith.scattering_reference import xray_non_resonant
from test_solution_phase_quality import case

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_ReflectionPredictionPlan"), reason="experimental-solution"
)


def make(symbol="P1", missing=True, extension=1.2, power=1.0):
    cell, group, rotations, h, i, s = case(symbol)
    plan = _core._ReflectionPredictionPlan(
        [32] * 3,
        list(cell.as_tuple()),
        group._native,
        h.ravel(),
        i,
        s,
        ["Fe", "O"],
        missing,
        extension,
        power,
    )
    return plan, cell, group, rotations


@pytest.mark.parametrize("seed", range(6))
def test_coefficient_values_and_fused_jacobian(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=2)
    for c in [0.0, 0.3, 1.0]:
        native = _core._prediction_coefficient(z, 1.3, 0.8, c, 0.7)
        expected = coefficient_reference(z, 1.3, 0.8, c, 0.7)
        np.testing.assert_allclose(native["value"], expected, rtol=3e-14, atol=3e-15)
        for a in range(2):
            delta = np.eye(2)[a] * 1e-7
            fd = (
                coefficient_reference(z + delta, 1.3, 0.8, c, 0.7)
                - coefficient_reference(z - delta, 1.3, 0.8, c, 0.7)
            ) / 2e-7
            np.testing.assert_allclose(
                np.asarray(native["jacobian"])[:, a], fd, rtol=3e-6, atol=2e-9
            )
        assert np.linalg.norm(native["value"]) <= 4 * 0.7 * c * 1.3 + 1e-14
    assert _core._prediction_coefficient([0.0, 0.0], 1.0, 1.0, 0.5, 1.0)["value"] == [0.0, 0.0]


@pytest.mark.parametrize("symbol", ["P1", "P21", "Fm-3m", "P 31 2 1"])
@pytest.mark.parametrize("power", [0.5, 1.0])
def test_envelope_extrapolation_matches_independent_wilson_and_support(symbol, power):
    plan, cell, group, rotations = make(symbol, power=power)
    meta = plan.metadata()
    h = meta["hkl"]
    x = reciprocal_s_squared(cell.as_tuple(), h)
    f, _ = xray_non_resonant(["Fe", "O"], np.sqrt(x))
    norm = wilson_reference(h, meta["intensity"], cell.as_tuple(), rotations, np.sum(f * f, axis=1))
    predicted = np.asarray(meta["predicted"])
    assert meta["extended_pairs"] > 0 and meta["missing_pairs"] >= 0
    index = predicted[:, 0].astype(int)
    coordinates = np.asarray(np.unravel_index(index, (32,) * 3)).T
    ph = (coordinates + 16) % 32 - 16
    px = reciprocal_s_squared(cell.as_tuple(), ph)
    pf, _ = xray_non_resonant(["Fe", "O"], np.sqrt(px))
    eps = np.sum(np.all(np.einsum("hi,gij->hgj", ph, rotations) == ph[:, None], axis=-1), axis=1)
    expectation = (
        eps
        * np.sum(pf * pf, axis=1)
        * np.exp(
            norm["log_scale"]
            + norm["slope"] * px
            + np.interp(px, norm["shell_x"], norm["shell_residuals"])
        )
    )
    np.testing.assert_allclose(
        predicted[:, 3], expectation ** (0.5 * power) / meta["observed_rms"], rtol=5e-10, atol=1e-12
    )
    assert not np.any(group.systematic_absences(ph))
    assert np.all(np.abs(ph) < 16)
    assert meta["measured_rows"] == len(h)
    assert np.count_nonzero(meta["support"] == 2) == len(h)
    assert meta["support"][0] == 1


@pytest.mark.parametrize("missing,extension", [(True, 1.0), (False, 1.2), (True, 1.2)])
def test_update_preserves_measurements_zeros_dc_and_hermitian_pairs(missing, extension):
    plan, *_ = make(missing=missing, extension=extension)
    meta = plan.metadata()
    rng = np.random.default_rng(315)
    coefficients = np.fft.fftn(rng.normal(size=(32,) * 3)).ravel()
    coefficients[meta["support"] == 0] = 0
    for h, a in zip(meta["hkl"], meta["amplitudes"], strict=True):
        index = np.ravel_multi_index(h % 32, (32,) * 3)
        coefficients[index] = a / meta["observed_rms"] * np.exp(1j * 0.13 * (h @ [1, 3, 7]))
    original = coefficients.copy()
    for cycle in [0, 8, 24]:
        value = plan.apply(coefficients.real.copy(), coefficients.imag.copy(), cycle)
        expected = prediction_update_reference(
            coefficients, meta["predicted"], meta["measured"], cycle
        )
        np.testing.assert_allclose(
            value["real"] + 1j * value["imag"], expected, rtol=3e-10, atol=3e-11
        )
        np.testing.assert_allclose(
            value["statistics"],
            statistics_reference(coefficients, meta["measured"]),
            rtol=3e-10,
            atol=3e-11,
        )
        actual = value["real"] + 1j * value["imag"]
        np.testing.assert_array_equal(actual[meta["support"] != 1], original[meta["support"] != 1])
        assert actual[0] == original[0]
        for h in meta["hkl"]:
            index = np.ravel_multi_index(h % 32, (32,) * 3)
            mate = np.ravel_multi_index((-h) % 32, (32,) * 3)
            assert actual[index] == pytest.approx(actual[mate].conjugate(), abs=1e-12)
    if not missing:
        assert meta["missing_pairs"] == 0
    if extension == 1.0:
        assert meta["extended_pairs"] == 0


def test_complete_prediction_cycles_against_independent_numpy():
    from phasesmith.experimental.solution_dual_reference import dual_space_reference
    from phasesmith.io.space_groups import space_group_by_symbol
    from test_solution_dual_space import inputs

    shape, cell, h, amplitudes, phases, _ = inputs()
    plan = _core._ReflectionPredictionPlan(
        shape,
        list(cell.as_tuple()),
        space_group_by_symbol("P1").space_group._native,
        h.ravel(),
        amplitudes**2,
        np.ones(len(h)),
        ["Fe", "O"],
        True,
        1.2,
        0.5,
    )
    meta = plan.metadata()
    native = plan.dual(phases, 20, 3, 71)

    def predict(values, cycle):
        return prediction_update_reference(values, meta["predicted"], meta["measured"], cycle)

    reference = dual_space_reference(
        shape,
        cell.geometry().direct_basis,
        meta["hkl"],
        meta["amplitudes"],
        phases,
        meta["support"],
        cycles=20,
        atom_count=3,
        sigma=0.25,
        seed=71,
        prediction=predict,
    )
    np.testing.assert_allclose(native["history"], reference["history"], rtol=3e-8, atol=3e-10)
    np.testing.assert_allclose(
        native["density"].reshape(shape), reference["density"], rtol=3e-7, atol=3e-9
    )
    assert native["updates"] > 0


@pytest.mark.parametrize("missing,extension", [(True, 1.0), (False, 1.2), (True, 1.2)])
@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_preserves_observations_and_recovers_synthetic_counts(
    symbol, missing, extension
):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, xyz, species = synthetic(symbol)
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=32,
        symmetry_recycling=True,
        tangent_refinement=True,
        wilson_normalization=True,
        phase_quality_selection=True,
        phase_origin_selection=True,
        chemical_assignment=True,
        iterative_refinement=True,
        missing_reflections=missing,
        resolution_extension=extension,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    np.testing.assert_array_equal(result.expanded_hkl, data.hkl)
    np.testing.assert_allclose(result.merged_intensity, data.intensity, rtol=2e-14, atol=1e-12)
    assert result.shared_search.prediction.measured_rows == len(data.hkl)
    assert result.shared_search.prediction_steps > 0
    assert result.shared_search.prediction_updates > 0
    best_error = float("inf")
    for hand in [-1, 1]:
        for anchor in xyz[species == result.best.species[0]]:
            candidate = (
                hand * result.best.fractional_xyz + anchor - hand * result.best.fractional_xyz[0]
            ) % 1
            error = 0.0
            for element in set(species):
                delta = (
                    xyz[species == element, None]
                    - candidate[None, np.asarray(result.best.species) == element]
                    + 0.5
                ) % 1 - 0.5
                distances = np.linalg.norm(delta @ cell.geometry().direct_basis.T, axis=-1)
                error = max(error, distances.min(axis=0).max(), distances.min(axis=1).max())
            best_error = min(best_error, error)
    assert best_error < 1e-5


@pytest.mark.parametrize("extension", [0.9, 1.51, float("nan")])
def test_prediction_rejects_invalid_extension(extension):
    with pytest.raises(ValueError):
        make(extension=extension)


def test_capped_and_zero_coefficient_derivatives_and_scale_equivariance():
    for z, c in [(np.array([7.0, 9.0]), 0.7), (np.zeros(2), 0.5), (np.zeros(2), 1.0)]:
        value = _core._prediction_coefficient(z, 1.0, 1.0, c, 1.0)
        for axis in range(2):
            step = 1e-9 if np.all(z == 0) and c == 1.0 else 1e-7
            d = np.eye(2)[axis] * step
            fd = (
                coefficient_reference(z + d, 1.0, 1.0, c, 1.0)
                - coefficient_reference(z - d, 1.0, 1.0, c, 1.0)
            ) / (2 * step)
            np.testing.assert_allclose(
                np.asarray(value["jacobian"])[:, axis], fd, rtol=2e-5, atol=2e-7
            )
        scaled = _core._prediction_coefficient(z * 31, 31.0, 1.0, c, 1.0)
        np.testing.assert_allclose(
            scaled["value"], np.asarray(value["value"]) * 31, rtol=1e-13, atol=1e-13
        )
