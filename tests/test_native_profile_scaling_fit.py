"""Native optimizer controls against independent dense NumPy/SciPy equations."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.empirical_scaling import (
    ScalingDesign,
    ScalingOptions,
    fit_empirical_scale,
)
from phasesmith.single_crystal.profile_refinement import ShapeFitOptions, _Spot, refine_profiles
from phasesmith.single_crystal.reference.intensity_reference import (
    empirical_model,
    fit_empirical_scale_reference,
)
from phasesmith.single_crystal.reference.profile_fit_reference import ReferenceSpot
from test_experimental_profiles import fixed_model, observations

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_profile_refine"), reason="experimental native fit build"
)


@pytest.mark.parametrize("expected", [False, True])
@pytest.mark.parametrize("choice", [1, 2])
def test_profile_noisy_masked_fit_and_joint_sigma_against_independent_reference(expected, choice):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(105315)
    o = observations((20000.0,), covariance=True)
    samples = o.samples.copy()
    samples[:, 0] += rng.normal(0, 1.5, len(samples))
    valid = o.valid.copy()
    valid[17::47] = False
    table = replace(
        o.summation, noise_model="expected_counts" if expected else o.summation.noise_model
    )
    o = replace(
        o,
        summation=table,
        samples=samples,
        valid=valid,
        centres=o.centres + np.array([0.17, -0.13, 0.07 if choice == 1 else 0]),
    )
    model = fixed_model(o)
    opt = ShapeFitOptions()
    result = refine_profiles(o, model, np.array([choice]), options=opt)
    control = ReferenceSpot(o, model, int(result.baseline.model_ids[0]))
    native = _Spot(o, model, int(result.baseline.model_ids[0]))
    p = np.array([0.071, -0.092, 0.037, 0.041, -0.029])
    np.testing.assert_allclose(native.mass(p), control.mass(p), atol=2e-15, rtol=2e-12)
    native.set_noise(native.mass(p)[:, 0], 20000)
    control.set_noise(control.mass(p)[:, 0], 20000)
    actual = native.objective(p, np.array(opt.regularizer))
    expected_objective = control.objective(p, np.array(opt.regularizer))
    np.testing.assert_allclose(actual[0], expected_objective[0], atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(actual[1], expected_objective[1], atol=2e-7, rtol=1e-10)
    fit = control.fit(result.baseline.values[0, 0], choice, opt)
    np.testing.assert_allclose(result.parameters[0], fit.parameters, atol=3e-5, rtol=0)
    np.testing.assert_allclose(
        result.integration.values[0, :2], [fit.intensity, fit.sigma], rtol=2e-5, atol=2e-5
    )
    assert not result.integration.flags[0] & 16384


def test_worker_determinism_frozen_shapes_and_cancellation():
    o = observations((7000.0, 8000.0, 0.0, -20.0), covariance=True)
    model = fixed_model(o)
    choices = np.array([1, 2, 0, 0])
    weak = np.array([False, False, True, True])

    def run(workers, **kwargs):
        return refine_profiles(
            o,
            model,
            choices,
            weak_mask=weak,
            sensor_offsets=np.zeros((4, 2)),
            options=ShapeFitOptions(workers=workers),
            **kwargs,
        )

    a = run(1)
    for workers in [4, 6, 8, 10]:
        b = run(workers)
        assert a.integration.values.tobytes() == b.integration.values.tobytes()
        assert a.parameters.tobytes() == b.parameters.tobytes()
        assert a.converged.tobytes() == b.converged.tobytes()
    frozen = run(4, frozen_parameters=a.parameters)
    np.testing.assert_array_equal(a.integration.values, frozen.integration.values)
    events = []

    def cancel(done, total):
        events.append((done, total))
        raise RuntimeError("test cancellation")

    with pytest.raises(RuntimeError, match="test cancellation"):
        run(1, progress=cancel)
    assert events == [(1, 4)]


@pytest.mark.parametrize("seed", [45011, 58121, 93311])
def test_robust_scaling_randomized_outliers_holdouts_and_signed_rows(seed):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(seed)
    n, nf, nd, na = 240, 7, 5, 3
    matrices = [rng.normal(scale=0.15, size=(n, width)) for width in (nf, nd, na)]
    p = nf + nd + na
    design = ScalingDesign(
        *matrices,
        np.eye(p) * 0.03,
        np.full(p, -0.8),
        np.full(p, 0.8),
    )
    groups = np.repeat(np.arange(n // 4), 4)
    truth = rng.normal(scale=0.05, size=p)
    values = empirical_model(*matrices, truth)[0]
    intensity = np.exp(
        np.repeat(rng.uniform(5, 8, n // 4), 4) + values + rng.normal(scale=0.015, size=n)
    )
    intensity[::37] *= 2
    variance = np.full(n, 0.015**2)
    teachers = groups < 45
    intensity[-2:] = [-20, 0]
    options = ScalingOptions(strength=1000)
    native = fit_empirical_scale(design, intensity, variance, groups, teachers, options=options)
    reference = fit_empirical_scale_reference(
        design, intensity, variance, groups, teachers, options=options
    )
    assert native.converged and reference.converged
    np.testing.assert_array_equal(native.reference_mask, reference.reference_mask)
    np.testing.assert_allclose(native.coefficients, reference.coefficients, atol=2e-6, rtol=0)
    np.testing.assert_allclose(native.correction, reference.correction, atol=2e-6, rtol=0)
    assert np.all(native.correction > 0) and intensity[-2] < 0
    changed = intensity.copy()
    changed[~teachers] *= 1.5
    rerun = fit_empirical_scale(design, changed, variance, groups, teachers, options=options)
    np.testing.assert_array_equal(native.coefficients, rerun.coefficients)


def test_profile_closed_bound_and_failed_frozen_row_retains_baseline():
    pytest.importorskip("scipy")
    obs = observations((20000.0,), covariance=True)
    obs = replace(obs, centres=obs.centres + np.array([1.3, 0.0, 0.0]))
    model = fixed_model(obs)
    native = refine_profiles(obs, model, np.array([2]))
    control = ReferenceSpot(obs, model, int(native.baseline.model_ids[0]))
    fit = control.fit(native.baseline.values[0, 0], 2, ShapeFitOptions())
    np.testing.assert_allclose(native.parameters[0], fit.parameters, atol=3e-5, rtol=0)
    assert native.parameters[0, 0] == -1.0
    failed = refine_profiles(
        obs, model, np.array([1]), frozen_parameters=np.array([[1000.0, 1000.0, 0.0, 0.0, 0.0]])
    )
    assert failed.integration.flags[0] & 16384 and not failed.converged[0]
    np.testing.assert_array_equal(failed.integration.values, failed.baseline.values)
    assert failed.parameters[0, 0] == 1000.0


def test_scaling_invalid_teachers_variance_bounds_and_effort_status():
    n = 8
    design = ScalingDesign(
        np.arange(n, dtype=float)[:, None] / n,
        np.empty((n, 0)),
        np.empty((n, 0)),
        np.eye(1),
        np.array([-0.5]),
        np.array([0.5]),
    )
    groups = np.repeat(np.arange(4), 2)
    mask = np.ones(n, bool)
    intensity = np.exp(np.arange(n, dtype=float) / n * 0.1 + np.repeat(np.arange(4), 2))
    for variance, y, start in [
        (np.zeros(n), intensity, None),
        (np.ones(n), -intensity, None),
        (np.ones(n), intensity, np.array([0.6])),
    ]:
        with pytest.raises(ValueError):
            fit_empirical_scale(design, y, variance, groups, mask, start=start)
    limited = fit_empirical_scale(
        design,
        intensity,
        np.full(n, 0.01),
        groups,
        mask,
        options=ScalingOptions(strength=0, iterations=1),
    )
    assert not limited.converged and len(limited.history) == 1
