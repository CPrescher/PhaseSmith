"""Independent equations, signed known-count controls and cache/provenance safety."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.empirical_scaling import (
    ScalingDesign,
    ScalingOptions,
    family_center,
    fit_empirical_scale,
    panel_mesh,
)
from phasesmith.single_crystal.intensity_response import SlabResponse
from phasesmith.single_crystal.profile_refinement import ShapeFitOptions, _Spot, refine_profiles
from phasesmith.single_crystal.profiles import evaluate_profile, evaluate_profile_shape
from phasesmith.single_crystal.reduction import (
    ReductionConfig,
    physical_factors,
    reduce_observations,
)
from phasesmith.single_crystal.reduction_cache import ReductionCache, cache_key
from phasesmith.single_crystal.reduction_io import load_case, save_case
from phasesmith.single_crystal.reference import intensity_reference as reference
from phasesmith.single_crystal.sensor_offsets import SensorLayout, fit_sensor_offsets
from test_experimental_profiles import fixed_model, observations, truth

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_fixed_profile"), reason="experimental intensity build"
)


@pytest.mark.parametrize("mu", [1e-9, 0.122453913, 8.9994525457, 1000.0])
def test_response_reference_and_derivatives(mu):
    c = np.linspace(0.51, 1.0, 97)
    slab = SlabResponse(0.75, mu, 0.28460, "explicit slab", "test calculation", 5.85)
    actual = slab.evaluate(c)
    np.testing.assert_allclose(actual, reference.slab_response(c, 0.75, mu), rtol=2e-11, atol=2e-12)
    for k in range(3):
        fd = (slab.evaluate(c[:-1] + 1e-6)[:, k] - slab.evaluate(c[:-1] - 1e-6)[:, k]) / 2e-6
        np.testing.assert_allclose(fd, actual[:-1, k + 3], rtol=1e-6, atol=2e-8)
    assert not actual.flags.writeable
    with pytest.raises(ValueError):
        slab.evaluate(np.array([0.0]))
    with pytest.raises(ValueError):
        SlabResponse(0.75, mu, 0.28460, "CdTe", "")


def test_shape_mass_normalization_and_fused_width_derivatives():
    rng = np.random.default_rng(2910202611)
    p = truth()
    b = rng.uniform(-3.1, 1.8, (173, 6))
    b[:, 1::2] = b[:, ::2] + rng.uniform(0.2, 1.7, (len(b), 3))
    step = np.array([0.71, 1.13, 0.47])
    centre = [0.163, -0.271, 0.319]
    for affine, scales in [
        (np.eye(2), [1.0, 1.0, 1.0]),
        (np.array([[0.9, -0.27], [0.13, 1.1]]), [1.0, 1.0, -0.83]),
    ]:
        actual = evaluate_profile_shape(
            p, b, step=step, centre=centre, scales=scales, planar_transform=affine
        )
        control = evaluate_profile(
            p, b, step=step, centre=centre, scales=scales, planar_transform=affine
        )
        np.testing.assert_array_equal(actual[:, :4], control)
        for k, dilation in [(3, [1, 1, 0]), (4, [0, 0, 1])]:
            plus = step * np.exp(np.array(dilation) * 1e-6)
            minus = step * np.exp(-np.array(dilation) * 1e-6)
            fd = (
                evaluate_profile(
                    p, b, step=plus, centre=centre, scales=scales, planar_transform=affine
                )[:, 0]
                - evaluate_profile(
                    p, b, step=minus, centre=centre, scales=scales, planar_transform=affine
                )[:, 0]
            ) / 2e-6
            np.testing.assert_allclose(actual[:, k + 1], fd, atol=6e-10, rtol=2e-7)
    full = evaluate_profile_shape(p, np.array([[-20.0, 20.0, -20.0, 20.0, -20.0, 20.0]]))
    np.testing.assert_allclose(full, [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], atol=3e-15)
    outside = evaluate_profile_shape(p, np.array([[10.0, 11.0, 10.0, 11.0, 10.0, 11.0]]))
    np.testing.assert_array_equal(outside, np.zeros((1, 6)))
    for bad in (np.zeros((3, 3, 2)), -p):
        with pytest.raises(ValueError):
            evaluate_profile_shape(bad, b)


def test_native_gls_and_fixed_filter_against_dense_independent_reference():
    o = observations((150.0,), covariance=True)
    m = fixed_model(o)
    spot = _Spot(o, m, 1)
    p = spot.mass(np.zeros(5))
    spot.set_noise(p[:, 0], 1000.0)
    w = reference.profile_precision(spot.weights, spot.basis, spot.offsets, spot.cov)
    design = np.c_[p, spot.signal]
    expected = design.T @ w @ design
    np.testing.assert_allclose(spot.normal(p), expected, rtol=3e-12, atol=2e-10)
    filt = w @ p[:, 0] / (p[:, 0] @ w @ p[:, 0])
    cov = np.diag(o.samples[:, 1])
    for g, (lo, hi) in enumerate(zip(spot.offsets[:-1], spot.offsets[1:], strict=True)):
        cov[lo:hi, lo:hi] += spot.basis[lo:hi] @ spot.cov[g].reshape(3, 3) @ spot.basis[lo:hi].T
    result = _core.experimental_fixed_profile(
        spot.signal,
        np.ascontiguousarray(o.samples[:, 1]),
        spot.weights,
        spot.basis,
        spot.offsets,
        spot.cov,
        p[:, 0].copy(),
    )
    np.testing.assert_allclose(
        result, [filt @ spot.signal, np.sqrt(filt @ cov @ filt)], rtol=2e-12, atol=2e-10
    )
    # Complete objective derivatives include both profile and signed-amplitude elimination.
    parameters = np.array([0.071, -0.092, 0.037, 0.041, -0.029])
    value, jac = spot.objective(parameters, np.zeros(5))
    for k in range(5):
        d = np.eye(5)[k] * 1e-6
        fd = (
            spot.objective(parameters + d, np.zeros(5))[0]
            - spot.objective(parameters - d, np.zeros(5))[0]
        ) / 2e-6
        np.testing.assert_allclose(jac[k], fd, atol=3e-6, rtol=3e-6)
    assert np.isfinite(value)


def test_signed_weak_reflections_and_strong_fit_no_population_change():
    o = observations((5000.0, -20.0, 0.0), covariance=True)
    m = fixed_model(o)
    options = ShapeFitOptions(weak_spatial_width=1.0)
    result = refine_profiles(
        o,
        m,
        np.array([2, 0, 0]),
        weak_mask=np.array([False, True, True]),
        sensor_offsets=np.zeros((3, 2)),
        options=options,
    )
    assert len(result.integration.values) == 3
    np.testing.assert_allclose(result.integration.values[:, 0], [5000.0, -20.0, 0.0], atol=1e-8)
    assert np.all(result.integration.values[:, 1] > 0)
    assert result.integration.summation is o.summation
    np.testing.assert_allclose(result.parameters[1, :3], 0.0, atol=1e-16)
    with pytest.raises(ValueError, match="require"):
        refine_profiles(o, m, np.zeros(3, int), weak_mask=np.ones(3, bool))


def test_empirical_partials_centering_and_recovery_without_structural_targets():
    rng = np.random.default_rng(58211)
    n, nf = 200, 5
    frames = rng.normal(size=(n, nf))
    detector = rng.normal(scale=0.1, size=(n, 2))
    harmonic = rng.normal(scale=0.1, size=(n, 3))
    design = ScalingDesign(
        frames, detector, harmonic, np.eye(10) * 0.01, np.full(10, -0.5), np.full(10, 0.5)
    )
    b = rng.normal(scale=0.04, size=10)
    actual = design.evaluate(b)
    control = reference.empirical_model(frames, detector, harmonic, b)
    for a, e in zip(actual, control, strict=True):
        np.testing.assert_allclose(a, e, rtol=2e-13, atol=2e-15)
    for k in range(10):
        d = np.eye(10)[k] * 1e-6
        np.testing.assert_allclose(
            actual[2][:, k],
            (design.evaluate(b + d)[0] - design.evaluate(b - d)[0]) / 2e-6,
            atol=2e-10,
        )
    groups = np.repeat(np.arange(40), 5)
    weight = rng.uniform(0.2, 3, n)
    np.testing.assert_allclose(
        family_center(frames, groups, weight),
        reference.family_center(frames, groups, weight),
        atol=2e-15,
    )
    intensity = np.repeat(rng.uniform(100.0, 1000.0, 40), 5) * np.exp(actual[0])
    intensity[-1] = -12.0
    training = np.ones(n, bool)
    training[-5:] = False
    fit = fit_empirical_scale(
        design,
        intensity,
        np.full(n, 0.02**2),
        groups,
        training,
        options=ScalingOptions(strength=1.0),
    )
    assert fit.converged
    np.testing.assert_allclose(fit.coefficients, b, atol=2e-7)
    corrected = fit.apply(intensity, np.ones(n))
    assert corrected[-1, 0] < 0 and len(corrected) == n
    assert not fit.reference_mask[-5:].any()
    with pytest.raises(ValueError):
        fit_empirical_scale(design, intensity, np.ones(n), groups, np.ones(n, bool))
    mesh = panel_mesh(np.array([[0.0, 0.0], [199.0, 49.0], [100.0, 20.0]]), (50, 200))
    np.testing.assert_allclose(mesh.sum(axis=1), 1.0, atol=1e-15)


def test_sensor_model_known_shifts_rotation_shared_terms_and_holdout():
    rng = np.random.default_rng(1651)
    layout = SensorLayout([[0, 0, 100, 100], [100, 0, 200, 100]], [[50, 50], [150, 50]], 50.0)
    xy = np.r_[rng.uniform([1, 1], [99, 99], (80, 2)), rng.uniform([101, 1], [199, 99], (80, 2))]
    b = np.array([0.3, -0.1, -0.2, 0.1, 0.04, -0.03, 0.02, -0.01, 0.015])
    shift = np.einsum("nip,p->ni", layout.design(xy), b)
    training = np.arange(160) % 5 != 0
    fit = fit_sensor_offsets(
        layout,
        xy,
        shift,
        training,
        provenance="known synthetic pixel centroids",
        regularization=0.0,
    )
    np.testing.assert_allclose(fit.coefficients, b, atol=3e-15)
    changed = shift.copy()
    changed[~training] += 1000
    other = fit_sensor_offsets(
        layout, xy, changed, training, provenance="holdout test", regularization=0.0
    )
    np.testing.assert_array_equal(other.coefficients, fit.coefficients)
    with pytest.raises(ValueError):
        layout.sections([[300.0, 10.0]])


def test_safe_case_roundtrip_cached_pipeline_and_invalidation(tmp_path):
    o = observations((150.0, -20.0), covariance=True)
    m = fixed_model(o)
    path = tmp_path / "case.npz"
    save_case(path, o, m)
    oo, mm = load_case(path)
    assert oo.fingerprint() == o.fingerprint()
    np.testing.assert_array_equal(mm.profiles, m.profiles)
    with pytest.raises(FileExistsError):
        save_case(path, o, m)
    config = ReductionConfig(profile=m.options)
    first = reduce_observations(o, m.detector_shape, config, model=m, cache=tmp_path / "cache")
    second = reduce_observations(o, m.detector_shape, config, model=m, cache=tmp_path / "cache")
    assert not first.metadata["cache_hit"] and second.metadata["cache_hit"]
    np.testing.assert_array_equal(first.intensity_sigma, second.intensity_sigma)
    assert first.aperture is o.summation
    assert len(first.intensity_sigma) == 2 and first.intensity_sigma[1, 0] < 0
    changed = reduce_observations(
        o, m.detector_shape, replace(config, strong_snr=60.0), model=m, cache=tmp_path / "cache"
    )
    assert changed.metadata["profile_cache_key"] != first.metadata["profile_cache_key"]
    cache = ReductionCache(tmp_path / "cache")
    key = cache_key("test", {}, {"x": np.arange(3)}, implementation="v1")
    cache.store(key, {"x": np.arange(3)})
    cache.store(key, {"x": np.arange(3)})
    with pytest.raises(ValueError, match="conflicting"):
        cache.store(key, {"x": np.arange(4)})
    (cache._path(key) / "arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache.load(key)
    assert key != cache_key("test", {}, {"x": np.arange(3)}, implementation="v2")
    assert asdict(ReductionConfig.from_dict(asdict(config))) == asdict(config)


def test_physical_lp_axis_wavelength_and_scattering_map_contract():
    from phasesmith.single_crystal.reciprocal import PanelGeometry
    from phasesmith.single_crystal.reduction import profile_geometry

    g = PanelGeometry(
        np.array([[-0.05, -0.04, 0.15], [0.000075, 0, 0], [0, 0.000075, 0], [0, 0, 1], [1, 0, 0]]),
        0.28460,
    )
    xy = np.array([[200.0, 120.0, 0.0], [300.0, 100.0, 2.0]])
    config = ReductionConfig(polarization_axis=(0.0, 1.0, 0.0))
    ray = profile_geometry(xy[:, :2], g)[0]
    factor, applied = physical_factors(xy, g, config)
    expected = abs(np.cross(ray, [0, 0, 1]) @ np.array([1, 0, 0])) / (
        0.98 * (1 - ray[:, 1] ** 2) + 0.02 * (1 - ray[:, 0] ** 2)
    )
    np.testing.assert_allclose(factor, expected, atol=1e-16)
    assert len(applied) == 1
    with pytest.raises(ValueError, match="wavelength"):
        physical_factors(
            xy, g, replace(config, response=SlabResponse(0.75, 9.0, 0.3, "CdTe", "test"))
        )


def test_equivalent_error_recovery_derivative_and_whole_family_holdout():
    from phasesmith.single_crystal.intensity_uncertainty import estimate_equivalent_error

    groups = np.repeat(np.arange(50), 2)
    mu = np.repeat(np.linspace(200.0, 1000.0, 50), 2)
    sigma = np.full(100, 3.0)
    beta = 0.05
    half_difference = 0.6744897501960817 / np.sqrt(2) * np.hypot(sigma, beta * mu)
    intensity = mu + np.tile([-1, 1], 50) * half_difference
    train = groups % 5 != 0
    model = estimate_equivalent_error(intensity, sigma, groups, np.ones(100, bool), train)
    np.testing.assert_allclose(model.relative, beta, atol=2e-15)
    np.testing.assert_allclose(model.sigma, np.hypot(sigma, beta * mu), atol=1e-12)
    h = 1e-6
    fd = (np.hypot(sigma, (beta + h) * mu) - np.hypot(sigma, (beta - h) * mu)) / (2 * h)
    np.testing.assert_allclose(model.d_sigma_d_relative, fd, atol=1e-7)
    changed = intensity.copy()
    changed[~train] *= 2
    other = estimate_equivalent_error(changed, sigma, groups, np.ones(100, bool), train)
    assert model.relative == other.relative
    bad = train.copy()
    bad[2] = False
    with pytest.raises(ValueError, match="complete families"):
        estimate_equivalent_error(intensity, sigma, groups, np.ones(100, bool), bad)


def test_strong_shape_fit_recovers_known_position_without_an_atomic_model():
    o = observations((50000.0,), covariance=True)
    m = fixed_model(o)
    shifted = replace(o, centres=o.centres + np.array([[0.17, -0.13, 0.08]]))
    fitted = refine_profiles(shifted, m, np.array([1]))
    np.testing.assert_allclose(fitted.parameters[0, :3], [-0.17, 0.13, -0.08], atol=2e-5)
    np.testing.assert_allclose(fitted.parameters[0, 3:], 0.0, atol=2e-5)
    np.testing.assert_allclose(fitted.integration.values[0, 0], 50000.0, rtol=2e-6)
    assert fitted.integration.values[0, 1] > 0


def test_declared_recipe_fields_and_underflow_are_rejected_early():
    with pytest.raises(ValueError, match="underflows"):
        SlabResponse(1e-300, 1e-300, 0.3, "explicit", "test")
    with pytest.raises(ValueError, match="unknown fields"):
        ReductionConfig(empirical={"option": {}})
    with pytest.raises(ValueError, match="sensor recipe"):
        ReductionConfig(sensor={"coefficients": []})
    layout = SensorLayout([[0, 0, 100, 100]], [[50, 50]])
    from phasesmith.single_crystal.sensor_offsets import SensorOffsets

    sensor = SensorOffsets(layout, np.zeros(layout.parameters), "synthetic explicit recipe")
    config = ReductionConfig(sensor=sensor.to_dict())
    assert config.sensor["layout"]["bounds"] == [[0.0, 0.0, 100.0, 100.0]]


def test_sensor_penalty_probes_are_declared_independently_of_assignment_bounds():
    rng = np.random.default_rng(301020264)
    layout = SensorLayout([[0, 0, 100, 110], [100, 0, 210, 110]], [[50, 50], [150, 50]], 50.0)
    xy = np.r_[rng.uniform([1, 1], [99, 99], (40, 2)), rng.uniform([101, 1], [199, 99], (40, 2))]
    target = rng.normal(0.0, 0.2, (80, 2))
    training = np.arange(80) % 4 != 0
    offsets = np.tile([[45.0, 0.0], [-45.0, 0.0], [0.0, 42.0], [0.0, -42.0]], (2, 1, 1))
    fitted = fit_sensor_offsets(
        layout,
        xy,
        target,
        training,
        provenance="declared probe control",
        huber_pixels=1e6,
        regularization=2.0,
        iterations=1,
        penalty_offsets=offsets,
    )
    centre = np.repeat(layout.centres, 4, axis=0)
    sections = np.repeat(np.arange(2), 4)
    q = (
        layout.design(centre + offsets.reshape(-1, 2), sections) - layout.design(centre, sections)
    ).reshape(-1, layout.parameters)
    design = layout.design(xy[training]).reshape(-1, layout.parameters)
    reference = np.linalg.solve(
        design.T @ design + 2 * q.T @ q, design.T @ target[training].ravel()
    )
    np.testing.assert_allclose(fitted.coefficients, reference, atol=3e-15, rtol=3e-15)
    with pytest.raises(ValueError, match="sections,4,2"):
        fit_sensor_offsets(
            layout, xy, target, training, provenance="invalid", penalty_offsets=np.zeros((2, 2, 2))
        )
    with pytest.raises(ValueError, match="span"):
        fit_sensor_offsets(
            layout, xy, target, training, provenance="invalid", penalty_offsets=np.zeros((2, 4, 2))
        )
