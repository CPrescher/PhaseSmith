"""Known finite voxel truth, independent equations, signed fits and training isolation."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.integration import IntegrationOptions, ReflectionTable
from phasesmith.single_crystal.profiles import (
    ProfileModel,
    ProfileObservations,
    ProfileOptions,
    evaluate_profile,
    fit_profiles,
    learn_profiles,
)
from phasesmith.single_crystal.reference import profiles_reference as reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_profile_fit"), reason="experimental profiles build"
)


def truth():
    z, y, x = np.indices((5, 5, 5))
    p = np.exp(-0.5 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2))
    p[0] = 0
    p[-1] = 0
    p[:, 0] = 0
    p[:, -1] = 0
    p[:, :, 0] = 0
    p[:, :, -1] = 0
    return p / p.sum()


def observations(
    amplitudes=(5000.0, 6000.0, 7000.0, 8000.0, 100.0, -20.0), *, covariance=False, spacing=10
):
    p = truth()
    rng = np.random.default_rng(23451)
    n = len(amplitudes)
    centres = np.column_stack(
        [
            20 + np.arange(n) * spacing + rng.uniform(-0.3, 0.3, n),
            np.full(n, 20.0) + rng.uniform(-0.3, 0.3, n),
            rng.uniform(-0.3, 0.3, n),
        ]
    )
    boxes = []
    angles = []
    ids = []
    frame = []
    offsets = [0]
    bounds = np.column_stack([np.arange(7) - 3.5, np.arange(7) - 2.5])
    for f, b in enumerate(bounds):
        for i, c in enumerate(centres):
            cx, cy = np.floor(c[:2] + 0.5).astype(int)
            boxes.append([cx - 3, cy - 3, cx + 4, cy + 4])
            angles.append(b)
            ids.append(i)
            frame.append(f)
            offsets.append(offsets[-1] + 49)
    boxes = np.array(boxes)
    angles = np.array(angles)
    ids = np.array(ids)
    offsets = np.array(offsets)
    b, _ = reference.sample_bounds(offsets, boxes, angles)
    data = np.zeros(len(b))
    for i in range(n):
        sel = np.repeat(ids == i, 49)
        data[sel] = amplitudes[i] * reference.evaluate(p, b[sel], centre=centres[i])[:, 0]
    var = np.full(len(b), 4.0)
    cov = np.tile(np.diag([0.1, 0.05, 0.07]).ravel() if covariance else np.zeros(9), (len(ids), 1))
    fv = np.zeros((len(ids), 11))
    fv[:, 9] = 49
    fv[:, 10] = 100
    t = ReflectionTable(
        np.column_stack([np.arange(n) + 1, np.zeros((n, 2), int)]),
        centres,
        np.zeros(n, np.int64),
        np.zeros((n, 6)),
        np.zeros(n, np.uint32),
        ids,
        np.array(frame),
        boxes,
        fv,
        np.zeros(len(ids), np.uint32),
        bounds,
        tuple(str(i) for i in range(7)),
        IntegrationOptions(),
        "supplied_independent_pixel_variance",
        (1.0, 0.0, None),
    )
    return ProfileObservations(
        t,
        np.column_stack([data, var, np.full(len(b), 20.0)]),
        np.ones(len(b), bool),
        offsets,
        cov,
        centres,
    )


def fixed_model(o, p=None):
    p = truth() if p is None else p
    opt = ProfileOptions(shape=p.shape[::-1], regions=(1, 1, 1), min_references=2)
    return ProfileModel(
        np.tile(p.ravel(), (2, 1)),
        np.array([4, 4]),
        np.zeros((2, 3)),
        np.arange(min(4, len(o.centres))),
        o.fingerprint(),
        (50, 200),
        (-3.5, 3.5),
        opt,
    )


def test_integral_mass_partial_exposures_and_derivatives():
    p = truth()
    rng = np.random.default_rng(983)
    b = rng.uniform(-4, 3, (80, 6))
    b[:, 1::2] = b[:, ::2] + rng.uniform(0.2, 2, (80, 3))
    centre = np.array([0.163, -0.291, 0.317])
    step = (0.7, 1.2, 0.5)
    actual = evaluate_profile(p, b, centre=centre, step=step)
    expected = reference.evaluate(p, b, step, centre)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-15)
    for j in range(3):
        plus = centre.copy()
        minus = centre.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        fd = (
            evaluate_profile(p, b, centre=plus, step=step)[:, 0]
            - evaluate_profile(p, b, centre=minus, step=step)[:, 0]
        ) / 2e-6
        np.testing.assert_allclose(fd, actual[:, j + 1], atol=5e-10)
    full = evaluate_profile(p, np.array([[-10, 10, -10, 10, -10, 10.0]]), centre=centre)
    np.testing.assert_allclose(full, [[1, 0, 0, 0]], atol=1e-15)
    assert not actual.flags.writeable


def test_signed_gls_and_shared_background_covariance():
    o = observations((150.0, -20.0), covariance=True)
    m = fixed_model(o)
    result = fit_profiles(o, m, expected_noise=False)
    expected = reference.fit(
        o.samples,
        o.valid,
        o.offsets,
        o.summation.boxes,
        o.summation.observation_ids,
        np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1),
        o.covariance,
        o.centres,
        m.volumes,
        np.ones(2, int),
        m.shifts,
    )
    np.testing.assert_allclose(result.values, expected, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(result.values[:, 0], [150, -20], atol=1e-10)
    assert not np.any(result.flags)
    no_cov = fit_profiles(
        replace(o, covariance=np.zeros_like(o.covariance)), m, expected_noise=False
    )
    assert np.all(result.values[:, 1] > no_cov.values[:, 1])
    assert not result.values.flags.writeable


def test_fitted_objective_shape_derivatives():
    o = observations((150.0,), covariance=True)
    m = fixed_model(o)
    shifted = o.centres.copy()
    shifted[0] += [0.07, -0.08, 0.03]
    o = replace(o, centres=shifted)
    result = fit_profiles(o, m, expected_noise=False)
    dof = result.values[0, 4] - 1
    for j in range(3):
        plus = o.centres.copy()
        minus = o.centres.copy()
        plus[0, j] += 1e-5
        minus[0, j] -= 1e-5
        a = fit_profiles(replace(o, centres=plus), m, expected_noise=False).values[0, 3] * dof
        b = fit_profiles(replace(o, centres=minus), m, expected_noise=False).values[0, 3] * dof
        assert (a - b) / 2e-5 == pytest.approx(result.values[0, 7 + j], rel=2e-6, abs=1e-5)


def test_learning_normalization_and_no_heldout_leakage():
    o = observations()
    mask = np.array([True] * 4 + [False] * 2)
    opts = ProfileOptions(
        shape=(5, 5, 5),
        regions=(1, 1, 1),
        min_references=3,
        reference_snr=5,
        iterations=30,
        cutoff=0,
    )
    model = learn_profiles(o, mask, detector_shape=(50, 200), options=opts)
    np.testing.assert_allclose(model.profiles.sum(axis=1), 1, atol=1e-14)
    assert np.all(model.counts == 4)
    np.testing.assert_array_equal(model.reference_ids, np.arange(4))
    data = o.samples.copy()
    held = np.repeat(o.summation.observation_ids >= 4, np.diff(o.offsets))
    data[held, 0] *= 10000
    other = learn_profiles(replace(o, samples=data), mask, detector_shape=(50, 200), options=opts)
    assert model.profiles.tobytes() == other.profiles.tobytes()
    assert model.shifts.tobytes() == other.shifts.tobytes()
    fit = fit_profiles(o, model, expected_noise=False)
    np.testing.assert_allclose(fit.values[4:, 0], [100, -20], rtol=0.03)
    np.testing.assert_array_equal(fit.reference_membership, mask)
    assert not model.profiles.flags.writeable


def test_independent_dense_training_solver():
    o = observations((5000.0, 6000.0, 7000.0))
    mask = np.ones(3, bool)
    opts = ProfileOptions(
        shape=(5, 5, 5),
        regions=(1, 1, 1),
        min_references=2,
        reference_snr=5,
        iterations=12,
        cutoff=0,
    )
    model = learn_profiles(o, mask, detector_shape=(50, 200), options=opts)
    p, shift = reference.learn_one(
        o.samples,
        o.valid,
        o.offsets,
        o.summation.boxes,
        o.summation.observation_ids,
        np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1),
        o.centres,
        mask,
        opts.shape,
        np.array(opts.step),
        opts.iterations,
    )
    np.testing.assert_allclose(model.profiles[0], p, atol=2e-15, rtol=2e-12)
    np.testing.assert_allclose(model.shifts[0], shift, atol=2e-14)


def test_masked_mass_and_explicit_local_fallback():
    o = observations((150.0,))
    m = fixed_model(o)
    data = o.valid.copy()
    data[3 * 49 : 4 * 49] = False
    result = fit_profiles(replace(o, valid=data), m, expected_noise=False)
    assert result.flags[0] & 8192
    assert result.values[0, 0] == pytest.approx(150, abs=1e-10)
    assert result.values[0, 2] < 0.9
    model = replace(m, counts=np.array([4, 0]), profiles=np.vstack([m.profiles[0], np.zeros(125)]))
    result = fit_profiles(o, model, expected_noise=False)
    assert result.model_ids[0] == 0
    missing = fit_profiles(replace(o, valid=np.zeros_like(data)), m, expected_noise=False)
    assert missing.flags[0] & 32768 and np.isnan(missing.values[0, 0])


@pytest.mark.parametrize(
    "kw",
    [
        dict(shape=(4, 5, 5)),
        dict(step=(1, 0, 1)),
        dict(min_references=True),
        dict(iterations=0),
        dict(cutoff=0.2),
        dict(minimum_mass=np.nan),
        dict(regions=(5, 1, 1)),
    ],
)
def test_invalid_options(kw):
    with pytest.raises(ValueError):
        ProfileOptions(**kw)


def test_learned_foreground_ignores_collection_only_noise_and_flags():
    o = observations((150.0,))
    m = fixed_model(o)
    samples = o.samples.copy()
    samples[0, 0] = 1e6
    valid = o.valid.copy()
    valid[1] = False
    ff = o.summation.frame_flags.copy()
    ff[0] = 2
    t = replace(o.summation, flags=np.array([2], np.uint32), frame_flags=ff)
    result = fit_profiles(
        replace(o, summation=t, samples=samples, valid=valid), m, expected_noise=False
    )
    assert result.values[0, 0] == pytest.approx(150, abs=1e-10)
    assert result.flags[0] == 0 and result.summation.flags[0] == 2


def test_learned_foreground_overlap_differs_from_collection_boxes():
    for spacing, overlap in [(5, False), (1, True)]:
        o = observations((150.0, 150.0), spacing=spacing)
        m = fixed_model(o)
        t = replace(
            o.summation,
            flags=np.full(2, 8, np.uint32),
            frame_flags=np.full(len(o.summation.frame_flags), 8, np.uint32),
        )
        result = fit_profiles(replace(o, summation=t), m, expected_noise=False)
        assert bool(np.all(result.flags & 8)) == overlap


def test_expected_poisson_weights_retain_weak_signed_counts():
    rng = np.random.default_rng(6129)
    o = observations((0.0,))
    o = replace(
        o, summation=replace(o.summation, noise_model="poisson_plus_read_noise_with_variance_floor")
    )
    m = fixed_model(o)
    pulls = []
    for _ in range(200):
        samples = o.samples.copy()
        raw = rng.poisson(20, len(samples)).astype(float)
        samples[:, 0] = raw - 20
        samples[:, 1] = np.maximum(raw, 1)
        fit = fit_profiles(replace(o, samples=samples), m)
        pulls.append(fit.values[0, 0] / fit.values[0, 1])
    assert abs(np.mean(pulls)) < 0.25
    assert 0.8 < np.std(pulls, ddof=1) < 1.2
    assert np.count_nonzero(np.array(pulls) < 0) > 60


def test_incomplete_requested_coverage_cannot_teach_profile():
    o = observations((5000.0, 6000.0, 7000.0))
    flags = o.summation.flags.copy()
    flags[0] = 256
    o = replace(o, summation=replace(o.summation, flags=flags))
    opts = ProfileOptions(shape=(5, 5, 5), regions=(1, 1, 1), min_references=2, reference_snr=5)
    model = learn_profiles(o, np.ones(3, bool), detector_shape=(50, 200), options=opts)
    np.testing.assert_array_equal(model.reference_ids, [1, 2])
    with pytest.raises(ValueError):
        learn_profiles(o, np.array([True, True, False]), detector_shape=(50, 200), options=opts)


@pytest.mark.parametrize("scale", [(1.0, 1.0, 0.23), (1.0, 1.0, -0.41), (0.8, -1.2, 0.73)])
def test_signed_coordinate_scaling_mass_and_partials(scale):
    p = truth()
    centre = np.array([0.173, -0.217, 0.129])
    rng = np.random.default_rng(912)
    b = rng.uniform(-6, 3, (80, 6))
    b[:, 1::2] = b[:, ::2] + rng.uniform(0.2, 3, (80, 3))
    got = evaluate_profile(p, b, centre=centre, scales=scale)
    expected = reference.evaluate(p, b, centre=centre, scales=scale)
    np.testing.assert_allclose(got, expected, atol=2e-15, rtol=5e-14)
    for axis in range(3):
        plus = centre.copy()
        minus = centre.copy()
        plus[axis] += 1e-6
        minus[axis] -= 1e-6
        fd = (
            evaluate_profile(p, b, centre=plus, scales=scale)[:, 0]
            - evaluate_profile(p, b, centre=minus, scales=scale)[:, 0]
        ) / 2e-6
        np.testing.assert_allclose(fd, got[:, axis + 1], atol=4e-10)
    full = evaluate_profile(
        p, np.array([[-50.0, 50.0, -50.0, 50.0, -50.0, 50.0]]), centre=centre, scales=scale
    )
    np.testing.assert_allclose(full, [[1, 0, 0, 0]], atol=1e-15)


def test_scaled_signed_gls_matches_independent_equations():
    o = observations((160.0, -20.0), covariance=True)
    scales = np.array([[1.0, 1.0, 0.6], [1.0, 1.0, -0.8]])
    b, _ = reference.sample_bounds(
        o.offsets,
        o.summation.boxes,
        np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1),
    )
    samples = o.samples.copy()
    for i, amp in enumerate((160.0, -20.0)):
        sel = np.repeat(o.summation.observation_ids == i, np.diff(o.offsets))
        samples[sel, 0] = (
            amp * reference.evaluate(truth(), b[sel], centre=o.centres[i], scales=scales[i])[:, 0]
        )
    o = replace(o, samples=samples, scales=scales)
    m = fixed_model(o)
    got = fit_profiles(o, m, expected_noise=False)
    expected = reference.fit(
        o.samples,
        o.valid,
        o.offsets,
        o.summation.boxes,
        o.summation.observation_ids,
        np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1),
        o.covariance,
        o.centres,
        m.volumes,
        np.ones(2, int),
        m.shifts,
        scales=scales,
    )
    np.testing.assert_allclose(got.values, expected, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(got.values[:, 0], [160, -20], atol=1e-10)
    assert replace(o, scales=np.ones_like(scales)).fingerprint() != o.fingerprint()


def test_rotation_factor_signed_geometry():
    from phasesmith.single_crystal.profiles import rotation_factors
    from phasesmith.single_crystal.reciprocal import PanelGeometry

    geometry = PanelGeometry(
        np.array([[0, 0, 10], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0.0]]), 0.31
    )
    z = rotation_factors(np.array([[1.0, 0], [-1, 0], [0, 1], [1, 1], [0, 0]]), geometry)
    np.testing.assert_allclose(z, [-1, 1, 0, -1 / np.sqrt(2), 0], atol=1e-15)
    assert not z.flags.writeable


def test_prepare_streamed_known_counts_and_widths():
    from phasesmith.single_crystal.prediction import ScanPredictions
    from phasesmith.single_crystal.profiles import prepare_profiles

    centre = np.array([[30.17, 30.28, 0.19]])
    bounds = np.column_stack([np.arange(9) - 4.5, np.arange(9) - 3.5])
    yy, xx = np.indices((61, 61))
    scales = np.array([[1.0, 1.0, -0.65]])
    p = truth()
    frames = []
    for lo, hi in bounds:
        b = np.column_stack(
            [
                xx.ravel() - 0.5,
                xx.ravel() + 0.5,
                yy.ravel() - 0.5,
                yy.ravel() + 0.5,
                np.full(xx.size, lo),
                np.full(xx.size, hi),
            ]
        )
        frames.append(
            (
                20 + 1234 * reference.evaluate(p, b, centre=centre[0], scales=scales[0])[:, 0]
            ).reshape(61, 61)
        )
    predictions = ScanPredictions(np.array([[1, 0, 0]]), centre, np.array([4]), 1.0)
    o = prepare_profiles(
        lambda i: frames[i],
        bounds,
        predictions,
        variance_loader=lambda i: np.full((61, 61), 4.0),
        options=IntegrationOptions(
            pixel_radius=5,
            omega_half_width=1.0,
            background_model="plane",
            background_gap=2,
            background_margin=4,
        ),
        scales=scales,
        angular_half_widths=np.array([4.0]),
    )
    np.testing.assert_allclose(o.summation.values[0, 0], 1234, atol=1e-9)
    np.testing.assert_array_equal(o.summation.angular_half_widths, [4.0])
    got = fit_profiles(o, fixed_model(o), expected_noise=False)
    np.testing.assert_allclose(got.values[0, 0], 1234, atol=1e-9)
    assert not got.flags[0]


def test_signed_scaled_training_matches_independent_solver():
    o = observations((5000.0, 6000.0, 7000.0, 8000.0))
    scales = np.column_stack([np.ones((4, 2)), [0.6, -0.8, 0.9, -1.2]])
    angles = np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1)
    b, _ = reference.sample_bounds(o.offsets, o.summation.boxes, angles)
    samples = o.samples.copy()
    for i, amp in enumerate((5000.0, 6000.0, 7000.0, 8000.0)):
        sel = np.repeat(o.summation.observation_ids == i, np.diff(o.offsets))
        samples[sel, 0] = (
            amp * reference.evaluate(truth(), b[sel], centre=o.centres[i], scales=scales[i])[:, 0]
        )
    o = replace(o, samples=samples, scales=scales)
    mask = np.ones(4, bool)
    opts = ProfileOptions(
        shape=(5, 5, 5),
        regions=(1, 1, 1),
        min_references=2,
        reference_snr=5,
        iterations=12,
        cutoff=0,
    )
    model = learn_profiles(o, mask, detector_shape=(50, 200), options=opts)
    p, shift = reference.learn_one(
        o.samples,
        o.valid,
        o.offsets,
        o.summation.boxes,
        o.summation.observation_ids,
        angles,
        o.centres,
        mask,
        opts.shape,
        np.array(opts.step),
        opts.iterations,
        scales=scales,
    )
    np.testing.assert_allclose(model.profiles[0], p, atol=2e-15, rtol=2e-12)
    np.testing.assert_allclose(model.shifts[0], shift, atol=2e-14)
    again = learn_profiles(o, mask, detector_shape=(50, 200), options=opts)
    assert model.profiles.tobytes() == again.profiles.tobytes()
    assert model.shifts.tobytes() == again.shifts.tobytes()


@pytest.mark.parametrize(
    "scales", [np.zeros((2, 3)), np.ones((1, 3)), np.full((2, 3), np.nan), np.ones((2, 3), bool)]
)
def test_invalid_coordinate_scales(scales):
    with pytest.raises(ValueError, match="scales"):
        replace(observations((1.0, 2.0)), scales=scales)


def test_finite_uniform_voxel_moments():
    p = truth()
    step = np.array([0.5, 0.7, 0.25])
    centre = np.array([0.173, -0.247, 0.391])
    fine = step / 4
    z, y, x = np.indices((20, 20, 20))
    indices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    lo = (indices - 10) * fine + centre
    hi = lo + fine
    bounds = np.empty((len(lo), 6))
    bounds[:, ::2] = lo
    bounds[:, 1::2] = hi
    mass = evaluate_profile(p, bounds, step=step, centre=centre)[:, 0]
    xyz = (lo + hi) / 2
    mean = mass @ xyz
    np.testing.assert_allclose(mass.sum(), 1, atol=3e-15)
    np.testing.assert_allclose(mean, centre, atol=5e-15)
    variance = mass @ ((xyz - mean) ** 2)
    z, y, x = np.indices(p.shape)
    coarse = (np.column_stack([x.ravel(), y.ravel(), z.ravel()]) - 2) * step
    expected = p.ravel() @ (coarse**2) + (step**2 - fine**2) / 12
    np.testing.assert_allclose(variance, expected, atol=5e-15)


def test_exact_finite_support_boundaries():
    p = np.full((3, 3, 3), 1 / 27)
    bounds = np.array([[1.5, 2.5, -1.0, 1.0, -1.0, 1.0], [-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]])
    got = evaluate_profile(p, bounds)
    # Empty half-open contact has zero mass; coincident-edge partials use zero.
    np.testing.assert_allclose(got, [[0, 0, 0, 0], [1, 0, 0, 0]], atol=1e-15)


@pytest.mark.parametrize(
    "planar",
    [
        np.array([[0.83, 0.35], [-0.45, 1.12]]),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[-0.8, 0.2], [0.3, 1.1]]),
    ],
)
def test_affine_pixel_intersections_and_partials(planar):
    from phasesmith.single_crystal.reference.profiles_affine_reference import evaluate

    p = truth()
    rng = np.random.default_rng(436)
    b = rng.uniform(-2.8, 1.2, (8, 6))
    b[:, 1::2] = b[:, ::2] + rng.uniform(0.2, 2, (8, 3))
    centre = np.array([0.117, -0.231, 0.193])
    step = (0.8, 1.1, 0.6)
    scale = (0.9, 1.2, -0.8)
    actual = evaluate_profile(p, b, step=step, centre=centre, scales=scale, planar_transform=planar)
    expected = evaluate(p, b, step, centre, planar, scale)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], atol=1e-13, rtol=2e-11)
    np.testing.assert_allclose(actual[:, 1:], expected[:, 1:], atol=2e-9, rtol=2e-6)
    full = evaluate_profile(
        p,
        np.array([[-20.0, 20, -20.0, 20, -20.0, 20.0]]),
        step=step,
        centre=centre,
        scales=scale,
        planar_transform=planar,
    )
    np.testing.assert_allclose(full, [[1, 0, 0, 0]], atol=5e-13)


def test_affine_identity_and_shifted_native_fit():
    o = observations((500.0, -20.0), covariance=True)
    planar = np.array([[0.85, -0.3], [0.4, 1.1]])
    A = np.tile(planar, (len(o.centres), 1, 1))
    angles = np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1)
    bounds, _ = reference.sample_bounds(o.offsets, o.summation.boxes, angles)
    samples = o.samples.copy()
    for r, amp in enumerate([500.0, -20.0]):
        sel = np.repeat(o.summation.observation_ids == r, np.diff(o.offsets))
        samples[sel, 0] = (
            amp
            * evaluate_profile(truth(), bounds[sel], centre=o.centres[r], planar_transform=planar)[
                :, 0
            ]
        )
    o = replace(o, samples=samples, planar_transforms=A)
    m = fixed_model(o)
    f = fit_profiles(o, m, expected_noise=False)
    np.testing.assert_allclose(f.values[:, 0], [500.0, -20.0], atol=1e-10)
    shifted = o.centres.copy()
    shifted[0] += [0.04, -0.07, 0.03]
    o = replace(o, centres=shifted)
    f = fit_profiles(o, m, expected_noise=False)
    for j in range(3):
        plus = shifted.copy()
        minus = shifted.copy()
        plus[0, j] += 1e-5
        minus[0, j] -= 1e-5
        fp = fit_profiles(replace(o, centres=plus), m, expected_noise=False).values[0]
        fm = fit_profiles(replace(o, centres=minus), m, expected_noise=False).values[0]
        fd = (fp[3] * (fp[4] - 1) - fm[3] * (fm[4] - 1)) / 2e-5
        assert fd == pytest.approx(f.values[0, 7 + j], rel=2e-6, abs=2e-5)


def test_variation_gls_covariance_and_signed_counts():
    from phasesmith.single_crystal.profiles import SCATTERING_VARIATIONS
    from phasesmith.single_crystal.reference.profiles_variation_reference import fit

    o = observations((1.0, 1.0), covariance=True)
    options = replace(fixed_model(o).options, cutoff=0)
    model = replace(fixed_model(o), options=options)
    factors = np.array(SCATTERING_VARIATIONS)
    angles = np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1)
    bounds, _ = reference.sample_bounds(o.offsets, o.summation.boxes, angles)
    samples = o.samples.copy()
    beta = np.array([[120.0, 35.0, 25.0, 15.0], [-10.0, -4.0, -3.0, -2.0]])
    for r in range(2):
        selected = np.repeat(o.summation.observation_ids == r, np.diff(o.offsets))
        signal = np.zeros(selected.sum())
        for component, factor in enumerate(factors):
            signal += (
                beta[r, component]
                * reference.evaluate(truth(), bounds[selected], step=factor, centre=o.centres[r])[
                    :, 0
                ]
            )
        samples[selected, 0] = signal
    o = replace(o, samples=samples)
    result = fit_profiles(
        o, model, expected_noise=False, variation_factors=factors, allow_shape_refinement=True
    )
    values, coeff, cov = fit(o, model, factors, np.ones(2, int))
    np.testing.assert_allclose(result.values, values, atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(result.coefficients, coeff, atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(result.coefficient_covariance, cov, atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(result.values[:, 0], beta.sum(axis=1), atol=2e-8)
    assert not result.coefficients.flags.writeable
    assert not result.coefficient_covariance.flags.writeable


def test_variation_objective_derivatives_and_rank_failure():
    from phasesmith.single_crystal.profiles import SCATTERING_VARIATIONS

    o = observations((150.0,), covariance=True)
    m = replace(fixed_model(o), options=replace(fixed_model(o).options, cutoff=0))
    shifted = o.centres.copy()
    shifted[0] += [0.06, -0.08, 0.03]
    o = replace(o, centres=shifted)
    f = fit_profiles(
        o,
        m,
        expected_noise=False,
        variation_factors=SCATTERING_VARIATIONS,
        allow_shape_refinement=True,
    )
    for axis in range(3):
        plus = o.centres.copy()
        minus = o.centres.copy()
        plus[0, axis] += 1e-5
        minus[0, axis] -= 1e-5
        fp = fit_profiles(
            replace(o, centres=plus),
            m,
            expected_noise=False,
            variation_factors=SCATTERING_VARIATIONS,
            allow_shape_refinement=True,
        ).values[0]
        fm = fit_profiles(
            replace(o, centres=minus),
            m,
            expected_noise=False,
            variation_factors=SCATTERING_VARIATIONS,
            allow_shape_refinement=True,
        ).values[0]
        fd = (fp[3] * (fp[4] - 4) - fm[3] * (fm[4] - 4)) / 2e-5
        assert fd == pytest.approx(f.values[0, 7 + axis], rel=3e-6, abs=2e-5)
    bad = fit_profiles(
        o, m, expected_noise=False, variation_factors=np.ones((2, 3)), allow_shape_refinement=True
    )
    assert bad.flags[0] & 32768
    assert np.isnan(bad.values[0, 0])


def test_frozen_strong_shapes_and_heldout_isolation():
    from phasesmith.single_crystal.profiles import fit_learned_profiles, learn_profile_variations

    o = observations()
    m = replace(fixed_model(o), options=replace(fixed_model(o).options, cutoff=0))
    shapes = learn_profile_variations(o, m, minimum_snr=20, neighbours=3)
    fit = fit_learned_profiles(o, m, shapes, expected_noise=False)
    np.testing.assert_allclose(
        fit.values[:, 0], [5000, 6000, 7000, 8000, 100, -20], rtol=2e-10, atol=2e-8
    )
    assert np.all(shapes.reference_ids < 4)
    samples = o.samples.copy()
    held = np.repeat(o.summation.observation_ids >= 4, np.diff(o.offsets))
    samples[held, 0] *= 10000
    changed = replace(o, samples=samples)
    other = learn_profile_variations(
        changed,
        replace(m, training_fingerprint=changed.fingerprint()),
        minimum_snr=20,
        neighbours=3,
    )
    assert shapes.fractions.tobytes() == other.fractions.tobytes()
    assert not shapes.fractions.flags.writeable
    with pytest.raises(ValueError, match="training diagnostic"):
        fit_profiles(o, m, variation_factors=shapes.factors)


def test_frozen_shape_nulls_stay_signed_and_calibrated():
    from phasesmith.single_crystal.profiles import SCATTERING_VARIATIONS

    o = observations(np.zeros(240))
    m = replace(fixed_model(o), options=replace(fixed_model(o).options, cutoff=0))
    rng = np.random.default_rng(942)
    samples = o.samples.copy()
    samples[:, 0] = rng.normal(0, 2, len(samples))
    o = replace(o, samples=samples)
    fractions = np.tile([1.0, 0, 0, 0], (len(o.centres), 1))
    f = fit_profiles(
        o,
        m,
        expected_noise=False,
        variation_factors=SCATTERING_VARIATIONS,
        fixed_fractions=fractions,
    )
    pull = f.values[:, 0] / f.values[:, 1]
    assert abs(pull.mean()) < 0.2
    assert 0.8 < pull.std(ddof=1) < 1.2
    assert 85 < np.count_nonzero(f.values[:, 0] < 0) < 155


def test_fixed_mixed_profile_matches_dense_reference():
    from phasesmith.single_crystal.profiles import SCATTERING_VARIATIONS
    from phasesmith.single_crystal.reference.profiles_variation_reference import fit

    o = observations((1.0, 1.0), covariance=True)
    m = replace(fixed_model(o), options=replace(fixed_model(o).options, cutoff=0))
    factors = np.array(SCATTERING_VARIATIONS)
    fractions = np.array([[0.2, 0.4, 0.1, 0.3], [0.1, 0.2, 0.4, 0.3]])
    bounds, _ = reference.sample_bounds(
        o.offsets,
        o.summation.boxes,
        np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1),
    )
    samples = o.samples.copy()
    for r, amplitude in enumerate([195.0, -19.0]):
        selected = np.repeat(o.summation.observation_ids == r, np.diff(o.offsets))
        signal = np.zeros(selected.sum())
        for j, factor in enumerate(factors):
            signal += (
                fractions[r, j]
                * reference.evaluate(truth(), bounds[selected], step=factor, centre=o.centres[r])[
                    :, 0
                ]
            )
        samples[selected, 0] = amplitude * signal
    o = replace(o, samples=samples)
    actual = fit_profiles(
        o, m, expected_noise=False, variation_factors=factors, fixed_fractions=fractions
    )
    expected, coeff, cov = fit(o, m, factors, np.ones(2, int), fixed_fractions=fractions)
    np.testing.assert_allclose(actual.values, expected, rtol=3e-9, atol=3e-8)
    np.testing.assert_allclose(actual.values[:, 0], [195.0, -19.0], atol=2e-10)
    np.testing.assert_allclose(actual.coefficients, coeff, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual.coefficient_covariance, cov, rtol=2e-10, atol=2e-10)


def test_inverse_variance_reference_weighting_and_measured_centres():
    o = observations((5000.0, 6000.0, 7000.0))
    mask = np.ones(3, bool)
    opts = ProfileOptions(
        shape=(5, 5, 5),
        regions=(1, 1, 1),
        min_references=2,
        reference_snr=5,
        iterations=12,
        cutoff=0,
        reference_weighting="inverse_variance",
    )
    m = learn_profiles(o, mask, detector_shape=(50, 200), options=opts)
    p, shift = reference.learn_one(
        o.samples,
        o.valid,
        o.offsets,
        o.summation.boxes,
        o.summation.observation_ids,
        np.sort(o.summation.omega_bounds[o.summation.frame_indices], axis=1),
        o.centres,
        mask,
        opts.shape,
        np.array(opts.step),
        opts.iterations,
        reference_weights=np.array([5000.0, 6000.0, 7000.0]) ** 2,
    )
    np.testing.assert_allclose(m.profiles[0], p, atol=3e-14, rtol=3e-11)
    np.testing.assert_allclose(m.shifts[0], shift, atol=3e-14)
    assert m.reference_centres.shape == (3, 3)
    np.testing.assert_array_equal(m.reference_centres[:, 2], o.centres[:, 2])
    assert not m.reference_centres.flags.writeable
