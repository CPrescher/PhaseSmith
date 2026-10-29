"""Known-scale recovery, independent equations, signed application and holdouts."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import scan_scaling_reference as reference
from phasesmith.experimental.scan_scaling import ScanScaleOptions, fit_scan_scale

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_scan_scale_fit"), reason="experimental scaling build"
)


def case(seed=847, families=180, coefficient=-0.25, noise=False):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(families, dtype=np.int64), 4)
    omega = rng.uniform(-35, 36, len(groups))
    x = (omega - 0.5) / 35.5
    truth = np.exp(rng.uniform(9, 11, families))
    counts = truth[groups] * np.exp(coefficient * x)
    sigma = np.sqrt(counts)
    if noise:
        counts += rng.normal(0, sigma)
    return counts, sigma, omega, groups, truth


def test_known_scale_recovery_and_signed_weak_application():
    counts, sigma, om, groups, _ = case()
    model = fit_scan_scale(counts, sigma, om, groups, omega_range=(-35, 36))
    assert model.converged and model.coefficient == pytest.approx(-0.25, abs=2e-14)
    out = model.apply([-20.0, 0.0, 50.0], [3.0, 2.0, 4.0], [-35.0, 0.5, 36.0])
    assert out[0, 1] < 0 and out[1, 1] == 0 and out[2, 1] > 0
    assert out[1, 0] == 1
    np.testing.assert_allclose(
        out,
        reference.apply(
            [-20.0, 0.0, 50.0],
            [3.0, 2.0, 4.0],
            [-35.0, 0.5, 36.0],
            (-35, 36),
            model.coefficient,
            model.variance,
        ),
        rtol=1e-14,
        atol=1e-14,
    )
    assert not out.flags.writeable and not model.reference_ids.flags.writeable


def test_randomized_native_independent_reference_and_determinism():
    for seed in range(10):
        counts, sig, om, group, _ = case(seed, noise=True)
        rng = np.random.default_rng(seed + 32)
        counts[rng.choice(len(counts), 8, replace=False)] *= 3
        training = group % 5 != 0
        model = fit_scan_scale(counts, sig, om, group, omega_range=(-35, 36), training=training)
        ref = reference.fit(counts, sig, (om - 0.5) / 35.5, group, training)
        assert model.coefficient == pytest.approx(ref[0], rel=2e-10, abs=2e-11)
        assert model.variance == pytest.approx(ref[1], rel=2e-10, abs=1e-12)
        np.testing.assert_array_equal(model.reference_ids, ref[2])
        np.testing.assert_allclose(model.reference_weights, ref[3], rtol=2e-9, atol=1e-8)
        assert abs(model.coefficient + 0.25) < 0.008
        repeat = fit_scan_scale(counts, sig, om, group, omega_range=(-35, 36), training=training)
        assert repeat.coefficient == model.coefficient and repeat.variance == model.variance
        assert repeat.reference_weights.tobytes() == model.reference_weights.tobytes()


def test_heldout_counts_do_not_train_model():
    counts, sig, om, group, _ = case(noise=True)
    training = group % 5 != 0
    a = fit_scan_scale(counts, sig, om, group, omega_range=(-35, 36), training=training)
    counts[~training] = np.nan
    sig[~training] = np.nan
    b = fit_scan_scale(counts, sig, om, group, omega_range=(-35, 36), training=training)
    assert a.coefficient == b.coefficient and a.variance == b.variance
    assert training[b.reference_ids].all()


def test_sparse_labels_zero_signal_and_uninformative_families():
    counts, sig, om, group, _ = case()
    counts[:4] = 0
    labels = group.astype(np.uint64) * 10_000_000_000 + 2**63
    model = fit_scan_scale(counts, sig, om, labels, omega_range=(-35, 36))
    assert not np.any(model.reference_ids < 4)
    assert model.coefficient == pytest.approx(-0.25, abs=2e-14)
    with pytest.raises(ValueError, match=r"insufficient|identifiable"):
        fit_scan_scale(counts, sig, np.zeros(len(om)), group, omega_range=(-35, 36))
    with pytest.raises(ValueError, match="insufficient"):
        fit_scan_scale(counts, sig, om, np.arange(len(om)), omega_range=(-35, 36))


def test_analytical_partials_and_separate_correlated_model_error():
    counts, sig, om, group, _ = case()
    model = fit_scan_scale(counts, sig, om, group, omega_range=(-35, 36))
    chosen = np.arange(20)
    out = model.apply(counts[chosen], sig[chosen], om[chosen])
    epsilon = 1e-6
    d_beta = (
        replace(model, coefficient=model.coefficient + epsilon).apply(
            counts[chosen], sig[chosen], om[chosen]
        )[:, 1]
        - replace(model, coefficient=model.coefficient - epsilon).apply(
            counts[chosen], sig[chosen], om[chosen]
        )[:, 1]
    ) / (2 * epsilon)
    d_omega = (
        model.apply(counts[chosen], sig[chosen], om[chosen] + epsilon)[:, 1]
        - model.apply(counts[chosen], sig[chosen], om[chosen] - epsilon)[:, 1]
    ) / (2 * epsilon)
    np.testing.assert_allclose(out[:, 3], d_beta, rtol=2e-7, atol=1e-5)
    np.testing.assert_allclose(out[:, 4], d_omega, rtol=3e-7, atol=1e-5)
    np.testing.assert_allclose(out[:, 5], abs(out[:, 3]) * np.sqrt(model.variance), rtol=1e-14)
    np.testing.assert_allclose(out[:, 2], out[:, 0] * sig[chosen], rtol=1e-14)


@pytest.mark.parametrize(
    "bad",
    [
        dict(minimum_snr=2),
        dict(minimum_snr=True),
        dict(relative_error_floor=-1),
        dict(relative_error_floor=np.inf),
        dict(huber=0),
        dict(max_iterations=0),
        dict(max_iterations=2.5),
        dict(max_iterations=True),
        dict(tolerance=0),
    ],
)
def test_invalid_options(bad):
    with pytest.raises(ValueError):
        ScanScaleOptions(**bad)


def test_invalid_inputs_and_explicit_angular_support():
    counts, sig, om, group, _ = case()
    with pytest.raises(ValueError):
        fit_scan_scale(counts, sig, om, group, omega_range=(36, -35))
    with pytest.raises(ValueError):
        fit_scan_scale(counts, sig, om, group + 0.5, omega_range=(-35, 36))
    with pytest.raises(ValueError):
        fit_scan_scale(
            counts, sig, om, group, omega_range=(-35, 36), training=np.ones(len(counts), int)
        )
    with pytest.raises(ValueError):
        fit_scan_scale(
            counts, sig, om, group, omega_range=(-35, 36), eligible=np.zeros(len(counts) - 1, bool)
        )
    missing = counts.copy()
    missing[0] = np.nan
    with pytest.raises(ValueError):
        fit_scan_scale(missing, sig, om, group, omega_range=(-35, 36))
    model = fit_scan_scale(counts, sig, om, group, omega_range=(-35, 36))
    model.apply([1, 1], [1, 1], [-35, 36])
    for bad_omega in [-35.0001, 36.0001, np.nan]:
        with pytest.raises(ValueError):
            model.apply([1.0], [1.0], [bad_omega])
    for bad_sigma in [0, -1, np.nan]:
        with pytest.raises(ValueError):
            model.apply([1.0], [bad_sigma], [0.0])


def test_nonconvergence_is_reported():
    counts, sig, om, group, _ = case(noise=True)
    counts[3] *= 5
    model = fit_scan_scale(
        counts, sig, om, group, omega_range=(-35, 36), options=ScanScaleOptions(max_iterations=1)
    )
    assert not model.converged and model.iterations == 1


def test_model_validation_and_unrepresentable_application():
    counts, sig, om, group, _ = case()
    model = fit_scan_scale(counts, sig, om, group, omega_range=(-35, 36))
    for bad in (dict(iterations=0), dict(converged=1), dict(variance=-1)):
        with pytest.raises(ValueError):
            replace(model, **bad)
    for coefficient in (-1000.0, 1000.0):
        with pytest.raises(ValueError, match="overflow"):
            replace(model, coefficient=coefficient).apply([1.0], [1.0], [36.0])
