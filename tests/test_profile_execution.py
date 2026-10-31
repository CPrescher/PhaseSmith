"""Exact cached/streaming profile equivalence and explicit resource admission."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.profiles import ProfileExecution, ProfileOptions, learn_profiles
from test_experimental_profiles import observations

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_profile_learn"), reason="experimental profiles"
)


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_learning_cache_preserves_order_and_bounded_fallback(affine, weighted):
    obs = observations((5000.0, 6000.0, 7000.0, 8000.0), covariance=True)
    if affine:
        obs = replace(
            obs,
            scales=np.tile([0.8, 1.1, -0.9], (4, 1)),
            planar_transforms=np.tile([[0.9, -0.2], [0.1, 1.05]], (4, 1, 1)),
        )
    options = ProfileOptions(
        shape=(5, 5, 5),
        regions=(1, 1, 1),
        min_references=2,
        iterations=5,
        cutoff=0.0,
        reference_weighting="inverse_variance" if weighted else "equal",
    )
    refs = np.ones(4, bool)
    baseline = learn_profiles(
        obs, refs, detector_shape=(50, 100), options=options, execution=ProfileExecution(1, 0)
    )
    for budget in [1, 1024**2]:
        cached = learn_profiles(
            obs,
            refs,
            detector_shape=(50, 100),
            options=options,
            execution=ProfileExecution(1, budget),
        )
        for key in ["profiles", "counts", "shifts", "reference_ids", "reference_centres"]:
            np.testing.assert_array_equal(getattr(cached, key), getattr(baseline, key))
        assert cached.execution["cache_bytes"] <= budget
        assert (cached.execution["cache_entries"] > 0) == (budget > 1)
    for workers, cache in [(0, 1), (5, 1), (1, -1), (1, 2**30 + 1)]:
        with pytest.raises(ValueError):
            ProfileExecution(workers, cache)


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_learning_banks_preserve_every_scalar_sum(workers):
    obs = observations((5000.0, 6000.0, 7000.0, 8000.0), covariance=True)
    obs = replace(
        obs,
        scales=np.tile([0.8, 1.1, -0.9], (4, 1)),
        planar_transforms=np.tile([[0.9, -0.2], [0.1, 1.05]], (4, 1, 1)),
    )
    options = ProfileOptions(
        shape=(5, 5, 5),
        regions=(2, 1, 1),
        min_references=2,
        iterations=7,
        cutoff=0.0,
        reference_weighting="inverse_variance",
    )

    def run(n, budget):
        return learn_profiles(
            obs,
            np.ones(4, bool),
            detector_shape=(50, 100),
            options=options,
            execution=ProfileExecution(n, budget),
        )

    baseline, parallel = run(1, 0), run(workers, 1024**2)
    for name in ["profiles", "counts", "shifts", "reference_ids", "reference_centres"]:
        np.testing.assert_array_equal(getattr(baseline, name), getattr(parallel, name))
    assert parallel.execution["workers"] == min(workers, 3)
    assert parallel.execution["cache_bytes"] <= 1024**2
    assert run(workers, 1).execution["workers"] == 1


@pytest.mark.parametrize("expected", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_baseline_integration_cache_and_workers_are_bitwise_exact(expected, affine):
    from phasesmith.experimental.profiles import fit_profiles
    from test_experimental_profiles import fixed_model

    obs = observations((5000.0, -20.0, 0.0, 8000.0), covariance=True)
    valid = obs.valid.copy()
    valid[23::47] = False
    samples = obs.samples.copy()
    samples[~valid] = np.nan
    table = replace(
        obs.summation, noise_model="expected_counts" if expected else obs.summation.noise_model
    )
    obs = replace(obs, summation=table, samples=samples, valid=valid)
    if affine:
        obs = replace(
            obs,
            scales=np.tile([0.8, 1.1, -0.9], (4, 1)),
            planar_transforms=np.tile([[0.9, -0.2], [0.1, 1.05]], (4, 1, 1)),
        )
    bank = fixed_model(obs)
    baseline = fit_profiles(obs, bank, execution=ProfileExecution(1, 0))
    for workers in [1, 2, 4]:
        for budget in [1, 1024**2]:
            value = fit_profiles(obs, bank, execution=ProfileExecution(workers, budget))
            assert value.values.tobytes() == baseline.values.tobytes()
            np.testing.assert_array_equal(value.flags, baseline.flags)
            assert value.execution["cache_peak_bound_bytes"] <= budget
            if budget > 1:
                assert value.execution["cached_reflections"] > 0
