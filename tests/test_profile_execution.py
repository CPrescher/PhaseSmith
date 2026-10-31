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
