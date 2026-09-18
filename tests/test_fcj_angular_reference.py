"""Check the height-transformed kernel against the original angular equations."""

from pathlib import Path

import numpy as np
import pytest
from phasesmith import FcjGeometry, profile_fcj
from phasesmith.validation.fcj_angular_reference import angular_profile, investigate


def test_original_angular_integral_resolves_pawley_oracle_discrepancy():
    root = Path(__file__).parents[1]
    report = investigate(
        root / "oracle/fixtures/pawley_optimizer_v1",
        root / "oracle/diagnostics/pawley-profile-controls-20260917",
    )
    assert report["passed"]
    # Check retained black-box data, not a requirement for any future GSAS-II version.
    assert min(case["oracle_relative_l2"] for case in report["results"]) > 8e-4


@pytest.mark.parametrize("sample,detector", [(0.02, 0.006), (0.006, 0.02), (0.012, 0.012)])
def test_angular_integral_at_larger_heights_and_gaussian_only(sample, detector):
    x = np.array([17.92, 17.98, 18.0, 18.02, 18.08])
    expected = angular_profile(x, 18.0, 0.08, 0.0, sample, detector, digits=40)
    actual = profile_fcj(x, 18.0, 0.08, 0.0, FcjGeometry(sample, detector))
    np.testing.assert_allclose(actual.value, expected, rtol=2e-10, atol=1e-12)
    # Reflection about 90° reverses asymmetry without changing the density.
    mirrored = profile_fcj(180 - x[::-1], 162.0, 0.08, 0.0, FcjGeometry(sample, detector))
    np.testing.assert_allclose(mirrored.value[::-1], expected, rtol=2e-10, atol=1e-12)


def test_angular_audit_rejects_geometry_outside_its_declared_scope():
    with pytest.raises(ValueError, match="low-angle"):
        angular_profile([120], 120, 0.1, 0, 0.01, 0.01)
    with pytest.raises(ValueError, match="clipped"):
        angular_profile([1], 1, 0.1, 0, 0.1, 0.1)
