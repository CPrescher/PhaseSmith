"""Opt-in tests that execute the exact pinned GSAS-II profile routines.

These tests intentionally fail, rather than skip, when their external
environment is incomplete. Normal test runs exclude the ``external_oracle``
marker; the dedicated workflow selects it explicitly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from phasesmith.oracle import load_fixture
from phasesmith.oracle._pinned_probe import PINNED_REVISION, probe_symmetric_profile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "symmetric_pseudo_voigt_v1"
GAUSSIAN_FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))


def _required_directory(variable: str) -> Path:
    raw = os.environ.get(variable)
    if raw is None:
        pytest.fail(f"{variable} is required by the external-oracle job", pytrace=False)
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        pytest.fail(f"{variable} does not name a directory: {path}", pytrace=False)
    return path


def _import_pinned_gsasii() -> ModuleType:
    root = _required_directory("PHASESMITH_GSASII_ROOT")
    binary_dir = _required_directory("PHASESMITH_GSASII_BINARY_DIR")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(binary_dir))
    try:
        from GSASII import GSASIIpath

        GSASIIpath.binaryPath = str(binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
        from GSASII import GSASIIpwd
    except Exception as error:  # pragma: no cover - exercised only by the oracle runner
        pytest.fail(f"cannot import pinned GSAS-II: {error}", pytrace=False)
    if not hasattr(GSASIIpwd, "pyd"):
        pytest.fail("GSAS-II pypowder binary is unavailable", pytrace=False)
    return GSASIIpwd


@pytest.mark.external_oracle
def test_live_pinned_symmetric_profile_matches_committed_fixture() -> None:
    """Verify direct value and derivative conversions against live GSAS-II."""

    profile_module = _import_pinned_gsasii()
    fixture = load_fixture(FIXTURE_PATH)
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    for case in fixture.cases:
        if case["case_kind"] != "isolated_peak":
            continue
        parameters = case["parameters"]
        x = fixture.arrays[case["arrays"]["x"]]
        live_profile = probe_symmetric_profile(
            profile_module,
            x,
            position=parameters["position_deg"],
            gaussian_sigma=parameters["gaussian_sigma_deg"],
            lorentzian_fwhm=parameters["lorentzian_fwhm_deg"],
        )
        np.testing.assert_allclose(
            live_profile,
            fixture.arrays[case["arrays"]["profile"]],
            rtol=3.0e-14,
            atol=3.0e-14,
        )

        sigma2 = (100.0 * parameters["gaussian_sigma_deg"]) ** 2
        gamma = 100.0 * parameters["lorentzian_fwhm_deg"]
        _value, d_position_native, d_sigma2, d_gamma = profile_module.getdPsVoigt(
            parameters["position_deg"], sigma2, gamma, x
        )
        gaussian_fwhm = parameters["gaussian_fwhm_deg"]
        d_sigma2_d_gaussian_fwhm = 20_000.0 * gaussian_fwhm / GAUSSIAN_FWHM_PER_SIGMA**2
        converted = {
            "d_position": -100.0 * np.asarray(d_position_native, dtype=np.float64),
            "d_gaussian_fwhm": (
                100.0 * np.asarray(d_sigma2, dtype=np.float64) * d_sigma2_d_gaussian_fwhm
            ),
            "d_lorentzian_fwhm": 10_000.0 * np.asarray(d_gamma, dtype=np.float64),
        }
        for name, values in converted.items():
            np.testing.assert_allclose(
                values,
                fixture.arrays[case["arrays"][name]],
                rtol=8.0e-13,
                atol=2.0e-11,
            )
