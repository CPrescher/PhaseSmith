from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import phasesmith
from phasesmith.oracle import load_fixture
from phasesmith.oracle._pinned_probe import PINNED_REVISION
from phasesmith.refinement import lebail

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "lebail_v1"


def _models() -> tuple[
    object,
    phasesmith.ConstantWavelengthInstrument,
    phasesmith.PowderPattern,
    phasesmith.Phase,
]:
    fixture = load_fixture(FIXTURE_PATH)
    reflection_list = fixture.arrays["reflection_list"]
    values = fixture.manifest["input_parameters"]["instrument"]
    instrument = phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=values["wavelength_angstrom"],
        u_deg2=values["u_gsas_centideg2"] * 1.0e-4,
        v_deg2=values["v_gsas_centideg2"] * 1.0e-4,
        w_deg2=values["w_gsas_centideg2"] * 1.0e-4,
        x_deg=values["x_gsas_centideg"] * 1.0e-2,
        y_deg=values["y_gsas_centideg"] * 1.0e-2,
    )
    pattern = phasesmith.PowderPattern(
        fixture.arrays["x_deg"],
        observed_y=fixture.arrays["observed_y"],
        background=fixture.arrays["background"],
        uncertainty=1.0 / np.sqrt(fixture.arrays["weight"]),
    )
    reflections = phasesmith.ReflectionBatch(
        [f"oracle-{index}" for index in range(reflection_list.shape[0])],
        reflection_list[:, :3].astype(np.int64),
        reflection_list[:, 4],
        reflection_list[:, 5],
        np.ones(reflection_list.shape[0]),
    )
    return fixture, instrument, pattern, phasesmith.Phase("alpha", "Oracle alpha", reflections)


def test_lebail_fixture_records_pin_generator_and_public_outputs() -> None:
    fixture, _, _, _ = _models()
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_lebail.py"
    assert fixture.manifest["fixture_id"] == "gsasii_lebail_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert fixture.manifest["provenance"]["generator_sha256"] == hashlib.sha256(
        generator.read_bytes()
    ).hexdigest()
    assert fixture.arrays["x_deg"].shape == fixture.arrays["ycalc"].shape == (3_001,)
    assert fixture.arrays["reflection_list"].shape == (85, 15)
    assert fixture.arrays["convergence"].shape == (8, 5)
    assert fixture.arrays["convergence"][0, 2] == 100.0
    assert fixture.arrays["convergence"][-1, 2] < 1.0e-5


def test_lebail_group_intensities_profile_parameters_and_ycalc_against_oracle() -> None:
    fixture, instrument, pattern, phase = _models()
    reflection_list = fixture.arrays["reflection_list"]
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument, (phase,)),
        lebail.LeBailOptions(
            max_iterations=20,
            intensity_tolerance=1.0e-12,
            rwp_tolerance=1.0e-14,
            support_fwhm=10_000.0,
        ),
    )
    extracted = np.asarray(
        [record.integrated_intensity for record in result.intensities]
    )
    oracle_integrated = 0.01 * reflection_list[:, 8] * reflection_list[:, 11]
    for position in np.unique(reflection_list[:, 5]):
        group = reflection_list[:, 5] == position
        np.testing.assert_allclose(
            np.sum(extracted[group]),
            np.sum(oracle_integrated[group]),
            rtol=1.1e-3,
        )

    widths = phasesmith.cw_profile_parameters(reflection_list[:, 5], instrument)
    np.testing.assert_allclose(
        1.0e4 * widths.gaussian_variance_deg2,
        reflection_list[:, 6],
        rtol=7.0e-16,
    )
    np.testing.assert_allclose(
        100.0 * widths.lorentzian_fwhm_deg,
        reflection_list[:, 7],
        # The public reflection table rounds the stored gamma at ~1e-12.
        rtol=4.3e-12,
    )
    normalized_maximum_error = np.max(
        np.abs(result.calculation.y - fixture.arrays["ycalc"])
    ) / np.ptp(fixture.arrays["ycalc"])
    # GSAS-II's public histogram calculation and the independent TCH kernel
    # differ slightly in sampled peak values; group areas agree more tightly.
    assert normalized_maximum_error < 6.6e-3
    assert result.metrics.rwp < 1.4e-3
    assert result.history[-1].rwp <= result.history[0].rwp
