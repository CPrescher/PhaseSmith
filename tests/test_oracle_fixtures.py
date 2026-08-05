from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rietveld
from rietveld.oracle import FixtureValidationError, load_fixture
from rietveld.oracle._pinned_probe import PINNED_REVISION

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "symmetric_pseudo_voigt_v1"
HISTOGRAM_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "minimal_cw_histogram_v1"
CW_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "cw_instrument_profile_v1"
FCJ_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "fcj_profile_v1"
COMPONENT_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "wavelength_components_v1"


def test_fixture_schema_and_pin_metadata_are_valid_json() -> None:
    schema = json.loads((REPOSITORY_ROOT / "oracle" / "fixtures" / "schema.json").read_text())
    pin = json.loads((REPOSITORY_ROOT / "oracle" / "PINNED_GSASII.json").read_text())
    assert schema["properties"]["format_version"]["const"] == 1
    assert pin["revision"] == PINNED_REVISION


def test_committed_fixture_passes_hash_shape_and_provenance_validation() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    assert fixture.manifest["fixture_id"] == "gsasii_symmetric_pseudo_voigt_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert len(fixture.cases) == 7
    assert all(not array.flags.writeable for array in fixture.arrays.values())


def test_fixture_records_the_committed_generator() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_symmetric_profile.py"
    digest = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert fixture.manifest["provenance"]["generator_sha256"] == digest


def test_cw_fixture_records_pin_and_committed_generator() -> None:
    fixture = load_fixture(CW_FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_cw_instrument_profile.py"
    digest = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert fixture.manifest["fixture_id"] == "gsasii_cw_instrument_profile_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert fixture.manifest["provenance"]["generator_sha256"] == digest
    assert len(fixture.cases) == 4


def test_fcj_fixture_records_pin_and_committed_generator() -> None:
    fixture = load_fixture(FCJ_FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_fcj_profile.py"
    digest = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert fixture.manifest["fixture_id"] == "gsasii_fcj_profile_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert fixture.manifest["provenance"]["generator_sha256"] == digest
    assert len(fixture.cases) == 4


def test_component_fixture_records_pin_and_committed_generator() -> None:
    fixture = load_fixture(COMPONENT_FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_wavelength_components.py"
    digest = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert fixture.manifest["fixture_id"] == "gsasii_wavelength_components_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert fixture.manifest["provenance"]["generator_sha256"] == digest
    helper = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_fcj_profile.py"
    assert (
        fixture.manifest["source"]["helper_sha256"]
        == hashlib.sha256(helper.read_bytes()).hexdigest()
    )
    assert len(fixture.cases) == 3


def test_public_scripting_histogram_fixture() -> None:
    fixture = load_fixture(HISTOGRAM_FIXTURE_PATH)
    case = fixture.cases[0]
    x = fixture.arrays[case["arrays"]["x"]]
    ycalc = fixture.arrays[case["arrays"]["ycalc"]]
    background = fixture.arrays[case["arrays"]["background"]]
    assert fixture.manifest["source"]["private_probe"] is False
    assert x.shape == ycalc.shape == background.shape == (3_501,)
    assert np.all(np.diff(x) > 0.0)
    assert np.max(ycalc) > 0.0
    np.testing.assert_array_equal(background, 1.0)

    table_metadata = case["reflection_tables"][0]
    reflections = fixture.arrays[table_metadata["array"]]
    columns = tuple(table_metadata["columns"])
    assert reflections.shape == (125, 15)
    assert columns[:8] == (
        "h",
        "k",
        "l",
        "multiplicity",
        "d_spacing_angstrom",
        "position_deg",
        "sigma2_centideg2",
        "gamma_centideg",
    )
    assert np.all(reflections[:, columns.index("d_spacing_angstrom")] > 0.0)
    assert np.all(reflections[:, columns.index("sigma2_centideg2")] > 0.0)
    assert np.all(reflections[:, columns.index("gamma_centideg")] > 0.0)
    positions = reflections[:, columns.index("position_deg")]
    assert np.all((positions >= x[0]) & (positions <= x[-1]))

    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_powder_histogram.py"
    digest = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert fixture.manifest["provenance"]["generator_sha256"] == digest


@pytest.mark.parametrize("shape_class", ["gaussian_dominant", "mixed", "lorentzian_dominant"])
@pytest.mark.parametrize("width_scale", ["narrow", "broad"])
def test_tch_profile_and_derivatives_against_pinned_gsasii_fixture(
    shape_class: str, width_scale: str
) -> None:
    fixture = load_fixture(FIXTURE_PATH)
    case = next(
        case
        for case in fixture.cases
        if case["case_kind"] == "isolated_peak"
        and case["parameters"]["shape_class"] == shape_class
        and case["parameters"]["width_scale"] == width_scale
    )
    parameters = case["parameters"]
    x = fixture.arrays[case["arrays"]["x"]]
    oracle = fixture.arrays[case["arrays"]["profile"]]
    actual = rietveld.profile_tch(
        x - parameters["position_deg"],
        parameters["gaussian_fwhm_deg"],
        parameters["lorentzian_fwhm_deg"],
    )

    # GSAS-II's compiled TCH approximation differs most in the far Gaussian-
    # dominant tail. This tolerance is local to the pinned #5838 fixture.
    np.testing.assert_allclose(actual.value, oracle, rtol=1.6e-4, atol=3e-8)
    derivative_pairs = {
        "d_position": -actual.d_delta,
        "d_gaussian_fwhm": actual.d_gaussian_fwhm,
        "d_lorentzian_fwhm": actual.d_lorentzian_fwhm,
    }
    for derivative_name, native_derivative in derivative_pairs.items():
        oracle_derivative = fixture.arrays[case["arrays"][derivative_name]]
        peak_scale = float(np.max(np.abs(oracle_derivative)))
        normalized_maximum_error = float(
            np.max(np.abs(native_derivative - oracle_derivative)) / peak_scale
        )
        assert normalized_maximum_error < 6e-5
    sampled_integral = np.trapezoid(oracle, x)
    assert sampled_integral == pytest.approx(case["sampled_integral_per_degree"], rel=2e-15)
    centroid = np.trapezoid(x * oracle, x) / sampled_integral
    assert centroid == pytest.approx(parameters["position_deg"], abs=2e-5)


def test_fused_overlap_against_pinned_gsasii_fixture() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    case = next(case for case in fixture.cases if case["case_kind"] == "overlapping_peaks")
    peaks = case["parameters"]["peaks"]
    x = fixture.arrays[case["arrays"]["x"]]
    actual = rietveld.accumulate_tch(
        x,
        [peak["position_deg"] for peak in peaks],
        [peak["intensity"] for peak in peaks],
        [np.sqrt(8.0 * np.log(2.0)) * peak["gaussian_sigma_deg"] for peak in peaks],
        [peak["lorentzian_fwhm_deg"] for peak in peaks],
        support_fwhm=100.0,
    ).y
    oracle = fixture.arrays[case["arrays"]["ycalc"]]
    np.testing.assert_allclose(actual, oracle, rtol=1.2e-4, atol=2e-5)


@pytest.mark.parametrize("angular_regime", ["low", "middle", "high"])
def test_cw_profile_widths_and_derivatives_against_pinned_gsasii(
    angular_regime: str,
) -> None:
    fixture = load_fixture(CW_FIXTURE_PATH)
    case = next(
        case
        for case in fixture.cases
        if case["case_kind"] == "cw_isolated_reflection"
        and case["parameters"]["angular_regime"] == angular_regime
    )
    parameters = case["parameters"]
    model = rietveld.ConstantWavelengthInstrument(**dict(parameters["instrument"]))
    position = parameters["position_deg"]
    x = fixture.arrays[case["arrays"]["x"]]
    widths = rietveld.cw_profile_parameters([position], model)
    assert widths.gaussian_variance_deg2[0] == pytest.approx(
        parameters["gaussian_variance_deg2"], rel=3e-15
    )
    assert widths.gaussian_fwhm_deg[0] == pytest.approx(parameters["gaussian_fwhm_deg"], rel=3e-15)
    assert widths.lorentzian_fwhm_deg[0] == pytest.approx(
        parameters["lorentzian_fwhm_deg"], rel=3e-15
    )

    actual = rietveld.accumulate_cw(
        x, [position], [1.0], model, support_fwhm=100.0, jacobian_layout="dense"
    )
    comparisons = (
        (actual.y, fixture.arrays[case["arrays"]["profile"]]),
        (actual.jacobian[0, 1], fixture.arrays[case["arrays"]["d_position"]]),
        (
            actual.derivatives.global_jacobian,
            fixture.arrays[case["arrays"]["d_instrument"]],
        ),
    )
    for native, oracle in comparisons:
        normalized_maximum_error = float(np.max(np.abs(native - oracle)) / np.max(np.abs(oracle)))
        # Local to the pinned #5838 TCH implementation and public unit chain.
        assert normalized_maximum_error < 6e-6
    sampled_integral = np.trapezoid(actual.y, x)
    assert sampled_integral == pytest.approx(parameters["integrated_intensity"], rel=2.2e-3)
    centroid = np.trapezoid(x * actual.y, x) / sampled_integral
    assert centroid == pytest.approx(position, abs=2e-11)


def test_fused_cw_overlap_and_all_jacobians_against_pinned_gsasii() -> None:
    fixture = load_fixture(CW_FIXTURE_PATH)
    case = next(case for case in fixture.cases if case["case_kind"] == "cw_overlapping_reflections")
    parameters = case["parameters"]
    model = rietveld.ConstantWavelengthInstrument(**dict(parameters["instrument"]))
    x = fixture.arrays[case["arrays"]["x"]]
    actual = rietveld.accumulate_cw(
        x,
        parameters["positions_deg"],
        parameters["integrated_intensities"],
        model,
        support_fwhm=100.0,
        jacobian_layout="dense",
    )
    comparisons = (
        (actual.y, fixture.arrays[case["arrays"]["ycalc"]]),
        (actual.jacobian, fixture.arrays[case["arrays"]["local_jacobian"]]),
        (
            actual.derivatives.global_jacobian,
            fixture.arrays[case["arrays"]["global_jacobian"]],
        ),
    )
    for native, oracle in comparisons:
        normalized_maximum_error = float(np.max(np.abs(native - oracle)) / np.max(np.abs(oracle)))
        assert normalized_maximum_error < 6e-6


@pytest.mark.parametrize("angular_regime", ["low", "middle", "high"])
def test_fcj_values_and_moments_against_pinned_gsasii(angular_regime: str) -> None:
    fixture = load_fixture(FCJ_FIXTURE_PATH)
    case = next(case for case in fixture.cases if case["id"] == f"{angular_regime}_fcj")
    parameters = case["parameters"]
    x = fixture.arrays[case["arrays"]["x"]]
    oracle = fixture.arrays[case["arrays"]["profile"]]
    mapping = parameters["public_equal_height_mapping"]
    actual = rietveld.profile_fcj(
        x,
        parameters["position_deg"],
        parameters["gaussian_fwhm_deg"],
        parameters["lorentzian_fwhm_deg"],
        rietveld.FcjGeometry(**dict(mapping)),
    ).value
    normalized_maximum_error = float(np.max(np.abs(actual - oracle)) / np.max(np.abs(oracle)))
    # GSAS-II #5838 uses a discretized one-parameter SH/L convolution. The
    # independently quadrature-converged published integral differs most at
    # middle/high angle; this tolerance is local to that pinned behavior.
    assert normalized_maximum_error < 2.4e-2

    actual_area = np.trapezoid(actual, x)
    actual_centroid = np.trapezoid(x * actual, x) / actual_area
    actual_third = np.trapezoid((x - actual_centroid) ** 3 * actual, x) / actual_area
    oracle_moments = case["sampled_moments"]
    assert actual_area == pytest.approx(oracle_moments["integral"], rel=4e-6)
    assert actual_centroid == pytest.approx(oracle_moments["centroid_deg"], abs=1.6e-3)
    assert np.sign(actual_third) == np.sign(oracle_moments["third_central_moment_deg3"])
    assert actual_third == pytest.approx(oracle_moments["third_central_moment_deg3"], rel=0.33)


def test_fcj_zero_limit_against_pinned_gsasii() -> None:
    fixture = load_fixture(FCJ_FIXTURE_PATH)
    case = next(case for case in fixture.cases if case["id"] == "middle_zero")
    parameters = case["parameters"]
    x = fixture.arrays[case["arrays"]["x"]]
    oracle = fixture.arrays[case["arrays"]["profile"]]
    actual = rietveld.profile_fcj(
        x,
        parameters["position_deg"],
        parameters["gaussian_fwhm_deg"],
        parameters["lorentzian_fwhm_deg"],
        rietveld.FcjGeometry(0.0, 0.0),
    ).value
    normalized_maximum_error = float(np.max(np.abs(actual - oracle)) / np.max(np.abs(oracle)))
    assert normalized_maximum_error < 3e-6


@pytest.mark.parametrize("angular_regime", ["low", "middle", "high"])
def test_fcj_doublet_values_positions_and_moments_against_pinned_gsasii(
    angular_regime: str,
) -> None:
    fixture = load_fixture(COMPONENT_FIXTURE_PATH)
    case = next(
        case for case in fixture.cases if case["parameters"]["angular_regime"] == angular_regime
    )
    parameters = case["parameters"]
    instrument = rietveld.ConstantWavelengthInstrument(**dict(parameters["instrument"]))
    components = rietveld.WavelengthComponents(
        parameters["wavelengths_angstrom"], parameters["relative_intensities"]
    )
    geometry = rietveld.FcjGeometry(**dict(parameters["public_equal_height_mapping"]))
    x = fixture.arrays[case["arrays"]["x"]]
    oracle = fixture.arrays[case["arrays"]["ycalc"]]
    actual = rietveld.accumulate_cw_fcj_components(
        x,
        [parameters["base_position_deg"]],
        [1.0],
        instrument,
        components,
        geometry,
        support_fwhm=100.0,
    ).y
    normalized_maximum_error = float(np.max(np.abs(actual - oracle)) / np.max(np.abs(oracle)))
    assert normalized_maximum_error < 2.4e-2

    base_theta = np.deg2rad(parameters["base_position_deg"] / 2.0)
    component_positions = np.rad2deg(
        2.0
        * np.arcsin(
            components.wavelengths_angstrom / instrument.wavelength_angstrom * np.sin(base_theta)
        )
    )
    np.testing.assert_allclose(
        component_positions,
        fixture.arrays[case["arrays"]["component_positions"]],
        rtol=2e-15,
        atol=2e-14,
    )
    actual_area = np.trapezoid(actual, x)
    actual_centroid = np.trapezoid(x * actual, x) / actual_area
    actual_third = np.trapezoid((x - actual_centroid) ** 3 * actual, x) / actual_area
    oracle_moments = case["sampled_moments"]
    assert actual_area == pytest.approx(oracle_moments["integral"], rel=4e-6)
    assert actual_centroid == pytest.approx(oracle_moments["centroid_deg"], abs=1.6e-3)
    assert actual_third == pytest.approx(oracle_moments["third_central_moment_deg3"], rel=0.1)


def test_fixture_reader_rejects_revision_drift(tmp_path: Path) -> None:
    manifest = json.loads((FIXTURE_PATH / "manifest.json").read_text())
    manifest["provenance"]["gsasii_revision"] = "0" * 40
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FixtureValidationError, match="does not match"):
        load_fixture(manifest_path)


def test_fixture_reader_rejects_archive_corruption(tmp_path: Path) -> None:
    manifest = json.loads((FIXTURE_PATH / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = bytearray((FIXTURE_PATH / "data.npz").read_bytes())
    archive[len(archive) // 2] ^= 0x01
    (tmp_path / "data.npz").write_bytes(archive)
    with pytest.raises(FixtureValidationError, match="archive SHA-256 mismatch"):
        load_fixture(tmp_path)
