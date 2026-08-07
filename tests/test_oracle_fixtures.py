from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import phasesmith
import pytest
from phasesmith.oracle import FixtureValidationError, load_fixture
from phasesmith.oracle._pinned_probe import PINNED_REVISION

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "symmetric_pseudo_voigt_v1"
HISTOGRAM_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "minimal_cw_histogram_v1"
CW_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "cw_instrument_profile_v1"
FCJ_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "fcj_profile_v1"
COMPONENT_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "wavelength_components_v1"
SAMPLE_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "sample_physics_v1"
MULTIPHASE_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "multiphase_v1"
NEUTRON_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "neutron_cw_v1"
TOF_FIXTURE_PATH = REPOSITORY_ROOT / "oracle" / "fixtures" / "tof_v1"


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


def test_sample_fixture_records_pin_and_committed_generator() -> None:
    fixture = load_fixture(SAMPLE_FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_sample_physics.py"
    digest = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert fixture.manifest["fixture_id"] == "gsasii_sample_physics_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert fixture.manifest["provenance"]["generator_sha256"] == digest
    assert fixture.manifest["source"]["private_probe"] is True
    assert len(fixture.cases) == 3


def test_multiphase_fixture_records_pin_and_public_scripting_arrays() -> None:
    fixture = load_fixture(MULTIPHASE_FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_multiphase.py"
    assert fixture.manifest["fixture_id"] == "gsasii_multiphase_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert (
        fixture.manifest["provenance"]["generator_sha256"]
        == hashlib.sha256(generator.read_bytes()).hexdigest()
    )
    assert fixture.arrays["x_deg"].shape == fixture.arrays["ycalc"].shape == (4_501,)
    assert fixture.arrays["background"].shape == (4_501,)
    assert fixture.arrays["reflection_list_alpha"].shape == (125, 15)
    assert fixture.arrays["reflection_list_beta"].shape == (73, 15)
    assert np.max(fixture.arrays["ycalc"]) > np.max(fixture.arrays["background"])


def test_multiphase_values_components_and_scale_rows_against_pinned_gsasii() -> None:
    fixture = load_fixture(MULTIPHASE_FIXTURE_PATH)
    case = fixture.cases[0]
    values = fixture.manifest["input_parameters"]["instrument"]
    instrument = phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=values["wavelength_angstrom"],
        u_deg2=values["u_gsas_centideg2"] * 1.0e-4,
        v_deg2=values["v_gsas_centideg2"] * 1.0e-4,
        w_deg2=values["w_gsas_centideg2"] * 1.0e-4,
        x_deg=values["x_gsas_centideg"] * 1.0e-2,
        y_deg=values["y_gsas_centideg"] * 1.0e-2,
    )
    parameters = case["parameters"]
    phases = []
    for index, reflection in enumerate(parameters["reflections"]):
        batch = phasesmith.ReflectionBatch(
            [f"oracle-{index}"],
            [reflection["hkl"]],
            [reflection["d_spacing_angstrom"]],
            [reflection["position_deg"]],
            [parameters["base_integrated_intensities"][index]],
        )
        phases.append(
            phasesmith.Phase(
                reflection["phase_id"],
                reflection["phase_id"],
                batch,
                scale=parameters["phase_scales"][index],
            )
        )
        widths = phasesmith.cw_profile_parameters([reflection["position_deg"]], instrument)
        assert 1.0e4 * widths.gaussian_variance_deg2[0] == pytest.approx(
            reflection["sigma2_centideg2"], rel=7e-16
        )
        assert 100.0 * widths.lorentzian_fwhm_deg[0] == pytest.approx(
            reflection["gamma_centideg"], rel=2.5e-12
        )
    x = fixture.arrays[case["arrays"]["x"]]
    background = np.full(x.size, parameters["background"])
    actual = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x, background=background),
        instrument,
        phases,
        options=phasesmith.CalculationOptions(
            support_fwhm=10_000.0,
            return_phase_components=True,
        ),
    )
    oracle = fixture.arrays[case["arrays"]["ycalc"]]
    normalized_maximum_error = float(np.max(np.abs(actual.y - oracle)) / np.max(np.abs(oracle)))
    assert normalized_maximum_error < 2.8e-6
    for index, phase in enumerate(phases):
        profile = fixture.arrays[case["arrays"][f"phase_{index}_profile"]]
        expected_component = (
            parameters["base_integrated_intensities"][index]
            * parameters["phase_scales"][index]
            * profile
        )
        component_error = float(
            np.max(np.abs(actual.phase_y(phase.phase_id) - expected_component))
            / np.max(np.abs(expected_component))
        )
        assert component_error < 2.8e-6
        row = actual.derivatives.global_parameter_names.index(f"phase[{phase.phase_id}].scale")
        expected_scale_row = parameters["base_integrated_intensities"][index] * profile
        derivative_error = float(
            np.max(np.abs(actual.derivatives.global_jacobian[row] - expected_scale_row))
            / np.max(np.abs(expected_scale_row))
        )
        assert derivative_error < 2.8e-6


def test_neutron_fixture_records_pin_public_arrays_and_reflections() -> None:
    fixture = load_fixture(NEUTRON_FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_neutron_cw.py"
    assert fixture.manifest["fixture_id"] == "gsasii_neutron_cw_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert (
        fixture.manifest["provenance"]["generator_sha256"]
        == hashlib.sha256(generator.read_bytes()).hexdigest()
    )
    assert fixture.manifest["source"]["histogram_type"] == "PNC"
    assert fixture.arrays["x_deg"].shape == fixture.arrays["ycalc"].shape == (7_001,)
    assert fixture.arrays["background"].shape == (7_001,)
    assert fixture.arrays["reflection_list"].shape == (128, 15)
    assert np.max(fixture.arrays["ycalc"]) > np.max(fixture.arrays["background"])


def test_tof_fixture_records_pin_public_arrays_and_reflections() -> None:
    fixture = load_fixture(TOF_FIXTURE_PATH)
    generator = REPOSITORY_ROOT / "oracle" / "scripts" / "generate_tof.py"
    assert fixture.manifest["fixture_id"] == "gsasii_tof_v1"
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert (
        fixture.manifest["provenance"]["generator_sha256"]
        == hashlib.sha256(generator.read_bytes()).hexdigest()
    )
    assert fixture.manifest["source"]["histogram_type"] == "PNT"
    assert fixture.arrays["x_us"].shape == fixture.arrays["ycalc"].shape == (6_000,)
    assert fixture.arrays["background"].shape == (6_000,)
    assert fixture.arrays["reflection_list"].shape == (2_066, 18)
    assert np.max(fixture.arrays["ycalc"]) > np.max(fixture.arrays["background"])

    values = fixture.manifest["input_parameters"]["instrument"]
    instrument = phasesmith.TofInstrument(
        **{
            key: value
            for key, value in values.items()
            if key not in {"flight_path_m", "two_theta_deg"}
        }
    )
    reflections = fixture.arrays["reflection_list"]
    actual = phasesmith.tof_profile_parameters(reflections[:, 4], instrument)
    np.testing.assert_array_equal(actual.position_us, reflections[:, 5])
    np.testing.assert_array_equal(actual.gaussian_variance_us2, reflections[:, 6])
    np.testing.assert_allclose(actual.lorentzian_fwhm_us, reflections[:, 7], rtol=9e-13)
    np.testing.assert_allclose(actual.alpha_per_us, reflections[:, 12], rtol=3e-16)
    np.testing.assert_allclose(actual.beta_per_us, reflections[:, 13], rtol=4e-16)


@pytest.mark.parametrize("case_index", [1, 2, 3])
def test_tof_profiles_derivatives_and_moments_against_pinned_gsasii(
    case_index: int,
) -> None:
    fixture = load_fixture(TOF_FIXTURE_PATH)
    case = fixture.cases[case_index]
    parameters = case["parameters"]
    x = fixture.arrays[case["arrays"]["x"]]
    gaussian_fwhm = 2.3548200450309493 * np.sqrt(parameters["sigma2_us2"])
    actual = phasesmith.profile_tof(
        x,
        parameters["position_us"],
        parameters["alpha_per_us"],
        parameters["beta_per_us"],
        gaussian_fwhm,
        parameters["gamma_us"],
    )
    d_sigma2 = (
        actual.d_gaussian_fwhm
        * 2.3548200450309493
        / (2.0 * np.sqrt(parameters["sigma2_us2"]))
    )
    comparisons = {
        "value": (actual.value, 2.5e-4),
        "d_position": (actual.d_position, 2.2e-3),
        "d_alpha": (actual.d_alpha, 2.3e-2),
        "d_beta": (actual.d_beta, 4.0e-4),
        "d_sigma2": (d_sigma2, 2.2e-3),
        "d_gamma": (actual.d_lorentzian_fwhm, 1.8e-3),
    }
    for name, (computed, tolerance) in comparisons.items():
        oracle = fixture.arrays[case["arrays"][name]]
        normalized_error = float(
            np.max(np.abs(computed - oracle)) / np.max(np.abs(oracle))
        )
        assert normalized_error < tolerance

    area = np.trapezoid(actual.value, x)
    centroid = np.trapezoid(x * actual.value, x) / area
    third = np.trapezoid((x - centroid) ** 3 * actual.value, x) / area
    moments = case["sampled_moments"]
    assert area == pytest.approx(moments["integral"], rel=1.0e-5)
    assert centroid == pytest.approx(moments["centroid_us"], abs=3.0e-3)
    assert third == pytest.approx(moments["third_central_moment_us3"], rel=1.0e-3)


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_neutron_symmetric_and_fcj_profiles_against_pinned_gsasii(case_index: int) -> None:
    fixture = load_fixture(NEUTRON_FIXTURE_PATH)
    values = fixture.manifest["input_parameters"]["instrument"]
    instrument = phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=values["wavelength_angstrom"],
        u_deg2=values["u_gsas_centideg2"] * 1.0e-4,
        v_deg2=values["v_gsas_centideg2"] * 1.0e-4,
        w_deg2=values["w_gsas_centideg2"] * 1.0e-4,
        x_deg=values["x_gsas_centideg"] * 1.0e-2,
        y_deg=values["y_gsas_centideg"] * 1.0e-2,
    )
    experiment = phasesmith.ConstantWavelengthExperiment.neutron(instrument)
    case = fixture.cases[case_index]
    parameters = case["parameters"]
    batch = phasesmith.ReflectionGeometryBatch(
        [parameters["hkl"]],
        [parameters["d_spacing_angstrom"]],
        [parameters["position_deg"]],
        [1.0],
    )
    widths = phasesmith.cw_profile_parameters([parameters["position_deg"]], instrument)
    assert 1.0e4 * widths.gaussian_variance_deg2[0] == pytest.approx(
        parameters["sigma2_centideg2"], rel=5e-16
    )
    x = fixture.arrays[case["arrays"]["x"]]
    symmetric = phasesmith.calculate_monochromatic_cw_pattern(
        x, batch, experiment, support_fwhm=10_000.0
    ).y
    symmetric_oracle = fixture.arrays[case["arrays"]["symmetric_profile"]]
    symmetric_error = float(
        np.max(np.abs(symmetric - symmetric_oracle)) / np.max(np.abs(symmetric_oracle))
    )
    assert symmetric_error < 3.3e-5

    fcj = phasesmith.calculate_neutron_fcj_pattern(
        x,
        batch,
        experiment,
        phasesmith.FcjGeometry(**parameters["public_equal_height_mapping"]),
        support_fwhm=10_000.0,
    ).y
    fcj_oracle = fixture.arrays[case["arrays"]["fcj_profile"]]
    fcj_error = float(np.max(np.abs(fcj - fcj_oracle)) / np.max(np.abs(fcj_oracle)))
    assert fcj_error < 4.3e-4
    area = np.trapezoid(fcj, x)
    centroid = np.trapezoid(x * fcj, x) / area
    third = np.trapezoid((x - centroid) ** 3 * fcj, x) / area
    moments = case["sampled_fcj_moments"]
    assert area == pytest.approx(moments["integral"], rel=1.7e-6)
    assert centroid == pytest.approx(moments["centroid_deg"], abs=1.4e-4)
    if abs(moments["third_central_moment_deg3"]) > 1.0e-8:
        assert np.sign(third) == np.sign(moments["third_central_moment_deg3"])
        assert third == pytest.approx(moments["third_central_moment_deg3"], rel=0.55)


def _sample_fixture_models(
    fixture: phasesmith.oracle.OracleFixture,
) -> tuple[
    phasesmith.ConstantWavelengthInstrument,
    phasesmith.CompositePhysicsProvider,
    dict[str, object],
]:
    translation = fixture.cases[0]["parameters"]["public_translation"]
    instrument_values = fixture.manifest["input_parameters"]["instrument"]
    instrument = phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=instrument_values["wavelength_angstrom"],
        u_deg2=instrument_values["u_gsas_centideg2"] * 1.0e-4,
        v_deg2=instrument_values["v_gsas_centideg2"] * 1.0e-4,
        w_deg2=instrument_values["w_gsas_centideg2"] * 1.0e-4,
        x_deg=instrument_values["x_gsas_centideg"] * 1.0e-2,
        y_deg=instrument_values["y_gsas_centideg"] * 1.0e-2,
    )
    provider = phasesmith.CompositePhysicsProvider(
        (
            phasesmith.IsotropicSizeBroadening(
                translation["crystallite_size_nm"], translation["shape_factor"]
            ),
            phasesmith.IsotropicMicrostrainBroadening(translation["rms_microstrain"]),
            phasesmith.MarchDollasePreferredOrientation(
                translation["march_ratio"],
                tuple(translation["preferred_axis_hkl"]),
                phasesmith.ReciprocalMetric(translation["reciprocal_metric_angstrom_minus2"]),
            ),
        )
    )
    return instrument, provider, translation


def test_sample_reflection_parameters_against_pinned_gsasii() -> None:
    fixture = load_fixture(SAMPLE_FIXTURE_PATH)
    instrument, provider, _translation = _sample_fixture_models(fixture)
    reflections = fixture.arrays["reflection_list"]
    columns = fixture.manifest["source"]["reflection_columns"]
    column = {name: columns.index(name) for name in columns}
    batch = phasesmith.ReflectionGeometryBatch(
        reflections[:, :3].astype(np.int64),
        reflections[:, column["d_spacing_angstrom"]],
        reflections[:, column["position_deg"]],
        np.ones(reflections.shape[0]),
    )
    contribution = provider.evaluate(phasesmith.PhysicsContext(batch, instrument))
    widths = phasesmith.cw_profile_parameters(batch.two_theta_deg, instrument)
    np.testing.assert_allclose(
        1.0e4 * (widths.gaussian_variance_deg2 + contribution.gaussian_variance_deg2),
        reflections[:, column["sigma2_centideg2"]],
        rtol=8e-16,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        100.0 * (widths.lorentzian_fwhm_deg + contribution.lorentzian_fwhm_deg),
        reflections[:, column["gamma_centideg"]],
        rtol=5e-16,
        atol=1.1e-14,
    )
    np.testing.assert_allclose(
        contribution.intensity_multiplier,
        reflections[:, column["preferred_orientation"]],
        rtol=9e-16,
        atol=9e-16,
    )
    x = fixture.arrays["x_deg"]
    ycalc = fixture.arrays["ycalc"]
    background = fixture.arrays["background"]
    assert x.shape == ycalc.shape == background.shape == (4_501,)
    assert np.all(np.diff(x) > 0.0)
    assert np.max(ycalc) > 0.0


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_sample_profiles_and_moments_against_pinned_gsasii(case_index: int) -> None:
    fixture = load_fixture(SAMPLE_FIXTURE_PATH)
    instrument, provider, _translation = _sample_fixture_models(fixture)
    case = fixture.cases[case_index]
    parameters = case["parameters"]
    x = fixture.arrays[case["arrays"]["x"]]
    oracle = parameters["preferred_orientation"] * fixture.arrays[case["arrays"]["profile"]]
    batch = phasesmith.ReflectionGeometryBatch(
        [parameters["hkl"]],
        [parameters["d_spacing_angstrom"]],
        [parameters["position_deg"]],
        [1.0],
    )
    actual = phasesmith.calculate_cw_pattern(
        x, batch, instrument, physics=provider, support_fwhm=10_000.0
    ).y
    normalized_maximum_error = float(np.max(np.abs(actual - oracle)) / np.max(np.abs(oracle)))
    # Local to the pinned #5838 TCH implementation and the public unit
    # translation validated independently against every reflection above.
    assert normalized_maximum_error < 2.0e-5

    actual_area = np.trapezoid(actual, x)
    actual_centroid = np.trapezoid(x * actual, x) / actual_area
    actual_second = np.trapezoid((x - actual_centroid) ** 2 * actual, x) / actual_area
    oracle_moments = case["sampled_corrected_moments"]
    assert actual_area == pytest.approx(oracle_moments["integral"], rel=2.1e-5)
    assert actual_centroid == pytest.approx(oracle_moments["centroid_deg"], abs=3e-14)
    assert actual_second == pytest.approx(oracle_moments["second_central_moment_deg2"], rel=4.1e-5)


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
    actual = phasesmith.profile_tch(
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
    actual = phasesmith.accumulate_tch(
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
    model = phasesmith.ConstantWavelengthInstrument(**dict(parameters["instrument"]))
    position = parameters["position_deg"]
    x = fixture.arrays[case["arrays"]["x"]]
    widths = phasesmith.cw_profile_parameters([position], model)
    assert widths.gaussian_variance_deg2[0] == pytest.approx(
        parameters["gaussian_variance_deg2"], rel=3e-15
    )
    assert widths.gaussian_fwhm_deg[0] == pytest.approx(parameters["gaussian_fwhm_deg"], rel=3e-15)
    assert widths.lorentzian_fwhm_deg[0] == pytest.approx(
        parameters["lorentzian_fwhm_deg"], rel=3e-15
    )

    actual = phasesmith.accumulate_cw(
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
    model = phasesmith.ConstantWavelengthInstrument(**dict(parameters["instrument"]))
    x = fixture.arrays[case["arrays"]["x"]]
    actual = phasesmith.accumulate_cw(
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
    actual = phasesmith.profile_fcj(
        x,
        parameters["position_deg"],
        parameters["gaussian_fwhm_deg"],
        parameters["lorentzian_fwhm_deg"],
        phasesmith.FcjGeometry(**dict(mapping)),
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
    actual = phasesmith.profile_fcj(
        x,
        parameters["position_deg"],
        parameters["gaussian_fwhm_deg"],
        parameters["lorentzian_fwhm_deg"],
        phasesmith.FcjGeometry(0.0, 0.0),
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
    instrument = phasesmith.ConstantWavelengthInstrument(**dict(parameters["instrument"]))
    components = phasesmith.WavelengthComponents(
        parameters["wavelengths_angstrom"], parameters["relative_intensities"]
    )
    geometry = phasesmith.FcjGeometry(**dict(parameters["public_equal_height_mapping"]))
    x = fixture.arrays[case["arrays"]["x"]]
    oracle = fixture.arrays[case["arrays"]["ycalc"]]
    actual = phasesmith.accumulate_cw_fcj_components(
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
