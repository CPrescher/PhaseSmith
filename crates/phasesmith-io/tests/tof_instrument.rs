//! Integration tests for bounded legacy GSAS TOF instrument import.

use phasesmith_io::{
    GsasTofInstrumentIoError, GsasTofInstrumentReadLimits, parse_gsas_tof_instrument_text,
};

const BANKS: &str = concat!(
    "INS  1 ICONS12345 0 1 0\n",
    "INS  1PRCF1     3 21 0.002\n",
    "INS  1PRCF11 0.1 0.2 0.3 0\n",
    "INS  1PRCF12 4 5 0 0\n",
    "INS  2 ICONS22581.63 0 4.41 0\n",
    "INS  2BNKPAR     3.183    90.000     0.000     0.000     0.200    1    1\n",
    "INS  2PRCF1     3 21 0.002\n",
    "INS  2PRCF11 0.257460 0.091563 0.017334 0\n",
    "INS  2PRCF12 10 203.581 0 10.651\n",
);

#[test]
fn translates_selected_profile_function_three_bank() {
    let parsed =
        parse_gsas_tof_instrument_text(BANKS, 2, GsasTofInstrumentReadLimits::default()).unwrap();
    assert_eq!(parsed.bank, 2);
    assert_eq!(parsed.profile_function, 3);
    assert!(parsed.source_path.is_none());
    assert!(parsed.incident_spectrum.is_none());
    assert_eq!(
        parsed.bank_geometry.unwrap().two_theta_deg.to_bits(),
        90.0_f64.to_bits()
    );
    assert_eq!(parsed.instrument.zero_us.to_bits(), 4.41_f64.to_bits());
    assert_eq!(
        parsed.instrument.difc_us_per_angstrom.to_bits(),
        22_581.63_f64.to_bits()
    );
    assert_eq!(
        parsed.instrument.difb_us_angstrom.to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        parsed.instrument.sigma1_us2_per_angstrom2.to_bits(),
        10.0_f64.to_bits()
    );
    assert_eq!(
        parsed.instrument.sigma2_us2_per_angstrom4.to_bits(),
        203.581_f64.to_bits()
    );
}

#[test]
fn translates_legacy_profile_function_one_bank() {
    let text = concat!(
        "INS  2 ICONS   4368.97      0.02      2.11         0\n",
        "INS  2BNKPAR    1.0000     88.05     53.70    .01905     .3048   16   17\n",
        "INS  2PRCF      1   12   0.01000    0NNNNNNNNNNNNNNNNNNNN\n",
        "INS  2PRCF 1   0.000000E+00   0.142760E+00   0.557661E-01   0.344498E-02\n",
        "INS  2PRCF 2   0.000000E+00   0.977951E+02   0.000000E+00   0.000000E+00\n",
    );
    let parsed =
        parse_gsas_tof_instrument_text(text, 2, GsasTofInstrumentReadLimits::default()).unwrap();
    assert_eq!(parsed.profile_function, 1);
    assert_eq!(
        parsed.bank_geometry.unwrap().two_theta_deg.to_bits(),
        88.05_f64.to_bits()
    );
    assert_eq!(parsed.instrument.zero_us.to_bits(), 2.11_f64.to_bits());
    assert_eq!(
        parsed.instrument.alpha_coefficient.to_bits(),
        0.142_760_f64.to_bits()
    );
    assert_eq!(
        parsed.instrument.beta0_per_us.to_bits(),
        0.055_766_1_f64.to_bits()
    );
    assert_eq!(
        parsed.instrument.beta1_angstrom4_per_us.to_bits(),
        0.003_444_98_f64.to_bits()
    );
    assert_eq!(
        parsed.instrument.sigma1_us2_per_angstrom2.to_bits(),
        97.795_1_f64.to_bits()
    );
    assert_eq!(
        parsed.instrument.sigma2_us2_per_angstrom4.to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn translates_type_four_incident_spectrum_in_microseconds() {
    let text = concat!(
        "INS  2 ICONS   4368.97      0.02      2.11         0\n",
        "INS  2I ITYP    4    0.7500    8.1904     76288\n",
        "INS  2ICOFF1   0.177427E+04   0.783794E+07   0.237297E+02   0.305645E+04\n",
        "INS  2ICOFF2  -0.600307E+03  -0.146005E+03  -0.147656E+03   0.442342E+03\n",
        "INS  2ICOFF3  -0.302364E+03   0.885096E+02  -0.968997E+01   0.000000E+00\n",
        "INS  2PRCF      1   12   0.01000    0NNNNNNNNNNNNNNNNNNNN\n",
        "INS  2PRCF 1   0.000000E+00   0.142760E+00   0.557661E-01   0.344498E-02\n",
        "INS  2PRCF 2   0.000000E+00   0.977951E+02   0.000000E+00   0.000000E+00\n",
    );
    let parsed =
        parse_gsas_tof_instrument_text(text, 2, GsasTofInstrumentReadLimits::default()).unwrap();
    let spectrum = parsed.incident_spectrum.expect("incident spectrum");
    assert_eq!(spectrum.min_tof_us.to_bits(), 750.0_f64.to_bits());
    assert!((spectrum.max_tof_us - 8_190.4).abs() <= 1.0e-9);
    assert_eq!(spectrum.coefficients[0].to_bits(), 1_774.27_f64.to_bits());
    let point = spectrum.evaluate(2_500.0).expect("spectrum value");
    assert!(point.value.is_finite() && point.value > 0.0);
    assert!(point.d_value_d_tof_us.is_finite());
}

#[test]
fn accepts_utf8_bom() {
    let parsed = parse_gsas_tof_instrument_text(
        &format!("\u{feff}{BANKS}"),
        2,
        GsasTofInstrumentReadLimits::default(),
    )
    .unwrap();
    assert_eq!(parsed.instrument.zero_us.to_bits(), 4.41_f64.to_bits());
}

#[test]
fn rejects_missing_nonfinite_unsupported_and_oversized_inputs() {
    assert!(matches!(
        parse_gsas_tof_instrument_text(
            &BANKS.replace("90.000", "180.000"),
            2,
            GsasTofInstrumentReadLimits::default()
        ),
        Err(GsasTofInstrumentIoError::InvalidRecord {
            record: "BNKPAR",
            ..
        })
    ));
    assert!(matches!(
        parse_gsas_tof_instrument_text(
            "INS  1 ICONS1 2 3 4\n",
            2,
            GsasTofInstrumentReadLimits::default()
        ),
        Err(GsasTofInstrumentIoError::MissingRecord { .. })
    ));
    assert!(matches!(
        parse_gsas_tof_instrument_text(
            &BANKS.replace("22581.63", "nan"),
            2,
            GsasTofInstrumentReadLimits::default()
        ),
        Err(GsasTofInstrumentIoError::InvalidRecord { .. })
    ));
    assert!(matches!(
        parse_gsas_tof_instrument_text(
            &BANKS.replace("     3 21", "     2 21"),
            2,
            GsasTofInstrumentReadLimits::default()
        ),
        Err(GsasTofInstrumentIoError::UnsupportedProfileFunction { found: 2, .. })
    ));
    let unsupported_spectrum = format!("INS  2I ITYP    3    0.7500    8.1904     76288\n{BANKS}");
    assert!(matches!(
        parse_gsas_tof_instrument_text(
            &unsupported_spectrum,
            2,
            GsasTofInstrumentReadLimits::default()
        ),
        Err(GsasTofInstrumentIoError::UnsupportedIncidentSpectrumFunction { found: 3, .. })
    ));
    assert!(matches!(
        parse_gsas_tof_instrument_text(
            BANKS,
            2,
            GsasTofInstrumentReadLimits {
                max_bytes: BANKS.len() - 1,
            }
        ),
        Err(GsasTofInstrumentIoError::ByteLimitExceeded { .. })
    ));
}

#[test]
fn keeps_profile_only_legacy_files_without_bank_geometry_compatible() {
    let text = concat!(
        "INS  2 ICONS   4368.97      0.02      2.11         0\n",
        "INS  2PRCF      1   12   0.01000    0NNNNNNNNNNNNNNNNNNNN\n",
        "INS  2PRCF 1   0.000000E+00   0.142760E+00   0.557661E-01   0.344498E-02\n",
        "INS  2PRCF 2   0.000000E+00   0.977951E+02   0.000000E+00   0.000000E+00\n",
    );
    let parsed =
        parse_gsas_tof_instrument_text(text, 2, GsasTofInstrumentReadLimits::default()).unwrap();
    assert!(parsed.bank_geometry.is_none());
}

#[test]
fn rejects_invalid_limits_and_banks() {
    assert!(matches!(
        parse_gsas_tof_instrument_text(BANKS, 0, GsasTofInstrumentReadLimits::default()),
        Err(GsasTofInstrumentIoError::InvalidBank)
    ));
    assert!(matches!(
        parse_gsas_tof_instrument_text(BANKS, 2, GsasTofInstrumentReadLimits { max_bytes: 0 }),
        Err(GsasTofInstrumentIoError::InvalidLimits)
    ));
}
