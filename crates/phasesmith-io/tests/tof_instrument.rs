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
        "INS  2PRCF      1   12   0.01000    0NNNNNNNNNNNNNNNNNNNN\n",
        "INS  2PRCF 1   0.000000E+00   0.142760E+00   0.557661E-01   0.344498E-02\n",
        "INS  2PRCF 2   0.000000E+00   0.977951E+02   0.000000E+00   0.000000E+00\n",
    );
    let parsed =
        parse_gsas_tof_instrument_text(text, 2, GsasTofInstrumentReadLimits::default()).unwrap();
    assert_eq!(parsed.profile_function, 1);
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
