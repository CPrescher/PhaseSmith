//! Contract and real-fixture tests for bounded native powder readers.

use std::path::Path;

use phasesmith_io::{
    PowderFormat, PowderIoError, PowderReadLimits, parse_powder_text, read_powder_file,
};

fn parse(text: &str) -> phasesmith_io::PowderData {
    parse_powder_text(text, PowderFormat::Auto, 1, PowderReadLimits::default())
        .expect("fixture must parse")
}

#[test]
fn reads_two_and_three_column_patterns() {
    let two = parse("# 2theta intensity\n5.000 174\n5.020 176\n");
    assert_eq!(two.format, PowderFormat::Columns);
    assert_eq!(two.pattern.x_deg, [5.0, 5.02]);
    assert_eq!(two.pattern.observed_y.as_deref(), Some(&[174.0, 176.0][..]));
    assert_eq!(two.pattern.uncertainty, None);

    let three = parse("1, 10, 2\n2, 11, 3 # comment\n");
    assert_eq!(three.pattern.uncertainty.as_deref(), Some(&[2.0, 3.0][..]));
}

#[test]
fn reads_selected_fxye_bank_and_converts_centidegrees() {
    let text = "Example\n\
        BANK 1 2 2 CONS 50.0 2.0 0 0 fxye\n\
         500.0 100.0 10.0\n\
         502.0 121.0 11.0\n\
        BANK 2 1 1 CONS 1000.0 1.0 0 0 FXYE\n\
         1000.0 9.0 3.0\n";
    let data = parse_powder_text(text, PowderFormat::Auto, 2, PowderReadLimits::default())
        .expect("selected FXYE bank must parse");

    assert_eq!(data.format, PowderFormat::GsasFxye);
    assert_eq!(data.bank, Some(2));
    assert_eq!(data.pattern.x_deg, [10.0]);
    assert_eq!(data.pattern.observed_y.as_deref(), Some(&[9.0][..]));
}

#[test]
fn reads_packed_constant_step_std_bank() {
    let data = parse(concat!(
        "Packed example\n",
        "BANK 1 3 1 CONST 1000 2.5 0 0 STD\n",
        "     100 2    50     121\n",
    ));

    assert_eq!(data.format, PowderFormat::GsasStd);
    assert_eq!(data.bank, Some(1));
    assert_eq!(data.pattern.x_deg, [10.0, 10.025, 10.05]);
    assert_eq!(
        data.pattern.observed_y.as_deref(),
        Some(&[100.0, 50.0, 121.0][..])
    );
    assert_eq!(
        data.pattern.uncertainty.as_deref(),
        Some(&[10.0, 5.0, 11.0][..])
    );
}

#[test]
fn rejects_invalid_arrays_and_mixed_columns() {
    let cases = [
        ("1 2\n1 3\n", "strictly increasing"),
        ("1 2 0\n2 3 1\n", "positive"),
        ("1 2\n2 3 4\n", "expected 2 columns"),
        ("1 2 3 4\n", "expected two or three columns"),
        ("1 nan\n", "finite"),
    ];
    for (text, expected) in cases {
        let error = parse_powder_text(text, PowderFormat::Auto, 1, PowderReadLimits::default())
            .expect_err("invalid fixture must fail");
        assert!(
            error.to_string().contains(expected),
            "{error:?} did not contain {expected:?}"
        );
    }
}

#[test]
fn rejects_unsupported_or_malformed_gsas_records() {
    let cases = [
        (
            "BANK 1 2 2 CONS 1 1 0 0 ESD\n1 2 3\n",
            "packed constant-step",
        ),
        (
            "BANK 1 1 1 TIME_MAP 1 1 0 0 STD\n 1    10\n",
            "constant-step",
        ),
        (
            "BANK 1 1 1 CONST 1 1 0 0 STD ESD\n     10\n",
            "packed constant-step",
        ),
        (
            "BANK 1 1 1 CONST 1 1 0 0 STD\nxx    10\n",
            "invalid fixed-width record",
        ),
        (
            "BANK 1 1 1 CONST 1 1 0 0 STD\n      nan\n",
            "invalid fixed-width record",
        ),
        (
            "BANK 1 1 1 CONS 1 1 0 0 FXYE\n100 2 1\n\
             BANK 2 1 1 CONST 1 1 0 0 STD\n     1\n",
            "only unpacked GSAS FXYE",
        ),
    ];
    for (text, expected) in cases {
        let error = parse_powder_text(text, PowderFormat::Auto, 1, PowderReadLimits::default())
            .expect_err("unsupported fixture must fail");
        assert!(error.to_string().contains(expected), "{error:?}");
    }
}

#[test]
fn enforces_limits_and_reports_missing_bank() {
    let row_error = parse_powder_text(
        "1 2\n2 3\n",
        PowderFormat::Columns,
        1,
        PowderReadLimits {
            max_rows: 1,
            ..PowderReadLimits::default()
        },
    )
    .expect_err("row limit must be enforced");
    assert!(matches!(
        row_error,
        PowderIoError::RowLimitExceeded { maximum: 1 }
    ));

    let byte_error = parse_powder_text(
        "1 2\n",
        PowderFormat::Columns,
        1,
        PowderReadLimits {
            max_bytes: 2,
            ..PowderReadLimits::default()
        },
    )
    .expect_err("byte limit must be enforced");
    assert!(matches!(
        byte_error,
        PowderIoError::ByteLimitExceeded { .. }
    ));

    let missing = parse_powder_text(
        "BANK 1 1 1 CONS 1 1 0 0 FXYE\n100 2 1\n",
        PowderFormat::GsasFxye,
        2,
        PowderReadLimits::default(),
    )
    .expect_err("missing bank must fail");
    assert!(matches!(
        missing,
        PowderIoError::MissingBank {
            requested: 2,
            available
        } if available == [1]
    ));
}

#[test]
fn reads_pinned_real_fxye_fixture_without_python() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/data/aps-sucrose-11bmb/11bmb_8716.fxye");
    let data = read_powder_file(&path, PowderFormat::Auto, 1, PowderReadLimits::default())
        .expect("pinned APS FXYE dataset must parse natively");

    assert_eq!(data.source_path, Some(path));
    assert_eq!(data.format, PowderFormat::GsasFxye);
    assert_eq!(data.pattern.sample_count(), 49_494);
    assert_eq!(data.pattern.x_deg.first(), Some(&0.5));
    assert_eq!(data.pattern.x_deg.last(), Some(&49.986_431_79));
}
