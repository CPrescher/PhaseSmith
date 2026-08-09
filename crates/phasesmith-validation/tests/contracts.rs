//! Native validation report and checksum-manifest contracts.

use std::fs;

use phasesmith_validation::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationStatus,
    validation_dataset, validation_datasets, verify_validation_dataset,
};

fn check(id: &str, status: ValidationStatus) -> ValidationCheck {
    ValidationCheck::new(
        id,
        status,
        "stable detail",
        Some(0.125),
        Some("value <= 0.2".to_owned()),
    )
    .unwrap()
}

#[test]
fn report_status_and_json_match_the_python_contract() {
    let report = RealDataValidationReport::new(
        "fixture",
        12,
        Some(3),
        0.5,
        vec![
            check("passed", ValidationStatus::Passed),
            check("blocked", ValidationStatus::Blocked),
        ],
        vec!["diagnostic".to_owned()],
    )
    .unwrap();
    assert_eq!(report.status, ValidationStatus::Blocked);
    let json = report.to_json().unwrap();
    assert_eq!(RealDataValidationReport::from_json(&json).unwrap(), report);
    assert!(json.starts_with("{\"dataset_id\":\"fixture\",\"status\":\"blocked\""));

    let failed = RealDataValidationReport::new(
        "fixture",
        12,
        None,
        0.0,
        vec![
            check("blocked", ValidationStatus::Blocked),
            check("failed", ValidationStatus::Failed),
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(failed.status, ValidationStatus::Failed);
}

#[test]
fn decoded_reports_are_revalidated() {
    let invalid = r#"{
        "dataset_id":"fixture",
        "status":"passed",
        "sample_count":1,
        "reflection_count":null,
        "elapsed_seconds":0.0,
        "checks":[{
            "check_id":"failed",
            "status":"failed",
            "detail":"detail",
            "measured":null,
            "criterion":null
        }],
        "notes":[]
    }"#;
    assert!(RealDataValidationReport::from_json(invalid).is_err());
    assert!(
        ValidationCheck::new(
            "finite",
            ValidationStatus::Passed,
            "detail",
            Some(f64::NAN),
            None
        )
        .is_err()
    );
}

#[test]
fn checked_in_python_baseline_reports_decode_without_schema_translation() {
    let baseline = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/results/2026-08-07-baseline.json");
    let source = fs::read_to_string(baseline).unwrap();
    let suite: serde_json::Value = serde_json::from_str(&source).unwrap();
    let reports = suite["reports"].as_array().unwrap();
    assert!(!reports.is_empty());
    for report in reports {
        let encoded = serde_json::to_string(report).unwrap();
        RealDataValidationReport::from_json(&encoded).unwrap();
    }
}

#[test]
fn built_in_dataset_manifests_match_the_python_registry() {
    let datasets = validation_datasets();
    assert_eq!(
        datasets
            .iter()
            .map(|dataset| dataset.dataset_id.as_str())
            .collect::<Vec<_>>(),
        ["aps-sucrose-11bmb", "gsasii-pbso4-cw", "iucr-qarr-1g"]
    );
    assert_eq!(
        validation_dataset("aps-sucrose-11bmb").unwrap().files.len(),
        2
    );
    assert_eq!(
        validation_dataset("gsasii-pbso4-cw").unwrap().files.len(),
        5
    );
    assert_eq!(validation_dataset("iucr-qarr-1g").unwrap().files.len(), 5);
    assert!(matches!(
        validation_dataset("missing"),
        Err(DatasetVerificationError::UnknownDataset(_))
    ));
    for dataset in datasets {
        dataset.validate().unwrap();
    }
}

#[test]
fn local_dataset_verification_checks_size_before_digest() {
    let temporary = tempfile_dir("phasesmith-validation-size");
    fs::write(temporary.join("11bmb_8716.fxye"), b"wrong").unwrap();
    let error = verify_validation_dataset("aps-sucrose-11bmb", &temporary).unwrap_err();
    assert!(matches!(
        error,
        DatasetVerificationError::SizeMismatch { .. }
    ));

    fs::write(temporary.join("11bmb_8716.fxye"), vec![0_u8; 1_725_515]).unwrap();
    let error = verify_validation_dataset("aps-sucrose-11bmb", &temporary).unwrap_err();
    assert!(matches!(
        error,
        DatasetVerificationError::DigestMismatch { .. }
    ));
    fs::remove_dir_all(temporary).unwrap();
}

#[test]
fn checked_in_external_datasets_match_native_manifests_when_available() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/data");
    for dataset in validation_datasets() {
        let directory = root.join(&dataset.dataset_id);
        if directory.is_dir() {
            let paths = verify_validation_dataset(&dataset.dataset_id, &directory).unwrap();
            assert_eq!(paths.len(), dataset.files.len());
        }
    }
}

fn tempfile_dir(prefix: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    if path.exists() {
        fs::remove_dir_all(&path).unwrap();
    }
    fs::create_dir(&path).unwrap();
    path
}
