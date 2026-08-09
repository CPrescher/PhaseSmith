//! Checksum-pinned pure-Rust QARR validation and Python differential coverage.

use std::path::PathBuf;
use std::process::Command;

use phasesmith_validation::{ValidationStatus, run_qarr_1g_validation};

fn dataset() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("validation/data/iucr-qarr-1g")
}

#[test]
#[ignore = "expensive checksum-pinned three-phase real-data refinement"]
fn pure_rust_qarr_runner_passes_all_scientific_gates_and_is_deterministic() {
    let first = run_qarr_1g_validation(&dataset()).unwrap();
    let second = run_qarr_1g_validation(&dataset()).unwrap();
    assert_eq!(first.status, ValidationStatus::Passed);
    assert_eq!(first.sample_count, 7_251);
    assert_eq!(first.reflection_count, Some(110));
    assert!(
        first
            .checks
            .iter()
            .all(|check| check.status == ValidationStatus::Passed)
    );
    let measured = first
        .checks
        .iter()
        .filter_map(|check| check.measured.map(|value| (check.check_id.as_str(), value)))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert!((measured["poisson_rwp"] - 0.198_466_348_056_649_98).abs() < 2.0e-10);
    assert!((measured["unit_weight_rwp"] - 0.131_947_877_641_163_84).abs() < 2.0e-10);
    assert!((measured["profile_correlation"] - 0.990_603_724_009_948_3).abs() < 2.0e-10);
    assert!((measured["qpa_weight_fraction"] - 0.010_070_411_944_036_461).abs() < 2.0e-10);
    let measurements = |report: &phasesmith_validation::RealDataValidationReport| {
        report
            .checks
            .iter()
            .map(|check| (check.check_id.clone(), check.measured))
            .collect::<Vec<_>>()
    };
    assert_eq!(measurements(&first), measurements(&second));
    assert_eq!(first.notes, second.notes);
}

#[test]
#[ignore = "expensive checksum-pinned Rust/Python differential refinement"]
fn pure_rust_qarr_report_matches_the_python_scripting_runner() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let rust = run_qarr_1g_validation(&dataset()).unwrap();
    let script = r"
import json, os, sys
os.environ['PHASESMITH_VALIDATION_PYTHON_REFERENCE'] = '1'
from phasesmith.validation import run_qarr_1g_validation
r = run_qarr_1g_validation(sys.argv[1])
print(json.dumps(r.to_record(), sort_keys=True))
";
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(dataset())
        .env(
            "PYTHONPATH",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python"),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let python: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(python["dataset_id"], rust.dataset_id);
    assert_eq!(python["sample_count"], rust.sample_count);
    let rust_measurements = rust
        .checks
        .iter()
        .filter_map(|check| check.measured.map(|value| (check.check_id.as_str(), value)))
        .collect::<std::collections::BTreeMap<_, _>>();
    for python_check in python["checks"].as_array().unwrap() {
        let Some(value) = python_check["measured"].as_f64() else {
            continue;
        };
        let check_id = python_check["check_id"].as_str().unwrap();
        let tolerance = match check_id {
            "observed_grid" | "site_expansion" => 0.0,
            "poisson_rwp" => 3.0e-3,
            "unit_weight_rwp" => 2.0e-3,
            "profile_correlation" => 1.0e-3,
            "qpa_weight_fraction" => 1.0e-2,
            _ => continue,
        };
        assert!(
            (value - rust_measurements[check_id]).abs() <= tolerance,
            "{check_id}"
        );
    }
}
