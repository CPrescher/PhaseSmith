//! Pure-Rust `PbSO4` neutron validation regression and Python parity.

use std::path::Path;
use std::process::Command;

use phasesmith_validation::{ValidationStatus, run_pbso4_neutron_validation};

fn dataset() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/data/gsasii-pbso4-cw")
}

fn measurements(
    report: &phasesmith_validation::RealDataValidationReport,
) -> std::collections::BTreeMap<&str, f64> {
    report
        .checks
        .iter()
        .filter_map(|check| check.measured.map(|value| (check.check_id.as_str(), value)))
        .collect()
}

#[test]
#[ignore = "expensive checksum-pinned real-data refinement"]
fn pure_rust_neutron_runner_reproduces_the_reviewed_scientific_gate() {
    let report = run_pbso4_neutron_validation(&dataset()).unwrap();
    assert_eq!(report.status, ValidationStatus::Passed);
    assert_eq!(report.sample_count, 2_681);
    let measured = measurements(&report);
    assert!((measured["poisson_rwp"] - 0.042_171_782_496_769_33).abs() < 2.0e-10);
    assert!((measured["unit_weight_rwp"] - 0.045_260_138_152_294_72).abs() < 2.0e-10);
    assert!((measured["profile_correlation"] - 0.996_644_799_175_426_8).abs() < 2.0e-10);
    assert!(
        (measured["reference_cell_relative_error"] - 0.001_167_108_534_439_469_3).abs() < 2.0e-10
    );
}

#[test]
#[ignore = "expensive checksum-pinned Rust/Python differential refinement"]
fn pure_rust_neutron_report_matches_the_python_scripting_runner() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let rust = run_pbso4_neutron_validation(&dataset()).unwrap();
    let script = r"
import json, sys
from phasesmith import RadiationProbe
from phasesmith.validation import run_pbso4_cw_validation
r = run_pbso4_cw_validation(sys.argv[1], RadiationProbe.NEUTRON)
print(json.dumps(r.to_record(), sort_keys=True))
";
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(dataset())
        .env(
            "PYTHONPATH",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python"),
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
    let rust_measurements = measurements(&rust);
    for python_check in python["checks"].as_array().unwrap() {
        let Some(value) = python_check["measured"].as_f64() else {
            continue;
        };
        let check_id = python_check["check_id"].as_str().unwrap();
        assert!((value - rust_measurements[check_id]).abs() < 1.0e-5);
    }
}
