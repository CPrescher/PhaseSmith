//! Pure-Rust APS sucrose validation regression and Python parity.

use std::path::Path;
use std::process::Command;

use phasesmith_validation::{ValidationStatus, run_sucrose_lebail_validation};

fn dataset() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/data/aps-sucrose-11bmb")
}

#[test]
fn pure_rust_sucrose_runner_reproduces_the_reviewed_scientific_gate() {
    let report = run_sucrose_lebail_validation(&dataset()).unwrap();
    assert_eq!(report.status, ValidationStatus::Passed);
    assert_eq!(report.sample_count, 23_003);
    assert_eq!(report.reflection_count, Some(811));
    let measurements = report
        .checks
        .iter()
        .filter_map(|check| check.measured.map(|value| (check.check_id.as_str(), value)))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert!((measurements["profile_improvement"] - 0.675_690_921_328_791_4).abs() < 1.0e-12);
    assert!((measurements["smoke_rwp"] - 0.187_867_714_269_909_14).abs() < 1.0e-12);
    assert!((measurements["profile_correlation"] - 0.991_587_841_863_939_1).abs() < 1.0e-12);
}

#[test]
fn pure_rust_sucrose_report_matches_the_python_scripting_runner() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let rust = run_sucrose_lebail_validation(&dataset()).unwrap();
    let script = r"
import json, sys
from phasesmith.validation import run_sucrose_lebail_validation
r = run_sucrose_lebail_validation(sys.argv[1])
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
    assert_eq!(python["reflection_count"], rust.reflection_count.unwrap());
    for rust_check in &rust.checks {
        let python_check = python["checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|check| check["check_id"] == rust_check.check_id)
            .unwrap();
        assert_eq!(python_check["status"], rust_check.status.as_str());
        if let Some(rust_value) = rust_check.measured {
            let python_value = python_check["measured"].as_f64().unwrap();
            assert!((python_value - rust_value).abs() < 1.0e-12);
        }
    }
}
