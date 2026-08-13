//! Python-free validation CLI contracts.

use std::process::Command;

fn dataset(dataset_id: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/data")
        .join(dataset_id)
}

#[test]
fn datasets_are_stable_and_unknown_runners_fail() {
    let binary = env!("CARGO_BIN_EXE_phasesmith-validation");
    let listed = Command::new(binary).arg("datasets").output().unwrap();
    assert!(listed.status.success());
    assert_eq!(
        String::from_utf8(listed.stdout).unwrap(),
        concat!(
            "ansto-echidna-lab6-cw-neutron\n",
            "aps-sucrose-11bmb\n",
            "gsasii-pbso4-cw\n",
            "iucr-qarr-1g\n",
            "iucr-qarr-1h\n",
            "nist-srm660c-lab6-xray\n",
            "lanl-nickel-tof\n",
            "powgen-lab6-tof-calibration\n",
        )
    );
    let manifest = Command::new(binary).arg("manifest").output().unwrap();
    assert!(manifest.status.success());
    let manifest: serde_json::Value = serde_json::from_slice(&manifest.stdout).unwrap();
    let datasets = manifest.as_array().unwrap();
    assert_eq!(datasets.len(), 8);
    let qarr_1h = datasets
        .iter()
        .find(|dataset| dataset["dataset_id"] == "iucr-qarr-1h")
        .unwrap();
    assert_eq!(qarr_1h["purpose"], "holdout");
    assert_eq!(qarr_1h["expected_status"], "failed");
    let invalid = Command::new(binary)
        .args(["run", "missing", "."])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("unknown validation runner"));
}

#[test]
#[ignore = "requires checksum-pinned external validation data"]
fn cli_exit_status_distinguishes_accepted_and_reviewed_failure_results() {
    let binary = env!("CARGO_BIN_EXE_phasesmith-validation");
    let accepted = Command::new(binary)
        .args(["run", "nist-srm660c-lab6-xray"])
        .arg(dataset("nist-srm660c-lab6-xray"))
        .output()
        .unwrap();
    assert!(accepted.status.success());
    let accepted_json: serde_json::Value = serde_json::from_slice(&accepted.stdout).unwrap();
    assert_eq!(accepted_json["status"], "passed");

    let tof = Command::new(binary)
        .args(["run", "powgen-lab6-tof-calibration"])
        .arg(dataset("powgen-lab6-tof-calibration"))
        .output()
        .unwrap();
    assert!(tof.status.success());
    let tof_json: serde_json::Value = serde_json::from_slice(&tof.stdout).unwrap();
    assert_eq!(tof_json["status"], "passed");

    let reviewed_failure = Command::new(binary)
        .args(["run", "iucr-qarr-1h"])
        .arg(dataset("iucr-qarr-1h"))
        .output()
        .unwrap();
    assert_eq!(reviewed_failure.status.code(), Some(1));
    let reviewed_failure_json: serde_json::Value =
        serde_json::from_slice(&reviewed_failure.stdout).unwrap();
    assert_eq!(reviewed_failure_json["status"], "failed");
}
