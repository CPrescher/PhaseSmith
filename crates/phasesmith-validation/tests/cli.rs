//! Python-free validation CLI contracts.

use std::process::Command;

#[test]
fn datasets_are_stable_and_unknown_runners_fail() {
    let binary = env!("CARGO_BIN_EXE_phasesmith-validation");
    let listed = Command::new(binary).arg("datasets").output().unwrap();
    assert!(listed.status.success());
    assert_eq!(
        String::from_utf8(listed.stdout).unwrap(),
        "aps-sucrose-11bmb\ngsasii-pbso4-cw\niucr-qarr-1g\n"
    );
    let invalid = Command::new(binary)
        .args(["run", "missing", "."])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("unknown validation runner"));
}
