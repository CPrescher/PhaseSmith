//! Checksum-pinned ANSTO Echidna neutron-pattern smoke regression.

use std::path::PathBuf;

use phasesmith_validation::{ValidationStatus, run_echidna_lab6_validation};

fn dataset() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("validation/data/ansto-echidna-lab6-cw-neutron")
}

#[test]
#[ignore = "requires checksum-pinned external Echidna pattern"]
fn native_neutron_lebail_smoke_gate_passes() {
    let report = run_echidna_lab6_validation(&dataset()).unwrap();
    assert_eq!(report.status, ValidationStatus::Passed);
    assert_eq!(report.sample_count, 2_111);
    assert_eq!(report.reflection_count, Some(13));
    let correlation = report
        .checks
        .iter()
        .find(|check| check.check_id == "profile_correlation")
        .and_then(|check| check.measured)
        .unwrap();
    assert!(correlation >= 0.90);
}
