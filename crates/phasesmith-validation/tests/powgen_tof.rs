//! Checksum-pinned POWGEN TOF complete-workflow regression.

use std::path::PathBuf;

use phasesmith_validation::{ValidationStatus, run_powgen_tof_validation};

fn dataset() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("validation/data/powgen-lab6-tof-calibration")
}

#[test]
#[ignore = "requires checksum-pinned external POWGEN tutorial files"]
fn tof_pattern_passes_native_profile_and_lebail_acceptance() {
    let report = run_powgen_tof_validation(&dataset()).unwrap();
    assert_eq!(report.status, ValidationStatus::Passed);
    assert_eq!(report.sample_count, 6_824);
    assert_eq!(report.reflection_count, Some(330));
    assert!(
        report
            .checks
            .iter()
            .all(|check| check.status == ValidationStatus::Passed)
    );
}
