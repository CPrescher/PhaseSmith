//! Checksum-pinned NIST SRM 660c archive integrity regression.

use std::path::PathBuf;

use phasesmith_validation::{ValidationStatus, run_nist_srm660c_validation};

fn dataset() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("validation/data/nist-srm660c-lab6-xray")
}

#[test]
#[ignore = "requires checksum-pinned external NIST SRM 660c archive"]
fn official_archive_and_reference_profiles_pass_the_integrity_gates() {
    let report = run_nist_srm660c_validation(&dataset()).unwrap();
    assert_eq!(report.status, ValidationStatus::Passed);
    assert_eq!(report.sample_count, 106_640);
    assert_eq!(report.reflection_count, Some(24));
    assert!(
        report
            .checks
            .iter()
            .all(|check| check.status == ValidationStatus::Passed)
    );
}
