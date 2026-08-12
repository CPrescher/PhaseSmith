//! Native project bundle and reporting contract tests.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use phasesmith_engine::crystallography::{
    IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::profile::{ConstantWavelengthInstrument, FcjGeometry};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_model::{
    ExperimentRecord, FixedWavelengthSpectrum, HistogramRecord, PatternRecord, ProjectRecord,
    ProviderRequirement, RadiationDefinition, RadiationProbe, RecordId, StructuralPhaseRecord,
};
use phasesmith_persistence::{
    PROJECT_ARRAYS_NAME, PROJECT_FORMAT_VERSION, PROJECT_MANIFEST_NAME, PersistenceError,
    ProjectReadLimits, ProjectReportSaveOptions, ProjectSaveOptions, ProjectSummaryReport,
    load_project, project_summary_json, save_project, write_project_summary_json_with_options,
};

static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[test]
fn native_project_round_trips_every_current_domain_field() {
    let directory = temporary_path("round-trip");
    let project = project();
    let saved = save_project(&directory, &project, ProjectSaveOptions::default()).unwrap();
    assert_eq!(saved, directory);
    assert!(directory.join(PROJECT_MANIFEST_NAME).is_file());
    assert!(directory.join(PROJECT_ARRAYS_NAME).is_file());
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        project
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join(PROJECT_MANIFEST_NAME)).unwrap()).unwrap();
    assert_eq!(manifest["format_version"], PROJECT_FORMAT_VERSION);
    assert_eq!(
        manifest["project"]["histograms"].as_array().unwrap().len(),
        2
    );
    assert!(manifest["arrays"].as_object().unwrap().len() >= 20);
    cleanup(directory);
}

#[test]
fn overwrite_replaces_only_owned_files_and_preserves_unrelated_content() {
    let directory = temporary_path("overwrite");
    let mut first = project();
    save_project(&directory, &first, ProjectSaveOptions::default()).unwrap();
    fs::write(directory.join("notes.txt"), "caller-owned").unwrap();
    assert!(matches!(
        save_project(&directory, &first, ProjectSaveOptions::default()),
        Err(PersistenceError::InvalidDestination { .. })
    ));

    first.revision = 18;
    save_project(&directory, &first, ProjectSaveOptions { overwrite: true }).unwrap();
    assert_eq!(
        fs::read_to_string(directory.join("notes.txt")).unwrap(),
        "caller-owned"
    );
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default())
            .unwrap()
            .revision,
        18
    );
    cleanup(directory);
}

#[test]
fn load_recovers_the_previous_pair_after_an_interrupted_overwrite() {
    let directory = temporary_path("overwrite-recovery");
    let expected = project();
    save_project(&directory, &expected, ProjectSaveOptions::default()).unwrap();
    fs::rename(
        directory.join(PROJECT_MANIFEST_NAME),
        directory.join(".manifest.json.phasesmith-backup"),
    )
    .unwrap();
    fs::rename(
        directory.join(PROJECT_ARRAYS_NAME),
        directory.join(".arrays.npz.phasesmith-backup"),
    )
    .unwrap();
    fs::write(
        directory.join(PROJECT_ARRAYS_NAME),
        b"incomplete replacement",
    )
    .unwrap();

    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        expected
    );
    assert!(!directory.join(".manifest.json.phasesmith-backup").exists());
    assert!(!directory.join(".arrays.npz.phasesmith-backup").exists());
    cleanup(directory);
}

#[test]
fn hashes_unknown_fields_and_resource_limits_fail_before_domain_use() {
    let directory = temporary_path("validation");
    save_project(&directory, &project(), ProjectSaveOptions::default()).unwrap();

    let archive_path = directory.join(PROJECT_ARRAYS_NAME);
    let mut archive = fs::read(&archive_path).unwrap();
    let last = archive.last_mut().unwrap();
    *last ^= 0x01;
    fs::write(&archive_path, archive).unwrap();
    assert!(matches!(
        load_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::InvalidArchive { .. })
    ));

    save_project(
        &directory,
        &project(),
        ProjectSaveOptions { overwrite: true },
    )
    .unwrap();
    let manifest_path = directory.join(PROJECT_MANIFEST_NAME);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["unexpected"] = serde_json::json!(true);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        load_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::Json(_))
    ));

    save_project(
        &directory,
        &project(),
        ProjectSaveOptions { overwrite: true },
    )
    .unwrap();
    let limits = ProjectReadLimits {
        max_histograms: 1,
        ..ProjectReadLimits::default()
    };
    assert!(matches!(
        load_project(&directory, limits),
        Err(PersistenceError::LimitExceeded { .. })
    ));

    let limits = ProjectReadLimits {
        max_manifest_bytes: 1,
        ..ProjectReadLimits::default()
    };
    assert!(matches!(
        load_project(&directory, limits),
        Err(PersistenceError::LimitExceeded { .. })
    ));

    let limits = ProjectReadLimits {
        max_archive_bytes: 1,
        ..ProjectReadLimits::default()
    };
    assert!(matches!(
        load_project(&directory, limits),
        Err(PersistenceError::LimitExceeded { .. })
    ));
    cleanup(directory);
}

#[test]
fn project_summary_is_versioned_deterministic_and_array_free() {
    let project = project();
    let first = project_summary_json(&project).unwrap();
    let second = project_summary_json(&project).unwrap();
    assert_eq!(first, second);
    assert!(!first.contains("observed_y"));
    let report = ProjectSummaryReport::from_project(&project).unwrap();
    assert_eq!(report.format_version, PROJECT_FORMAT_VERSION);
    assert_eq!(report.histogram_count, 2);
    assert_eq!(report.phase_count, 2);
    assert_eq!(report.total_sample_count, 8);
    assert_eq!(report.histograms[0].phase_ids, ["alpha"]);
    assert_eq!(report.phases[0].required_providers, ["example.texture@2"]);
}

#[test]
fn project_summary_file_export_has_an_explicit_overwrite_policy() {
    let path = temporary_path("summary.json");
    let project = project();
    write_project_summary_json_with_options(&project, &path, ProjectReportSaveOptions::default())
        .unwrap();
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        project_summary_json(&project).unwrap()
    );
    fs::write(&path, "caller-owned").unwrap();
    assert!(matches!(
        write_project_summary_json_with_options(
            &project,
            &path,
            ProjectReportSaveOptions::default(),
        ),
        Err(PersistenceError::InvalidDestination { .. })
    ));
    assert_eq!(fs::read_to_string(&path).unwrap(), "caller-owned");
    write_project_summary_json_with_options(
        &project,
        &path,
        ProjectReportSaveOptions { overwrite: true },
    )
    .unwrap();
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        project_summary_json(&project).unwrap()
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn native_manifest_schema_is_valid_json_and_tracks_the_wire_version() {
    let schema_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/native-project-v4.schema.json");
    let schema: serde_json::Value =
        serde_json::from_slice(&fs::read(schema_path).unwrap()).unwrap();
    assert_eq!(schema["properties"]["format_version"]["const"], 4);
    assert_eq!(
        schema["properties"]["format_version"]["const"],
        PROJECT_FORMAT_VERSION
    );
    assert_eq!(schema["additionalProperties"], false);
}

#[test]
fn old_projects_migrate_but_each_version_requires_its_analysis_fields() {
    let directory = temporary_path("native-version-migration");
    let expected = project();
    save_project(&directory, &expected, ProjectSaveOptions::default()).unwrap();
    let manifest_path = directory.join(PROJECT_MANIFEST_NAME);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["format_version"] = serde_json::json!(1);
    manifest
        .as_object_mut()
        .unwrap()
        .remove("rietveld_analyses");
    manifest
        .as_object_mut()
        .unwrap()
        .remove("tof_lebail_analyses");
    manifest
        .as_object_mut()
        .unwrap()
        .remove("tof_multibank_geometry_analyses");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        expected
    );

    manifest["format_version"] = serde_json::json!(2);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        load_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::InvalidRecord { .. })
    ));

    manifest["rietveld_analyses"] = serde_json::json!([]);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        expected
    );

    manifest["format_version"] = serde_json::json!(3);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        load_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::InvalidRecord { .. })
    ));

    manifest["tof_lebail_analyses"] = serde_json::json!([]);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        expected
    );

    manifest["format_version"] = serde_json::json!(4);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        load_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::InvalidRecord { .. })
    ));

    manifest["tof_multibank_geometry_analyses"] = serde_json::json!([]);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        expected
    );

    manifest["format_version"] = serde_json::json!(PROJECT_FORMAT_VERSION + 1);
    manifest["future_field"] = serde_json::json!({"shape": "unknown"});
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        load_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::UnsupportedVersion { version })
            if version == PROJECT_FORMAT_VERSION + 1
    ));
    cleanup(directory);
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn numpy_reads_and_rewrites_native_archives_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let directory = temporary_path("numpy-cross-interface");
    let project = project();
    save_project(&directory, &project, ProjectSaveOptions::default()).unwrap();
    let script = r#"
import hashlib
import json
import pathlib
import sys

import numpy as np

directory = pathlib.Path(sys.argv[1])
archive_path = directory / "arrays.npz"
manifest_path = directory / "manifest.json"
with np.load(archive_path, allow_pickle=False) as archive:
    arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}

assert arrays["histogram.xray-bank.x_deg"].dtype == np.dtype("<f8")
assert arrays["histogram.xray-bank.x_deg"].shape == (4,)
assert arrays["histogram.xray-bank.mask"].dtype == np.dtype("bool")
assert arrays["phase.alpha.hkl"].dtype == np.dtype("<i4")
assert arrays["phase.alpha.hkl"].shape == (2, 3)
assert arrays["phase.alpha.multiplicity"].dtype == np.dtype("<u8")
np.savez_compressed(archive_path, **arrays)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["archive"]["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .arg(&directory)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "NumPy interoperability script failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        project
    );
    cleanup(directory);
}

fn project() -> ProjectRecord {
    ProjectRecord {
        project_id: id("project-1"),
        revision: 17,
        name: "Native persistence".to_owned(),
        histograms: vec![
            histogram("xray-bank", RadiationProbe::Xray, &["alpha"]),
            histogram("neutron-bank", RadiationProbe::Neutron, &["beta"]),
        ],
        tof_histograms: Vec::new(),
        phases: vec![
            phase(
                "alpha",
                BuiltInScatteringModel::XrayNonResonant,
                IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                    wavelength_angstrom: 1.5406,
                    polarization: 0.73,
                },
                vec![ProviderRequirement::new("example.texture", "2").unwrap()],
                true,
            ),
            phase(
                "beta",
                BuiltInScatteringModel::NeutronNuclear,
                IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
                    wavelength_angstrom: 1.8,
                },
                Vec::new(),
                false,
            ),
        ],
        metadata: BTreeMap::from([
            ("operator".to_owned(), "Ada".to_owned()),
            ("sample".to_owned(), "reference".to_owned()),
        ]),
    }
}

fn histogram(id_value: &str, probe: RadiationProbe, phase_ids: &[&str]) -> HistogramRecord {
    let wavelength = match probe {
        RadiationProbe::Xray => 1.5406,
        RadiationProbe::Neutron => 1.8,
    };
    let experiment = ExperimentRecord::new(
        ConstantWavelengthInstrument {
            wavelength_angstrom: wavelength,
            u_deg2: 1.0e-4,
            v_deg2: -2.0e-5,
            w_deg2: 3.0e-4,
            x_deg: 4.0e-3,
            y_deg: 5.0e-3,
        },
        if probe == RadiationProbe::Xray {
            RadiationDefinition::FixedSpectrum {
                probe,
                spectrum: FixedWavelengthSpectrum::new(vec![wavelength, 1.54439], vec![1.0, 0.5])
                    .unwrap(),
            }
        } else {
            RadiationDefinition::Monochromatic {
                probe,
                wavelength_angstrom: wavelength,
            }
        },
        (probe == RadiationProbe::Xray).then_some(FcjGeometry {
            sample_over_radius: 0.01,
            detector_over_radius: 0.02,
        }),
        if probe == RadiationProbe::Xray {
            MonochromaticPositionCorrection {
                zero_shift_deg: 0.03,
                bragg_brentano_mm: Some((0.02, 150.0)),
                debye_scherrer_micrometre: None,
            }
        } else {
            MonochromaticPositionCorrection {
                zero_shift_deg: -0.01,
                bragg_brentano_mm: None,
                debye_scherrer_micrometre: Some((2.0, -3.0, 200.0)),
            }
        },
    )
    .unwrap();
    HistogramRecord {
        histogram_id: id(id_value),
        name: id_value.to_owned(),
        pattern: PatternRecord::new(
            vec![10.0, 10.1, 10.2, 10.3],
            Some(vec![2.0, 3.0, 4.0, 5.0]),
            Some(vec![1.0, 1.1, 1.2, 1.3]),
            Some(vec![true, true, false, true]),
            Some(vec![0.1, 0.2, 0.3, 0.4]),
        )
        .unwrap(),
        experiment,
        phase_ids: phase_ids.iter().map(|value| id(value)).collect(),
    }
}

fn phase(
    phase_id: &str,
    scattering_model: BuiltInScatteringModel,
    correction_model: IntegratedIntensityCorrectionModel,
    required_providers: Vec<ProviderRequirement>,
    with_offsets: bool,
) -> StructuralPhaseRecord {
    StructuralPhaseRecord {
        phase_id: id(phase_id),
        name: format!("Phase {phase_id}"),
        definition: StructuralPhaseDefinition {
            cell: UnitCell {
                a_angstrom: 5.0,
                b_angstrom: 5.1,
                c_angstrom: 5.2,
                alpha_deg: 90.0,
                beta_deg: 91.0,
                gamma_deg: 90.0,
            },
            space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap(),
            hkl: vec![[1, 0, 0], [1, 1, 0]],
            multiplicity: vec![2, 4],
            fractional_xyz: vec![[0.1, 0.2, 0.3]],
            occupancy: vec![0.9],
            u_iso_angstrom2: vec![0.01],
            anisotropic_mask: vec![true],
            u_aniso_cif_angstrom2: vec![[0.01, 0.02, 0.03, 0.0, 0.0, 0.001]],
            scattering_species: vec!["Si".to_owned()],
            scattering_real_offset: with_offsets.then_some(vec![0.2]).unwrap_or_default(),
            scattering_imag_offset: with_offsets.then_some(vec![0.03]).unwrap_or_default(),
            scale: 1.25,
            coordinate_tolerance: 1.0e-9,
            scattering_model,
            correction_model,
        },
        required_providers,
    }
}

fn id(value: &str) -> RecordId {
    RecordId::new(value).unwrap()
}

fn temporary_path(label: &str) -> PathBuf {
    let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "phasesmith-persistence-{label}-{}-{sequence}",
        std::process::id()
    ))
}

fn cleanup(path: PathBuf) {
    if path.exists() {
        fs::remove_dir_all(path).unwrap();
    }
}
