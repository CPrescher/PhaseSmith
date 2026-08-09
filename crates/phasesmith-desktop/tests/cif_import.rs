//! Native CIF phase-import tests for the desktop adapter.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use phasesmith_crystallography::IntegratedIntensityCorrectionModel;
use phasesmith_desktop::{
    AnalysisSelectionInput, AnalysisSolverInput, CifPhaseImportRequest, CreateAnalysisRequest,
    DesktopErrorCode, DesktopExperimentInput, DesktopIntensityCorrectionInput,
    DesktopPositionCorrectionInput, DesktopProjectStore, DesktopRadiationProbe, PowderFormatInput,
    PowderHistogramImportRequest,
};
use phasesmith_engine::BuiltInScatteringModel;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temporary_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "phasesmith-desktop-cif-{label}-{}-{}",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn creates_an_analysis_and_later_cif_imports_extend_it_atomically() {
    let store = DesktopProjectStore::new();
    let (powder, revision) = prepare_histogram(&store, DesktopRadiationProbe::Xray);
    let first_cif = temporary_path("analysis-first.cif");
    let second_cif = temporary_path("analysis-second.cif");
    std::fs::write(&first_cif, cif_text()).unwrap();
    std::fs::write(&second_cif, cif_text()).unwrap();

    let first = store
        .import_cif_phase(revision, request(first_cif.clone()))
        .unwrap();
    let created = store
        .create_analysis(
            first.revision,
            &CreateAnalysisRequest {
                histogram_id: "histogram-1".to_owned(),
                selection: AnalysisSelectionInput::default(),
                solver: AnalysisSolverInput::default(),
                lattice_relative_length: 0.05,
                lattice_angle_delta_deg: 5.0,
            },
        )
        .unwrap();
    assert_eq!(created.phase_count, 1);
    assert_eq!(created.parameter_count, 1);

    let mut second_request = request(second_cif.clone());
    second_request.phase_id = "phase-si-2".to_owned();
    let second = store
        .import_cif_phase(created.revision, second_request)
        .unwrap();
    let snapshot = store.snapshot().unwrap();
    assert_eq!(snapshot.revision(), second.revision);
    assert_eq!(snapshot.state().analyses[0].input.phases.len(), 2);
    assert_eq!(snapshot.state().analyses[0].lattice_bounds.len(), 2);
    assert!(snapshot.state().analyses[0].checkpoint.is_none());

    std::fs::remove_file(powder).unwrap();
    std::fs::remove_file(first_cif).unwrap();
    std::fs::remove_file(second_cif).unwrap();
}

fn prepare_histogram(store: &DesktopProjectStore, probe: DesktopRadiationProbe) -> (PathBuf, u64) {
    store.create_project("project", "CIF project").unwrap();
    let powder = temporary_path("pattern.xy");
    std::fs::write(&powder, "10 100 2\n50 80 2\n90 60 2\n").unwrap();
    let response = store
        .import_powder_histogram(
            0,
            PowderHistogramImportRequest {
                path: powder.clone(),
                histogram_id: "histogram-1".to_owned(),
                name: "Observed".to_owned(),
                format: PowderFormatInput::Columns,
                bank: 1,
                experiment: DesktopExperimentInput {
                    probe,
                    wavelength_angstrom: 1.5406,
                    u_deg2: 0.01,
                    v_deg2: 0.0,
                    w_deg2: 0.01,
                    x_deg: 0.01,
                    y_deg: 0.0,
                    axial_geometry: None,
                    position_correction: DesktopPositionCorrectionInput::default(),
                },
                phase_ids: Vec::new(),
            },
        )
        .unwrap();
    (powder, response.revision)
}

fn cif_text() -> &'static str {
    "data_silicon\n\
     _chemical_name_common 'Silicon test'\n\
     _cell_length_a 5.43\n\
     _cell_length_b 5.43\n\
     _cell_length_c 5.43\n\
     _cell_angle_alpha 90\n\
     _cell_angle_beta 90\n\
     _cell_angle_gamma 90\n\
     _symmetry_equiv_pos_as_xyz 'x,y,z'\n\
     loop_\n\
     _atom_site_label\n\
     _atom_site_type_symbol\n\
     _atom_site_fract_x\n\
     _atom_site_fract_y\n\
     _atom_site_fract_z\n\
     _atom_site_occupancy\n\
     _atom_site_U_iso_or_equiv\n\
     Si1 Si 0 0 0 1 0.005\n"
}

fn request(path: PathBuf) -> CifPhaseImportRequest {
    CifPhaseImportRequest {
        path,
        histogram_id: "histogram-1".to_owned(),
        phase_id: "phase-si".to_owned(),
        name: None,
        block: None,
        strict: true,
        scale: 1.0,
        coordinate_tolerance: 1.0e-10,
        merge_friedel: true,
        max_candidates: 1_000_000,
        correction: DesktopIntensityCorrectionInput::Neutral,
    }
}

#[test]
fn imports_cif_generates_reflections_and_attaches_the_phase_atomically() {
    let store = DesktopProjectStore::new();
    let (powder, revision) = prepare_histogram(&store, DesktopRadiationProbe::Xray);
    let cif = temporary_path("silicon.cif");
    std::fs::write(&cif, cif_text()).unwrap();

    let response = store
        .import_cif_phase(revision, request(cif.clone()))
        .unwrap();

    assert_eq!(response.revision, 2);
    assert_eq!(response.phase_id, "phase-si");
    assert_eq!(response.histogram_id, "histogram-1");
    assert_eq!(response.site_count, 1);
    assert!(response.reflection_count > 0);
    assert_eq!(response.selected_block, "silicon");
    assert!(response.diagnostics.is_empty());
    let snapshot = store.snapshot().unwrap();
    let phase = &snapshot.state().project.phases[0];
    assert_eq!(phase.name, "Silicon test");
    assert_eq!(phase.definition.scattering_species, ["Si"]);
    assert_eq!(
        phase.definition.scattering_model,
        BuiltInScatteringModel::XrayNonResonant
    );
    let attached = &snapshot.state().project.histograms[0].phase_ids;
    assert_eq!(attached.len(), 1);
    assert_eq!(attached[0], phase.phase_id);
    assert_eq!(
        snapshot.state().project.metadata["phase.phase-si.cif_block"],
        "silicon"
    );
    std::fs::remove_file(powder).unwrap();
    std::fs::remove_file(cif).unwrap();
}

#[test]
fn stale_invalid_and_missing_target_imports_preserve_the_snapshot() {
    let store = DesktopProjectStore::new();
    let (powder, revision) = prepare_histogram(&store, DesktopRadiationProbe::Xray);
    let cif = temporary_path("invalid.cif");
    std::fs::write(&cif, "data_broken\n_cell_length_a nope\n").unwrap();

    let stale = store
        .import_cif_phase(revision + 1, request(cif.clone()))
        .unwrap_err();
    assert_eq!(stale.code, DesktopErrorCode::RevisionConflict);
    let invalid = store
        .import_cif_phase(revision, request(cif.clone()))
        .unwrap_err();
    assert_eq!(invalid.code, DesktopErrorCode::Import);
    let mut missing = request(cif.clone());
    missing.histogram_id = "missing".to_owned();
    let missing = store.import_cif_phase(revision, missing).unwrap_err();
    assert_eq!(missing.code, DesktopErrorCode::Import);
    assert_eq!(store.snapshot().unwrap().revision(), revision);
    assert!(store.snapshot().unwrap().state().project.phases.is_empty());
    std::fs::remove_file(powder).unwrap();
    std::fs::remove_file(cif).unwrap();
}

#[test]
fn correction_probe_contract_and_command_json_are_explicit() {
    let store = DesktopProjectStore::new();
    let (powder, revision) = prepare_histogram(&store, DesktopRadiationProbe::Xray);
    let cif = temporary_path("correction.cif");
    std::fs::write(&cif, cif_text()).unwrap();
    let mut incompatible = request(cif.clone());
    incompatible.correction = DesktopIntensityCorrectionInput::ConstantWavelengthNeutronLorentz;
    let error = store.import_cif_phase(revision, incompatible).unwrap_err();
    assert_eq!(error.code, DesktopErrorCode::Import);
    assert!(store.snapshot().unwrap().state().project.phases.is_empty());

    let value = serde_json::to_value(request(cif.clone())).unwrap();
    assert_eq!(value["correction"]["kind"], "neutral");
    let restored: CifPhaseImportRequest = serde_json::from_value(value).unwrap();
    assert_eq!(restored.phase_id, "phase-si");
    std::fs::remove_file(powder).unwrap();
    std::fs::remove_file(cif).unwrap();
}

#[test]
fn neutron_import_preserves_isotope_identity_and_probe_specific_correction() {
    let store = DesktopProjectStore::new();
    let (powder, revision) = prepare_histogram(&store, DesktopRadiationProbe::Neutron);
    let cif = temporary_path("deuterium.cif");
    let deuterium = cif_text().replace("Si1 Si", "D1 2H");
    std::fs::write(&cif, deuterium).unwrap();
    let mut request = request(cif.clone());
    request.phase_id = "phase-d".to_owned();
    request.correction = DesktopIntensityCorrectionInput::ConstantWavelengthNeutronLorentz;

    store.import_cif_phase(revision, request).unwrap();

    let snapshot = store.snapshot().unwrap();
    let definition = &snapshot.state().project.phases[0].definition;
    assert_eq!(definition.scattering_species, ["H-2"]);
    assert_eq!(
        definition.scattering_model,
        BuiltInScatteringModel::NeutronNuclear
    );
    assert_eq!(
        definition.correction_model,
        IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
            wavelength_angstrom: 1.5406,
        }
    );
    std::fs::remove_file(powder).unwrap();
    std::fs::remove_file(cif).unwrap();
}
