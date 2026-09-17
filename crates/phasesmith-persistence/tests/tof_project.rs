//! Native TOF histogram and resumable analysis state coverage.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use phasesmith_core::{
    OwnedCwContributions, TofBankGeometry, TofInstrument, TofInstrumentParameter,
};
use phasesmith_engine::crystallography::{
    IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::{BuiltInScatteringModel, StructuralPhaseDefinition};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{
    ProjectRecord, RecordId, StructuralPhaseRecord, TofExperimentRecord, TofHistogramRecord,
    TofPatternRecord,
};
use phasesmith_persistence::{
    PROJECT_FORMAT_VERSION, PROJECT_MANIFEST_NAME, PersistenceError, ProjectReadLimits,
    ProjectSaveOptions, ProjectSummaryReport, load_project, load_rietveld_project,
    load_structural_tof_multibank_project, load_tof_lebail_project,
    load_tof_multibank_geometry_project, save_structural_tof_multibank_project,
    save_tof_lebail_project, save_tof_multibank_geometry_project,
};
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, ParameterBounds,
    PreparedStructuralTofMultiBankObjective, RefinementLimits, RietveldPhase,
    RietveldStructuralSelection, StructuralTofBank, StructuralTofMultiBankAnalysis,
    StructuralTofMultiBankInput, StructuralTofMultiBankLayout, StructuralTofMultiBankProjectState,
    StructuralTofMultiBankRefinementOptions, TofBankInstrumentModel, TofChebyshevBackground,
    TofInstrumentParameterBound, TofLeBailAnalysis, TofLeBailBank, TofLeBailInput,
    TofLeBailOptions, TofLeBailPhase, TofLeBailProjectState, TofMultiBankGeometryAnalysis,
    TofMultiBankGeometryInput, TofMultiBankGeometryOptions, TofMultiBankGeometryProjectState,
    TofMultiBankInput, TofMultiBankLatticeInput, TofMultiBankProjectError, TofProjectError,
    TofSharedLatticePhase, calculate_tof_lebail_pattern, refine_structural_tof_multibank,
    refine_tof_lebail, refine_tof_multibank_geometry,
};

static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[test]
fn format_three_round_trips_tof_histogram_analysis_and_checkpoint() {
    let state = state();
    let directory = temporary_path("tof-round-trip");
    save_tof_lebail_project(&directory, &state, ProjectSaveOptions::default()).unwrap();

    let restored = load_tof_lebail_project(&directory, ProjectReadLimits::default()).unwrap();
    assert_eq!(restored, state);
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        state.project
    );
    assert!(
        load_rietveld_project(&directory, ProjectReadLimits::default())
            .unwrap()
            .analyses
            .is_empty()
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join(PROJECT_MANIFEST_NAME)).unwrap()).unwrap();
    assert_eq!(manifest["format_version"], PROJECT_FORMAT_VERSION);
    assert_eq!(
        manifest["project"]["tof_histograms"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(manifest["tof_lebail_analyses"].as_array().unwrap().len(), 1);
    assert!(
        manifest["arrays"]
            .as_object()
            .unwrap()
            .keys()
            .any(|name| name.contains("checkpoint"))
    );

    let report = ProjectSummaryReport::from_project(&state.project).unwrap();
    assert_eq!(report.histogram_count, 1);
    assert_eq!(report.total_sample_count, 401);
    assert_eq!(report.histograms[0].coordinate_kind, "tof_us");
    assert_eq!(report.histograms[0].probe, "neutron");

    let mut legacy = manifest;
    legacy.as_object_mut().unwrap().remove("pawley_analyses");
    legacy["format_version"] = serde_json::json!(3);
    legacy
        .as_object_mut()
        .unwrap()
        .remove("tof_multibank_geometry_analyses");
    legacy
        .as_object_mut()
        .unwrap()
        .remove("structural_tof_multibank_analyses");
    fs::write(
        directory.join(PROJECT_MANIFEST_NAME),
        serde_json::to_vec_pretty(&legacy).unwrap(),
    )
    .unwrap();
    assert_eq!(
        load_tof_lebail_project(&directory, ProjectReadLimits::default()).unwrap(),
        state
    );
    cleanup(directory);
}

#[test]
fn tof_project_rejects_duplicate_ownership_and_histogram_drift() {
    let mut duplicate = state();
    duplicate.analyses.push(duplicate.analyses[0].clone());
    assert!(matches!(
        duplicate.validate(),
        Err(TofProjectError::DuplicateAnalysis { .. })
    ));

    let mut drifted = state();
    drifted.analyses[0].input.pattern.background_y[0] = 1.0;
    assert!(matches!(
        drifted.validate(),
        Err(TofProjectError::HistogramStateMismatch { .. })
    ));
}

#[test]
fn format_four_round_trips_joint_geometry_and_complete_checkpoint() {
    let state = multibank_state();
    let directory = temporary_path("tof-multibank-round-trip");
    save_tof_multibank_geometry_project(&directory, &state, ProjectSaveOptions::default()).unwrap();

    let restored =
        load_tof_multibank_geometry_project(&directory, ProjectReadLimits::default()).unwrap();
    assert_eq!(restored, state);
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        state.project
    );
    assert!(
        load_tof_lebail_project(&directory, ProjectReadLimits::default())
            .unwrap()
            .analyses
            .is_empty()
    );
    assert!(
        load_rietveld_project(&directory, ProjectReadLimits::default())
            .unwrap()
            .analyses
            .is_empty()
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join(PROJECT_MANIFEST_NAME)).unwrap()).unwrap();
    assert_eq!(manifest["format_version"], PROJECT_FORMAT_VERSION);
    assert_eq!(
        manifest["tof_multibank_geometry_analyses"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(
        manifest["arrays"]
            .as_object()
            .unwrap()
            .keys()
            .any(|name| name.contains("tof_multibank") && name.contains("history"))
    );
    cleanup(directory);
}

#[test]
fn joint_project_rejects_duplicate_histogram_ownership_and_drift() {
    let mut duplicate = multibank_state();
    let mut second = duplicate.analyses[0].clone();
    second.analysis_id = id("joint-geometry-copy");
    duplicate.analyses.push(second);
    assert!(matches!(
        duplicate.validate(),
        Err(TofMultiBankProjectError::DuplicateHistogramOwnership { .. })
    ));

    let mut drifted = multibank_state();
    drifted.analyses[0].input.lattice.multibank.banks[0]
        .input
        .pattern
        .background_y[0] = 1.0;
    assert!(matches!(
        drifted.validate(),
        Err(TofMultiBankProjectError::HistogramStateMismatch { .. })
    ));
}

#[test]
fn format_five_round_trips_structural_tof_and_complete_checkpoint() {
    let state = structural_tof_state(2);
    let directory = temporary_path("structural-tof-round-trip");
    save_structural_tof_multibank_project(&directory, &state, ProjectSaveOptions::default())
        .unwrap();

    let restored =
        load_structural_tof_multibank_project(&directory, ProjectReadLimits::default()).unwrap();
    assert_eq!(restored, state);
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        state.project
    );
    assert!(
        load_tof_multibank_geometry_project(&directory, ProjectReadLimits::default())
            .unwrap()
            .analyses
            .is_empty()
    );

    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join(PROJECT_MANIFEST_NAME)).unwrap()).unwrap();
    assert_eq!(manifest["format_version"], 6);
    assert_eq!(
        manifest["structural_tof_multibank_analyses"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(
        manifest["structural_tof_multibank_analyses"][0]["checkpoint"]["history"]
            .as_array()
            .is_some_and(|history| !history.is_empty())
    );
    assert!(
        manifest["structural_tof_multibank_analyses"][0]["banks"][0]["scale_bounds"][0].is_null()
    );

    let restrictive = ProjectReadLimits {
        max_histograms: 1,
        ..ProjectReadLimits::default()
    };
    assert!(matches!(
        load_structural_tof_multibank_project(&directory, restrictive),
        Err(PersistenceError::LimitExceeded { .. })
    ));

    manifest["structural_tof_multibank_analyses"][0]["checkpoint"]["parameters"][0]["name"] =
        serde_json::json!("corrupt_parameter_identity");
    fs::write(
        directory.join(PROJECT_MANIFEST_NAME),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        load_structural_tof_multibank_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::InvalidRecord { .. })
    ));
    cleanup(directory);
}

#[test]
fn format_five_round_trips_a_single_bank_structural_tof_analysis() {
    let state = structural_tof_state(1);
    let directory = temporary_path("structural-tof-single-bank-round-trip");
    save_structural_tof_multibank_project(&directory, &state, ProjectSaveOptions::default())
        .unwrap();

    let restored =
        load_structural_tof_multibank_project(&directory, ProjectReadLimits::default()).unwrap();
    assert_eq!(restored, state);
    assert_eq!(restored.analyses[0].input.banks.len(), 1);
    assert_eq!(restored.project.tof_histograms.len(), 1);
    cleanup(directory);
}

fn state() -> TofLeBailProjectState {
    let options = TofLeBailOptions::new(
        2,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2).unwrap(),
    )
    .unwrap()
    .with_redistribution_uncertainty(false);
    let tof_us = (0..401)
        .map(|index| 3_000.0 + 10.0 * f64::from(index))
        .collect::<Vec<_>>();
    let blank = TofPatternRecord::new(
        tof_us.clone(),
        Some(vec![0.0; tof_us.len()]),
        Some(vec![1.0; tof_us.len()]),
        None,
        None,
    )
    .unwrap();
    let truth = TofLeBailInput::new(blank, instrument(), vec![tof_phase(vec![80.0, 120.0])])
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(id("background"), vec![2.0, 0.2], [3_000.0, 7_000.0])
                .unwrap(),
        )
        .unwrap();
    let observed = calculate_tof_lebail_pattern(&truth, &options).unwrap().y;
    let pattern = TofPatternRecord::new(
        tof_us,
        Some(observed),
        Some(vec![1.0; 401]),
        Some((0..401).map(|index| index != 0).collect()),
        None,
    )
    .unwrap();
    let input = TofLeBailInput::new(pattern.clone(), instrument(), vec![tof_phase(vec![1.0; 2])])
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(id("background"), vec![1.0, 0.0], [3_000.0, 7_000.0])
                .unwrap(),
        )
        .unwrap();
    let checkpoint = refine_tof_lebail(&input, &options).unwrap().checkpoint;
    let project = ProjectRecord {
        project_id: id("tof-project"),
        revision: 3,
        name: "TOF project".to_owned(),
        histograms: Vec::new(),
        tof_histograms: vec![TofHistogramRecord {
            histogram_id: id("bank-1"),
            name: "Bank 1".to_owned(),
            pattern,
            experiment: TofExperimentRecord::new(instrument()).unwrap(),
            phase_ids: vec![id("alpha")],
        }],
        phases: vec![structural_phase()],
        metadata: BTreeMap::from([("coordinate".to_owned(), "microseconds".to_owned())]),
    };
    TofLeBailProjectState {
        project,
        analyses: vec![TofLeBailAnalysis {
            histogram_id: id("bank-1"),
            input,
            options,
            checkpoint: Some(checkpoint),
        }],
    }
}

#[allow(clippy::too_many_lines)]
fn structural_tof_state(bank_count: usize) -> StructuralTofMultiBankProjectState {
    assert!((1..=2).contains(&bank_count));
    let definition = StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom: 4.0,
            b_angstrom: 4.0,
            c_angstrom: 4.0,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        },
        space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap(),
        hkl: vec![[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]],
        multiplicity: vec![6, 12, 8, 6],
        fractional_xyz: vec![[0.0, 0.0, 0.0]],
        occupancy: vec![1.0],
        u_iso_angstrom2: vec![0.01],
        anisotropic_mask: vec![false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]],
        scattering_species: vec!["Ni".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::NeutronNuclear,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    let phase = RietveldPhase::new_with_site_ids(
        id("structural-phase"),
        "Structural phase",
        vec![id("nickel-site")],
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap();
    let mut banks = Vec::new();
    for (bank_id, angle, zero_us, scale) in [
        ("structural-bank-1", 88.0, 1.0, 1.1),
        ("structural-bank-2", 130.0, -0.5, 0.9),
    ]
    .into_iter()
    .take(bank_count)
    {
        let tof_us = (0..1_001)
            .map(|index| 5_000.0 + 20.0 * f64::from(index))
            .collect::<Vec<_>>();
        let pattern = TofPatternRecord::new(
            tof_us.clone(),
            Some(vec![0.0; tof_us.len()]),
            Some(vec![1.0; tof_us.len()]),
            Some((0..tof_us.len()).map(|index| index % 41 != 0).collect()),
            Some(vec![0.1; tof_us.len()]),
        )
        .unwrap();
        banks.push(StructuralTofBank {
            bank_id: id(bank_id),
            pattern,
            instrument: TofInstrument {
                zero_us,
                ..instrument()
            },
            geometry: TofBankGeometry {
                two_theta_deg: angle,
            },
            correction_model: IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: angle,
            },
            scale,
            scale_bounds: ParameterBounds::default(),
            refine_scale: true,
            background: Some(
                TofChebyshevBackground::new(
                    id(&format!("{bank_id}-background")),
                    vec![0.1, 0.0],
                    [tof_us[0], *tof_us.last().unwrap()],
                )
                .unwrap(),
            ),
            refine_background: false,
            instrument_bounds: vec![
                TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0).unwrap(),
            ],
        });
    }
    let mut input = StructuralTofMultiBankInput {
        phase,
        structural_selection: RietveldStructuralSelection::default(),
        lattice_bounds: None,
        banks,
        support_fwhm: 20.0,
        tail_log: 20.0,
        use_uncertainty: true,
        execution: ExecutionPolicy::new(Some(1), 2).unwrap(),
    };
    let layout = StructuralTofMultiBankLayout::new(&input).unwrap();
    let truth_values = [1.3, 2.0, 0.7, -1.5];
    let truth = layout
        .apply_values(&input, &truth_values[..2 * bank_count])
        .unwrap();
    let calculated = PreparedStructuralTofMultiBankObjective::new(truth)
        .unwrap()
        .calculate()
        .unwrap();
    for (bank, calculated) in input.banks.iter_mut().zip(calculated.banks) {
        bank.pattern.observed_y = Some(calculated.y);
    }
    let options = StructuralTofMultiBankRefinementOptions::new(
        RefinementLimits::new(8, 100, None, 8).unwrap(),
        1,
        1.0e-12,
        1.0e-9,
        1.0e-3,
        10.0,
        0.3,
        1.0,
        8,
    )
    .unwrap();
    let checkpoint = refine_structural_tof_multibank(&input, options, None, None)
        .unwrap()
        .checkpoint;
    assert!(!checkpoint.history.is_empty());
    let project = ProjectRecord {
        project_id: id("structural-tof-project"),
        revision: 5,
        name: "Structural TOF project".to_owned(),
        histograms: Vec::new(),
        tof_histograms: input
            .banks
            .iter()
            .map(|bank| TofHistogramRecord {
                histogram_id: bank.bank_id.clone(),
                name: bank.bank_id.as_str().to_owned(),
                pattern: bank.pattern.clone(),
                experiment: TofExperimentRecord::new(bank.instrument).unwrap(),
                phase_ids: vec![input.phase.phase_id().clone()],
            })
            .collect(),
        phases: vec![StructuralPhaseRecord {
            phase_id: input.phase.phase_id().clone(),
            name: input.phase.name().to_owned(),
            definition,
            required_providers: Vec::new(),
        }],
        metadata: BTreeMap::new(),
    };
    StructuralTofMultiBankProjectState {
        project,
        analyses: vec![StructuralTofMultiBankAnalysis {
            analysis_id: id("structural-analysis"),
            input,
            options,
            checkpoint: Some(checkpoint),
        }],
    }
}

#[allow(clippy::too_many_lines)]
fn multibank_state() -> TofMultiBankGeometryProjectState {
    let initial_cell = UnitCell {
        a_angstrom: 4.0,
        b_angstrom: 4.0,
        c_angstrom: 4.0,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
    let parameterization = LatticeParameterization::new(group, initial_cell).unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.01, 1.0).unwrap();
    let lattice =
        TofSharedLatticePhase::new(id("shared-alpha"), parameterization, bounds, initial_cell)
            .unwrap();
    let geometry = initial_cell.geometry().unwrap();
    let d_spacing = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]
        .into_iter()
        .map(|hkl| geometry.d_spacing_and_derivatives(hkl).unwrap().0)
        .collect::<Vec<_>>();
    let options = TofMultiBankGeometryOptions::new(
        TofLeBailOptions::new(
            2,
            1.0,
            1.0e-12,
            1.0e-15,
            20.0,
            20.0,
            true,
            ExecutionPolicy::new(Some(1), 2).unwrap(),
        )
        .unwrap(),
        1.0e-10,
        0.08,
        10,
        0.5,
    )
    .unwrap();
    let bank_specs = [
        ("joint-bank-1", instrument(), 6_000.0, 21_000.0, 1_501),
        (
            "joint-bank-2",
            TofInstrument {
                zero_us: 1.3,
                difc_us_per_angstrom: 4_400.0,
                ..instrument()
            },
            5_000.0,
            18_000.0,
            1_301,
        ),
    ];
    let mut banks = Vec::new();
    let mut histograms = Vec::new();
    let mut models = Vec::new();
    for (bank_id, bank_instrument, start, end, count) in bank_specs {
        let count_usize = usize::try_from(count).unwrap();
        let step = (end - start) / f64::from(count - 1);
        let tof_us = (0..count)
            .map(|index| start + step * f64::from(index))
            .collect::<Vec<_>>();
        let phase = TofLeBailPhase::new(
            id("shared-alpha"),
            "Shared alpha",
            vec!["100".into(), "110".into(), "111".into(), "200".into()],
            vec![[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]],
            d_spacing.clone(),
            vec![80.0, 120.0, 65.0, 90.0],
            1.0,
        )
        .unwrap();
        let blank = TofPatternRecord::new(
            tof_us.clone(),
            Some(vec![0.0; count_usize]),
            Some(vec![1.0; count_usize]),
            None,
            None,
        )
        .unwrap();
        let truth = TofLeBailInput::new(blank, bank_instrument, vec![phase.clone()]).unwrap();
        let observed = calculate_tof_lebail_pattern(&truth, &options.lebail)
            .unwrap()
            .y;
        let pattern = TofPatternRecord::new(
            tof_us,
            Some(observed),
            Some(vec![1.0; count_usize]),
            Some((0..count_usize).map(|index| index % 37 != 0).collect()),
            None,
        )
        .unwrap();
        banks.push(TofLeBailBank {
            bank_id: id(bank_id),
            input: TofLeBailInput::new(pattern.clone(), bank_instrument, vec![phase]).unwrap(),
        });
        histograms.push(TofHistogramRecord {
            histogram_id: id(bank_id),
            name: bank_id.to_owned(),
            pattern,
            experiment: TofExperimentRecord::new(bank_instrument).unwrap(),
            phase_ids: vec![id("shared-alpha")],
        });
        models.push(
            TofBankInstrumentModel::new(
                id(bank_id),
                vec![
                    TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0)
                        .unwrap(),
                ],
            )
            .unwrap(),
        );
    }
    let input = TofMultiBankGeometryInput {
        lattice: TofMultiBankLatticeInput {
            multibank: TofMultiBankInput { banks },
            lattice_phases: vec![lattice],
        },
        instrument_models: models,
    };
    let checkpoint = refine_tof_multibank_geometry(&input, &options)
        .unwrap()
        .checkpoint;
    TofMultiBankGeometryProjectState {
        project: ProjectRecord {
            project_id: id("joint-tof-project"),
            revision: 4,
            name: "Joint TOF project".to_owned(),
            histograms: Vec::new(),
            tof_histograms: histograms,
            phases: vec![StructuralPhaseRecord {
                phase_id: id("shared-alpha"),
                name: "Shared alpha".to_owned(),
                definition: StructuralPhaseDefinition {
                    cell: initial_cell,
                    space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap(),
                    hkl: vec![[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]],
                    multiplicity: vec![1; 4],
                    fractional_xyz: vec![[0.0, 0.0, 0.0]],
                    occupancy: vec![1.0],
                    u_iso_angstrom2: vec![0.01],
                    anisotropic_mask: vec![false],
                    u_aniso_cif_angstrom2: vec![[0.0; 6]],
                    scattering_species: vec!["Ni".to_owned()],
                    scattering_real_offset: Vec::new(),
                    scattering_imag_offset: Vec::new(),
                    scale: 1.0,
                    coordinate_tolerance: 1.0e-10,
                    scattering_model: BuiltInScatteringModel::NeutronNuclear,
                    correction_model: IntegratedIntensityCorrectionModel::Neutral,
                },
                required_providers: Vec::new(),
            }],
            metadata: BTreeMap::new(),
        },
        analyses: vec![TofMultiBankGeometryAnalysis {
            analysis_id: id("joint-geometry"),
            input,
            options,
            checkpoint: Some(checkpoint),
        }],
    }
}

fn instrument() -> TofInstrument {
    TofInstrument {
        zero_us: -0.7,
        difc_us_per_angstrom: 5_000.0,
        difa_us_per_angstrom2: -1.5,
        difb_us_angstrom: 0.8,
        alpha_coefficient: 0.18,
        beta0_per_us: 0.04,
        beta1_angstrom4_per_us: 0.000_5,
        betaq_angstrom2_per_us: 0.001,
        sigma0_us2: 1.0,
        sigma1_us2_per_angstrom2: 12.0,
        sigma2_us2_per_angstrom4: 0.05,
        sigmaq_us2_per_angstrom: 0.2,
        x_us_per_angstrom: 0.3,
        y_us_per_angstrom2: 0.05,
        z_us: 0.4,
    }
}

fn tof_phase(intensities: Vec<f64>) -> TofLeBailPhase {
    TofLeBailPhase::new(
        id("alpha"),
        "Phase alpha",
        vec!["100".to_owned(), "110".to_owned()],
        vec![[1, 0, 0], [1, 1, 0]],
        vec![0.72, 1.17],
        intensities,
        1.0,
    )
    .unwrap()
}

fn structural_phase() -> StructuralPhaseRecord {
    StructuralPhaseRecord {
        phase_id: id("alpha"),
        name: "Phase alpha".to_owned(),
        definition: StructuralPhaseDefinition {
            cell: UnitCell {
                a_angstrom: 5.0,
                b_angstrom: 5.0,
                c_angstrom: 5.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap(),
            hkl: vec![[1, 0, 0], [1, 1, 0]],
            multiplicity: vec![2, 4],
            fractional_xyz: vec![[0.0, 0.0, 0.0]],
            occupancy: vec![1.0],
            u_iso_angstrom2: vec![0.01],
            anisotropic_mask: vec![false],
            u_aniso_cif_angstrom2: vec![[0.0; 6]],
            scattering_species: vec!["Si".to_owned()],
            scattering_real_offset: Vec::new(),
            scattering_imag_offset: Vec::new(),
            scale: 1.0,
            coordinate_tolerance: 1.0e-10,
            scattering_model: BuiltInScatteringModel::NeutronNuclear,
            correction_model: IntegratedIntensityCorrectionModel::Neutral,
        },
        required_providers: Vec::new(),
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
