//! Native Rietveld analysis and checkpoint persistence contracts.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions};
use phasesmith_engine::crystallography::{
    IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{
    ExperimentRecord, FixedWavelengthSpectrum, HistogramRecord, PatternRecord, ProjectRecord,
    RadiationDefinition, RadiationProbe, RecordId, StructuralPhaseRecord,
};
use phasesmith_persistence::{
    PROJECT_MANIFEST_NAME, PersistenceError, ProjectReadLimits, ProjectSaveOptions,
    load_rietveld_project, save_rietveld_project,
};
use phasesmith_workflows::{
    AffineConstraint, AmorphousBackground, AmorphousPeak, BackgroundModel, ChebyshevBackground,
    CompositeBackground, Constraint, FixedConstraint, LatticeBounds, LatticeParameterization,
    LatticeReflectionDomain, LinearConstraint, LinearTerm, ParameterKey, PointBackground,
    PolynomialBackground, RefinementLimits, RietveldAnalysis, RietveldCalculationOptions,
    RietveldCovarianceOptions, RietveldInput, RietveldParameterSelection, RietveldPhase,
    RietveldProjectState, RietveldRefinementOptions, RietveldSamplePhysicsModel,
    RietveldStructuralSelection, refine_general_rietveld,
};

static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[test]
fn complete_native_analysis_round_trips_and_rejects_corrupt_parameter_identity() {
    let directory = temporary_path("round-trip");
    let state = state();
    save_rietveld_project(&directory, &state, ProjectSaveOptions::default()).unwrap();
    assert_eq!(
        load_rietveld_project(&directory, ProjectReadLimits::default()).unwrap(),
        state
    );

    let manifest_path = directory.join(PROJECT_MANIFEST_NAME);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["rietveld_analyses"][0]["checkpoint"]["parameters"][0]["key"]["name"] =
        serde_json::json!("changed");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        load_rietveld_project(&directory, ProjectReadLimits::default()),
        Err(PersistenceError::InvalidRecord { .. })
    ));
    cleanup(directory);
}

#[test]
fn native_analysis_rejects_invalid_tail_accuracy_on_load_and_save() {
    let directory = temporary_path("invalid-tail-accuracy");
    let original = state();
    save_rietveld_project(&directory, &original, ProjectSaveOptions::default()).unwrap();
    let manifest_path = directory.join(PROJECT_MANIFEST_NAME);
    let original_manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    for bad in [0.0, -0.1, 1.0e-9, 0.11] {
        let mut manifest = original_manifest.clone();
        // Matching corrupt policies must not bypass checkpoint identity checks.
        manifest["rietveld_analyses"][0]["options"]["tail_area_tolerance"] = serde_json::json!(bad);
        manifest["rietveld_analyses"][0]["checkpoint"]["tail_area_tolerance"] =
            serde_json::json!(bad);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(
            load_rietveld_project(&directory, ProjectReadLimits::default()).is_err(),
            "invalid tail-area tolerance {bad} must fail during load"
        );
        let mut invalid = original.clone();
        invalid.analyses[0]
            .options
            .calculation
            .profile_accuracy
            .tail_area_tolerance = Some(bad);
        invalid.analyses[0]
            .checkpoint
            .as_mut()
            .unwrap()
            .profile_accuracy
            .tail_area_tolerance = Some(bad);
        assert!(invalid.validate().is_err());
        let analysis = &invalid.analyses[0];
        assert!(
            analysis
                .checkpoint
                .as_ref()
                .unwrap()
                .validate_for(
                    &analysis.input,
                    &analysis.selection,
                    &analysis.lattice_bounds,
                    &analysis.constraints,
                )
                .is_err()
        );
        assert!(
            save_rietveld_project(&directory, &invalid, ProjectSaveOptions::default()).is_err()
        );
    }
    cleanup(directory);
}

#[test]
fn native_analysis_counts_obey_project_read_limits() {
    let directory = temporary_path("analysis-limits");
    let state = state();
    save_rietveld_project(&directory, &state, ProjectSaveOptions::default()).unwrap();
    let manifest_path = directory.join(PROJECT_MANIFEST_NAME);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let duplicate = manifest["rietveld_analyses"][0].clone();
    manifest["rietveld_analyses"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        load_rietveld_project(
            &directory,
            ProjectReadLimits {
                max_histograms: 1,
                ..ProjectReadLimits::default()
            }
        ),
        Err(PersistenceError::LimitExceeded { .. })
    ));
    cleanup(directory);
}

#[test]
fn every_native_background_and_sample_physics_variant_round_trips() {
    let backgrounds = vec![
        BackgroundModel::Polynomial(PolynomialBackground::new("polynomial", vec![0.2]).unwrap()),
        BackgroundModel::Chebyshev(
            ChebyshevBackground::new("chebyshev", vec![0.2, -0.01], [20.0, 30.0]).unwrap(),
        ),
        BackgroundModel::Point(
            PointBackground::new("points", vec![20.0, 25.0, 30.0], vec![0.2, 0.3, 0.1]).unwrap(),
        ),
        BackgroundModel::Amorphous(
            AmorphousBackground::new(
                "amorphous",
                vec![AmorphousPeak::new(2.0, 25.0, 4.0).unwrap()],
            )
            .unwrap(),
        ),
        BackgroundModel::Composite(
            CompositeBackground::new(
                "composite",
                vec![
                    BackgroundModel::Polynomial(
                        PolynomialBackground::new("broad", vec![0.2]).unwrap(),
                    ),
                    BackgroundModel::Point(
                        PointBackground::new("local", vec![20.0, 30.0], vec![0.0, 0.1]).unwrap(),
                    ),
                ],
            )
            .unwrap(),
        ),
    ];
    for (index, background) in backgrounds.into_iter().enumerate() {
        let mut value = state_without_checkpoint();
        value.analyses[0].input.background = Some(background);
        assert_native_round_trip(&format!("background-{index}"), &value);
    }

    let mut value = state_without_checkpoint();
    let phase = value.analyses[0].input.phases[0].clone();
    value.analyses[0].input.phases[0] =
        phase.with_sample_physics(RietveldSamplePhysicsModel::Composite(vec![
            RietveldSamplePhysicsModel::IsotropicSize {
                crystallite_size_nm: 80.0,
                shape_factor: 0.9,
            },
            RietveldSamplePhysicsModel::IsotropicMicrostrain {
                rms_microstrain: 5.0e-4,
            },
            RietveldSamplePhysicsModel::IsotropicLorentzianMicrostrain {
                microstrain: 7.0e-4,
            },
            RietveldSamplePhysicsModel::StephensOrthorhombic {
                coefficients_angstrom_minus4: [2.0e-8, 3.0e-8, 1.0e-8, 8.0e-9, 6.0e-9, 7.0e-9],
                lorentzian_fraction: 0.35,
            },
            RietveldSamplePhysicsModel::MarchDollase {
                ratio: 0.85,
                preferred_axis_hkl: [1.0, 1.0, 0.0],
            },
        ]));
    assert_native_round_trip("sample-physics-composite", &value);
}

#[test]
fn fixed_spectrum_native_analysis_round_trips() {
    let mut value = state_without_checkpoint();
    let spectrum = FixedWavelengthSpectrum::new(vec![1.5406, 1.54439], vec![1.0, 0.48]).unwrap();
    value.project.histograms[0].experiment.radiation = RadiationDefinition::FixedSpectrum {
        probe: RadiationProbe::Xray,
        spectrum: spectrum.clone(),
    };
    let previous = value.analyses[0].input.clone();
    value.analyses[0].input = RietveldInput::new_fixed_spectrum_with_background(
        previous.pattern,
        previous.instrument,
        spectrum,
        previous.axial_geometry,
        previous.position_correction,
        previous.background.unwrap(),
        previous.phases,
    )
    .unwrap();
    assert_native_round_trip("fixed-spectrum", &value);
}

#[test]
fn affine_linear_constraints_and_guarded_domains_round_trip() {
    let mut constrained = state();
    constrained.analyses[0].checkpoint = None;
    let background = ParameterKey::new("background", "main", "coefficient_0").unwrap();
    constrained.analyses[0].constraints = vec![
        Constraint::Affine(
            AffineConstraint::new(
                ParameterKey::new("phase", "alpha", "scale").unwrap(),
                background.clone(),
                2.0,
                0.6,
            )
            .unwrap(),
        ),
        Constraint::Linear(
            LinearConstraint::new(
                ParameterKey::new("sample", "alpha", "isotropic_microstrain.rms").unwrap(),
                vec![LinearTerm::new(background, 0.0025).unwrap()],
                0.0,
            )
            .unwrap(),
        ),
    ];
    assert_native_round_trip("constraint-variants", &constrained);

    let mut dynamic = state_without_checkpoint();
    let stored = &dynamic.project.phases[0];
    let parameterization = LatticeParameterization::new(
        stored.definition.space_group.clone(),
        stored.definition.cell,
    )
    .unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.01, 1.0).unwrap();
    let domain = LatticeReflectionDomain::new(
        parameterization,
        bounds.clone(),
        1.5406,
        [20.0, 30.0],
        0.25,
        true,
        1_000_000,
        1.001,
    )
    .unwrap();
    let phase = RietveldPhase::from_lattice_domain(
        stored.phase_id.clone(),
        stored.name.clone(),
        vec![id("Si1")],
        stored.definition.clone(),
        domain,
    )
    .unwrap()
    .with_sample_physics(RietveldSamplePhysicsModel::IsotropicMicrostrain {
        rms_microstrain: 5.0e-4,
    });
    dynamic.project.phases[0].definition = phase.definition().clone();
    dynamic.analyses[0].input.phases[0] = phase;
    dynamic.analyses[0].lattice_bounds = vec![Some(bounds)];
    assert_native_round_trip("guarded-domain", &dynamic);
}

fn state_without_checkpoint() -> RietveldProjectState {
    let mut value = state();
    value.analyses[0].checkpoint = None;
    value.analyses[0].selection = RietveldParameterSelection::default();
    value.analyses[0].constraints.clear();
    value
}

fn assert_native_round_trip(label: &str, state: &RietveldProjectState) {
    let directory = temporary_path(label);
    save_rietveld_project(&directory, state, ProjectSaveOptions::default()).unwrap();
    assert_eq!(
        load_rietveld_project(&directory, ProjectReadLimits::default()).unwrap(),
        *state
    );
    cleanup(directory);
}

#[allow(clippy::too_many_lines)]
fn state() -> RietveldProjectState {
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 1.2e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    };
    let position = MonochromaticPositionCorrection {
        zero_shift_deg: 0.0,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: None,
    };
    let definition = StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 4.7,
            c_angstrom: 4.7,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        },
        space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap(),
        hkl: vec![[1, 0, 0]],
        multiplicity: vec![2],
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
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    let phase = RietveldPhase::new_with_site_ids(
        id("alpha"),
        "Alpha",
        vec![id("Si1")],
        definition.clone(),
        OwnedCwContributions::neutral(1),
    )
    .unwrap()
    .with_sample_physics(RietveldSamplePhysicsModel::IsotropicMicrostrain {
        rms_microstrain: 5.0e-4,
    });
    let x_deg = (0..101)
        .map(|index| 20.0 + f64::from(index) * 0.1)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        None,
    )
    .unwrap();
    let background =
        BackgroundModel::Polynomial(PolynomialBackground::new("main", vec![0.2]).unwrap());
    let input = RietveldInput::new_with_background(
        pattern.clone(),
        instrument,
        None,
        position,
        background,
        vec![phase.clone()],
    )
    .unwrap();
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        true,
        true,
    )
    .unwrap();
    let constraints = vec![
        fixed("sample", "alpha", "isotropic_microstrain.rms", 5.0e-4),
        fixed("background", "main", "coefficient_0", 0.2),
        fixed("phase", "alpha", "scale", 1.0),
    ];
    let options = RietveldRefinementOptions::new(
        RietveldCalculationOptions::new(20.0, true, ExecutionPolicy::new(Some(1), 2).unwrap())
            .unwrap(),
        RefinementLimits::new(4, 100, None, 10).unwrap(),
        1,
        1.0e-8,
        1.0e-7,
        1.0e-6,
        10.0,
        0.3,
        1.0e-6,
        20,
        0.25,
        4,
    )
    .unwrap();
    let covariance = RietveldCovarianceOptions::new(true, 16, 0.999).unwrap();
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &constraints,
        &options,
        covariance,
        None,
        None,
    )
    .unwrap();
    RietveldProjectState {
        project: ProjectRecord {
            project_id: id("project"),
            revision: 7,
            name: "Rietveld persistence".to_owned(),
            histograms: vec![HistogramRecord {
                histogram_id: id("histogram"),
                name: "Histogram".to_owned(),
                pattern,
                experiment: ExperimentRecord::new(
                    instrument,
                    RadiationDefinition::Monochromatic {
                        probe: RadiationProbe::Xray,
                        wavelength_angstrom: instrument.wavelength_angstrom,
                    },
                    None,
                    position,
                )
                .unwrap(),
                phase_ids: vec![id("alpha")],
            }],
            tof_histograms: Vec::new(),
            phases: vec![StructuralPhaseRecord {
                phase_id: id("alpha"),
                name: "Alpha".to_owned(),
                definition,
                required_providers: Vec::new(),
            }],
            metadata: BTreeMap::new(),
        },
        analyses: vec![RietveldAnalysis {
            histogram_id: id("histogram"),
            input,
            selection,
            lattice_bounds: vec![None],
            constraints,
            options,
            covariance,
            checkpoint: Some(result.checkpoint),
        }],
    }
}

fn fixed(module: &str, owner: &str, name: &str, value: f64) -> Constraint {
    Constraint::Fixed(
        FixedConstraint::new(ParameterKey::new(module, owner, name).unwrap(), value).unwrap(),
    )
}

fn id(value: &str) -> RecordId {
    RecordId::new(value).unwrap()
}

fn temporary_path(label: &str) -> PathBuf {
    let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "phasesmith-native-rietveld-{label}-{}-{sequence}",
        std::process::id()
    ))
}

fn cleanup(path: PathBuf) {
    if path.exists() {
        fs::remove_dir_all(path).unwrap();
    }
}

#[test]
fn mixed_rietveld_and_pawley_bundle_preserves_methods_and_resume() {
    use phasesmith_persistence::{ProjectBundle, load_project_bundle, save_project_bundle};
    use phasesmith_workflows::{
        PawleyAnalysis, PawleyInput, PawleyOptions, PawleyPhase, RefinementRuntime,
        pawley_parameters, refine_pawley, refine_pawley_with_runtime,
    };
    let structural = state();
    let histogram = &structural.project.histograms[0];
    let phases: Vec<_> = histogram
        .phase_ids
        .iter()
        .map(|id| PawleyPhase {
            id: id.as_str().into(),
            reflection_ids: vec!["family".into()],
            two_theta_deg: vec![25.0],
            intensities: vec![1.0],
            hkl: Vec::new(),
            lattice: None,
        })
        .collect();
    let input = PawleyInput {
        fixed_spectrum: None,
        pattern: histogram.pattern.clone(),
        instrument: histogram.experiment.instrument,
        axial: histogram.experiment.axial_geometry,
        parameters: pawley_parameters(&phases, histogram.experiment.instrument, None, false)
            .unwrap(),
        phases,
        background: None,
        signed_intensities: false,
        constraints: Vec::new(),
    };
    let options = PawleyOptions::default();
    let fitted = refine_pawley(&input, &options).unwrap();
    let mut bundle = ProjectBundle::new(structural.project.clone());
    bundle.rietveld_analyses.clone_from(&structural.analyses);
    bundle.pawley_analyses.push(PawleyAnalysis {
        histogram_id: histogram.histogram_id.clone(),
        input,
        options,
        checkpoint: Some(fitted.checkpoint),
    });
    let directory = temporary_path("mixed-pawley");
    save_project_bundle(&directory, &bundle, ProjectSaveOptions::default()).unwrap();
    let restored = load_project_bundle(&directory, ProjectReadLimits::default()).unwrap();
    assert_eq!(restored, bundle);
    let a = &restored.pawley_analyses[0];
    let mut runtime =
        RefinementRuntime::new(phasesmith_workflows::RefinementLimits::default(), None).unwrap();
    let resumed =
        refine_pawley_with_runtime(&a.input, &a.options, a.checkpoint.as_ref(), &mut runtime)
            .unwrap();
    assert_eq!(
        resumed.evaluation.calculated_y,
        fitted.evaluation.calculated_y
    );
    let manifest_path = directory.join(PROJECT_MANIFEST_NAME);
    let mut wire: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert!(wire["pawley_analyses"][0].get("x_deg").is_none());
    wire["pawley_analyses"][0]["histogram_id"] = serde_json::json!("missing");
    std::fs::write(&manifest_path, serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(load_project_bundle(&directory, ProjectReadLimits::default()).is_err());
    cleanup(directory);
}

#[test]
fn pawley_cell_metadata_must_match_a_shared_structural_phase() {
    use phasesmith_workflows::{
        PawleyAnalysis, PawleyInput, PawleyOptions, PawleyPhase, PawleyProjectState,
        pawley_parameters,
    };
    let structural = state();
    let shared = &structural.project.phases[0];
    let par = LatticeParameterization::new(
        shared.definition.space_group.clone(),
        shared.definition.cell,
    )
    .unwrap();
    let bounds = LatticeBounds::around(&par, 0.01, 1.0).unwrap();
    let domain = LatticeReflectionDomain::new(
        par,
        bounds,
        1.5406,
        [20.0, 30.0],
        1.0,
        true,
        1_000_000,
        1.001,
    )
    .unwrap();
    let phases = vec![PawleyPhase::from_domain(shared.phase_id.as_str().into(), domain).unwrap()];
    let h = &structural.project.histograms[0];
    let input = PawleyInput {
        fixed_spectrum: None,
        pattern: h.pattern.clone(),
        instrument: h.experiment.instrument,
        axial: h.experiment.axial_geometry,
        parameters: pawley_parameters(&phases, h.experiment.instrument, None, false).unwrap(),
        phases,
        background: None,
        signed_intensities: false,
        constraints: Vec::new(),
    };
    let mut state = PawleyProjectState {
        project: structural.project.clone(),
        analyses: vec![PawleyAnalysis {
            histogram_id: h.histogram_id.clone(),
            input,
            options: PawleyOptions::default(),
            checkpoint: None,
        }],
    };
    state.validate().unwrap();
    state.project.phases[0].definition.cell.a_angstrom += 0.001;
    assert!(
        state
            .validate()
            .unwrap_err()
            .to_string()
            .contains("cell/symmetry")
    );
}
