//! Rust-only joint X-ray/neutron benchmark over the pinned `PbSO4` tutorial data.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use phasesmith_core::{
    ConstantWavelengthInstrument, FcjGeometry, OwnedCwContributions, smooth_bruckner,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedReflectionGenerator, ReflectionRange,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::{
    CifAtomSite, CifReadLimits, CifStructure, PowderFormat, PowderReadLimits, read_cif_file,
    read_powder_file,
};
use phasesmith_model::{PatternRecord, RadiationProbe, RecordId};
use phasesmith_workflows::{
    BackgroundModel, ChebyshevBackground, JointRietveldHistogram, JointRietveldRefinementOptions,
    LatticeBounds, LatticeParameterization, RefinementLimits, RietveldCalculationOptions,
    RietveldInput, RietveldParameterSelection, RietveldPhase, RietveldSamplePhysicsModel,
    RietveldStructuralSelection, calculate_rietveld_pattern, refine_joint_rietveld,
};

const PHASE_ID: &str = "PbSO4";

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args_os().nth(1).map_or_else(
        || PathBuf::from("validation/data/gsasii-pbso4-cw"),
        PathBuf::from,
    );
    let workers = std::env::var("PHASESMITH_BENCHMARK_THREADS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let execution = ExecutionPolicy::new(Some(workers), 256)?;
    let structure = read_cif_file(
        directory.join("PbSO4-Wyckoff.cif"),
        None,
        true,
        CifReadLimits::default(),
    )?
    .structure;
    let histograms = vec![
        build_histogram(
            &directory,
            &structure,
            RadiationProbe::Xray,
            execution.clone(),
        )?,
        build_histogram(&directory, &structure, RadiationProbe::Neutron, execution)?,
    ];
    let options = JointRietveldRefinementOptions::new(
        RefinementLimits::new(40, 3_000, None, 100)?,
        3,
        1.0e-7,
        1.0e-8,
        1.0e-6,
        10.0,
        0.3,
        1.0e-8,
        40,
        0.15,
        12,
    )?;
    let initial = histograms
        .iter()
        .map(|histogram| {
            calculate_rietveld_pattern(&histogram.input, &histogram.calculation)
                .map(|calculation| 0.5 * calculation.metrics.chi_square)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<f64>();
    let start = Instant::now();
    let result = refine_joint_rietveld(&histograms, &[], options, None, None)?;
    let elapsed = start.elapsed().as_secs_f64();
    let cell = result.histograms[0].input.phases[0].definition().cell;
    let neutron_cell = result.histograms[1].input.phases[0].definition().cell;
    if cell != neutron_cell {
        return Err("joint result installed inconsistent shared cells".into());
    }
    if result.checkpoint.objective >= initial {
        return Err("joint PbSO4 objective did not decrease".into());
    }
    if result.metrics.included_samples != 8_378 {
        return Err("joint PbSO4 benchmark selected an unexpected sample count".into());
    }
    if !result.metrics.rwp.is_finite() || result.metrics.rwp > 0.30 {
        return Err("joint PbSO4 benchmark exceeded its native Rwp regression gate".into());
    }
    let reference = [8.480, 5.398, 6.958];
    let maximum_cell_relative_error = [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom]
        .iter()
        .zip(reference)
        .map(|(actual, expected)| (actual - expected).abs() / expected)
        .fold(0.0_f64, f64::max);
    if maximum_cell_relative_error > 0.005 {
        return Err("joint PbSO4 cell exceeded the reference regression gate".into());
    }
    println!("dataset=gsasii-pbso4-cw");
    println!("mode=joint_native_xray_neutron");
    println!("workers={workers}");
    println!("samples={}", result.metrics.included_samples);
    println!("iterations={}", result.history.len());
    println!("evaluations={}", result.evaluations);
    println!("termination={}", result.termination_reason.as_str());
    println!("initial_objective={initial:.12e}");
    println!("final_objective={:.12e}", result.checkpoint.objective);
    println!("joint_rwp={:.10}", result.metrics.rwp);
    println!("joint_rp={:.10}", result.metrics.rp);
    println!("cell_a_angstrom={:.10}", cell.a_angstrom);
    println!("cell_b_angstrom={:.10}", cell.b_angstrom);
    println!("cell_c_angstrom={:.10}", cell.c_angstrom);
    println!("maximum_cell_relative_error={maximum_cell_relative_error:.12e}");
    println!("elapsed_seconds={elapsed:.6}");
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn build_histogram(
    directory: &Path,
    structure: &CifStructure,
    probe: RadiationProbe,
    execution: ExecutionPolicy,
) -> Result<JointRietveldHistogram, Box<dyn Error>> {
    let (id, filename, angular_range, instrument, axial_geometry, position, correction, points) =
        match probe {
            RadiationProbe::Xray => (
                "xray",
                "PBSO4.XRA",
                [16.0, 158.4],
                ConstantWavelengthInstrument {
                    wavelength_angstrom: 1.5405,
                    u_deg2: 2.0e-4,
                    v_deg2: -2.0e-4,
                    w_deg2: 5.0e-4,
                    x_deg: 1.0e-3,
                    y_deg: 0.0,
                },
                Some(FcjGeometry {
                    sample_over_radius: 0.0075,
                    detector_over_radius: 0.0075,
                }),
                MonochromaticPositionCorrection {
                    zero_shift_deg: 0.0,
                    bragg_brentano_mm: None,
                    debye_scherrer_micrometre: None,
                },
                IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                    wavelength_angstrom: 1.5405,
                    polarization: 0.7,
                },
                40,
            ),
            RadiationProbe::Neutron => (
                "neutron",
                "PBSO4.CWN",
                [19.0, 153.0],
                ConstantWavelengthInstrument {
                    wavelength_angstrom: 1.909,
                    u_deg2: 354.031e-4,
                    v_deg2: -760.404e-4,
                    w_deg2: 651.592e-4,
                    x_deg: 0.0,
                    y_deg: 0.0,
                },
                None,
                MonochromaticPositionCorrection {
                    zero_shift_deg: -0.1,
                    bragg_brentano_mm: None,
                    debye_scherrer_micrometre: Some((0.0, 0.0, 650.0)),
                },
                IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
                    wavelength_angstrom: 1.909,
                },
                20,
            ),
        };
    let imported = read_powder_file(
        directory.join(filename),
        PowderFormat::GsasStd,
        1,
        PowderReadLimits::default(),
    )?;
    let source = imported.pattern;
    let indices = source
        .x_deg
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            (angular_range[0] <= *value && *value <= angular_range[1]).then_some(index)
        })
        .collect::<Vec<_>>();
    let select = |values: &[f64]| {
        indices
            .iter()
            .map(|index| values[*index])
            .collect::<Vec<_>>()
    };
    let x_deg = select(&source.x_deg);
    let observed = select(
        source
            .observed_y
            .as_ref()
            .ok_or("powder observations are missing")?,
    );
    let uncertainty = source.uncertainty.as_ref().map(|values| select(values));
    let fixed_background = smooth_bruckner(&observed, points, 50)?;
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(observed.clone()),
        uncertainty,
        None,
        Some(fixed_background.clone()),
    )?;
    let definition = phase_definition(
        structure,
        probe,
        angular_range,
        instrument.wavelength_angstrom,
        correction,
    )?;
    let mut phase = RietveldPhase::new_with_site_ids(
        RecordId::new(PHASE_ID)?,
        "PbSO4",
        structure
            .sites
            .iter()
            .map(|site| RecordId::new(&site.site_id))
            .collect::<Result<Vec<_>, _>>()?,
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )?;
    if probe == RadiationProbe::Xray {
        phase = phase.with_sample_physics(RietveldSamplePhysicsModel::Composite(vec![
            RietveldSamplePhysicsModel::IsotropicSize {
                crystallite_size_nm: 100.0,
                shape_factor: 0.9,
            },
            RietveldSamplePhysicsModel::IsotropicMicrostrain {
                rms_microstrain: 8.0e-4,
            },
        ]));
    }
    phase = with_estimated_scale(
        &pattern,
        instrument,
        axial_geometry,
        position,
        &phase,
        execution.clone(),
    )?;
    let background = BackgroundModel::Chebyshev(ChebyshevBackground::new(
        format!("{id}_residual"),
        vec![0.0; 3],
        angular_range,
    )?);
    let input = RietveldInput::new_with_background(
        pattern,
        instrument,
        axial_geometry,
        position,
        background,
        vec![phase],
    )?;
    let parameterization =
        LatticeParameterization::new(structure.space_group.clone(), structure.cell)?;
    Ok(JointRietveldHistogram {
        histogram_id: RecordId::new(id)?,
        input,
        selection: RietveldParameterSelection::new(
            RietveldStructuralSelection {
                lattice: true,
                phase_scale: true,
                ..RietveldStructuralSelection::default()
            },
            Vec::new(),
            true,
            false,
        )?,
        lattice_bounds: vec![Some(LatticeBounds::around(&parameterization, 0.02, 1.0)?)],
        calculation: RietveldCalculationOptions::new(30.0, true, execution)?,
    })
}

fn with_estimated_scale(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    axial_geometry: Option<FcjGeometry>,
    position: MonochromaticPositionCorrection,
    phase: &RietveldPhase,
    execution: ExecutionPolicy,
) -> Result<RietveldPhase, Box<dyn Error>> {
    let input = RietveldInput::new(
        pattern.clone(),
        instrument,
        axial_geometry,
        position,
        vec![phase.clone()],
    )?;
    let calculation = calculate_rietveld_pattern(
        &input,
        &RietveldCalculationOptions::new(30.0, true, execution)?,
    )?;
    let observed = pattern
        .observed_y
        .as_ref()
        .ok_or("observations are missing")?;
    let uncertainty = pattern.uncertainty.as_deref();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for index in 0..pattern.sample_count() {
        let weight = uncertainty.map_or(1.0, |sigma| 1.0 / sigma[index].powi(2));
        let target = observed[index] - pattern.background_y[index];
        numerator += weight * calculation.profile_y[index] * target;
        denominator += weight * calculation.profile_y[index].powi(2);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("cannot estimate a positive PbSO4 phase scale".into());
    }
    let mut definition = phase.definition().clone();
    definition.scale = (numerator / denominator).max(f64::MIN_POSITIVE);
    let updated = RietveldPhase::new_with_site_ids(
        phase.phase_id().clone(),
        phase.name(),
        phase.site_ids().to_vec(),
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )?;
    Ok(match phase.sample_physics().cloned() {
        Some(model) => updated.with_sample_physics(model),
        None => updated,
    })
}

fn phase_definition(
    structure: &CifStructure,
    probe: RadiationProbe,
    angular_range: [f64; 2],
    wavelength_angstrom: f64,
    correction_model: IntegratedIntensityCorrectionModel,
) -> Result<StructuralPhaseDefinition, Box<dyn Error>> {
    let reflections =
        PreparedReflectionGenerator::new(structure.space_group.clone(), true, 500_000)?.generate(
            structure.cell,
            ReflectionRange::CwTwoTheta {
                min_deg: angular_range[0],
                max_deg: angular_range[1],
                wavelength_angstrom,
            },
        )?;
    let definition = StructuralPhaseDefinition {
        cell: structure.cell,
        space_group: structure.space_group.clone(),
        hkl: reflections
            .iter()
            .map(|reflection| reflection.hkl)
            .collect(),
        multiplicity: reflections
            .iter()
            .map(|reflection| reflection.multiplicity)
            .collect(),
        fractional_xyz: structure
            .sites
            .iter()
            .map(|site| site.fractional_xyz)
            .collect(),
        occupancy: structure.sites.iter().map(|site| site.occupancy).collect(),
        u_iso_angstrom2: structure
            .sites
            .iter()
            .map(|site| site.u_iso_angstrom2.unwrap_or(0.0))
            .collect(),
        anisotropic_mask: structure
            .sites
            .iter()
            .map(|site| site.anisotropic_displacement.is_some())
            .collect(),
        u_aniso_cif_angstrom2: structure
            .sites
            .iter()
            .map(|site| {
                site.anisotropic_displacement
                    .as_ref()
                    .map_or([0.0; 6], |value| value.u_cif_angstrom2)
            })
            .collect(),
        scattering_species: structure
            .sites
            .iter()
            .map(|site| scattering_key(site, probe))
            .collect(),
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-4,
        scattering_model: match probe {
            RadiationProbe::Xray => BuiltInScatteringModel::XrayNonResonant,
            RadiationProbe::Neutron => BuiltInScatteringModel::NeutronNuclear,
        },
        correction_model,
    };
    definition.validate()?;
    Ok(definition)
}

fn scattering_key(site: &CifAtomSite, probe: RadiationProbe) -> String {
    match probe {
        RadiationProbe::Neutron => site.isotope.map_or_else(
            || site.element_symbol.clone(),
            |isotope| format!("{}-{isotope}", site.element_symbol),
        ),
        RadiationProbe::Xray => site.charge.map_or_else(
            || site.element_symbol.clone(),
            |charge| {
                format!(
                    "{}{}{}",
                    site.element_symbol,
                    charge.unsigned_abs(),
                    if charge > 0 { '+' } else { '-' }
                )
            },
        ),
    }
}
