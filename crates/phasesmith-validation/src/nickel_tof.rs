//! Non-POWGEN TOF transferability gate using the classic LANL nickel example.

#![allow(clippy::cast_precision_loss)]

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::time::Instant;

use phasesmith_core::{
    BackgroundError, OwnedCwContributions, TofIncidentSpectrumError, TofInstrumentParameter,
    smooth_bruckner,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedReflectionGenerator, ReflectionGenerationError,
    ReflectionRange, UnitCell,
};
use phasesmith_engine::{BuiltInScatteringModel, StructuralPhaseDefinition};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_io::{
    GsasTofInstrumentIoError, GsasTofInstrumentReadLimits, PowderIoError, PowderReadLimits,
    SpaceGroupLookupError, TofPowderFormat, read_gsas_tof_instrument_file, read_tof_powder_file,
    space_group_by_number,
};
use phasesmith_model::{DomainError, RecordId, TofPatternRecord};
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, ParameterBounds, ParameterError,
    PreparedStructuralTofMultiBankObjective, RefinementLimits, RietveldError, RietveldPhase,
    RietveldStructuralSelection, RuntimeError, StructuralTofBank, StructuralTofMultiBankError,
    StructuralTofMultiBankInput, StructuralTofMultiBankRefinementError,
    StructuralTofMultiBankRefinementOptions, TofBankInstrumentModel, TofChebyshevBackground,
    TofGeometryParameterKey, TofInstrumentParameterBound, TofLeBailBank, TofLeBailError,
    TofLeBailInput, TofLeBailOptions, TofLeBailPhase, TofMultiBankGeometryError,
    TofMultiBankGeometryInput, TofMultiBankGeometryOptions, TofMultiBankInput,
    TofMultiBankLatticeInput, TofSharedLatticePhase, refine_structural_tof_multibank,
    refine_tof_lebail, refine_tof_multibank_geometry,
};

use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "lanl-nickel-tof";
const BANK: usize = 2;
const FIT_MIN_US: f64 = 1_101.6;
const FIT_MAX_US: f64 = 8_189.6;
const NICKEL_LATTICE_ANGSTROM: f64 = 3.523_4;
const MULTIBANK_INITIAL_LATTICE_ANGSTROM: f64 = 3.523;
const MULTIBANK_MIN_D_ANGSTROM: f64 = 0.2;
const MULTIBANK_MAX_D_ANGSTROM: f64 = 3.0;
const STRUCTURAL_INITIAL_LATTICE_ANGSTROM: f64 = 3.522;

/// Run the complete native fixed-instrument TOF Le Bail transferability gate.
///
/// This intentionally uses the LANL/GSAS packed constant-step format and legacy
/// profile function 1, rather than POWGEN's SLOG FXYE/profile-function-3 pair.
///
/// # Errors
///
/// Returns [`NickelTofValidationError`] for corrupt data, unsupported records,
/// invalid numerical state, or report construction failures.
#[allow(clippy::too_many_lines)]
pub fn run_nickel_tof_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, NickelTofValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let imported = read_tof_powder_file(
        dataset_directory.join("nickel.raw"),
        BANK,
        PowderReadLimits::default(),
    )?;
    let calibration = read_gsas_tof_instrument_file(
        dataset_directory.join("inst_tof.prm"),
        BANK,
        GsasTofInstrumentReadLimits::default(),
    )?;
    let bank_two_theta_deg = calibration
        .bank_geometry
        .map(|geometry| geometry.two_theta_deg);
    let instrument = calibration.instrument;
    let start = imported
        .pattern
        .tof_us
        .partition_point(|value| *value < FIT_MIN_US);
    let end = imported
        .pattern
        .tof_us
        .partition_point(|value| *value <= FIT_MAX_US);
    if start >= end {
        return Err(NickelTofValidationError::InvalidData(
            "LANL nickel fit interval is empty".to_owned(),
        ));
    }
    let tof_us = imported.pattern.tof_us[start..end].to_vec();
    let observed = imported
        .pattern
        .observed_y
        .as_deref()
        .ok_or(NickelTofValidationError::MissingObservations)?[start..end]
        .to_vec();
    let uncertainty = imported
        .pattern
        .uncertainty
        .as_deref()
        .map(|values| values[start..end].to_vec());
    let mask = imported
        .pattern
        .mask
        .as_deref()
        .map(|values| values[start..end].to_vec());
    let fixed_background = smooth_bruckner(&observed, 20, 50)?;
    let pattern = TofPatternRecord::new(
        tof_us,
        Some(observed.clone()),
        uncertainty,
        mask,
        Some(fixed_background),
    )?;
    let reflections =
        PreparedReflectionGenerator::new(space_group_by_number(225)?.space_group, true, 100_000)?
            .generate(
            nickel_cell(),
            ReflectionRange::Tof {
                min_us: pattern.tof_us[0],
                max_us: pattern.tof_us[pattern.sample_count() - 1],
                search_min_d_angstrom: 0.2,
                search_max_d_angstrom: 3.0,
                zero_us: instrument.zero_us,
                difc_us_per_angstrom: instrument.difc_us_per_angstrom,
                difa_us_per_angstrom2: instrument.difa_us_per_angstrom2,
                difb_us_angstrom: instrument.difb_us_angstrom,
            },
        )?;
    let phase = TofLeBailPhase::new(
        RecordId::new("nickel")?,
        "FCC nickel powder standard",
        reflections
            .iter()
            .map(|reflection| reflection.reflection_id.clone())
            .collect(),
        reflections
            .iter()
            .map(|reflection| reflection.hkl)
            .collect(),
        reflections
            .iter()
            .map(|reflection| reflection.d_spacing_angstrom)
            .collect(),
        vec![0.0; reflections.len()],
        1.0,
    )?;
    let background = TofChebyshevBackground::new(
        RecordId::new("nickel-chebyshev-background")?,
        vec![0.0; 12],
        [
            pattern.tof_us[0],
            pattern.tof_us[pattern.sample_count() - 1],
        ],
    )?;
    let request = TofLeBailInput::new(pattern, instrument, vec![phase])?
        .with_refinable_background(background)?;
    let options = TofLeBailOptions::new(
        20,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2)?,
    )?
    .with_redistribution_uncertainty(false);
    let result = refine_tof_lebail(&request, &options)?;
    let correlation = pearson_correlation(
        &observed,
        &result.calculation.background_y,
        &result.calculation.profile_y,
        request.pattern.mask.as_deref(),
    )?;
    let monotonic = result
        .history
        .windows(2)
        .all(|pair| pair[1].metrics.rwp <= pair[0].metrics.rwp);
    let multibank = run_multibank_geometry(dataset_directory)?;
    let structural = run_multibank_structural(dataset_directory)?;
    let mut checks = vec![
        check(
            "tof_nickel_bank_geometry",
            bank_two_theta_deg.is_some_and(|value| value.to_bits() == 88.05_f64.to_bits()),
            "The LANL BNKPAR scattering-angle field is imported into the same facility-neutral bank geometry used for POWGEN.",
            bank_two_theta_deg,
            "LANL bank 2 two_theta_deg = 88.05",
        )?,
        check(
            "tof_non_powgen_format",
            imported.format == TofPowderFormat::GsasConstStd
                && imported.bank == Some(BANK)
                && imported.pattern.sample_count() == 5_120,
            "The native adapter resolves the packed constant-step microsecond convention independently of POWGEN SLOG FXYE.",
            Some(imported.pattern.sample_count() as f64),
            "bank 2, gsas_const_std, 5120 imported bins",
        )?,
        check(
            "tof_profile_function_one",
            calibration.profile_function == 1
                && instrument.difc_us_per_angstrom.to_bits() == 4_368.97_f64.to_bits()
                && instrument.alpha_coefficient.to_bits() == 0.142_760_f64.to_bits(),
            "Legacy GSAS profile function 1 is translated into the common typed 15-coefficient model.",
            Some(calibration.profile_function as f64),
            "profile function 1 with DIFC=4368.97 us/A and alpha=0.142760",
        )?,
        check(
            "tof_nickel_incident_spectrum",
            calibration.incident_spectrum.is_some_and(|spectrum| {
                (spectrum.min_tof_us - 750.0).abs() <= 1.0e-9
                    && (spectrum.max_tof_us - 8_190.4).abs() <= 1.0e-9
            }),
            "The bounded adapter translates the bank-local Maxwellian/Chebyshev spectrum into the facility-neutral microsecond contract.",
            calibration
                .incident_spectrum
                .map(|spectrum| spectrum.max_tof_us),
            "type 4 incident spectrum valid on 750.0..8190.4 us",
        )?,
        check(
            "tof_nickel_fit_window",
            request.pattern.sample_count() == 4_431
                && (request.pattern.tof_us[0] - FIT_MIN_US).abs() <= 1.0e-9
                && (request.pattern.tof_us[request.pattern.sample_count() - 1] - FIT_MAX_US).abs()
                    <= 1.0e-9,
            "The published tutorial fit interval uses bin centers in microseconds.",
            Some(request.pattern.sample_count() as f64),
            "4431 centers from 1101.6 through 8189.6 us",
        )?,
        check(
            "tof_nickel_reflections",
            reflections.len() == 102,
            "The Fm-3m nickel cell generates stable reflection families across the selected bank.",
            Some(reflections.len() as f64),
            "102 reflection families",
        )?,
        check(
            "tof_nickel_profile_fit",
            result.metrics.rwp <= 0.026,
            "The native fixed-instrument TOF extraction fits the non-POWGEN nickel observations.",
            Some(result.metrics.rwp),
            "uncertainty-weighted Rwp <= 0.026",
        )?,
        check(
            "tof_nickel_profile_correlation",
            correlation >= 0.998,
            "The background-subtracted observations and extracted profile remain aligned.",
            Some(correlation),
            "masked Pearson correlation >= 0.998",
        )?,
        check(
            "tof_nickel_convergence",
            monotonic && result.history.len() == 20,
            "Every accepted nonnegative redistribution cycle is non-increasing in Rwp.",
            Some(result.history.len() as f64),
            "20 cycles with non-increasing Rwp",
        )?,
        check(
            "tof_nickel_derivatives",
            result
                .calculation
                .accumulation
                .derivatives
                .global
                .as_ref()
                .is_some_and(|jacobian| jacobian.parameter_count == 15),
            "The real-pattern calculation retains all analytical instrument rows from the fused pass.",
            Some(15.0),
            "15 instrument derivative rows",
        )?,
    ];
    checks.extend([
        check(
            "tof_nickel_multibank_atomic",
            multibank.bank_count == 3
                && multibank.history_count == 20
                && multibank.included_samples == 13_293,
            "Banks 2--4 advance in one atomic shared-cell/local-instrument workflow.",
            Some(multibank.included_samples as f64),
            "3 banks, 20 accepted cycles, and 13293 included observations",
        )?,
        check(
            "tof_nickel_multibank_fit",
            multibank.rwp <= 0.03
                && multibank
                    .bank_rwp
                    .iter()
                    .all(|value| value.is_finite() && *value <= 0.03),
            "The concatenated real observations and every member bank pass explicit Rwp gates.",
            Some(multibank.rwp),
            "joint Rwp <= 0.03 and every bank Rwp <= 0.03",
        )?,
        check(
            "tof_nickel_multibank_lattice",
            (multibank.lattice_angstrom - NICKEL_LATTICE_ANGSTROM).abs() <= 0.0005,
            "The displaced shared cubic cell returns to the published nickel lattice.",
            Some(multibank.lattice_angstrom),
            "|a - 3.5234 A| <= 0.0005 A",
        )?,
        check(
            "tof_nickel_multibank_identifiability",
            multibank.parameter_count == 4
                && multibank.jacobian_rank == 4
                && multibank.cross_family_correlations > 0,
            "Rank and lattice/instrument correlations are retained for the real joint system.",
            Some(multibank.jacobian_rank as f64),
            "4 selected columns, rank 4, and at least one cross-family correlation",
        )?,
        check(
            "tof_nickel_structural_multibank",
            structural.bank_count == 3
                && structural.included_samples == 13_293
                && structural.parameter_count == 8
                && structural.history_count > 0,
            "One structural Fm-3m nickel phase and three explicit bank models refine in one accepted-state objective.",
            Some(structural.history_count as f64),
            "3 banks, 13293 included observations, 8 parameters, and at least one accepted step",
        )?,
        check(
            "tof_nickel_structural_fit",
            structural.rwp <= 0.04,
            "The native neutron structure-factor intensities reproduce every real bank without Le Bail intensity redistribution.",
            Some(structural.rwp),
            "joint Rwp <= 0.04",
        )?,
        check(
            "tof_nickel_structural_correlation",
            structural.minimum_profile_correlation >= 0.995,
            "Every incident-normalized bank retains a high background-subtracted structural profile correlation.",
            Some(structural.minimum_profile_correlation),
            "minimum bank profile correlation >= 0.995",
        )?,
        check(
            "tof_nickel_structural_lattice",
            (structural.lattice_angstrom - NICKEL_LATTICE_ANGSTROM).abs() <= 0.002,
            "The real structural objective returns the displaced shared cubic cell to the published nickel lattice.",
            Some(structural.lattice_angstrom),
            "|a - 3.5234 A| <= 0.002 A",
        )?,
        check(
            "tof_nickel_structural_u_iso",
            (0.0..=0.05).contains(&structural.u_iso_angstrom2),
            "The shared Ni isotropic displacement remains in an explicit physical validation interval.",
            Some(structural.u_iso_angstrom2),
            "0 <= Uiso <= 0.05 A^2",
        )?,
    ]);
    RealDataValidationReport::new(
        DATASET_ID,
        request.pattern.sample_count(),
        Some(reflections.len()),
        started.elapsed().as_secs_f64(),
        checks,
        vec![
            format!(
                "LANL bank 2 profile function 1; range={FIT_MIN_US:.1}..{FIT_MAX_US:.1} us; DIFC={:.5} us/A.",
                instrument.difc_us_per_angstrom,
            ),
            format!(
                "Final Rwp={:.8}; Rp={:.8}; correlation={correlation:.8}; background=Smooth Bruckner + 12-term Chebyshev.",
                result.metrics.rwp, result.metrics.rp,
            ),
            format!(
                "Joint banks 2--4 Rwp={:.8}; bank Rwp={:?}; a={:.8} A; rank={}/{}.",
                multibank.rwp,
                multibank.bank_rwp,
                multibank.lattice_angstrom,
                multibank.jacobian_rank,
                multibank.parameter_count,
            ),
            format!(
                "Structural banks 2--4 Rwp={:.8}; minimum correlation={:.8}; a={:.8} A; Ni Uiso={:.8} A^2; accepted steps={}; evaluations={}.",
                structural.rwp,
                structural.minimum_profile_correlation,
                structural.lattice_angstrom,
                structural.u_iso_angstrom2,
                structural.history_count,
                structural.evaluation_count,
            ),
        ],
    )
    .map_err(Into::into)
}

fn nickel_cell() -> UnitCell {
    UnitCell {
        a_angstrom: NICKEL_LATTICE_ANGSTROM,
        b_angstrom: NICKEL_LATTICE_ANGSTROM,
        c_angstrom: NICKEL_LATTICE_ANGSTROM,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct MultiBankAcceptance {
    bank_count: usize,
    included_samples: usize,
    history_count: usize,
    rwp: f64,
    bank_rwp: Vec<f64>,
    lattice_angstrom: f64,
    parameter_count: usize,
    jacobian_rank: usize,
    cross_family_correlations: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct StructuralAcceptance {
    bank_count: usize,
    included_samples: usize,
    history_count: usize,
    evaluation_count: usize,
    parameter_count: usize,
    rwp: f64,
    minimum_profile_correlation: f64,
    lattice_angstrom: f64,
    u_iso_angstrom2: f64,
}

#[allow(clippy::too_many_lines)]
fn run_multibank_geometry(
    dataset_directory: &Path,
) -> Result<MultiBankAcceptance, NickelTofValidationError> {
    let initial_cell = UnitCell {
        a_angstrom: MULTIBANK_INITIAL_LATTICE_ANGSTROM,
        b_angstrom: MULTIBANK_INITIAL_LATTICE_ANGSTROM,
        c_angstrom: MULTIBANK_INITIAL_LATTICE_ANGSTROM,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let space_group = space_group_by_number(225)?.space_group;
    let reflections = PreparedReflectionGenerator::new(space_group.clone(), true, 100_000)?
        .generate(
            initial_cell,
            ReflectionRange::DSpacing {
                min_angstrom: MULTIBANK_MIN_D_ANGSTROM,
                max_angstrom: MULTIBANK_MAX_D_ANGSTROM,
            },
        )?;
    let phase_id = RecordId::new("nickel")?;
    let mut banks = Vec::with_capacity(3);
    let mut instrument_models = Vec::with_capacity(3);
    for bank_number in 2..=4 {
        let imported = read_tof_powder_file(
            dataset_directory.join("nickel.raw"),
            bank_number,
            PowderReadLimits::default(),
        )?;
        let calibration = read_gsas_tof_instrument_file(
            dataset_directory.join("inst_tof.prm"),
            bank_number,
            GsasTofInstrumentReadLimits::default(),
        )?;
        if imported.format != TofPowderFormat::GsasConstStd || calibration.profile_function != 1 {
            return Err(NickelTofValidationError::InvalidData(format!(
                "LANL nickel bank {bank_number} does not use the pinned packed/function-1 contract"
            )));
        }
        let instrument = calibration.instrument;
        let start = imported
            .pattern
            .tof_us
            .partition_point(|value| *value < FIT_MIN_US);
        let end = imported
            .pattern
            .tof_us
            .partition_point(|value| *value <= FIT_MAX_US);
        if start >= end {
            return Err(NickelTofValidationError::InvalidData(format!(
                "LANL nickel bank {bank_number} common TOF interval is empty"
            )));
        }
        let tof_us = imported.pattern.tof_us[start..end].to_vec();
        let observed = imported
            .pattern
            .observed_y
            .as_deref()
            .ok_or(NickelTofValidationError::MissingObservations)?[start..end]
            .to_vec();
        let uncertainty = imported
            .pattern
            .uncertainty
            .as_deref()
            .map(|values| values[start..end].to_vec());
        let mask = imported
            .pattern
            .mask
            .as_deref()
            .map(|values| values[start..end].to_vec());
        let fixed_background = smooth_bruckner(&observed, 20, 50)?;
        let pattern = TofPatternRecord::new(
            tof_us,
            Some(observed),
            uncertainty,
            mask,
            Some(fixed_background),
        )?;
        let phase = TofLeBailPhase::new(
            phase_id.clone(),
            "FCC nickel powder standard",
            reflections
                .iter()
                .map(|reflection| reflection.reflection_id.clone())
                .collect(),
            reflections
                .iter()
                .map(|reflection| reflection.hkl)
                .collect(),
            reflections
                .iter()
                .map(|reflection| reflection.d_spacing_angstrom)
                .collect(),
            vec![0.0; reflections.len()],
            1.0,
        )?;
        let bank_id = RecordId::new(format!("nickel-bank-{bank_number}"))?;
        let background = TofChebyshevBackground::new(
            RecordId::new(format!("nickel-bank-{bank_number}-background"))?,
            vec![0.0; 12],
            [
                pattern.tof_us[0],
                pattern.tof_us[pattern.sample_count() - 1],
            ],
        )?;
        let input = TofLeBailInput::new(pattern, instrument, vec![phase])?
            .with_refinable_background(background)?;
        banks.push(TofLeBailBank {
            bank_id: bank_id.clone(),
            input,
        });
        instrument_models.push(
            TofBankInstrumentModel::new(
                bank_id,
                vec![
                    TofInstrumentParameterBound::new(
                        TofInstrumentParameter::Zero,
                        instrument.zero_us - 20.0,
                        instrument.zero_us + 20.0,
                    )
                    .map_err(TofMultiBankGeometryError::from)?,
                ],
            )
            .map_err(TofMultiBankGeometryError::from)?,
        );
    }
    let parameterization = LatticeParameterization::new(space_group, initial_cell)
        .map_err(TofMultiBankGeometryError::from)?;
    let bounds = LatticeBounds::around(&parameterization, 0.01, 1.0)
        .map_err(TofMultiBankGeometryError::from)?;
    let lattice_phase =
        TofSharedLatticePhase::new(phase_id, parameterization, bounds, initial_cell)
            .map_err(TofMultiBankGeometryError::from)?;
    let input = TofMultiBankGeometryInput {
        lattice: TofMultiBankLatticeInput {
            multibank: TofMultiBankInput { banks },
            lattice_phases: vec![lattice_phase],
        },
        instrument_models,
    };
    let lebail = TofLeBailOptions::new(
        20,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2)?,
    )?
    .with_redistribution_uncertainty(false);
    let options = TofMultiBankGeometryOptions::new(lebail, 1.0e-10, 0.03, 10, 0.5)?;
    let result = refine_tof_multibank_geometry(&input, &options)?;
    let cross_family_correlations = result
        .diagnostics
        .unresolved_correlations
        .iter()
        .filter(|pair| {
            matches!(
                (&pair.left, &pair.right),
                (
                    TofGeometryParameterKey::Lattice { .. },
                    TofGeometryParameterKey::Instrument { .. }
                ) | (
                    TofGeometryParameterKey::Instrument { .. },
                    TofGeometryParameterKey::Lattice { .. }
                )
            )
        })
        .count();
    Ok(MultiBankAcceptance {
        bank_count: result.banks.len(),
        included_samples: result.metrics.included_samples,
        history_count: result.history.len(),
        rwp: result.metrics.rwp,
        bank_rwp: result.banks.iter().map(|bank| bank.metrics.rwp).collect(),
        lattice_angstrom: result.lattice_phases[0].cell.a_angstrom,
        parameter_count: result.diagnostics.parameter_count,
        jacobian_rank: result.diagnostics.jacobian_rank,
        cross_family_correlations,
    })
}

#[allow(clippy::too_many_lines)]
fn run_multibank_structural(
    dataset_directory: &Path,
) -> Result<StructuralAcceptance, NickelTofValidationError> {
    let initial_cell = UnitCell {
        a_angstrom: STRUCTURAL_INITIAL_LATTICE_ANGSTROM,
        b_angstrom: STRUCTURAL_INITIAL_LATTICE_ANGSTROM,
        c_angstrom: STRUCTURAL_INITIAL_LATTICE_ANGSTROM,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let space_group = space_group_by_number(225)?.space_group;
    let reflections = PreparedReflectionGenerator::new(space_group.clone(), true, 100_000)?
        .generate(
            initial_cell,
            ReflectionRange::DSpacing {
                min_angstrom: MULTIBANK_MIN_D_ANGSTROM,
                max_angstrom: MULTIBANK_MAX_D_ANGSTROM,
            },
        )?;
    let definition = StructuralPhaseDefinition {
        cell: initial_cell,
        space_group: space_group.clone(),
        hkl: reflections
            .iter()
            .map(|reflection| reflection.hkl)
            .collect(),
        multiplicity: reflections
            .iter()
            .map(|reflection| reflection.multiplicity)
            .collect(),
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
        RecordId::new("nickel")?,
        "FCC nickel powder standard",
        vec![RecordId::new("nickel-origin")?],
        definition,
        OwnedCwContributions::neutral(reflections.len()),
    )?;
    let mut banks = Vec::with_capacity(3);
    for bank_number in 2..=4 {
        let imported = read_tof_powder_file(
            dataset_directory.join("nickel.raw"),
            bank_number,
            PowderReadLimits::default(),
        )?;
        let calibration = read_gsas_tof_instrument_file(
            dataset_directory.join("inst_tof.prm"),
            bank_number,
            GsasTofInstrumentReadLimits::default(),
        )?;
        let geometry = calibration.bank_geometry.ok_or_else(|| {
            NickelTofValidationError::InvalidData(format!(
                "LANL nickel bank {bank_number} has no scattering angle"
            ))
        })?;
        let incident_spectrum = calibration.incident_spectrum.ok_or_else(|| {
            NickelTofValidationError::InvalidData(format!(
                "LANL nickel bank {bank_number} has no calibrated incident spectrum"
            ))
        })?;
        let start = imported
            .pattern
            .tof_us
            .partition_point(|value| *value < FIT_MIN_US);
        let end = imported
            .pattern
            .tof_us
            .partition_point(|value| *value <= FIT_MAX_US);
        let tof_us = imported.pattern.tof_us[start..end].to_vec();
        let mut observed = imported
            .pattern
            .observed_y
            .as_deref()
            .ok_or(NickelTofValidationError::MissingObservations)?[start..end]
            .to_vec();
        let mut uncertainty = imported
            .pattern
            .uncertainty
            .as_deref()
            .map(|values| values[start..end].to_vec());
        for (sample, (&tof_us, observed)) in tof_us.iter().zip(&mut observed).enumerate() {
            let incident = incident_spectrum.evaluate(tof_us)?.value;
            *observed /= incident;
            if let Some(values) = uncertainty.as_mut() {
                values[sample] /= incident;
            }
        }
        let fixed_background = smooth_bruckner(&observed, 20, 50)?;
        let pattern = TofPatternRecord::new(
            tof_us,
            Some(observed),
            uncertainty,
            imported
                .pattern
                .mask
                .as_deref()
                .map(|values| values[start..end].to_vec()),
            Some(fixed_background),
        )?;
        banks.push(StructuralTofBank {
            bank_id: RecordId::new(format!("nickel-bank-{bank_number}"))?,
            pattern,
            instrument: calibration.instrument,
            geometry,
            correction_model: IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: geometry.two_theta_deg,
            },
            scale: 1.0,
            scale_bounds: ParameterBounds::default(),
            refine_scale: true,
            background: None,
            refine_background: false,
            instrument_bounds: vec![
                TofInstrumentParameterBound::new(
                    TofInstrumentParameter::Zero,
                    calibration.instrument.zero_us - 20.0,
                    calibration.instrument.zero_us + 20.0,
                )
                .map_err(TofMultiBankGeometryError::from)?,
            ],
        });
    }
    let parameterization = LatticeParameterization::new(space_group, initial_cell)
        .map_err(TofMultiBankGeometryError::from)?;
    let lattice_bounds = LatticeBounds::around(&parameterization, 0.01, 1.0)
        .map_err(TofMultiBankGeometryError::from)?;
    let mut input = StructuralTofMultiBankInput {
        phase,
        structural_selection: RietveldStructuralSelection {
            lattice: true,
            u_iso: true,
            ..RietveldStructuralSelection::default()
        },
        lattice_bounds: Some(lattice_bounds),
        banks,
        support_fwhm: 20.0,
        tail_log: 20.0,
        use_uncertainty: true,
        execution: ExecutionPolicy::new(Some(1), 2)?,
    };
    let unit = PreparedStructuralTofMultiBankObjective::new(input.clone())?.calculate()?;
    for (bank, calculated) in input.banks.iter_mut().zip(unit.banks) {
        let observed = bank
            .pattern
            .observed_y
            .as_deref()
            .ok_or(NickelTofValidationError::MissingObservations)?;
        let uncertainty = bank.pattern.uncertainty.as_deref();
        let mask = bank.pattern.mask.as_deref();
        let (numerator, denominator) = calculated.profile_y.iter().enumerate().fold(
            (0.0, 0.0),
            |(numerator, denominator), (sample, profile)| {
                if mask.is_none_or(|values| values[sample]) {
                    let weight = uncertainty.map_or(1.0, |values| 1.0 / values[sample].powi(2));
                    let target = observed[sample] - calculated.background_y[sample];
                    (
                        numerator + weight * profile * target,
                        denominator + weight * profile * profile,
                    )
                } else {
                    (numerator, denominator)
                }
            },
        );
        let scale = (numerator / denominator).max(f64::EPSILON);
        if !scale.is_finite() {
            return Err(NickelTofValidationError::InvalidData(
                "structural TOF scale estimate is invalid".to_owned(),
            ));
        }
        bank.scale = scale;
        bank.scale_bounds = ParameterBounds::new(0.1 * scale, 10.0 * scale)?;
    }
    input.validate()?;
    let options = StructuralTofMultiBankRefinementOptions::new(
        RefinementLimits::new(12, 200, None, 8)?,
        1,
        1.0e-10,
        1.0e-8,
        1.0e-3,
        10.0,
        0.3,
        0.5,
        8,
    )?;
    let result = refine_structural_tof_multibank(&input, options, None, None)?;
    let (rwp, included_samples) = structural_rwp(&result.input, &result.calculation.banks)?;
    let minimum_profile_correlation = result
        .input
        .banks
        .iter()
        .zip(&result.calculation.banks)
        .map(|(bank, calculation)| {
            pearson_correlation(
                bank.pattern
                    .observed_y
                    .as_deref()
                    .ok_or(NickelTofValidationError::MissingObservations)?,
                &calculation.background_y,
                &calculation.profile_y,
                bank.pattern.mask.as_deref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .reduce(f64::min)
        .ok_or_else(|| NickelTofValidationError::InvalidData("no structural banks".to_owned()))?;
    Ok(StructuralAcceptance {
        bank_count: result.input.banks.len(),
        included_samples,
        history_count: result.history.len(),
        evaluation_count: result.evaluations,
        parameter_count: result.parameters.specs().len(),
        rwp,
        minimum_profile_correlation,
        lattice_angstrom: result.input.phase.definition().cell.a_angstrom,
        u_iso_angstrom2: result.input.phase.definition().u_iso_angstrom2[0],
    })
}

fn structural_rwp(
    input: &StructuralTofMultiBankInput,
    calculations: &[phasesmith_workflows::StructuralTofBankCalculation],
) -> Result<(f64, usize), NickelTofValidationError> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut included = 0;
    for (bank, calculation) in input.banks.iter().zip(calculations) {
        let observed = bank
            .pattern
            .observed_y
            .as_deref()
            .ok_or(NickelTofValidationError::MissingObservations)?;
        for sample in 0..observed.len() {
            if bank.pattern.mask.as_deref().is_none_or(|mask| mask[sample]) {
                let weight = bank
                    .pattern
                    .uncertainty
                    .as_deref()
                    .map_or(1.0, |sigma| 1.0 / sigma[sample].powi(2));
                numerator += weight * (calculation.y[sample] - observed[sample]).powi(2);
                denominator += weight * observed[sample].powi(2);
                included += 1;
            }
        }
    }
    let rwp = (numerator / denominator).sqrt();
    rwp.is_finite()
        .then_some((rwp, included))
        .ok_or_else(|| NickelTofValidationError::InvalidData("invalid structural Rwp".to_owned()))
}

fn pearson_correlation(
    observed: &[f64],
    background: &[f64],
    profile: &[f64],
    mask: Option<&[bool]>,
) -> Result<f64, NickelTofValidationError> {
    if observed.len() != background.len() || observed.len() != profile.len() {
        return Err(NickelTofValidationError::InvalidData(
            "nickel correlation arrays have different lengths".to_owned(),
        ));
    }
    let selected = (0..observed.len())
        .filter(|index| mask.is_none_or(|values| values[*index]))
        .map(|index| (observed[index] - background[index], profile[index]))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(NickelTofValidationError::InvalidData(
            "nickel correlation selection is empty".to_owned(),
        ));
    }
    let count = selected.len() as f64;
    let left_mean = selected.iter().map(|item| item.0).sum::<f64>() / count;
    let right_mean = selected.iter().map(|item| item.1).sum::<f64>() / count;
    let (cross, left_square, right_square) = selected.iter().fold(
        (0.0, 0.0, 0.0),
        |(cross, left_square, right_square), (left, right)| {
            let left = left - left_mean;
            let right = right - right_mean;
            (
                cross + left * right,
                left_square + left * left,
                right_square + right * right,
            )
        },
    );
    let value = cross / (left_square * right_square).sqrt();
    value.is_finite().then_some(value).ok_or_else(|| {
        NickelTofValidationError::InvalidData("invalid nickel profile correlation".to_owned())
    })
}

fn check(
    check_id: &str,
    passed: bool,
    detail: &str,
    measured: Option<f64>,
    criterion: &str,
) -> Result<ValidationCheck, ValidationContractError> {
    ValidationCheck::new(
        check_id,
        if passed {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        },
        detail,
        measured,
        Some(criterion.to_owned()),
    )
}

/// LANL nickel TOF setup, input, numerical, or report failure.
#[derive(Debug)]
pub enum NickelTofValidationError {
    /// Dataset checksum verification failed.
    Dataset(DatasetVerificationError),
    /// Typed TOF/domain validation failed.
    Domain(DomainError),
    /// Powder text import failed.
    Powder(PowderIoError),
    /// Instrument calibration import failed.
    Instrument(GsasTofInstrumentIoError),
    /// Space-group lookup failed.
    SpaceGroup(SpaceGroupLookupError),
    /// Reflection generation failed.
    Reflection(ReflectionGenerationError),
    /// Fixed background estimation failed.
    Background(BackgroundError),
    /// Incident-spectrum evaluation failed.
    IncidentSpectrum(TofIncidentSpectrumError),
    /// Native TOF workflow failed.
    Workflow(TofLeBailError),
    /// Joint multi-bank geometry workflow failed.
    Geometry(TofMultiBankGeometryError),
    /// Structural phase construction failed.
    Rietveld(RietveldError),
    /// Structural TOF request or products failed.
    Structural(StructuralTofMultiBankError),
    /// Structural TOF solver failed.
    StructuralRefinement(StructuralTofMultiBankRefinementError),
    /// Scalar parameter bounds failed validation.
    Parameter(ParameterError),
    /// Structural solver limits failed validation.
    Runtime(RuntimeError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
    /// The imported bank contains no observations.
    MissingObservations,
    /// Pinned records or returned arrays violate the gate contract.
    InvalidData(String),
    /// Stable report construction failed.
    Report(ValidationContractError),
}

impl Display for NickelTofValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(error) => Display::fmt(error, formatter),
            Self::Domain(error) => Display::fmt(error, formatter),
            Self::Powder(error) => Display::fmt(error, formatter),
            Self::Instrument(error) => Display::fmt(error, formatter),
            Self::SpaceGroup(error) => Display::fmt(error, formatter),
            Self::Reflection(error) => Display::fmt(error, formatter),
            Self::Background(error) => Display::fmt(error, formatter),
            Self::IncidentSpectrum(error) => Display::fmt(error, formatter),
            Self::Workflow(error) => Display::fmt(error, formatter),
            Self::Geometry(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Structural(error) => Display::fmt(error, formatter),
            Self::StructuralRefinement(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
            Self::MissingObservations => {
                formatter.write_str("LANL nickel observations are missing")
            }
            Self::InvalidData(message) => formatter.write_str(message),
            Self::Report(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for NickelTofValidationError {}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for NickelTofValidationError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

from_error!(DatasetVerificationError, Dataset);
from_error!(DomainError, Domain);
from_error!(PowderIoError, Powder);
from_error!(GsasTofInstrumentIoError, Instrument);
from_error!(SpaceGroupLookupError, SpaceGroup);
from_error!(ReflectionGenerationError, Reflection);
from_error!(BackgroundError, Background);
from_error!(TofIncidentSpectrumError, IncidentSpectrum);
from_error!(TofLeBailError, Workflow);
from_error!(TofMultiBankGeometryError, Geometry);
from_error!(RietveldError, Rietveld);
from_error!(StructuralTofMultiBankError, Structural);
from_error!(StructuralTofMultiBankRefinementError, StructuralRefinement);
from_error!(ParameterError, Parameter);
from_error!(RuntimeError, Runtime);
from_error!(ExecutionPolicyError, Execution);
from_error!(ValidationContractError, Report);
