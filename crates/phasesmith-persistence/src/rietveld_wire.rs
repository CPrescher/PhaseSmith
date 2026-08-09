//! Explicit wire records for native per-histogram Rietveld analyses.

use std::collections::BTreeMap;

use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{ProjectRecord, RecordId};
use phasesmith_workflows::{
    AffineConstraint, AmorphousBackground, AmorphousPeak, BackgroundModel, ChebyshevBackground,
    CompositeBackground, Constraint, DifferentiableBackground, FixedConstraint, LatticeBounds,
    LatticeParameterization, LatticeReflectionDomain, LinearConstraint, LinearTerm,
    ParameterChange, ParameterKey, PointBackground, PolynomialBackground, RefinementLimits,
    RietveldAnalysis, RietveldCalculationOptions, RietveldCovarianceOptions,
    RietveldGeneralCheckpoint, RietveldInstrumentParameter, RietveldIterationRecord,
    RietveldParameterLayout, RietveldParameterSelection, RietveldPhase, RietveldProjectState,
    RietveldRefinementOptions, RietveldSamplePhysicsModel, RietveldStructuralSelection,
    RietveldTopologyChange,
};
use serde::{Deserialize, Serialize};

use crate::{PersistenceError, ProjectReadLimits};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireRietveldAnalysis {
    histogram_id: String,
    phases: Vec<WirePhaseState>,
    background: Option<WireBackground>,
    selection: WireSelection,
    constraints: Vec<WireConstraint>,
    options: WireOptions,
    covariance: WireCovarianceOptions,
    checkpoint: Option<WireCheckpoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhaseState {
    phase_id: String,
    site_ids: Vec<String>,
    domain: Option<WireDomain>,
    sample_physics: Option<WireSamplePhysics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDomain {
    lower: Vec<f64>,
    upper: Vec<f64>,
    wavelength_angstrom: f64,
    visible_two_theta_deg: [f64; 2],
    initial_intensity: f64,
    merge_friedel: bool,
    max_candidates: usize,
    guard_scale: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireSamplePhysics {
    IsotropicSize {
        crystallite_size_nm: f64,
        shape_factor: f64,
    },
    IsotropicMicrostrain {
        rms_microstrain: f64,
    },
    MarchDollase {
        ratio: f64,
        preferred_axis_hkl: [f64; 3],
    },
    Composite {
        components: Vec<WireSamplePhysics>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireBackground {
    Polynomial {
        background_id: String,
        coefficients: Vec<f64>,
    },
    Chebyshev {
        background_id: String,
        coefficients: Vec<f64>,
        domain_deg: [f64; 2],
    },
    Point {
        background_id: String,
        knot_x: Vec<f64>,
        values: Vec<f64>,
    },
    Amorphous {
        background_id: String,
        peaks: Vec<[f64; 3]>,
    },
    Composite {
        background_id: String,
        components: Vec<WireBackground>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
// These flags intentionally mirror the independent public refinement selections.
#[allow(clippy::struct_excessive_bools)]
struct WireSelection {
    phase_scale: bool,
    lattice: bool,
    coordinates: bool,
    occupancy: bool,
    u_iso: bool,
    instrument: Vec<String>,
    background: bool,
    sample_physics: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireKey {
    module: String,
    owner_id: String,
    name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireConstraint {
    Fixed {
        target: WireKey,
        value: f64,
    },
    Affine {
        target: WireKey,
        source: WireKey,
        multiplier: f64,
        offset: f64,
    },
    Linear {
        target: WireKey,
        terms: Vec<WireLinearTerm>,
        offset: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLinearTerm {
    source: WireKey,
    coefficient: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOptions {
    support_fwhm: f64,
    use_uncertainty: bool,
    requested_threads: Option<usize>,
    minimum_parallel_tasks: usize,
    max_iterations: usize,
    max_evaluations: usize,
    max_runtime_seconds: Option<f64>,
    max_consecutive_rejections: usize,
    min_iterations: usize,
    objective_tolerance: f64,
    parameter_tolerance: f64,
    initial_damping: f64,
    damping_increase: f64,
    damping_decrease: f64,
    cg_tolerance: f64,
    max_cg_iterations: usize,
    max_scaled_parameter_step: f64,
    max_backtracks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCovarianceOptions {
    enabled: bool,
    max_parameters: usize,
    unresolved_correlation: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCheckpoint {
    completed_iterations: usize,
    parameters: Vec<WireParameterValue>,
    objective: f64,
    damping: f64,
    history: Vec<WireIteration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParameterValue {
    key: WireKey,
    value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireIteration {
    iteration: usize,
    objective: f64,
    objective_change: f64,
    scaled_step_norm: f64,
    damping: f64,
    cg_iterations: usize,
    backtracks: usize,
    parameter_changes: Vec<WireParameterChange>,
    topology_changes: Vec<WireTopologyChange>,
    rwp: f64,
    rp: f64,
    chi_square: f64,
    reduced_chi_square: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParameterChange {
    key: WireKey,
    before: f64,
    after: f64,
    scaled_change: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTopologyChange {
    phase_id: String,
    added_reflection_ids: Vec<String>,
    removed_reflection_ids: Vec<String>,
    preserved_reflection_count: usize,
}

pub(crate) fn encode_analyses(state: &RietveldProjectState) -> Vec<WireRietveldAnalysis> {
    state.analyses.iter().map(encode_analysis).collect()
}

fn encode_analysis(value: &RietveldAnalysis) -> WireRietveldAnalysis {
    WireRietveldAnalysis {
        histogram_id: value.histogram_id.as_str().to_owned(),
        phases: value
            .input
            .phases
            .iter()
            .map(|phase| WirePhaseState {
                phase_id: phase.phase_id().as_str().to_owned(),
                site_ids: phase
                    .site_ids()
                    .iter()
                    .map(|item| item.as_str().to_owned())
                    .collect(),
                domain: phase.reflection_domain().map(encode_domain),
                sample_physics: phase.sample_physics().map(encode_sample_physics),
            })
            .collect(),
        background: value.input.background.as_ref().map(encode_background),
        selection: encode_selection(&value.selection),
        constraints: value.constraints.iter().map(encode_constraint).collect(),
        options: encode_options(&value.options),
        covariance: WireCovarianceOptions {
            enabled: value.covariance.enabled,
            max_parameters: value.covariance.max_parameters,
            unresolved_correlation: value.covariance.unresolved_correlation,
        },
        checkpoint: value.checkpoint.as_ref().map(encode_checkpoint),
    }
}

fn encode_domain(value: &LatticeReflectionDomain) -> WireDomain {
    WireDomain {
        lower: value.bounds().lower().to_vec(),
        upper: value.bounds().upper().to_vec(),
        wavelength_angstrom: value.wavelength_angstrom(),
        visible_two_theta_deg: value.visible_two_theta_deg(),
        initial_intensity: value.initial_intensity(),
        merge_friedel: value.merge_friedel(),
        max_candidates: value.max_candidates(),
        guard_scale: value.guard_scale(),
    }
}

fn encode_sample_physics(value: &RietveldSamplePhysicsModel) -> WireSamplePhysics {
    match value {
        RietveldSamplePhysicsModel::IsotropicSize {
            crystallite_size_nm,
            shape_factor,
        } => WireSamplePhysics::IsotropicSize {
            crystallite_size_nm: *crystallite_size_nm,
            shape_factor: *shape_factor,
        },
        RietveldSamplePhysicsModel::IsotropicMicrostrain { rms_microstrain } => {
            WireSamplePhysics::IsotropicMicrostrain {
                rms_microstrain: *rms_microstrain,
            }
        }
        RietveldSamplePhysicsModel::MarchDollase {
            ratio,
            preferred_axis_hkl,
        } => WireSamplePhysics::MarchDollase {
            ratio: *ratio,
            preferred_axis_hkl: *preferred_axis_hkl,
        },
        RietveldSamplePhysicsModel::Composite(components) => WireSamplePhysics::Composite {
            components: components.iter().map(encode_sample_physics).collect(),
        },
    }
}

fn encode_background(value: &BackgroundModel) -> WireBackground {
    let background_id = value.background_id().to_owned();
    match value {
        BackgroundModel::Polynomial(value) => WireBackground::Polynomial {
            background_id,
            coefficients: value.coefficients(),
        },
        BackgroundModel::Chebyshev(value) => WireBackground::Chebyshev {
            background_id,
            coefficients: value.coefficients(),
            domain_deg: value.domain_deg(),
        },
        BackgroundModel::Point(value) => WireBackground::Point {
            background_id,
            knot_x: value.knot_x().to_vec(),
            values: value.coefficients(),
        },
        BackgroundModel::Amorphous(value) => WireBackground::Amorphous {
            background_id,
            peaks: value
                .peaks()
                .iter()
                .map(|peak| [peak.area(), peak.center_deg(), peak.fwhm_deg()])
                .collect(),
        },
        BackgroundModel::Composite(value) => WireBackground::Composite {
            background_id,
            components: value.components().iter().map(encode_background).collect(),
        },
    }
}

fn encode_selection(value: &RietveldParameterSelection) -> WireSelection {
    WireSelection {
        phase_scale: value.structural.phase_scale,
        lattice: value.structural.lattice,
        coordinates: value.structural.coordinates,
        occupancy: value.structural.occupancy,
        u_iso: value.structural.u_iso,
        instrument: value
            .instrument
            .iter()
            .map(|item| item.as_str().to_owned())
            .collect(),
        background: value.background,
        sample_physics: value.sample_physics,
    }
}

fn encode_key(value: &ParameterKey) -> WireKey {
    WireKey {
        module: value.module().to_owned(),
        owner_id: value.owner_id().to_owned(),
        name: value.name().to_owned(),
    }
}

fn encode_constraint(value: &Constraint) -> WireConstraint {
    match value {
        Constraint::Fixed(value) => WireConstraint::Fixed {
            target: encode_key(value.target()),
            value: value.value(),
        },
        Constraint::Affine(value) => WireConstraint::Affine {
            target: encode_key(value.target()),
            source: encode_key(value.source()),
            multiplier: value.multiplier(),
            offset: value.offset(),
        },
        Constraint::Linear(value) => WireConstraint::Linear {
            target: encode_key(value.target()),
            terms: value
                .terms()
                .iter()
                .map(|term| WireLinearTerm {
                    source: encode_key(term.source()),
                    coefficient: term.coefficient(),
                })
                .collect(),
            offset: value.offset(),
        },
    }
}

fn encode_options(value: &RietveldRefinementOptions) -> WireOptions {
    WireOptions {
        support_fwhm: value.calculation.support_fwhm,
        use_uncertainty: value.calculation.use_uncertainty,
        requested_threads: value.calculation.execution.requested_threads(),
        minimum_parallel_tasks: value.calculation.execution.minimum_parallel_tasks(),
        max_iterations: value.limits.max_iterations(),
        max_evaluations: value.limits.max_evaluations(),
        max_runtime_seconds: value.limits.max_runtime_seconds(),
        max_consecutive_rejections: value.limits.max_consecutive_rejections(),
        min_iterations: value.min_iterations,
        objective_tolerance: value.objective_tolerance,
        parameter_tolerance: value.parameter_tolerance,
        initial_damping: value.initial_damping,
        damping_increase: value.damping_increase,
        damping_decrease: value.damping_decrease,
        cg_tolerance: value.cg_tolerance,
        max_cg_iterations: value.max_cg_iterations,
        max_scaled_parameter_step: value.max_scaled_parameter_step,
        max_backtracks: value.max_backtracks,
    }
}

fn encode_checkpoint(value: &RietveldGeneralCheckpoint) -> WireCheckpoint {
    WireCheckpoint {
        completed_iterations: value.completed_iterations,
        parameters: value
            .parameters
            .specs()
            .iter()
            .map(|spec| WireParameterValue {
                key: encode_key(spec.key()),
                value: spec.value(),
            })
            .collect(),
        objective: value.objective,
        damping: value.damping,
        history: value.history.iter().map(encode_iteration).collect(),
    }
}

fn encode_iteration(value: &RietveldIterationRecord) -> WireIteration {
    WireIteration {
        iteration: value.iteration,
        objective: value.objective,
        objective_change: value.objective_change,
        scaled_step_norm: value.scaled_step_norm,
        damping: value.damping,
        cg_iterations: value.cg_iterations,
        backtracks: value.backtracks,
        parameter_changes: value
            .parameter_changes
            .iter()
            .map(|change| WireParameterChange {
                key: encode_key(&change.key),
                before: change.before,
                after: change.after,
                scaled_change: change.scaled_change,
            })
            .collect(),
        topology_changes: value
            .topology_changes
            .iter()
            .map(|change| WireTopologyChange {
                phase_id: change.phase_id.as_str().to_owned(),
                added_reflection_ids: change.added_reflection_ids.clone(),
                removed_reflection_ids: change.removed_reflection_ids.clone(),
                preserved_reflection_count: change.preserved_reflection_count,
            })
            .collect(),
        rwp: value.rwp,
        rp: value.rp,
        chi_square: value.chi_square,
        reduced_chi_square: value.reduced_chi_square,
    }
}

pub(crate) fn decode_state(
    project: ProjectRecord,
    analyses: Vec<WireRietveldAnalysis>,
    limits: ProjectReadLimits,
) -> Result<RietveldProjectState, PersistenceError> {
    check_count(
        analyses.len(),
        limits.max_histograms,
        "Rietveld analysis count exceeds max_histograms",
    )?;
    let analyses = analyses
        .into_iter()
        .map(|value| decode_analysis(&project, value, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let state = RietveldProjectState { project, analyses };
    state
        .validate()
        .map_err(|error| invalid(format!("invalid native Rietveld project state: {error}")))?;
    Ok(state)
}

#[allow(clippy::too_many_lines)]
fn decode_analysis(
    project: &ProjectRecord,
    value: WireRietveldAnalysis,
    limits: ProjectReadLimits,
) -> Result<RietveldAnalysis, PersistenceError> {
    check_count(
        value.phases.len(),
        limits.max_phases,
        "Rietveld phase-state count exceeds max_phases",
    )?;
    check_count(
        value.constraints.len(),
        limits.max_array_elements,
        "Rietveld constraint count exceeds max_array_elements",
    )?;
    if let Some(checkpoint) = &value.checkpoint {
        check_count(
            checkpoint.parameters.len(),
            limits.max_array_elements,
            "Rietveld checkpoint parameter count exceeds max_array_elements",
        )?;
        check_count(
            checkpoint.history.len(),
            limits.max_array_elements,
            "Rietveld checkpoint history exceeds max_array_elements",
        )?;
    }
    let histogram_id = RecordId::new(value.histogram_id)
        .map_err(|error| invalid(format!("invalid analysis histogram ID: {error}")))?;
    let histogram = project
        .histograms
        .iter()
        .find(|item| item.histogram_id == histogram_id)
        .ok_or_else(|| {
            invalid(format!(
                "analysis references unknown histogram {histogram_id}"
            ))
        })?;
    let mut phases = Vec::with_capacity(value.phases.len());
    let mut lattice_bounds = Vec::with_capacity(value.phases.len());
    for phase_state in value.phases {
        let phase_id = RecordId::new(phase_state.phase_id)
            .map_err(|error| invalid(format!("invalid analysis phase ID: {error}")))?;
        let stored = project
            .phases
            .iter()
            .find(|item| item.phase_id == phase_id)
            .ok_or_else(|| invalid(format!("analysis references unknown phase {phase_id}")))?;
        let site_ids = phase_state
            .site_ids
            .into_iter()
            .map(|item| {
                RecordId::new(item)
                    .map_err(|error| invalid(format!("invalid analysis site ID: {error}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (mut phase, bounds) = if let Some(domain) = phase_state.domain {
            let parameterization = LatticeParameterization::new(
                stored.definition.space_group.clone(),
                stored.definition.cell,
            )
            .map_err(|error| invalid(format!("invalid lattice parameterization: {error}")))?;
            let bounds = LatticeBounds::new(&parameterization, domain.lower, domain.upper)
                .map_err(|error| invalid(format!("invalid lattice bounds: {error}")))?;
            let native_domain = LatticeReflectionDomain::new(
                parameterization,
                bounds.clone(),
                domain.wavelength_angstrom,
                domain.visible_two_theta_deg,
                domain.initial_intensity,
                domain.merge_friedel,
                domain.max_candidates,
                domain.guard_scale,
            )
            .map_err(|error| invalid(format!("invalid reflection domain: {error}")))?;
            (
                RietveldPhase::from_lattice_domain(
                    phase_id,
                    stored.name.clone(),
                    site_ids,
                    stored.definition.clone(),
                    native_domain,
                )
                .map_err(|error| invalid(format!("invalid dynamic phase: {error}")))?,
                Some(bounds),
            )
        } else {
            let count = stored.definition.hkl.len();
            (
                RietveldPhase::new_with_site_ids(
                    phase_id,
                    stored.name.clone(),
                    site_ids,
                    stored.definition.clone(),
                    phasesmith_core::OwnedCwContributions::neutral(count),
                )
                .map_err(|error| invalid(format!("invalid fixed phase: {error}")))?,
                None,
            )
        };
        if let Some(model) = phase_state.sample_physics {
            phase = phase.with_sample_physics(decode_sample_physics(model));
        }
        phases.push(phase);
        lattice_bounds.push(bounds);
    }
    let background = value.background.map(decode_background).transpose()?;
    let input = match (&histogram.experiment.radiation, background) {
        (phasesmith_model::RadiationDefinition::Monochromatic { .. }, Some(background)) => {
            phasesmith_workflows::RietveldInput::new_with_background(
                histogram.pattern.clone(),
                histogram.experiment.instrument,
                histogram.experiment.axial_geometry,
                histogram.experiment.position_correction,
                background,
                phases,
            )
        }
        (phasesmith_model::RadiationDefinition::Monochromatic { .. }, None) => {
            phasesmith_workflows::RietveldInput::new(
                histogram.pattern.clone(),
                histogram.experiment.instrument,
                histogram.experiment.axial_geometry,
                histogram.experiment.position_correction,
                phases,
            )
        }
        (
            phasesmith_model::RadiationDefinition::FixedSpectrum { spectrum, .. },
            Some(background),
        ) => phasesmith_workflows::RietveldInput::new_fixed_spectrum_with_background(
            histogram.pattern.clone(),
            histogram.experiment.instrument,
            spectrum.clone(),
            histogram.experiment.axial_geometry,
            histogram.experiment.position_correction,
            background,
            phases,
        ),
        (phasesmith_model::RadiationDefinition::FixedSpectrum { spectrum, .. }, None) => {
            phasesmith_workflows::RietveldInput::new_fixed_spectrum(
                histogram.pattern.clone(),
                histogram.experiment.instrument,
                spectrum.clone(),
                histogram.experiment.axial_geometry,
                histogram.experiment.position_correction,
                phases,
            )
        }
    }
    .map_err(|error| invalid(format!("invalid analysis input: {error}")))?;
    let selection = decode_selection(value.selection)?;
    let constraints = value
        .constraints
        .into_iter()
        .map(decode_constraint)
        .collect::<Result<Vec<_>, _>>()?;
    let options = decode_options(&value.options)?;
    let covariance = RietveldCovarianceOptions::new(
        value.covariance.enabled,
        value.covariance.max_parameters,
        value.covariance.unresolved_correlation,
    )
    .map_err(|error| invalid(format!("invalid covariance options: {error}")))?;
    let checkpoint = value
        .checkpoint
        .map(|checkpoint| {
            decode_checkpoint(
                &input,
                &selection,
                &lattice_bounds,
                &constraints,
                checkpoint,
            )
        })
        .transpose()?;
    Ok(RietveldAnalysis {
        histogram_id,
        input,
        selection,
        lattice_bounds,
        constraints,
        options,
        covariance,
        checkpoint,
    })
}

fn check_count(
    actual: usize,
    maximum: usize,
    message: &'static str,
) -> Result<(), PersistenceError> {
    if actual > maximum {
        return Err(PersistenceError::LimitExceeded {
            message: message.to_owned(),
        });
    }
    Ok(())
}

fn decode_sample_physics(value: WireSamplePhysics) -> RietveldSamplePhysicsModel {
    match value {
        WireSamplePhysics::IsotropicSize {
            crystallite_size_nm,
            shape_factor,
        } => RietveldSamplePhysicsModel::IsotropicSize {
            crystallite_size_nm,
            shape_factor,
        },
        WireSamplePhysics::IsotropicMicrostrain { rms_microstrain } => {
            RietveldSamplePhysicsModel::IsotropicMicrostrain { rms_microstrain }
        }
        WireSamplePhysics::MarchDollase {
            ratio,
            preferred_axis_hkl,
        } => RietveldSamplePhysicsModel::MarchDollase {
            ratio,
            preferred_axis_hkl,
        },
        WireSamplePhysics::Composite { components } => RietveldSamplePhysicsModel::Composite(
            components.into_iter().map(decode_sample_physics).collect(),
        ),
    }
}

fn decode_background(value: WireBackground) -> Result<BackgroundModel, PersistenceError> {
    let result = match value {
        WireBackground::Polynomial {
            background_id,
            coefficients,
        } => BackgroundModel::Polynomial(
            PolynomialBackground::new(background_id, coefficients)
                .map_err(|error| invalid(format!("invalid polynomial background: {error}")))?,
        ),
        WireBackground::Chebyshev {
            background_id,
            coefficients,
            domain_deg,
        } => BackgroundModel::Chebyshev(
            ChebyshevBackground::new(background_id, coefficients, domain_deg)
                .map_err(|error| invalid(format!("invalid Chebyshev background: {error}")))?,
        ),
        WireBackground::Point {
            background_id,
            knot_x,
            values,
        } => BackgroundModel::Point(
            PointBackground::new(background_id, knot_x, values)
                .map_err(|error| invalid(format!("invalid point background: {error}")))?,
        ),
        WireBackground::Amorphous {
            background_id,
            peaks,
        } => BackgroundModel::Amorphous(
            AmorphousBackground::new(
                background_id,
                peaks
                    .into_iter()
                    .map(|peak| {
                        AmorphousPeak::new(peak[0], peak[1], peak[2])
                            .map_err(|error| invalid(format!("invalid amorphous peak: {error}")))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .map_err(|error| invalid(format!("invalid amorphous background: {error}")))?,
        ),
        WireBackground::Composite {
            background_id,
            components,
        } => BackgroundModel::Composite(
            CompositeBackground::new(
                background_id,
                components
                    .into_iter()
                    .map(decode_background)
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .map_err(|error| invalid(format!("invalid composite background: {error}")))?,
        ),
    };
    Ok(result)
}

fn decode_selection(value: WireSelection) -> Result<RietveldParameterSelection, PersistenceError> {
    RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: value.phase_scale,
            lattice: value.lattice,
            coordinates: value.coordinates,
            occupancy: value.occupancy,
            u_iso: value.u_iso,
        },
        value
            .instrument
            .into_iter()
            .map(|item| decode_instrument_parameter(&item))
            .collect::<Result<Vec<_>, _>>()?,
        value.background,
        value.sample_physics,
    )
    .map_err(|error| invalid(format!("invalid parameter selection: {error}")))
}

fn decode_instrument_parameter(
    value: &str,
) -> Result<RietveldInstrumentParameter, PersistenceError> {
    Ok(match value {
        "u_deg2" => RietveldInstrumentParameter::UDeg2,
        "v_deg2" => RietveldInstrumentParameter::VDeg2,
        "w_deg2" => RietveldInstrumentParameter::WDeg2,
        "x_deg" => RietveldInstrumentParameter::XDeg,
        "y_deg" => RietveldInstrumentParameter::YDeg,
        "wavelength_angstrom" => RietveldInstrumentParameter::WavelengthAngstrom,
        "zero_shift_deg" => RietveldInstrumentParameter::ZeroShiftDeg,
        "sample_displacement_mm" => RietveldInstrumentParameter::SampleDisplacementMm,
        "displace_x_micrometre" => RietveldInstrumentParameter::DisplaceXMicrometre,
        "displace_y_micrometre" => RietveldInstrumentParameter::DisplaceYMicrometre,
        _ => {
            return Err(invalid(format!(
                "unknown Rietveld instrument parameter {value:?}"
            )));
        }
    })
}

fn decode_key(value: WireKey) -> Result<ParameterKey, PersistenceError> {
    ParameterKey::new(value.module, value.owner_id, value.name)
        .map_err(|error| invalid(format!("invalid parameter key: {error}")))
}

fn decode_constraint(value: WireConstraint) -> Result<Constraint, PersistenceError> {
    Ok(match value {
        WireConstraint::Fixed { target, value } => Constraint::Fixed(
            FixedConstraint::new(decode_key(target)?, value)
                .map_err(|error| invalid(format!("invalid fixed constraint: {error}")))?,
        ),
        WireConstraint::Affine {
            target,
            source,
            multiplier,
            offset,
        } => Constraint::Affine(
            AffineConstraint::new(decode_key(target)?, decode_key(source)?, multiplier, offset)
                .map_err(|error| invalid(format!("invalid affine constraint: {error}")))?,
        ),
        WireConstraint::Linear {
            target,
            terms,
            offset,
        } => Constraint::Linear(
            LinearConstraint::new(
                decode_key(target)?,
                terms
                    .into_iter()
                    .map(|term| {
                        LinearTerm::new(decode_key(term.source)?, term.coefficient)
                            .map_err(|error| invalid(format!("invalid linear term: {error}")))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                offset,
            )
            .map_err(|error| invalid(format!("invalid linear constraint: {error}")))?,
        ),
    })
}

fn decode_options(value: &WireOptions) -> Result<RietveldRefinementOptions, PersistenceError> {
    let execution = ExecutionPolicy::new(value.requested_threads, value.minimum_parallel_tasks)
        .map_err(|error| invalid(format!("invalid execution policy: {error}")))?;
    let calculation =
        RietveldCalculationOptions::new(value.support_fwhm, value.use_uncertainty, execution)
            .map_err(|error| invalid(format!("invalid calculation options: {error}")))?;
    let limits = RefinementLimits::new(
        value.max_iterations,
        value.max_evaluations,
        value.max_runtime_seconds,
        value.max_consecutive_rejections,
    )
    .map_err(|error| invalid(format!("invalid refinement limits: {error}")))?;
    RietveldRefinementOptions::new(
        calculation,
        limits,
        value.min_iterations,
        value.objective_tolerance,
        value.parameter_tolerance,
        value.initial_damping,
        value.damping_increase,
        value.damping_decrease,
        value.cg_tolerance,
        value.max_cg_iterations,
        value.max_scaled_parameter_step,
        value.max_backtracks,
    )
    .map_err(|error| invalid(format!("invalid Rietveld options: {error}")))
}

fn decode_checkpoint(
    input: &phasesmith_workflows::RietveldInput,
    selection: &RietveldParameterSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    constraints: &[Constraint],
    value: WireCheckpoint,
) -> Result<RietveldGeneralCheckpoint, PersistenceError> {
    let layout = RietveldParameterLayout::new(input, selection, lattice_bounds)
        .map_err(|error| invalid(format!("invalid checkpoint layout: {error}")))?;
    if value.parameters.len() != layout.parameters().specs().len() {
        return Err(invalid("checkpoint parameter count changed".to_owned()));
    }
    let mut replacements = BTreeMap::new();
    for (stored, spec) in value
        .parameters
        .into_iter()
        .zip(layout.parameters().specs())
    {
        let key = decode_key(stored.key)?;
        if &key != spec.key() {
            return Err(invalid("checkpoint parameter order changed".to_owned()));
        }
        replacements.insert(key, stored.value);
    }
    let parameters = layout
        .parameters()
        .replace_values(&replacements)
        .map_err(|error| invalid(format!("invalid checkpoint parameters: {error}")))?;
    let accepted_values = parameters
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let accepted_input = layout
        .apply_values(input, &accepted_values)
        .map_err(|error| invalid(format!("invalid checkpoint accepted state: {error}")))?;
    let history = value
        .history
        .into_iter()
        .map(decode_iteration)
        .collect::<Result<Vec<_>, _>>()?;
    let checkpoint = RietveldGeneralCheckpoint {
        completed_iterations: value.completed_iterations,
        input: accepted_input,
        selection: selection.clone(),
        lattice_bounds: lattice_bounds.to_vec(),
        constraints: constraints.to_vec(),
        parameters,
        objective: value.objective,
        damping: value.damping,
        history,
    };
    checkpoint
        .validate_for(input, selection, lattice_bounds, constraints)
        .map_err(|error| invalid(format!("invalid checkpoint: {error}")))?;
    Ok(checkpoint)
}

fn decode_iteration(value: WireIteration) -> Result<RietveldIterationRecord, PersistenceError> {
    Ok(RietveldIterationRecord {
        iteration: value.iteration,
        objective: value.objective,
        objective_change: value.objective_change,
        scaled_step_norm: value.scaled_step_norm,
        damping: value.damping,
        cg_iterations: value.cg_iterations,
        backtracks: value.backtracks,
        parameter_changes: value
            .parameter_changes
            .into_iter()
            .map(|change| {
                Ok(ParameterChange {
                    key: decode_key(change.key)?,
                    before: change.before,
                    after: change.after,
                    scaled_change: change.scaled_change,
                })
            })
            .collect::<Result<Vec<_>, PersistenceError>>()?,
        topology_changes: value
            .topology_changes
            .into_iter()
            .map(|change| {
                Ok(RietveldTopologyChange {
                    phase_id: RecordId::new(change.phase_id)
                        .map_err(|error| invalid(format!("invalid topology phase ID: {error}")))?,
                    added_reflection_ids: change.added_reflection_ids,
                    removed_reflection_ids: change.removed_reflection_ids,
                    preserved_reflection_count: change.preserved_reflection_count,
                })
            })
            .collect::<Result<Vec<_>, PersistenceError>>()?,
        rwp: value.rwp,
        rp: value.rp,
        chi_square: value.chi_square,
        reduced_chi_square: value.reduced_chi_square,
    })
}

fn invalid(message: String) -> PersistenceError {
    PersistenceError::InvalidRecord { message }
}
