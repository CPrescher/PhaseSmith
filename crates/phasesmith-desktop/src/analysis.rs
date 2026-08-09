//! Explicit desktop authoring boundary for runnable native analyses.

use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::RecordId;
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, RefinementLimits, RietveldAnalysis,
    RietveldCalculationOptions, RietveldCovarianceOptions, RietveldInstrumentParameter,
    RietveldParameterLayout, RietveldParameterSelection, RietveldRefinementOptions,
    RietveldStructuralSelection,
};
use serde::{Deserialize, Serialize};

use crate::calculation::calculation_input;
use crate::{DesktopError, DesktopErrorCode, DesktopProjectStore};

/// Stable desktop names for refinable instrument and position parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopInstrumentParameter {
    /// Gaussian Caglioti U in square degrees.
    UDeg2,
    /// Gaussian Caglioti V in square degrees.
    VDeg2,
    /// Gaussian Caglioti W in square degrees.
    WDeg2,
    /// Lorentzian X in degrees.
    XDeg,
    /// Lorentzian Y in degrees.
    YDeg,
    /// Monochromatic wavelength in angstroms.
    WavelengthAngstrom,
    /// Constant two-theta offset in degrees.
    ZeroShiftDeg,
    /// Bragg--Brentano sample displacement in millimetres.
    SampleDisplacementMm,
    /// Debye--Scherrer X displacement in micrometres.
    DisplaceXMicrometre,
    /// Debye--Scherrer Y displacement in micrometres.
    DisplaceYMicrometre,
}

impl From<DesktopInstrumentParameter> for RietveldInstrumentParameter {
    fn from(value: DesktopInstrumentParameter) -> Self {
        match value {
            DesktopInstrumentParameter::UDeg2 => Self::UDeg2,
            DesktopInstrumentParameter::VDeg2 => Self::VDeg2,
            DesktopInstrumentParameter::WDeg2 => Self::WDeg2,
            DesktopInstrumentParameter::XDeg => Self::XDeg,
            DesktopInstrumentParameter::YDeg => Self::YDeg,
            DesktopInstrumentParameter::WavelengthAngstrom => Self::WavelengthAngstrom,
            DesktopInstrumentParameter::ZeroShiftDeg => Self::ZeroShiftDeg,
            DesktopInstrumentParameter::SampleDisplacementMm => Self::SampleDisplacementMm,
            DesktopInstrumentParameter::DisplaceXMicrometre => Self::DisplaceXMicrometre,
            DesktopInstrumentParameter::DisplaceYMicrometre => Self::DisplaceYMicrometre,
        }
    }
}

/// Caller-owned parameter-family selection for a new analysis.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
#[allow(clippy::struct_excessive_bools)]
pub struct AnalysisSelectionInput {
    /// Refine symmetry-independent lattice variables.
    pub lattice: bool,
    /// Refine symmetry-allowed site coordinates.
    pub coordinates: bool,
    /// Refine site occupancies.
    pub occupancy: bool,
    /// Refine isotropic displacement values.
    pub u_iso: bool,
    /// Refine one scale per phase.
    pub phase_scale: bool,
    /// Ordered unique instrument and position parameters.
    pub instrument: Vec<DesktopInstrumentParameter>,
    /// Refine all coefficients of the attached analytical background.
    pub background: bool,
    /// Refine supported per-phase sample-physics parameters.
    pub sample_physics: bool,
}

impl Default for AnalysisSelectionInput {
    fn default() -> Self {
        Self {
            lattice: false,
            coordinates: false,
            occupancy: false,
            u_iso: false,
            phase_scale: true,
            instrument: Vec::new(),
            background: false,
            sample_physics: false,
        }
    }
}

impl AnalysisSelectionInput {
    fn native(&self) -> Result<RietveldParameterSelection, DesktopError> {
        RietveldParameterSelection::new(
            RietveldStructuralSelection {
                lattice: self.lattice,
                coordinates: self.coordinates,
                occupancy: self.occupancy,
                u_iso: self.u_iso,
                phase_scale: self.phase_scale,
            },
            self.instrument.iter().copied().map(Into::into).collect(),
            self.background,
            self.sample_physics,
        )
        .map_err(invalid_analysis)
    }
}

/// Bounded calculation, solver, and covariance controls for a new analysis.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct AnalysisSolverInput {
    /// Finite profile support in multiples of FWHM.
    pub support_fwhm: f64,
    /// Use supplied one-sigma uncertainties in the objective.
    pub use_uncertainty: bool,
    /// Requested worker count, or automatic when absent.
    pub threads: Option<usize>,
    /// Minimum phase count before parallel scheduling.
    pub minimum_parallel_phases: usize,
    /// Maximum attempted iterations.
    pub max_iterations: usize,
    /// Maximum model evaluations.
    pub max_evaluations: usize,
    /// Optional wall-clock ceiling in seconds.
    pub max_runtime_seconds: Option<f64>,
    /// Maximum consecutive rejected steps.
    pub max_consecutive_rejections: usize,
    /// Minimum accepted iterations before convergence.
    pub min_iterations: usize,
    /// Relative objective-change tolerance.
    pub objective_tolerance: f64,
    /// Scaled parameter-step tolerance.
    pub parameter_tolerance: f64,
    /// Initial Levenberg damping.
    pub initial_damping: f64,
    /// Damping multiplier after rejection.
    pub damping_increase: f64,
    /// Damping multiplier after acceptance.
    pub damping_decrease: f64,
    /// Relative conjugate-gradient residual tolerance.
    pub cg_tolerance: f64,
    /// Maximum conjugate-gradient iterations per attempted step.
    pub max_cg_iterations: usize,
    /// Maximum scaled Euclidean parameter step.
    pub max_scaled_parameter_step: f64,
    /// Maximum half-step backtracks after a full trial.
    pub max_backtracks: usize,
    /// Evaluate final covariance diagnostics.
    pub covariance_enabled: bool,
    /// Maximum free dimension for explicit covariance diagnostics.
    pub covariance_max_parameters: usize,
    /// Absolute correlation treated as unresolved.
    pub unresolved_correlation: f64,
}

impl Default for AnalysisSolverInput {
    fn default() -> Self {
        let covariance = RietveldCovarianceOptions::default();
        Self {
            support_fwhm: 20.0,
            use_uncertainty: true,
            threads: Some(2),
            minimum_parallel_phases: 2,
            max_iterations: 100,
            max_evaluations: 1_000,
            max_runtime_seconds: None,
            max_consecutive_rejections: 20,
            min_iterations: 1,
            objective_tolerance: 1.0e-10,
            parameter_tolerance: 1.0e-7,
            initial_damping: 1.0e-6,
            damping_increase: 10.0,
            damping_decrease: 0.3,
            cg_tolerance: 1.0e-6,
            max_cg_iterations: 30,
            max_scaled_parameter_step: 1.0,
            max_backtracks: 10,
            covariance_enabled: covariance.enabled,
            covariance_max_parameters: covariance.max_parameters,
            unresolved_correlation: covariance.unresolved_correlation,
        }
    }
}

impl AnalysisSolverInput {
    fn native(
        &self,
    ) -> Result<(RietveldRefinementOptions, RietveldCovarianceOptions), DesktopError> {
        let execution = ExecutionPolicy::new(self.threads, self.minimum_parallel_phases)
            .map_err(invalid_analysis)?;
        let calculation =
            RietveldCalculationOptions::new(self.support_fwhm, self.use_uncertainty, execution)
                .map_err(invalid_analysis)?;
        let limits = RefinementLimits::new(
            self.max_iterations,
            self.max_evaluations,
            self.max_runtime_seconds,
            self.max_consecutive_rejections,
        )
        .map_err(invalid_analysis)?;
        let options = RietveldRefinementOptions::new(
            calculation,
            limits,
            self.min_iterations,
            self.objective_tolerance,
            self.parameter_tolerance,
            self.initial_damping,
            self.damping_increase,
            self.damping_decrease,
            self.cg_tolerance,
            self.max_cg_iterations,
            self.max_scaled_parameter_step,
            self.max_backtracks,
        )
        .map_err(invalid_analysis)?;
        let covariance = RietveldCovarianceOptions::new(
            self.covariance_enabled,
            self.covariance_max_parameters,
            self.unresolved_correlation,
        )
        .map_err(invalid_analysis)?;
        Ok((options, covariance))
    }
}

/// Create one runnable analysis for an existing histogram and its attached phases.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CreateAnalysisRequest {
    /// Existing histogram receiving the analysis.
    pub histogram_id: String,
    /// Parameter families to refine.
    #[serde(default)]
    pub selection: AnalysisSelectionInput,
    /// Calculation, solver, and covariance controls.
    #[serde(default)]
    pub solver: AnalysisSolverInput,
    /// Relative half-width for generated lattice-length bounds.
    #[serde(default = "default_lattice_relative_length")]
    pub lattice_relative_length: f64,
    /// Absolute half-width for generated lattice-angle bounds.
    #[serde(default = "default_lattice_angle_delta")]
    pub lattice_angle_delta_deg: f64,
}

/// Successful analysis creation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CreateAnalysisResponse {
    /// New project revision.
    pub revision: u64,
    /// Histogram owning the analysis.
    pub histogram_id: String,
    /// Number of attached phases.
    pub phase_count: usize,
    /// Number of physical parameters in the resulting layout.
    pub parameter_count: usize,
}

impl DesktopProjectStore {
    /// Validate and install a runnable native analysis using an exact project snapshot.
    ///
    /// # Errors
    ///
    /// Returns a project/revision, input, parameter-layout, execution-policy,
    /// solver-control, covariance, or shared-state error.
    pub fn create_analysis(
        &self,
        expected_revision: u64,
        request: &CreateAnalysisRequest,
    ) -> Result<CreateAnalysisResponse, DesktopError> {
        let starting = self.snapshot()?;
        if starting.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                starting.revision(),
            ));
        }
        let histogram_id = RecordId::new(&request.histogram_id).map_err(invalid_analysis)?;
        if starting
            .state()
            .analyses
            .iter()
            .any(|analysis| analysis.histogram_id == histogram_id)
        {
            return Err(invalid_analysis(format!(
                "histogram {histogram_id} already has a native analysis"
            )));
        }
        let input = calculation_input(&starting, &histogram_id)?;
        if input.phases.is_empty() {
            return Err(invalid_analysis(
                "an analysis requires at least one phase attached to the histogram",
            ));
        }
        let selection = request.selection.native()?;
        let lattice_bounds = input
            .phases
            .iter()
            .map(|phase| {
                if selection.structural.lattice {
                    let parameterization = LatticeParameterization::new(
                        phase.definition().space_group.clone(),
                        phase.definition().cell,
                    )
                    .map_err(invalid_analysis)?;
                    LatticeBounds::around(
                        &parameterization,
                        request.lattice_relative_length,
                        request.lattice_angle_delta_deg,
                    )
                    .map(Some)
                    .map_err(invalid_analysis)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>, DesktopError>>()?;
        let (options, covariance) = request.solver.native()?;
        let analysis = RietveldAnalysis {
            histogram_id: histogram_id.clone(),
            input,
            selection,
            lattice_bounds,
            constraints: Vec::new(),
            options,
            covariance,
            checkpoint: None,
        };
        analysis.validate().map_err(invalid_analysis)?;
        let parameter_count = RietveldParameterLayout::new(
            &analysis.input,
            &analysis.selection,
            &analysis.lattice_bounds,
        )
        .map_err(invalid_analysis)?
        .parameters()
        .specs()
        .len();
        let phase_count = analysis.input.phases.len();
        let mut next = starting.state().clone();
        next.analyses.push(analysis);
        let installed = self.replace_snapshot(&starting, next)?;
        Ok(CreateAnalysisResponse {
            revision: installed.revision(),
            histogram_id: histogram_id.as_str().to_owned(),
            phase_count,
            parameter_count,
        })
    }
}

const fn default_lattice_relative_length() -> f64 {
    0.05
}

const fn default_lattice_angle_delta() -> f64 {
    5.0
}

fn invalid_analysis(error: impl std::fmt::Display) -> DesktopError {
    DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
}
