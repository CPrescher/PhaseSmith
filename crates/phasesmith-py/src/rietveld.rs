//! Thin Python adapter for the application-neutral native Rietveld workflow.

use std::collections::BTreeMap;

use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use phasesmith_core::OwnedCwContributions;
use phasesmith_execution::ExecutionPolicy as NativeExecutionPolicyModel;
use phasesmith_model::{
    ExperimentRecord, HistogramRecord, PatternRecord, ProjectRecord, RadiationDefinition,
    RadiationProbe, RecordId, StructuralPhaseRecord,
};
use phasesmith_persistence::{
    ProjectReadLimits, ProjectSaveOptions, load_rietveld_project, save_rietveld_project,
};
use phasesmith_workflows::{
    AffineConstraint, AmorphousBackground, AmorphousPeak, BackgroundModel, CancellationToken,
    ChebyshevBackground, CompositeBackground, Constraint, FixedConstraint, LatticeBounds,
    LatticeParameterization, LatticeReflectionDomain, LinearConstraint, LinearTerm, ParameterKey,
    ParameterSet, PointBackground, PolynomialBackground, RefinementLimits, RietveldAnalysis,
    RietveldCalculationOptions, RietveldCovarianceOptions, RietveldGeneralCheckpoint,
    RietveldGeneralRefinementResult, RietveldInput, RietveldInstrumentParameter,
    RietveldParameterSelection, RietveldPhase, RietveldProjectState, RietveldRefinementOptions,
    RietveldSamplePhysicsModel, RietveldStructuralSelection, refine_general_rietveld,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use super::{
    NativeExecutionPolicy, NativeStructuralPhase, axial_geometry, bool_slice, contiguous_slice,
    cw_instrument, position_correction,
};

type PyParameterRecord = (String, String, String, f64, String, f64, f64, f64, bool);
type PyKeyRecord = (String, String, String);
type PyCorrelationRecord = (PyKeyRecord, PyKeyRecord, f64);

fn parameter_records(parameters: &ParameterSet) -> Vec<PyParameterRecord> {
    parameters
        .specs()
        .iter()
        .map(|spec| {
            (
                spec.key().module().to_owned(),
                spec.key().owner_id().to_owned(),
                spec.key().name().to_owned(),
                spec.value(),
                spec.unit().to_owned(),
                spec.bounds().lower(),
                spec.bounds().upper(),
                spec.scale(),
                spec.refine(),
            )
        })
        .collect()
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn key(value: (String, String, String)) -> PyResult<ParameterKey> {
    ParameterKey::new(value.0, value.1, value.2).map_err(value_error)
}

fn physics_model(records: Vec<(String, Vec<f64>)>) -> PyResult<Option<RietveldSamplePhysicsModel>> {
    let mut models = Vec::with_capacity(records.len());
    for (kind, values) in records {
        let model = match (kind.as_str(), values.as_slice()) {
            ("isotropic_size", [crystallite_size_nm, shape_factor]) => {
                RietveldSamplePhysicsModel::IsotropicSize {
                    crystallite_size_nm: *crystallite_size_nm,
                    shape_factor: *shape_factor,
                }
            }
            ("isotropic_microstrain", [rms_microstrain]) => {
                RietveldSamplePhysicsModel::IsotropicMicrostrain {
                    rms_microstrain: *rms_microstrain,
                }
            }
            ("isotropic_lorentzian_microstrain", [microstrain]) => {
                RietveldSamplePhysicsModel::IsotropicLorentzianMicrostrain {
                    microstrain: *microstrain,
                }
            }
            ("march_dollase", [ratio, h, k, l]) => RietveldSamplePhysicsModel::MarchDollase {
                ratio: *ratio,
                preferred_axis_hkl: [*h, *k, *l],
            },
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported native sample-physics record {kind:?}"
                )));
            }
        };
        models.push(model);
    }
    Ok(match models.len() {
        0 => None,
        1 => models.pop(),
        _ => Some(RietveldSamplePhysicsModel::Composite(models)),
    })
}

/// One owned built-in phase plus its optional guarded lattice bounds.
#[pyclass(name = "_RietveldPhase")]
pub(super) struct NativeRietveldPhase {
    phase: RietveldPhase,
    bounds: Option<LatticeBounds>,
}

#[pymethods]
impl NativeRietveldPhase {
    #[staticmethod]
    fn fixed(
        base: PyRef<'_, NativeStructuralPhase>,
        phase_id: String,
        name: String,
        site_ids: Vec<String>,
        physics: Vec<(String, Vec<f64>)>,
    ) -> PyResult<Self> {
        let definition = base.phase.definition().clone();
        let reflection_count = definition.hkl.len();
        let mut phase = RietveldPhase::new_with_site_ids(
            RecordId::new(phase_id).map_err(value_error)?,
            name,
            site_ids
                .into_iter()
                .map(|value| RecordId::new(value).map_err(value_error))
                .collect::<PyResult<Vec<_>>>()?,
            definition,
            OwnedCwContributions::neutral(reflection_count),
        )
        .map_err(value_error)?;
        if let Some(model) = physics_model(physics)? {
            phase = phase.with_sample_physics(model);
        }
        Ok(Self {
            phase,
            bounds: None,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn dynamic(
        base: PyRef<'_, NativeStructuralPhase>,
        phase_id: String,
        name: String,
        site_ids: Vec<String>,
        physics: Vec<(String, Vec<f64>)>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        wavelength_angstrom: f64,
        visible_two_theta_min_deg: f64,
        visible_two_theta_max_deg: f64,
        merge_friedel: bool,
        max_candidates: usize,
        guard_scale: f64,
    ) -> PyResult<Self> {
        let definition = base.phase.definition().clone();
        let parameterization =
            LatticeParameterization::new(definition.space_group.clone(), definition.cell)
                .map_err(value_error)?;
        let bounds = LatticeBounds::new(&parameterization, lower, upper).map_err(value_error)?;
        let domain = LatticeReflectionDomain::new(
            parameterization,
            bounds.clone(),
            wavelength_angstrom,
            [visible_two_theta_min_deg, visible_two_theta_max_deg],
            0.0,
            merge_friedel,
            max_candidates,
            guard_scale,
        )
        .map_err(value_error)?;
        let mut phase = RietveldPhase::from_lattice_domain(
            RecordId::new(phase_id).map_err(value_error)?,
            name,
            site_ids
                .into_iter()
                .map(|value| RecordId::new(value).map_err(value_error))
                .collect::<PyResult<Vec<_>>>()?,
            definition,
            domain,
        )
        .map_err(value_error)?;
        if let Some(model) = physics_model(physics)? {
            phase = phase.with_sample_physics(model);
        }
        Ok(Self {
            phase,
            bounds: Some(bounds),
        })
    }
}

/// One owned built-in analytical background.
#[pyclass(name = "_RietveldBackground")]
pub(super) struct NativeRietveldBackground {
    model: BackgroundModel,
}

#[pymethods]
impl NativeRietveldBackground {
    #[staticmethod]
    fn polynomial(background_id: String, coefficients: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            model: BackgroundModel::Polynomial(
                PolynomialBackground::new(background_id, coefficients).map_err(value_error)?,
            ),
        })
    }

    #[staticmethod]
    fn chebyshev(
        background_id: String,
        coefficients: Vec<f64>,
        domain_deg: [f64; 2],
    ) -> PyResult<Self> {
        Ok(Self {
            model: BackgroundModel::Chebyshev(
                ChebyshevBackground::new(background_id, coefficients, domain_deg)
                    .map_err(value_error)?,
            ),
        })
    }

    #[staticmethod]
    fn point(background_id: String, knot_x: Vec<f64>, values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            model: BackgroundModel::Point(
                PointBackground::new(background_id, knot_x, values).map_err(value_error)?,
            ),
        })
    }

    #[staticmethod]
    fn amorphous(background_id: String, peaks: Vec<(f64, f64, f64)>) -> PyResult<Self> {
        Ok(Self {
            model: BackgroundModel::Amorphous(
                AmorphousBackground::new(
                    background_id,
                    peaks
                        .into_iter()
                        .map(|(area, center, fwhm)| {
                            AmorphousPeak::new(area, center, fwhm).map_err(value_error)
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
                .map_err(value_error)?,
            ),
        })
    }

    #[staticmethod]
    fn composite(background_id: String, components: &Bound<'_, PyList>) -> PyResult<Self> {
        let models = components
            .iter()
            .map(|item| -> PyResult<_> {
                Ok(item
                    .extract::<PyRef<'_, Self>>()
                    .map_err(PyErr::from)?
                    .model
                    .clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            model: BackgroundModel::Composite(
                CompositeBackground::new(background_id, models).map_err(value_error)?,
            ),
        })
    }
}

/// One native fixed, affine, or multi-source constraint.
#[pyclass(name = "_RietveldConstraint")]
pub(super) struct NativeRietveldConstraint {
    constraint: Constraint,
}

#[pymethods]
impl NativeRietveldConstraint {
    #[staticmethod]
    fn fixed(target: (String, String, String), value: f64) -> PyResult<Self> {
        Ok(Self {
            constraint: Constraint::Fixed(
                FixedConstraint::new(key(target)?, value).map_err(value_error)?,
            ),
        })
    }

    #[staticmethod]
    fn affine(
        target: (String, String, String),
        source: (String, String, String),
        multiplier: f64,
        offset: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            constraint: Constraint::Affine(
                AffineConstraint::new(key(target)?, key(source)?, multiplier, offset)
                    .map_err(value_error)?,
            ),
        })
    }

    #[staticmethod]
    fn linear(
        target: (String, String, String),
        terms: Vec<((String, String, String), f64)>,
        offset: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            constraint: Constraint::Linear(
                LinearConstraint::new(
                    key(target)?,
                    terms
                        .into_iter()
                        .map(|(source, coefficient)| {
                            LinearTerm::new(key(source)?, coefficient).map_err(value_error)
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                    offset,
                )
                .map_err(value_error)?,
            ),
        })
    }
}

fn instrument_parameter(name: &str) -> PyResult<RietveldInstrumentParameter> {
    match name {
        "u_deg2" => Ok(RietveldInstrumentParameter::UDeg2),
        "v_deg2" => Ok(RietveldInstrumentParameter::VDeg2),
        "w_deg2" => Ok(RietveldInstrumentParameter::WDeg2),
        "x_deg" => Ok(RietveldInstrumentParameter::XDeg),
        "y_deg" => Ok(RietveldInstrumentParameter::YDeg),
        "wavelength_angstrom" => Ok(RietveldInstrumentParameter::WavelengthAngstrom),
        "zero_shift_deg" => Ok(RietveldInstrumentParameter::ZeroShiftDeg),
        "sample_displacement_mm" => Ok(RietveldInstrumentParameter::SampleDisplacementMm),
        "displace_x_micrometre" => Ok(RietveldInstrumentParameter::DisplaceXMicrometre),
        "displace_y_micrometre" => Ok(RietveldInstrumentParameter::DisplaceYMicrometre),
        _ => Err(PyValueError::new_err(format!(
            "unsupported native Rietveld instrument parameter {name:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn refinement_options(
    execution: &NativeExecutionPolicyModel,
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
    use_uncertainty: bool,
    support_fwhm: f64,
) -> PyResult<RietveldRefinementOptions> {
    let limits = RefinementLimits::new(
        max_iterations,
        max_evaluations,
        max_runtime_seconds,
        max_consecutive_rejections,
    )
    .map_err(value_error)?;
    let calculation =
        RietveldCalculationOptions::new(support_fwhm, use_uncertainty, execution.clone())
            .map_err(value_error)?;
    RietveldRefinementOptions::new(
        calculation,
        limits,
        min_iterations,
        objective_tolerance,
        parameter_tolerance,
        initial_damping,
        damping_increase,
        damping_decrease,
        cg_tolerance,
        max_cg_iterations,
        max_scaled_parameter_step,
        max_backtracks,
    )
    .map_err(value_error)
}

/// Complete owned native request assembled by the Python adapter.
#[pyclass(name = "_RietveldRequest")]
pub(super) struct NativeRietveldRequest {
    input: RietveldInput,
    selection: RietveldParameterSelection,
    bounds: Vec<Option<LatticeBounds>>,
    constraints: Vec<Constraint>,
    options: RietveldRefinementOptions,
    covariance: RietveldCovarianceOptions,
}

/// Thread-safe native cancellation shared with a detached solver call.
#[pyclass(name = "_RietveldCancellation")]
pub(super) struct NativeRietveldCancellation {
    token: CancellationToken,
}

#[pymethods]
impl NativeRietveldCancellation {
    #[new]
    fn new() -> Self {
        Self {
            token: CancellationToken::default(),
        }
    }

    fn request(&self, reason: String) -> PyResult<bool> {
        self.token.request(reason).map_err(value_error)
    }

    #[getter]
    fn reason(&self) -> PyResult<Option<String>> {
        self.token.reason().map_err(value_error)
    }
}

#[pymethods]
impl NativeRietveldRequest {
    #[new]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::fn_params_excessive_bools
    )]
    fn new<'py>(
        x_deg: PyReadonlyArray1<'py, f64>,
        observed_y: PyReadonlyArray1<'py, f64>,
        uncertainty: Option<PyReadonlyArray1<'py, f64>>,
        mask: Option<PyReadonlyArray1<'py, bool>>,
        fixed_background_y: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        background: Option<PyRef<'py, NativeRietveldBackground>>,
        phases: &Bound<'py, PyList>,
        phase_scale: bool,
        lattice: bool,
        coordinates: bool,
        occupancy: bool,
        u_iso: bool,
        instrument_parameters: Vec<String>,
        refine_background: bool,
        sample_physics: bool,
        constraints: &Bound<'py, PyList>,
        execution: PyRef<'py, NativeExecutionPolicy>,
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
        use_uncertainty: bool,
        support_fwhm: f64,
        estimate_covariance: bool,
        max_covariance_parameters: usize,
        unresolved_correlation: f64,
    ) -> PyResult<Self> {
        let pattern = PatternRecord::new(
            contiguous_slice(&x_deg, "x_deg")?.to_vec(),
            Some(contiguous_slice(&observed_y, "observed_y")?.to_vec()),
            uncertainty
                .as_ref()
                .map(|value| contiguous_slice(value, "uncertainty").map(<[_]>::to_vec))
                .transpose()?,
            mask.as_ref()
                .map(|value| bool_slice(value, "mask").map(<[_]>::to_vec))
                .transpose()?,
            Some(contiguous_slice(&fixed_background_y, "fixed_background_y")?.to_vec()),
        )
        .map_err(value_error)?;
        let instrument = cw_instrument(
            wavelength_angstrom,
            u_deg2,
            v_deg2,
            w_deg2,
            x_width_deg,
            y_width_deg,
        );
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let mut native_phases = Vec::with_capacity(phases.len());
        let mut bounds = Vec::with_capacity(phases.len());
        for item in phases.iter() {
            let phase = item.extract::<PyRef<'_, NativeRietveldPhase>>()?;
            native_phases.push(phase.phase.clone());
            bounds.push(phase.bounds.clone());
        }
        let background_model = background.as_ref().map(|value| value.model.clone());
        let input = match background_model {
            Some(background) => RietveldInput::new_with_background(
                pattern,
                instrument,
                axial,
                correction,
                background,
                native_phases,
            ),
            None => RietveldInput::new(pattern, instrument, axial, correction, native_phases),
        }
        .map_err(value_error)?;
        let selection = RietveldParameterSelection::new(
            RietveldStructuralSelection {
                phase_scale,
                lattice,
                coordinates,
                occupancy,
                u_iso,
            },
            instrument_parameters
                .iter()
                .map(|value| instrument_parameter(value))
                .collect::<PyResult<Vec<_>>>()?,
            refine_background,
            sample_physics,
        )
        .map_err(value_error)?;
        let constraints = constraints
            .iter()
            .map(|item| -> PyResult<_> {
                Ok(item
                    .extract::<PyRef<'_, NativeRietveldConstraint>>()
                    .map_err(PyErr::from)?
                    .constraint
                    .clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let options = refinement_options(
            &execution.policy,
            max_iterations,
            max_evaluations,
            max_runtime_seconds,
            max_consecutive_rejections,
            min_iterations,
            objective_tolerance,
            parameter_tolerance,
            initial_damping,
            damping_increase,
            damping_decrease,
            cg_tolerance,
            max_cg_iterations,
            max_scaled_parameter_step,
            max_backtracks,
            use_uncertainty,
            support_fwhm,
        )?;
        let covariance = RietveldCovarianceOptions::new(
            estimate_covariance,
            max_covariance_parameters,
            unresolved_correlation,
        )
        .map_err(value_error)?;
        Ok(Self {
            input,
            selection,
            bounds,
            constraints,
            options,
            covariance,
        })
    }

    fn refine(
        &self,
        py: Python<'_>,
        cancellation: Option<PyRef<'_, NativeRietveldCancellation>>,
        checkpoint: Option<PyRef<'_, NativeRietveldCheckpoint>>,
    ) -> PyResult<NativeRietveldResult> {
        let input = self.input.clone();
        let selection = self.selection.clone();
        let bounds = self.bounds.clone();
        let constraints = self.constraints.clone();
        let options = self.options.clone();
        let covariance = self.covariance;
        let cancellation = cancellation.map(|value| value.token.clone());
        let checkpoint = checkpoint.map(|value| value.checkpoint.clone());
        let result = py
            .detach(move || {
                refine_general_rietveld(
                    &input,
                    &selection,
                    &bounds,
                    &constraints,
                    &options,
                    covariance,
                    checkpoint.as_ref(),
                    cancellation,
                )
            })
            .map_err(value_error)?;
        Ok(NativeRietveldResult { result })
    }

    #[allow(clippy::too_many_arguments)]
    fn save_project(
        &self,
        py: Python<'_>,
        path: String,
        project_id: String,
        revision: u64,
        project_name: String,
        histogram_id: String,
        histogram_name: String,
        probe: &str,
        checkpoint: Option<PyRef<'_, NativeRietveldCheckpoint>>,
        overwrite: bool,
    ) -> PyResult<String> {
        let project_id = RecordId::new(project_id).map_err(value_error)?;
        let histogram_id = RecordId::new(histogram_id).map_err(value_error)?;
        let probe = match probe {
            "xray" => RadiationProbe::Xray,
            "neutron" => RadiationProbe::Neutron,
            _ => return Err(PyValueError::new_err("probe must be 'xray' or 'neutron'")),
        };
        let phase_ids = self
            .input
            .phases
            .iter()
            .map(|phase| phase.phase_id().clone())
            .collect::<Vec<_>>();
        let phases = self
            .input
            .phases
            .iter()
            .map(|phase| StructuralPhaseRecord {
                phase_id: phase.phase_id().clone(),
                name: phase.name().to_owned(),
                definition: phase.definition().clone(),
                required_providers: Vec::new(),
            })
            .collect();
        let experiment = ExperimentRecord::new(
            self.input.instrument,
            RadiationDefinition::Monochromatic {
                probe,
                wavelength_angstrom: self.input.instrument.wavelength_angstrom,
            },
            self.input.axial_geometry,
            self.input.position_correction,
        )
        .map_err(value_error)?;
        let state = RietveldProjectState {
            project: ProjectRecord {
                project_id,
                revision,
                name: project_name,
                histograms: vec![HistogramRecord {
                    histogram_id: histogram_id.clone(),
                    name: histogram_name,
                    pattern: self.input.pattern.clone(),
                    experiment,
                    phase_ids,
                }],
                phases,
                metadata: BTreeMap::default(),
            },
            analyses: vec![RietveldAnalysis {
                histogram_id,
                input: self.input.clone(),
                selection: self.selection.clone(),
                lattice_bounds: self.bounds.clone(),
                constraints: self.constraints.clone(),
                options: self.options.clone(),
                covariance: self.covariance,
                checkpoint: checkpoint.map(|value| value.checkpoint.clone()),
            }],
        };
        let destination = py
            .detach(move || save_rietveld_project(path, &state, ProjectSaveOptions { overwrite }))
            .map_err(value_error)?;
        Ok(destination.to_string_lossy().into_owned())
    }
}

/// Native checkpoint handle retained independently of result diagnostics.
#[pyclass(name = "_RietveldCheckpoint")]
pub(super) struct NativeRietveldCheckpoint {
    checkpoint: RietveldGeneralCheckpoint,
}

#[pymethods]
impl NativeRietveldCheckpoint {
    fn parameter_records(&self) -> Vec<PyParameterRecord> {
        parameter_records(&self.checkpoint.parameters)
    }
}

/// Validated native project loaded without Python-side scientific reconstruction.
#[pyclass(name = "_StoredRietveldProject")]
pub(super) struct NativeStoredRietveldProject {
    state: RietveldProjectState,
}

#[pymethods]
impl NativeStoredRietveldProject {
    #[staticmethod]
    fn load(py: Python<'_>, path: String) -> PyResult<Self> {
        Ok(Self {
            state: py
                .detach(move || load_rietveld_project(path, ProjectReadLimits::default()))
                .map_err(value_error)?,
        })
    }

    fn project_record(&self) -> (String, u64, String) {
        (
            self.state.project.project_id.as_str().to_owned(),
            self.state.project.revision,
            self.state.project.name.clone(),
        )
    }

    fn histogram_records(&self) -> Vec<(String, String)> {
        self.state
            .project
            .histograms
            .iter()
            .map(|value| (value.histogram_id.as_str().to_owned(), value.name.clone()))
            .collect()
    }

    fn checkpoint(&self, histogram_id: &str) -> PyResult<Option<NativeRietveldCheckpoint>> {
        let histogram_id = RecordId::new(histogram_id).map_err(value_error)?;
        Ok(self
            .state
            .analyses
            .iter()
            .find(|value| value.histogram_id == histogram_id)
            .and_then(|value| value.checkpoint.clone())
            .map(|checkpoint| NativeRietveldCheckpoint { checkpoint }))
    }
}

/// Complete native result retained for deterministic continuation.
#[pyclass(name = "_RietveldResult")]
pub(super) struct NativeRietveldResult {
    result: RietveldGeneralRefinementResult,
}

#[pymethods]
impl NativeRietveldResult {
    fn checkpoint(&self) -> NativeRietveldCheckpoint {
        NativeRietveldCheckpoint {
            checkpoint: self.result.checkpoint.clone(),
        }
    }

    #[getter]
    fn termination_reason(&self) -> &'static str {
        self.result.termination_reason.as_str()
    }

    #[getter]
    fn evaluations(&self) -> usize {
        self.result.evaluations
    }

    #[getter]
    fn jacobian_rank(&self) -> Option<usize> {
        self.result.jacobian_rank
    }

    #[getter]
    fn objective(&self) -> f64 {
        self.result.checkpoint.objective
    }

    #[getter]
    fn damping(&self) -> f64 {
        self.result.checkpoint.damping
    }

    fn calculated_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.result.calculation.y.clone().into_pyarray(py)
    }

    fn profile_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.result.calculation.profile_y.clone().into_pyarray(py)
    }

    fn background_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.result
            .calculation
            .background_y
            .clone()
            .into_pyarray(py)
    }

    fn residual<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.result
            .calculation
            .metrics
            .residual
            .clone()
            .into_pyarray(py)
    }

    fn weighted_residual<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.result
            .calculation
            .metrics
            .weighted_residual
            .clone()
            .into_pyarray(py)
    }

    fn included<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.result
            .calculation
            .metrics
            .included
            .clone()
            .into_pyarray(py)
    }

    fn metrics(&self) -> (f64, f64, f64, f64) {
        let value = &self.result.calculation.metrics;
        (
            value.rp,
            value.rwp,
            value.chi_square,
            value.reduced_chi_square,
        )
    }

    fn parameter_records(&self) -> Vec<PyParameterRecord> {
        parameter_records(&self.result.parameters)
    }

    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let records = PyList::empty(py);
        for row in &self.result.history {
            let record = PyDict::new(py);
            record.set_item("iteration", row.iteration)?;
            record.set_item("rp", row.rp)?;
            record.set_item("rwp", row.rwp)?;
            record.set_item("chi_square", row.chi_square)?;
            record.set_item("reduced_chi_square", row.reduced_chi_square)?;
            record.set_item("objective", row.objective)?;
            record.set_item("objective_change", row.objective_change)?;
            record.set_item("scaled_step_norm", row.scaled_step_norm)?;
            record.set_item("damping", row.damping)?;
            record.set_item("cg_iterations", row.cg_iterations)?;
            record.set_item("backtracks", row.backtracks)?;
            let changes = row
                .parameter_changes
                .iter()
                .map(|change| {
                    (
                        (
                            change.key.module().to_owned(),
                            change.key.owner_id().to_owned(),
                            change.key.name().to_owned(),
                        ),
                        change.before,
                        change.after,
                        change.scaled_change,
                    )
                })
                .collect::<Vec<_>>();
            record.set_item("parameter_changes", changes)?;
            let topology = row
                .topology_changes
                .iter()
                .map(|change| {
                    format!(
                        "phase {}: +{} -{} guarded families",
                        change.phase_id.as_str(),
                        change.added_reflection_ids.len(),
                        change.removed_reflection_ids.len()
                    )
                })
                .collect::<Vec<_>>();
            record.set_item("topology_changes", topology)?;
            records.append(record)?;
        }
        Ok(records)
    }

    fn covariance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.result
            .covariance
            .as_ref()
            .map(|value| {
                Array2::from_shape_vec((value.size, value.size), value.values.clone())
                    .map(|array| array.into_pyarray(py))
                    .map_err(value_error)
            })
            .transpose()
    }

    fn unresolved_correlations(&self) -> Vec<PyCorrelationRecord> {
        self.result
            .unresolved_correlations
            .iter()
            .map(|value| {
                (
                    (
                        value.left.module().to_owned(),
                        value.left.owner_id().to_owned(),
                        value.left.name().to_owned(),
                    ),
                    (
                        value.right.module().to_owned(),
                        value.right.owner_id().to_owned(),
                        value.right.name().to_owned(),
                    ),
                    value.correlation,
                )
            })
            .collect()
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRietveldPhase>()?;
    module.add_class::<NativeRietveldBackground>()?;
    module.add_class::<NativeRietveldConstraint>()?;
    module.add_class::<NativeRietveldRequest>()?;
    module.add_class::<NativeRietveldCancellation>()?;
    module.add_class::<NativeRietveldCheckpoint>()?;
    module.add_class::<NativeStoredRietveldProject>()?;
    module.add_class::<NativeRietveldResult>()?;
    Ok(())
}
