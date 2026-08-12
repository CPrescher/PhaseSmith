//! Python adapter for guarded structural multi-bank neutron TOF refinement.

use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_core::{OwnedCwContributions, TofBankGeometry};
use phasesmith_engine::crystallography::IntegratedIntensityCorrectionModel;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, ParameterBounds, RefinementEvent, RefinementLimits,
    RefinementRuntime, RietveldPhase, RietveldStructuralSelection, StructuralTofBank,
    StructuralTofMultiBankCheckpoint, StructuralTofMultiBankInput,
    StructuralTofMultiBankRefinementOptions, StructuralTofMultiBankRefinementResult,
    TofChebyshevBackground, TofInstrumentParameterBound,
    refine_structural_tof_multibank_with_runtime,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::tof_lebail::{
    NativeTofLeBailCancellation, event_to_python, metrics_to_python, tof_instrument,
};
use crate::tof_multibank::parameter;
use crate::{NativeExecutionPolicy, NativeStructuralPhase};

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// One owned neutral unit-scale neutron phase for structural TOF.
#[pyclass(name = "_StructuralTofPhase")]
struct NativeStructuralTofPhase {
    phase: RietveldPhase,
}

#[pymethods]
impl NativeStructuralTofPhase {
    #[new]
    fn new(
        base: PyRef<'_, NativeStructuralPhase>,
        phase_id: String,
        name: String,
        site_ids: Vec<String>,
    ) -> PyResult<Self> {
        let definition = base.phase.definition().clone();
        let reflection_count = definition.hkl.len();
        Ok(Self {
            phase: RietveldPhase::new_with_site_ids(
                RecordId::new(phase_id).map_err(value_error)?,
                name,
                site_ids
                    .into_iter()
                    .map(|value| RecordId::new(value).map_err(value_error))
                    .collect::<PyResult<Vec<_>>>()?,
                definition,
                OwnedCwContributions::neutral(reflection_count),
            )
            .map_err(value_error)?,
        })
    }
}

/// One owned bank-local structural TOF request.
#[pyclass(name = "_StructuralTofBank")]
struct NativeStructuralTofBank {
    bank: StructuralTofBank,
}

#[pymethods]
impl NativeStructuralTofBank {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
        bank_id: String,
        tof_us: PyReadonlyArray1<'py, f64>,
        observed_y: PyReadonlyArray1<'py, f64>,
        uncertainty: Option<PyReadonlyArray1<'py, f64>>,
        mask: Option<PyReadonlyArray1<'py, bool>>,
        fixed_background_y: PyReadonlyArray1<'py, f64>,
        instrument_values: Vec<f64>,
        two_theta_deg: f64,
        correction: &str,
        scale: f64,
        scale_lower: f64,
        scale_upper: f64,
        refine_scale: bool,
        background_coefficients: Option<Vec<f64>>,
        background_id: String,
        refine_background: bool,
        instrument_bounds: Vec<(String, f64, f64)>,
    ) -> PyResult<Self> {
        let tof_us = tof_us.as_slice()?.to_vec();
        let observed_y = observed_y.as_slice()?.to_vec();
        let uncertainty = uncertainty
            .map(|values| values.as_slice().map(<[f64]>::to_vec))
            .transpose()?;
        let mask = mask
            .map(|values| values.as_slice().map(<[bool]>::to_vec))
            .transpose()?;
        let fixed_background_y = fixed_background_y.as_slice()?.to_vec();
        let pattern = TofPatternRecord::new(
            tof_us,
            Some(observed_y),
            uncertainty,
            mask,
            Some(fixed_background_y),
        )
        .map_err(value_error)?;
        let geometry = TofBankGeometry { two_theta_deg };
        geometry.validate().map_err(value_error)?;
        let correction_model = match correction {
            "neutral" => IntegratedIntensityCorrectionModel::Neutral,
            "time_of_flight_neutron_lorentz" => {
                IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "correction must be 'neutral' or 'time_of_flight_neutron_lorentz'",
                ));
            }
        };
        let background = background_coefficients
            .map(|coefficients| {
                TofChebyshevBackground::new(
                    RecordId::new(background_id).map_err(value_error)?,
                    coefficients,
                    [pattern.tof_us[0], pattern.tof_us[pattern.tof_us.len() - 1]],
                )
                .map_err(value_error)
            })
            .transpose()?;
        let instrument_bounds = instrument_bounds
            .into_iter()
            .map(|(name, lower, upper)| {
                TofInstrumentParameterBound::new(parameter(&name)?, lower, upper)
                    .map_err(value_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            bank: StructuralTofBank {
                bank_id: RecordId::new(bank_id).map_err(value_error)?,
                pattern,
                instrument: tof_instrument(&instrument_values)?,
                geometry,
                correction_model,
                scale,
                scale_bounds: ParameterBounds::new(scale_lower, scale_upper)
                    .map_err(value_error)?,
                refine_scale,
                background,
                refine_background,
                instrument_bounds,
            },
        })
    }
}

/// Opaque complete last-accepted structural TOF state.
#[pyclass(name = "_StructuralTofMultiBankCheckpoint")]
struct NativeStructuralTofMultiBankCheckpoint {
    checkpoint: StructuralTofMultiBankCheckpoint,
}

#[pymethods]
impl NativeStructuralTofMultiBankCheckpoint {
    #[getter]
    fn completed_iterations(&self) -> usize {
        self.checkpoint.completed_iterations()
    }
}

#[allow(
    clippy::fn_params_excessive_bools,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
#[pyfunction(name = "_refine_structural_tof_multibank")]
fn refine_structural_tof_multibank_for_python<'py>(
    py: Python<'py>,
    phase: Py<NativeStructuralTofPhase>,
    lattice: bool,
    coordinates: bool,
    occupancy: bool,
    u_iso: bool,
    lattice_lower: Option<Vec<f64>>,
    lattice_upper: Option<Vec<f64>>,
    banks: Vec<Py<NativeStructuralTofBank>>,
    support_fwhm: f64,
    tail_log: f64,
    use_uncertainty: bool,
    execution: &NativeExecutionPolicy,
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
    max_scaled_parameter_step: f64,
    max_backtracks: usize,
    cancellation: Option<PyRef<'py, NativeTofLeBailCancellation>>,
    checkpoint: Option<PyRef<'py, NativeStructuralTofMultiBankCheckpoint>>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let phase = phase.borrow(py).phase.clone();
    let lattice_bounds = match (lattice_lower, lattice_upper) {
        (Some(lower), Some(upper)) => {
            let parameterization = LatticeParameterization::new(
                phase.definition().space_group.clone(),
                phase.definition().cell,
            )
            .map_err(value_error)?;
            Some(LatticeBounds::new(&parameterization, lower, upper).map_err(value_error)?)
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "lattice lower and upper bounds must be supplied together",
            ));
        }
    };
    let banks = banks
        .iter()
        .map(|bank| bank.borrow(py).bank.clone())
        .collect();
    let execution = execution.policy.clone();
    let cancellation = cancellation.map(|value| value.token.clone());
    let checkpoint = checkpoint.map(|value| value.checkpoint.clone());
    let result = py
        .detach(move || {
            let limits = RefinementLimits::new(
                max_iterations,
                max_evaluations,
                max_runtime_seconds,
                max_consecutive_rejections,
            )
            .map_err(|error| error.to_string())?;
            let options = StructuralTofMultiBankRefinementOptions::new(
                limits,
                min_iterations,
                objective_tolerance,
                parameter_tolerance,
                initial_damping,
                damping_increase,
                damping_decrease,
                max_scaled_parameter_step,
                max_backtracks,
            )
            .map_err(|error| error.to_string())?;
            let input = StructuralTofMultiBankInput {
                phase,
                structural_selection: RietveldStructuralSelection {
                    phase_scale: false,
                    lattice,
                    coordinates,
                    occupancy,
                    u_iso,
                },
                lattice_bounds,
                banks,
                support_fwhm,
                tail_log,
                use_uncertainty,
                execution,
            };
            let mut runtime =
                RefinementRuntime::new(limits, cancellation).map_err(|error| error.to_string())?;
            if let Some(progress) = progress {
                runtime.set_event_sink(move |event: &RefinementEvent| {
                    Python::attach(|py| {
                        let record =
                            event_to_python(py, event).map_err(|error| error.to_string())?;
                        progress
                            .call1(py, (record,))
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    })
                });
            }
            refine_structural_tof_multibank_with_runtime(
                &input,
                options,
                checkpoint.as_ref(),
                &mut runtime,
            )
            .map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    result_to_python(py, result)
}

fn cell_values(cell: phasesmith_engine::crystallography::UnitCell) -> [f64; 6] {
    [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ]
}

#[allow(clippy::too_many_lines)]
fn result_to_python(
    py: Python<'_>,
    result: StructuralTofMultiBankRefinementResult,
) -> PyResult<Bound<'_, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("termination_reason", result.termination_reason.as_str())?;
    output.set_item("evaluations", result.evaluations)?;
    output.set_item("objective", result.calculation.objective)?;
    output.set_item(
        "checkpoint",
        Py::new(
            py,
            NativeStructuralTofMultiBankCheckpoint {
                checkpoint: result.checkpoint,
            },
        )?,
    )?;

    let definition = result.input.phase.definition();
    let phase_state = PyDict::new(py);
    phase_state.set_item("cell", cell_values(definition.cell))?;
    phase_state.set_item("fractional_xyz", definition.fractional_xyz.clone())?;
    phase_state.set_item("occupancy", definition.occupancy.clone().into_pyarray(py))?;
    phase_state.set_item(
        "u_iso_angstrom2",
        definition.u_iso_angstrom2.clone().into_pyarray(py),
    )?;
    output.set_item("phase", phase_state)?;

    let banks = PyList::empty(py);
    for (state, calculation) in result.input.banks.into_iter().zip(result.calculation.banks) {
        let item = PyDict::new(py);
        item.set_item("bank_id", state.bank_id.as_str())?;
        item.set_item("scale", state.scale)?;
        item.set_item("instrument", state.instrument.values().to_vec())?;
        item.set_item(
            "background_coefficients",
            state
                .background
                .map(|value| value.coefficients().to_vec().into_pyarray(py)),
        )?;
        item.set_item("y", calculation.y.into_pyarray(py))?;
        item.set_item("profile_y", calculation.profile_y.into_pyarray(py))?;
        item.set_item("background_y", calculation.background_y.into_pyarray(py))?;
        item.set_item(
            "d_spacing_angstrom",
            calculation.structural.d_spacing_angstrom.into_pyarray(py),
        )?;
        item.set_item(
            "integrated_intensity",
            calculation
                .structural
                .structure_factors
                .intensity
                .into_pyarray(py),
        )?;
        item.set_item("metrics", metrics_to_python(py, &calculation.metrics)?)?;
        banks.append(item)?;
    }
    output.set_item("banks", banks)?;

    output.set_item(
        "parameters",
        result
            .parameters
            .specs()
            .iter()
            .map(|spec| {
                (
                    (
                        spec.key().module().to_owned(),
                        spec.key().owner_id().to_owned(),
                        spec.key().name().to_owned(),
                    ),
                    spec.value(),
                    spec.unit().to_owned(),
                    spec.bounds().lower(),
                    spec.bounds().upper(),
                    spec.scale(),
                )
            })
            .collect::<Vec<_>>(),
    )?;
    let history = PyList::empty(py);
    for row in result.history {
        let item = PyDict::new(py);
        item.set_item("iteration", row.iteration)?;
        item.set_item("objective", row.objective)?;
        item.set_item("objective_change", row.objective_change)?;
        item.set_item("scaled_step_norm", row.scaled_step_norm)?;
        item.set_item("damping", row.damping)?;
        item.set_item("backtracks", row.backtracks)?;
        item.set_item(
            "parameter_changes",
            row.parameter_changes
                .into_iter()
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
                .collect::<Vec<_>>(),
        )?;
        history.append(item)?;
    }
    output.set_item("history", history)?;
    Ok(output)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeStructuralTofPhase>()?;
    module.add_class::<NativeStructuralTofBank>()?;
    module.add_class::<NativeStructuralTofMultiBankCheckpoint>()?;
    module.add_function(wrap_pyfunction!(
        refine_structural_tof_multibank_for_python,
        module
    )?)?;
    Ok(())
}
