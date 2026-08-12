//! Python adapter for native joint multi-bank TOF geometry refinement.

use npy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use phasesmith_core::{TOF_GLOBAL_PARAMETER_NAMES, TofInstrumentParameter};
use phasesmith_engine::crystallography::UnitCell;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, RefinementEvent, RefinementLimits, RefinementRuntime,
    TofBankInstrumentModel, TofGeometryParameterKey, TofInstrumentParameterBound, TofLeBailBank,
    TofLeBailInput, TofMultiBankGeometryCheckpoint, TofMultiBankGeometryInput,
    TofMultiBankGeometryOptions, TofMultiBankGeometryResult, TofMultiBankInput,
    TofMultiBankLatticeInput, TofSharedLatticePhase, refine_tof_multibank_geometry_with_runtime,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::tof_lebail::{
    NativeTofLeBailCancellation, event_to_python, metrics_to_python, tof_instrument, tof_phases,
};
use crate::{NativeExecutionPolicy, NativePreparedReflectionGenerator};

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Owned bank-local request reused by fixed and geometry-coupled multi-bank calls.
#[pyclass(name = "_TofLeBailBank")]
struct NativeTofLeBailBank {
    bank: TofLeBailBank,
}

#[pymethods]
impl NativeTofLeBailBank {
    #[new]
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn new<'py>(
        bank_id: String,
        tof_us: PyReadonlyArray1<'py, f64>,
        observed_y: PyReadonlyArray1<'py, f64>,
        uncertainty: Option<PyReadonlyArray1<'py, f64>>,
        mask: Option<PyReadonlyArray1<'py, bool>>,
        fixed_background_y: PyReadonlyArray1<'py, f64>,
        instrument_values: Vec<f64>,
        phase_ids: Vec<String>,
        phase_names: Vec<String>,
        phase_offsets: PyReadonlyArray1<'py, i64>,
        reflection_ids: Vec<String>,
        hkl: PyReadonlyArray2<'py, i64>,
        d_spacing_angstrom: PyReadonlyArray1<'py, f64>,
        integrated_intensity: PyReadonlyArray1<'py, f64>,
        phase_scales: PyReadonlyArray1<'py, f64>,
        background_coefficients: Option<Vec<f64>>,
        background_id: String,
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
        let phase_offsets = phase_offsets.as_slice()?.to_vec();
        let d_spacing_angstrom = d_spacing_angstrom.as_slice()?.to_vec();
        let integrated_intensity = integrated_intensity.as_slice()?.to_vec();
        let phase_scales = phase_scales.as_slice()?.to_vec();
        let hkl_shape = hkl.shape();
        if hkl_shape.len() != 2 || hkl_shape[1] != 3 {
            return Err(PyValueError::new_err("hkl must have shape (n, 3)"));
        }
        let hkl = hkl
            .as_array()
            .rows()
            .into_iter()
            .map(|row| {
                Ok([
                    i32::try_from(row[0])
                        .map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
                    i32::try_from(row[1])
                        .map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
                    i32::try_from(row[2])
                        .map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
                ])
            })
            .collect::<PyResult<Vec<_>>>()?;
        let phases = tof_phases(
            phase_ids,
            phase_names,
            &phase_offsets,
            reflection_ids,
            hkl,
            d_spacing_angstrom,
            integrated_intensity,
            phase_scales,
        )?;
        let pattern = TofPatternRecord::new(
            tof_us,
            Some(observed_y),
            uncertainty,
            mask,
            Some(fixed_background_y),
        )
        .map_err(value_error)?;
        let domain_us = [pattern.tof_us[0], pattern.tof_us[pattern.tof_us.len() - 1]];
        let mut input = TofLeBailInput::new(pattern, tof_instrument(&instrument_values)?, phases)
            .map_err(value_error)?;
        if let Some(coefficients) = background_coefficients {
            input = input
                .with_refinable_background(
                    phasesmith_workflows::TofChebyshevBackground::new(
                        RecordId::new(background_id).map_err(value_error)?,
                        coefficients,
                        domain_us,
                    )
                    .map_err(value_error)?,
                )
                .map_err(value_error)?;
        }
        Ok(Self {
            bank: TofLeBailBank {
                bank_id: RecordId::new(bank_id).map_err(value_error)?,
                input,
            },
        })
    }
}

/// Owned setting-aware bounded shared cell.
#[pyclass(name = "_TofSharedLatticePhase")]
struct NativeTofSharedLatticePhase {
    phase: TofSharedLatticePhase,
}

#[pymethods]
impl NativeTofSharedLatticePhase {
    #[new]
    fn new(
        generator: PyRef<'_, NativePreparedReflectionGenerator>,
        phase_id: String,
        cell_values: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> PyResult<Self> {
        let cell = unit_cell(&cell_values)?;
        let parameterization = LatticeParameterization::new(generator.space_group().clone(), cell)
            .map_err(value_error)?;
        let bounds = LatticeBounds::new(&parameterization, lower, upper).map_err(value_error)?;
        Ok(Self {
            phase: TofSharedLatticePhase::new(
                RecordId::new(phase_id).map_err(value_error)?,
                parameterization,
                bounds,
                cell,
            )
            .map_err(value_error)?,
        })
    }
}

/// Owned selected bank-local instrument coefficients.
#[pyclass(name = "_TofBankInstrumentModel")]
struct NativeTofBankInstrumentModel {
    model: TofBankInstrumentModel,
}

#[pymethods]
impl NativeTofBankInstrumentModel {
    #[new]
    fn new(bank_id: String, bounds: Vec<(String, f64, f64)>) -> PyResult<Self> {
        let bounds = bounds
            .into_iter()
            .map(|(name, lower, upper)| {
                TofInstrumentParameterBound::new(parameter(&name)?, lower, upper)
                    .map_err(value_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            model: TofBankInstrumentModel::new(
                RecordId::new(bank_id).map_err(value_error)?,
                bounds,
            )
            .map_err(value_error)?,
        })
    }
}

/// Opaque complete last-accepted joint state.
#[pyclass(name = "_TofMultiBankGeometryCheckpoint")]
struct NativeTofMultiBankGeometryCheckpoint {
    checkpoint: TofMultiBankGeometryCheckpoint,
}

#[pymethods]
impl NativeTofMultiBankGeometryCheckpoint {
    #[getter]
    fn completed_iterations(&self) -> usize {
        self.checkpoint.completed_iterations
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[pyfunction(name = "_refine_tof_multibank_geometry")]
fn refine_tof_multibank_geometry_for_python<'py>(
    py: Python<'py>,
    banks: Vec<Py<NativeTofLeBailBank>>,
    lattice_phases: Vec<Py<NativeTofSharedLatticePhase>>,
    instrument_models: Vec<Py<NativeTofBankInstrumentModel>>,
    cycles: usize,
    redistribution_damping: f64,
    initial_intensity_floor: f64,
    minimum_calculated: f64,
    support_fwhm: f64,
    tail_log: f64,
    use_uncertainty: bool,
    redistribution_use_uncertainty: bool,
    geometry_damping: f64,
    max_scaled_geometry_step: f64,
    max_geometry_backtracks: usize,
    unresolved_correlation: f64,
    execution: &NativeExecutionPolicy,
    cancellation: Option<PyRef<'py, NativeTofLeBailCancellation>>,
    checkpoint: Option<PyRef<'py, NativeTofMultiBankGeometryCheckpoint>>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let banks = banks
        .iter()
        .map(|bank| bank.borrow(py).bank.clone())
        .collect();
    let lattice_phases = lattice_phases
        .iter()
        .map(|phase| phase.borrow(py).phase.clone())
        .collect();
    let instrument_models = instrument_models
        .iter()
        .map(|model| model.borrow(py).model.clone())
        .collect();
    let execution = execution.policy.clone();
    let cancellation = cancellation.map(|value| value.token.clone());
    let checkpoint = checkpoint.map(|value| value.checkpoint.clone());
    let result = py
        .detach(move || {
            let lebail = phasesmith_workflows::TofLeBailOptions::new(
                cycles,
                redistribution_damping,
                initial_intensity_floor,
                minimum_calculated,
                support_fwhm,
                tail_log,
                use_uncertainty,
                execution,
            )
            .map_err(|error| error.to_string())?
            .with_redistribution_uncertainty(redistribution_use_uncertainty);
            let options = TofMultiBankGeometryOptions::new(
                lebail,
                geometry_damping,
                max_scaled_geometry_step,
                max_geometry_backtracks,
                unresolved_correlation,
            )
            .map_err(|error| error.to_string())?;
            let input = TofMultiBankGeometryInput {
                lattice: TofMultiBankLatticeInput {
                    multibank: TofMultiBankInput { banks },
                    lattice_phases,
                },
                instrument_models,
            };
            let evaluations_per_cycle = max_geometry_backtracks
                .checked_add(4)
                .ok_or_else(|| "TOF geometry evaluation budget overflow".to_owned())?;
            let max_evaluations = cycles
                .checked_mul(evaluations_per_cycle)
                .ok_or_else(|| "TOF geometry evaluation budget overflow".to_owned())?;
            let limits = RefinementLimits::new(cycles, max_evaluations, None, 1)
                .map_err(|error| error.to_string())?;
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
            refine_tof_multibank_geometry_with_runtime(
                &input,
                &options,
                checkpoint.as_ref(),
                &mut runtime,
            )
            .map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    result_to_python(py, result)
}

fn unit_cell(values: &[f64]) -> PyResult<UnitCell> {
    let [
        a_angstrom,
        b_angstrom,
        c_angstrom,
        alpha_deg,
        beta_deg,
        gamma_deg,
    ] = values
    else {
        return Err(PyValueError::new_err("cell_values must contain six values"));
    };
    Ok(UnitCell {
        a_angstrom: *a_angstrom,
        b_angstrom: *b_angstrom,
        c_angstrom: *c_angstrom,
        alpha_deg: *alpha_deg,
        beta_deg: *beta_deg,
        gamma_deg: *gamma_deg,
    })
}

fn parameter(name: &str) -> PyResult<TofInstrumentParameter> {
    TofInstrumentParameter::ALL
        .into_iter()
        .find(|parameter| parameter.name() == name)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown TOF instrument parameter {name:?}; expected one of {}",
                TOF_GLOBAL_PARAMETER_NAMES.join(", ")
            ))
        })
}

fn aggregate_metrics_to_python(
    py: Python<'_>,
    metrics: phasesmith_workflows::TofMultiBankMetrics,
) -> PyResult<Bound<'_, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("included_samples", metrics.included_samples)?;
    output.set_item("rp", metrics.rp)?;
    output.set_item("rwp", metrics.rwp)?;
    output.set_item("chi_square", metrics.chi_square)?;
    output.set_item("reduced_chi_square", metrics.reduced_chi_square)?;
    Ok(output)
}

fn key_record(key: &TofGeometryParameterKey) -> (String, String, String) {
    match key {
        TofGeometryParameterKey::Lattice {
            phase_id,
            parameter_name,
        } => (
            "lattice".to_owned(),
            phase_id.as_str().to_owned(),
            parameter_name.clone(),
        ),
        TofGeometryParameterKey::Instrument { bank_id, parameter } => (
            "instrument".to_owned(),
            bank_id.as_str().to_owned(),
            parameter.name().to_owned(),
        ),
    }
}

#[allow(clippy::too_many_lines)]
fn result_to_python(
    py: Python<'_>,
    result: TofMultiBankGeometryResult,
) -> PyResult<Bound<'_, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("termination_reason", result.termination_reason.as_str())?;
    output.set_item(
        "checkpoint",
        Py::new(
            py,
            NativeTofMultiBankGeometryCheckpoint {
                checkpoint: result.checkpoint,
            },
        )?,
    )?;
    output.set_item("metrics", aggregate_metrics_to_python(py, result.metrics)?)?;

    let banks = PyList::empty(py);
    for bank in result.banks {
        let item = PyDict::new(py);
        item.set_item("bank_id", bank.bank_id.as_str())?;
        item.set_item("y", bank.calculation.y.into_pyarray(py))?;
        item.set_item("profile_y", bank.calculation.profile_y.into_pyarray(py))?;
        item.set_item(
            "background_y",
            bank.calculation.background_y.into_pyarray(py),
        )?;
        item.set_item("reflection_keys", bank.calculation.reflection_keys)?;
        item.set_item(
            "phase_offsets",
            bank.calculation
                .phase_offsets
                .into_iter()
                .map(|value| i64::try_from(value).unwrap_or(i64::MAX))
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        item.set_item(
            "integrated_intensity",
            bank.intensities
                .into_iter()
                .map(|value| value.integrated_intensity)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        item.set_item(
            "background_coefficients",
            bank.background
                .map(|value| value.coefficients().to_vec().into_pyarray(py)),
        )?;
        item.set_item("metrics", metrics_to_python(py, &bank.metrics)?)?;
        banks.append(item)?;
    }
    output.set_item("banks", banks)?;

    output.set_item(
        "lattice_phases",
        result
            .lattice_phases
            .into_iter()
            .map(|state| (state.phase_id.as_str().to_owned(), cell_values(state.cell)))
            .collect::<Vec<_>>(),
    )?;
    output.set_item(
        "instruments",
        result
            .instruments
            .into_iter()
            .map(|state| {
                (
                    state.bank_id.as_str().to_owned(),
                    state.instrument.values().to_vec(),
                )
            })
            .collect::<Vec<_>>(),
    )?;

    let diagnostics = PyDict::new(py);
    diagnostics.set_item("parameter_count", result.diagnostics.parameter_count)?;
    diagnostics.set_item("jacobian_rank", result.diagnostics.jacobian_rank)?;
    diagnostics.set_item(
        "maximum_absolute_correlation",
        result.diagnostics.maximum_absolute_correlation,
    )?;
    diagnostics.set_item(
        "unresolved_correlations",
        result
            .diagnostics
            .unresolved_correlations
            .iter()
            .map(|value| {
                (
                    key_record(&value.left),
                    key_record(&value.right),
                    value.correlation,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    output.set_item("diagnostics", diagnostics)?;

    let history = PyList::empty(py);
    for record in result.history {
        let item = PyDict::new(py);
        item.set_item("iteration", record.iteration)?;
        item.set_item("metrics", aggregate_metrics_to_python(py, record.metrics)?)?;
        let bank_metrics = PyList::empty(py);
        for metrics in &record.bank_metrics {
            bank_metrics.append(metrics_to_python(py, metrics)?)?;
        }
        item.set_item("bank_metrics", bank_metrics)?;
        item.set_item(
            "maximum_relative_intensity_change",
            record.maximum_relative_intensity_change,
        )?;
        item.set_item(
            "maximum_absolute_background_change",
            record.maximum_absolute_background_change,
        )?;
        item.set_item(
            "scaled_geometry_step_norm",
            record.scaled_geometry_step_norm,
        )?;
        item.set_item(
            "lattice_parameter_changes",
            record
                .lattice_parameter_changes
                .into_iter()
                .map(|value| {
                    (
                        value.phase_id.as_str().to_owned(),
                        value.parameter_name,
                        value.before,
                        value.after,
                        value.scaled_change,
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        item.set_item(
            "instrument_parameter_changes",
            record
                .instrument_parameter_changes
                .into_iter()
                .map(|value| {
                    (
                        value.bank_id.as_str().to_owned(),
                        value.parameter.name(),
                        value.before,
                        value.after,
                        value.scaled_change,
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        history.append(item)?;
    }
    output.set_item("history", history)?;
    Ok(output)
}

const fn cell_values(cell: UnitCell) -> [f64; 6] {
    [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ]
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTofLeBailBank>()?;
    module.add_class::<NativeTofSharedLatticePhase>()?;
    module.add_class::<NativeTofBankInstrumentModel>()?;
    module.add_class::<NativeTofMultiBankGeometryCheckpoint>()?;
    module.add_function(wrap_pyfunction!(
        refine_tof_multibank_geometry_for_python,
        module
    )?)?;
    Ok(())
}
