//! Python adapter for the native effective-profile workflow.

use npy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    LeBailOptions, LeBailPhase, ProfileEstimationInput, ProfileEstimationMode,
    ProfileEstimationOptions, ProfileEstimationResult, estimate_effective_profile,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::NativeExecutionPolicy;

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "_estimate_effective_profile")]
fn estimate_effective_profile_for_python<'py>(
    py: Python<'py>,
    x_deg: PyReadonlyArray1<'py, f64>,
    observed_y: PyReadonlyArray1<'py, f64>,
    uncertainty: Option<PyReadonlyArray1<'py, f64>>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
    background_y: PyReadonlyArray1<'py, f64>,
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_width_deg: f64,
    y_width_deg: f64,
    phase_id: String,
    phase_name: String,
    reflection_ids: Vec<String>,
    hkl: PyReadonlyArray2<'py, i64>,
    d_spacing_angstrom: PyReadonlyArray1<'py, f64>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    integrated_intensity: PyReadonlyArray1<'py, f64>,
    phase_scale: f64,
    mode: &str,
    minimum_relative_rwp_improvement: f64,
    minimum_absolute_rwp_improvement: f64,
    maximum_absolute_correlation: f64,
    max_iterations: usize,
    min_iterations: usize,
    intensity_tolerance: f64,
    rwp_tolerance: f64,
    redistribution_damping: f64,
    minimum_calculated: f64,
    initial_intensity_floor: f64,
    use_uncertainty: bool,
    profile_damping: f64,
    max_scaled_parameter_step: f64,
    max_profile_backtracks: usize,
    unresolved_correlation: f64,
    diagnose_rank_deficiency: bool,
    support_fwhm: f64,
    execution: &NativeExecutionPolicy,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = parse_mode(mode)?;
    let x_deg = x_deg.as_slice()?.to_vec();
    let observed_y = observed_y.as_slice()?.to_vec();
    let uncertainty = uncertainty
        .map(|values| values.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let mask = mask
        .map(|values| values.as_slice().map(<[bool]>::to_vec))
        .transpose()?;
    let background_y = background_y.as_slice()?.to_vec();
    let d_spacing_angstrom = d_spacing_angstrom.as_slice()?.to_vec();
    let two_theta_deg = two_theta_deg.as_slice()?.to_vec();
    let integrated_intensity = integrated_intensity.as_slice()?.to_vec();
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
                i32::try_from(row[0]).map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
                i32::try_from(row[1]).map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
                i32::try_from(row[2]).map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
            ])
        })
        .collect::<PyResult<Vec<_>>>()?;
    let execution = execution.policy.clone();

    let result = py
        .detach(move || {
            let pattern = PatternRecord::new(
                x_deg,
                Some(observed_y),
                uncertainty,
                mask,
                Some(background_y),
            )
            .map_err(|error| error.to_string())?;
            let instrument = ConstantWavelengthInstrument {
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_deg: x_width_deg,
                y_deg: y_width_deg,
            };
            let phase = LeBailPhase::new(
                phase_id,
                phase_name,
                reflection_ids,
                hkl,
                d_spacing_angstrom,
                two_theta_deg,
                integrated_intensity,
                phase_scale,
                Vec::new(),
            )
            .map_err(|error| error.to_string())?;
            let input = ProfileEstimationInput::new(pattern, instrument, phase)
                .map_err(|error| error.to_string())?;
            let lebail = LeBailOptions::new(
                max_iterations,
                min_iterations,
                intensity_tolerance,
                rwp_tolerance,
                redistribution_damping,
                minimum_calculated,
                initial_intensity_floor,
                use_uncertainty,
                unresolved_correlation,
                diagnose_rank_deficiency,
                support_fwhm,
                execution,
            )
            .and_then(|options| {
                options.with_profile_controls(
                    profile_damping,
                    max_scaled_parameter_step,
                    max_profile_backtracks,
                )
            })
            .map_err(|error| error.to_string())?;
            let options = ProfileEstimationOptions::new(
                mode,
                false,
                minimum_relative_rwp_improvement,
                minimum_absolute_rwp_improvement,
                maximum_absolute_correlation,
                lebail,
            )
            .map_err(|error| error.to_string())?;
            estimate_effective_profile(&input, &options).map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;

    result_to_python(py, result)
}

fn result_to_python(
    py: Python<'_>,
    result: ProfileEstimationResult,
) -> PyResult<Bound<'_, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("wavelength_angstrom", result.instrument.wavelength_angstrom)?;
    output.set_item("u_deg2", result.instrument.u_deg2)?;
    output.set_item("v_deg2", result.instrument.v_deg2)?;
    output.set_item("w_deg2", result.instrument.w_deg2)?;
    output.set_item("x_deg", result.instrument.x_deg)?;
    output.set_item("y_deg", result.instrument.y_deg)?;
    output.set_item("active_parameters", result.active_parameters)?;
    output.set_item("rwp", result.lebail.metrics.rwp)?;
    output.set_item("rp", result.lebail.metrics.rp)?;
    output.set_item("chi_square", result.lebail.metrics.chi_square)?;
    output.set_item(
        "termination_reason",
        result.lebail.termination_reason.as_str(),
    )?;
    output.set_item("warnings", result.warnings)?;
    output.set_item("calculated_y", result.lebail.calculation.y.into_pyarray(py))?;
    output.set_item(
        "integrated_intensity",
        result
            .lebail
            .intensities
            .iter()
            .map(|value| value.integrated_intensity)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    let stages = PyList::empty(py);
    for stage in result.stages {
        let item = PyDict::new(py);
        item.set_item("kind", stage.kind.as_str())?;
        item.set_item("instrument_parameters", stage.instrument_parameters)?;
        item.set_item("accepted", stage.accepted)?;
        item.set_item("decision", stage.decision)?;
        item.set_item("rwp", stage.rwp)?;
        item.set_item(
            "maximum_absolute_correlation",
            stage.maximum_absolute_correlation,
        )?;
        stages.append(item)?;
    }
    output.set_item("stages", stages)?;
    Ok(output)
}

fn parse_mode(value: &str) -> PyResult<ProfileEstimationMode> {
    match value {
        "w_only" => Ok(ProfileEstimationMode::WOnly),
        "uvw" => Ok(ProfileEstimationMode::Uvw),
        "uvwxy" => Ok(ProfileEstimationMode::Uvwxy),
        "automatic" => Ok(ProfileEstimationMode::Automatic),
        _ => Err(PyValueError::new_err(
            "mode must be 'w_only', 'uvw', 'uvwxy', or 'automatic'",
        )),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        estimate_effective_profile_for_python,
        module
    )?)?;
    Ok(())
}
