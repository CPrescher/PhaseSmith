//! Python adapter for native fixed-instrument TOF Le Bail extraction.

use npy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use phasesmith_core::TofInstrument;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    ResidualEvaluation, TofChebyshevBackground, TofLeBailInput, TofLeBailOptions, TofLeBailPhase,
    TofLeBailResult, refine_tof_lebail,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::NativeExecutionPolicy;

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "_refine_tof_lebail")]
fn refine_tof_lebail_for_python<'py>(
    py: Python<'py>,
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
    cycles: usize,
    redistribution_damping: f64,
    initial_intensity_floor: f64,
    minimum_calculated: f64,
    support_fwhm: f64,
    tail_log: f64,
    use_uncertainty: bool,
    redistribution_use_uncertainty: bool,
    execution: &NativeExecutionPolicy,
) -> PyResult<Bound<'py, PyDict>> {
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
                i32::try_from(row[0]).map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
                i32::try_from(row[1]).map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
                i32::try_from(row[2]).map_err(|_| PyValueError::new_err("hkl exceeds int32"))?,
            ])
        })
        .collect::<PyResult<Vec<_>>>()?;
    let instrument = tof_instrument(&instrument_values)?;
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
    let execution = execution.policy.clone();
    let result = py
        .detach(move || {
            let pattern = TofPatternRecord::new(
                tof_us,
                Some(observed_y),
                uncertainty,
                mask,
                Some(fixed_background_y),
            )
            .map_err(|error| error.to_string())?;
            let domain_us = [pattern.tof_us[0], pattern.tof_us[pattern.tof_us.len() - 1]];
            let mut input = TofLeBailInput::new(pattern, instrument, phases)
                .map_err(|error| error.to_string())?;
            if let Some(coefficients) = background_coefficients {
                input = input
                    .with_refinable_background(
                        TofChebyshevBackground::new(
                            RecordId::new(background_id).map_err(|error| error.to_string())?,
                            coefficients,
                            domain_us,
                        )
                        .map_err(|error| error.to_string())?,
                    )
                    .map_err(|error| error.to_string())?;
            }
            let options = TofLeBailOptions::new(
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
            refine_tof_lebail(&input, &options).map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    result_to_python(py, result)
}

fn tof_instrument(values: &[f64]) -> PyResult<TofInstrument> {
    if values.len() != 15 {
        return Err(PyValueError::new_err(
            "instrument_values must contain 15 coefficients",
        ));
    }
    Ok(TofInstrument {
        zero_us: values[0],
        difc_us_per_angstrom: values[1],
        difa_us_per_angstrom2: values[2],
        difb_us_angstrom: values[3],
        alpha_coefficient: values[4],
        beta0_per_us: values[5],
        beta1_angstrom4_per_us: values[6],
        betaq_angstrom2_per_us: values[7],
        sigma0_us2: values[8],
        sigma1_us2_per_angstrom2: values[9],
        sigma2_us2_per_angstrom4: values[10],
        sigmaq_us2_per_angstrom: values[11],
        x_us_per_angstrom: values[12],
        y_us_per_angstrom2: values[13],
        z_us: values[14],
    })
}

#[allow(clippy::too_many_arguments)]
fn tof_phases(
    phase_ids: Vec<String>,
    phase_names: Vec<String>,
    offsets: &[i64],
    reflection_ids: Vec<String>,
    hkl: Vec<[i32; 3]>,
    d_spacing_angstrom: Vec<f64>,
    integrated_intensity: Vec<f64>,
    scales: Vec<f64>,
) -> PyResult<Vec<TofLeBailPhase>> {
    let phase_count = phase_ids.len();
    if phase_names.len() != phase_count
        || scales.len() != phase_count
        || offsets.len() != phase_count + 1
        || offsets.first() != Some(&0)
    {
        return Err(PyValueError::new_err("inconsistent TOF phase metadata"));
    }
    let offsets = offsets
        .iter()
        .map(|value| {
            usize::try_from(*value)
                .map_err(|_| PyValueError::new_err("phase_offsets must be nonnegative"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    if offsets.windows(2).any(|pair| pair[0] > pair[1])
        || offsets.last().copied() != Some(reflection_ids.len())
        || hkl.len() != reflection_ids.len()
        || d_spacing_angstrom.len() != reflection_ids.len()
        || integrated_intensity.len() != reflection_ids.len()
    {
        return Err(PyValueError::new_err("inconsistent TOF reflection arrays"));
    }
    (0..phase_count)
        .map(|index| {
            let begin = offsets[index];
            let end = offsets[index + 1];
            TofLeBailPhase::new(
                RecordId::new(phase_ids[index].clone())
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
                phase_names[index].clone(),
                reflection_ids[begin..end].to_vec(),
                hkl[begin..end].to_vec(),
                d_spacing_angstrom[begin..end].to_vec(),
                integrated_intensity[begin..end].to_vec(),
                scales[index],
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))
        })
        .collect()
}

fn result_to_python(py: Python<'_>, result: TofLeBailResult) -> PyResult<Bound<'_, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("y", result.calculation.y.into_pyarray(py))?;
    output.set_item("profile_y", result.calculation.profile_y.into_pyarray(py))?;
    output.set_item(
        "background_y",
        result.calculation.background_y.into_pyarray(py),
    )?;
    output.set_item("reflection_keys", result.calculation.reflection_keys)?;
    output.set_item(
        "phase_offsets",
        result
            .calculation
            .phase_offsets
            .into_iter()
            .map(|value| i64::try_from(value).unwrap_or(i64::MAX))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item(
        "integrated_intensity",
        result
            .intensities
            .iter()
            .map(|item| item.integrated_intensity)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item(
        "background_coefficients",
        result
            .background
            .map(|value| value.coefficients().to_vec().into_pyarray(py)),
    )?;
    output.set_item("metrics", metrics_to_python(py, &result.metrics)?)?;
    let history = PyList::empty(py);
    for record in result.history {
        let item = PyDict::new(py);
        item.set_item("iteration", record.iteration)?;
        item.set_item("metrics", metrics_to_python(py, &record.metrics)?)?;
        item.set_item(
            "maximum_relative_intensity_change",
            record.maximum_relative_intensity_change,
        )?;
        item.set_item(
            "maximum_absolute_background_change",
            record.maximum_absolute_background_change,
        )?;
        history.append(item)?;
    }
    output.set_item("history", history)?;
    Ok(output)
}

fn metrics_to_python<'py>(
    py: Python<'py>,
    metrics: &ResidualEvaluation,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("included", metrics.included.clone().into_pyarray(py))?;
    output.set_item("residual", metrics.residual.clone().into_pyarray(py))?;
    output.set_item(
        "weighted_residual",
        metrics.weighted_residual.clone().into_pyarray(py),
    )?;
    output.set_item("rp", metrics.rp)?;
    output.set_item("rwp", metrics.rwp)?;
    output.set_item("chi_square", metrics.chi_square)?;
    output.set_item("reduced_chi_square", metrics.reduced_chi_square)?;
    Ok(output)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(refine_tof_lebail_for_python, module)?)?;
    Ok(())
}
