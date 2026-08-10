//! Built-in Python-free sample-physics records for structural workflows.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter};

use nalgebra::{Matrix3, Vector3};
use phasesmith_core::{CwContributionsError, OwnedCwContributionArrays, OwnedCwContributions};
use phasesmith_crystallography::UnitCell;

use crate::ParameterBounds;

const DEG_PER_RAD: f64 = 180.0 / PI;
const HALF_ANGLE_RAD_PER_DEG: f64 = PI / 360.0;
const CELL_PARAMETER_NAMES: [&str; 6] = [
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
];
/// One closed built-in sample-physics model.
#[derive(Clone, Debug, PartialEq)]
pub enum RietveldSamplePhysicsModel {
    /// Lorentzian Scherrer broadening.
    IsotropicSize {
        /// Coherent-domain size in nanometres.
        crystallite_size_nm: f64,
        /// Fixed positive Scherrer shape factor.
        shape_factor: f64,
    },
    /// Gaussian broadening from RMS `delta d / d`.
    IsotropicMicrostrain {
        /// Non-negative dimensionless RMS microstrain.
        rms_microstrain: f64,
    },
    /// Lorentzian broadening from a distribution of `delta d / d`.
    IsotropicLorentzianMicrostrain {
        /// Non-negative dimensionless Lorentzian microstrain.
        microstrain: f64,
    },
    /// March--Dollase integrated-intensity correction around a fixed axis.
    MarchDollase {
        /// Positive March ratio.
        ratio: f64,
        /// Non-zero preferred reciprocal-lattice axis.
        preferred_axis_hkl: [f64; 3],
    },
    /// Ordered multiplicative/additive composition.
    Composite(Vec<Self>),
}

/// Evaluated contribution arrays plus stable provider derivative names.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedSamplePhysics {
    /// Validated contribution arrays.
    pub contributions: OwnedCwContributions,
    /// Provider parameter names in derivative-row order.
    pub parameter_names: Vec<String>,
}

/// One stable refinable built-in sample-physics scalar.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplePhysicsParameter {
    /// Stable provider-local name.
    pub name: String,
    /// Current physical value.
    pub value: f64,
    /// Physical unit label.
    pub unit: &'static str,
    /// Closed physical bounds.
    pub bounds: ParameterBounds,
    /// Positive solver scaling.
    pub scale: f64,
}

impl RietveldSamplePhysicsModel {
    /// Return refinable model scalars in stable composition order.
    ///
    /// # Errors
    ///
    /// Returns [`SamplePhysicsError`] for invalid or duplicate model records.
    pub fn parameters(&self) -> Result<Vec<SamplePhysicsParameter>, SamplePhysicsError> {
        let result = match self {
            Self::IsotropicSize {
                crystallite_size_nm,
                shape_factor,
            } => {
                if crystallite_size_nm.is_nan()
                    || *crystallite_size_nm <= 0.0
                    || !crystallite_size_nm.is_finite()
                    || !shape_factor.is_finite()
                    || *shape_factor <= 0.0
                {
                    return Err(SamplePhysicsError::InvalidModel);
                }
                vec![SamplePhysicsParameter {
                    name: "isotropic_size.crystallite_size_nm".to_owned(),
                    value: *crystallite_size_nm,
                    unit: "nanometre",
                    bounds: ParameterBounds::new(f64::MIN_POSITIVE, f64::INFINITY)
                        .map_err(|_| SamplePhysicsError::InvalidModel)?,
                    scale: crystallite_size_nm.abs().max(1.0),
                }]
            }
            Self::IsotropicMicrostrain { rms_microstrain } => {
                if !rms_microstrain.is_finite() || *rms_microstrain < 0.0 {
                    return Err(SamplePhysicsError::InvalidModel);
                }
                vec![SamplePhysicsParameter {
                    name: "isotropic_microstrain.rms".to_owned(),
                    value: *rms_microstrain,
                    unit: "fraction",
                    bounds: ParameterBounds::new(0.0, f64::INFINITY)
                        .map_err(|_| SamplePhysicsError::InvalidModel)?,
                    scale: rms_microstrain.abs().max(1.0e-4),
                }]
            }
            Self::IsotropicLorentzianMicrostrain { microstrain } => {
                if !microstrain.is_finite() || *microstrain < 0.0 {
                    return Err(SamplePhysicsError::InvalidModel);
                }
                vec![SamplePhysicsParameter {
                    name: "isotropic_lorentzian_microstrain.fraction".to_owned(),
                    value: *microstrain,
                    unit: "fraction",
                    bounds: ParameterBounds::new(0.0, f64::INFINITY)
                        .map_err(|_| SamplePhysicsError::InvalidModel)?,
                    scale: microstrain.abs().max(1.0e-4),
                }]
            }
            Self::MarchDollase {
                ratio,
                preferred_axis_hkl,
            } => {
                if !ratio.is_finite()
                    || *ratio <= 0.0
                    || preferred_axis_hkl.iter().any(|value| !value.is_finite())
                    || preferred_axis_hkl.iter().all(|value| *value == 0.0)
                {
                    return Err(SamplePhysicsError::InvalidModel);
                }
                vec![SamplePhysicsParameter {
                    name: "march_dollase.ratio".to_owned(),
                    value: *ratio,
                    unit: "relative",
                    bounds: ParameterBounds::new(f64::MIN_POSITIVE, f64::INFINITY)
                        .map_err(|_| SamplePhysicsError::InvalidModel)?,
                    scale: ratio.abs().max(1.0),
                }]
            }
            Self::Composite(models) => {
                if models.is_empty() {
                    return Err(SamplePhysicsError::EmptyComposite);
                }
                models
                    .iter()
                    .map(Self::parameters)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect()
            }
        };
        if result
            .iter()
            .map(|parameter| &parameter.name)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != result.len()
        {
            return Err(SamplePhysicsError::DuplicateParameterName);
        }
        Ok(result)
    }

    /// Clone the model with replacement refinable scalar values.
    ///
    /// # Errors
    ///
    /// Returns [`SamplePhysicsError`] for missing, unknown, or invalid values.
    pub fn replace_parameters(
        &self,
        values: &std::collections::BTreeMap<String, f64>,
    ) -> Result<Self, SamplePhysicsError> {
        let expected = self
            .parameters()?
            .into_iter()
            .map(|parameter| parameter.name)
            .collect::<std::collections::BTreeSet<_>>();
        if values.len() != expected.len() || values.keys().any(|name| !expected.contains(name)) {
            return Err(SamplePhysicsError::ParameterSetMismatch);
        }
        let result = match self {
            Self::IsotropicSize { shape_factor, .. } => Self::IsotropicSize {
                crystallite_size_nm: values["isotropic_size.crystallite_size_nm"],
                shape_factor: *shape_factor,
            },
            Self::IsotropicMicrostrain { .. } => Self::IsotropicMicrostrain {
                rms_microstrain: values["isotropic_microstrain.rms"],
            },
            Self::IsotropicLorentzianMicrostrain { .. } => Self::IsotropicLorentzianMicrostrain {
                microstrain: values["isotropic_lorentzian_microstrain.fraction"],
            },
            Self::MarchDollase {
                preferred_axis_hkl, ..
            } => Self::MarchDollase {
                ratio: values["march_dollase.ratio"],
                preferred_axis_hkl: *preferred_axis_hkl,
            },
            Self::Composite(models) => Self::Composite(
                models
                    .iter()
                    .map(|model| {
                        let names = model
                            .parameters()?
                            .into_iter()
                            .map(|parameter| parameter.name)
                            .collect::<std::collections::BTreeSet<_>>();
                        let child = values
                            .iter()
                            .filter(|(name, _)| names.contains(*name))
                            .map(|(name, value)| (name.clone(), *value))
                            .collect();
                        model.replace_parameters(&child)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        result.parameters()?;
        Ok(result)
    }

    /// Validate and evaluate the model for one reflection batch.
    ///
    /// # Errors
    ///
    /// Returns [`SamplePhysicsError`] for invalid model, reflection, cell, or
    /// contribution state.
    pub fn evaluate(
        &self,
        hkl: &[[i32; 3]],
        two_theta_deg: &[f64],
        cell: UnitCell,
        wavelength_angstrom: f64,
    ) -> Result<EvaluatedSamplePhysics, SamplePhysicsError> {
        if hkl.len() != two_theta_deg.len()
            || hkl.is_empty()
            || two_theta_deg
                .iter()
                .any(|value| !value.is_finite() || !(0.0..180.0).contains(value))
            || !wavelength_angstrom.is_finite()
            || wavelength_angstrom <= 0.0
        {
            return Err(SamplePhysicsError::InvalidInput);
        }
        cell.geometry()
            .map_err(|_| SamplePhysicsError::InvalidInput)?;
        match self {
            Self::IsotropicSize {
                crystallite_size_nm,
                shape_factor,
            } => size(
                *crystallite_size_nm,
                *shape_factor,
                two_theta_deg,
                wavelength_angstrom,
            ),
            Self::IsotropicMicrostrain { rms_microstrain } => {
                microstrain(*rms_microstrain, two_theta_deg)
            }
            Self::IsotropicLorentzianMicrostrain { microstrain } => {
                lorentzian_microstrain(*microstrain, two_theta_deg)
            }
            Self::MarchDollase {
                ratio,
                preferred_axis_hkl,
            } => march(*ratio, *preferred_axis_hkl, hkl, cell),
            Self::Composite(models) => {
                if models.is_empty() {
                    return Err(SamplePhysicsError::EmptyComposite);
                }
                let evaluated = models
                    .iter()
                    .map(|model| model.evaluate(hkl, two_theta_deg, cell, wavelength_angstrom))
                    .collect::<Result<Vec<_>, _>>()?;
                compose(&evaluated)
            }
        }
    }
}

fn size(
    size_nm: f64,
    shape_factor: f64,
    positions: &[f64],
    wavelength: f64,
) -> Result<EvaluatedSamplePhysics, SamplePhysicsError> {
    if size_nm.is_nan() || size_nm <= 0.0 || !shape_factor.is_finite() || shape_factor <= 0.0 {
        return Err(SamplePhysicsError::InvalidModel);
    }
    let count = positions.len();
    let mut lorentzian = vec![0.0; count];
    let mut d_position = vec![0.0; count];
    let mut d_parameter = vec![0.0; count];
    if size_nm.is_finite() {
        let scale = DEG_PER_RAD * shape_factor * wavelength / (10.0 * size_nm);
        for (index, position) in positions.iter().enumerate() {
            let theta = position * HALF_ANGLE_RAD_PER_DEG;
            lorentzian[index] = scale / theta.cos();
            d_position[index] = lorentzian[index] * HALF_ANGLE_RAD_PER_DEG * theta.tan();
            d_parameter[index] = -lorentzian[index] / size_nm;
        }
    }
    width_result(
        vec![0.0; count],
        lorentzian,
        vec![0.0; count],
        d_position,
        "isotropic_size.crystallite_size_nm",
        vec![0.0; count],
        d_parameter,
    )
}

fn microstrain(
    strain: f64,
    positions: &[f64],
) -> Result<EvaluatedSamplePhysics, SamplePhysicsError> {
    if !strain.is_finite() || strain < 0.0 {
        return Err(SamplePhysicsError::InvalidModel);
    }
    let coefficient = (2.0 * DEG_PER_RAD).powi(2);
    let mut variance = Vec::with_capacity(positions.len());
    let mut d_position = Vec::with_capacity(positions.len());
    let mut d_parameter = Vec::with_capacity(positions.len());
    for position in positions {
        let theta = position * HALF_ANGLE_RAD_PER_DEG;
        let tangent = theta.tan();
        variance.push(coefficient * strain * strain * tangent * tangent);
        d_parameter.push(2.0 * coefficient * strain * tangent * tangent);
        d_position.push(
            2.0 * coefficient * strain * strain * tangent / theta.cos().powi(2)
                * HALF_ANGLE_RAD_PER_DEG,
        );
    }
    let count = positions.len();
    width_result(
        variance,
        vec![0.0; count],
        d_position,
        vec![0.0; count],
        "isotropic_microstrain.rms",
        d_parameter,
        vec![0.0; count],
    )
}

fn lorentzian_microstrain(
    strain: f64,
    positions: &[f64],
) -> Result<EvaluatedSamplePhysics, SamplePhysicsError> {
    if !strain.is_finite() || strain < 0.0 {
        return Err(SamplePhysicsError::InvalidModel);
    }
    let mut lorentzian = Vec::with_capacity(positions.len());
    let mut d_position = Vec::with_capacity(positions.len());
    let mut d_parameter = Vec::with_capacity(positions.len());
    for position in positions {
        let theta = position * HALF_ANGLE_RAD_PER_DEG;
        lorentzian.push(DEG_PER_RAD * strain * theta.tan());
        d_parameter.push(DEG_PER_RAD * theta.tan());
        d_position.push(0.5 * strain / theta.cos().powi(2));
    }
    let count = positions.len();
    width_result(
        vec![0.0; count],
        lorentzian,
        vec![0.0; count],
        d_position,
        "isotropic_lorentzian_microstrain.fraction",
        vec![0.0; count],
        d_parameter,
    )
}

#[allow(clippy::too_many_arguments)]
fn width_result(
    gaussian: Vec<f64>,
    lorentzian: Vec<f64>,
    d_gaussian_position: Vec<f64>,
    d_lorentzian_position: Vec<f64>,
    name: &str,
    d_gaussian_parameter: Vec<f64>,
    d_lorentzian_parameter: Vec<f64>,
) -> Result<EvaluatedSamplePhysics, SamplePhysicsError> {
    let count = gaussian.len();
    Ok(EvaluatedSamplePhysics {
        contributions: OwnedCwContributions::new(
            count,
            1,
            OwnedCwContributionArrays {
                gaussian_variance_deg2: gaussian,
                lorentzian_fwhm_deg: lorentzian,
                intensity_multiplier: vec![1.0; count],
                d_gaussian_variance_d_position: d_gaussian_position,
                d_lorentzian_fwhm_d_position: d_lorentzian_position,
                d_intensity_multiplier_d_position: vec![0.0; count],
                d_gaussian_variance_d_parameters: d_gaussian_parameter,
                d_lorentzian_fwhm_d_parameters: d_lorentzian_parameter,
                d_intensity_multiplier_d_parameters: vec![0.0; count],
            },
        )?,
        parameter_names: vec![name.to_owned()],
    })
}

fn march(
    ratio: f64,
    axis: [f64; 3],
    hkl: &[[i32; 3]],
    cell: UnitCell,
) -> Result<EvaluatedSamplePhysics, SamplePhysicsError> {
    if !ratio.is_finite()
        || ratio <= 0.0
        || axis.iter().any(|value| !value.is_finite())
        || axis.iter().all(|value| *value == 0.0)
    {
        return Err(SamplePhysicsError::InvalidModel);
    }
    let reciprocal = cell
        .geometry()
        .map_err(|_| SamplePhysicsError::InvalidInput)?
        .reciprocal_metric;
    let metric = Matrix3::from_row_slice(&reciprocal.concat());
    let metric_derivatives = reciprocal_metric_derivatives(cell, metric)?;
    let axis = Vector3::from_row_slice(&axis);
    let axis_norm = (axis.transpose() * metric * axis)[0];
    let count = hkl.len();
    let mut multiplier = Vec::with_capacity(count);
    let mut d_ratio = Vec::with_capacity(count);
    let cell_derivative_count =
        CELL_PARAMETER_NAMES
            .len()
            .checked_mul(count)
            .ok_or(SamplePhysicsError::Contributions(
                CwContributionsError::AllocationOverflow,
            ))?;
    let mut d_cell = vec![0.0; cell_derivative_count];
    for reflection in hkl {
        let vector = Vector3::new(
            f64::from(reflection[0]),
            f64::from(reflection[1]),
            f64::from(reflection[2]),
        );
        let reflection_norm = (vector.transpose() * metric * vector)[0];
        let projection = (vector.transpose() * metric * axis)[0];
        if reflection_norm <= 0.0 || axis_norm <= 0.0 {
            return Err(SamplePhysicsError::InvalidInput);
        }
        let raw_cosine = projection * projection / (reflection_norm * axis_norm);
        let tolerance = 64.0 * f64::EPSILON;
        if !raw_cosine.is_finite() || raw_cosine < -tolerance || raw_cosine > 1.0 + tolerance {
            return Err(SamplePhysicsError::InvalidInput);
        }
        let cosine = raw_cosine.clamp(0.0, 1.0);
        let sine = 1.0 - cosine;
        let denominator = ratio * ratio * cosine + sine / ratio;
        multiplier.push(denominator.powf(-1.5));
        let derivative = 2.0 * ratio * cosine - sine / (ratio * ratio);
        d_ratio.push(-1.5 * denominator.powf(-2.5) * derivative);
        let d_multiplier_d_cosine = -1.5 * denominator.powf(-2.5) * (ratio * ratio - ratio.recip());
        for (parameter, derivative_metric) in metric_derivatives.iter().enumerate() {
            let d_reflection_norm = (vector.transpose() * derivative_metric * vector)[0];
            let d_axis_norm = (axis.transpose() * derivative_metric * axis)[0];
            let d_projection = (vector.transpose() * derivative_metric * axis)[0];
            let d_cosine = 2.0 * projection * d_projection / (reflection_norm * axis_norm)
                - cosine * (d_reflection_norm / reflection_norm + d_axis_norm / axis_norm);
            d_cell[parameter * count + multiplier.len() - 1] = d_multiplier_d_cosine * d_cosine;
        }
    }
    let mut intensity_derivatives = d_ratio;
    intensity_derivatives.extend(d_cell);
    let zeros = vec![0.0; count];
    Ok(EvaluatedSamplePhysics {
        contributions: OwnedCwContributions::new(
            count,
            1 + CELL_PARAMETER_NAMES.len(),
            OwnedCwContributionArrays {
                gaussian_variance_deg2: zeros.clone(),
                lorentzian_fwhm_deg: zeros.clone(),
                intensity_multiplier: multiplier,
                d_gaussian_variance_d_position: zeros.clone(),
                d_lorentzian_fwhm_d_position: zeros.clone(),
                d_intensity_multiplier_d_position: zeros.clone(),
                d_gaussian_variance_d_parameters: vec![0.0; intensity_derivatives.len()],
                d_lorentzian_fwhm_d_parameters: vec![0.0; intensity_derivatives.len()],
                d_intensity_multiplier_d_parameters: intensity_derivatives,
            },
        )?,
        parameter_names: std::iter::once("march_dollase.ratio".to_owned())
            .chain(
                CELL_PARAMETER_NAMES
                    .iter()
                    .map(|name| format!("march_dollase.cell.{name}")),
            )
            .collect(),
    })
}

fn reciprocal_metric_derivatives(
    cell: UnitCell,
    reciprocal: Matrix3<f64>,
) -> Result<[Matrix3<f64>; 6], SamplePhysicsError> {
    let [a, b, c, alpha_deg, beta_deg, gamma_deg] = [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ];
    let [alpha, beta, gamma] = [alpha_deg, beta_deg, gamma_deg].map(f64::to_radians);
    let mut direct = std::array::from_fn(|_| Matrix3::zeros());
    direct[0] = Matrix3::new(
        2.0 * a,
        b * gamma.cos(),
        c * beta.cos(),
        b * gamma.cos(),
        0.0,
        0.0,
        c * beta.cos(),
        0.0,
        0.0,
    );
    direct[1] = Matrix3::new(
        0.0,
        a * gamma.cos(),
        0.0,
        a * gamma.cos(),
        2.0 * b,
        c * alpha.cos(),
        0.0,
        c * alpha.cos(),
        0.0,
    );
    direct[2] = Matrix3::new(
        0.0,
        0.0,
        a * beta.cos(),
        0.0,
        0.0,
        b * alpha.cos(),
        a * beta.cos(),
        b * alpha.cos(),
        2.0 * c,
    );
    let per_degree = PI / 180.0;
    direct[3][(1, 2)] = -b * c * alpha.sin() * per_degree;
    direct[3][(2, 1)] = direct[3][(1, 2)];
    direct[4][(0, 2)] = -a * c * beta.sin() * per_degree;
    direct[4][(2, 0)] = direct[4][(0, 2)];
    direct[5][(0, 1)] = -a * b * gamma.sin() * per_degree;
    direct[5][(1, 0)] = direct[5][(0, 1)];
    if direct
        .iter()
        .flat_map(Matrix3::iter)
        .any(|value| !value.is_finite())
    {
        return Err(SamplePhysicsError::InvalidInput);
    }
    Ok(direct.map(|derivative| -reciprocal * derivative * reciprocal))
}

fn compose(items: &[EvaluatedSamplePhysics]) -> Result<EvaluatedSamplePhysics, SamplePhysicsError> {
    let count = items[0].contributions.reflection_count();
    if items
        .iter()
        .any(|item| item.contributions.reflection_count() != count)
    {
        return Err(SamplePhysicsError::InvalidInput);
    }
    let names = items
        .iter()
        .flat_map(|item| item.parameter_names.iter().cloned())
        .collect::<Vec<_>>();
    if names
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != names.len()
    {
        return Err(SamplePhysicsError::DuplicateParameterName);
    }
    let parameter_count = names.len();
    let derivative_count =
        parameter_count
            .checked_mul(count)
            .ok_or(SamplePhysicsError::Contributions(
                CwContributionsError::AllocationOverflow,
            ))?;
    let mut arrays = OwnedCwContributionArrays {
        gaussian_variance_deg2: vec![0.0; count],
        lorentzian_fwhm_deg: vec![0.0; count],
        intensity_multiplier: vec![1.0; count],
        d_gaussian_variance_d_position: vec![0.0; count],
        d_lorentzian_fwhm_d_position: vec![0.0; count],
        d_intensity_multiplier_d_position: vec![0.0; count],
        d_gaussian_variance_d_parameters: vec![0.0; derivative_count],
        d_lorentzian_fwhm_d_parameters: vec![0.0; derivative_count],
        d_intensity_multiplier_d_parameters: vec![0.0; derivative_count],
    };
    let mut row_offset = 0;
    for item in items {
        let source = item.contributions.arrays();
        let previous_multipliers = arrays.intensity_multiplier.clone();
        for (reflection, old_multiplier) in previous_multipliers.iter().copied().enumerate() {
            let child_multiplier = source.intensity_multiplier[reflection];
            arrays.gaussian_variance_deg2[reflection] += source.gaussian_variance_deg2[reflection];
            arrays.lorentzian_fwhm_deg[reflection] += source.lorentzian_fwhm_deg[reflection];
            arrays.d_gaussian_variance_d_position[reflection] +=
                source.d_gaussian_variance_d_position[reflection];
            arrays.d_lorentzian_fwhm_d_position[reflection] +=
                source.d_lorentzian_fwhm_d_position[reflection];
            arrays.d_intensity_multiplier_d_position[reflection] =
                arrays.d_intensity_multiplier_d_position[reflection] * child_multiplier
                    + old_multiplier * source.d_intensity_multiplier_d_position[reflection];
            arrays.intensity_multiplier[reflection] *= child_multiplier;
            for prior in 0..row_offset {
                arrays.d_intensity_multiplier_d_parameters[prior * count + reflection] *=
                    child_multiplier;
            }
        }
        for row in 0..item.parameter_names.len() {
            for (reflection, previous_multiplier) in
                previous_multipliers.iter().copied().enumerate()
            {
                let source_index = row * count + reflection;
                let target_index = (row_offset + row) * count + reflection;
                arrays.d_gaussian_variance_d_parameters[target_index] =
                    source.d_gaussian_variance_d_parameters[source_index];
                arrays.d_lorentzian_fwhm_d_parameters[target_index] =
                    source.d_lorentzian_fwhm_d_parameters[source_index];
                arrays.d_intensity_multiplier_d_parameters[target_index] =
                    source.d_intensity_multiplier_d_parameters[source_index] * previous_multiplier;
            }
        }
        row_offset += item.parameter_names.len();
    }
    Ok(EvaluatedSamplePhysics {
        contributions: OwnedCwContributions::new(count, parameter_count, arrays)?,
        parameter_names: names,
    })
}

/// Invalid built-in sample-physics state.
#[derive(Debug)]
pub enum SamplePhysicsError {
    /// Model scalar or axis state is invalid.
    InvalidModel,
    /// Reflection/cell/wavelength state is invalid.
    InvalidInput,
    /// A composite model must not be empty.
    EmptyComposite,
    /// Composite child parameter names must be unique.
    DuplicateParameterName,
    /// Replacement values do not exactly match model parameters.
    ParameterSetMismatch,
    /// Contribution array construction failed.
    Contributions(CwContributionsError),
}

impl Display for SamplePhysicsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidModel => formatter.write_str("native sample-physics model is invalid"),
            Self::InvalidInput => formatter.write_str("native sample-physics input is invalid"),
            Self::EmptyComposite => formatter.write_str("sample-physics composite is empty"),
            Self::DuplicateParameterName => {
                formatter.write_str("sample-physics parameter names are duplicated")
            }
            Self::ParameterSetMismatch => {
                formatter.write_str("sample-physics replacement parameters do not match")
            }
            Self::Contributions(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for SamplePhysicsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Contributions(error) => Some(error),
            Self::InvalidModel
            | Self::InvalidInput
            | Self::EmptyComposite
            | Self::DuplicateParameterName
            | Self::ParameterSetMismatch => None,
        }
    }
}

impl From<CwContributionsError> for SamplePhysicsError {
    fn from(value: CwContributionsError) -> Self {
        Self::Contributions(value)
    }
}
