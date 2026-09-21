//! Masked residual arrays and standard powder-diffraction metrics.

use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_model::{DomainError, PatternRecord, TofPatternRecord};

/// Weighting and degrees-of-freedom controls for residual evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualOptions {
    /// Use available one-sigma uncertainties when true.
    pub use_uncertainty: bool,
    /// Number of fitted free parameters subtracted from included samples.
    pub parameter_count: usize,
}

impl Default for ResidualOptions {
    fn default() -> Self {
        Self {
            use_uncertainty: true,
            parameter_count: 0,
        }
    }
}

/// Full residual arrays and standard masked powder metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualEvaluation {
    /// Sample-aligned inclusion mask; true means included.
    pub included: Vec<bool>,
    /// Sample-aligned `calculated - observed` values.
    pub residual: Vec<f64>,
    /// Residual divided by one-sigma uncertainty when requested and available.
    pub weighted_residual: Vec<f64>,
    /// Unweighted profile residual fraction.
    pub rp: f64,
    /// Weighted profile residual fraction.
    pub rwp: f64,
    /// Sum of squared selected weighted residuals.
    pub chi_square: f64,
    /// Chi-square divided by positive residual degrees of freedom.
    pub reduced_chi_square: f64,
}

/// Evaluate masked residual arrays and standard powder residual metrics.
///
/// Residual is `calculated - observed`. If enabled and present, uncertainty is
/// interpreted as one standard deviation. Zero `Rp`/`Rwp` denominators and
/// non-positive residual degrees of freedom produce positive infinity, matching
/// the scripting contract.
///
/// # Errors
///
/// Returns [`ResidualError`] if the pattern is invalid, has no observations,
/// or `calculated_y` has the wrong length or contains a non-finite sample.
pub fn evaluate_residuals(
    pattern: &PatternRecord,
    calculated_y: &[f64],
    options: ResidualOptions,
) -> Result<ResidualEvaluation, ResidualError> {
    pattern.validate().map_err(ResidualError::Pattern)?;
    evaluate_residual_arrays(
        pattern.sample_count(),
        pattern.observed_y.as_deref(),
        pattern.uncertainty.as_deref(),
        pattern.mask.as_deref(),
        calculated_y,
        options,
    )
}

/// Evaluate residuals on an explicitly microsecond-domain TOF pattern.
///
/// This is deliberately separate from [`evaluate_residuals`]: callers cannot
/// reinterpret TOF coordinates as constant-wavelength angles to reach shared
/// residual mathematics.
///
/// # Errors
///
/// Returns [`ResidualError`] for an invalid pattern, absent observations, or
/// invalid calculated values.
pub fn evaluate_tof_residuals(
    pattern: &TofPatternRecord,
    calculated_y: &[f64],
    options: ResidualOptions,
) -> Result<ResidualEvaluation, ResidualError> {
    pattern.validate().map_err(ResidualError::Pattern)?;
    evaluate_residual_arrays(
        pattern.sample_count(),
        pattern.observed_y.as_deref(),
        pattern.uncertainty.as_deref(),
        pattern.mask.as_deref(),
        calculated_y,
        options,
    )
}

pub(crate) fn evaluate_residual_arrays(
    sample_count: usize,
    observed_y: Option<&[f64]>,
    uncertainty: Option<&[f64]>,
    mask: Option<&[bool]>,
    calculated_y: &[f64],
    options: ResidualOptions,
) -> Result<ResidualEvaluation, ResidualError> {
    let observed_y = observed_y.ok_or(ResidualError::MissingObservations)?;
    if calculated_y.len() != sample_count {
        return Err(ResidualError::CalculatedLengthMismatch {
            expected: sample_count,
            actual: calculated_y.len(),
        });
    }
    if let Some(index) = calculated_y.iter().position(|value| !value.is_finite()) {
        return Err(ResidualError::NonFiniteCalculated { index });
    }
    let included = mask.map_or_else(|| vec![true; sample_count], <[bool]>::to_vec);
    let residual = calculated_y
        .iter()
        .zip(observed_y)
        .map(|(calculated, observed)| calculated - observed)
        .collect::<Vec<_>>();
    let uncertainty = options.use_uncertainty.then_some(uncertainty).flatten();
    let weighted_residual = match uncertainty {
        Some(uncertainty) => residual
            .iter()
            .zip(uncertainty)
            .map(|(value, sigma)| value / sigma)
            .collect(),
        None => residual.clone(),
    };

    let mut included_count = 0_usize;
    let mut absolute_residual_sum = 0.0;
    let mut absolute_observed_sum = 0.0;
    let mut weighted_observed_square_sum = 0.0;
    let mut chi_square = 0.0;
    for index in 0..sample_count {
        if !included[index] {
            continue;
        }
        included_count += 1;
        absolute_residual_sum += residual[index].abs();
        absolute_observed_sum += observed_y[index].abs();
        chi_square += weighted_residual[index] * weighted_residual[index];
        weighted_observed_square_sum += match uncertainty {
            Some(uncertainty) => {
                let sigma = uncertainty[index];
                (1.0 / (sigma * sigma)) * (observed_y[index] * observed_y[index])
            }
            None => observed_y[index] * observed_y[index],
        };
    }
    let rp = if absolute_observed_sum == 0.0 {
        f64::INFINITY
    } else {
        absolute_residual_sum / absolute_observed_sum
    };
    let rwp = if weighted_observed_square_sum == 0.0 {
        f64::INFINITY
    } else {
        (chi_square / weighted_observed_square_sum).sqrt()
    };
    let degrees_of_freedom = included_count.checked_sub(options.parameter_count);
    let reduced_chi_square = match degrees_of_freedom {
        Some(degrees) if degrees > 0 => chi_square / count_as_f64(degrees),
        _ => f64::INFINITY,
    };
    Ok(ResidualEvaluation {
        included,
        residual,
        weighted_residual,
        rp,
        rwp,
        chi_square,
        reduced_chi_square,
    })
}

// IEEE-754 conversion follows Python/NumPy's metric convention. Counts above
// 2^53 may round, but cannot arise without an already-impossible dense array.
#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> f64 {
    value as f64
}

/// Invalid input to native residual evaluation.
#[derive(Debug)]
pub enum ResidualError {
    /// The live pattern failed domain validation.
    Pattern(DomainError),
    /// Observed intensity is absent.
    MissingObservations,
    /// Calculated intensity length differs from the pattern.
    CalculatedLengthMismatch {
        /// Expected sample count.
        expected: usize,
        /// Received sample count.
        actual: usize,
    },
    /// One calculated sample is non-finite.
    NonFiniteCalculated {
        /// Rejected sample index.
        index: usize,
    },
}

impl Display for ResidualError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::MissingObservations => {
                formatter.write_str("observed_y is required for residual evaluation")
            }
            Self::CalculatedLengthMismatch { expected, actual } => write!(
                formatter,
                "calculated_y length {actual} does not match pattern length {expected}"
            ),
            Self::NonFiniteCalculated { index } => {
                write!(
                    formatter,
                    "calculated_y contains a non-finite value at index {index}"
                )
            }
        }
    }
}

impl Error for ResidualError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pattern(error) => Some(error),
            _ => None,
        }
    }
}
