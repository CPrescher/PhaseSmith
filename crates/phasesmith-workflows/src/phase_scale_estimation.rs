//! Deterministic weighted non-negative initialization of Rietveld phase scales.
//!
//! The estimator evaluates every structural phase at unit scale on the input's
//! exact finite-support convention. For each mask-included sample `i`, it solves
//!
//! `min_(s >= 0) Σ_i ((y_obs,i - y_background,i - Σ_p A_ip s_p) / σ_i)^2`,
//!
//! where `A_ip` is phase `p`'s unit-scale profile. One-sigma uncertainties are
//! used only when the supplied calculation options request them. The complete
//! pattern grid remains unchanged; the result owns a cloned [`RietveldInput`]
//! whose phase definitions contain the estimated scales.

use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{DMatrix, DVector};

use crate::{RietveldCalculationOptions, RietveldError, RietveldInput, calculate_rietveld_pattern};

/// Diagnostics and restart-ready input produced by phase-scale initialization.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseScaleEstimationResult {
    /// Cloned input with the estimated scale installed in every phase.
    pub input: RietveldInput,
    /// Estimated non-negative phase scales in input phase order.
    pub scales: Vec<f64>,
    /// Number of mask-included observations used by the solve.
    pub included_points: usize,
    /// Number of strictly positive fitted phase scales.
    pub active_phases: usize,
    /// Number of deterministic outer active-set iterations.
    pub iterations: usize,
    /// Final weighted residual sum of squares of the linear initialization.
    pub weighted_residual_sum_squares: f64,
}

/// Failure to build or solve the weighted phase-scale initialization problem.
#[derive(Debug)]
pub enum PhaseScaleEstimationError {
    /// The input has no observation array.
    MissingObservations,
    /// The mask excludes every observation.
    EmptyDomain,
    /// Matrix allocation size overflowed `usize`.
    AllocationOverflow,
    /// A native Rietveld calculation or phase replacement failed.
    Rietveld(RietveldError),
    /// An active least-squares subproblem was singular or non-finite.
    LinearSolve,
    /// The bounded deterministic active-set solve did not converge.
    DidNotConverge,
}

impl Display for PhaseScaleEstimationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingObservations => {
                formatter.write_str("phase-scale estimation requires observed intensities")
            }
            Self::EmptyDomain => {
                formatter.write_str("phase-scale estimation has no mask-included observations")
            }
            Self::AllocationOverflow => {
                formatter.write_str("phase-scale estimation matrix allocation overflow")
            }
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::LinearSolve => {
                formatter.write_str("phase-scale active least-squares solve failed")
            }
            Self::DidNotConverge => {
                formatter.write_str("phase-scale active-set solve did not converge")
            }
        }
    }
}

impl Error for PhaseScaleEstimationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rietveld(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RietveldError> for PhaseScaleEstimationError {
    fn from(value: RietveldError) -> Self {
        Self::Rietveld(value)
    }
}

/// Estimate and install deterministic weighted non-negative phase scales.
///
/// # Errors
///
/// Returns [`PhaseScaleEstimationError`] when observations are absent, the
/// mask excludes every sample, native profile evaluation fails, or an active
/// least-squares subproblem cannot be solved deterministically.
pub fn estimate_initial_phase_scales(
    input: &RietveldInput,
    options: &RietveldCalculationOptions,
) -> Result<PhaseScaleEstimationResult, PhaseScaleEstimationError> {
    input.validate()?;
    let observed = input
        .pattern
        .observed_y
        .as_ref()
        .ok_or(PhaseScaleEstimationError::MissingObservations)?;
    let mut unit_input = input.clone();
    unit_input.phases = input
        .phases
        .iter()
        .map(|phase| {
            let mut definition = phase.definition().clone();
            definition.scale = 1.0;
            phase.with_definition(definition)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let calculation = calculate_rietveld_pattern(&unit_input, options)?;
    let included_indices = (0..input.pattern.sample_count())
        .filter(|index| input.pattern.mask.as_ref().is_none_or(|mask| mask[*index]))
        .collect::<Vec<_>>();
    if included_indices.is_empty() {
        return Err(PhaseScaleEstimationError::EmptyDomain);
    }
    let rows = included_indices.len();
    let columns = calculation.phases.len();
    let capacity = rows
        .checked_mul(columns)
        .ok_or(PhaseScaleEstimationError::AllocationOverflow)?;
    let mut design = Vec::with_capacity(capacity);
    let mut target = Vec::with_capacity(rows);
    for index in included_indices {
        let sigma = if options.use_uncertainty {
            input
                .pattern
                .uncertainty
                .as_ref()
                .map_or(1.0, |values| values[index])
        } else {
            1.0
        };
        for phase in &calculation.phases {
            design.push(phase.result.accumulation.y[index] / sigma);
        }
        target.push((observed[index] - calculation.background_y[index]) / sigma);
    }
    let matrix = DMatrix::from_row_slice(rows, columns, &design);
    let target = DVector::from_vec(target);
    let (scales, iterations) = solve_non_negative_least_squares(&matrix, &target)?;
    let residual = &matrix * &scales - target;
    let weighted_residual_sum_squares = residual.dot(&residual);
    let scales = scales.as_slice().to_vec();
    if !weighted_residual_sum_squares.is_finite()
        || scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
    {
        return Err(PhaseScaleEstimationError::LinearSolve);
    }
    let mut estimated_input = input.clone();
    estimated_input.phases = input
        .phases
        .iter()
        .zip(&scales)
        .map(|(phase, scale)| {
            let mut definition = phase.definition().clone();
            definition.scale = *scale;
            phase.with_definition(definition)
        })
        .collect::<Result<Vec<_>, _>>()?;
    estimated_input.validate()?;
    Ok(PhaseScaleEstimationResult {
        input: estimated_input,
        active_phases: scales.iter().filter(|scale| **scale > 0.0).count(),
        included_points: rows,
        scales,
        iterations,
        weighted_residual_sum_squares,
    })
}

fn solve_non_negative_least_squares(
    matrix: &DMatrix<f64>,
    target: &DVector<f64>,
) -> Result<(DVector<f64>, usize), PhaseScaleEstimationError> {
    let columns = matrix.ncols();
    let gram = matrix.transpose() * matrix;
    let correlation = matrix.transpose() * target;
    let tolerance = 1.0e-12
        * correlation
            .iter()
            .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
    let maximum_iterations = columns
        .checked_mul(columns)
        .and_then(|value| value.checked_mul(10))
        .and_then(|value| value.checked_add(10))
        .ok_or(PhaseScaleEstimationError::AllocationOverflow)?;
    let mut solution = DVector::zeros(columns);
    let mut passive = vec![false; columns];
    let mut iterations = 0usize;

    loop {
        let gradient = &correlation - &gram * &solution;
        let candidate = (0..columns)
            .filter(|index| !passive[*index] && gradient[*index] > tolerance)
            .max_by(|left, right| {
                gradient[*left]
                    .total_cmp(&gradient[*right])
                    .then_with(|| right.cmp(left))
            });
        let Some(candidate) = candidate else {
            return Ok((solution, iterations));
        };
        passive[candidate] = true;
        iterations += 1;
        if iterations > maximum_iterations {
            return Err(PhaseScaleEstimationError::DidNotConverge);
        }

        loop {
            let active = passive
                .iter()
                .enumerate()
                .filter_map(|(index, selected)| selected.then_some(index))
                .collect::<Vec<_>>();
            // A boundary step can remove every passive variable. Restart the
            // outer search instead of asking nalgebra to decompose an m x 0
            // matrix; SVD deliberately rejects empty matrices.
            if active.is_empty() {
                break;
            }
            let active_matrix = DMatrix::from_fn(matrix.nrows(), active.len(), |row, column| {
                matrix[(row, active[column])]
            });
            let active_solution = active_matrix
                .svd(true, true)
                .solve(target, f64::EPSILON)
                .map_err(|_| PhaseScaleEstimationError::LinearSolve)?;
            if active_solution.iter().any(|value| !value.is_finite()) {
                return Err(PhaseScaleEstimationError::LinearSolve);
            }
            let mut candidate_solution = DVector::zeros(columns);
            for (active_index, value) in active.into_iter().zip(active_solution.iter()) {
                candidate_solution[active_index] = *value;
            }
            // Feasibility is a condition on the scale itself. The dual
            // gradient tolerance above has different units and can be very
            // large for high-count diffraction patterns, so it must not be
            // reused as a lower bound on phase scales.
            if (0..columns).all(|index| !passive[index] || candidate_solution[index] > 0.0) {
                solution = candidate_solution;
                break;
            }
            let alpha = (0..columns)
                .filter(|index| passive[*index] && candidate_solution[*index] <= 0.0)
                .map(|index| solution[index] / (solution[index] - candidate_solution[index]))
                .fold(1.0_f64, f64::min);
            solution += alpha * (candidate_solution - &solution);
            for index in 0..columns {
                if passive[index] && solution[index] <= 0.0 {
                    solution[index] = 0.0;
                    passive[index] = false;
                }
            }
            iterations += 1;
            if iterations > maximum_iterations {
                return Err(PhaseScaleEstimationError::DidNotConverge);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_count_design_accepts_a_small_positive_scale() {
        let matrix = DMatrix::from_row_slice(1, 1, &[1.0e8]);
        let target = DVector::from_vec(vec![1.0]);

        let (solution, iterations) =
            solve_non_negative_least_squares(&matrix, &target).expect("bounded scale solve");

        assert_eq!(iterations, 1);
        assert!((solution[0] - 1.0e-8).abs() < 1.0e-20);
    }
}
