//! Small convex subproblems for the existing coupled CW width domain.
//!
//! This constructs proposal constraints; every trial still goes through the
//! complete physical-model evaluation and ordinary strict descent test.

use crate::{ConstraintDerivativeMatrix, ParameterSet, RietveldCalculation, RietveldInput};
use nalgebra::{DMatrix, DVector};

/// Rows encode `a . step >= lower` in scaled free coordinates.
pub(crate) struct StepInequalities {
    pub rows: Vec<Vec<f64>>,
    pub lower: Vec<f64>,
}

impl StepInequalities {
    fn push(&mut self, row: Vec<f64>, lower: f64) {
        if lower.is_finite() && row.iter().any(|v| *v != 0.0) {
            self.rows.push(row);
            self.lower.push(lower);
        }
    }
}

/// Linearize Gaussian and Lorentzian instrument widths at every calculated
/// component reflection. Only fixed cells and optional additive zero shift
/// are covered; other position models retain the established solver.
pub(crate) fn width_inequalities(
    input: &RietveldInput,
    calculation: &RietveldCalculation,
    parameters: &ParameterSet,
    derivative: &ConstraintDerivativeMatrix,
) -> Option<StepInequalities> {
    let count = derivative.columns;
    let reflection_count = calculation
        .phases
        .iter()
        .map(|phase| phase.result.two_theta_deg.len())
        .sum::<usize>();
    if reflection_count.saturating_add(parameters.specs().len()) > 2048 {
        return None;
    }
    if count == 0 || count > 64 {
        return None;
    }
    let row = |index: usize| &derivative.values[index * count..(index + 1) * count];
    let mut width_indices = [None; 6];
    for (index, spec) in parameters.specs().iter().enumerate() {
        if !row(index).iter().any(|v| *v != 0.0) {
            continue;
        }
        let key = spec.key();
        if key.module() == "lattice" {
            return None;
        }
        if key.module() == "instrument" {
            let slot = match key.name() {
                "u_deg2" => 0,
                "v_deg2" => 1,
                "w_deg2" => 2,
                "x_deg" => 3,
                "y_deg" => 4,
                "zero_shift_deg" => 5,
                _ => return None,
            };
            width_indices[slot] = Some(index);
        }
    }
    if width_indices[..5].iter().all(Option::is_none) {
        return None;
    }
    let mut result = StepInequalities {
        rows: Vec::new(),
        lower: Vec::new(),
    };
    let instrument = input.instrument;
    for phase in &calculation.phases {
        for &position in &phase.result.two_theta_deg {
            let theta = position * std::f64::consts::PI / 360.0;
            let tangent = theta.tan();
            let secant = theta.cos().recip();
            let gaussian = instrument.u_deg2 * tangent * tangent
                + instrument.v_deg2 * tangent
                + instrument.w_deg2;
            let lorentzian = instrument.x_deg * secant + instrument.y_deg * tangent;
            let angular = std::f64::consts::PI / 360.0;
            let coefficients = [
                [
                    tangent * tangent,
                    tangent,
                    1.0,
                    0.0,
                    0.0,
                    (2.0 * instrument.u_deg2 * tangent + instrument.v_deg2)
                        * secant
                        * secant
                        * angular,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    secant,
                    tangent,
                    (instrument.x_deg * secant * tangent + instrument.y_deg * secant * secant)
                        * angular,
                ],
            ];
            for (coefficients, value) in coefficients.iter().zip([gaussian, lorentzian]) {
                let mut free_row = vec![0.0; count];
                for (index, coefficient) in width_indices.iter().zip(coefficients) {
                    if let Some(index) = index {
                        for (target, chain) in free_row.iter_mut().zip(row(*index)) {
                            *target += coefficient * chain;
                        }
                    }
                }
                // Fraction to the boundary preserves strict Gaussian positivity.
                result.push(free_row, -0.99 * value);
            }
        }
    }
    // Include dependent physical bounds, not just independent-variable boxes.
    for (index, spec) in parameters.specs().iter().enumerate() {
        result.push(row(index).to_vec(), spec.bounds().lower() - spec.value());
        result.push(
            row(index).iter().map(|v| -v).collect(),
            spec.value() - spec.bounds().upper(),
        );
    }
    Some(result)
}

/// Primal active-set quadratic minimization from the feasible zero step.
/// A QR null space avoids subtracting ill-conditioned inverse-Hessian products.
/// Failure to verify a solution returns `None`, preserving the caller's fallback.
#[allow(clippy::too_many_lines)]
pub(crate) fn solve_quadratic(
    normal: &DMatrix<f64>,
    rhs: &[f64],
    constraints: &StepInequalities,
    tolerance: f64,
) -> Option<Vec<f64>> {
    let count = rhs.len();
    if count == 0 || count > 64 || normal.nrows() != count || normal.ncols() != count {
        return None;
    }
    let scale = DVector::from_iterator(count, (0..count).map(|i| normal[(i, i)].sqrt()));
    if scale.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return None;
    }
    let hessian = DMatrix::from_fn(count, count, |i, j| normal[(i, j)] / scale[i] / scale[j]);
    let initial_gradient =
        DVector::from_iterator(count, rhs.iter().enumerate().map(|(i, v)| -v / scale[i]));
    let mut rows = Vec::new();
    let mut bounds = Vec::new();
    for (row, &bound) in constraints.rows.iter().zip(&constraints.lower) {
        let mut row =
            DVector::from_iterator(count, row.iter().enumerate().map(|(i, v)| v / scale[i]));
        let norm = row.norm();
        if norm > 0.0 && norm.is_finite() {
            row /= norm;
            rows.push(row);
            bounds.push(bound / norm);
        }
    }
    if bounds.iter().any(|b| !b.is_finite() || *b > 1e-12) {
        return None;
    }
    let mut step = DVector::zeros(count);
    let mut active: Vec<usize> = Vec::new();
    for _ in 0..(20 * (count + 1)) {
        let gradient = &hessian * &step + &initial_gradient;
        let (direction, multipliers) = if active.is_empty() {
            (
                -hessian.clone().cholesky()?.solve(&gradient),
                DVector::zeros(0),
            )
        } else {
            let rank = active.len();
            let basis = DMatrix::from_fn(count, count, |i, j| {
                if j < rank { rows[active[j]][i] } else { 0.0 }
            })
            .qr();
            let r = basis.r();
            if (0..rank).any(|i| r[(i, i)].abs() < 1e-10) {
                return None;
            }
            let q = basis.q();
            let z = q.columns(rank, count - rank);
            let direction = if rank == count {
                DVector::zeros(count)
            } else {
                let reduced = z.transpose() * &hessian * z;
                -z * reduced.cholesky()?.solve(&(z.transpose() * &gradient))
            };
            let multipliers = r
                .view((0, 0), (rank, rank))
                .solve_upper_triangular(&(q.columns(0, rank).transpose() * &gradient))?;
            (direction, multipliers)
        };
        if direction.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if direction.norm() <= 1e-9 * (1.0 + step.norm()) {
            if let Some((index, _)) = multipliers
                .iter()
                .enumerate()
                .filter(|(_, v)| **v < -1e-8)
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
            {
                active.remove(index);
                continue;
            }
            let mut residual = gradient;
            for (&index, multiplier) in active.iter().zip(&multipliers) {
                residual -= &rows[index] * *multiplier;
            }
            if residual.norm() > tolerance * initial_gradient.norm().max(1.0)
                || rows
                    .iter()
                    .zip(&bounds)
                    .any(|(a, b)| a.dot(&step) - b < -1e-10 * (1.0 + b.abs()))
            {
                return None;
            }
            return Some(step.iter().zip(&scale).map(|(v, s)| v / s).collect());
        }
        let mut alpha = 1.0;
        let mut blocker = None;
        for (index, (row, bound)) in rows.iter().zip(&bounds).enumerate() {
            if active.contains(&index) {
                continue;
            }
            let change = row.dot(&direction);
            if change < -1e-12 {
                let candidate = (row.dot(&step) - bound).max(0.0) / -change;
                if candidate < alpha {
                    alpha = candidate;
                    blocker = Some(index);
                }
            }
        }
        step += alpha * direction;
        if let Some(index) = blocker {
            if active.len() == count {
                return None;
            }
            active.push(index);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupled_step_and_bound_have_verified_kkt_solution() {
        let h = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let mut constraints = StepInequalities {
            rows: vec![vec![1.0, 2.0]],
            lower: vec![0.0],
        };
        let result = solve_quadratic(&h, &[2.0, -2.0], &constraints, 1e-10).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-10 && (result[1] + 1.0).abs() < 1e-10);
        constraints.push(vec![-1.0, 0.0], -1.0);
        let result = solve_quadratic(&h, &[2.0, -2.0], &constraints, 1e-10).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10 && (result[1] + 0.5).abs() < 1e-10);
    }

    #[test]
    fn deterministic_quadratics_match_exhaustive_active_sets() {
        for seed in 1..=32 {
            let b = DMatrix::from_fn(3, 3, |i, j| {
                (f64::from(seed) * 0.7 + f64::from(u32::try_from(i * 3 + j).unwrap())).sin()
            });
            let h = b.transpose() * b + DMatrix::identity(3, 3) * 0.25;
            let rhs = DVector::from_fn(3, |i, _| {
                (f64::from(seed) + f64::from(u32::try_from(i).unwrap()) * 0.3).cos()
            });
            let constraints = StepInequalities {
                rows: vec![
                    vec![1.0, 2.0, 0.0],
                    vec![0.0, -1.0, 1.0],
                    vec![-1.0, 0.0, 0.0],
                    vec![0.0, 0.0, -1.0],
                ],
                lower: vec![-0.1, -0.2, -0.4, -0.3],
            };
            let result =
                DVector::from_vec(solve_quadratic(&h, rhs.as_slice(), &constraints, 1e-9).unwrap());
            let objective = |x: &DVector<f64>| 0.5 * x.dot(&(&h * x)) - rhs.dot(x);
            let mut best = f64::INFINITY;
            for mask in 0_usize..16 {
                let active = (0..4).filter(|i| mask & (1 << i) != 0).collect::<Vec<_>>();
                let k = active.len();
                let mut system = DMatrix::zeros(3 + k, 3 + k);
                system.view_mut((0, 0), (3, 3)).copy_from(&h);
                let mut right = DVector::zeros(3 + k);
                right.rows_mut(0, 3).copy_from(&rhs);
                for (index, &row) in active.iter().enumerate() {
                    right[3 + index] = constraints.lower[row];
                    for column in 0..3 {
                        system[(column, 3 + index)] = -constraints.rows[row][column];
                        system[(3 + index, column)] = constraints.rows[row][column];
                    }
                }
                let Some(solution) = system.lu().solve(&right) else {
                    continue;
                };
                let point = solution.rows(0, 3).into_owned();
                if solution.rows(3, k).iter().any(|v| *v < -1e-8) {
                    continue;
                }
                if constraints
                    .rows
                    .iter()
                    .zip(&constraints.lower)
                    .any(|(a, b)| {
                        a.iter().zip(point.iter()).map(|(x, y)| x * y).sum::<f64>() < b - 1e-8
                    })
                {
                    continue;
                }
                best = best.min(objective(&point));
            }
            assert!((objective(&result) - best).abs() < 1e-9, "seed {seed}");
        }
    }
}
