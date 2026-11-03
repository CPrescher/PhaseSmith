//! Small deterministic Cartesian matrix operations; no process-global state.

use crate::solution::{SolutionError, error};
use phasesmith_crystallography::{CellGeometry, Matrix3};

pub(crate) const IDENTITY: Matrix3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
pub(crate) fn transpose(a: Matrix3) -> Matrix3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
pub(crate) fn multiply(a: Matrix3, b: Matrix3) -> Matrix3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
pub(crate) fn mv(a: Matrix3, x: [f64; 3]) -> [f64; 3] {
    a.map(|row| dot(row, x))
}
pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
pub(crate) fn scaled(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|v| v * s)
}
pub(crate) fn matrix_add(a: &mut Matrix3, b: Matrix3) {
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] += b[i][j];
        }
    }
}
pub(crate) fn frobenius(a: Matrix3, b: Matrix3) -> f64 {
    a.into_iter()
        .flatten()
        .zip(b.into_iter().flatten())
        .map(|(a, b)| a * b)
        .sum()
}
pub(crate) fn conjugate(q: Matrix3, u: Matrix3) -> Matrix3 {
    multiply(multiply(q, u), transpose(q))
}

/// Jacobi rotations for a real symmetric 3x3 matrix. Eigenvectors are columns.
#[allow(clippy::many_single_char_names)]
pub(crate) fn eigen(mut a: Matrix3) -> ([f64; 3], Matrix3) {
    let mut vectors = IDENTITY;
    for _ in 0..48 {
        let (p, q) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|&(i, j), &(k, l)| a[i][j].abs().total_cmp(&a[k][l].abs()))
            .unwrap_or((0, 1));
        if a[p][q].abs()
            <= 2e-15
                * a.iter()
                    .flatten()
                    .fold(f64::MIN_POSITIVE, |x, y| x.max(y.abs()))
        {
            break;
        }
        let theta = 0.5 * (2. * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (s, c) = theta.sin_cos();
        let mut rotation = IDENTITY;
        rotation[p][p] = c;
        rotation[q][q] = c;
        rotation[p][q] = s;
        rotation[q][p] = -s;
        a = conjugate(transpose(rotation), a);
        vectors = multiply(vectors, rotation);
    }
    let values = [a[0][0], a[1][1], a[2][2]];
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| values[i].total_cmp(&values[j]).then(i.cmp(&j)));
    (
        order.map(|i| values[i]),
        std::array::from_fn(|i| order.map(|j| vectors[i][j])),
    )
}

pub(crate) fn pseudo_solve(a: Matrix3, rhs: [f64; 3]) -> [f64; 3] {
    let (values, vectors) = eigen(a);
    let maximum = values.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let rotated = mv(transpose(vectors), rhs);
    mv(
        vectors,
        std::array::from_fn(|i| {
            if values[i] > 1e-10 * maximum.max(1e-20) {
                rotated[i] / values[i]
            } else {
                0.
            }
        }),
    )
}

pub(crate) fn orthogonal_vectors(
    columns: impl IntoIterator<Item = [f64; 3]>,
    metric: Matrix3,
) -> Vec<[f64; 3]> {
    let columns: Vec<_> = columns.into_iter().collect();
    let threshold = columns
        .iter()
        .map(|&v| dot(v, mv(metric, v)).max(0.).sqrt())
        .fold(1.0_f64, f64::max)
        * 1e-9;
    let mut basis = Vec::new();
    for mut value in columns {
        for &other in &basis {
            value = sub(value, scaled(other, dot(other, mv(metric, value))));
        }
        let norm = dot(value, mv(metric, value)).max(0.).sqrt();
        if norm > threshold.max(f64::MIN_POSITIVE) {
            basis.push(scaled(value, 1. / norm));
        }
    }
    basis
}

/// Exact bounded shortest-image enumeration, rather than fractional-axis wrapping.
pub(crate) fn periodic_difference(
    geometry: &CellGeometry,
    difference: [f64; 3],
) -> Result<[f64; 3], SolutionError> {
    let initial = difference.map(|v| v - v.round());
    let metric = geometry.direct_metric;
    if [(0, 1), (0, 2), (1, 2)]
        .iter()
        .all(|&(i, j)| metric[i][j].abs() < 1e-13 * metric[i][i].sqrt() * metric[j][j].sqrt())
    {
        return Ok(initial);
    }
    let mut best = initial;
    let mut best2 = dot(best, mv(metric, best));
    let radius = best2.sqrt();
    let reciprocal = geometry.reciprocal_metric;
    let bounds = std::array::from_fn::<_, 3, _>(|i| radius * reciprocal[i][i].sqrt() + 1e-12);
    if bounds.iter().any(|b| !b.is_finite() || *b > 2047.) {
        return Err(error(
            "cell exceeds the 4096-image periodic-distance search budget",
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let ranges = std::array::from_fn::<_, 3, _>(|i| {
        let bound = bounds[i];
        (
            (initial[i] - bound).ceil() as i32,
            (initial[i] + bound).floor() as i32,
        )
    });
    let count = ranges
        .iter()
        .try_fold(1_i64, |n, (a, b)| n.checked_mul(i64::from(b - a + 1)))
        .unwrap_or(i64::MAX);
    if !(0..=4096).contains(&count) {
        return Err(error(
            "cell exceeds the 4096-image periodic-distance search budget",
        ));
    }
    for i in ranges[0].0..=ranges[0].1 {
        for j in ranges[1].0..=ranges[1].1 {
            for k in ranges[2].0..=ranges[2].1 {
                let point = sub(initial, [f64::from(i), f64::from(j), f64::from(k)]);
                let d2 = dot(point, mv(metric, point));
                if d2 < best2 {
                    best = point;
                    best2 = d2;
                }
            }
        }
    }
    Ok(best)
}

pub(crate) fn exponential(
    h: Matrix3,
    basis: &[Matrix3],
) -> Result<(Matrix3, Vec<Matrix3>), SolutionError> {
    let (lambda, v) = eigen(h);
    if lambda
        .iter()
        .any(|&x| !x.is_finite() || !(1e-5_f64.ln() - 1e-12..=0.2_f64.ln() + 1e-12).contains(&x))
    {
        return Err(error(
            "ADP eigenvalues must lie in [1e-5,0.2] angstrom squared",
        ));
    }
    let values = lambda.map(f64::exp);
    let diag =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { values[i] } else { 0. }));
    let u = conjugate(v, diag);
    let derivatives = basis
        .iter()
        .map(|&direction| {
            let mut local = conjugate(transpose(v), direction);
            for i in 0..3 {
                for j in 0..3 {
                    let half = (lambda[i] - lambda[j]) * 0.5;
                    let sinc = if half.abs() < 1e-5 {
                        1. + half * half / 6. + half.powi(4) / 120.
                    } else {
                        half.sinh() / half
                    };
                    local[i][j] *= (lambda[i] * 0.5 + lambda[j] * 0.5).exp() * sinc;
                }
            }
            conjugate(v, local)
        })
        .collect();
    Ok((u, derivatives))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spectral_reconstruction_and_degenerate_frechet() {
        let h = [[-3.8, 0.2, -0.1], [0.2, -4.1, 0.3], [-0.1, 0.3, -3.9]];
        let (values, v) = eigen(h);
        let rebuilt = conjugate(
            v,
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { values[i] } else { 0. })),
        );
        for (a, b) in h.iter().flatten().zip(rebuilt.iter().flatten()) {
            assert!((a - b).abs() < 2e-13);
        }
        let (u, d) =
            exponential(IDENTITY.map(|r| r.map(|x| x * 0.01_f64.ln())), &[IDENTITY]).unwrap();
        assert!((u[0][0] - 0.01).abs() < 1e-15);
        assert!((d[0][1][1] - 0.01).abs() < 1e-15);
    }
    #[test]
    fn closest_periodic_image_and_skew_budget() {
        use phasesmith_crystallography::UnitCell;
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 6.,
            c_angstrom: 7.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 25.,
        };
        let geometry = cell.geometry().unwrap();
        let difference = [0.49, 0.47, 0.1];
        let actual = periodic_difference(&geometry, difference).unwrap();
        let mut expected = f64::INFINITY;
        for i in -3..=3 {
            for j in -3..=3 {
                for k in -3..=3 {
                    let delta = sub(difference, [f64::from(i), f64::from(j), f64::from(k)]);
                    expected = expected.min(dot(delta, mv(geometry.direct_metric, delta)));
                }
            }
        }
        assert!((dot(actual, mv(geometry.direct_metric, actual)) - expected).abs() < 1e-12);
        assert!(dot(difference, mv(geometry.direct_metric, difference)) > expected + 1.);
        // A very long third axis must not hide the first two axes' skew.
        let long_axis = UnitCell {
            c_angstrom: 1e9,
            ..cell
        }
        .geometry()
        .unwrap();
        let delta = periodic_difference(&long_axis, [0.49, 0.47, 0.]).unwrap();
        assert!(dot(delta, mv(long_axis.direct_metric, delta)) < 2.);
        let skew = UnitCell {
            gamma_deg: 0.001,
            ..cell
        }
        .geometry()
        .unwrap();
        assert!(
            periodic_difference(&skew, difference)
                .unwrap_err()
                .to_string()
                .contains("4096-image")
        );
    }
}
