//! Dense Cholesky solves for bounded native fits and small symmetric eigensystems.
use crate::{SpotError, invalid};

pub(super) fn cholesky(matrix: &[f64], n: usize) -> Result<Vec<f64>, SpotError> {
    let mut l = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * n + j];
            for k in 0..j {
                value -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                if !value.is_finite() || value <= 0. {
                    return Err(invalid("nonpositive geometry normal"));
                }
                value.sqrt()
            } else {
                value / l[j * n + j]
            };
        }
    }
    Ok(l)
}

pub(super) fn solve(matrix: &[f64], rhs: &[f64]) -> Result<Vec<f64>, SpotError> {
    let n = rhs.len();
    let l = cholesky(matrix, n)?;
    let mut x = rhs.to_vec();
    for i in 0..n {
        for j in 0..i {
            x[i] -= l[i * n + j] * x[j];
        }
        x[i] /= l[i * n + i];
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= l[j * n + i] * x[j];
        }
        x[i] /= l[i * n + i];
    }
    Ok(x)
}

/// Jacobi eigensystem of a real symmetric small matrix. Eigenvectors are columns.
pub(super) fn eigen(matrix: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), SpotError> {
    let mut a = matrix.to_vec();
    let mut vectors = vec![0.; n * n];
    for i in 0..n {
        vectors[i * n + i] = 1.;
    }
    for _ in 0..100 * n * n {
        let mut largest = 0.;
        let (mut p, mut q) = (0, 0);
        let mut scale: f64 = 1e-300;
        for i in 0..n {
            scale = scale.max(a[i * n + i].abs());
            for j in i + 1..n {
                if a[i * n + j].abs() > largest {
                    largest = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= 2e-15 * scale {
            let mut order: Vec<_> = (0..n).collect();
            order.sort_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j]));
            let values = order.iter().map(|&i| a[i * n + i]).collect();
            let sorted = (0..n * n)
                .map(|i| vectors[(i / n) * n + order[i % n]])
                .collect();
            return Ok((values, sorted));
        }
        let tau = (a[q * n + q] - a[p * n + p]) / (2. * a[p * n + q]);
        let t = if tau >= 0. {
            1. / (tau + tau.hypot(1.))
        } else {
            -1. / (-tau + tau.hypot(1.))
        };
        let c = 1. / (1. + t * t).sqrt();
        let s = t * c;
        let off = a[p * n + q];
        a[p * n + p] -= t * off;
        a[q * n + q] += t * off;
        a[p * n + q] = 0.;
        a[q * n + p] = 0.;
        for k in 0..n {
            if k != p && k != q {
                let x = a[k * n + p];
                let y = a[k * n + q];
                a[k * n + p] = c * x - s * y;
                a[p * n + k] = a[k * n + p];
                a[k * n + q] = s * x + c * y;
                a[q * n + k] = a[k * n + q];
            }
            let x = vectors[k * n + p];
            let y = vectors[k * n + q];
            vectors[k * n + p] = c * x - s * y;
            vectors[k * n + q] = s * x + c * y;
        }
    }
    Err(invalid("geometry eigensystem did not converge"))
}
