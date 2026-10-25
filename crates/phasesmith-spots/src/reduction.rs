//! Selling reduction of a three-dimensional lattice, with exact integer transforms.
//! Four superbase vectors sum to zero. A positive pair dot product is removed by
//! negating one vector and adding it to the two non-partner vectors. The sum of
//! squared lengths decreases by twice that dot product. No structure is involved.
use crate::{SpotError, invalid};

/// Row-vector lattice basis.
pub type Basis = [[f64; 3]; 3];
/// Exact integer change of basis, reduced = transform * input.
pub type Transform = [[i64; 3]; 3];

/// A right-handed basis selected from an obtuse superbase, within tolerance.
pub struct Reduction {
    /// Reduced Cartesian row vectors, in input units.
    pub basis: Basis,
    /// Unimodular integer transform; absolute determinant one.
    pub transform: Transform,
    /// Number of Selling moves.
    pub iterations: usize,
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn determinant(a: Basis) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}
#[allow(clippy::cast_precision_loss)] // Coefficients are bounded by 1e9, exactly representable.
fn vectors(coefficients: [[i64; 3]; 4], basis: Basis) -> [[f64; 3]; 4] {
    coefficients.map(|r| std::array::from_fn(|j| (0..3).map(|k| r[k] as f64 * basis[k][j]).sum()))
}

/// Obtain a Selling-reduced superbase and retain its three shortest vectors.
/// The result preserves the supplied lattice; this does not prove a measured
/// diffraction lattice primitive, and is not a unique Niggli representative.
///
/// # Errors
/// Rejects nonfinite/singular bases, invalid tolerance/budgets, coefficients
/// exceeding 1e9, and nonconvergence. All integer updates are checked.
pub fn reduce(basis: Basis, tolerance: f64, max_iterations: usize) -> Result<Reduction, SpotError> {
    if basis
        .iter()
        .flatten()
        .any(|x| !x.is_finite() || x.abs() > 1e6)
        || !determinant(basis).is_finite()
        || determinant(basis).abs() < 1e-12
        || !tolerance.is_finite()
        || !(0.0..=1e-6).contains(&tolerance)
        || !(1..=10_000).contains(&max_iterations)
    {
        return Err(invalid(
            "invalid lattice, Selling tolerance or iteration budget",
        ));
    }
    let mut coefficients = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]];
    for iteration in 0..=max_iterations {
        let v = vectors(coefficients, basis);
        let scale = v.iter().map(|&r| dot(r, r)).fold(0.0, f64::max);
        let mut pair = None;
        let mut largest = tolerance * scale;
        for i in 0..4 {
            for j in i + 1..4 {
                let product = dot(v[i], v[j]);
                if product > largest {
                    largest = product;
                    pair = Some((i, j));
                }
            }
        }
        let Some((i, j)) = pair else {
            let mut order = [0, 1, 2, 3];
            order.sort_by(|&i, &j| {
                dot(v[i], v[i])
                    .total_cmp(&dot(v[j], v[j]))
                    .then(coefficients[i].cmp(&coefficients[j]))
            });
            let mut transform = std::array::from_fn(|k| coefficients[order[k]]);
            let mut reduced = std::array::from_fn(|k| v[order[k]]);
            if determinant(reduced) < 0.0 {
                reduced.swap(0, 1);
                transform.swap(0, 1);
            }
            return Ok(Reduction {
                basis: reduced,
                transform,
                iterations: iteration,
            });
        };
        if iteration == max_iterations {
            break;
        }
        let original = coefficients[i];
        for (k, row) in coefficients.iter_mut().enumerate() {
            if k != i && k != j {
                for c in 0..3 {
                    row[c] = row[c]
                        .checked_add(original[c])
                        .ok_or_else(|| invalid("Selling transform overflow"))?;
                }
            }
        }
        coefficients[i] = original.map(|x| -x);
        if coefficients
            .iter()
            .flatten()
            .any(|x| x.abs() > 1_000_000_000)
        {
            return Err(invalid("Selling transform coefficient budget exceeded"));
        }
    }
    Err(invalid(
        "Selling reduction did not converge within the iteration budget",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sheared_basis_preserves_lattice_and_obtuse_superbase() {
        let a = [[4.0, 0.0, 0.0], [32.0, 7.0, 0.0], [-44.0, 49.0, 11.0]];
        let r = reduce(a, 1e-12, 10_000).unwrap();
        assert!((determinant(r.basis) - 308.0).abs() < 1e-10);
        let v = vectors(
            [
                r.transform[0],
                r.transform[1],
                r.transform[2],
                std::array::from_fn(|j| -r.transform.iter().map(|x| x[j]).sum::<i64>()),
            ],
            a,
        );
        for i in 0..4 {
            for j in i + 1..4 {
                assert!(dot(v[i], v[j]) <= 1e-9);
            }
        }
    }
    #[test]
    fn invalid_inputs_and_nonconvergence() {
        assert!(reduce([[0.0; 3]; 3], 1e-12, 100).is_err());
        assert!(reduce([[1.0, 0.0, 0.0], [20.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 0.0, 1).is_err());
    }
}
