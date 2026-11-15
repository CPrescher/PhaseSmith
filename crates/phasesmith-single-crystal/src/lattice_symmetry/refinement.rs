//! Symmetry-constrained cell refinement on indexed reciprocal positions.
//!
//! Model: `q = R (h B₀(p))`, row index `h` in the conventional setting,
//! `B₀(p)` the reciprocal basis of the cell in the standard Cartesian frame
//! of [`cell_basis`] and `R` a rotation. Parameters are a
//! small rotation (three, re-linearised every cycle) and the free cell
//! parameters `p` of the lattice system (lengths in Å, angles in radians).
//! The derivative of `q` with respect to the rotation is `eₖ × q`; with
//! respect to `p` it is `R (h ∂B₀/∂p)`, with `∂B₀/∂p` from central
//! differences of the 3×3 matrix `B₀` (relative step 10⁻⁶, truncation error
//! about 10⁻¹² relative), so no per-reflection numerical derivative is taken.
//! Gauss–Newton steps are halved until the cost does not increase. The
//! covariance is `s² (JᵀJ)⁻¹`, `s² = Σ|Δq|² / (3N − P)`.
use super::{
    Bravais, LatticeSystem, Result, best_rotation, cell_basis, cell_parameters, constrain_cell,
    dot, rotate_rows,
};
use crate::grains::{M3, V3};
use crate::spots::SpotError;
use nalgebra::{DMatrix, DVector, Matrix3};

/// Iteration settings of [`refine_cell`]. Non-exhaustive: start from
/// [`RefineOptions::default`] and assign fields.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RefineOptions {
    /// Gauss–Newton cycles, 1 through 1000; default 50.
    pub max_iterations: usize,
    /// Converged when a cycle lowers the residual sum of squares by at most
    /// this fraction; default 10⁻¹², range [0, 10⁻³].
    pub tolerance: f64,
}
impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-12,
        }
    }
}
impl RefineOptions {
    /// Rejects out-of-range settings.
    ///
    /// # Errors
    /// `max_iterations` outside 1..=1000 or `tolerance` outside [0, 10⁻³].
    pub fn check(&self) -> Result<()> {
        if !(1..=1000).contains(&self.max_iterations) || !(0. ..=1e-3).contains(&self.tolerance) {
            return Err(SpotError(
                "max_iterations must lie in [1,1000] and tolerance in [0,1e-3]".into(),
            ));
        }
        Ok(())
    }
}

/// How a refinement ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Convergence {
    /// The relative decrease fell below the tolerance, or no step along the
    /// Gauss–Newton direction lowered the cost (the minimum to precision).
    Converged,
    /// `max_iterations` cycles ran without meeting the criterion.
    IterationLimit,
}

/// Result of [`refine_cell`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CellRefinement {
    /// The constrained Bravais type refined.
    pub bravais: Bravais,
    /// Refined conventional direct row basis, ångström; exactly the
    /// constrained cell, in the refined orientation.
    pub conventional: M3,
    /// a, b, c (Å), α, β, γ (degrees), constraints exact.
    pub cell: [f64; 6],
    /// Standard uncertainties of `cell` from the fit covariance; a length
    /// tied to another carries that one's uncertainty, a fixed angle 0.
    pub cell_su: [f64; 6],
    /// Cell volume, cubic ångström.
    pub volume: f64,
    /// Standard uncertainty of `volume`.
    pub volume_su: f64,
    /// Root mean square of `|q_observed − q_calculated|`, inverse ångström.
    pub rms: f64,
    /// `|q_observed − q_calculated|` per position, inverse ångström.
    pub residuals: Vec<f64>,
    /// Number of positions fitted.
    pub used: usize,
    /// Free cell parameters fitted (the orientation adds three).
    pub parameters: usize,
    /// Gauss–Newton cycles run.
    pub iterations: usize,
    /// How the iteration ended.
    pub status: Convergence,
}

fn inverse_transpose(a: M3) -> Option<M3> {
    let m = Matrix3::from_fn(|i, j| a[i][j]);
    let inv = m.try_inverse()?.transpose();
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| inv[(i, j)])
    }))
}
fn row(h: V3, b: M3) -> V3 {
    std::array::from_fn(|j| (0..3).map(|k| h[k] * b[k][j]).sum())
}

/// Conventional indices of the positions within `tolerance` (inverse
/// ångström) of a node of `conventional` that the type's centring allows;
/// None for the rest and for 000. Indices are the rounded fractional
/// coordinates `q · aᵢ`, so `conventional` should be close to the truth.
///
/// # Errors
/// Non-finite or singular basis, non-positive or non-finite tolerance.
pub fn index(
    q: &[V3],
    conventional: M3,
    bravais: Bravais,
    tolerance: f64,
) -> Result<Vec<Option<[i64; 3]>>> {
    if !(tolerance > 0. && tolerance.is_finite()) {
        return Err(SpotError("index tolerance must be positive".into()));
    }
    let b = conventional_reciprocal(conventional)?;
    let centring = bravais.centring();
    Ok(q.iter()
        .map(|&q| {
            if !q.iter().all(|x| x.is_finite()) {
                return None;
            }
            let f: V3 = conventional.map(|a| dot(q, a));
            if f.iter().any(|x| x.abs() > 1e6) {
                return None;
            }
            let h = f.map(f64::round);
            let hkl = h.map(|x| x as i64);
            let p = row(h, b);
            let d = (0..3).map(|j| (q[j] - p[j]).powi(2)).sum::<f64>().sqrt();
            (hkl != [0; 3] && centring.allows(hkl) && d < tolerance).then_some(hkl)
        })
        .collect())
}
fn conventional_reciprocal(a: M3) -> Result<M3> {
    if a.iter().flatten().any(|x| !x.is_finite()) {
        return Err(SpotError("non-finite basis".into()));
    }
    inverse_transpose(a)
        .filter(|b| b.iter().flatten().all(|x| x.is_finite()))
        .ok_or_else(|| SpotError("singular basis".into()))
}

/// Free cell parameters (Å, radians) of a constrained cell (Å, degrees).
fn free(system: LatticeSystem, c: [f64; 6]) -> Vec<f64> {
    let r = f64::to_radians;
    match system {
        LatticeSystem::Triclinic => vec![c[0], c[1], c[2], r(c[3]), r(c[4]), r(c[5])],
        LatticeSystem::Monoclinic => vec![c[0], c[1], c[2], r(c[4])],
        LatticeSystem::Orthorhombic => vec![c[0], c[1], c[2]],
        LatticeSystem::Tetragonal | LatticeSystem::Rhombohedral | LatticeSystem::Hexagonal => {
            vec![c[0], c[2]]
        }
        LatticeSystem::Cubic => vec![c[0]],
    }
}
/// Which free parameter each of a, b, c, α, β, γ is; None when fixed.
fn source(system: LatticeSystem) -> [Option<usize>; 6] {
    match system {
        LatticeSystem::Triclinic => [Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)],
        LatticeSystem::Monoclinic => [Some(0), Some(1), Some(2), None, Some(3), None],
        LatticeSystem::Orthorhombic => [Some(0), Some(1), Some(2), None, None, None],
        LatticeSystem::Tetragonal | LatticeSystem::Rhombohedral | LatticeSystem::Hexagonal => {
            [Some(0), Some(0), Some(1), None, None, None]
        }
        LatticeSystem::Cubic => [Some(0), Some(0), Some(0), None, None, None],
    }
}
/// Cell (Å, degrees) from free parameters (Å, radians).
fn expand(system: LatticeSystem, p: &[f64]) -> [f64; 6] {
    let fixed = constrain_cell(system, [1., 1., 1., 90., 90., 90.]);
    let src = source(system);
    std::array::from_fn(|k| match src[k] {
        Some(i) if k >= 3 => p[i].to_degrees(),
        Some(i) => p[i],
        None => fixed[k],
    })
}
fn reciprocal(system: LatticeSystem, p: &[f64]) -> Option<M3> {
    inverse_transpose(cell_basis(expand(system, p))?)
}
fn rotate(r: &Matrix3<f64>, v: V3) -> V3 {
    std::array::from_fn(|i| (0..3).map(|k| r[(i, k)] * v[k]).sum())
}
fn rodrigues(w: V3) -> Matrix3<f64> {
    let theta = dot(w, w).sqrt();
    let k = Matrix3::new(0., -w[2], w[1], w[2], 0., -w[0], -w[1], w[0], 0.);
    let (s, c) = if theta < 1e-8 {
        (1. - theta * theta / 6., 0.5 - theta * theta / 24.)
    } else {
        (theta.sin() / theta, (1. - theta.cos()) / (theta * theta))
    };
    Matrix3::identity() + k * s + k * k * c
}

struct Model<'a> {
    system: LatticeSystem,
    q: &'a [V3],
    h: Vec<V3>,
}
impl Model<'_> {
    fn residuals(&self, p: &[f64], r: &Matrix3<f64>) -> Option<Vec<V3>> {
        let b = reciprocal(self.system, p)?;
        Some(
            self.q
                .iter()
                .zip(&self.h)
                .map(|(q, &h)| {
                    let c = rotate(r, row(h, b));
                    std::array::from_fn(|j| q[j] - c[j])
                })
                .collect(),
        )
    }
    fn cost(&self, p: &[f64], r: &Matrix3<f64>) -> Option<f64> {
        let s = self
            .residuals(p, r)?
            .iter()
            .map(|d| dot(*d, *d))
            .sum::<f64>();
        s.is_finite().then_some(s)
    }
    /// Jacobian of the calculated positions (3N × (P + 3)): cell parameters,
    /// then the rotation increment.
    fn jacobian(&self, p: &[f64], r: &Matrix3<f64>) -> Option<DMatrix<f64>> {
        let n = p.len();
        let b = reciprocal(self.system, p)?;
        let mut derivatives = Vec::with_capacity(n);
        for k in 0..n {
            let step = 1e-6 * p[k].abs().max(1.);
            let mut plus = p.to_vec();
            let mut minus = p.to_vec();
            plus[k] += step;
            minus[k] -= step;
            let (bp, bm) = (
                reciprocal(self.system, &plus)?,
                reciprocal(self.system, &minus)?,
            );
            let d: M3 = std::array::from_fn(|i| {
                std::array::from_fn(|j| (bp[i][j] - bm[i][j]) / (2. * step))
            });
            derivatives.push(d);
        }
        let mut j = DMatrix::zeros(3 * self.h.len(), n + 3);
        for (i, &h) in self.h.iter().enumerate() {
            let c = rotate(r, row(h, b));
            for (k, d) in derivatives.iter().enumerate() {
                let v = rotate(r, row(h, *d));
                for a in 0..3 {
                    j[(3 * i + a, k)] = v[a];
                }
            }
            // e_k × c
            let cross = [[0., -c[2], c[1]], [c[2], 0., -c[0]], [-c[1], c[0], 0.]];
            for a in 0..3 {
                for k in 0..3 {
                    j[(3 * i + a, n + k)] = cross[k][a];
                }
            }
        }
        Some(j)
    }
}

/// Column-scaled normal matrix inverse, `(JᵀJ)⁻¹`; None when singular.
fn normal_inverse(j: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let normal = j.transpose() * j;
    let scale = DVector::from_iterator(
        normal.nrows(),
        (0..normal.nrows()).map(|i| normal[(i, i)].sqrt()),
    );
    if scale
        .iter()
        .any(|&x| x.is_nan() || x <= 0. || !x.is_finite())
    {
        return None;
    }
    let scaled = DMatrix::from_fn(normal.nrows(), normal.ncols(), |a, b| {
        normal[(a, b)] / (scale[a] * scale[b])
    });
    let svd = scaled.clone().svd(false, false);
    let max = svd.singular_values.max();
    if svd.singular_values.min() <= max * 1e-12 {
        return None;
    }
    let inverse = scaled.cholesky()?.inverse();
    Some(DMatrix::from_fn(normal.nrows(), normal.ncols(), |a, b| {
        inverse[(a, b)] / (scale[a] * scale[b])
    }))
}

/// Least-squares fit of the orientation and the free cell parameters of
/// `bravais`'s lattice system to observed reciprocal positions `q` (inverse
/// ångström) with conventional indices `hkl`, starting from `start` (any
/// conventional basis of the setting; its cell is constrained first and
/// oriented onto it).
///
/// Free parameters: cubic a; tetragonal, hexagonal and rhombohedral (hexagonal
/// axes) a, c; orthorhombic a, b, c; monoclinic (unique b) a, b, c, β;
/// triclinic all six. All positions carry equal weight.
///
/// `start` must be right handed, as every [`BravaisMatch`](super::BravaisMatch)
/// basis is, and every index must satisfy the type's centring condition
/// ([`Centring::allows`](super::Centring::allows)); [`index`] only returns such
/// indices.
///
/// # Errors
/// A left-handed or singular `start`, an index forbidden by the centring,
/// mismatched lengths, non-finite positions, a 000 index, fewer than
/// `(P + 3) / 3 + 1` positions, an impossible starting cell, invalid options,
/// or indices that do not determine the parameters (singular normal matrix).
pub fn refine_cell(
    q: &[V3],
    hkl: &[[i64; 3]],
    start: M3,
    bravais: Bravais,
    options: &RefineOptions,
) -> Result<CellRefinement> {
    options.check()?;
    let system = bravais.system();
    let parameters = system.free_parameters();
    if q.len() != hkl.len() {
        return Err(SpotError("q and hkl differ in length".into()));
    }
    if q.iter().flatten().any(|x| !x.is_finite()) {
        return Err(SpotError("non-finite reciprocal position".into()));
    }
    if hkl
        .iter()
        .any(|h| *h == [0; 3] || h.iter().any(|x| x.abs() > 1_000_000))
    {
        return Err(SpotError("indices must be nonzero and at most 1e6".into()));
    }
    if 3 * q.len() <= parameters + 3 {
        return Err(SpotError(
            "too few positions for the free parameters".into(),
        ));
    }
    conventional_reciprocal(start)?;
    if super::det(start) <= 0. {
        return Err(SpotError(
            "start must be a right-handed basis (positive determinant)".into(),
        ));
    }
    let centring = bravais.centring();
    if let Some(h) = hkl.iter().find(|h| !centring.allows(**h)) {
        return Err(SpotError(format!(
            "index {h:?} is forbidden by the {:?} centring of {}",
            centring,
            bravais.symbol()
        )));
    }
    let model = Model {
        system,
        q,
        h: hkl.iter().map(|h| h.map(|x| x as f64)).collect(),
    };
    let mut p = free(system, constrain_cell(system, cell_parameters(start)));
    let a0 = cell_basis(expand(system, &p))
        .ok_or_else(|| SpotError("impossible starting cell".into()))?;
    let mut r = best_rotation(a0, start);
    let mut ssr = model
        .cost(&p, &r)
        .ok_or_else(|| SpotError("impossible starting cell".into()))?;
    let singular = || SpotError("indices do not determine the cell parameters".into());
    let mut status = Convergence::IterationLimit;
    let mut iterations = 0;
    while iterations < options.max_iterations {
        iterations += 1;
        let j = model.jacobian(&p, &r).ok_or_else(singular)?;
        let residual = model.residuals(&p, &r).ok_or_else(singular)?;
        let rv = DVector::from_iterator(residual.len() * 3, residual.into_iter().flatten());
        let inverse = normal_inverse(&j).ok_or_else(singular)?;
        let delta = inverse * (j.transpose() * rv);
        let mut lambda = 1.;
        let mut accepted = None;
        for _ in 0..30 {
            let trial: Vec<f64> = (0..parameters).map(|k| p[k] + lambda * delta[k]).collect();
            let w = [0, 1, 2].map(|k| lambda * delta[parameters + k]);
            let tr = rodrigues(w) * r;
            if let Some(c) = model.cost(&trial, &tr).filter(|&c| c <= ssr) {
                accepted = Some((trial, tr, c));
                break;
            }
            lambda *= 0.5;
        }
        let Some((trial, tr, c)) = accepted else {
            status = Convergence::Converged;
            break;
        };
        let decrease = ssr - c;
        (p, r) = (trial, tr);
        let previous = ssr;
        ssr = c;
        if decrease <= options.tolerance * previous {
            status = Convergence::Converged;
            break;
        }
    }

    let n = q.len();
    let j = model.jacobian(&p, &r).ok_or_else(singular)?;
    let inverse = normal_inverse(&j).ok_or_else(singular)?;
    let s2 = ssr / (3 * n - parameters - 3) as f64;
    let cell = expand(system, &p);
    let src = source(system);
    let cell_su = std::array::from_fn(|k| {
        src[k].map_or(0., |i| {
            let su = (s2 * inverse[(i, i)]).max(0.).sqrt();
            if k >= 3 { su.to_degrees() } else { su }
        })
    });
    let a0 = cell_basis(cell).ok_or_else(singular)?;
    let volume_of = |p: &[f64]| cell_basis(expand(system, p)).map(super::det);
    let volume = super::det(a0);
    let mut gradient = vec![0.; parameters];
    for k in 0..parameters {
        let step = 1e-6 * p[k].abs().max(1.);
        let mut plus = p.clone();
        let mut minus = p.clone();
        plus[k] += step;
        minus[k] -= step;
        gradient[k] = (volume_of(&plus).ok_or_else(singular)?
            - volume_of(&minus).ok_or_else(singular)?)
            / (2. * step);
    }
    let mut variance = 0.;
    for a in 0..parameters {
        for b in 0..parameters {
            variance += gradient[a] * inverse[(a, b)] * gradient[b];
        }
    }
    let residuals: Vec<f64> = model
        .residuals(&p, &r)
        .ok_or_else(singular)?
        .iter()
        .map(|d| dot(*d, *d).sqrt())
        .collect();
    Ok(CellRefinement {
        bravais,
        conventional: rotate_rows(a0, &r),
        cell,
        cell_su,
        volume,
        volume_su: (s2 * variance).max(0.).sqrt(),
        rms: (ssr / n as f64).sqrt(),
        residuals,
        used: n,
        parameters,
        iterations,
        status,
    })
}

#[cfg(test)]
pub(super) fn jacobian_for_test(
    system: LatticeSystem,
    q: &[V3],
    hkl: &[V3],
    p: &[f64],
    w: V3,
) -> Option<(DMatrix<f64>, DMatrix<f64>)> {
    // Analytic rotation and matrix-difference cell columns against a
    // full-model central difference of the calculated positions.
    let model = Model {
        system,
        q,
        h: hkl.to_vec(),
    };
    let r = rodrigues(w);
    let analytic = model.jacobian(p, &r)?;
    let calc = |p: &[f64], r: &Matrix3<f64>| -> Option<Vec<f64>> {
        Some(
            model
                .residuals(p, r)?
                .iter()
                .zip(q)
                .flat_map(|(d, q)| [q[0] - d[0], q[1] - d[1], q[2] - d[2]])
                .collect(),
        )
    };
    let n = p.len();
    let mut numeric = DMatrix::zeros(3 * hkl.len(), n + 3);
    for k in 0..n + 3 {
        let step = 1e-5;
        let (pp, rp, pm, rm) = if k < n {
            let mut plus = p.to_vec();
            let mut minus = p.to_vec();
            plus[k] += step;
            minus[k] -= step;
            (plus, r, minus, r)
        } else {
            let mut e = [0.; 3];
            e[k - n] = step;
            let rp = rodrigues(e) * r;
            e[k - n] = -step;
            (p.to_vec(), rp, p.to_vec(), rodrigues(e) * r)
        };
        let (cp, cm) = (calc(&pp, &rp)?, calc(&pm, &rm)?);
        for i in 0..cp.len() {
            numeric[(i, k)] = (cp[i] - cm[i]) / (2. * step);
        }
    }
    Some((analytic, numeric))
}
#[cfg(test)]
pub(super) fn free_for_test(system: LatticeSystem, cell: [f64; 6]) -> Vec<f64> {
    free(system, cell)
}
