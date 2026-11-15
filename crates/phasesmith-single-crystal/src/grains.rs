//! Position-only multi-grain lattice discovery with frozen-pool validation.
//!
//! Unindexed reciprocal positions (inverse ångström, no 2π) in; competing
//! direct lattices and their exclusive ownership out. No cell, HKL, chemistry
//! or symmetry input. This is the native counterpart of
//! `phasesmith.single_crystal.grains.find_grains` (equations and boundaries:
//! `docs/experimental-grain-search.md`); the native single-crystal workflow
//! uses it with [`Options::default`].
//!
//! - [`discover`]: training positions only. Each round searches the positions
//!   no model owns or explains, refines the strongest Fourier triplets, keeps
//!   the best new lattice and refits all models with exclusive ownership.
//!   [`Method::Rows`] proposes one lattice per row-search group and
//!   consolidates them; [`Method::Both`] pools those with the FFT models.
//! - [`validate`]: the first use of withheld positions. The frozen pool is
//!   assigned once, and once per rotated control; acceptance is per model.
//! - [`find`]: both, for a caller-supplied training partition.
//!
//! Defaults reproduce the native workflow bit for bit. [`Options::round_controls`],
//! [`Options::min_tight_fraction`] and the row methods are opt-in.
// Moved with the workflow's coordinator code, which allows these casts and
// short names for its whole module.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]
use crate::spots::{
    SpotError,
    fft::{Engine, SupportBits},
};
use nalgebra::{DMatrix, Matrix3};

/// Three Cartesian components, inverse ångström or ångström by context.
pub type V3 = [f64; 3];
/// Row-major 3×3 matrix; direct bases have one lattice vector per row.
pub type M3 = [V3; 3];
type Result<T> = std::result::Result<T, SpotError>;

/// Bounded discovery and validation settings. Defaults are the native
/// workflow's values. Non-exhaustive so settings can be added: start from
/// [`Options::default`] and assign fields.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Options {
    /// Shortest direct basis vector, ångström.
    pub min_length: f64,
    /// Longest direct basis vector, ångström.
    pub max_length: f64,
    /// Even Fourier grid edge, 16 through 256.
    pub grid_size: usize,
    /// Fourier maxima refined per round.
    pub fft_peaks: usize,
    /// Distinct translation directions kept per round.
    pub basis_vectors: usize,
    /// Strongest support pairs screened for triplets.
    pub max_pairs: usize,
    /// Triplets refined per round.
    pub cell_refinements: usize,
    /// Discovery rounds, at most one new model each.
    pub max_grains: usize,
    /// Integer-phase window of translation support screening.
    pub fractional_tolerance: f64,
    /// Strict reciprocal ownership distance, inverse ångström.
    pub reciprocal_tolerance: f64,
    /// Fewest training positions a model must own.
    pub min_indexed: usize,
    /// Lower direct cell volume bound, cubic ångström.
    pub min_volume: f64,
    /// Upper direct cell volume bound, cubic ångström.
    pub max_volume: f64,
    /// Largest direct basis condition number.
    pub max_condition: f64,
    /// A position is unowned unless its best residual is below this share of
    /// the second best.
    pub ambiguity_ratio: f64,
    /// Private Rayon workers, 1 through 4.
    pub workers: usize,
    /// Round selection. 0 (default): largest support minus the uniform
    /// expectation `N V 4π tolerance³/3`, as in the workflow. `n > 0`: among
    /// the 16 best distinct proposals, the largest support beyond the same
    /// cell in the first `n` rotated controls of the round's positions, then
    /// a smaller cell on the same positions keeping 90 % of that. Structured
    /// clouds catch far more than the uniform expectation on large cells; a
    /// near-supercell can otherwise win a round, fail validation and hide the
    /// grain whose positions it took.
    pub round_controls: usize,
    /// Validation: share of withheld matches that must lie within half the
    /// reciprocal tolerance. 0 (default) disables the check. Matches of a
    /// real lattice cluster at its nodes; chance matches fill the sphere.
    pub min_tight_fraction: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            min_length: 1.5,
            max_length: 30.,
            grid_size: 256,
            fft_peaks: 192,
            basis_vectors: 64,
            max_pairs: 96,
            cell_refinements: 32,
            max_grains: 5,
            fractional_tolerance: 0.15,
            reciprocal_tolerance: 0.015,
            min_indexed: 20,
            min_volume: 15.,
            max_volume: 20000.,
            max_condition: 20.,
            ambiguity_ratio: 0.8,
            workers: 4,
            round_controls: 0,
            min_tight_fraction: 0.,
        }
    }
}
impl Options {
    /// Rejects out-of-range or non-finite settings.
    /// # Errors
    /// Names the first invalid setting.
    pub fn check(&self) -> Result<()> {
        let bad = |m: &str| Err(SpotError(m.into()));
        let finite = [
            self.min_length,
            self.max_length,
            self.fractional_tolerance,
            self.reciprocal_tolerance,
            self.min_volume,
            self.max_volume,
            self.max_condition,
            self.ambiguity_ratio,
            self.min_tight_fraction,
        ];
        if finite.iter().any(|x| !x.is_finite()) {
            return bad("grain options must be finite");
        }
        if !(0. < self.min_length && self.min_length < self.max_length && self.max_length <= 100.) {
            return bad("require 0 < min_length < max_length <= 100 angstrom");
        }
        if !(16..=256).contains(&self.grid_size) || !self.grid_size.is_multiple_of(2) {
            return bad("grid_size must be even in [16,256]");
        }
        if !(3..=400).contains(&self.fft_peaks)
            || !(3..=80).contains(&self.basis_vectors)
            || self.basis_vectors > self.fft_peaks
            || !(1..=256).contains(&self.max_pairs)
            || !(1..=80).contains(&self.cell_refinements)
            || !(1..=40).contains(&self.max_grains)
            || !(1..=4).contains(&self.workers)
            || !(6..=100_000).contains(&self.min_indexed)
            || self.round_controls > 128
        {
            return bad("grain search budget out of range");
        }
        let open = |x: f64, lo: f64, hi: f64| lo < x && x < hi;
        if !open(self.fractional_tolerance, 0., 0.5) || !open(self.reciprocal_tolerance, 0., 1.) {
            return bad("invalid positional tolerances");
        }
        if !open(self.min_volume, 0., self.max_volume)
            || self.max_volume > 1e6
            || self.max_condition <= 1.
            || self.max_condition > 100.
        {
            return bad("invalid volume or conditioning bounds");
        }
        if !open(self.ambiguity_ratio, 0., 1.) || !(0. ..=1.).contains(&self.min_tight_fraction) {
            return bad("invalid ambiguity ratio or tight fraction");
        }
        Ok(())
    }
}

/// How discovery proposes lattices. Non-exhaustive: the row variants exist
/// only with the `row-search` feature, and methods may be added.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Method {
    /// Fourier triplets on the whole remaining training cloud, per round.
    Fft,
    /// Row search (DAFi-based) groups, one inferred lattice per group, then
    /// consolidation over all training positions
    /// (`phasesmith.single_crystal.grains.find_grains(method="rows")`).
    #[cfg(feature = "row-search")]
    Rows(crate::row_search::Options),
    /// The FFT models and the row proposals in one consolidated pool.
    #[cfg(feature = "row-search")]
    Both(crate::row_search::Options),
}

/// Position-supported competing lattice, with independently withheld ownership.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Grain {
    /// Right-handed direct row basis in angstroms.
    pub direct: M3,
    /// Integer Miller indices in the selected supplied setting.
    pub hkl: Vec<[i64; 3]>,
    /// Boolean frozen-pool ownership for each supplied reciprocal point.
    pub indexed: Vec<bool>,
    /// Frozen independent training membership or retained training count.
    pub training: usize,
    /// Independent withheld position count.
    pub withheld: usize,
    /// Whether every frozen training, withheld and rotated-null gate passed.
    pub accepted: bool,
    /// Retained ownership counts for the 64 predeclared null rotations.
    pub null_counts: Vec<usize>,
}

fn matrix(a: M3) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| a[i][j])
}
fn array(a: Matrix3<f64>) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)]))
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn inv_t(a: M3) -> Result<M3> {
    matrix(a)
        .try_inverse()
        .map(|x| array(x.transpose()))
        .ok_or_else(|| SpotError("singular basis".into()))
}
fn det(a: M3) -> f64 {
    matrix(a).determinant()
}
fn mul(a: M3, b: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn row(v: V3, a: M3) -> V3 {
    std::array::from_fn(|j| (0..3).map(|k| v[k] * a[k][j]).sum())
}
fn least(h: &[V3], q: &[V3]) -> Result<M3> {
    let a = DMatrix::from_fn(h.len(), 3, |i, j| h[i][j]);
    let b = DMatrix::from_fn(q.len(), 3, |i, j| q[i][j]);
    let svd = a.svd(true, true);
    let max = svd.singular_values.max();
    if svd
        .singular_values
        .iter()
        .filter(|&&x| x > max * (h.len() as f64) * f64::EPSILON)
        .count()
        < 3
    {
        return Err(SpotError("rank-deficient indexed positions".into()));
    }
    let x = svd
        .solve(&b, max * (h.len() as f64) * f64::EPSILON)
        .map_err(|e| SpotError(e.into()))?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| x[(i, j)])))
}
fn condition(a: M3) -> f64 {
    let s = matrix(a).svd(false, false).singular_values;
    s.max() / s.min()
}

/// Pairwise length reduction, shortest first, right-handed. An integer basis
/// change, not a Niggli or conventional-cell guarantee.
#[must_use]
pub fn reduce(mut a: M3) -> M3 {
    for _ in 0..100 {
        a.sort_by(|x, y| dot(*x, *x).total_cmp(&dot(*y, *y)));
        let mut changed = false;
        for j in 1..3 {
            for i in (0..j).rev() {
                let k = (dot(a[i], a[j]) / dot(a[i], a[i])).round_ties_even();
                let t = std::array::from_fn(|n| a[j][n] - k * a[i][n]);
                if dot(t, t) < dot(a[j], a[j]) - 1e-10 {
                    a[j] = t;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if det(a) < 0. {
        a[2] = a[2].map(|x| -x);
    }
    a
}

/// Integer Hermite row basis of the lattice the indices span; None for
/// non-integer, oversized or rank-deficient input.
#[allow(clippy::float_cmp)] // Integer HKLs and exact zero/signed determinant sentinels.
pub(crate) fn integer_basis(indices: &[V3], stop_if_primitive: bool) -> Option<M3> {
    // Validate the entire input before the private index-only shortcut.
    if indices
        .iter()
        .flatten()
        .any(|x| !x.is_finite() || x.abs() > 1_000_000. || *x != x.round_ties_even())
    {
        return None;
    }
    let mut basis: [Option<[i128; 3]>; 3] = [None; 3];
    for h in indices {
        let mut row = h.map(|x| x as i128);
        for col in 0..3 {
            while row[col] != 0 {
                if let Some(pivot) = basis[col] {
                    let k = row[col].div_euclid(pivot[col]);
                    for j in 0..3 {
                        row[j] = row[j].checked_sub(k.checked_mul(pivot[j])?)?;
                    }
                    if row[col] != 0 {
                        if row[col] < 0 {
                            row = row.map(|x| -x);
                        }
                        basis[col] = Some(row);
                        row = pivot;
                    }
                } else {
                    basis[col] = Some(if row[col] > 0 { row } else { row.map(|x| -x) });
                    row = [0; 3];
                    break;
                }
            }
        }
        // Upper triangular positive unit pivots generate Z³. Later integer
        // rows cannot change its index; this consumer does not need its basis.
        if stop_if_primitive && (0..3).all(|i| basis[i].is_some_and(|r| r[i] == 1)) {
            return Some([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        }
    }
    if basis
        .iter()
        .flatten()
        .flatten()
        .any(|x| x.abs() > 9_007_199_254_740_992)
    {
        return None;
    }
    Some([
        basis[0]?.map(|x| x as f64),
        basis[1]?.map(|x| x as f64),
        basis[2]?.map(|x| x as f64),
    ])
}

pub(crate) struct RefinementFit {
    pub(crate) h: Vec<V3>,
    pub(crate) points: Vec<V3>,
    pub(crate) updated: M3,
}
pub(crate) struct ScoredBasis {
    pub(crate) direct: M3,
    pub(crate) scores: Vec<(V3, f64)>,
}
impl RefinementFit {
    pub(crate) fn matches(&self, h: &[V3], points: &[V3]) -> bool {
        fn same(a: &[V3], b: &[V3]) -> bool {
            a.len() == b.len()
                && a.iter()
                    .flatten()
                    .zip(b.iter().flatten())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }
        same(&self.h, h) && same(&self.points, points)
    }
}

/// One search over a fixed engine and options.
pub(crate) struct Search<'a> {
    pub(crate) e: Engine,
    pub(crate) o: &'a Options,
}
impl Search<'_> {
    fn valid(&self, a: M3) -> bool {
        let o = self.o;
        let l = a.map(norm);
        a.iter().flatten().all(|x| x.is_finite())
            && l.iter()
                .all(|&x| (o.min_length..=o.max_length).contains(&x))
            && (o.min_volume..=o.max_volume).contains(&det(a).abs())
            && condition(a) <= o.max_condition
    }
    pub(crate) fn score(&self, q: &[V3], a: M3) -> Result<Vec<(V3, f64)>> {
        Ok(self.e.score(q, a, inv_t(a)?))
    }
    #[allow(clippy::float_cmp)] // Integer HKLs and exact zero/signed determinant sentinels.
    fn mask(&self, s: &[(V3, f64)]) -> Vec<bool> {
        s.iter()
            .map(|(h, d)| *d < self.o.reciprocal_tolerance && *h != [0.; 3])
            .collect()
    }
    fn assign(&self, q: &[V3], a: &[M3]) -> Result<Vec<i32>> {
        let models = assignment_models(a)?;
        Ok(self
            .e
            .assign(
                q,
                &models,
                self.o.reciprocal_tolerance,
                self.o.ambiguity_ratio,
            )
            .0)
    }
    #[allow(clippy::float_cmp)] // Inferred HKLs are exact rounded integers.
    pub(crate) fn refine(&self, q: &[V3], mut a: M3) -> Result<Option<ScoredBasis>> {
        let tolerance = self.o.reciprocal_tolerance;
        let mut previous: Option<RefinementFit> = None;
        for cycle in 0..8 {
            let s = self.score(q, a)?;
            let mut h = Vec::new();
            let mut points = Vec::new();
            for (&q, (indices, d)) in q.iter().zip(s) {
                if d < if cycle < 2 { 2. * tolerance } else { tolerance } && indices != [0.; 3] {
                    h.push(indices);
                    points.push(q);
                }
            }
            if h.len() < 6 {
                return Ok(None);
            }
            let updated =
                if let Some(fit) = previous.as_ref().filter(|fit| fit.matches(&h, &points)) {
                    fit.updated
                } else {
                    let Ok(b) = least(&h, &points) else {
                        return Ok(None);
                    };
                    if det(b).abs() < 1e-12 {
                        return Ok(None);
                    }
                    let updated = inv_t(b)?;
                    previous = Some(RefinementFit { h, points, updated });
                    updated
                };
            let delta = a
                .iter()
                .flatten()
                .zip(updated.iter().flatten())
                .map(|(x, y)| (x - y).abs())
                .fold(0., f64::max);
            a = updated;
            if cycle >= 2 && delta < 1e-8 {
                break;
            }
        }
        a = reduce(a);
        if !self.valid(a) {
            return Ok(None);
        }
        let s = self.score(q, a)?;
        let selected = self.mask(&s);
        let mut ids: Vec<_> = (0..q.len()).filter(|&i| selected[i]).collect();
        ids.sort_by(|&i, &j| s[i].1.total_cmp(&s[j].1));
        let mut best = a;
        let mut best_score = None;
        for size in [24, 48, 96] {
            if ids.len() < size {
                continue;
            }
            let hs: Vec<_> = ids[..size].iter().map(|&i| s[i].0).collect();
            if let Some(h) = integer_basis(&hs, true) {
                if det(h).round().abs() <= 1. {
                    continue;
                }
                let t = reduce(inv_t(mul(h, inv_t(a)?))?);
                if !self.valid(t) {
                    continue;
                }
                let scored = self.score(q, t)?;
                let m = self.mask(&scored);
                if (0..q.len()).filter(|&i| m[i] && selected[i]).count() as f64
                    >= 0.9 * selected.iter().filter(|&&v| v).count() as f64
                    && det(t).abs() < det(best).abs()
                {
                    best = t;
                    best_score = Some(scored);
                }
            }
        }
        Ok(Some(ScoredBasis {
            direct: best,
            scores: best_score.unwrap_or(s),
        }))
    }
    #[allow(clippy::float_cmp)] // Integer HKLs and exact zero/signed determinant sentinels.
    fn joint(&self, q: &[V3], mut models: Vec<M3>) -> Result<(Vec<M3>, Vec<i32>)> {
        let mut labels = self.assign(q, &models)?;
        for _ in 0..8 {
            let mut updated = Vec::new();
            let mut stable = true;
            for (i, &a) in models.iter().enumerate() {
                let block: Vec<_> = q
                    .iter()
                    .zip(&labels)
                    .filter_map(|(&q, &l)| (l == i as i32 + 1).then_some(q))
                    .collect();
                let h: Vec<_> = self.score(&block, a)?.into_iter().map(|x| x.0).collect();
                let mut b = a;
                if block.len() >= 6
                    && let Ok(fit) = least(&h, &block)
                    && det(fit).abs() > 1e-12
                {
                    let t = inv_t(fit)?;
                    if self.valid(t) {
                        b = t;
                    }
                }
                stable &= block.iter().zip(h).all(|(&q, h)| {
                    std::array::from_fn::<_, 3, _>(|j| dot(q, b[j]).round_ties_even()) == h
                });
                updated.push(b);
            }
            let next = self.assign(q, &updated)?;
            let changed = labels != next;
            models = updated;
            labels = next;
            if !changed && stable {
                break;
            }
        }
        Ok((models, labels))
    }
    fn vectors(&self, q: &[V3]) -> Result<Vec<V3>> {
        let o = self.o;
        let n = o.grid_size;
        let max = q.iter().flatten().map(|x| x.abs()).fold(0., f64::max);
        let step = (2. * max / (n - 4) as f64).min(1. / (2.1 * o.max_length));
        let (initial, _, _) = self
            .e
            .windowed_maxima(q, n, step, o.min_length, o.max_length, o.fft_peaks)
            .map_err(SpotError)?;
        let radius = 1.5 / (n as f64 * step);
        let refined = self.e.refine_fourier(q, &initial, radius)?;
        let v: Vec<_> = refined.iter().map(|r| r.0).collect();
        let m: Vec<_> = refined.iter().map(|r| r.1).collect();
        let mut order: Vec<_> = (0..v.len()).collect();
        order.sort_by(|&i, &j| m[j][0].total_cmp(&m[i][0]));
        let mut chosen: Vec<V3> = Vec::new();
        for i in order {
            let n = norm(v[i]);
            if !(o.min_length..=o.max_length).contains(&n)
                || chosen
                    .iter()
                    .any(|&x| dot(v[i], x).abs() / (n * norm(x)) > 0.9995)
            {
                continue;
            }
            chosen.push(v[i]);
            if chosen.len() == o.basis_vectors {
                break;
            }
        }
        Ok(chosen)
    }
    /// Refined, non-duplicate proposals of one round on `block`, each as
    /// (-excess, rms, volume, order, basis, support count).
    #[allow(clippy::type_complexity)]
    fn proposals(
        &self,
        block: &[V3],
        models: &[M3],
    ) -> Result<Vec<(f64, f64, f64, usize, M3, usize)>> {
        let o = self.o;
        let v = self.vectors(block)?;
        let rows = v
            .iter()
            .map(|&a| {
                let mut bits = vec![0u64; block.len().div_ceil(64)];
                for (i, &q) in block.iter().enumerate() {
                    let p = dot(q, a);
                    if (p - p.round_ties_even()).abs() < o.fractional_tolerance {
                        bits[i / 64] |= 1 << (i % 64);
                    }
                }
                bits
            })
            .collect();
        let bits = SupportBits {
            rows,
            count: block.len(),
            pool: self.e.pool.clone(),
        };
        let keys: Vec<_> = bits
            .candidate_keys(o.max_pairs)
            .into_iter()
            .filter(|k| {
                let a = k.map(|i| v[i]);
                (o.min_volume..=o.max_volume).contains(&det(a).abs())
            })
            .collect();
        let volumes: Vec<_> = keys.iter().map(|k| det(k.map(|i| v[i])).abs()).collect();
        let (order, _) = bits.rank_triplets(&keys, &volumes);
        let mut candidates = Vec::new();
        let mut tried = 0;
        for i in order {
            let a = keys[i].map(|i| v[i]);
            if condition(a) > o.max_condition {
                continue;
            }
            tried += 1;
            if tried > o.cell_refinements {
                break;
            }
            let Some(fit) = self.refine(block, a)? else {
                continue;
            };
            let a = fit.direct;
            let s = fit.scores;
            let selected = self.mask(&s);
            let count = selected.iter().filter(|&&v| v).count();
            let own: Vec<_> = block
                .iter()
                .zip(&selected)
                .filter_map(|(&q, &ok)| ok.then_some(q))
                .collect();
            let mut duplicate = own.is_empty();
            for &old in models {
                let n = self
                    .mask(&self.score(&own, old)?)
                    .iter()
                    .filter(|&&x| x)
                    .count();
                duplicate |= n as f64 >= 0.8 * own.len() as f64;
            }
            if count < o.min_indexed || duplicate {
                continue;
            }
            let volume = det(a).abs();
            let excess = count as f64
                - block.len() as f64 * volume * 4. * std::f64::consts::PI / 3.
                    * o.reciprocal_tolerance.powi(3);
            let rms = (s
                .iter()
                .zip(&selected)
                .filter_map(|(s, &ok)| ok.then_some(s.1 * s.1))
                .sum::<f64>()
                / count as f64)
                .sqrt();
            candidates.push((-excess, rms, volume, tried, a, count));
        }
        candidates.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.total_cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        Ok(candidates)
    }
    /// `Options::round_controls`: the proposal indexing the most positions
    /// beyond its rotated controls, preferring a smaller cell on the same
    /// positions that keeps 90 % of that.
    fn controlled(
        &self,
        block: &[V3],
        ranked: &[(f64, f64, f64, usize, M3, usize)],
        controls: &[M3],
    ) -> Result<Option<M3>> {
        let mut distinct: Vec<&(f64, f64, f64, usize, M3, usize)> = Vec::new();
        for c in ranked {
            if !distinct.iter().any(|d| related(d.4, c.4)) {
                distinct.push(c);
                if distinct.len() == 16 {
                    break;
                }
            }
        }
        let rotated: Vec<Vec<V3>> = controls
            .iter()
            .take(self.o.round_controls)
            .map(|&r| block.iter().map(|&q| row(q, r)).collect())
            .collect();
        let mut scored = Vec::new();
        for c in &distinct {
            let mut null = 0;
            for q in &rotated {
                null += self
                    .mask(&self.score(q, c.4)?)
                    .iter()
                    .filter(|&&x| x)
                    .count();
            }
            let beyond = c.5 as f64 - null as f64 / rotated.len().max(1) as f64;
            scored.push((beyond, self.mask(&self.score(block, c.4)?)));
        }
        let Some(mut chosen) = (0..distinct.len())
            .max_by(|&i, &j| scored[i].0.total_cmp(&scored[j].0).then(j.cmp(&i)))
        else {
            return Ok(None);
        };
        loop {
            let (beyond, mask) = &scored[chosen];
            let smaller = (0..distinct.len())
                .filter(|&i| det(distinct[i].4).abs() < 0.67 * det(distinct[chosen].4).abs())
                .filter(|&i| {
                    let shared = mask
                        .iter()
                        .zip(&scored[i].1)
                        .filter(|(a, b)| **a && **b)
                        .count();
                    scored[i].0 >= 0.9 * beyond && shared as f64 >= 0.8 * distinct[i].5 as f64
                })
                .max_by(|&i, &j| scored[i].0.total_cmp(&scored[j].0).then(j.cmp(&i)));
            match smaller {
                Some(i) => chosen = i,
                None => break,
            }
        }
        Ok(Some(distinct[chosen].4))
    }
    fn fft(
        &self,
        q: &[V3],
        controls: &[M3],
        progress: &mut dyn FnMut(usize, usize) -> Result<()>,
    ) -> Result<Vec<M3>> {
        let o = self.o;
        let mut models = Vec::new();
        let mut labels = vec![0; q.len()];
        for iteration in 0..o.max_grains {
            progress(iteration, o.max_grains)?;
            let mut explained = vec![false; q.len()];
            for &a in &models {
                for (x, m) in explained.iter_mut().zip(self.mask(&self.score(q, a)?)) {
                    *x |= m;
                }
            }
            let block: Vec<_> = q
                .iter()
                .enumerate()
                .filter_map(|(i, &q)| (labels[i] == 0 && !explained[i]).then_some(q))
                .collect();
            if block.len() < o.min_indexed {
                break;
            }
            let candidates = self.proposals(&block, &models)?;
            let best = if o.round_controls > 0 {
                self.controlled(&block, &candidates, controls)?
            } else {
                candidates.first().map(|c| c.4)
            };
            let Some(best) = best else {
                break;
            };
            models.push(best);
            (models, labels) = self.joint(q, models)?;
        }
        Ok(models)
    }
    /// Training-only consolidation of a proposal pool
    /// (`grains._consolidate`): volume-normalised local evidence, symmetric
    /// overlap and novelty, then joint ownership.
    #[cfg(feature = "row-search")]
    #[allow(clippy::float_cmp, clippy::too_many_lines)] // Exact zero-HKL sentinel; one ordered greedy pass as in `_consolidate`.
    fn consolidate(&self, fit: &[V3], models: &[M3]) -> Result<Vec<M3>> {
        let o = self.o;
        if models.is_empty() {
            return Ok(vec![]);
        }
        if fit.len() * models.len() > 8_000_000 {
            return Err(SpotError(
                "consolidation exceeds 8 million position/model pairs".into(),
            ));
        }
        let sigma = o.reciprocal_tolerance / 4.;
        let mut errors = Vec::with_capacity(models.len());
        for &a in models {
            errors.push(
                self.score(fit, a)?
                    .into_iter()
                    .map(|(h, d)| if h == [0.; 3] { f64::INFINITY } else { d })
                    .collect::<Vec<_>>(),
            );
        }
        let volumes: Vec<f64> = models.iter().map(|&a| det(a).abs()).collect();
        let masks: Vec<Vec<bool>> = errors
            .iter()
            .map(|e| e.iter().map(|&d| d < sigma).collect())
            .collect();
        let counts: Vec<usize> = masks
            .iter()
            .map(|m| m.iter().filter(|&&x| x).count())
            .collect();
        let tau = 2. * std::f64::consts::PI;
        let evidence: Vec<Vec<f64>> = errors
            .iter()
            .zip(&volumes)
            .map(|(e, &v)| {
                e.iter()
                    .map(|&d| {
                        let x =
                            -0.5 * (d / sigma).powi(2) - (v * tau.powf(1.5) * sigma.powi(3)).ln();
                        // logaddexp(0, x)
                        if x > 0. {
                            x + (-x).exp().ln_1p()
                        } else {
                            x.exp().ln_1p()
                        }
                    })
                    .collect()
            })
            .collect();
        let mut previous = vec![0.; fit.len()];
        let mut covered = vec![false; fit.len()];
        let mut selected: Vec<usize> = Vec::new();
        for _ in 0..o.max_grains {
            let mut best: Option<(f64, f64, usize)> = None;
            for i in 0..models.len() {
                if selected.contains(&i) || counts[i] < o.min_indexed {
                    continue;
                }
                let overlaps = selected.iter().any(|&j| {
                    let both = masks[i]
                        .iter()
                        .zip(&masks[j])
                        .filter(|(a, b)| **a && **b)
                        .count();
                    both as f64 >= 0.8 * counts[i].max(counts[j]) as f64
                });
                if overlaps {
                    continue;
                }
                let novel = masks[i]
                    .iter()
                    .zip(&covered)
                    .filter(|(m, c)| **m && !**c)
                    .count();
                if novel < o.min_indexed {
                    continue;
                }
                let gain: f64 = evidence[i]
                    .iter()
                    .zip(&previous)
                    .map(|(e, p)| (e - p).max(0.))
                    .sum();
                if gain > 0. {
                    let key = (-gain, volumes[i], i);
                    if best.is_none_or(|b| {
                        key.0
                            .total_cmp(&b.0)
                            .then(key.1.total_cmp(&b.1))
                            .then(key.2.cmp(&b.2))
                            .is_lt()
                    }) {
                        best = Some(key);
                    }
                }
            }
            let Some((_, _, i)) = best else {
                break;
            };
            selected.push(i);
            for (p, e) in previous.iter_mut().zip(&evidence[i]) {
                *p = p.max(*e);
            }
            for (c, m) in covered.iter_mut().zip(&masks[i]) {
                *c |= m;
            }
        }
        let chosen: Vec<M3> = selected.iter().map(|&i| models[i]).collect();
        if chosen.is_empty() {
            return Ok(chosen);
        }
        Ok(self.joint(fit, chosen)?.0)
    }
    /// One lattice per row-search group (`row_grains.discover_rows`). The
    /// grouping itself runs to completion or its own deadline
    /// (`row_search::Options::seconds`); `progress` can cancel between groups.
    #[cfg(feature = "row-search")]
    fn rows(
        &self,
        q: &[V3],
        options: &crate::row_search::Options,
        progress: &mut dyn FnMut(usize, usize) -> Result<()>,
    ) -> Result<Vec<M3>> {
        use std::sync::atomic::AtomicBool;
        progress(0, self.o.max_grains)?;
        let cancel = AtomicBool::new(false);
        let grouping = crate::row_search::run(q, options, &cancel).map_err(SpotError)?;
        if grouping.budget_exhausted || grouping.cancelled {
            return Err(SpotError(
                "Row search exhausted its work/deadline budget".into(),
            ));
        }
        // One lattice per group with the plain round selection, as
        // `discover_rows` does; consolidation decides among the groups.
        let per_group = Options {
            max_grains: 1,
            round_controls: 0,
            ..self.o.clone()
        };
        let search = Search {
            e: Engine::new(self.o.workers).map_err(SpotError)?,
            o: &per_group,
        };
        let mut proposals = Vec::new();
        for (number, members) in grouping.groups.iter().enumerate() {
            progress(number.min(self.o.max_grains - 1), self.o.max_grains)?;
            let block: Vec<V3> = members.iter().map(|&i| q[i]).collect();
            if block.len() < self.o.min_indexed || least(&block, &block).is_err() {
                continue;
            }
            // A failed inference leaves the group without a proposal.
            if let Ok(models) = search.fft(&block, &[], &mut |_, _| Ok(())) {
                proposals.extend(models);
            }
        }
        Ok(proposals)
    }
}

fn assignment_models(a: &[M3]) -> Result<Vec<(M3, M3)>> {
    a.iter().map(|&a| Ok((a, inv_t(a)?))).collect()
}

/// Same lattice within refinement scatter: an integer, unimodular change of
/// basis within 0.05.
#[must_use]
pub fn related(a: M3, b: M3) -> bool {
    let Ok(inverse) = inv_t(b) else {
        return false;
    };
    let t = a.map(|r| inverse.map(|dual| dot(r, dual)));
    t.iter().flatten().all(|v| (v - v.round()).abs() < 0.05) && (det(t).abs() - 1.).abs() < 0.05
}

/// Training-only discovery. `controls` are rotations (row-vector convention,
/// `q·R`) used by [`Options::round_controls`]; `progress(done, total)` runs
/// before each round and cancels with its error.
/// # Errors
/// Invalid options or positions, kernel failures, an incomplete row search,
/// or the error `progress` returns.
pub fn discover(
    train: &[V3],
    options: &Options,
    controls: &[M3],
    method: &Method,
    progress: &mut dyn FnMut(usize, usize) -> Result<()>,
) -> Result<Vec<M3>> {
    options.check()?;
    if options.round_controls > controls.len() {
        return Err(SpotError(
            "round_controls exceeds the supplied rotations".into(),
        ));
    }
    let search = Search {
        e: Engine::new(options.workers).map_err(SpotError)?,
        o: options,
    };
    match method {
        Method::Fft => search.fft(train, controls, progress),
        #[cfg(feature = "row-search")]
        Method::Rows(rows) => {
            let proposals = search.rows(train, rows, progress)?;
            search.consolidate(train, &proposals)
        }
        #[cfg(feature = "row-search")]
        Method::Both(rows) => {
            // One progress range: FFT rounds, then row groups.
            let mut proposals = search.fft(train, controls, &mut |done, total| {
                progress(done, 2 * total)
            })?;
            proposals.extend(search.rows(train, rows, &mut |done, total| {
                progress(total + done, 2 * total)
            })?);
            search.consolidate(train, &proposals)
        }
    }
}

/// Frozen-pool validation: the first use of withheld positions.
///
/// Acceptance per model: at least `min_indexed` training and six withheld
/// positions owned; withheld count above every rotated control, at least
/// twice `max(mean,1)` and `mean + 4 sqrt(max(variance, mean, 1))`; at least
/// half the training share extrapolated to the withheld population; rank-three
/// training HKLs; and, when set, [`Options::min_tight_fraction`].
/// # Errors
/// Mismatched lengths, invalid options or singular models.
#[allow(clippy::float_cmp)]
pub fn validate(
    q: &[V3],
    training: &[bool],
    models: &[M3],
    controls: &[M3],
    options: &Options,
) -> Result<Vec<Grain>> {
    options.check()?;
    if training.len() != q.len() || controls.len() < 2 {
        return Err(SpotError(
            "validation needs one training flag per position and two or more controls".into(),
        ));
    }
    let search = Search {
        e: Engine::new(options.workers).map_err(SpotError)?,
        o: options,
    };
    let assignments = assignment_models(models)?;
    let (tol, ratio) = (options.reciprocal_tolerance, options.ambiguity_ratio);
    let labels = search.e.assign(q, &assignments, tol, ratio).0;
    let held: Vec<_> = q
        .iter()
        .zip(training.iter())
        .filter_map(|(&q, &t)| (!t).then_some(q))
        .collect();
    let train_count = training.iter().filter(|&&t| t).count();
    let mut null = vec![vec![]; models.len()];
    for &rotation in controls {
        let rotated: Vec<_> = held.iter().map(|&q| row(q, rotation)).collect();
        let l = search.e.assign(&rotated, &assignments, tol, ratio).0;
        for (i, n) in null.iter_mut().enumerate() {
            n.push(l.iter().filter(|&&x| x == i as i32 + 1).count());
        }
    }
    let count = controls.len() as f64;
    let mut grains = Vec::new();
    for (i, &direct) in models.iter().enumerate() {
        let s = search.score(q, direct)?;
        let indexed: Vec<_> = labels.iter().map(|&l| l == i as i32 + 1).collect();
        let ntrain = indexed
            .iter()
            .zip(training.iter())
            .filter(|(a, b)| **a && **b)
            .count();
        let ntest = indexed
            .iter()
            .zip(training.iter())
            .filter(|(a, b)| **a && !**b)
            .count();
        let mean = null[i].iter().sum::<usize>() as f64 / count;
        let var = null[i]
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / (count - 1.);
        let trained: Vec<_> = s
            .iter()
            .zip(&indexed)
            .zip(training.iter())
            .filter_map(|((s, &own), &train)| (own && train).then_some(s.0))
            .collect();
        let unique = trained
            .iter()
            .map(|h| h.map(|x| x as i64))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let rank = unique >= options.min_indexed && least(&trained, &trained).is_ok();
        let tight = if options.min_tight_fraction > 0. {
            let inner = s
                .iter()
                .zip(&indexed)
                .zip(training.iter())
                .filter(|((s, own), train)| **own && !**train && s.1 < tol / 2.)
                .count();
            inner as f64 >= options.min_tight_fraction * ntest as f64
        } else {
            true
        };
        let accepted = ntrain >= options.min_indexed
            && ntest >= 6
            && ntest > *null[i].iter().max().unwrap_or(&0)
            && ntest as f64 >= 2. * mean.max(1.)
            && ntest as f64 >= mean + 4. * var.max(mean).max(1.).sqrt()
            && ntest as f64 >= 0.5 * ntrain as f64 * held.len() as f64 / train_count as f64
            && rank
            && tight;
        grains.push(Grain {
            direct,
            hkl: s.iter().map(|s| s.0.map(|x| x as i64)).collect(),
            indexed,
            training: ntrain,
            withheld: ntest,
            accepted,
            null_counts: null[i].clone(),
        });
    }
    Ok(grains)
}

/// [`discover`] on the training positions, then [`validate`] on all.
/// # Errors
/// As [`discover`] and [`validate`]; fewer than `min_indexed` training or six
/// withheld positions.
pub fn find(
    q: &[V3],
    training: &[bool],
    controls: &[M3],
    options: &Options,
    method: &Method,
    progress: &mut dyn FnMut(usize, usize) -> Result<()>,
) -> Result<Vec<Grain>> {
    if training.len() != q.len() {
        return Err(SpotError("one training flag per position".into()));
    }
    let train: Vec<V3> = q
        .iter()
        .zip(training)
        .filter_map(|(&p, &t)| t.then_some(p))
        .collect();
    if train.len() < options.min_indexed || q.len() - train.len() < 6 {
        return Err(SpotError(
            "insufficient training or held-out positions".into(),
        ));
    }
    let models = discover(&train, options, controls, method, progress)?;
    validate(q, training, &models, controls, options)
}

#[cfg(test)]
#[path = "grains_tests.rs"]
mod tests;
