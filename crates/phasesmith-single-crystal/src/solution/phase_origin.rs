//! Measured-phase origin alignment, independently derived from phase equivalence.
//! Conventions, bounds and equations: docs/solution-phase-origin.md.
use crate::solution::fourier::{Transform, grid_index, grid_len};
use crate::solution::math3::{dot, pseudo_solve};
use crate::solution::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{Matrix3, SpaceGroup};
use rustfft::num_complex::Complex;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Pair {
    a: usize,
    b: usize,
    q: [i32; 3],
    translation: f64,
    weight: f64,
}
/// One count-free origin/hand hypothesis in the supplied space-group setting.
#[derive(Clone, Debug)]
pub struct PhaseOriginCandidate {
    /// Aligned coordinate = hand * source coordinate + shift.
    pub hand: i32,
    /// Fractional origin shift, reduced modulo one.
    pub shift: [f64; 3],
    /// F-squared-weighted wrapped phase error, random expectation one.
    pub alpha: f64,
    /// Smooth circular loss, random expectation one.
    pub chord: f64,
}
/// Value and analytic fractional-origin derivatives, evaluated together.
#[derive(Clone, Debug)]
pub struct PhaseOriginEvaluation {
    /// Weighted 1-cos(eta).
    pub chord: f64,
    /// Gradient of chord loss.
    pub gradient: [f64; 3],
    /// Hessian of chord loss.
    pub hessian: Matrix3,
    /// Weighted wrapped eta-squared divided by pi-squared/3.
    pub alpha: f64,
    /// Alpha gradient away from phase-wrap boundaries.
    pub alpha_gradient: [f64; 3],
}
/// Immutable observed-family phase relations; contains no atom information.
#[derive(Clone, Debug)]
pub struct PreparedPhaseOrigin {
    hkl: Vec<[i32; 3]>,
    pairs: Vec<Pair>,
    group: SpaceGroup,
    /// Independent family representatives in canonical HKL order.
    pub representatives: Vec<usize>,
    /// Families retained for phase alignment, capped deterministically at 2048.
    pub selected_families: usize,
}
/// Phases extracted once from an arbitrary-origin density.
#[derive(Clone, Debug)]
pub struct PhaseOriginMap {
    terms: Vec<([i32; 3], Complex<f64>)>,
    weight: f64,
    /// Number of nonzero-amplitude measured phase pairs.
    pub pairs: usize,
}

/// Immutable Fourier coefficients of one real density, reusable across settings/hands.
/// The transform and coefficient normalization exactly match `PreparedPhaseOrigin::map`.
#[derive(Clone, Debug)]
pub struct PhaseOriginDensity {
    values: Vec<Complex<f64>>,
    shape: [usize; 3],
}
impl PhaseOriginDensity {
    /// Transform a finite density once. No observations or symmetry are imposed.
    /// # Errors
    /// Invalid grid, nonfinite or excessive density, or mismatched array length.
    pub fn new(density: &[f64], shape: [usize; 3]) -> Result<Self, SolutionError> {
        if density.len() != grid_len(shape)?
            || density.iter().any(|v| !v.is_finite() || v.abs() > 1e100)
        {
            return Err(error("invalid cached phase-origin density/grid"));
        }
        let mut values: Vec<_> = density.iter().map(|&v| Complex::new(v, 0.)).collect();
        Transform::new(shape).apply(&mut values, true);
        Ok(Self { values, shape })
    }
}

impl PreparedPhaseOrigin {
    /// Prepare unique-family F-squared weights and equivalent-reflection pairs.
    /// # Errors
    /// Rejects unmerged/unclosed, invalid or excessive reciprocal arrays.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    pub fn new(
        group: &SpaceGroup,
        hkl: &[[i32; 3]],
        observed: &[f64],
    ) -> Result<Self, SolutionError> {
        if hkl.is_empty()
            || hkl.len() > 100_000
            || observed.len() != hkl.len()
            || hkl.windows(2).any(|p| p[0] >= p[1])
            || hkl
                .iter()
                .any(|h| *h == [0; 3] || h.iter().any(|v| v.unsigned_abs() > 1024))
            || observed
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 1e100)
            || group.operations().len() > 192
        {
            return Err(error("invalid phase-origin arrays/symmetry"));
        }
        let maximum = observed.iter().copied().fold(0., f64::max);
        if maximum <= 0. {
            return Err(error("phase-origin alignment needs positive amplitudes"));
        }
        let locations: BTreeMap<_, _> = hkl.iter().enumerate().map(|(i, &h)| (h, i)).collect();
        for &h in hkl {
            if group
                .is_systematically_absent(h)
                .map_err(|e| error(e.to_string()))?
            {
                return Err(error("phase-origin input contains a systematic absence"));
            }
        }
        let mut remaining: BTreeSet<_> = hkl.iter().copied().collect();
        let mut representatives = Vec::new();
        while let Some(&h) = remaining.first() {
            representatives.push(locations[&h]);
            for mate in group
                .reflection_orbit(h, true)
                .map_err(|e| error(e.to_string()))?
            {
                let j = *locations
                    .get(&mate)
                    .ok_or_else(|| error("phase-origin families are not closed"))?;
                if (observed[j] / maximum - observed[locations[&h]] / maximum).abs() > 1e-10 {
                    return Err(error("phase-origin amplitudes are not symmetry merged"));
                }
                remaining.remove(&mate);
            }
        }
        let mut chosen = representatives.clone();
        chosen.sort_by(|&i, &j| {
            observed[j]
                .total_cmp(&observed[i])
                .then(hkl[i].cmp(&hkl[j]))
        });
        chosen.truncate(2048);
        let mut rotations = BTreeSet::new();
        let ops: Vec<_> = group
            .operations()
            .iter()
            .filter(|op| rotations.insert(op.rotation()))
            .collect();
        let mut pairs = Vec::new();
        for i in &chosen {
            let h = hkl[*i];
            let mut family = Vec::new();
            for &op in &ops {
                let mapped = op.reciprocal_index(h).map_err(|e| error(e.to_string()))?;
                if mapped == h {
                    continue;
                }
                let b = *locations
                    .get(&mapped)
                    .ok_or_else(|| error("missing phase-origin mate"))?;
                let translation = op
                    .translation()
                    .map(phasesmith_crystallography::Rational::as_f64);
                family.push(Pair {
                    a: *i,
                    b,
                    q: std::array::from_fn(|a| mapped[a] - h[a]),
                    translation: std::f64::consts::TAU * dot(h.map(f64::from), translation),
                    weight: (observed[*i] / maximum).powi(2),
                });
            }
            let count = family.len().max(1) as f64;
            for pair in &mut family {
                pair.weight /= count;
            }
            pairs.extend(family);
        }
        Ok(Self {
            hkl: hkl.to_vec(),
            pairs,
            group: group.clone(),
            representatives,
            selected_families: chosen.len(),
        })
    }
    /// Extract measured phases from a real density, once per hand.
    /// # Errors
    /// Rejects nonfinite density, invalid hand/grid or vanishing phase information.
    pub fn map(
        &self,
        density: &[f64],
        shape: [usize; 3],
        hand: i32,
    ) -> Result<PhaseOriginMap, SolutionError> {
        self.map_cached(&PhaseOriginDensity::new(density, shape)?, hand)
    }
    /// Extract phase relations from one checked immutable Fourier density.
    /// # Errors
    /// Invalid hand or measured HKLs outside strict Nyquist support.
    pub fn map_cached(
        &self,
        density: &PhaseOriginDensity,
        hand: i32,
    ) -> Result<PhaseOriginMap, SolutionError> {
        let shape = density.shape;
        if !matches!(hand, -1 | 1)
            || self.hkl.iter().any(|h| {
                (0..3).any(|a| 2 * i64::from(h[a]).abs() >= i64::try_from(shape[a]).unwrap_or(0))
            })
        {
            return Err(error("invalid phase-origin hand/support"));
        }
        let phases: Vec<_> = self
            .hkl
            .iter()
            .map(|h| density.values[grid_index(h.map(|v| v * hand), shape)])
            .collect();
        let threshold = phases.iter().map(|v| v.norm()).fold(0., f64::max) * 1e-12;
        let mut terms = Vec::new();
        for pair in &self.pairs {
            let (a, b) = (phases[pair.a], phases[pair.b]);
            if a.norm() <= threshold || b.norm() <= threshold || pair.weight == 0. {
                continue;
            }
            let z = b / b.norm()
                * (a / a.norm()).conj()
                * Complex::from_polar(pair.weight, pair.translation);
            terms.push((pair.q, z));
        }
        if !self.pairs.is_empty() && terms.is_empty() {
            return Err(error("no usable measured phase pairs"));
        }
        let weight = terms.iter().map(|(_, z)| z.norm()).sum();
        Ok(PhaseOriginMap {
            pairs: terms.len(),
            terms,
            weight,
        })
    }
    /// Nonzero-weight phase-equivalence frequency extent on each axis.
    #[must_use]
    pub fn sampling_extent(&self) -> [usize; 3] {
        std::array::from_fn(|a| {
            self.pairs
                .iter()
                .filter(|p| p.weight > 0.)
                .map(|p| p.q[a].unsigned_abs() as usize)
                .max()
                .unwrap_or(0)
        })
    }
    /// Frequency-derived origin sampling, independent of the density grid.
    /// At least 16 samples per axis; powers of two through 128. Larger frequencies
    /// are explicitly bounded (seed aliasing is possible), while polishing remains exact.
    #[must_use]
    pub fn sampling_shape(&self) -> [usize; 3] {
        self.sampling_extent()
            .map(|q| (2 * q + 1).next_power_of_two().clamp(16, 128))
    }
    /// Legacy density-grid-dependent origin search.
    /// # Errors
    /// Propagates phase extraction and invalid budget errors.
    pub fn search(
        &self,
        density: &[f64],
        shape: [usize; 3],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<Vec<PhaseOriginCandidate>, SolutionError> {
        self.search_sampled(density, shape, maximum, deadline, cancellation, false)
    }
    /// FFT-seeded smooth-origin search, retaining non-equivalent low-alpha settings.
    /// # Errors
    /// Propagates phase extraction errors; malformed budgets fail explicitly.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    pub fn search_sampled(
        &self,
        density: &[f64],
        shape: [usize; 3],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        independent: bool,
    ) -> Result<Vec<PhaseOriginCandidate>, SolutionError> {
        self.search_impl(
            density,
            shape,
            maximum,
            deadline,
            cancellation,
            independent,
            None,
        )
    }
    /// Same origin search using a Fourier density cached across settings and hands.
    /// No phase relation, seed, derivative or hypothesis ordering changes.
    /// # Errors
    /// Invalid search budget, coefficient support or phase relations.
    #[allow(clippy::too_many_arguments)]
    pub fn search_cached(
        &self,
        density: &PhaseOriginDensity,
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        independent: bool,
    ) -> Result<Vec<PhaseOriginCandidate>, SolutionError> {
        self.search_impl(
            &[],
            density.shape,
            maximum,
            deadline,
            cancellation,
            independent,
            Some(density),
        )
    }
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    fn search_impl(
        &self,
        density: &[f64],
        shape: [usize; 3],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        independent: bool,
        cached: Option<&PhaseOriginDensity>,
    ) -> Result<Vec<PhaseOriginCandidate>, SolutionError> {
        if !(1..=8).contains(&maximum) {
            return Err(error("phase-origin hypothesis bound must be 1..8"));
        }
        if self.pairs.is_empty() {
            return Ok(vec![PhaseOriginCandidate {
                hand: 1,
                shift: [0.; 3],
                alpha: 0.,
                chord: 0.,
            }]);
        }
        let centric = self
            .group
            .operations()
            .iter()
            .any(|op| op.rotation() == [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]);
        let mut candidates: Vec<PhaseOriginCandidate> = Vec::new();
        let coarse = if independent {
            self.sampling_shape()
        } else {
            shape.map(|n| n.clamp(16, 64))
        };
        for hand in [1, -1] {
            if hand == -1 && centric {
                break;
            }
            if crate::solution::stopped(deadline, cancellation) {
                break;
            }
            let map = if let Some(cached) = cached {
                self.map_cached(cached, hand)?
            } else {
                self.map(density, shape, hand)?
            };
            let mut harmonics = vec![Complex::default(); grid_len(coarse)?];
            for (q, z) in &map.terms {
                harmonics[grid_index(*q, coarse)] += z;
            }
            Transform::new(coarse).apply(&mut harmonics, true);
            let mut indices: Vec<_> = (0..harmonics.len()).collect();
            indices.sort_by(|&i, &j| harmonics[j].re.total_cmp(&harmonics[i].re).then(i.cmp(&j)));
            let mut seeds: Vec<[f64; 3]> = Vec::new();
            for index in indices {
                let coords = [
                    index / (coarse[1] * coarse[2]),
                    index / coarse[2] % coarse[1],
                    index % coarse[2],
                ];
                let mut local = true;
                for a in -1..=1 {
                    for b in -1..=1 {
                        for c in -1..=1 {
                            let offsets = [a, b, c];
                            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                            let near = grid_index(
                                std::array::from_fn(|i| coords[i] as i32 + offsets[i]),
                                coarse,
                            );
                            if harmonics[near].re > harmonics[index].re + 1e-12 * map.weight {
                                local = false;
                            }
                        }
                    }
                }
                if !local {
                    continue;
                }
                let shift = [
                    index / (coarse[1] * coarse[2]),
                    index / coarse[2] % coarse[1],
                    index % coarse[2],
                ];
                let shift = std::array::from_fn(|a| shift[a] as f64 / coarse[a] as f64);
                if seeds.iter().any(|&s| self.equivalent(s, shift)) {
                    continue;
                }
                seeds.push(shift);
                if seeds.len() >= 8 {
                    break;
                }
            }
            for mut shift in seeds {
                if crate::solution::stopped(deadline, cancellation) {
                    break;
                }
                for _ in 0..24 {
                    if crate::solution::stopped(deadline, cancellation) {
                        break;
                    }
                    let old = map.evaluate(shift)?;
                    let mut hessian = old.hessian;
                    for (a, row) in hessian.iter_mut().enumerate() {
                        row[a] += 1e-6;
                    }
                    let mut step = pseudo_solve(hessian, old.gradient.map(|v| -v));
                    let largest = step.iter().map(|v| v.abs()).fold(0., f64::max);
                    if largest > 0.1 {
                        step = step.map(|v| v * 0.1 / largest);
                    }
                    if dot(step, step) < 1e-20 {
                        break;
                    }
                    let mut accepted = false;
                    for _ in 0..12 {
                        let next = std::array::from_fn(|a| (shift[a] + step[a]).rem_euclid(1.));
                        if map.evaluate(next)?.chord < old.chord - 1e-14 {
                            shift = next;
                            accepted = true;
                            break;
                        }
                        step = step.map(|v| v * 0.5);
                    }
                    if !accepted {
                        break;
                    }
                }
                let e = map.evaluate(shift)?;
                let candidate = PhaseOriginCandidate {
                    hand,
                    shift,
                    alpha: e.alpha,
                    chord: e.chord,
                };
                if let Some(old) = candidates.iter_mut().find(|c| {
                    let relative = c.hand * hand;
                    let delta =
                        std::array::from_fn(|a| shift[a] - f64::from(relative) * c.shift[a]);
                    self.normalizes(delta, relative)
                }) {
                    if candidate.alpha < old.alpha - 1e-12
                        || ((candidate.alpha - old.alpha).abs() <= 1e-12
                            && candidate.hand > old.hand)
                    {
                        *old = candidate;
                    }
                } else {
                    candidates.push(candidate);
                }
            }
        }
        candidates.sort_by(|a, b| {
            a.alpha
                .total_cmp(&b.alpha)
                .then(a.chord.total_cmp(&b.chord))
                .then(b.hand.cmp(&a.hand))
        });
        candidates.truncate(maximum);
        Ok(candidates)
    }
    fn equivalent(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        let d = std::array::from_fn::<_, 3, _>(|i| a[i] - b[i]);
        self.normalizes(d, 1)
    }
    fn normalizes(&self, d: [f64; 3], hand: i32) -> bool {
        self.group.operations().iter().all(|op| {
            let rotation = op.rotation();
            let translation = op
                .translation()
                .map(phasesmith_crystallography::Rational::as_f64);
            let mapped = std::array::from_fn::<_, 3, _>(|i| {
                f64::from(hand) * translation[i] + d[i]
                    - (0..3)
                        .map(|j| f64::from(rotation[i][j]) * d[j])
                        .sum::<f64>()
            });
            self.group.operations().iter().any(|other| {
                other.rotation() == rotation
                    && mapped.iter().zip(other.translation()).all(|(a, b)| {
                        let delta = (*a - b.as_f64() + 0.5).rem_euclid(1.) - 0.5;
                        delta.abs() < 1e-5
                    })
            })
        })
    }
}
impl PhaseOriginMap {
    /// Smooth and wrapped phase merit with analytical first/second derivatives.
    /// # Errors
    /// Rejects nonfinite origin shifts.
    pub fn evaluate(&self, shift: [f64; 3]) -> Result<PhaseOriginEvaluation, SolutionError> {
        if shift.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite phase origin"));
        }
        let mut out = PhaseOriginEvaluation {
            chord: 0.,
            gradient: [0.; 3],
            hessian: [[0.; 3]; 3],
            alpha: 0.,
            alpha_gradient: [0.; 3],
        };
        if self.weight == 0. {
            return Ok(out);
        }
        let tau = std::f64::consts::TAU;
        let a_scale = 3. / std::f64::consts::PI.powi(2);
        for &(q, z) in &self.terms {
            let q = q.map(f64::from);
            let value = z * Complex::from_polar(1., tau * dot(q, shift));
            let w = z.norm();
            let eta = value.arg();
            out.chord += w - value.re;
            out.alpha += w * eta * eta * a_scale;
            for a in 0..3 {
                out.gradient[a] += tau * q[a] * value.im;
                out.alpha_gradient[a] += 2. * a_scale * w * eta * tau * q[a];
                for b in 0..3 {
                    out.hessian[a][b] += tau * tau * q[a] * q[b] * value.re;
                }
            }
        }
        out.chord /= self.weight;
        out.alpha /= self.weight;
        for a in 0..3 {
            out.gradient[a] /= self.weight;
            out.alpha_gradient[a] /= self.weight;
            for b in 0..3 {
                out.hessian[a][b] /= self.weight;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use phasesmith_crystallography::{Rational, SymmetryOperation};
    #[test]
    fn cached_fourier_preserves_both_hands_origins_and_fused_derivatives_bitwise() {
        let shape = [16; 3];
        let density: Vec<_> = (0..4096)
            .map(|i| {
                let node = [i / 256, i / 16 % 16, i % 16];
                [[0.13, 0.27, 0.41], [0.71, 0.39, 0.11], [0.37, 0.63, 0.83]]
                    .iter()
                    .map(|p| {
                        let d2: f64 = (0..3)
                            .map(|a| {
                                let d =
                                    (f64::from(node[a]) / 16. - p[a] + 0.5).rem_euclid(1.) - 0.5;
                                d * d
                            })
                            .sum();
                        (-d2 / 0.004).exp()
                    })
                    .sum::<f64>()
            })
            .collect();
        let cached = PhaseOriginDensity::new(&density, shape).unwrap();
        for rotation in [
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ] {
            let group = SpaceGroup::new(vec![
                SymmetryOperation::identity(),
                SymmetryOperation::new(rotation, [Rational::zero(); 3]).unwrap(),
            ])
            .unwrap();
            let mut hkl = Vec::new();
            let mut observed = Vec::new();
            for h in -2i32..=2 {
                for k in -2i32..=2 {
                    for l in -2i32..=2 {
                        if [h, k, l] != [0; 3] {
                            hkl.push([h, k, l]);
                            observed.push(f64::from(1 + h * h + 2 * k * k + 3 * l * l).sqrt());
                        }
                    }
                }
            }
            let plan = PreparedPhaseOrigin::new(&group, &hkl, &observed).unwrap();
            for hand in [1, -1] {
                let a = plan
                    .map(&density, shape, hand)
                    .unwrap()
                    .evaluate([0.17, 0.29, 0.43])
                    .unwrap();
                let b = plan
                    .map_cached(&cached, hand)
                    .unwrap()
                    .evaluate([0.17, 0.29, 0.43])
                    .unwrap();
                assert_eq!(a.alpha.to_bits(), b.alpha.to_bits());
                assert_eq!(a.chord.to_bits(), b.chord.to_bits());
                assert_eq!(a.gradient.map(f64::to_bits), b.gradient.map(f64::to_bits));
                assert_eq!(
                    a.hessian.map(|v| v.map(f64::to_bits)),
                    b.hessian.map(|v| v.map(f64::to_bits))
                );
                assert_eq!(
                    a.alpha_gradient.map(f64::to_bits),
                    b.alpha_gradient.map(f64::to_bits)
                );
            }
            let a = plan
                .search_sampled(&density, shape, 4, None, None, true)
                .unwrap();
            let b = plan.search_cached(&cached, 4, None, None, true).unwrap();
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.hand, b.hand);
                assert_eq!(a.shift.map(f64::to_bits), b.shift.map(f64::to_bits));
                assert_eq!(a.alpha.to_bits(), b.alpha.to_bits());
                assert_eq!(a.chord.to_bits(), b.chord.to_bits());
            }
            assert!(plan.map_cached(&cached, 0).is_err());
            assert!(PhaseOriginDensity::new(&[0.; 64], [4; 3]).is_err());
            let edge =
                PreparedPhaseOrigin::new(&group, &[[-4, 0, 0], [4, 0, 0]], &[1., 1.]).unwrap();
            assert!(
                edge.map_cached(&PhaseOriginDensity::new(&[0.; 512], [8; 3]).unwrap(), 1)
                    .is_err()
            );
        }
        assert!(PhaseOriginDensity::new(&density[..4095], shape).is_err());
        let mut bad = density;
        bad[0] = f64::NAN;
        assert!(PhaseOriginDensity::new(&bad, shape).is_err());
    }
}
