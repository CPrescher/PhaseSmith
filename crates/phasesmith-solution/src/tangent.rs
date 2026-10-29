//! Count-independent phase invariants and a bounded annealed tangent projection.
//! Equations and explicit statistical limitations: docs/solution-tangent-refinement.md.
use crate::math3::{dot, mv};
use crate::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{SpaceGroup, UnitCell};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// One signed independent Friedel phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhaseTerm {
    /// Independent-variable index.
    pub index: usize,
    /// +1 for h, -1 for its Friedel mate.
    pub sign: i32,
}
impl PhaseTerm {
    fn value(self, phases: &[f64]) -> f64 {
        f64::from(self.sign) * phases[self.index]
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn code(self) -> i32 {
        self.sign * (self.index as i32 + 1)
    }
}
/// One tangent target and its measured strong partners.
#[derive(Clone, Debug)]
pub struct PhaseTriplet {
    /// Canonical target h.
    pub target: usize,
    /// Signed k and h-k phases.
    pub partners: [PhaseTerm; 2],
    /// Product of partner normalized magnitudes.
    pub weight: f64,
}
/// Four measured strong phases whose signed HKLs sum to zero.
#[derive(Clone, Debug)]
pub struct NegativeQuartet {
    /// Four distinct independent phase variables.
    pub terms: [PhaseTerm; 4],
    /// Principal-magnitude product times weak-cross evidence.
    pub weight: f64,
}
/// Fused phase-consistency values and analytical derivatives.
#[derive(Clone, Debug)]
pub struct TangentEvaluation {
    /// Weighted mean 1-cos(triplet sum), or zero if unavailable.
    pub triplet: f64,
    /// Weighted mean 1+cos(quartet sum), or zero if unavailable.
    pub negative_quartet: f64,
    /// Derivative of the triplet value.
    pub triplet_gradient: Vec<f64>,
    /// Derivative of the quartet value.
    pub quartet_gradient: Vec<f64>,
}
/// One actual optional reciprocal step, before the hybrid magnitude update.
#[derive(Clone, Debug)]
pub struct TangentStep {
    /// Returned independent phases, in radians.
    pub phases: Vec<f64>,
    /// Declared annealing temperature this cycle.
    pub temperature: f64,
    /// Non-frozen variables with an available tangent estimate.
    pub proposals: usize,
    /// Actual changed independent phases.
    pub changed: usize,
    /// Actual accepted blend, zero on decline.
    pub blend: f64,
    /// Quartet consistency before the step.
    pub quartet_before: f64,
    /// Quartet consistency after the step.
    pub quartet_after: f64,
}
/// Actual phase-plan and reciprocal work for one phasing start.
#[derive(Clone, Debug, Default)]
pub struct TangentAudit {
    /// Phase start index (filled by the shared-search caller).
    pub trial: usize,
    /// Number of independent measured Friedel pairs.
    pub variables: usize,
    /// Number of strong selected variables (a bound, not atoms).
    pub strong: usize,
    /// Actual retained directed triplets.
    pub triplets: usize,
    /// Actual retained negative quartets.
    pub quartets: usize,
    /// Whether quartet discovery reached a declared work/candidate cap.
    pub truncated: bool,
    /// Completed reciprocal updates.
    pub steps: usize,
    /// Completed steps with nonzero annealing temperature.
    pub annealed_steps: usize,
    /// Total proposed independent-phase changes.
    pub proposals: usize,
    /// Total accepted independent-phase changes.
    pub changed: usize,
    /// Whole steps declined by the quartet guard.
    pub declined_steps: usize,
    /// Candidate phase steps subsequently restored to supplied symmetry.
    pub symmetry_restorations: usize,
    /// Such restored candidates declined by the additional quartet check.
    pub symmetry_declined_steps: usize,
    /// Per-step quartet values before projection.
    pub quartet_before: Vec<f64>,
    /// Per-step quartet values after projection.
    pub quartet_after: Vec<f64>,
}
/// Prepared measured-only relations; never requires composition or atom counts.
#[derive(Clone, Debug)]
pub struct PreparedTangent {
    /// Canonical h > -h, in input order.
    pub(crate) hkl: Vec<[i32; 3]>,
    /// Normalized measured amplitudes for independent pairs.
    pub(crate) normalized: Vec<f64>,
    /// Two-sigma upper normalized amplitudes for weak-cross eligibility.
    pub(crate) upper: Vec<f64>,
    /// Strong-variable order, decreasing E, ties by index.
    pub(crate) strong: Vec<usize>,
    /// Retained triplets.
    pub(crate) triplets: Vec<PhaseTriplet>,
    /// Retained negative quartets.
    pub(crate) quartets: Vec<NegativeQuartet>,
    /// Full input rows mapped to independent phases.
    pub(crate) full_terms: Vec<PhaseTerm>,
    /// Resolution-shell centers in squared reciprocal angstrom.
    pub(crate) shell_centers: Vec<f64>,
    /// Corresponding scaled mean intensity/epsilon.
    pub(crate) shell_means: Vec<f64>,
    /// Explicit quartet resource-cap outcome.
    pub(crate) truncated: bool,
    by_target: Vec<Vec<usize>>,
    by_quartet: Vec<Vec<usize>>,
}
fn stopped(
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<(), SolutionError> {
    if crate::stopped(deadline, cancellation) {
        Err(error("tangent preparation stopped"))
    } else {
        Ok(())
    }
}
type Normalization = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
type Pair = ([PhaseTerm; 2], f64);

#[allow(clippy::cast_precision_loss)]
fn normalize(
    q: &[f64],
    intensity: &[f64],
    sigma: &[f64],
    epsilon: &[f64],
) -> Result<Normalization, SolutionError> {
    let n = q.len();
    let maximum = intensity.iter().copied().fold(0., f64::max);
    if maximum <= 0. {
        return Err(error("tangent normalization needs positive observations"));
    }
    let values: Vec<_> = intensity
        .iter()
        .zip(epsilon)
        .map(|(&i, &e)| i.max(0.) / maximum / e)
        .collect();
    let global = values.iter().sum::<f64>() / n as f64;
    let bins = (n / 32).clamp(1, 8);
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| q[a].total_cmp(&q[b]).then(a.cmp(&b)));
    let mut centers = Vec::new();
    let mut means = Vec::new();
    for b in 0..bins {
        let indices = &order[b * n / bins..(b + 1) * n / bins];
        centers.push(indices.iter().map(|&i| q[i]).sum::<f64>() / indices.len() as f64);
        let mean = indices.iter().map(|&i| values[i]).sum::<f64>() / indices.len() as f64;
        means.push(if mean > global * 1e-12 { mean } else { global });
    }
    // Coincident resolution shells must not acquire arbitrary intensity steps.
    let mut merged_centers: Vec<f64> = Vec::new();
    let mut merged_means = Vec::new();
    let mut populations = Vec::new();
    for b in 0..bins {
        let population = (b + 1) * n / bins - b * n / bins;
        if let Some(last) = merged_centers
            .last()
            .filter(|&&c| (c - centers[b]).abs() <= 1e-12 * (1. + c.abs()))
        {
            let index = merged_means.len() - 1;
            let total = populations[index] + population;
            let center =
                (*last * populations[index] as f64 + centers[b] * population as f64) / total as f64;
            merged_centers[index] = center;
            merged_means[index] = (merged_means[index] * populations[index] as f64
                + means[b] * population as f64)
                / total as f64;
            populations[index] = total;
        } else {
            merged_centers.push(centers[b]);
            merged_means.push(means[b]);
            populations.push(population);
        }
    }
    let centers = merged_centers;
    let means = merged_means;
    let bins = centers.len();
    let mut e = Vec::new();
    let mut upper = Vec::new();
    for (i, &value) in q.iter().enumerate() {
        let right = centers.partition_point(|&c| c < value);
        let mean = if right == 0 {
            means[0]
        } else if right == bins {
            means[bins - 1]
        } else {
            let width = centers[right] - centers[right - 1];
            if width <= 0. {
                means[right]
            } else {
                let fraction = (value - centers[right - 1]) / width;
                means[right - 1] * (1. - fraction) + means[right] * fraction
            }
        };
        e.push((values[i] / mean).sqrt());
        let bound = (intensity[i] + 2. * sigma[i]).max(0.) / maximum / epsilon[i] / mean;
        if !bound.is_finite() {
            return Err(error("tangent normalized uncertainty overflow"));
        }
        upper.push(bound.sqrt());
    }
    Ok((e, upper, centers, means))
}
fn canonical_codes(terms: [PhaseTerm; 4]) -> [i32; 4] {
    let mut codes = terms.map(PhaseTerm::code);
    codes.sort_unstable();
    let mut inverse = codes.map(|v| -v);
    inverse.sort_unstable();
    codes.min(inverse)
}
impl PreparedTangent {
    /// Immutable independent reciprocal indices.
    #[must_use]
    pub fn independent_hkl(&self) -> &[[i32; 3]] {
        &self.hkl
    }
    /// Immutable normalized magnitudes and two-sigma upper bounds.
    #[must_use]
    pub fn normalized_values(&self) -> (&[f64], &[f64]) {
        (&self.normalized, &self.upper)
    }
    /// Immutable selected independent-variable indices.
    #[must_use]
    pub fn selected(&self) -> &[usize] {
        &self.strong
    }
    /// Immutable retained triplet relations.
    #[must_use]
    pub fn triplet_relations(&self) -> &[PhaseTriplet] {
        &self.triplets
    }
    /// Immutable retained negative-quartet relations.
    #[must_use]
    pub fn quartet_relations(&self) -> &[NegativeQuartet] {
        &self.quartets
    }
    /// Immutable full-input/Friedel mapping.
    #[must_use]
    pub fn mapping(&self) -> &[PhaseTerm] {
        &self.full_terms
    }
    pub(crate) fn matches_hkl(&self, hkl: &[[i32; 3]]) -> bool {
        self.full_terms.len() == hkl.len()
            && hkl
                .iter()
                .zip(&self.full_terms)
                .all(|(h, t)| self.hkl[t.index].map(|value| value * t.sign) == *h)
    }
    /// Immutable resolution-shell centers and scaled means.
    #[must_use]
    pub fn shells(&self) -> (&[f64], &[f64]) {
        (&self.shell_centers, &self.shell_means)
    }
    /// Prepare bounded relations from sorted, symmetry-expanded measured observations.
    /// # Errors
    /// Rejects malformed/nonfinite arrays, absent reflections, missing/inconsistent
    /// Friedel mates, normalization overflow, or cancellation during preparation.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    pub fn new(
        cell: UnitCell,
        group: &SpaceGroup,
        hkl: &[[i32; 3]],
        intensity: &[f64],
        sigma: &[f64],
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<Self, SolutionError> {
        if hkl.is_empty()
            || hkl.len() > 100_000
            || hkl.len() != intensity.len()
            || hkl.len() != sigma.len()
            || hkl.windows(2).any(|p| p[0] >= p[1])
            || hkl
                .iter()
                .any(|h| *h == [0; 3] || h.iter().any(|v| i64::from(*v).abs() > 1_000_000))
            || intensity.iter().any(|v| !v.is_finite() || v.abs() > 1e100)
            || sigma
                .iter()
                .any(|v| !v.is_finite() || *v <= 0. || *v > 1e100)
        {
            return Err(error("invalid tangent observation arrays"));
        }
        let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
        let mut independent = Vec::new();
        let mut ii = Vec::new();
        let mut ss = Vec::new();
        let mut q = Vec::new();
        let mut epsilon = Vec::new();
        for (i, &h) in hkl.iter().enumerate() {
            stopped(deadline, cancellation)?;
            if group
                .is_systematically_absent(h)
                .map_err(|e| error(e.to_string()))?
            {
                return Err(error("tangent input includes systematic absence"));
            }
            let negative = h.map(|v| -v);
            let mate = hkl
                .binary_search(&negative)
                .map_err(|_| error("tangent missing Friedel mate"))?;
            if (intensity[i] - intensity[mate]).abs() > 1e-10 * (1. + intensity[i].abs())
                || (sigma[i] - sigma[mate]).abs() > 1e-10 * (1. + sigma[i])
            {
                return Err(error("tangent Friedel observations disagree"));
            }
            if h < negative {
                continue;
            }
            independent.push(h);
            ii.push(intensity[i]);
            ss.push(sigma[i]);
            let vector = mv(geometry.reciprocal_basis, h.map(f64::from));
            q.push(dot(vector, vector));
            let count = group
                .operations()
                .iter()
                .filter(|op| {
                    let r = op.rotation();
                    (0..3).all(|a| {
                        (0..3)
                            .map(|b| i64::from(h[b]) * i64::from(r[b][a]))
                            .sum::<i64>()
                            == i64::from(h[a])
                    })
                })
                .count();
            epsilon.push(count as f64);
        }
        let n = independent.len();
        let (normalized, upper, shell_centers, shell_means) = normalize(&q, &ii, &ss, &epsilon)?;
        let mut lookup = BTreeMap::new();
        for (index, &h) in independent.iter().enumerate() {
            lookup.insert(h, PhaseTerm { index, sign: 1 });
            lookup.insert(h.map(|v| -v), PhaseTerm { index, sign: -1 });
        }
        let full_terms = hkl.iter().map(|h| lookup[h]).collect();
        let mut strong: Vec<_> = (0..n)
            .filter(|&i| normalized[i] >= 1.2 && ii[i] >= 2. * ss[i])
            .collect();
        strong.sort_by(|&a, &b| normalized[b].total_cmp(&normalized[a]).then(a.cmp(&b)));
        strong.truncate(256);
        let selected: BTreeSet<_> = strong.iter().copied().collect();
        let signed: Vec<_> = lookup
            .iter()
            .filter(|(_, t)| selected.contains(&t.index))
            .map(|(&h, &t)| (h, t))
            .collect();
        let mut triplets = Vec::new();
        let mut by_target = vec![Vec::new(); n];
        for &target in &strong {
            stopped(deadline, cancellation)?;
            let h = independent[target];
            let mut candidates = Vec::new();
            let mut seen = BTreeSet::new();
            for &(k, first) in &signed {
                let l = std::array::from_fn(|i| h[i] - k[i]);
                let Some(&second) = lookup.get(&l) else {
                    continue;
                };
                if !selected.contains(&second.index)
                    || first.index == second.index
                    || first.index == target
                    || second.index == target
                {
                    continue;
                }
                let mut key = [first.code(), second.code()];
                key.sort_unstable();
                if !seen.insert(key) {
                    continue;
                }
                candidates.push((
                    normalized[first.index] * normalized[second.index],
                    key,
                    [first, second],
                ));
            }
            candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (weight, _, partners) in candidates.into_iter().take(64) {
                by_target[target].push(triplets.len());
                triplets.push(PhaseTriplet {
                    target,
                    partners,
                    weight,
                });
            }
        }
        // Pair groups only exist when their cross reflection is actually measured weak.
        let mut pairs: BTreeMap<[i32; 3], Vec<Pair>> = BTreeMap::new();
        for (a, &(h, first)) in signed.iter().enumerate() {
            stopped(deadline, cancellation)?;
            for &(k, second) in &signed[a + 1..] {
                if first.index == second.index {
                    continue;
                }
                let cross = std::array::from_fn(|i| h[i] + k[i]);
                if lookup.get(&cross).is_some_and(|t| upper[t.index] < 0.9) {
                    pairs.entry(cross).or_default().push((
                        [first, second],
                        normalized[first.index] * normalized[second.index],
                    ));
                }
            }
        }
        for values in pairs.values_mut() {
            values.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            values.truncate(32);
        }
        let mut candidates = Vec::new();
        let mut seen = BTreeSet::new();
        let mut work = 0;
        let mut truncated = false;
        'groups: for (&cross, first) in &pairs {
            stopped(deadline, cancellation)?;
            let negative = cross.map(|v| -v);
            if cross < negative {
                continue;
            }
            let Some(second) = pairs.get(&negative) else {
                continue;
            };
            for &(ab, _) in first {
                for &(cd, _) in second {
                    work += 1;
                    if work > 2_000_000 || candidates.len() >= 32_768 {
                        truncated = true;
                        break 'groups;
                    }
                    let terms = [ab[0], ab[1], cd[0], cd[1]];
                    if terms.iter().map(|t| t.index).collect::<BTreeSet<_>>().len() != 4 {
                        continue;
                    }
                    let codes = canonical_codes(terms);
                    if seen.contains(&codes) {
                        continue;
                    }
                    let left = independent[ab[0].index].map(|v| v * ab[0].sign);
                    let right0 = independent[cd[0].index].map(|v| v * cd[0].sign);
                    let right1 = independent[cd[1].index].map(|v| v * cd[1].sign);
                    let crosses = [
                        cross,
                        std::array::from_fn(|i| left[i] + right0[i]),
                        std::array::from_fn(|i| left[i] + right1[i]),
                    ];
                    let mut evidence = 1.;
                    let mut valid = true;
                    for key in crosses {
                        if let Some(t) = lookup.get(&key).filter(|t| upper[t.index] < 0.9) {
                            evidence *= 1. - upper[t.index] / 0.9;
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    seen.insert(codes);
                    let weight =
                        terms.iter().map(|t| normalized[t.index]).product::<f64>() * evidence;
                    candidates.push((weight, codes, terms));
                }
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut quartets = Vec::new();
        let mut by_quartet = vec![Vec::new(); n];
        for (weight, _, terms) in candidates {
            if quartets.len() >= 4096 {
                break;
            }
            if terms.iter().any(|t| by_quartet[t.index].len() >= 16) {
                continue;
            }
            for term in terms {
                by_quartet[term.index].push(quartets.len());
            }
            quartets.push(NegativeQuartet { terms, weight });
        }
        Ok(Self {
            hkl: independent,
            normalized,
            upper,
            strong,
            triplets,
            quartets,
            full_terms,
            shell_centers,
            shell_means,
            truncated,
            by_target,
            by_quartet,
        })
    }
    fn validate_phases(&self, phases: &[f64]) -> Result<(), SolutionError> {
        if phases.len() != self.hkl.len() || phases.iter().any(|v| !v.is_finite() || v.abs() > 1e8)
        {
            Err(error("invalid independent tangent phases"))
        } else {
            Ok(())
        }
    }
    /// Evaluate relation values and phase gradients in the same passes.
    /// # Errors
    /// Rejects nonfinite phases or a wrong independent-variable length.
    pub fn evaluate(&self, phases: &[f64]) -> Result<TangentEvaluation, SolutionError> {
        self.validate_phases(phases)?;
        let mut result = TangentEvaluation {
            triplet: 0.,
            negative_quartet: 0.,
            triplet_gradient: vec![0.; phases.len()],
            quartet_gradient: vec![0.; phases.len()],
        };
        let total = self.triplets.iter().map(|t| t.weight).sum::<f64>();
        for t in &self.triplets {
            let angle = phases[t.target] - t.partners.iter().map(|p| p.value(phases)).sum::<f64>();
            let (sin, cos) = angle.sin_cos();
            let w = t.weight / total;
            result.triplet += w * (1. - cos);
            result.triplet_gradient[t.target] += w * sin;
            for p in t.partners {
                result.triplet_gradient[p.index] -= w * f64::from(p.sign) * sin;
            }
        }
        let (value, gradient) = self.quartet_value_gradient(phases);
        result.negative_quartet = value;
        result.quartet_gradient = gradient;
        Ok(result)
    }
    // The acceptance guard needs only quartets. Keep their values and derivatives
    // fused without reevaluating every triplet during each backtracking attempt.
    pub(crate) fn quartet_value_gradient(&self, phases: &[f64]) -> (f64, Vec<f64>) {
        let mut value = 0.;
        let mut gradient = vec![0.; phases.len()];
        let total = self.quartets.iter().map(|t| t.weight).sum::<f64>();
        for t in &self.quartets {
            let angle = t.terms.iter().map(|p| p.value(phases)).sum::<f64>();
            let (sin, cos) = angle.sin_cos();
            let w = t.weight / total;
            value += w * (1. + cos);
            for p in t.terms {
                gradient[p.index] -= w * f64::from(p.sign) * sin;
            }
        }
        (value, gradient)
    }
    fn local_quartet(&self, index: usize, value: f64, phases: &[f64]) -> f64 {
        self.by_quartet[index]
            .iter()
            .map(|&q| {
                let relation = &self.quartets[q];
                let sum = relation
                    .terms
                    .iter()
                    .map(|t| {
                        f64::from(t.sign)
                            * if t.index == index {
                                value
                            } else {
                                phases[t.index]
                            }
                    })
                    .sum::<f64>();
                relation.weight * (1. + sum.cos())
            })
            .sum()
    }
    /// One deterministic tangent/annealing step; measured magnitudes stay untouched.
    /// # Errors
    /// Rejects invalid phases, nonfinite/negative calculated magnitudes or controls.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn step(
        &self,
        phases: &[f64],
        magnitudes: &[f64],
        cycle: usize,
        seed: u64,
    ) -> Result<TangentStep, SolutionError> {
        self.step_with_tie_tolerance(phases, magnitudes, cycle, seed, 0.)
    }
    pub(crate) fn symmetry_step(
        &self,
        phases: &[f64],
        magnitudes: &[f64],
        cycle: usize,
        seed: u64,
    ) -> Result<TangentStep, SolutionError> {
        self.step_with_tie_tolerance(phases, magnitudes, cycle, seed, 1e-12)
    }
    // Symmetry phase lines create exact +/- cost ties. Resolve their floating
    // point neighborhood consistently instead of letting roundoff choose a sign.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    fn step_with_tie_tolerance(
        &self,
        phases: &[f64],
        magnitudes: &[f64],
        cycle: usize,
        seed: u64,
        relative_tie_tolerance: f64,
    ) -> Result<TangentStep, SolutionError> {
        self.validate_phases(phases)?;
        if magnitudes.len() != phases.len()
            || magnitudes.iter().any(|v| !v.is_finite() || *v < 0.)
            || cycle > 20_000
        {
            return Err(error("invalid tangent calculated magnitudes/cycle"));
        }
        let temperature = if cycle < 12 {
            0.35 * 0.85_f64.powi(i32::try_from(cycle).unwrap_or(12))
        } else {
            0.
        };
        let mut frozen = self.strong.clone();
        frozen.sort_by(|&a, &b| magnitudes[b].total_cmp(&magnitudes[a]).then(a.cmp(&b)));
        frozen.truncate(2 * frozen.len() / 5);
        let frozen: BTreeSet<_> = frozen.into_iter().collect();
        let mut proposed = phases.to_vec();
        let mut proposals = 0;
        for &target in &self.strong {
            if frozen.contains(&target) || self.by_target[target].is_empty() {
                continue;
            }
            let mut real = 0.;
            let mut imaginary = 0.;
            for &j in &self.by_target[target] {
                let t = &self.triplets[j];
                let angle = t.partners.iter().map(|p| p.value(phases)).sum::<f64>();
                let (s, c) = angle.sin_cos();
                real += t.weight * c;
                imaginary += t.weight * s;
            }
            if real.hypot(imaginary) <= 1e-12 {
                continue;
            }
            let angle = imaginary.atan2(real);
            let u = uniform(seed, cycle, target, 0);
            let delta = (temperature * (-u.ln())).min(std::f64::consts::PI / 3.);
            let plus = angle + delta;
            let minus = angle - delta;
            proposed[target] = if self.by_quartet[target].is_empty() {
                if uniform(seed, cycle, target, 1) < 0.5 {
                    minus
                } else {
                    plus
                }
            } else {
                let plus_cost = self.local_quartet(target, plus, phases);
                let minus_cost = self.local_quartet(target, minus, phases);
                let tolerance = relative_tie_tolerance * (1. + plus_cost.abs() + minus_cost.abs());
                if plus_cost <= minus_cost + tolerance {
                    plus
                } else {
                    minus
                }
            };
            proposals += 1;
        }
        let before = self.quartet_value_gradient(phases).0;
        let mut blend = 0.25;
        for _ in 0..=6 {
            let values: Vec<_> = phases
                .iter()
                .zip(&proposed)
                .map(|(&old, &new)| old + blend * wrap(new - old))
                .collect();
            let after = self.quartet_value_gradient(&values).0;
            if after <= before + 1e-12 {
                let changed = values
                    .iter()
                    .zip(phases)
                    .filter(|(a, b)| (*a - *b).abs() > 1e-14)
                    .count();
                return Ok(TangentStep {
                    phases: values,
                    temperature,
                    proposals,
                    changed,
                    blend,
                    quartet_before: before,
                    quartet_after: after,
                });
            }
            blend *= 0.5;
        }
        Ok(TangentStep {
            phases: phases.to_vec(),
            temperature,
            proposals,
            changed: 0,
            blend: 0.,
            quartet_before: before,
            quartet_after: before,
        })
    }
    /// Seed an actual-work diagnostic record before a phasing start.
    #[must_use]
    pub fn audit(&self) -> TangentAudit {
        TangentAudit {
            variables: self.hkl.len(),
            strong: self.strong.len(),
            triplets: self.triplets.len(),
            quartets: self.quartets.len(),
            truncated: self.truncated,
            ..TangentAudit::default()
        }
    }
}
fn wrap(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}
#[allow(clippy::cast_precision_loss)]
fn uniform(seed: u64, cycle: usize, index: usize, lane: u64) -> f64 {
    let mut value = seed
        ^ (cycle as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ lane.wrapping_mul(0x94d0_49bb_1331_11eb);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value >> 12) as f64 + 0.5) / 4_503_599_627_370_496.
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_crystallography::SymmetryOperation;
    #[test]
    fn preparation_obeys_stop_and_input_bounds() {
        let cell = UnitCell {
            a_angstrom: 8.,
            b_angstrom: 9.,
            c_angstrom: 10.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let hkl = [[-1, 0, 0], [1, 0, 0]];
        let token = Cancellation::default();
        let _ = token.cancel();
        assert!(
            PreparedTangent::new(
                cell,
                &group,
                &hkl,
                &[1., 1.],
                &[0.1, 0.1],
                None,
                Some(&token)
            )
            .unwrap_err()
            .to_string()
            .contains("stopped")
        );
        assert!(
            PreparedTangent::new(
                cell,
                &group,
                &hkl,
                &[1., 1.],
                &[0.1, 0.1],
                Some(Instant::now()),
                None
            )
            .is_err()
        );
        assert!(
            PreparedTangent::new(cell, &group, &hkl, &[1., 1.], &[0., 0.1], None, None).is_err()
        );
        assert!(
            PreparedTangent::new(
                cell,
                &group,
                &[[-1_000_001, 0, 0], [1_000_001, 0, 0]],
                &[1., 1.],
                &[0.1, 0.1],
                None,
                None
            )
            .is_err()
        );
    }
    #[test]
    fn stateless_randoms_are_open_unit_interval() {
        for seed in [0, u64::MAX] {
            for cycle in 0..20 {
                for index in 0..256 {
                    let value = uniform(seed, cycle, index, 0);
                    assert!(value > 0. && value < 1.);
                }
            }
        }
    }
}
