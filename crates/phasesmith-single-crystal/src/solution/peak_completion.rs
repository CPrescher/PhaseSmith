//! Cached, coordinate-fixed whole-orbit count/label proposals before refinement.
use crate::solution::math3::{dot, mv};
use crate::solution::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{CellGeometry, PreparedXrayScattering, SpaceGroup, UnitCell};
use rustfft::num_complex::Complex;
use std::time::Instant;

#[path = "assignment_ambiguity.rs"]
pub(crate) mod ambiguity;
#[path = "joint_assignment.rs"]
mod joint;
pub use joint::{JointAssignment, JointAssignments};
#[path = "chemical_labels.rs"]
mod chemical;
pub use ambiguity::{AssignmentAlternative, AssignmentAlternatives};
pub use chemical::{ChemicalScale, chemical_density_scales};

/// Shared peak-orbit phases and jets, without assigned element quantities.
#[derive(Clone, Debug)]
pub struct PreparedPeakCompletion {
    /// Proposed orbit representatives in the supplied setting.
    pub points: Vec<[f64; 3]>,
    /// Unique full-cell multiplicities.
    pub multiplicities: Vec<usize>,
    /// Candidate label order.
    pub elements: Vec<String>,
    phase: Vec<Vec<Complex<f64>>>,
    jets: Vec<Vec<[Complex<f64>; 3]>>,
    factors: Vec<f64>,
    attenuation: Vec<Vec<f64>>,
    observed: Vec<f64>,
    maximum_atoms: usize,
    s_squared: Vec<f64>,
    displacement_applied: bool,
    hkl: Vec<[i32; 3]>,
    group: SpaceGroup,
    geometry: CellGeometry,
}
/// Cheap complete-model proposal, with every required type represented.
#[derive(Clone, Debug)]
pub struct PeakCompletionCandidate {
    /// Label index per peak orbit, None denotes an omitted orbit.
    pub labels: Vec<Option<usize>>,
    /// Global isotropic displacement seed in square angstrom.
    pub u_iso: f64,
    /// Profiled amplitude SSE on the screening families.
    pub objective: f64,
    /// Descriptive isotropic BIC-like ranking, not a correctness certificate.
    pub score: f64,
    /// Full-cell non-H atom count.
    pub atoms: usize,
}
/// Actual cheap work, including rejected candidate topologies.
#[derive(Clone, Debug, Default)]
pub struct PeakCompletionResult {
    /// Complete, distinct candidates ordered by cheap ranking.
    pub candidates: Vec<PeakCompletionCandidate>,
    /// Coordinate-fixed amplitude model scores.
    pub evaluations: usize,
    /// Accepted categorical count/label changes.
    pub changes: usize,
}
impl PreparedPeakCompletion {
    /// Cache phase values and representative-coordinate jets together per image/reflection.
    /// # Errors
    /// Rejects malformed arrays, species, orbit counts or excessive cache memory.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    pub fn new(
        cell: UnitCell,
        group: &SpaceGroup,
        points: &[[f64; 3]],
        elements: &[String],
        hkl: &[[i32; 3]],
        observed: &[f64],
        maximum_atoms: usize,
    ) -> Result<Self, SolutionError> {
        if points.is_empty()
            || points.len() > 128
            || points.iter().flatten().any(|x| !x.is_finite())
            || elements.is_empty()
            || elements.len() > 8
            || hkl.is_empty()
            || hkl.len() > 2048
            || hkl.len() != observed.len()
            || observed
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 1e100)
            || hkl.windows(2).any(|p| p[0] >= p[1])
            || !(1..=128).contains(&maximum_atoms)
            || points.len() * hkl.len() > 262_144
            || hkl
                .iter()
                .any(|h| *h == [0; 3] || h.iter().any(|v| v.unsigned_abs() > 1024))
        {
            return Err(error("invalid peak completion arrays/bounds"));
        }
        if elements
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != elements.len()
            || elements.iter().any(|e| {
                e.len() > 2
                    || e.is_empty()
                    || phasesmith_crystallography::xray_species_metadata(e)
                        .is_none_or(|m| m.atomic_number <= 1)
                    || !e.as_bytes()[0].is_ascii_uppercase()
                    || e.as_bytes().get(1).is_some_and(|c| !c.is_ascii_lowercase())
            })
        {
            return Err(error(
                "completion needs distinct neutral non-H element types",
            ));
        }
        let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
        let g = geometry.direct_metric;
        let components = [g[0][0], g[1][1], g[2][2], g[1][2], g[0][2], g[0][1]];
        let metric_scale = components.iter().map(|v| v.abs()).fold(1., f64::max);
        for equation in &group.metric_constraints().equations {
            #[allow(clippy::cast_precision_loss)]
            let residual = equation
                .iter()
                .zip(components)
                .map(|(&a, b)| a as f64 * b)
                .sum::<f64>();
            #[allow(clippy::cast_precision_loss)]
            let magnitude = equation
                .iter()
                .map(|a| a.unsigned_abs() as f64)
                .sum::<f64>();
            if residual.abs() > 1e-10 * metric_scale * magnitude.max(1.) {
                return Err(error(
                    "completion cell is incompatible with supplied symmetry",
                ));
            }
        }
        let maximum = observed.iter().copied().fold(0., f64::max);
        if maximum <= 0. {
            return Err(error("no positive completion amplitudes"));
        }
        let observed = observed.iter().map(|v| v / maximum).collect();
        let s2: Vec<_> = hkl
            .iter()
            .map(|h| {
                let q = mv(geometry.reciprocal_basis, h.map(f64::from));
                dot(q, q) / 4.
            })
            .collect();
        let factors = PreparedXrayScattering::new(elements)
            .and_then(|p| p.evaluate(&s2.iter().map(|s| s.sqrt()).collect::<Vec<_>>()))
            .map_err(|e| error(e.to_string()))?
            .real;
        let mut phase = Vec::new();
        let mut jets = Vec::new();
        let mut multiplicities = Vec::new();
        let mut image_budget = 0;
        for &point in points {
            let expansion = group
                .expand_sites(&[point], 1e-8)
                .map_err(|e| error(e.to_string()))?;
            let count = expansion.fractional_xyz.len();
            image_budget += count;
            if count > maximum_atoms || image_budget * hkl.len() > 2_000_000 {
                return Err(error("peak completion image budget exceeded"));
            }
            multiplicities.push(count);
            let mut values = Vec::new();
            let mut derivatives = Vec::new();
            for &h in hkl {
                let h = h.map(f64::from);
                let mut value = Complex::default();
                let mut derivative = [Complex::default(); 3];
                for (&x, rotation) in expansion
                    .fractional_xyz
                    .iter()
                    .zip(&expansion.representative_rotation)
                {
                    let f = Complex::from_polar(1., std::f64::consts::TAU * dot(h, x));
                    value += f;
                    for a in 0..3 {
                        let k = std::f64::consts::TAU
                            * (0..3)
                                .map(|i| h[i] * f64::from(rotation[i][a]))
                                .sum::<f64>();
                        derivative[a] += Complex::new(-f.im * k, f.re * k);
                    }
                }
                values.push(value);
                derivatives.push(derivative);
            }
            phase.push(values);
            jets.push(derivatives);
        }
        let attenuation = [0., 0.005, 0.01, 0.02, 0.04, 0.08]
            .into_iter()
            .map(|u| {
                s2.iter()
                    .map(|s| (-8. * std::f64::consts::PI.powi(2) * u * s).exp())
                    .collect()
            })
            .collect();
        Ok(Self {
            points: points.to_vec(),
            multiplicities,
            elements: elements.to_vec(),
            phase,
            jets,
            factors,
            attenuation,
            observed,
            maximum_atoms,
            s_squared: s2,
            displacement_applied: false,
            hkl: hkl.to_vec(),
            group: group.clone(),
            geometry,
        })
    }
    fn sum(&self, labels: &[Option<usize>]) -> Vec<Complex<f64>> {
        let n = self.observed.len();
        let e = self.elements.len();
        let mut total = vec![Complex::default(); n];
        for (site, label) in labels.iter().enumerate() {
            if let Some(label) = label {
                for (h, t) in total.iter_mut().enumerate() {
                    *t += self.phase[site][h] * self.factors[h * e + label];
                }
            }
        }
        total
    }
    fn updated(
        &self,
        total: &[Complex<f64>],
        site: usize,
        old: Option<usize>,
        new: Option<usize>,
    ) -> Vec<Complex<f64>> {
        let e = self.elements.len();
        total
            .iter()
            .enumerate()
            .map(|(h, &t)| {
                let a = old.map_or(0., |l| self.factors[h * e + l]);
                let b = new.map_or(0., |l| self.factors[h * e + l]);
                t + self.phase[site][h] * (b - a)
            })
            .collect()
    }
    fn profile(&self, total: &[Complex<f64>]) -> (f64, usize) {
        let amplitudes: Vec<_> = total.iter().map(|t| t.norm()).collect();
        let norm: f64 = self.observed.iter().map(|v| v * v).sum();
        let mut best = (f64::INFINITY, 0);
        for (u, attenuation) in self.attenuation.iter().enumerate() {
            let mut cross = 0.;
            let mut square = 0.;
            for ((&a, &d), &o) in amplitudes.iter().zip(attenuation).zip(&self.observed) {
                cross += a * d * o;
                square += (a * d).powi(2);
            }
            if square <= 1e-250 {
                continue;
            }
            // Profiled nonnegative scale, using the residual form to avoid cancellation.
            let scale = cross / square;
            let objective = amplitudes
                .iter()
                .zip(attenuation)
                .zip(&self.observed)
                .map(|((&a, &d), &o)| (scale * a * d - o).powi(2))
                .sum::<f64>()
                / norm;
            if objective < best.0 {
                best = (objective, u);
            }
        }
        best
    }
    fn rank(&self, objective: f64, labels: &[Option<usize>]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let n = self.observed.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        let p = 4. * labels.iter().filter(|l| l.is_some()).count() as f64 + 2.;
        n * objective.max(1e-300).ln() + p * n.ln()
    }
    fn atoms(&self, labels: &[Option<usize>]) -> usize {
        labels
            .iter()
            .zip(&self.multiplicities)
            .filter(|(l, _)| l.is_some())
            .map(|(_, n)| n)
            .sum()
    }
    fn covered(&self, labels: &[Option<usize>]) -> bool {
        (0..self.elements.len()).all(|e| labels.contains(&Some(e)))
    }
    /// Evaluate one supplied label topology and all fractional-coordinate gradients.
    /// # Errors
    /// Rejects malformed labels, displacement-grid index or a zero model.
    pub fn evaluate(
        &self,
        labels: &[Option<usize>],
        u: usize,
    ) -> Result<(f64, Vec<[f64; 3]>), SolutionError> {
        if labels.len() != self.points.len()
            || labels.iter().flatten().any(|&e| e >= self.elements.len())
            || u >= self.attenuation.len()
            || self.atoms(labels) > self.maximum_atoms
        {
            return Err(error("invalid completion labels/U index"));
        }
        let total = self.sum(labels);
        let e = self.elements.len();
        let amplitudes: Vec<_> = total
            .iter()
            .zip(&self.attenuation[u])
            .map(|(c, d)| c.norm() * d)
            .collect();
        let square: f64 = amplitudes.iter().map(|a| a * a).sum();
        if square <= 1e-250 {
            return Err(error("zero completion model"));
        }
        let k = amplitudes
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| a * o)
            .sum::<f64>()
            / square;
        let norm = self.observed.iter().map(|v| v * v).sum::<f64>();
        let mut objective = 0.;
        let mut gradient = vec![[0.; 3]; self.points.len()];
        for (h, (&a, &o)) in amplitudes.iter().zip(&self.observed).enumerate() {
            let residual = k * a - o;
            objective += residual * residual / norm;
            let magnitude = total[h].norm();
            if magnitude <= 1e-100 {
                continue;
            }
            for (site, label) in labels.iter().enumerate() {
                if let Some(label) = label {
                    for (axis, g) in gradient[site].iter_mut().enumerate() {
                        let da = (total[h].conj() * self.jets[site][h][axis]).re / magnitude
                            * self.factors[h * e + label]
                            * self.attenuation[u][h];
                        *g += 2. * k * residual * da / norm;
                    }
                }
            }
        }
        Ok((objective, gradient))
    }
    /// Fit categorical assignments and whole-orbit counts on shared cached phases.
    /// # Errors
    /// Rejects nonpositive/invalid mass calibration; no coordinates are optimized.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    pub fn search(
        &self,
        masses: &[f64],
        scales: &[f64],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<PeakCompletionResult, SolutionError> {
        if masses.len() != self.points.len()
            || masses.iter().any(|v| !v.is_finite() || *v <= 0.)
            || scales.is_empty()
            || scales.len() > 8
            || scales.iter().any(|v| !v.is_finite() || *v <= 0.)
            || !(1..=8).contains(&maximum)
        {
            return Err(error("invalid completion masses/scales"));
        }
        let numbers: Vec<_> = self
            .elements
            .iter()
            .map(|e| {
                phasesmith_crystallography::xray_species_metadata(e)
                    .map_or(0., |m| f64::from(m.atomic_number))
            })
            .collect();
        if numbers.iter().any(|z| *z <= 1.) {
            return Err(error("completion needs neutral non-H types"));
        }
        let mut out = PeakCompletionResult::default();
        let mut raw = Vec::new();
        for &scale in scales {
            let mut labels = vec![None; self.points.len()];
            let mut prefix_total = vec![Complex::default(); self.observed.len()];
            let mut atoms = 0;
            for (site, &mass) in masses.iter().enumerate() {
                if crate::solution::stopped(deadline, cancellation) {
                    break;
                }
                if atoms + self.multiplicities[site] > self.maximum_atoms {
                    continue;
                }
                let estimate = mass / scale;
                let label = (0..numbers.len())
                    .min_by(|&i, &j| {
                        (estimate / numbers[i])
                            .ln()
                            .abs()
                            .total_cmp(&(estimate / numbers[j]).ln().abs())
                    })
                    .ok_or_else(|| error("completion has no allowed types"))?;
                labels[site] = Some(label);
                prefix_total = self.updated(&prefix_total, site, None, Some(label));
                atoms += self.multiplicities[site];
                if labels.iter().filter(|l| l.is_some()).count() < self.elements.len() {
                    continue;
                }
                // Rescue required types using density evidence; diffraction sweeps follow the shortlist.
                let mut seed = labels.clone();
                let mut total = prefix_total.clone();
                for missing in 0..numbers.len() {
                    if seed.contains(&Some(missing)) {
                        continue;
                    }
                    let mut best = None;
                    for i in 0..seed.len() {
                        let Some(old) = seed[i] else { continue };
                        if seed.iter().filter(|&&e| e == Some(old)).count() == 1 {
                            continue;
                        }
                        let value = (masses[i] / scale / numbers[missing]).ln().powi(2)
                            - (masses[i] / scale / numbers[old]).ln().powi(2);
                        if best.as_ref().is_none_or(|&(_, v)| value < v) {
                            best = Some((i, value));
                        }
                    }
                    if let Some((i, _)) = best {
                        total = self.updated(&total, i, seed[i], Some(missing));
                        seed[i] = Some(missing);
                    }
                }
                if !self.covered(&seed) {
                    continue;
                }
                let (j, u) = self.profile(&total);
                out.evaluations += 1;
                raw.push((self.rank(j, &seed), j, u, seed));
            }
        }
        raw.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut seeds: Vec<Vec<Option<usize>>> = Vec::new();
        for (_, _, _, seed) in raw {
            if !seeds.contains(&seed) {
                seeds.push(seed);
            }
            if seeds.len() >= maximum.max(4) {
                break;
            }
        }
        for mut labels in seeds {
            if crate::solution::stopped(deadline, cancellation) {
                break;
            }
            let mut total = self.sum(&labels);
            let (mut value, mut u) = self.profile(&total);
            let mut score = self.rank(value, &labels);
            for _ in 0..4 {
                let mut changed = false;
                for site in 0..labels.len() {
                    if crate::solution::stopped(deadline, cancellation) {
                        break;
                    }
                    let old = labels[site];
                    let mut best = None;
                    for new in std::iter::once(None).chain((0..numbers.len()).map(Some)) {
                        if new == old {
                            continue;
                        }
                        let mut next = labels.clone();
                        next[site] = new;
                        if !self.covered(&next) || self.atoms(&next) > self.maximum_atoms {
                            continue;
                        }
                        let proposal = self.updated(&total, site, old, new);
                        let (j, nu) = self.profile(&proposal);
                        out.evaluations += 1;
                        let rank = self.rank(j, &next);
                        if rank < score - 1e-8
                            && best.as_ref().is_none_or(|(s, _, _, _, _)| rank < *s)
                        {
                            best = Some((rank, j, nu, new, proposal));
                        }
                    }
                    if let Some((rank, j, nu, new, proposal)) = best {
                        labels[site] = new;
                        total = proposal;
                        score = rank;
                        value = j;
                        u = nu;
                        out.changes += 1;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            if !out.candidates.iter().any(|c| c.labels == labels) {
                let atoms = self.atoms(&labels);
                out.candidates.push(PeakCompletionCandidate {
                    labels,
                    u_iso: [0., 0.005, 0.01, 0.02, 0.04, 0.08][u],
                    objective: value,
                    score,
                    atoms,
                });
            }
        }
        out.candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        out.candidates.truncate(maximum);
        Ok(out)
    }
}
