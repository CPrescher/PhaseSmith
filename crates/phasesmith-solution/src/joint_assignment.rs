//! Bounded categorical beam repair using refined anisotropic orbit columns.
use super::PreparedPeakCompletion;
use crate::math3::{conjugate, dot, eigen, mv, transpose};
use crate::{Cancellation, SolutionError, SymmetryModel, error};
use phasesmith_crystallography::Matrix3;
use rustfft::num_complex::Complex;
use std::time::Instant;
#[path = "repair_evaluation.rs"]
mod evaluation;

/// Complete discrete proposal with isotropic shifts of the original site tensors.
#[derive(Clone, Debug)]
pub struct JointAssignment {
    /// One label per original orbit, with None denoting a vacancy.
    pub labels: Vec<Option<usize>>,
    /// Additive diagonal shifts in square angstrom, relative to the original tensors.
    pub shifts: Vec<f64>,
    /// Profiled fixed-geometry amplitude SSE.
    pub objective: f64,
    /// Descriptive isotropic topology rank used only for proposal allocation.
    pub score: f64,
}
/// Actual beam work and retained distinct proposals.
#[derive(Clone, Debug, Default)]
pub struct JointAssignments {
    /// Distinct categorical hypotheses retained for physical refinement.
    pub candidates: Vec<JointAssignment>,
    /// Includes discarded numerical scores.
    pub evaluations: usize,
    /// Completed categorical beam expansions.
    pub depths: usize,
    /// Cached reflection fields actually built.
    pub profile_builds: usize,
    /// Reused cached fields, including categorical rescoring.
    pub profile_reuses: usize,
    /// Line-search requests that avoid objective derivatives.
    pub score_only_requests: usize,
    /// Maximum per-call bytes for base columns and reflection buffers.
    pub workspace_bytes: usize,
}
impl PreparedPeakCompletion {
    /// Apply supplied-symmetry Cartesian tensors to orbit values and coordinate jets together.
    /// # Errors
    /// Rejects changed orbit topology, invalid tensors, or mismatched reciprocal rows.
    #[allow(clippy::too_many_lines)]
    pub fn with_tensor_model(
        mut self,
        model: &SymmetryModel,
        tensors: &[Matrix3],
        hkl: &[[i32; 3]],
    ) -> Result<Self, SolutionError> {
        if self.displacement_applied
            || tensors.len() != self.points.len()
            || model.sites.len() != self.points.len()
            || hkl != self.hkl
            || model.group.operations() != self.group.operations()
            || model.geometry.direct_metric != self.geometry.direct_metric
        {
            return Err(error(
                "tensor completion requires matching bare orbit arrays",
            ));
        }
        for (i, site) in model.sites.iter().enumerate() {
            if site.images.len() != self.multiplicities[i]
                || site
                    .fractional_xyz
                    .iter()
                    .zip(self.points[i])
                    .any(|(a, b)| {
                        let d = (a - b).rem_euclid(1.);
                        d.min(1. - d) > 1e-8
                    })
            {
                return Err(error("tensor completion orbit setting differs"));
            }
            let tensor = tensors[i];
            if tensor.iter().flatten().any(|x| !x.is_finite())
                || (0..3).any(|a| (0..3).any(|b| (tensor[a][b] - tensor[b][a]).abs() > 1e-12))
            {
                return Err(error("invalid Cartesian displacement tensor"));
            }
            let (values, _) = eigen(tensor);
            if values
                .iter()
                .any(|x| !(1e-5 - 1e-12..=0.2 + 1e-12).contains(x))
                || site.stabilizer.iter().any(|&q| {
                    let image = conjugate(q, tensor);
                    image
                        .iter()
                        .flatten()
                        .zip(tensor.iter().flatten())
                        .any(|(a, b)| (a - b).abs() > 1e-9)
                })
            {
                return Err(error(
                    "tensor completion requires bounded site-invariant tensors",
                ));
            }
            for (row, &h) in hkl.iter().enumerate() {
                let hf = h.map(f64::from);
                let q = mv(model.geometry.reciprocal_basis, hf);
                if (dot(q, q) / 4. - self.s_squared[row]).abs() > 1e-10 * (1. + self.s_squared[row])
                {
                    return Err(error("tensor completion reciprocal rows differ"));
                }
                let mut value = Complex::default();
                let mut jet = [Complex::default(); 3];
                for image in &site.images {
                    let position = image.operation.apply_fractional(site.fractional_xyz);
                    let vector = mv(transpose(image.cartesian_rotation), q);
                    let attenuation =
                        (-2. * std::f64::consts::PI.powi(2) * dot(vector, mv(tensor, vector)))
                            .exp();
                    let f =
                        Complex::from_polar(attenuation, std::f64::consts::TAU * dot(hf, position));
                    value += f;
                    for (axis, derivative) in jet.iter_mut().enumerate() {
                        let factor = std::f64::consts::TAU
                            * (0..3)
                                .map(|a| hf[a] * f64::from(image.operation.rotation()[a][axis]))
                                .sum::<f64>();
                        *derivative += Complex::new(-f.im * factor, f.re * factor);
                    }
                }
                self.phase[i][row] = value;
                self.jets[i][row] = jet;
            }
        }
        self.displacement_applied = true;
        Ok(self)
    }
    fn shift_profile(
        &self,
        rest: &[Complex<f64>],
        site: usize,
        label: Option<usize>,
        shift: f64,
    ) -> (f64, f64, f64) {
        let beta = 8. * std::f64::consts::PI.powi(2);
        let width = self.elements.len();
        let mut amplitude = Vec::with_capacity(rest.len());
        let mut derivative = Vec::with_capacity(rest.len());
        for (h, &background) in rest.iter().enumerate() {
            let column = label.map_or(Complex::default(), |e| {
                self.phase[site][h]
                    * self.factors[h * width + e]
                    * (-beta * shift * self.s_squared[h]).exp()
            });
            let total = background + column;
            let a = total.norm();
            let da = if a > 1e-100 {
                (total.conj() * column).re * (-beta * self.s_squared[h]) / a
            } else {
                0.
            };
            amplitude.push(a);
            derivative.push(da);
        }
        let square: f64 = amplitude.iter().map(|a| a * a).sum();
        if square <= 1e-250 {
            return (f64::INFINITY, 0., 0.);
        }
        let scale = amplitude
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| a * o)
            .sum::<f64>()
            / square;
        let scale_derivative = derivative
            .iter()
            .zip(&self.observed)
            .zip(&amplitude)
            .map(|((da, o), a)| da * (o - 2. * scale * a))
            .sum::<f64>()
            / square;
        let norm: f64 = self.observed.iter().map(|o| o * o).sum();
        let (mut value, mut gradient, mut curvature) = (0., 0., 0.);
        for ((a, da), o) in amplitude.into_iter().zip(derivative).zip(&self.observed) {
            let r = scale * a - o;
            let dr = scale * da + scale_derivative * a;
            value += r * r / norm;
            gradient += 2. * r * dr / norm;
            curvature += 2. * dr * dr / norm;
        }
        (value, gradient, curvature)
    }
    /// Profiled amplitude objective, diagonal-tensor-shift derivative and GN curvature.
    /// Values and derivatives are accumulated from the same orbit/reflection evaluation.
    /// # Errors
    /// Rejects malformed labels/shifts, unsafe exponents or a site outside the plan.
    pub fn shift_score(
        &self,
        labels: &[Option<usize>],
        shifts: &[f64],
        site: usize,
    ) -> Result<(f64, f64, f64), SolutionError> {
        self.evaluate(labels, 0)?;
        let beta = 8. * std::f64::consts::PI.powi(2);
        if !self.displacement_applied
            || shifts.len() != labels.len()
            || site >= labels.len()
            || shifts.iter().any(|v| {
                !v.is_finite()
                    || v.abs() > 0.2
                    || self.s_squared.iter().any(|s| (beta * v * s).abs() > 600.)
            })
        {
            return Err(error("invalid diagonal tensor shifts/site"));
        }
        let width = self.elements.len();
        let mut rest = vec![Complex::default(); self.observed.len()];
        for (i, &label) in labels.iter().enumerate().filter(|(i, _)| *i != site) {
            if let Some(e) = label {
                for (h, t) in rest.iter_mut().enumerate() {
                    *t += self.phase[i][h]
                        * self.factors[h * width + e]
                        * (-beta * shifts[i] * self.s_squared[h]).exp();
                }
            }
        }
        Ok(self.shift_profile(&rest, site, labels[site], shifts[site]))
    }

    /// Coordinate-fixed categorical sweeps under an explicit tensor-shift proposal.
    /// This is a seed generator, not a displacement restraint or an accepted model.
    /// # Errors
    /// Rejects invalid shifts/bounds or incomplete required element coverage.
    #[allow(clippy::too_many_lines)]
    pub fn coherent_reassign(
        &self,
        labels: &[Option<usize>],
        shifts: &[f64],
        bounds: &[[f64; 2]],
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<JointAssignments, SolutionError> {
        self.shift_score(labels, shifts, 0)?;
        if bounds.len() != labels.len()
            || !self.covered(labels)
            || shifts.iter().zip(bounds).any(|(&v, b)| {
                !b.iter().all(|x| x.is_finite())
                    || b[0] > 0.
                    || b[1] < 0.
                    || b[0] < -0.2
                    || b[1] > 0.2
                    || v < b[0]
                    || v > b[1]
            })
        {
            return Err(error("invalid coherent tensor shifts/bounds"));
        }
        let beta = 8. * std::f64::consts::PI.powi(2);
        let mut plan = self.clone();
        for (i, &shift) in shifts.iter().enumerate() {
            for (h, &s) in self.s_squared.iter().enumerate() {
                let attenuation = (-beta * shift * s).exp();
                plan.phase[i][h] *= attenuation;
                for jet in &mut plan.jets[i][h] {
                    *jet *= attenuation;
                }
            }
        }
        let mut current = labels.to_vec();
        let mut total = plan.sum(&current);
        let mut value = plan.fixed_objective(&total);
        let numbers: Vec<_> = self
            .elements
            .iter()
            .map(|e| {
                phasesmith_crystallography::xray_species_metadata(e).map_or(0, |m| m.atomic_number)
            })
            .collect();
        let mut out = JointAssignments::default();
        for _ in 0..4 {
            let mut changed = false;
            for i in 0..current.len() {
                if crate::stopped(deadline, cancellation) {
                    return Ok(out);
                }
                let Some(old) = current[i] else { continue };
                let mut best = None;
                for new in 0..self.elements.len() {
                    if new == old || numbers[old].abs_diff(numbers[new]) > 2 {
                        continue;
                    }
                    let mut assignment = current.clone();
                    assignment[i] = Some(new);
                    if !plan.covered(&assignment) {
                        continue;
                    }
                    let next = plan.updated(&total, i, Some(old), Some(new));
                    let objective = plan.fixed_objective(&next);
                    out.evaluations += 1;
                    if objective < value - 1e-12
                        && best.as_ref().is_none_or(|(v, _, _)| objective < *v)
                    {
                        best = Some((objective, assignment, next));
                    }
                }
                if let Some((objective, assignment, next)) = best {
                    value = objective;
                    current = assignment;
                    total = next;
                    changed = true;
                }
            }
            out.depths += 1;
            if !changed {
                break;
            }
        }
        if current != labels {
            out.candidates.push(JointAssignment {
                score: self.rank(value, &current),
                objective: value,
                labels: current,
                shifts: shifts.to_vec(),
            });
        }
        Ok(out)
    }

    /// Beam-search coordinated labels and orbit vacancies/births with bounded tensor shifts.
    /// The cached phases already contain physical displacement. Bounds must contain zero.
    /// # Errors
    /// Rejects uncovered labels, invalid bounds, or maximum outside [1,4].
    /// # Panics
    /// Only if public element metadata was mutated after preparation.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn joint_neighbors(
        &self,
        labels: &[Option<usize>],
        bounds: &[[f64; 2]],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<JointAssignments, SolutionError> {
        self.joint_neighbors_with_cache(labels, bounds, maximum, deadline, cancellation, false)
    }
    /// Evaluate the same bounded categorical search with optional call-local numerical reuse.
    /// # Errors
    /// Uses the same input validation and resource limits as `joint_neighbors`.
    /// # Panics
    /// Only if public element metadata was mutated after preparation.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn joint_neighbors_with_cache(
        &self,
        labels: &[Option<usize>],
        bounds: &[[f64; 2]],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        cached: bool,
    ) -> Result<JointAssignments, SolutionError> {
        if cached {
            self.joint_neighbors_impl::<true>(labels, bounds, maximum, deadline, cancellation)
        } else {
            self.joint_neighbors_impl::<false>(labels, bounds, maximum, deadline, cancellation)
        }
    }
    #[allow(clippy::too_many_lines)]
    fn joint_neighbors_impl<const CACHED: bool>(
        &self,
        labels: &[Option<usize>],
        bounds: &[[f64; 2]],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<JointAssignments, SolutionError> {
        let cached = CACHED;
        self.evaluate(labels, 0)?;
        if !self.displacement_applied
            || !self.covered(labels)
            || bounds.len() != labels.len()
            || !(1..=4).contains(&maximum)
            || bounds.iter().any(|b| {
                !b.iter().all(|x| x.is_finite())
                    || b[0] > 0.
                    || b[1] < 0.
                    || b[0] < -0.2
                    || b[1] > 0.2
            })
        {
            return Err(error("invalid tensor-aware assignment bounds/labels"));
        }
        let width = self.elements.len();
        let numbers: Vec<_> = self
            .elements
            .iter()
            .map(|e| {
                phasesmith_crystallography::xray_species_metadata(e)
                    .expect("validated element")
                    .atomic_number
            })
            .collect();
        let beta = 8. * std::f64::consts::PI.powi(2);
        let norm: f64 = self.observed.iter().map(|o| o * o).sum();
        let s2 = self
            .s_squared
            .iter()
            .zip(&self.observed)
            .map(|(s, o)| s * o * o)
            .sum::<f64>()
            / norm;
        let anchor = self
            .s_squared
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - s2).abs().total_cmp(&(*b - s2).abs()))
            .expect("nonempty rows")
            .0;
        let columns = cached.then(|| evaluation::Columns::new(self));
        let mut workspace = cached.then(|| evaluation::Workspace::new(self));
        let mut rest_buffer = if cached {
            vec![Complex::default(); self.observed.len()]
        } else {
            Vec::new()
        };
        let column = |site: usize, label: Option<usize>, shift: f64| -> Vec<Complex<f64>> {
            self.s_squared
                .iter()
                .enumerate()
                .map(|(h, &s)| {
                    label.map_or(Complex::default(), |e| {
                        columns.as_ref().map_or_else(
                            || self.phase[site][h] * self.factors[h * width + e],
                            |cache| cache.value(self, site, e, h),
                        ) * (-beta * shift * s).clamp(-600., 600.).exp()
                    })
                })
                .collect()
        };
        let sum = |assignment: &JointAssignment| -> Vec<_> {
            let mut total = vec![Complex::default(); self.observed.len()];
            for (site, (&label, &shift)) in
                assignment.labels.iter().zip(&assignment.shifts).enumerate()
            {
                for (t, c) in total.iter_mut().zip(column(site, label, shift)) {
                    *t += c;
                }
            }
            total
        };
        let mut parent = JointAssignment {
            labels: labels.to_vec(),
            shifts: vec![0.; labels.len()],
            objective: 0.,
            score: 0.,
        };
        parent.objective = self.fixed_objective(&sum(&parent));
        parent.score = self.rank(parent.objective, labels);
        if !parent.objective.is_finite() {
            return Err(error("degenerate tensor assignment parent"));
        }
        let mut beam = vec![parent.clone()];
        let mut visited = vec![labels.to_vec()];
        let mut retained = Vec::<JointAssignment>::new();
        let mut out = JointAssignments::default();
        for _ in 0..8.min(labels.len().max(1)) {
            let mut next = Vec::<JointAssignment>::new();
            for state in &beam {
                let total = sum(state);
                for i in 0..labels.len() {
                    if crate::stopped(deadline, cancellation) {
                        break;
                    }
                    let old = state.labels[i];
                    let old_column = column(i, old, state.shifts[i]);
                    if cached {
                        for (value, (t, o)) in
                            rest_buffer.iter_mut().zip(total.iter().zip(&old_column))
                        {
                            *value = t - o;
                        }
                    }
                    for new in std::iter::once(None).chain((0..width).map(Some)) {
                        if new == old
                            || old
                                .zip(new)
                                .is_some_and(|(a, b)| numbers[a].abs_diff(numbers[b]) > 2)
                        {
                            continue;
                        }
                        let mut assignment = state.labels.clone();
                        assignment[i] = new;
                        if !self.covered(&assignment)
                            || self.atoms(&assignment) > self.maximum_atoms
                            || visited.contains(&assignment)
                        {
                            continue;
                        }
                        let delta = old.zip(new).map_or(0., |(a, b)| {
                            let ratio =
                                self.factors[anchor * width + b] / self.factors[anchor * width + a];
                            if s2 > 0. && ratio > 0. {
                                (ratio.ln() / (beta * s2)).clamp(-0.02, 0.02)
                            } else {
                                0.
                            }
                        });
                        let shifted = (state.shifts[i] + delta).clamp(bounds[i][0], bounds[i][1]);
                        let owned_rest: Vec<_>;
                        let rest = if cached {
                            rest_buffer.as_slice()
                        } else {
                            owned_rest =
                                total.iter().zip(&old_column).map(|(t, o)| t - o).collect();
                            &owned_rest
                        };
                        if let Some(work) = workspace.as_mut() {
                            work.begin();
                        }
                        let mut polished = if self
                            .s_squared
                            .iter()
                            .any(|s| (beta * shifted * s).abs() > 600.)
                        {
                            state.shifts[i]
                        } else {
                            shifted
                        };
                        for _ in 0..3 {
                            let (value, gradient, curvature) = if let (Some(work), Some(columns)) =
                                (workspace.as_mut(), columns.as_ref())
                            {
                                work.profile(self, columns, rest, i, new, polished, true)
                            } else {
                                self.shift_profile(rest, i, new, polished)
                            };
                            out.evaluations += 1;
                            if curvature <= 1e-20 {
                                break;
                            }
                            let step = (-gradient / curvature).clamp(-0.01, 0.01);
                            if step.abs() < 1e-8 {
                                break;
                            }
                            let mut accepted = false;
                            for factor in [1., 0.5, 0.25] {
                                let next =
                                    (polished + factor * step).clamp(bounds[i][0], bounds[i][1]);
                                if self
                                    .s_squared
                                    .iter()
                                    .any(|s| (beta * next * s).abs() > 600.)
                                {
                                    continue;
                                }
                                let (objective, _, _) = if let (Some(work), Some(columns)) =
                                    (workspace.as_mut(), columns.as_ref())
                                {
                                    work.profile(self, columns, rest, i, new, next, false)
                                } else {
                                    self.shift_profile(rest, i, new, next)
                                };
                                out.evaluations += 1;
                                if objective < value - 1e-14 {
                                    polished = next;
                                    accepted = true;
                                    break;
                                }
                            }
                            if !accepted {
                                break;
                            }
                        }
                        for (variant, shift) in
                            [state.shifts[i], shifted, polished].into_iter().enumerate()
                        {
                            if (variant > 0 && shift.to_bits() == state.shifts[i].to_bits())
                                || (variant == 2 && shift.to_bits() == shifted.to_bits())
                            {
                                continue;
                            }
                            // Extreme reciprocal ranges retain only numerically safe seeds.
                            if self
                                .s_squared
                                .iter()
                                .any(|s| (beta * shift * s).abs() > 600.)
                            {
                                continue;
                            }
                            let objective = if let (Some(work), Some(columns)) =
                                (workspace.as_mut(), columns.as_ref())
                            {
                                work.fixed_trial(self, columns, rest, i, new, shift)
                            } else {
                                let changed = column(i, new, shift);
                                let trial: Vec<_> = total
                                    .iter()
                                    .zip(&old_column)
                                    .zip(changed)
                                    .map(|((t, o), c)| t - o + c)
                                    .collect();
                                self.fixed_objective(&trial)
                            };
                            out.evaluations += 1;
                            if !objective.is_finite() || objective > 1.25 * parent.objective + 1e-4
                            {
                                continue;
                            }
                            let mut candidate = JointAssignment {
                                labels: assignment.clone(),
                                shifts: state.shifts.clone(),
                                objective,
                                score: self.rank(objective, &assignment),
                            };
                            candidate.shifts[i] = shift;
                            if let Some(existing) = next.iter_mut().find(|c| c.labels == assignment)
                            {
                                if candidate.score < existing.score {
                                    *existing = candidate;
                                }
                            } else {
                                next.push(candidate);
                            }
                        }
                    }
                }
            }
            if next.is_empty() || crate::stopped(deadline, cancellation) {
                break;
            }
            next.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.labels.cmp(&b.labels)));
            next.truncate(4);
            out.depths += 1;
            for candidate in &next {
                visited.push(candidate.labels.clone());
                if candidate.score <= parent.score + 6. {
                    retained.push(candidate.clone());
                }
            }
            retained.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.labels.cmp(&b.labels)));
            retained.truncate(32);
            beam = next;
        }
        let mut counts = Vec::<Vec<usize>>::new();
        for candidate in retained {
            let signature: Vec<_> = (0..width)
                .map(|e| {
                    candidate
                        .labels
                        .iter()
                        .zip(&self.multiplicities)
                        .filter_map(|(&l, &m)| (l == Some(e)).then_some(m))
                        .sum()
                })
                .collect();
            if !counts.contains(&signature) {
                counts.push(signature);
                out.candidates.push(candidate);
            }
            if out.candidates.len() == maximum {
                break;
            }
        }
        if let (Some(work), Some(columns)) = (workspace.as_ref(), columns.as_ref()) {
            out.profile_builds = work.builds;
            out.profile_reuses = work.reuses;
            out.score_only_requests = work.score_only;
            out.workspace_bytes = work.bytes()
                + columns.bytes()
                + rest_buffer.len() * std::mem::size_of::<Complex<f64>>();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expired_deadline_returns_without_beam_work() {
        let cell = phasesmith_crystallography::UnitCell {
            a_angstrom: 6.,
            b_angstrom: 7.,
            c_angstrom: 8.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let group = phasesmith_crystallography::SpaceGroup::new(vec![
            phasesmith_crystallography::SymmetryOperation::identity(),
        ])
        .unwrap();
        let plan = PreparedPeakCompletion::new(
            cell,
            &group,
            &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.3, 0.1, 0.7]],
            &["C".into(), "N".into()],
            &[[0, 0, -1], [0, 0, 1], [0, 1, 0]],
            &[1., 1., 2.],
            8,
        )
        .unwrap()
        .with_site_displacements(&[0.02; 3])
        .unwrap();
        let result = plan
            .joint_neighbors(
                &[Some(0), Some(1), Some(0)],
                &[[-0.019, 0.18]; 3],
                2,
                Some(Instant::now()),
                None,
            )
            .unwrap();
        assert_eq!(result.evaluations, 0);
        assert_eq!(result.depths, 0);
        assert!(result.candidates.is_empty());
    }
}
