//! Motif scale proposals and fixed-coordinate coupled label improvements.
use super::{PeakCompletionCandidate, PeakCompletionResult, PreparedPeakCompletion};
use crate::solution::math3::{dot, mv, periodic_difference, sub};
use crate::solution::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{SpaceGroup, UnitCell};
use std::time::Instant;

/// Relative-density scale inferred from a bounded geometric motif, never an observation.
#[derive(Clone, Debug)]
pub struct ChemicalScale {
    /// `boron_cage` or `oxygen_coordination`.
    pub kind: &'static str,
    /// Density integral per neutral atomic number.
    pub scale: f64,
    /// Number of supporting ligand images.
    pub support: usize,
}
/// Propose additional scales from similar-mass coordination shells.
/// # Errors
/// Rejects malformed peaks, masses, species or cells. Distances are exact periodic.
#[allow(clippy::too_many_lines)]
pub fn chemical_density_scales(
    cell: UnitCell,
    group: &SpaceGroup,
    points: &[[f64; 3]],
    masses: &[f64],
    elements: &[String],
) -> Result<Vec<ChemicalScale>, SolutionError> {
    if points.len() != masses.len()
        || points.is_empty()
        || points.len() > 128
        || points.iter().flatten().any(|v| !v.is_finite())
        || masses.iter().any(|v| !v.is_finite() || *v <= 0.)
        || elements.is_empty()
        || elements.len() > 8
        || elements
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != elements.len()
        || elements.iter().any(|e| {
            e.len() > 2
                || phasesmith_crystallography::xray_species_metadata(e)
                    .is_none_or(|v| v.atomic_number <= 1)
        })
    {
        return Err(error("invalid chemical density proposals"));
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let mut images = Vec::new();
    for (&p, &m) in points.iter().zip(masses) {
        for q in group
            .expand_sites(&[p], 1e-8)
            .map_err(|e| error(e.to_string()))?
            .fractional_xyz
        {
            images.push((q, m));
            if images.len() > 512 {
                return Err(error("chemical motif image budget exceeded"));
            }
        }
    }
    let oxygen = elements.iter().any(|e| e == "O")
        && elements.iter().any(|e| {
            phasesmith_crystallography::xray_species_metadata(e)
                .is_some_and(|v| v.atomic_number > 8)
        });
    let boron = elements.iter().any(|e| e == "B");
    let mut out = Vec::new();
    for (center, central_mass) in &images {
        let mut neighbors = Vec::new();
        for (point, mass) in &images {
            let delta = periodic_difference(&geometry, sub(*point, *center))?;
            let d = dot(delta, mv(geometry.direct_metric, delta)).sqrt();
            if (0.9..=2.3).contains(&d) {
                neighbors.push((d, *mass));
            }
        }
        for &(distance, mass) in &neighbors {
            if oxygen && *central_mass > mass * 1.1 {
                let mut shell: Vec<_> = neighbors
                    .iter()
                    .filter(|(d, m)| {
                        (d / distance - 1.).abs() <= 0.12 && (m / mass - 1.).abs() <= 0.20
                    })
                    .map(|(_, m)| *m)
                    .collect();
                if shell.len() >= 3 {
                    shell.sort_by(f64::total_cmp);
                    out.push(ChemicalScale {
                        kind: "oxygen_coordination",
                        scale: shell[shell.len() / 2] / 8.,
                        support: shell.len(),
                    });
                }
            }
            if boron && (1.65..=1.8).contains(&distance) && (mass / central_mass - 1.).abs() < 0.2 {
                let shell: Vec<_> = neighbors
                    .iter()
                    .filter(|(d, m)| {
                        (1.65..=1.8).contains(d) && (m / central_mass - 1.).abs() < 0.2
                    })
                    .collect();
                if shell.len() >= 3 {
                    out.push(ChemicalScale {
                        kind: "boron_cage",
                        scale: *central_mass / 5.,
                        support: shell.len(),
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| {
        b.support
            .cmp(&a.support)
            .then(a.kind.cmp(b.kind))
            .then(a.scale.total_cmp(&b.scale))
    });
    let mut distinct: Vec<ChemicalScale> = Vec::new();
    for proposal in out {
        if distinct
            .iter()
            .all(|p| (p.scale / proposal.scale - 1.).abs() > 0.1)
        {
            distinct.push(proposal);
        }
        if distinct.len() == 2 {
            break;
        }
    }
    Ok(distinct)
}

impl PreparedPeakCompletion {
    /// Apply fixed per-orbit Uiso to cached phases and jets. Global U remains additive.
    /// # Errors
    /// Rejects malformed, nonfinite or negative displacements.
    pub fn with_site_displacements(mut self, site_u: &[f64]) -> Result<Self, SolutionError> {
        if site_u.len() != self.points.len()
            || site_u
                .iter()
                .any(|v| !v.is_finite() || !(0. ..=0.5).contains(v))
        {
            return Err(error("invalid fixed completion displacements"));
        }
        self.displacement_applied = true;
        for (i, &u) in site_u.iter().enumerate() {
            for (h, &s2) in self.s_squared.iter().enumerate() {
                let factor = (-8. * std::f64::consts::PI.powi(2) * u * s2).exp();
                self.phase[i][h] *= factor;
                for jet in &mut self.jets[i][h] {
                    *jet *= factor;
                }
            }
        }
        Ok(self)
    }
    pub(super) fn fixed_objective(&self, total: &[rustfft::num_complex::Complex<f64>]) -> f64 {
        let square: f64 = total
            .iter()
            .map(rustfft::num_complex::Complex::norm_sqr)
            .sum();
        if square <= 1e-250 {
            return f64::INFINITY;
        }
        let scale = total
            .iter()
            .zip(&self.observed)
            .map(|(v, o)| v.norm() * o)
            .sum::<f64>()
            / square;
        total
            .iter()
            .zip(&self.observed)
            .map(|(v, o)| (scale * v.norm() - o).powi(2))
            .sum::<f64>()
            / self.observed.iter().map(|v| v * v).sum::<f64>()
    }
    /// Retain nearby complete single relabels and paired swaps for subsequent relaxation.
    /// Unlike `reassign`, alternatives need not improve at fixed geometry/displacement.
    /// The explicit proposal band is `J <= 1.25*J_parent + 1e-4`; it is not an error bar.
    /// # Errors
    /// Rejects incomplete labels or a maximum outside `[1,8]`.
    #[allow(clippy::too_many_lines)]
    pub fn assignment_neighbors(
        &self,
        labels: &[Option<usize>],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<PeakCompletionResult, SolutionError> {
        self.evaluate(labels, 0)?;
        if labels.iter().any(Option::is_none)
            || !self.covered(labels)
            || !(1..=8).contains(&maximum)
        {
            return Err(error(
                "assignment neighbors require complete labels and maximum in [1,8]",
            ));
        }
        let total = self.sum(labels);
        let ceiling = 1.25 * self.fixed_objective(&total) + 1e-4;
        let mut out = PeakCompletionResult::default();
        let mut priorities = Vec::new();
        let mut retain = |next: Vec<Option<usize>>, value: f64| {
            out.evaluations += 1;
            if value.is_finite() && value <= ceiling && self.covered(&next) {
                out.candidates.push(PeakCompletionCandidate {
                    atoms: self.atoms(&next),
                    score: self.rank(value, &next),
                    labels: next,
                    u_iso: 0.,
                    objective: value,
                });
            }
        };
        for i in 0..labels.len() {
            if crate::solution::stopped(deadline, cancellation) {
                break;
            }
            let mut priority = f64::INFINITY;
            for e in 0..self.elements.len() {
                if labels[i] == Some(e) {
                    continue;
                }
                let value = self.fixed_objective(&self.updated(&total, i, labels[i], Some(e)));
                priority = priority.min(value);
                let mut next = labels.to_vec();
                next[i] = Some(e);
                retain(next, value);
            }
            priorities.push((priority, i));
        }
        priorities.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        priorities.truncate(24);
        for (slot, &(_, i)) in priorities.iter().enumerate() {
            if crate::solution::stopped(deadline, cancellation) {
                break;
            }
            for &(_, j) in &priorities[..slot] {
                if labels[i] == labels[j] {
                    continue;
                }
                let proposal = self.updated(
                    &self.updated(&total, i, labels[i], labels[j]),
                    j,
                    labels[j],
                    labels[i],
                );
                let mut next = labels.to_vec();
                next.swap(i, j);
                retain(next, self.fixed_objective(&proposal));
            }
        }
        out.candidates
            .sort_by(|a, b| a.score.total_cmp(&b.score).then(a.labels.cmp(&b.labels)));
        out.candidates.dedup_by(|a, b| a.labels == b.labels);
        out.candidates.truncate(maximum);
        Ok(out)
    }
    /// Improve full supplied labels using single changes and paired swaps at fixed geometry/U.
    /// Returns only improving alternatives; counts/topology and required types are preserved.
    /// # Errors
    /// Rejects invalid initial labels or an empty/incomplete model.
    #[allow(clippy::too_many_lines)]
    pub fn reassign(
        &self,
        labels: &[Option<usize>],
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<PeakCompletionResult, SolutionError> {
        self.evaluate(labels, 0)?;
        if labels.iter().any(Option::is_none) || !self.covered(labels) {
            return Err(error("reassignment requires complete labels"));
        }
        let mut current = labels.to_vec();
        let mut total = self.sum(&current);
        let mut value = self.fixed_objective(&total);
        let mut out = PeakCompletionResult::default();
        for _ in 0..3 {
            let mut best = None;
            let mut priorities = Vec::new();
            for i in 0..current.len() {
                if crate::solution::stopped(deadline, cancellation) {
                    return Ok(out);
                }
                let mut priority = f64::INFINITY;
                for e in 0..self.elements.len() {
                    if current[i] == Some(e) {
                        continue;
                    }
                    let proposal = self.updated(&total, i, current[i], Some(e));
                    let score = self.fixed_objective(&proposal);
                    out.evaluations += 1;
                    priority = priority.min(score);
                    let mut next = current.clone();
                    next[i] = Some(e);
                    if self.covered(&next)
                        && score < value - 1e-12
                        && best.as_ref().is_none_or(|(v, _, _)| score < *v)
                    {
                        best = Some((score, next, proposal));
                    }
                }
                priorities.push((priority, i));
            }
            priorities.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            priorities.truncate(24);
            for (slot, &(_, i)) in priorities.iter().enumerate() {
                if crate::solution::stopped(deadline, cancellation) {
                    return Ok(out);
                }
                for &(_, j) in &priorities[..slot] {
                    if current[i] == current[j] {
                        continue;
                    }
                    let proposal = self.updated(
                        &self.updated(&total, i, current[i], current[j]),
                        j,
                        current[j],
                        current[i],
                    );
                    let score = self.fixed_objective(&proposal);
                    out.evaluations += 1;
                    if score < value - 1e-12 && best.as_ref().is_none_or(|(v, _, _)| score < *v) {
                        let mut next = current.clone();
                        next.swap(i, j);
                        best = Some((score, next, proposal));
                    }
                }
            }
            let Some((score, next, proposal)) = best else {
                break;
            };
            value = score;
            current = next;
            total = proposal;
            out.changes += 1;
            out.candidates.push(PeakCompletionCandidate {
                labels: current.clone(),
                u_iso: 0.,
                objective: value,
                score: self.rank(value, &current),
                atoms: self.atoms(&current),
            });
        }
        out.candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        Ok(out)
    }
}
