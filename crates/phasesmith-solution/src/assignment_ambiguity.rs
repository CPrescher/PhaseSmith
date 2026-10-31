//! Bounded coupled relabels with displacement-compensated proposal seeds.
use super::PreparedPeakCompletion;
use crate::{Cancellation, SolutionError, error};
use rustfft::num_complex::Complex;
use std::time::Instant;

pub(crate) fn replacement_supported(
    score: f64,
    r1: f64,
    parent_score: f64,
    parent_r1: f64,
) -> bool {
    [score, r1, parent_score, parent_r1]
        .iter()
        .all(|v| v.is_finite())
        && score <= parent_score + 1e-7
        && r1 <= parent_r1 + 1e-10
}

/// A complete assignment hypothesis, not an accepted chemical identification.
#[derive(Clone, Debug)]
pub struct AssignmentAlternative {
    /// Label per original orbit.
    pub labels: Vec<Option<usize>>,
    /// Independent isotropic displacement seeds in square angstrom.
    pub site_u: Vec<f64>,
    /// Profiled amplitude SSE at the proposed fixed geometry.
    pub objective: f64,
    /// Number of simultaneously changed orbit labels.
    pub edits: usize,
}

/// Actual cheap scores and retained alternatives.
#[derive(Clone, Debug, Default)]
pub struct AssignmentAlternatives {
    /// Deterministically selected proposals, with count-vector diversity.
    pub candidates: Vec<AssignmentAlternative>,
    /// Includes rejected fixed-geometry scores.
    pub evaluations: usize,
}
struct Single {
    site: usize,
    label: usize,
    u: f64,
    objective: f64,
    delta: Vec<Complex<f64>>,
}
fn ordering(a: &AssignmentAlternative, b: &AssignmentAlternative) -> std::cmp::Ordering {
    a.objective
        .total_cmp(&b.objective)
        .then(a.labels.cmp(&b.labels))
}
impl PreparedPeakCompletion {
    /// Propose one/two coordinated near-element relabels without fixing stoichiometry.
    /// The plan must contain bare orbit phases; `site_u` supplies attenuation once.
    /// # Errors
    /// Rejects incomplete labels, invalid displacements or a maximum outside `[1,8]`.
    /// # Panics
    /// Only if public plan metadata has been mutated inconsistently with its cached arrays.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    pub fn ambiguity_neighbors(
        &self,
        labels: &[Option<usize>],
        site_u: &[f64],
        maximum: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<AssignmentAlternatives, SolutionError> {
        self.evaluate(labels, 0)?;
        if self.displacement_applied
            || labels.iter().any(Option::is_none)
            || !self.covered(labels)
            || site_u.len() != self.points.len()
            || site_u
                .iter()
                .any(|u| !u.is_finite() || !(1e-5..=0.2).contains(u))
            || !(1..=8).contains(&maximum)
        {
            return Err(error(
                "ambiguity search needs complete labels, bounded U and maximum in [1,8]",
            ));
        }
        let beta = 8. * std::f64::consts::PI.powi(2);
        let z: Vec<_> = self
            .elements
            .iter()
            .map(|e| {
                phasesmith_crystallography::xray_species_metadata(e)
                    .expect("validated element")
                    .atomic_number
            })
            .collect();
        if !z
            .iter()
            .enumerate()
            .any(|(i, a)| z[..i].iter().any(|b| a.abs_diff(*b) <= 2))
        {
            return Ok(AssignmentAlternatives::default());
        }
        let width = self.elements.len();
        let column = |site: usize, label: usize, u: f64| -> Vec<_> {
            self.s_squared
                .iter()
                .enumerate()
                .map(|(h, &s2)| {
                    self.phase[site][h] * self.factors[h * width + label] * (-beta * u * s2).exp()
                })
                .collect()
        };
        let columns: Vec<_> = labels
            .iter()
            .zip(site_u)
            .enumerate()
            .map(|(i, (&e, &u))| column(i, e.expect("complete labels"), u))
            .collect();
        let total: Vec<_> = (0..self.observed.len())
            .map(|h| columns.iter().map(|p| p[h]).sum::<Complex<f64>>())
            .collect();
        let parent = self.fixed_objective(&total);
        if !parent.is_finite() {
            return Err(error("degenerate ambiguity parent"));
        }
        let ceiling = 1.25 * parent + 1e-4;
        let norm: f64 = self.observed.iter().map(|v| v * v).sum();
        let s2 = self
            .s_squared
            .iter()
            .zip(&self.observed)
            .map(|(&s, &o)| s * o * o)
            .sum::<f64>()
            / norm;
        let anchor = self
            .s_squared
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - s2).abs().total_cmp(&(*b - s2).abs()))
            .expect("nonempty observations")
            .0;
        let mut out = AssignmentAlternatives::default();
        let mut singles = Vec::<Single>::new();
        let mut retained = Vec::<AssignmentAlternative>::new();
        let mut retain = |labels: Vec<Option<usize>>, site_u: Vec<f64>, objective: f64, edits| {
            if objective.is_finite() && objective <= ceiling && self.covered(&labels) {
                if let Some(index) = retained.iter().position(|p| p.labels == labels) {
                    if retained[index].objective <= objective {
                        return;
                    }
                    retained.remove(index);
                }
                retained.push(AssignmentAlternative {
                    labels,
                    site_u,
                    objective,
                    edits,
                });
                retained.sort_by(ordering);
                retained.truncate(32);
            }
        };
        for i in 0..labels.len() {
            if crate::stopped(deadline, cancellation) {
                break;
            }
            let old = labels[i].expect("complete labels");
            for e in 0..width {
                if e == old || z[e].abs_diff(z[old]) > 2 {
                    continue;
                }
                let ratio = self.factors[anchor * width + e] / self.factors[anchor * width + old];
                let shifted = if ratio.is_finite() && ratio > 0. && s2 > 0. {
                    (site_u[i] + (ratio.ln() / (beta * s2)).clamp(-0.02, 0.02)).clamp(1e-5, 0.2)
                } else {
                    site_u[i]
                };
                for (variant, u) in [site_u[i], shifted].into_iter().enumerate() {
                    if variant == 1 && u.to_bits() == site_u[i].to_bits() {
                        continue;
                    }
                    let delta: Vec<_> = column(i, e, u)
                        .iter()
                        .zip(&columns[i])
                        .map(|(a, b)| a - b)
                        .collect();
                    let next_total: Vec<_> = total.iter().zip(&delta).map(|(a, b)| a + b).collect();
                    let objective = self.fixed_objective(&next_total);
                    out.evaluations += 1;
                    let mut next = labels.to_vec();
                    next[i] = Some(e);
                    let mut displacement = site_u.to_vec();
                    displacement[i] = u;
                    retain(next, displacement, objective, 1);
                    singles.push(Single {
                        site: i,
                        label: e,
                        u,
                        objective,
                        delta,
                    });
                    singles.sort_by(|a, b| {
                        a.objective
                            .total_cmp(&b.objective)
                            .then(a.site.cmp(&b.site))
                            .then(a.label.cmp(&b.label))
                            .then(a.u.total_cmp(&b.u))
                    });
                    singles.truncate(12);
                }
            }
        }
        for (slot, a) in singles.iter().enumerate() {
            if crate::stopped(deadline, cancellation) {
                break;
            }
            for b in &singles[..slot] {
                if a.site == b.site {
                    continue;
                }
                let next_total: Vec<_> = total
                    .iter()
                    .zip(&a.delta)
                    .zip(&b.delta)
                    .map(|((v, x), y)| v + x + y)
                    .collect();
                let objective = self.fixed_objective(&next_total);
                out.evaluations += 1;
                let mut next = labels.to_vec();
                next[a.site] = Some(a.label);
                next[b.site] = Some(b.label);
                let mut displacement = site_u.to_vec();
                displacement[a.site] = a.u;
                displacement[b.site] = b.u;
                retain(next, displacement, objective, 2);
            }
        }
        let counts = |p: &AssignmentAlternative| -> Vec<usize> {
            (0..width)
                .map(|e| {
                    p.labels
                        .iter()
                        .zip(&self.multiplicities)
                        .filter_map(|(&label, &m)| (label == Some(e)).then_some(m))
                        .sum()
                })
                .collect()
        };
        let mut signatures = Vec::new();
        for p in &retained {
            let signature = counts(p);
            if !signatures.contains(&signature) && out.candidates.len() < maximum {
                signatures.push(signature);
                out.candidates.push(p.clone());
            }
        }
        for p in retained {
            if out.candidates.len() == maximum {
                break;
            }
            if !out.candidates.iter().any(|q| q.labels == p.labels) {
                out.candidates.push(p);
            }
        }
        out.candidates.sort_by(ordering);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::replacement_supported;
    #[test]
    fn squared_error_improvement_cannot_hide_a_worse_absolute_residual() {
        let parent = [0.12_f64, 0.];
        let challenger = [0.08_f64, 0.08];
        let square = |r: [f64; 2]| r.iter().map(|v| v * v).sum::<f64>();
        let absolute = |r: [f64; 2]| r.iter().map(|v| v.abs()).sum::<f64>();
        assert!(square(challenger) < square(parent));
        assert!(!replacement_supported(
            square(challenger),
            absolute(challenger),
            square(parent),
            absolute(parent)
        ));
        assert!(replacement_supported(
            square([0.06, 0.]),
            absolute([0.06, 0.]),
            square(parent),
            absolute(parent)
        ));
        assert!(!replacement_supported(f64::NAN, 0., 1., 1.));
    }
}
