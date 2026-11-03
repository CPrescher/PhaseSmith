//! Conservative full-cell, element-preserving model agreement without reference data.
use crate::solution::math3::{dot, mv, periodic_difference, sub};
use crate::solution::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{UnitCell, xray_species_metadata};
use std::{collections::BTreeMap, time::Instant};

/// Compare complete models under one translation and optional inversion.
/// Returns the maximum matched distance for the first bijection within 0.02 A.
/// This is geometric agreement, not independent evidence of structural correctness.
/// Greedy matching may conservatively reject overlapping/ambiguous input clouds.
/// # Errors
/// Rejects malformed, nonfinite, unknown-element or over-128-atom inputs and stops.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn model_agreement_distance(
    cell: UnitCell,
    a_species: &[String],
    a: &[[f64; 3]],
    b_species: &[String],
    b: &[[f64; 3]],
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<Option<f64>, SolutionError> {
    if a.is_empty()
        || b.is_empty()
        || a.len() > 128
        || b.len() > 128
        || a_species.len() != a.len()
        || b_species.len() != b.len()
        || a.iter().chain(b).flatten().any(|v| !v.is_finite())
        || a_species
            .iter()
            .chain(b_species)
            .any(|s| xray_species_metadata(s).is_none())
    {
        return Err(error("invalid model agreement arrays"));
    }
    let g = cell.geometry().map_err(|e| error(e.to_string()))?;
    if crate::solution::stopped(deadline, cancellation) {
        return Err(error("model agreement stopped"));
    }
    let counts = |species: &[String]| {
        let mut counts = BTreeMap::new();
        for s in species {
            *counts.entry(s.clone()).or_insert(0usize) += 1;
        }
        counts
    };
    let ca = counts(a_species);
    if ca != counts(b_species) {
        return Ok(None);
    }
    let anchor_species = ca
        .iter()
        .min_by_key(|(s, n)| (**n, *s))
        .ok_or_else(|| error("empty model agreement"))?
        .0;
    let anchor = a_species
        .iter()
        .position(|s| s == anchor_species)
        .ok_or_else(|| error("missing anchor element"))?;
    for hand in [1., -1.] {
        for (j, target) in b
            .iter()
            .enumerate()
            .filter(|(j, _)| b_species[*j] == *anchor_species)
        {
            if crate::solution::stopped(deadline, cancellation) {
                return Err(error("model agreement stopped"));
            }
            let shift: [f64; 3] = std::array::from_fn(|axis| target[axis] - hand * a[anchor][axis]);
            let mut used = vec![false; b.len()];
            let mut maximum = 0_f64;
            let mut matched = true;
            // Match the chosen anchor exactly, then the remaining source atoms.
            used[j] = true;
            for i in (0..a.len()).filter(|&i| i != anchor) {
                if crate::solution::stopped(deadline, cancellation) {
                    return Err(error("model agreement stopped"));
                }
                let point = std::array::from_fn(|axis| hand * a[i][axis] + shift[axis]);
                let mut nearest = None;
                let mut best = f64::INFINITY;
                for k in (0..b.len()).filter(|&k| !used[k] && a_species[i] == b_species[k]) {
                    let delta = periodic_difference(&g, sub(point, b[k]))?;
                    let d2 = dot(delta, mv(g.direct_metric, delta));
                    if d2 < best {
                        best = d2;
                        nearest = Some(k);
                    }
                }
                if best > 0.02_f64.powi(2) || !best.is_finite() {
                    matched = false;
                    break;
                }
                if let Some(k) = nearest {
                    used[k] = true;
                    maximum = maximum.max(best);
                } else {
                    matched = false;
                    break;
                }
            }
            if matched {
                return Ok(Some(maximum.sqrt()));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cell() -> UnitCell {
        UnitCell {
            a_angstrom: 6.,
            b_angstrom: 7.,
            c_angstrom: 8.,
            alpha_deg: 90.,
            beta_deg: 100.,
            gamma_deg: 90.,
        }
    }
    #[test]
    fn matching_is_periodic_species_preserving_and_honors_stops() {
        let species = ["Si", "O", "O"].map(str::to_owned);
        let a: [[f64; 3]; 3] = [[0.1, 0.2, 0.3], [0.4, 0.6, 0.2], [0.7, 0.1, 0.8]];
        let order = [2, 0, 1];
        for hand in [1., -1.] {
            let b: Vec<[f64; 3]> = order
                .iter()
                .map(|&i| {
                    std::array::from_fn(|j| (hand * a[i][j] + [0.37, 0.11, 0.29][j]).rem_euclid(1.))
                })
                .collect();
            let labels: Vec<_> = order.iter().map(|&i| species[i].clone()).collect();
            assert!(
                model_agreement_distance(cell(), &species, &a, &labels, &b, None, None)
                    .unwrap()
                    .unwrap()
                    < 1e-12
            );
            let mut changed = b.clone();
            changed[0][0] += 0.01;
            assert!(
                model_agreement_distance(cell(), &species, &a, &labels, &changed, None, None)
                    .unwrap()
                    .is_none()
            );
        }
        assert!(
            model_agreement_distance(
                cell(),
                &species,
                &a,
                &species,
                &a,
                Some(Instant::now()),
                None
            )
            .is_err()
        );
        let token = Cancellation::default();
        let _ = token.cancel();
        assert!(
            model_agreement_distance(cell(), &species, &a, &species, &a, None, Some(&token))
                .is_err()
        );
        assert!(model_agreement_distance(cell(), &[], &[], &species, &a, None, None).is_err());
    }
}
