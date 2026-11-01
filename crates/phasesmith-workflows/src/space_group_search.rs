//! Measured-only space-group evidence; no absent measurements are synthesized.
//! Equations and limitations: `docs/space-group-search.md`.
#![allow(clippy::cast_precision_loss)]
use crate::single_crystal::{Error, Result};
use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation, UnitCell};
use phasesmith_solution::Observation;
use std::collections::{BTreeMap, BTreeSet};
/// Integer direct-space rotation.
pub type Rotation = [[i32; 3]; 3];
/// One conventional setting supplied by a database adapter, not by a reference model.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    /// International Tables number; used only for identification.
    pub number: i32,
    /// Human-readable setting name.
    pub symbol: String,
    /// Unambiguous Hall identifier in the supplied index basis.
    pub hall: String,
    /// Exact conventional operations.
    pub group: SpaceGroup,
}
/// Explicit preliminary measurement-model and evidence controls.
#[derive(Clone, Debug)]
pub struct AssessmentOptions {
    /// Relative pre-empirical discrepancy allowance.
    pub relative_error: f64,
    /// Largest admitted reduced intensity-symmetry chi square.
    pub maximum_chi_square: f64,
    /// Minimum measured independent intensity constraints for nontrivial symmetry.
    pub minimum_constraints: usize,
    /// Equal-population resolution bins for the allowed-intensity prescreen.
    pub shells: usize,
    /// Per-family clipped log-likelihood evidence interval.
    pub evidence_clip: [f64; 2],
}
impl Default for AssessmentOptions {
    fn default() -> Self {
        Self {
            relative_error: 0.25,
            maximum_chi_square: 4.,
            minimum_constraints: 20,
            shells: 8,
            evidence_clip: [-25., 5.],
        }
    }
}
/// Measured intensity-symmetry evidence including Friedel symmetry.
#[derive(Clone, Debug)]
pub struct LaueEvidence {
    /// Exact direct-space rotational set, including inversion.
    pub rotations: Vec<Rotation>,
    /// Independent measured equivalence classes.
    pub families: usize,
    /// Measured constraints after Friedel merging; missing rows supply none.
    pub constraints: usize,
    /// Reduced weighted discrepancy, absent when there is no redundancy.
    pub reduced_chi_square: Option<f64>,
    /// Unweighted within-family absolute intensity discrepancy.
    pub r_int: Option<f64>,
    /// `supported`, `insufficient_data`, `contradicted`, or `friedel_only`.
    pub status: &'static str,
}
/// One candidate's prescreen, in exactly the corresponding catalogue order.
#[derive(Clone, Debug)]
pub struct Evidence {
    /// Index into `Assessment.laue`.
    pub laue: usize,
    /// Number of independent measured Friedel families forbidden by this setting.
    pub absent_families: usize,
    /// Forbidden measured means with I > 5 sigma; not an automatic deletion rule.
    pub strong_violations: usize,
    /// Sum of clipped measured-family log BF against the common shell model.
    pub absence_log_evidence: f64,
}
/// Complete prescreen; the chosen processing Laue set remains provisional.
#[derive(Clone, Debug)]
pub struct Assessment {
    /// Original supplied measured rows, in source order, for independent replay/audit.
    pub observations: Vec<Observation>,
    /// Number of supplied measured rows.
    pub rows: usize,
    /// Independent measured Friedel families.
    pub friedel_families: usize,
    /// Distinct compatible intensity-symmetry hypotheses.
    pub laue: Vec<LaueEvidence>,
    /// Largest supported rotational set; Friedel-only when higher symmetry lacks evidence.
    pub processing_laue: usize,
    /// Candidate evidence in input catalogue order.
    pub evidence: Vec<Evidence>,
    /// Stable candidate order: contradicted intensity groups last, then descending absence evidence.
    pub ranking: Vec<usize>,
}
/// Construct an exact symmorphic group from a closed rotational set.
/// # Errors
/// Nonclosed, duplicate, nonunimodular or excessive operations.
pub fn rotational_group(rotations: &[Rotation]) -> Result<SpaceGroup> {
    SpaceGroup::new(
        rotations
            .iter()
            .map(|&r| SymmetryOperation::new(r, [Rational::zero(); 3]))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error(e.to_string()))?,
    )
    .map_err(|e| Error(e.to_string()))
}
/// Non-resonant Laue rotational closure of a space group.
#[must_use]
pub fn laue_rotations(group: &SpaceGroup) -> Vec<Rotation> {
    group
        .rotations()
        .iter()
        .flat_map(|&r| [r, r.map(|row| row.map(|x| -x))])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
fn canonical(h: [i32; 3], rotations: &[Rotation]) -> [i32; 3] {
    rotations
        .iter()
        .map(|r| std::array::from_fn(|j| (0..3).map(|i| r[i][j] * h[i]).sum()))
        .min()
        .expect("nonempty rotational set")
}
fn validate(options: &AssessmentOptions, candidates: &[Hypothesis]) -> Result<()> {
    if candidates.is_empty()
        || candidates.len() > 530
        || options.shells == 0
        || options.shells > 64
        || options.minimum_constraints == 0
        || !options.relative_error.is_finite()
        || !(0.0..=1.0).contains(&options.relative_error)
        || !options.maximum_chi_square.is_finite()
        || options.maximum_chi_square <= 0.
        || options.evidence_clip.iter().any(|x| !x.is_finite())
        || options.evidence_clip[0] >= 0.
        || options.evidence_clip[1] <= 0.
        || candidates
            .iter()
            .any(|c| !(1..=230).contains(&c.number) || c.hall.is_empty() || c.symbol.is_empty())
        || candidates
            .iter()
            .map(|c| &c.hall)
            .collect::<BTreeSet<_>>()
            .len()
            != candidates.len()
    {
        return Err(Error(
            "invalid bounded symmetry assessment controls/catalogue".into(),
        ));
    }
    Ok(())
}
/// Assess signed, genuinely measured rows with no exact space group supplied.
/// Every catalogue setting must preserve the supplied cell metric. Inputs are never mutated.
/// # Errors
/// Invalid observations/catalogue/metric, numerical overflow or exceeded input budgets.
pub fn assess(
    observations: &[Observation],
    cell: UnitCell,
    candidates: &[Hypothesis],
    options: &AssessmentOptions,
) -> Result<Assessment> {
    validate(options, candidates)?;
    let rows = merge_rows(observations)?;
    let geometry = cell.geometry().map_err(|e| Error(e.to_string()))?;
    let sets = validate_catalogue(cell, candidates, options)?;
    let mut laue = Vec::new();
    for rotations in sets {
        laue.push(laue_evidence(&rows, rotations, options)?);
    }
    let processing_laue = (0..laue.len())
        .filter(|&i| matches!(laue[i].status, "supported" | "friedel_only"))
        .max_by(|&a, &b| {
            laue[a]
                .rotations
                .len()
                .cmp(&laue[b].rotations.len())
                .then_with(|| {
                    laue[b]
                        .reduced_chi_square
                        .unwrap_or(0.)
                        .total_cmp(&laue[a].reduced_chi_square.unwrap_or(0.))
                })
                .then_with(|| b.cmp(&a))
        })
        .ok_or_else(|| Error("Friedel baseline missing".into()))?;
    let (bins, means) = shell_model(&rows, &geometry, options.shells)?;
    let mut evidence = Vec::new();
    for c in candidates {
        let rotations = laue_rotations(&c.group);
        let li = laue
            .iter()
            .position(|l| l.rotations == rotations)
            .ok_or_else(|| Error("catalogued Laue set missing".into()))?;
        let mut e = Evidence {
            laue: li,
            absent_families: 0,
            strong_violations: 0,
            absence_log_evidence: 0.,
        };
        for (i, r) in rows.iter().enumerate() {
            if !c
                .group
                .is_systematically_absent(r.hkl)
                .map_err(|e| Error(e.to_string()))?
            {
                continue;
            }
            e.absent_families += 1;
            e.strong_violations += usize::from(r.intensity > 5. * r.sigma);
            let z = r.intensity / r.sigma;
            let mu = means[bins[i]] / r.sigma;
            // Ratios keep finite observation units from entering an avoidable square overflow.
            let ratio = 1. + mu * mu;
            let raw = 0.5 * (ratio.ln() + ((z - mu) / ratio.sqrt()).powi(2) - z * z);
            if raw.is_nan() {
                return Err(Error("absence evidence overflow".into()));
            }
            e.absence_log_evidence += raw.clamp(options.evidence_clip[0], options.evidence_clip[1]);
        }
        evidence.push(e);
    }
    let mut ranking: Vec<_> = (0..candidates.len()).collect();
    ranking.sort_by(|&a, &b| {
        let rejected = |i: usize| laue[evidence[i].laue].status == "contradicted";
        rejected(a)
            .cmp(&rejected(b))
            .then_with(|| {
                evidence[b]
                    .absence_log_evidence
                    .total_cmp(&evidence[a].absence_log_evidence)
            })
            .then_with(|| candidates[a].hall.cmp(&candidates[b].hall))
    });
    Ok(Assessment {
        observations: observations.to_vec(),
        rows: observations.len(),
        friedel_families: rows.len(),
        laue,
        processing_laue,
        evidence,
        ranking,
    })
}
fn laue_evidence(
    rows: &[Observation],
    rotations: Vec<Rotation>,
    o: &AssessmentOptions,
) -> Result<LaueEvidence> {
    let mut families: BTreeMap<_, Vec<&Observation>> = BTreeMap::new();
    for row in rows {
        families
            .entry(canonical(row.hkl, &rotations))
            .or_default()
            .push(row);
    }
    let mut chi = 0.;
    let mut constraints = 0;
    let mut residual = 0.;
    let mut signal = 0.;
    for members in families.values().filter(|m| m.len() > 1) {
        let min = members
            .iter()
            .map(|r| r.sigma.hypot(o.relative_error * r.intensity))
            .fold(f64::INFINITY, f64::min);
        let weights: Vec<_> = members
            .iter()
            .map(|r| (min / r.sigma.hypot(o.relative_error * r.intensity)).powi(2))
            .collect();
        let total: f64 = weights.iter().sum();
        let mean = members
            .iter()
            .zip(&weights)
            .map(|(r, &w)| r.intensity * (w / total))
            .sum::<f64>();
        for (r, &w) in members.iter().zip(&weights) {
            chi += ((r.intensity - mean) / min).powi(2) * w;
            residual += (r.intensity - mean).abs();
            signal += r.intensity.abs();
        }
        constraints += members.len() - 1;
    }
    if !chi.is_finite() || !residual.is_finite() || !signal.is_finite() {
        return Err(Error("Laue evidence overflow".into()));
    }
    let reduced_chi_square = (constraints > 0).then(|| chi / constraints as f64);
    let status = if rotations.len() == 2 {
        "friedel_only"
    } else if constraints < o.minimum_constraints {
        "insufficient_data"
    } else if reduced_chi_square.is_some_and(|v| v > o.maximum_chi_square) {
        "contradicted"
    } else {
        "supported"
    };
    Ok(LaueEvidence {
        rotations,
        families: families.len(),
        constraints,
        reduced_chi_square,
        r_int: (signal > 0.).then(|| residual / signal),
        status,
    })
}

fn merge_rows(observations: &[Observation]) -> Result<Vec<Observation>> {
    if observations.is_empty()
        || observations.len() > 100_000
        || observations.iter().any(|r| {
            r.hkl == [0; 3]
                || r.hkl.iter().any(|h| h.unsigned_abs() > 1024)
                || !r.intensity.is_finite()
                || r.intensity.abs() > 1e100
                || !r.sigma.is_finite()
                || r.sigma <= 0.
        })
    {
        return Err(Error("invalid measured symmetry observations".into()));
    }
    let mut measured: BTreeMap<[i32; 3], Vec<&Observation>> = BTreeMap::new();
    for r in observations {
        measured
            .entry(r.hkl.min(r.hkl.map(|h| -h)))
            .or_default()
            .push(r);
    }
    let mut rows = Vec::new();
    for (h, members) in measured {
        let minimum = members
            .iter()
            .map(|r| r.sigma)
            .fold(f64::INFINITY, f64::min);
        let weights: Vec<_> = members
            .iter()
            .map(|r| (minimum / r.sigma).powi(2))
            .collect();
        let total: f64 = weights.iter().sum();
        let intensity = members
            .iter()
            .zip(&weights)
            .map(|(r, w)| r.intensity * (w / total))
            .sum::<f64>();
        let sigma = minimum / total.sqrt();
        if !intensity.is_finite() || sigma <= 0. {
            return Err(Error("Friedel mean exceeds finite range".into()));
        }
        rows.push(Observation {
            hkl: h,
            intensity,
            sigma,
        });
    }
    Ok(rows)
}
/// Validate a complete conventional catalogue before loading original detector frames.
/// Returns distinct metric-compatible Laue sets, including the Friedel baseline.
/// # Errors
/// Invalid controls, duplicate identifiers, unsupported rotations or incompatible metric.
pub fn validate_catalogue(
    cell: UnitCell,
    candidates: &[Hypothesis],
    options: &AssessmentOptions,
) -> Result<Vec<Vec<Rotation>>> {
    validate(options, candidates)?;
    let geometry = cell.geometry().map_err(|e| Error(e.to_string()))?;
    let identity =
        SpaceGroup::new(vec![SymmetryOperation::identity()]).map_err(|e| Error(e.to_string()))?;
    let mut sets = BTreeSet::new();
    sets.insert(laue_rotations(&identity));
    for c in candidates {
        if c.group
            .rotations()
            .iter()
            .flatten()
            .flatten()
            .any(|r| r.unsigned_abs() > 8)
        {
            return Err(Error("conventional setting rotation bound exceeded".into()));
        }
        let rotations = laue_rotations(&c.group);
        if sets.insert(rotations.clone()) {
            let group = rotational_group(&rotations)?;
            let g = geometry.direct_metric;
            let components = [g[0][0], g[1][1], g[2][2], g[1][2], g[0][2], g[0][1]];
            let scale = components.iter().fold(1f64, |a, b| a.max(b.abs()));
            for eq in &group.metric_constraints().equations {
                let error = eq
                    .iter()
                    .zip(components)
                    .map(|(&a, b)| a as f64 * b)
                    .sum::<f64>();
                let coefficients = eq.iter().map(|x| x.unsigned_abs() as f64).sum::<f64>();
                if error.abs() > 1e-10 * scale * coefficients.max(1.) {
                    return Err(Error(
                        "catalogue setting incompatible with supplied metric".into(),
                    ));
                }
            }
        }
    }
    Ok(sets.into_iter().collect())
}
fn shell_model(
    rows: &[Observation],
    geometry: &phasesmith_crystallography::CellGeometry,
    shell_count: usize,
) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut ordered: Vec<_> = (0..rows.len()).collect();
    let resolutions: Vec<_> = rows
        .iter()
        .map(|r| {
            Ok({
                geometry
                    .d_spacing_and_derivatives(r.hkl)
                    .map_err(|e| Error(e.to_string()))?
                    .0
                    .recip()
                    .powi(2)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let maximum = resolutions.iter().copied().fold(0f64, f64::max);
    ordered.sort_by(|&a, &b| {
        (resolutions[a] / maximum * 1e12)
            .round_ties_even()
            .total_cmp(&(resolutions[b] / maximum * 1e12).round_ties_even())
            .then(a.cmp(&b))
    });
    let shells = shell_count.min(rows.len());
    let mut bins = vec![0; rows.len()];
    let mut means = vec![0.; shells];
    let mut noise = vec![0.; shells];
    let mut counts = vec![0usize; shells];
    for (rank, &i) in ordered.iter().enumerate() {
        let bin = rank * shells / rows.len();
        bins[i] = bin;
        means[bin] += rows[i].intensity;
        noise[bin] += rows[i].sigma;
        counts[bin] += 1;
    }
    for j in 0..shells {
        means[j] = (means[j] / counts[j] as f64)
            .max(noise[j] / counts[j] as f64)
            .max(f64::MIN_POSITIVE);
    }
    Ok((bins, means))
}
