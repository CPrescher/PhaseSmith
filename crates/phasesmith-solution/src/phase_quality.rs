//! Composition-free Wilson/shell normalization and CC/Rweak phase-map selection.
//! Equations, assumptions and boundary conventions: docs/solution-phase-quality.md.
use crate::math3::{dot, mv};
use crate::{Observation, PreparedData, SolutionError, error};
use phasesmith_crystallography::{PreparedXrayScattering, SpaceGroup, UnitCell};
use std::collections::{BTreeMap, BTreeSet};

/// One equal-population Wilson shell, before interpolation.
#[derive(Clone, Debug)]
pub struct WilsonShell {
    /// Mean (sin(theta)/lambda)^2, in inverse squared angstrom.
    pub s_squared: f64,
    /// Independent reflection families in this shell.
    pub families: usize,
    /// Signed mean intensity divided by epsilon and type-average form-factor sum.
    pub reduced_mean: f64,
    /// Shrunk log residual relative to the fitted Wilson line.
    pub log_residual: f64,
}
/// Fitted normalization provenance; no input composition or atom count.
#[derive(Clone, Debug)]
pub struct WilsonAudit {
    /// Number of independent allowed symmetry/Friedel families.
    pub families: usize,
    /// Families with nonpositive signed intensity, retained in shell estimation.
    pub nonpositive_families: usize,
    /// Log fitted scale, including the final global E^2 rescale.
    pub log_scale: f64,
    /// Slope versus s^2; effective envelope B is -slope/2.
    pub slope: f64,
    /// Mean E^2 over independent allowed families after rescaling.
    pub mean_e_squared: f64,
    /// Actual shells and signed means.
    pub shells: Vec<WilsonShell>,
}
/// Normalized amplitudes and expected intensities aligned with merged input rows.
#[derive(Clone, Debug)]
pub struct WilsonNormalization {
    /// Expected positive intensity epsilon * C(s) * fitted envelope.
    pub expected_intensity: Vec<f64>,
    /// Derived sqrt(max(I,0)/expected), leaving original signed I untouched.
    pub e_values: Vec<f64>,
    /// Statistical symmetry factors, including centring.
    pub epsilon: Vec<usize>,
    /// One input-row index per independent family, in canonical HKL order.
    pub representatives: Vec<usize>,
    /// Fitted parameters and diagnostics.
    pub audit: WilsonAudit,
    form_squares: Vec<f64>,
}

fn epsilon(group: &SpaceGroup, h: [i32; 3]) -> Result<usize, SolutionError> {
    let mut count = 0;
    for op in group.operations() {
        count += usize::from(op.reciprocal_index(h).map_err(|e| error(e.to_string()))? == h);
    }
    Ok(count.max(1))
}
fn resolution(cell: UnitCell, hkl: &[[i32; 3]]) -> Result<Vec<f64>, SolutionError> {
    let g = cell.geometry().map_err(|e| error(e.to_string()))?;
    Ok(hkl
        .iter()
        .map(|h| {
            let q = mv(g.reciprocal_basis, h.map(f64::from));
            dot(q, q) / 4.
        })
        .collect())
}
fn form_squares(elements: &[String], x: &[f64]) -> Result<Vec<f64>, SolutionError> {
    if elements.is_empty()
        || elements.len() > 8
        || elements.iter().collect::<BTreeSet<_>>().len() != elements.len()
        || elements.iter().any(|e| {
            phasesmith_crystallography::xray_species_metadata(e)
                .is_none_or(|m| m.atomic_number <= 1)
                || e.len() > 2
                || !e.as_bytes()[0].is_ascii_uppercase()
                || e.as_bytes().get(1).is_some_and(|c| !c.is_ascii_lowercase())
        })
    {
        return Err(error("normalization needs 1..8 element types"));
    }
    let factors = PreparedXrayScattering::new(elements)
        .and_then(|p| p.evaluate(&x.iter().map(|s| s.sqrt()).collect::<Vec<_>>()))
        .map_err(|e| error(e.to_string()))?
        .real;
    Ok(factors
        .chunks_exact(elements.len())
        .map(|f| f.iter().map(|v| v * v).sum::<f64>())
        .collect())
}
fn residual_at(shells: &[WilsonShell], x: f64) -> f64 {
    if x <= shells[0].s_squared {
        return shells[0].log_residual;
    }
    for pair in shells.windows(2) {
        if x <= pair[1].s_squared {
            let span = pair[1].s_squared - pair[0].s_squared;
            let t = if span > 1e-30 {
                (x - pair[0].s_squared) / span
            } else {
                0.
            };
            return (1. - t) * pair[0].log_residual + t * pair[1].log_residual;
        }
    }
    shells.last().expect("nonempty shells").log_residual
}
impl WilsonNormalization {
    fn expected(&self, x: f64, c: f64, eps: usize) -> Result<f64, SolutionError> {
        #[allow(clippy::cast_precision_loss)]
        let value = eps as f64
            * c
            * (self.audit.log_scale + self.audit.slope * x + residual_at(&self.audit.shells, x))
                .exp();
        if !value.is_finite() || value <= 1e-250 {
            return Err(error("nonfinite or vanishing Wilson expectation"));
        }
        Ok(value)
    }
}
/// Fit isotropic Wilson decay plus regularized radial-shell residuals.
/// Uses each symmetry/Friedel family once, signed means, and epsilon factors.
/// # Errors
/// Rejects invalid/unmerged rows, no positive shell means, or singular expectations.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub fn normalize_wilson(
    cell: UnitCell,
    group: &SpaceGroup,
    hkl: &[[i32; 3]],
    intensity: &[f64],
    elements: &[String],
) -> Result<WilsonNormalization, SolutionError> {
    if hkl.is_empty()
        || hkl.len() > 100_000
        || hkl.len() != intensity.len()
        || intensity.iter().any(|v| !v.is_finite() || v.abs() > 1e100)
        || hkl
            .iter()
            .any(|h| *h == [0; 3] || h.iter().any(|v| v.unsigned_abs() > 1024))
    {
        return Err(error("invalid Wilson input arrays"));
    }
    let x = resolution(cell, hkl)?;
    let c = form_squares(elements, &x)?;
    let mut by_hkl = BTreeMap::new();
    for (i, &h) in hkl.iter().enumerate() {
        if by_hkl.insert(h, i).is_some() {
            return Err(error("duplicate unmerged normalization reflection"));
        }
        if group
            .is_systematically_absent(h)
            .map_err(|e| error(e.to_string()))?
        {
            return Err(error("Wilson normalization requires allowed reflections"));
        }
    }
    let mut remaining: BTreeSet<_> = hkl.iter().copied().collect();
    let mut reps = Vec::new();
    while let Some(&h) = remaining.first() {
        let index = by_hkl[&h];
        reps.push(index);
        for mate in group
            .reflection_orbit(h, true)
            .map_err(|e| error(e.to_string()))?
        {
            if let Some(&j) = by_hkl.get(&mate) {
                if (intensity[j] - intensity[index]).abs()
                    > 1e-10 * intensity[j].abs().max(intensity[index].abs()).max(1e-100)
                {
                    return Err(error(
                        "normalization requires merged symmetry-equivalent intensities",
                    ));
                }
            }
            remaining.remove(&mate);
        }
    }
    let eps: Vec<_> = hkl
        .iter()
        .map(|&h| epsilon(group, h))
        .collect::<Result<_, _>>()?;
    let maximum = intensity.iter().map(|v| v.abs()).fold(0., f64::max);
    if maximum <= 0. {
        return Err(error("no positive Wilson shell signal"));
    }
    let reduced: Vec<_> = intensity
        .iter()
        .zip(&eps)
        .zip(&c)
        .map(|((&v, &e), &f)| v / maximum / (e as f64 * f))
        .collect();
    let mut ordered = reps.clone();
    let xmax = x.iter().copied().fold(0., f64::max);
    ordered.sort_by(|&i, &j| {
        (x[i] / xmax * 1e12)
            .round()
            .total_cmp(&(x[j] / xmax * 1e12).round())
            .then(hkl[i].cmp(&hkl[j]))
    });
    let bins = (reps.len() / 80).clamp(2, 20).min(reps.len());
    let mut shells = Vec::new();
    for b in 0..bins {
        let rows = &ordered[b * ordered.len() / bins..(b + 1) * ordered.len() / bins];
        shells.push(WilsonShell {
            s_squared: rows.iter().map(|&i| x[i]).sum::<f64>() / rows.len() as f64,
            families: rows.len(),
            reduced_mean: rows.iter().map(|&i| reduced[i]).sum::<f64>() / rows.len() as f64,
            log_residual: 0.,
        });
    }
    let mut merged: Vec<WilsonShell> = Vec::new();
    for shell in shells {
        if let Some(last) = merged.last_mut()
            && (last.s_squared - shell.s_squared).abs() <= xmax * 1e-12
        {
            let total = (last.families + shell.families) as f64;
            last.s_squared = (last.s_squared * last.families as f64
                + shell.s_squared * shell.families as f64)
                / total;
            last.reduced_mean = (last.reduced_mean * last.families as f64
                + shell.reduced_mean * shell.families as f64)
                / total;
            last.families += shell.families;
        } else {
            merged.push(shell);
        }
    }
    let mut shells = merged;
    let positive: Vec<_> = shells.iter().filter(|s| s.reduced_mean > 0.).collect();
    if positive.is_empty() {
        return Err(error("no positive signed Wilson shell mean"));
    }
    let weight = positive.iter().map(|s| s.families).sum::<usize>() as f64;
    let mx = positive
        .iter()
        .map(|s| s.s_squared * s.families as f64)
        .sum::<f64>()
        / weight;
    let my = positive
        .iter()
        .map(|s| s.reduced_mean.ln() * s.families as f64)
        .sum::<f64>()
        / weight;
    let var = positive
        .iter()
        .map(|s| (s.s_squared - mx).powi(2) * s.families as f64)
        .sum::<f64>();
    let slope = if var > 1e-30 {
        positive
            .iter()
            .map(|s| (s.s_squared - mx) * (s.reduced_mean.ln() - my) * s.families as f64)
            .sum::<f64>()
            / var
    } else {
        0.
    };
    let intercept = my - slope * mx;
    for shell in &mut shells {
        if shell.reduced_mean > 0. {
            shell.log_residual = (shell.reduced_mean.ln() - intercept - slope * shell.s_squared)
                * (shell.families as f64 / (shell.families as f64 + 80.));
        }
        shell.reduced_mean *= maximum;
    }
    let mut fit = WilsonNormalization {
        expected_intensity: Vec::new(),
        e_values: Vec::new(),
        epsilon: eps,
        representatives: reps,
        form_squares: c,
        audit: WilsonAudit {
            families: ordered.len(),
            nonpositive_families: ordered.iter().filter(|&&i| intensity[i] <= 0.).count(),
            log_scale: intercept + maximum.ln(),
            slope,
            mean_e_squared: 0.,
            shells,
        },
    };
    for (i, &resolution) in x.iter().enumerate() {
        fit.expected_intensity.push(fit.expected(
            resolution,
            fit.form_squares[i],
            fit.epsilon[i],
        )?);
    }
    let mean = fit
        .representatives
        .iter()
        .map(|&i| intensity[i].max(0.) / fit.expected_intensity[i])
        .sum::<f64>()
        / fit.representatives.len() as f64;
    if !mean.is_finite() || mean <= 0. {
        return Err(error("degenerate normalized amplitudes"));
    }
    fit.audit.log_scale += mean.ln();
    for (i, value) in fit.expected_intensity.iter_mut().enumerate() {
        *value *= mean;
        fit.e_values.push((intensity[i].max(0.) / *value).sqrt());
    }
    fit.audit.mean_e_squared = fit
        .representatives
        .iter()
        .map(|&i| fit.e_values[i].powi(2))
        .sum::<f64>()
        / fit.representatives.len() as f64;
    Ok(fit)
}

/// Measured-only phase-quality statistics and their derivatives versus calculated magnitudes.
#[derive(Clone, Debug)]
pub struct PhaseQualityScore {
    /// Pearson correlation, as a fraction rather than percent.
    pub correlation: f64,
    /// Mean normalized calculated E^2 in the weakest observed set.
    pub r_weak: f64,
    /// correlation - Rweak (published `0.01*CC_percent - Rweak`).
    pub cfom: f64,
    /// Whether both correlation variances are nonzero.
    pub correlation_defined: bool,
    /// Analytic d(CFOM)/d(calculated magnitude), in plan row order.
    pub gradient: Vec<f64>,
}
/// Per-trial actual quality histories and handoff decisions.
#[derive(Clone, Debug, Default)]
pub struct PhaseQualityTrace {
    /// Phase start index.
    pub trial: usize,
    /// Unique allowed measured families.
    pub allowed_families: usize,
    /// Observed systematic-absence constraints included in the weak set.
    pub absent_families: usize,
    /// Actual weak-set size including all ties at its boundary.
    pub weak_families: usize,
    /// Per-completed-cycle CC.
    pub correlation: Vec<f64>,
    /// Per-completed-cycle Rweak.
    pub r_weak: Vec<f64>,
    /// Per-completed-cycle CFOM.
    pub cfom: Vec<f64>,
    /// Early handoffs deferred because CFOM < 0.4.
    pub deferred_handoffs: usize,
    /// Checkpoint quality alternatives evaluated after the amplitude-SSE map.
    pub alternative_handoffs: usize,
    /// Calculated-amplitude resolution shells (zero for the earlier fixed envelope).
    pub calculated_shells: usize,
}
/// Prepared unique-family comparison, including actually observed forbidden reflections.
#[derive(Clone, Debug)]
pub struct PreparedPhaseQuality {
    /// Unique allowed then observed-absent indices.
    pub hkl: Vec<[i32; 3]>,
    /// Phasing target G at allowed indices, zero for absence constraints.
    pub target: Vec<f64>,
    /// Conversion from Gc^2 to an unscaled Ec^2.
    pub e_weights: Vec<f64>,
    /// Weak-set membership in plan order.
    pub weak: Vec<bool>,
    /// Count of allowed rows used for CC.
    pub allowed: usize,
    /// Fixed equal-population calculated-normalization shells; empty selects the legacy statistic.
    pub calculated_shell: Vec<usize>,
}
impl PreparedPhaseQuality {
    /// Construct a measured-only comparison with the same amplitude transform as phasing.
    /// # Errors
    /// Propagates normalization, symmetry and shape errors.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn new(
        cell: UnitCell,
        group: &SpaceGroup,
        data: &PreparedData,
        raw: &[Observation],
        elements: &[String],
        norm: &WilsonNormalization,
        wilson: bool,
        calculated_shell_normalization: bool,
    ) -> Result<Self, SolutionError> {
        let mut hkl: Vec<_> = norm.representatives.iter().map(|&i| data.hkl[i]).collect();
        let mut target = Vec::new();
        let mut e_weights = Vec::new();
        let mut observed_e = Vec::new();
        for &i in &norm.representatives {
            let envelope = if wilson {
                norm.expected_intensity[i]
            } else {
                norm.form_squares[i]
            };
            target.push(data.amplitudes[i] / envelope.powf(0.25));
            e_weights.push(envelope.sqrt() / norm.expected_intensity[i]);
            observed_e.push(norm.e_values[i]);
        }
        let maximum = target.iter().copied().fold(0., f64::max);
        for t in &mut target {
            *t /= maximum;
        }
        let allowed = hkl.len();
        let mut absent: BTreeMap<[i32; 3], Vec<&Observation>> = BTreeMap::new();
        for row in raw {
            if group
                .is_systematically_absent(row.hkl)
                .map_err(|e| error(e.to_string()))?
            {
                absent
                    .entry(
                        group
                            .reflection_orbit(row.hkl, true)
                            .map_err(|e| error(e.to_string()))?[0],
                    )
                    .or_default()
                    .push(row);
            }
        }
        let all_extra: Vec<_> = absent.keys().copied().collect();
        let extra: Vec<_> = all_extra
            .iter()
            .zip(resolution(cell, &all_extra)?)
            .filter(|(_, x)| *x <= 1. / (4. * data.d_min.powi(2)) * (1. + 1e-12))
            .map(|(&h, _)| h)
            .collect();
        let x = resolution(cell, &extra)?;
        let c = form_squares(elements, &x)?;
        for (j, &h) in extra.iter().enumerate() {
            let expected = norm.expected(x[j], c[j], epsilon(group, h)?)?;
            let envelope = if wilson { expected } else { c[j] };
            hkl.push(h);
            target.push(0.);
            e_weights.push(envelope.sqrt() / expected);
            let rows = &absent[&h];
            let minimum = rows.iter().map(|r| r.sigma).fold(f64::INFINITY, f64::min);
            let weights: Vec<_> = rows.iter().map(|r| (minimum / r.sigma).powi(2)).collect();
            let total: f64 = weights.iter().sum();
            let mean = rows
                .iter()
                .zip(weights)
                .map(|(r, w)| r.intensity * (w / total))
                .sum::<f64>();
            observed_e.push((mean.max(0.) / expected).sqrt());
        }
        let mut order: Vec<_> = (0..hkl.len()).collect();
        order.sort_by(|&i, &j| {
            observed_e[i]
                .total_cmp(&observed_e[j])
                .then(hkl[i].cmp(&hkl[j]))
        });
        let cut = observed_e[order[hkl.len().div_ceil(10) - 1]];
        let weak = observed_e
            .iter()
            .map(|&e| e <= cut + 1e-12 * cut.abs())
            .collect();
        let mut calculated_shell = Vec::new();
        if calculated_shell_normalization {
            let x = resolution(cell, &hkl)?;
            let xmax = x.iter().copied().fold(0., f64::max);
            let mut order: Vec<_> = (0..hkl.len()).collect();
            order.sort_by(|&i, &j| {
                (x[i] / xmax * 1e12)
                    .round()
                    .total_cmp(&(x[j] / xmax * 1e12).round())
                    .then(hkl[i].cmp(&hkl[j]))
            });
            let bins = (hkl.len() / 80).clamp(1, 20);
            calculated_shell.resize(hkl.len(), 0);
            for b in 0..bins {
                for &i in &order[b * hkl.len() / bins..(b + 1) * hkl.len() / bins] {
                    calculated_shell[i] = b;
                }
            }
            e_weights.fill(1.);
        }
        Ok(Self {
            hkl,
            target,
            e_weights,
            weak,
            allowed,
            calculated_shell,
        })
    }
    /// Evaluate scale-invariant CC/Rweak and a fused analytic CFOM gradient.
    /// # Errors
    /// Rejects mismatched/nonfinite/negative calculated magnitudes or a zero map.
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self, calculated: &[f64]) -> Result<PhaseQualityScore, SolutionError> {
        let n = calculated.len();
        if n != self.hkl.len()
            || n == 0
            || self.allowed == 0
            || self.allowed > n
            || self.target.len() != n
            || self.e_weights.len() != n
            || self.weak.len() != n
            || (!self.calculated_shell.is_empty() && self.calculated_shell.len() != n)
            || self.calculated_shell.iter().any(|&b| b >= n)
            || self.target.iter().any(|v| !v.is_finite())
            || self.e_weights.iter().any(|v| !v.is_finite() || *v <= 0.)
            || !self.weak.iter().any(|&w| w)
            || calculated.iter().any(|c| !c.is_finite() || *c < 0.)
        {
            return Err(error("invalid quality magnitudes"));
        }
        let max = calculated.iter().copied().fold(0., f64::max);
        if max <= 0. {
            return Err(error("zero quality map"));
        }
        let c: Vec<_> = calculated.iter().map(|v| v / max).collect();
        let a = self.allowed as f64;
        let mt = self.target[..self.allowed].iter().sum::<f64>() / a;
        let mc = c[..self.allowed].iter().sum::<f64>() / a;
        let vt = self.target[..self.allowed]
            .iter()
            .map(|t| (t - mt).powi(2))
            .sum::<f64>();
        let vc = c[..self.allowed]
            .iter()
            .map(|t| (t - mc).powi(2))
            .sum::<f64>();
        let defined = vt > 1e-28 && vc > 1e-28;
        let cc = if defined {
            self.target[..self.allowed]
                .iter()
                .zip(&c)
                .map(|(t, v)| (t - mt) * (v - mc))
                .sum::<f64>()
                / (vt * vc).sqrt()
        } else {
            0.
        };
        let weak_n = self.weak.iter().filter(|&&w| w).count() as f64;
        let bins = self.calculated_shell.iter().max().map_or(1, |b| b + 1);
        let shell = |i| self.calculated_shell.get(i).copied().unwrap_or(0);
        let mut populations = vec![0.; bins];
        let mut sums = vec![0.; bins];
        let mut weak_sums = vec![0.; bins];
        for (i, &amplitude) in c.iter().enumerate() {
            let b = shell(i);
            let value = amplitude * amplitude * self.e_weights[i];
            populations[b] += 1.;
            sums[b] += value;
            if self.weak[i] {
                weak_sums[b] += value;
            }
        }
        if sums.iter().sum::<f64>() <= 1e-250 || sums.iter().any(|s| !s.is_finite()) {
            return Err(error("degenerate calculated E normalization"));
        }
        let means: Vec<_> = sums
            .iter()
            .zip(&populations)
            .map(|(s, n)| if *n > 0. { s / n } else { 0. })
            .collect();
        // A completely zero shell contributes zero and has zero derivative by
        // explicit convention; positive shells are independently normalized.
        let rw = means
            .iter()
            .zip(&weak_sums)
            .filter(|(d, _)| **d > 0.)
            .map(|(d, a)| a / d / weak_n)
            .sum::<f64>();
        let gradient = (0..n)
            .map(|i| {
                let dcc = if defined && i < self.allowed {
                    (self.target[i] - mt) / (vt * vc).sqrt() - cc * (c[i] - mc) / vc
                } else {
                    0.
                };
                let b = shell(i);
                let dr = if means[b] > 0. {
                    2. * c[i] * self.e_weights[i] / (weak_n * means[b])
                        * (f64::from(self.weak[i]) - weak_sums[b] / sums[b])
                } else {
                    0.
                };
                (dcc - dr) / max
            })
            .collect();
        Ok(PhaseQualityScore {
            correlation: cc,
            r_weak: rw,
            cfom: cc - rw,
            correlation_defined: defined,
            gradient,
        })
    }
    /// Empty trace with exact comparison-set counts.
    #[must_use]
    pub fn trace(&self) -> PhaseQualityTrace {
        PhaseQualityTrace {
            allowed_families: self.allowed,
            absent_families: self.hkl.len() - self.allowed,
            weak_families: self.weak.iter().filter(|&&v| v).count(),
            calculated_shells: self.calculated_shell.iter().max().map_or(0, |b| b + 1),
            ..PhaseQualityTrace::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weak_reflection_evidence_can_reverse_amplitude_sse_ranking() {
        let target = vec![10., 10., 9., 9., 2., 2., 0.1, 0.1];
        let plan = PreparedPhaseQuality {
            hkl: vec![[1, 0, 0]; 8],
            target: target.clone(),
            e_weights: vec![1.; 8],
            weak: vec![false, false, false, false, false, false, true, true],
            allowed: 8,
            calculated_shell: Vec::new(),
        };
        let a = [10., 10., 9., 9., 2., 2., 1., 1.];
        let b = [10.8, 9.2, 9.8, 8.2, 2., 2., 0.1, 0.1];
        let sse = |v: &[f64]| {
            let scale = v.iter().zip(&target).map(|(x, y)| x * y).sum::<f64>()
                / v.iter().map(|x| x * x).sum::<f64>();
            v.iter()
                .zip(&target)
                .map(|(x, y)| (scale * x - y).powi(2))
                .sum::<f64>()
        };
        assert!(sse(&a) < sse(&b));
        let qa = plan.evaluate(&a).unwrap();
        let qb = plan.evaluate(&b).unwrap();
        assert!(qa.cfom < qb.cfom);
        assert!(qa.r_weak > qb.r_weak);
        assert!(plan.evaluate(&[0.; 8]).is_err());
        assert!(plan.evaluate(&[1.; 7]).is_err());
    }
    #[test]
    fn flat_correlations_and_scaling_are_explicit() {
        let plan = PreparedPhaseQuality {
            hkl: vec![[1, 0, 0]; 3],
            target: vec![1.; 3],
            e_weights: vec![1.; 3],
            weak: vec![true; 3],
            allowed: 3,
            calculated_shell: Vec::new(),
        };
        let value = plan.evaluate(&[1., 2., 3.]).unwrap();
        assert!(!value.correlation_defined);
        assert!((value.r_weak - 1.).abs() < 1e-14);
        assert!(value.gradient.iter().all(|g| g.abs() < 1e-14));
    }
}
