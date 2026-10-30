//! Value-only phase merits with immutable-plan statistics and reusable work arrays.
use crate::{PhaseQualityScore, PreparedPhaseQuality, SolutionError, error};

/// The four scalar merits, without a derivative allocation.
#[derive(Clone, Copy, Debug)]
pub struct PhaseQualityValue {
    /// Pearson correlation as a fraction.
    pub correlation: f64,
    /// Mean calculated E-squared in the weak set.
    pub r_weak: f64,
    /// Correlation minus Rweak.
    pub cfom: f64,
    /// Both correlation variances are nonzero.
    pub correlation_defined: bool,
}
impl From<PhaseQualityScore> for PhaseQualityValue {
    fn from(value: PhaseQualityScore) -> Self {
        Self {
            correlation: value.correlation,
            r_weak: value.r_weak,
            cfom: value.cfom,
            correlation_defined: value.correlation_defined,
        }
    }
}

/// Borrowed immutable plan, cached target moments/populations, and owned scratch.
/// The full fused value/gradient evaluator remains the independent native control.
pub struct PreparedPhaseScore<'a> {
    plan: &'a PreparedPhaseQuality,
    mean_target: f64,
    variance_target: f64,
    weak_count: f64,
    shell: Vec<usize>,
    populations: Vec<f64>,
    normalized: Vec<f64>,
    sums: Vec<f64>,
    weak_sums: Vec<f64>,
    means: Vec<f64>,
}
impl<'a> PreparedPhaseScore<'a> {
    /// Validate fixed arrays once and prepare scalar-only scoring.
    /// # Errors
    /// Rejects invalid plan lengths, indices, weights, target values or weak sets.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(plan: &'a PreparedPhaseQuality) -> Result<Self, SolutionError> {
        let n = plan.hkl.len();
        if n == 0
            || plan.allowed == 0
            || plan.allowed > n
            || plan.target.len() != n
            || plan.e_weights.len() != n
            || plan.weak.len() != n
            || (!plan.calculated_shell.is_empty() && plan.calculated_shell.len() != n)
            || plan.calculated_shell.iter().any(|&b| b >= n)
            || plan.target.iter().any(|v| !v.is_finite())
            || plan.e_weights.iter().any(|v| !v.is_finite() || *v <= 0.)
            || !plan.weak.iter().any(|&v| v)
        {
            return Err(error("invalid quality magnitudes"));
        }
        let a = plan.allowed as f64;
        let mean_target = plan.target[..plan.allowed].iter().sum::<f64>() / a;
        let variance_target = plan.target[..plan.allowed]
            .iter()
            .map(|t| (t - mean_target).powi(2))
            .sum::<f64>();
        let weak_count = plan.weak.iter().filter(|&&w| w).count() as f64;
        let bins = plan.calculated_shell.iter().max().map_or(1, |b| b + 1);
        let shell = if plan.calculated_shell.is_empty() {
            vec![0; n]
        } else {
            plan.calculated_shell.clone()
        };
        let mut populations = vec![0.; bins];
        for &b in &shell {
            populations[b] += 1.;
        }
        Ok(Self {
            plan,
            mean_target,
            variance_target,
            weak_count,
            shell,
            populations,
            normalized: vec![0.; n],
            sums: vec![0.; bins],
            weak_sums: vec![0.; bins],
            means: vec![0.; bins],
        })
    }
    /// Score one changing map with the control's arithmetic/reduction order.
    /// # Errors
    /// Rejects invalid or zero calculated amplitudes and degenerate normalization.
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&mut self, calculated: &[f64]) -> Result<PhaseQualityValue, SolutionError> {
        if calculated.len() != self.normalized.len()
            || calculated.iter().any(|c| !c.is_finite() || *c < 0.)
        {
            return Err(error("invalid quality magnitudes"));
        }
        let max = calculated.iter().copied().fold(0., f64::max);
        if max <= 0. {
            return Err(error("zero quality map"));
        }
        for (out, &value) in self.normalized.iter_mut().zip(calculated) {
            *out = value / max;
        }
        let c = &self.normalized;
        let allowed = self.plan.allowed;
        let mc = c[..allowed].iter().sum::<f64>() / allowed as f64;
        let vc = c[..allowed].iter().map(|t| (t - mc).powi(2)).sum::<f64>();
        let vt = self.variance_target;
        let mt = self.mean_target;
        let defined = vt > 1e-28 && vc > 1e-28;
        let cc = if defined {
            self.plan.target[..allowed]
                .iter()
                .zip(c)
                .map(|(t, v)| (t - mt) * (v - mc))
                .sum::<f64>()
                / (vt * vc).sqrt()
        } else {
            0.
        };
        self.sums.fill(0.);
        self.weak_sums.fill(0.);
        for (i, &amplitude) in c.iter().enumerate() {
            let b = self.shell[i];
            let value = amplitude * amplitude * self.plan.e_weights[i];
            self.sums[b] += value;
            if self.plan.weak[i] {
                self.weak_sums[b] += value;
            }
        }
        if self.sums.iter().sum::<f64>() <= 1e-250 || self.sums.iter().any(|s| !s.is_finite()) {
            return Err(error("degenerate calculated E normalization"));
        }
        for ((out, s), n) in self.means.iter_mut().zip(&self.sums).zip(&self.populations) {
            *out = if *n > 0. { s / n } else { 0. };
        }
        let rw = self
            .means
            .iter()
            .zip(&self.weak_sums)
            .filter(|(d, _)| **d > 0.)
            .map(|(d, a)| a / d / self.weak_count)
            .sum::<f64>();
        Ok(PhaseQualityValue {
            correlation: cc,
            r_weak: rw,
            cfom: cc - rw,
            correlation_defined: defined,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> PreparedPhaseQuality {
        PreparedPhaseQuality {
            hkl: vec![[1, 0, 0]; 8],
            target: vec![1., 2., 3., 4., 5., 6., 0., 0.],
            e_weights: vec![1.; 8],
            weak: vec![false, false, false, false, false, false, true, true],
            allowed: 6,
            calculated_shell: vec![0, 0, 2, 2, 4, 4, 4, 4],
        }
    }
    #[test]
    fn cached_values_match_fused_for_empty_and_zero_shells_and_repeated_maps() {
        let p = plan();
        let mut cache = PreparedPhaseScore::new(&p).unwrap();
        for row in [
            [1.; 8],
            [0., 0., 2., 3., 4., 1., 0., 0.],
            [2., 1., 3., 4., 5., 6., 0.1, 0.2],
        ] {
            let a = cache.evaluate(&row).unwrap();
            let b = p.evaluate(&row).unwrap();
            assert_eq!(a.correlation.to_bits(), b.correlation.to_bits());
            assert_eq!(a.r_weak.to_bits(), b.r_weak.to_bits());
            assert_eq!(a.cfom.to_bits(), b.cfom.to_bits());
            assert_eq!(a.correlation_defined, b.correlation_defined);
        }
    }
    #[test]
    fn invalid_plans_are_rejected_before_fast_scoring() {
        for i in 0..7 {
            let mut p = plan();
            match i {
                0 => p.allowed = 0,
                1 => p.weak.fill(false),
                2 => p.target[0] = f64::NAN,
                3 => p.e_weights[0] = 0.,
                4 => p.calculated_shell[0] = 8,
                5 => p.e_weights.pop().map(|_| ()).unwrap(),
                _ => p.hkl.clear(),
            }
            assert!(PreparedPhaseScore::new(&p).is_err());
        }
    }
}
