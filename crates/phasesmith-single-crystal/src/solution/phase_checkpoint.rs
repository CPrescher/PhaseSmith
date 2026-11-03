//! Empirical measured-map handoff policy; no atom or reference information.
use crate::solution::{SolutionError, error};

/// One actual adaptive decision, separate from full-model acceptance.
#[derive(Clone, Debug)]
pub struct PhaseCheckpoint {
    /// Completed Fourier cycles.
    pub cycles: usize,
    /// Live checkpoint map's measured-only phase quality.
    pub cfom: f64,
    /// Running-best quality (the map eventually handed off).
    pub best_cfom: f64,
    /// Live checkpoint map's profiled amplitude SSE.
    pub score: f64,
    /// Cosine against the preceding checkpoint's live atomic map.
    pub map_cosine: Option<f64>,
    /// Whether this checkpoint selects a map for downstream model testing.
    pub selected: bool,
    /// `quality_target`, `stable_map`, `continue`, or `cycle_ceiling`.
    pub reason: &'static str,
}

/// Normalized map agreement, invariant to a positive density scale.
/// # Errors
/// Rejects empty, mismatched, nonfinite or zero maps.
pub fn checkpoint_map_cosine(a: &[f64], b: &[f64]) -> Result<f64, SolutionError> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(error("invalid checkpoint maps"));
    }
    let scale_a = a.iter().map(|v| v.abs()).fold(0., f64::max);
    let scale_b = b.iter().map(|v| v.abs()).fold(0., f64::max);
    if scale_a == 0. || scale_b == 0. {
        return Err(error("zero checkpoint map"));
    }
    let (mut ab, mut aa, mut bb) = (0., 0., 0.);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (a / scale_a, b / scale_b);
        ab += a * b;
        aa += a * a;
        bb += b * b;
    }
    Ok((ab / (aa * bb).sqrt()).clamp(-1., 1.))
}

/// Stateful checkpoint selection. A plateau can hand off at 64 or 256 cycles;
/// the existing quality target remains available from cycle 32 on every cycle.
#[derive(Default)]
pub struct PhaseCheckpointSelector {
    previous: Option<(Vec<f64>, f64, f64)>,
}
impl PhaseCheckpointSelector {
    /// Evaluate a scheduled checkpoint; no decision changes the Fourier iterate.
    /// # Errors
    /// Rejects invalid metrics or maps.
    pub fn consider(
        &mut self,
        map: &[f64],
        cycles: usize,
        cfom: f64,
        score: f64,
        best_cfom: f64,
        ceiling: bool,
    ) -> Result<PhaseCheckpoint, SolutionError> {
        if cycles == 0
            || !cfom.is_finite()
            || !best_cfom.is_finite()
            || !score.is_finite()
            || score < 0.
        {
            return Err(error("invalid checkpoint metrics"));
        }
        let cosine = if let Some((previous, _, _)) = &self.previous {
            Some(checkpoint_map_cosine(map, previous)?)
        } else {
            checkpoint_map_cosine(map, map)?;
            None
        };
        let stable = self
            .previous
            .as_ref()
            .is_some_and(|(_, old_quality, old_score)| {
                cycles >= 64
                    && cfom >= 0.65
                    && (cfom - old_quality).abs() <= 0.015
                    && best_cfom - cfom <= 0.015
                    && (score - old_score).abs() <= 0.02 * old_score.max(1e-12)
                    && cosine.is_some_and(|c| c >= 0.995)
            });
        let reason = if cycles >= 32 && best_cfom >= 0.9 {
            "quality_target"
        } else if stable {
            "stable_map"
        } else if ceiling {
            "cycle_ceiling"
        } else {
            "continue"
        };
        self.previous = Some((map.to_vec(), cfom, score));
        Ok(PhaseCheckpoint {
            cycles,
            cfom,
            best_cfom,
            score,
            map_cosine: cosine,
            selected: reason != "continue",
            reason,
        })
    }
}
