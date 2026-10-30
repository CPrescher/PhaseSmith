//! Work-sliced phase-first scheduling; no composition or reference inputs.
use crate::{SolutionError, error};

/// Deterministic cycle quota: early half-starts use half work, later starts full work.
/// The 32-cycle floor preserves the existing quality-target warm-up.
/// # Errors
/// Rejects controls outside the solver's established bounds or an invalid start.
pub fn phase_cycle_quota(
    cycles: usize,
    trials: usize,
    trial: usize,
) -> Result<usize, SolutionError> {
    phase_cycle_quota_with_short_starts(cycles, trials, trial, false)
}

/// Quarter-work early starts with original full-work fallback starts.
/// # Errors
/// Rejects the same controls as `phase_cycle_quota`.
pub fn phase_cycle_quota_with_short_starts(
    cycles: usize,
    trials: usize,
    trial: usize,
    short: bool,
) -> Result<usize, SolutionError> {
    if !(1..=20_000).contains(&cycles) || !(1..=128).contains(&trials) || trial >= trials {
        return Err(error("invalid phase work-slice controls"));
    }
    Ok(if trial < trials / 2 {
        cycles
            .div_ceil(if short { 4 } else { 2 })
            .max(32)
            .min(cycles)
    } else {
        cycles
    })
}

/// Actual phase slice, including wall/deadline exits before its work quota.
#[derive(Clone, Debug)]
pub struct PhaseSliceAudit {
    /// Zero-based phase-start index.
    pub trial: usize,
    /// This start's work ceiling, separate from the global requested ceiling.
    pub cycle_quota: usize,
    /// Scored/completed Fourier cycles.
    pub completed_cycles: usize,
    /// Time inside the P1 solver, including preparation.
    pub seconds: f64,
    /// `cycle_limit`, `phase_time_limit`, `quality_target`, `cancelled`, or `degenerate`.
    pub stop_reason: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn early_work_is_bounded_and_full_fallback_is_retained() {
        assert_eq!(
            (0..4)
                .map(|i| phase_cycle_quota(800, 4, i).unwrap())
                .collect::<Vec<_>>(),
            vec![400, 400, 800, 800]
        );
        assert_eq!(phase_cycle_quota(800, 1, 0).unwrap(), 800);
        assert_eq!(
            (0..3)
                .map(|i| phase_cycle_quota(63, 3, i).unwrap())
                .collect::<Vec<_>>(),
            vec![32, 63, 63]
        );
        for cycles in [1, 2, 31, 32, 33, 64, 800, 20_000] {
            for trials in 1..=128 {
                for trial in 0..trials {
                    let quota = phase_cycle_quota(cycles, trials, trial).unwrap();
                    assert!((1..=cycles).contains(&quota));
                    if trial >= trials / 2 {
                        assert_eq!(quota, cycles);
                    }
                }
            }
        }
    }
    #[test]
    fn invalid_work_requests_fail() {
        for (cycles, trials, trial) in [
            (0, 4, 0),
            (20_001, 4, 0),
            (800, 0, 0),
            (800, 129, 0),
            (800, 4, 4),
        ] {
            assert!(phase_cycle_quota(cycles, trials, trial).is_err());
        }
    }
    #[test]
    fn short_early_work_retains_full_later_starts_and_original_bounds() {
        assert_eq!(
            (0..4)
                .map(|i| phase_cycle_quota_with_short_starts(800, 4, i, true).unwrap())
                .collect::<Vec<_>>(),
            vec![200, 200, 800, 800]
        );
        for cycles in [1, 2, 31, 32, 33, 63, 64, 127, 128, 799, 800, 20_000] {
            for trials in 1..=128 {
                for trial in 0..trials {
                    let q =
                        phase_cycle_quota_with_short_starts(cycles, trials, trial, true).unwrap();
                    assert!((1..=cycles).contains(&q));
                    assert!(q <= phase_cycle_quota(cycles, trials, trial).unwrap());
                    if trial >= trials / 2 {
                        assert_eq!(q, cycles);
                    }
                }
            }
        }
        for (c, t, j) in [
            (0, 4, 0),
            (20_001, 4, 0),
            (800, 0, 0),
            (800, 129, 0),
            (800, 4, 4),
        ] {
            assert!(phase_cycle_quota_with_short_starts(c, t, j, true).is_err());
        }
    }
}
