//! Deterministic background-estimation kernels.
//!
//! Background estimation is preprocessing, not a refinable profile term. The
//! Bruckner smoother intentionally preserves the scan and boundary semantics of
//! xypattern's pinned Cython implementation.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Validation failures for background-estimation kernels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackgroundError {
    /// At least one intensity sample is required for endpoint padding.
    EmptyInput,
    /// One input sample is NaN or infinite.
    NonFiniteSample {
        /// Zero-based location of the invalid intensity.
        index: usize,
    },
    /// Requested padding cannot be represented by the platform index type.
    SizeOverflow,
    /// The padded work buffer cannot be allocated.
    AllocationFailed,
}

impl Display for BackgroundError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("background input must not be empty"),
            Self::NonFiniteSample { index } => {
                write!(formatter, "background input sample {index} must be finite")
            }
            Self::SizeOverflow => formatter.write_str("background smoothing size overflow"),
            Self::AllocationFailed => {
                formatter.write_str("background smoothing work buffer allocation failed")
            }
        }
    }
}

impl Error for BackgroundError {}

/// Reproduce the pinned xypattern Cython Smooth Bruckner estimator.
///
/// `smooth_points` is the half-window in samples, so each moving mean contains
/// `2 * smooth_points + 1` values. Each iteration scans extended indices
/// `smooth_points..y.len() - smooth_points - 2`; this intentionally leaves the
/// same trailing samples unchanged as the compatibility implementation.
///
/// # Errors
///
/// Returns [`BackgroundError`] when `y` is empty, contains a non-finite sample,
/// or the padded allocation length overflows `usize`.
pub fn smooth_bruckner(
    y: &[f64],
    smooth_points: usize,
    iterations: usize,
) -> Result<Vec<f64>, BackgroundError> {
    if y.is_empty() {
        return Err(BackgroundError::EmptyInput);
    }
    if let Some((index, _)) = y.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(BackgroundError::NonFiniteSample { index });
    }

    let double_padding = smooth_points
        .checked_mul(2)
        .ok_or(BackgroundError::SizeOverflow)?;
    let extended_len = y
        .len()
        .checked_add(double_padding)
        .ok_or(BackgroundError::SizeOverflow)?;
    let window_len = double_padding
        .checked_add(1)
        .ok_or(BackgroundError::SizeOverflow)?;
    let window_len_u32 = u32::try_from(window_len).map_err(|_| BackgroundError::SizeOverflow)?;

    let mut extended = Vec::new();
    extended
        .try_reserve_exact(extended_len)
        .map_err(|_| BackgroundError::AllocationFailed)?;
    extended.resize(extended_len, 0.0);
    extended[..smooth_points].fill(y[0]);
    extended[smooth_points..smooth_points + y.len()].copy_from_slice(y);
    extended[smooth_points + y.len()..].fill(y[y.len() - 1]);

    let scan_end = y.len().saturating_sub(smooth_points.saturating_add(2));
    let window_scale = f64::from(window_len_u32);
    for _ in 0..iterations {
        let mut window_sum = 0.0;
        for value in &extended[..window_len] {
            window_sum += value;
        }
        let mut window_average = window_sum / window_scale;

        for index in smooth_points..scan_end {
            let outgoing = extended[index - smooth_points];
            let incoming = extended[index + smooth_points + 1];
            if extended[index] > window_average {
                let old_value = extended[index];
                extended[index] = window_average;
                window_average +=
                    ((window_average - old_value) + (incoming - outgoing)) / window_scale;
            } else {
                window_average += (incoming - outgoing) / window_scale;
            }
        }
    }

    Ok(extended[smooth_points..smooth_points + y.len()].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_iterations_only_round_trips_the_padded_center() {
        let y = [1.0, 5.0, 2.0, 9.0];
        assert_eq!(smooth_bruckner(&y, 3, 0), Ok(y.to_vec()));
    }

    #[test]
    fn zero_half_window_preserves_every_sample() {
        let y = [1.0, 5.0, 2.0, 9.0];
        assert_eq!(smooth_bruckner(&y, 0, 20), Ok(y.to_vec()));
    }

    #[test]
    fn clipping_and_trailing_compatibility_range_are_explicit() {
        let y = [0.0, 0.0, 9.0, 0.0, 0.0, 7.0, 8.0, 9.0, 10.0];
        let actual = smooth_bruckner(&y, 1, 1).expect("valid smoother input");
        assert_eq!(actual, vec![0.0, 0.0, 3.0, 0.0, 0.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn window_larger_than_signal_is_a_safe_noop() {
        let y = [1.0, 4.0, 2.0];
        assert_eq!(smooth_bruckner(&y, 20, 5), Ok(y.to_vec()));
    }

    #[test]
    fn invalid_samples_are_rejected_with_their_index() {
        assert_eq!(smooth_bruckner(&[], 1, 1), Err(BackgroundError::EmptyInput));
        assert_eq!(
            smooth_bruckner(&[1.0, f64::NAN], 1, 1),
            Err(BackgroundError::NonFiniteSample { index: 1 })
        );
    }
}
