//! Symmetric pseudo-Voigt profile and fused peak accumulation.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::mem::size_of;

const FOUR_LN_2: f64 = 4.0 * std::f64::consts::LN_2;
const GAUSSIAN_NORMALIZATION: f64 = 0.939_437_278_699_651_3; // sqrt(4 ln(2) / pi)

/// Parameters for one symmetric pseudo-Voigt peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// Peak center in the same coordinate system as the sampling grid.
    pub position: f64,
    /// Integrated intensity before applying the finite support window.
    pub intensity: f64,
    /// Full width at half maximum; must be positive.
    pub fwhm: f64,
    /// Lorentzian mixing fraction in the inclusive range `[0, 1]`.
    pub eta: f64,
}

/// A validated, borrowed view of a strictly increasing sampling grid.
#[derive(Clone, Copy, Debug)]
pub struct GridView<'a> {
    values: &'a [f64],
}

impl<'a> GridView<'a> {
    /// Validate and borrow a sampling grid.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileError`] if a coordinate is non-finite or the grid is
    /// not strictly increasing.
    pub fn new(values: &'a [f64]) -> Result<Self, ProfileError> {
        validate_grid(values)?;
        Ok(Self { values })
    }

    /// Return the borrowed coordinates.
    #[must_use]
    pub const fn as_slice(self) -> &'a [f64] {
        self.values
    }
}

/// A validated structure-of-arrays view of peak parameters.
#[derive(Clone, Copy, Debug)]
pub struct PeakBatchView<'a> {
    positions: &'a [f64],
    intensities: &'a [f64],
    fwhms: &'a [f64],
    etas: &'a [f64],
}

impl<'a> PeakBatchView<'a> {
    /// Validate and borrow equal-length peak parameter arrays.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileError`] if the arrays differ in length or a parameter
    /// lies outside its domain.
    pub fn new(
        positions: &'a [f64],
        intensities: &'a [f64],
        fwhms: &'a [f64],
        etas: &'a [f64],
    ) -> Result<Self, ProfileError> {
        let peak_count = positions.len();
        if intensities.len() != peak_count || fwhms.len() != peak_count || etas.len() != peak_count
        {
            return Err(ProfileError::PeakLengthMismatch);
        }
        for peak_index in 0..peak_count {
            validate_peak(
                peak_index,
                Peak {
                    position: positions[peak_index],
                    intensity: intensities[peak_index],
                    fwhm: fwhms[peak_index],
                    eta: etas[peak_index],
                },
            )?;
        }
        Ok(Self {
            positions,
            intensities,
            fwhms,
            etas,
        })
    }

    /// Number of peaks in this batch.
    #[must_use]
    pub const fn len(self) -> usize {
        self.positions.len()
    }

    /// Whether this batch contains no peaks.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.positions.is_empty()
    }

    fn peak(self, index: usize) -> Peak {
        Peak {
            position: self.positions[index],
            intensity: self.intensities[index],
            fwhm: self.fwhms[index],
            eta: self.etas[index],
        }
    }
}

/// Deterministic rule used to choose a peak's finite support.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SupportPolicy {
    /// Include coordinates at most this many FWHM from the peak center.
    FwhmMultiple(f64),
}

/// Inclusive physical support limits for one peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupportRange {
    /// Inclusive left coordinate.
    pub left: f64,
    /// Inclusive right coordinate.
    pub right: f64,
}

impl SupportPolicy {
    fn validate(self) -> Result<(), ProfileError> {
        match self {
            Self::FwhmMultiple(multiple) if multiple.is_finite() && multiple > 0.0 => Ok(()),
            Self::FwhmMultiple(_) => Err(ProfileError::InvalidSupport),
        }
    }

    fn range(self, peak: Peak) -> SupportRange {
        match self {
            Self::FwhmMultiple(multiple) => {
                let radius = multiple * peak.fwhm;
                SupportRange {
                    left: peak.position - radius,
                    right: peak.position + radius,
                }
            }
        }
    }
}

/// A profile value and its analytical first derivatives.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProfilePoint {
    /// Unit-area profile value.
    pub value: f64,
    /// Derivative with respect to `delta = x - position`.
    pub d_delta: f64,
    /// Derivative with respect to FWHM.
    pub d_fwhm: f64,
    /// Derivative with respect to the Lorentzian fraction.
    pub d_eta: f64,
}

#[derive(Clone, Copy)]
struct ProfileComponents {
    inverse_fwhm: f64,
    z_squared: f64,
    gaussian: f64,
    lorentzian_denominator: f64,
    lorentzian: f64,
}

/// Per-peak Jacobian stored only over each peak's active support.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportJacobian {
    /// First active grid index for each peak.
    pub starts: Vec<usize>,
    /// Prefix sum of active sample counts, with length `peak_count + 1`.
    pub offsets: Vec<usize>,
    /// Sample-major derivative rows with four values per active sample.
    ///
    /// The local parameter order is intensity, position, FWHM, and eta.
    pub values: Vec<f64>,
}

impl SupportJacobian {
    /// Number of represented peaks.
    #[must_use]
    pub fn peak_count(&self) -> usize {
        self.starts.len()
    }

    /// Total number of active peak/sample pairs.
    #[must_use]
    pub fn active_sample_count(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// Materialize a parameter-major dense `(peak, 4, sample)` Jacobian.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileError`] if the dense allocation size overflows or the
    /// supplied sample count is inconsistent with a stored support block.
    pub fn to_dense(&self, sample_count: usize) -> Result<Vec<f64>, ProfileError> {
        self.validate_structure(sample_count)?;
        let dense_length = checked_matrix_len(
            self.peak_count()
                .checked_mul(4)
                .ok_or(ProfileError::AllocationOverflow)?,
            sample_count,
        )?;
        let mut dense = zeroed_f64_vec(dense_length)?;
        for peak_index in 0..self.peak_count() {
            let active_begin = self.offsets[peak_index];
            let active_end = self.offsets[peak_index + 1];
            let active_count = active_end - active_begin;
            let start = self.starts[peak_index];
            for relative_index in 0..active_count {
                let sample_index = start + relative_index;
                let sparse_base = (active_begin + relative_index) * 4;
                for parameter_index in 0..4 {
                    let dense_index =
                        (peak_index * 4 + parameter_index) * sample_count + sample_index;
                    dense[dense_index] = self.values[sparse_base + parameter_index];
                }
            }
        }
        Ok(dense)
    }

    fn validate_structure(&self, sample_count: usize) -> Result<(), ProfileError> {
        if self.offsets.len() != self.starts.len().saturating_add(1)
            || self.offsets.first() != Some(&0)
            || self.offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self.offsets.last().and_then(|count| count.checked_mul(4)) != Some(self.values.len())
        {
            return Err(ProfileError::InconsistentSupport);
        }
        for peak_index in 0..self.peak_count() {
            let active_count = self.offsets[peak_index + 1] - self.offsets[peak_index];
            if self.starts[peak_index]
                .checked_add(active_count)
                .is_none_or(|end| end > sample_count)
            {
                return Err(ProfileError::InconsistentSupport);
            }
        }
        Ok(())
    }
}

/// Derivatives of one calculated pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternDerivatives {
    /// Sparse per-peak derivatives.
    pub local: SupportJacobian,
    /// Optional parameter-major dense shared derivatives.
    pub global: Option<DenseJacobian>,
}

/// Parameter-major dense derivatives shared across many peaks.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseJacobian {
    /// Row-major values with shape `(parameter_count, sample_count)`.
    pub values: Vec<f64>,
    /// Number of global parameter rows.
    pub parameter_count: usize,
    /// Number of grid samples.
    pub sample_count: usize,
}

/// Result from fused multi-peak accumulation.
#[derive(Clone, Debug, PartialEq)]
pub struct Accumulation {
    /// Summed calculated profile, with one value per grid sample.
    pub y: Vec<f64>,
    /// Local and global analytical derivatives.
    pub derivatives: PatternDerivatives,
    /// Number of grid samples represented by the result.
    pub sample_count: usize,
}

impl Accumulation {
    /// Explicitly materialize the compatibility dense local Jacobian.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileError`] if allocation arithmetic overflows.
    pub fn dense_local_jacobian(&self) -> Result<Vec<f64>, ProfileError> {
        self.derivatives.local.to_dense(self.sample_count)
    }
}

/// Input validation errors for profile evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileError {
    /// The sampling grid contains NaN or infinity.
    NonFiniteGrid {
        /// Index of the invalid grid value.
        index: usize,
    },
    /// The sampling grid is not strictly increasing.
    UnsortedGrid {
        /// Index of the first value not greater than its predecessor.
        index: usize,
    },
    /// A peak parameter contains NaN or infinity.
    NonFinitePeak {
        /// Index of the invalid peak.
        peak: usize,
    },
    /// A peak FWHM is zero or negative.
    NonPositiveFwhm {
        /// Index of the invalid peak.
        peak: usize,
    },
    /// A peak eta is outside `[0, 1]`.
    InvalidEta {
        /// Index of the invalid peak.
        peak: usize,
    },
    /// The requested support is zero, negative, NaN, or infinity.
    InvalidSupport,
    /// Peak parameter arrays do not all have the same length.
    PeakLengthMismatch,
    /// A Jacobian allocation would overflow `usize`.
    AllocationOverflow,
    /// A stored support block does not fit the requested dense grid.
    InconsistentSupport,
}

impl Display for ProfileError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteGrid { index } => {
                write!(formatter, "grid value at index {index} is not finite")
            }
            Self::UnsortedGrid { index } => write!(
                formatter,
                "grid must be strictly increasing (violation at index {index})"
            ),
            Self::NonFinitePeak { peak } => {
                write!(formatter, "peak {peak} contains a non-finite parameter")
            }
            Self::NonPositiveFwhm { peak } => {
                write!(formatter, "peak {peak} has a non-positive FWHM")
            }
            Self::InvalidEta { peak } => {
                write!(formatter, "peak {peak} has eta outside [0, 1]")
            }
            Self::InvalidSupport => write!(formatter, "support_fwhm must be positive and finite"),
            Self::PeakLengthMismatch => write!(
                formatter,
                "positions, intensities, fwhms, and etas must have equal length"
            ),
            Self::AllocationOverflow => write!(formatter, "requested Jacobian is too large"),
            Self::InconsistentSupport => {
                write!(
                    formatter,
                    "support block lies outside the requested sample grid"
                )
            }
        }
    }
}

impl Error for ProfileError {}

/// Evaluate a unit-area symmetric pseudo-Voigt profile and derivatives.
///
/// `delta` is `x - position`, `fwhm` is the common full width at half maximum,
/// and `eta` is the Lorentzian fraction. Callers should validate `fwhm` and
/// `eta` at their API boundary; this low-level scalar function is branch-free.
#[must_use]
pub fn symmetric_pseudo_voigt(delta: f64, fwhm: f64, eta: f64) -> ProfilePoint {
    let components = profile_components(delta, fwhm);
    let ProfileComponents {
        inverse_fwhm,
        z_squared,
        gaussian,
        lorentzian_denominator,
        lorentzian,
    } = components;

    let d_gaussian_delta = gaussian * (-2.0 * FOUR_LN_2 * delta * inverse_fwhm.powi(2));
    let d_lorentzian_delta =
        lorentzian * (-8.0 * delta * inverse_fwhm.powi(2) / lorentzian_denominator);

    let d_gaussian_fwhm = gaussian * inverse_fwhm * (-1.0 + 2.0 * FOUR_LN_2 * z_squared);
    let d_lorentzian_fwhm =
        lorentzian * inverse_fwhm * (-1.0 + 8.0 * z_squared / lorentzian_denominator);

    let gaussian_weight = 1.0 - eta;
    ProfilePoint {
        value: eta * lorentzian + gaussian_weight * gaussian,
        d_delta: eta * d_lorentzian_delta + gaussian_weight * d_gaussian_delta,
        d_fwhm: eta * d_lorentzian_fwhm + gaussian_weight * d_gaussian_fwhm,
        d_eta: lorentzian - gaussian,
    }
}

#[inline]
fn symmetric_pseudo_voigt_value(delta: f64, fwhm: f64, eta: f64) -> f64 {
    let components = profile_components(delta, fwhm);
    eta * components.lorentzian + (1.0 - eta) * components.gaussian
}

#[inline]
fn profile_components(delta: f64, fwhm: f64) -> ProfileComponents {
    let inverse_fwhm = fwhm.recip();
    let z = delta * inverse_fwhm;
    let z_squared = z * z;
    let gaussian = GAUSSIAN_NORMALIZATION * inverse_fwhm * (-FOUR_LN_2 * z_squared).exp();
    let lorentzian_denominator = 1.0 + 4.0 * z_squared;
    let lorentzian = 2.0 * inverse_fwhm / (std::f64::consts::PI * lorentzian_denominator);
    ProfileComponents {
        inverse_fwhm,
        z_squared,
        gaussian,
        lorentzian_denominator,
        lorentzian,
    }
}

/// Accumulate a borrowed peak batch and all derivatives on a validated grid.
///
/// Each peak is evaluated only at samples satisfying
/// `abs(x - position) <= support_fwhm * fwhm`. Sparse derivative values are
/// sample-major within each peak support and use the parameter order intensity,
/// position, FWHM, and eta. The active sample set is held fixed when
/// differentiating.
///
/// # Errors
///
/// Returns [`ProfileError`] if support is invalid or allocation arithmetic
/// overflows. Grid and peak validation occurs when their views are constructed.
pub fn accumulate_batch(
    grid: GridView<'_>,
    peaks: PeakBatchView<'_>,
    support: SupportPolicy,
) -> Result<Accumulation, ProfileError> {
    support.validate()?;
    accumulate_source(
        grid.as_slice(),
        peaks.len(),
        |index| peaks.peak(index),
        support,
    )
}

/// Accumulate only calculated profile values for a borrowed peak batch.
///
/// This values-only path preserves peak-order summation and exact support
/// semantics while avoiding derivative evaluation and storage.
///
/// # Errors
///
/// Returns [`ProfileError`] if support is invalid or allocation fails.
pub fn accumulate_values_batch(
    grid: GridView<'_>,
    peaks: PeakBatchView<'_>,
    support: SupportPolicy,
) -> Result<Vec<f64>, ProfileError> {
    support.validate()?;
    let x = grid.as_slice();
    let mut y = zeroed_f64_vec(x.len())?;
    for peak_index in 0..peaks.len() {
        let peak = peaks.peak(peak_index);
        let range = support.range(peak);
        let lower = x.partition_point(|value| *value < range.left);
        let upper = x.partition_point(|value| *value <= range.right);
        for sample_index in lower..upper {
            y[sample_index] += peak.intensity
                * symmetric_pseudo_voigt_value(
                    x[sample_index] - peak.position,
                    peak.fwhm,
                    peak.eta,
                );
        }
    }
    Ok(y)
}

/// Convenience accumulator for an array-of-structs peak collection.
///
/// The Python binding uses [`accumulate_batch`] so its `NumPy` structure-of-arrays
/// input is borrowed directly without constructing a temporary `Vec<Peak>`.
///
/// # Errors
///
/// Returns [`ProfileError`] when an input is invalid or allocation arithmetic
/// overflows.
pub fn accumulate_peaks(
    x: &[f64],
    peaks: &[Peak],
    support_fwhm: f64,
) -> Result<Accumulation, ProfileError> {
    let grid = GridView::new(x)?;
    let support = SupportPolicy::FwhmMultiple(support_fwhm);
    support.validate()?;
    for (peak_index, peak) in peaks.iter().copied().enumerate() {
        validate_peak(peak_index, peak)?;
    }
    accumulate_source(grid.as_slice(), peaks.len(), |index| peaks[index], support)
}

fn accumulate_source(
    x: &[f64],
    peak_count: usize,
    peak_at: impl Fn(usize) -> Peak,
    support: SupportPolicy,
) -> Result<Accumulation, ProfileError> {
    let mut starts: Vec<usize> = Vec::new();
    starts
        .try_reserve_exact(peak_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    let offset_count = peak_count
        .checked_add(1)
        .ok_or(ProfileError::AllocationOverflow)?;
    let mut offsets: Vec<usize> = Vec::new();
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets.push(0);

    for peak_index in 0..peak_count {
        let peak = peak_at(peak_index);
        let range = support.range(peak);
        let lower = x.partition_point(|value| *value < range.left);
        let upper = x.partition_point(|value| *value <= range.right);
        let next_offset = offsets[peak_index]
            .checked_add(upper - lower)
            .ok_or(ProfileError::AllocationOverflow)?;
        starts.push(lower);
        offsets.push(next_offset);
    }

    let active_sample_count = offsets.last().copied().unwrap_or(0);
    let value_count = checked_matrix_len(active_sample_count, 4)?;
    let mut y = zeroed_f64_vec(x.len())?;
    let mut values = zeroed_f64_vec(value_count)?;

    for peak_index in 0..peak_count {
        let peak = peak_at(peak_index);
        let start = starts[peak_index];
        let active_begin = offsets[peak_index];
        let active_end = offsets[peak_index + 1];
        for active_index in active_begin..active_end {
            let sample_index = start + active_index - active_begin;
            let profile =
                symmetric_pseudo_voigt(x[sample_index] - peak.position, peak.fwhm, peak.eta);
            y[sample_index] += peak.intensity * profile.value;
            let value_base = active_index * 4;
            values[value_base] = profile.value;
            values[value_base + 1] = -peak.intensity * profile.d_delta;
            values[value_base + 2] = peak.intensity * profile.d_fwhm;
            values[value_base + 3] = peak.intensity * profile.d_eta;
        }
    }

    Ok(Accumulation {
        y,
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts,
                offsets,
                values,
            },
            global: None,
        },
        sample_count: x.len(),
    })
}

fn validate_grid(x: &[f64]) -> Result<(), ProfileError> {
    for (index, value) in x.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ProfileError::NonFiniteGrid { index });
        }
        if index > 0 && value <= x[index - 1] {
            return Err(ProfileError::UnsortedGrid { index });
        }
    }
    Ok(())
}

fn validate_peak(peak_index: usize, peak: Peak) -> Result<(), ProfileError> {
    if !peak.position.is_finite()
        || !peak.intensity.is_finite()
        || !peak.fwhm.is_finite()
        || !peak.eta.is_finite()
    {
        return Err(ProfileError::NonFinitePeak { peak: peak_index });
    }
    if peak.fwhm <= 0.0 {
        return Err(ProfileError::NonPositiveFwhm { peak: peak_index });
    }
    if !(0.0..=1.0).contains(&peak.eta) {
        return Err(ProfileError::InvalidEta { peak: peak_index });
    }
    Ok(())
}

fn checked_matrix_len(rows: usize, columns: usize) -> Result<usize, ProfileError> {
    let length = rows
        .checked_mul(columns)
        .ok_or(ProfileError::AllocationOverflow)?;
    length
        .checked_mul(size_of::<f64>())
        .filter(|bytes| isize::try_from(*bytes).is_ok())
        .ok_or(ProfileError::AllocationOverflow)?;
    Ok(length)
}

fn zeroed_f64_vec(length: usize) -> Result<Vec<f64>, ProfileError> {
    checked_matrix_len(length, 1)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.1e}"
        );
    }

    #[test]
    fn components_have_expected_height_and_half_maximum() {
        for eta in [0.0, 0.25, 1.0] {
            let center = symmetric_pseudo_voigt(0.0, 2.0, eta).value;
            let half_maximum = symmetric_pseudo_voigt(1.0, 2.0, eta).value;
            assert_close(half_maximum, center / 2.0, 1e-15);
        }
    }

    #[test]
    fn scalar_derivatives_match_centered_differences() {
        let delta = 0.37;
        let fwhm = 0.82;
        let eta = 0.41;
        let h = 1e-6;
        let analytic = symmetric_pseudo_voigt(delta, fwhm, eta);

        let d_delta = (symmetric_pseudo_voigt(delta + h, fwhm, eta).value
            - symmetric_pseudo_voigt(delta - h, fwhm, eta).value)
            / (2.0 * h);
        let d_fwhm = (symmetric_pseudo_voigt(delta, fwhm + h, eta).value
            - symmetric_pseudo_voigt(delta, fwhm - h, eta).value)
            / (2.0 * h);
        let d_eta = (symmetric_pseudo_voigt(delta, fwhm, eta + h).value
            - symmetric_pseudo_voigt(delta, fwhm, eta - h).value)
            / (2.0 * h);

        assert_close(analytic.d_delta, d_delta, 2e-10);
        assert_close(analytic.d_fwhm, d_fwhm, 2e-10);
        assert_close(analytic.d_eta, d_eta, 2e-10);
    }

    #[test]
    fn accumulation_is_exactly_support_limited() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let peaks = [Peak {
            position: 0.0,
            intensity: 3.0,
            fwhm: 1.0,
            eta: 0.5,
        }];
        let result = accumulate_peaks(&x, &peaks, 1.0).expect("valid profile");
        assert_close(result.y[0], 0.0, 0.0);
        assert!(result.y[1] > 0.0);
        assert!(result.y[2] > 0.0);
        assert!(result.y[3] > 0.0);
        assert_close(result.y[4], 0.0, 0.0);
        assert_eq!(result.derivatives.local.starts, [1]);
        assert_eq!(result.derivatives.local.offsets, [0, 3]);
        assert_eq!(result.derivatives.local.values.len(), 12);
        let dense = result.dense_local_jacobian().expect("dense compatibility");
        assert_close(dense[0], 0.0, 0.0);
    }

    #[test]
    fn overlapping_peaks_sum_without_overwriting_jacobian_rows() {
        let x = [-0.25, 0.0, 0.25];
        let peaks = [
            Peak {
                position: 0.0,
                intensity: 2.0,
                fwhm: 1.0,
                eta: 0.2,
            },
            Peak {
                position: 0.1,
                intensity: 4.0,
                fwhm: 0.7,
                eta: 0.8,
            },
        ];
        let result = accumulate_peaks(&x, &peaks, 10.0).expect("valid profile");
        let dense = result.dense_local_jacobian().expect("dense compatibility");
        for (sample, coordinate) in x.iter().copied().enumerate() {
            let first = symmetric_pseudo_voigt(coordinate, 1.0, 0.2);
            let second = symmetric_pseudo_voigt(coordinate - 0.1, 0.7, 0.8);
            assert_close(
                result.y[sample],
                2.0 * first.value + 4.0 * second.value,
                1e-14,
            );
            assert_close(dense[sample], first.value, 1e-14);
            assert_close(dense[4 * x.len() + sample], second.value, 1e-14);
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let peak = Peak {
            position: 0.0,
            intensity: 1.0,
            fwhm: 1.0,
            eta: 0.5,
        };
        assert!(matches!(
            accumulate_peaks(&[0.0, 0.0], &[peak], 5.0),
            Err(ProfileError::UnsortedGrid { index: 1 })
        ));
        assert!(matches!(
            accumulate_peaks(&[0.0], &[Peak { fwhm: 0.0, ..peak }], 5.0),
            Err(ProfileError::NonPositiveFwhm { peak: 0 })
        ));
        assert!(matches!(
            accumulate_peaks(&[0.0], &[Peak { eta: 1.1, ..peak }], 5.0),
            Err(ProfileError::InvalidEta { peak: 0 })
        ));
    }

    #[test]
    fn borrowed_batch_matches_peak_convenience_api() {
        let x = [-0.4, 0.0, 0.4];
        let positions = [0.0, 0.2];
        let intensities = [2.0, 3.0];
        let fwhms = [0.5, 0.7];
        let etas = [0.1, 0.8];
        let peaks = [
            Peak {
                position: positions[0],
                intensity: intensities[0],
                fwhm: fwhms[0],
                eta: etas[0],
            },
            Peak {
                position: positions[1],
                intensity: intensities[1],
                fwhm: fwhms[1],
                eta: etas[1],
            },
        ];
        let borrowed = accumulate_batch(
            GridView::new(&x).expect("grid"),
            PeakBatchView::new(&positions, &intensities, &fwhms, &etas).expect("peaks"),
            SupportPolicy::FwhmMultiple(3.0),
        )
        .expect("borrowed accumulation");
        let convenience = accumulate_peaks(&x, &peaks, 3.0).expect("peak accumulation");
        assert_eq!(borrowed, convenience);
        let values_only = accumulate_values_batch(
            GridView::new(&x).expect("grid"),
            PeakBatchView::new(&positions, &intensities, &fwhms, &etas).expect("peaks"),
            SupportPolicy::FwhmMultiple(3.0),
        )
        .expect("values-only accumulation");
        assert_eq!(values_only, borrowed.y);
    }

    #[test]
    fn empty_and_outside_support_blocks_are_well_formed() {
        let empty = accumulate_peaks(&[], &[], 2.0).expect("empty accumulation");
        assert!(empty.y.is_empty());
        assert!(empty.derivatives.local.starts.is_empty());
        assert_eq!(empty.derivatives.local.offsets, [0]);
        assert!(empty.derivatives.local.values.is_empty());

        let outside = accumulate_peaks(
            &[0.0, 1.0],
            &[Peak {
                position: 10.0,
                intensity: 1.0,
                fwhm: 0.1,
                eta: 0.5,
            }],
            1.0,
        )
        .expect("outside accumulation");
        assert_eq!(outside.y, [0.0, 0.0]);
        assert_eq!(outside.derivatives.local.starts, [2]);
        assert_eq!(outside.derivatives.local.offsets, [0, 0]);
        assert!(outside.derivatives.local.values.is_empty());
    }

    #[test]
    fn checked_dense_allocation_rejects_overflow() {
        assert_eq!(
            checked_matrix_len(usize::MAX, 2),
            Err(ProfileError::AllocationOverflow)
        );
        let sparse = SupportJacobian {
            starts: vec![1],
            offsets: vec![0, 1],
            values: vec![0.0; 4],
        };
        assert_eq!(sparse.to_dense(1), Err(ProfileError::InconsistentSupport));
    }
}
