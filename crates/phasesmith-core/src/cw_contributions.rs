//! CW accumulation with externally prepared sample-physics contributions.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::cw::{ConstantWavelengthInstrument, CwBatchError, CwError, CwProfileParameters};
use crate::profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, ProfileError, SupportJacobian,
    SupportPolicy, zeroed_f64_vec,
};
use crate::tch::{TchShape, TchWidths};

const GAUSSIAN_FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949_3;
const INSTRUMENT_PARAMETER_COUNT: usize = 5;
const LOCAL_PARAMETER_COUNT: usize = 2;

/// Validated sample-physics contributions for one CW reflection batch.
#[derive(Clone, Copy, Debug)]
pub struct CwContributionsView<'a> {
    gaussian_variance_deg2: &'a [f64],
    lorentzian_fwhm_deg: &'a [f64],
    intensity_multiplier: &'a [f64],
    d_gaussian_variance_d_position: &'a [f64],
    d_lorentzian_fwhm_d_position: &'a [f64],
    d_intensity_multiplier_d_position: &'a [f64],
    d_gaussian_variance_d_parameters: &'a [f64],
    d_lorentzian_fwhm_d_parameters: &'a [f64],
    d_intensity_multiplier_d_parameters: &'a [f64],
    parameter_count: usize,
    reflection_count: usize,
}

/// Input arrays used to construct [`CwContributionsView`].
#[derive(Clone, Copy, Debug)]
pub struct CwContributionArrays<'a> {
    /// Additive Gaussian variance for each reflection.
    pub gaussian_variance_deg2: &'a [f64],
    /// Additive Lorentzian FWHM for each reflection.
    pub lorentzian_fwhm_deg: &'a [f64],
    /// Multiplicative integrated-intensity correction for each reflection.
    pub intensity_multiplier: &'a [f64],
    /// Position derivative of the Gaussian-variance contribution.
    pub d_gaussian_variance_d_position: &'a [f64],
    /// Position derivative of the Lorentzian-FWHM contribution.
    pub d_lorentzian_fwhm_d_position: &'a [f64],
    /// Position derivative of the intensity multiplier.
    pub d_intensity_multiplier_d_position: &'a [f64],
    /// Parameter-major Gaussian-variance chains, flattened from `(parameter, reflection)`.
    pub d_gaussian_variance_d_parameters: &'a [f64],
    /// Parameter-major Lorentzian-FWHM chains, flattened from `(parameter, reflection)`.
    pub d_lorentzian_fwhm_d_parameters: &'a [f64],
    /// Parameter-major intensity-multiplier chains, flattened from `(parameter, reflection)`.
    pub d_intensity_multiplier_d_parameters: &'a [f64],
}

fn validate_reflection_arrays(
    reflection_count: usize,
    arrays: CwContributionArrays<'_>,
) -> Result<(), CwContributionsError> {
    for (name, values) in [
        ("gaussian_variance_deg2", arrays.gaussian_variance_deg2),
        ("lorentzian_fwhm_deg", arrays.lorentzian_fwhm_deg),
        ("intensity_multiplier", arrays.intensity_multiplier),
        (
            "d_gaussian_variance_d_position",
            arrays.d_gaussian_variance_d_position,
        ),
        (
            "d_lorentzian_fwhm_d_position",
            arrays.d_lorentzian_fwhm_d_position,
        ),
        (
            "d_intensity_multiplier_d_position",
            arrays.d_intensity_multiplier_d_position,
        ),
    ] {
        if values.len() != reflection_count {
            return Err(CwContributionsError::LengthMismatch { name });
        }
    }
    for reflection in 0..reflection_count {
        for (quantity, value, non_negative) in [
            (
                "gaussian_variance_deg2",
                arrays.gaussian_variance_deg2[reflection],
                true,
            ),
            (
                "lorentzian_fwhm_deg",
                arrays.lorentzian_fwhm_deg[reflection],
                true,
            ),
            (
                "intensity_multiplier",
                arrays.intensity_multiplier[reflection],
                true,
            ),
            (
                "d_gaussian_variance_d_position",
                arrays.d_gaussian_variance_d_position[reflection],
                false,
            ),
            (
                "d_lorentzian_fwhm_d_position",
                arrays.d_lorentzian_fwhm_d_position[reflection],
                false,
            ),
            (
                "d_intensity_multiplier_d_position",
                arrays.d_intensity_multiplier_d_position[reflection],
                false,
            ),
        ] {
            if !value.is_finite() || (non_negative && value < 0.0) {
                return Err(CwContributionsError::InvalidContribution {
                    reflection,
                    quantity,
                });
            }
        }
    }
    Ok(())
}

fn validate_parameter_arrays(
    reflection_count: usize,
    parameter_count: usize,
    arrays: CwContributionArrays<'_>,
) -> Result<(), CwContributionsError> {
    let derivative_count = parameter_count
        .checked_mul(reflection_count)
        .ok_or(CwContributionsError::AllocationOverflow)?;
    let derivative_arrays = [
        (
            "d_gaussian_variance_d_parameters",
            arrays.d_gaussian_variance_d_parameters,
        ),
        (
            "d_lorentzian_fwhm_d_parameters",
            arrays.d_lorentzian_fwhm_d_parameters,
        ),
        (
            "d_intensity_multiplier_d_parameters",
            arrays.d_intensity_multiplier_d_parameters,
        ),
    ];
    for (name, values) in derivative_arrays {
        if values.len() != derivative_count {
            return Err(CwContributionsError::LengthMismatch { name });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(CwContributionsError::InvalidDerivative {
                parameter: index / reflection_count.max(1),
                reflection: index % reflection_count.max(1),
                quantity: name,
            });
        }
    }
    Ok(())
}

impl<'a> CwContributionsView<'a> {
    /// Validate and borrow one reflection-batch contribution set.
    ///
    /// # Errors
    ///
    /// Returns [`CwContributionsError`] for inconsistent lengths, non-finite
    /// values, negative broadening, or negative intensity multipliers.
    pub fn new(
        reflection_count: usize,
        parameter_count: usize,
        arrays: CwContributionArrays<'a>,
    ) -> Result<Self, CwContributionsError> {
        validate_reflection_arrays(reflection_count, arrays)?;
        validate_parameter_arrays(reflection_count, parameter_count, arrays)?;
        Ok(Self {
            gaussian_variance_deg2: arrays.gaussian_variance_deg2,
            lorentzian_fwhm_deg: arrays.lorentzian_fwhm_deg,
            intensity_multiplier: arrays.intensity_multiplier,
            d_gaussian_variance_d_position: arrays.d_gaussian_variance_d_position,
            d_lorentzian_fwhm_d_position: arrays.d_lorentzian_fwhm_d_position,
            d_intensity_multiplier_d_position: arrays.d_intensity_multiplier_d_position,
            d_gaussian_variance_d_parameters: arrays.d_gaussian_variance_d_parameters,
            d_lorentzian_fwhm_d_parameters: arrays.d_lorentzian_fwhm_d_parameters,
            d_intensity_multiplier_d_parameters: arrays.d_intensity_multiplier_d_parameters,
            parameter_count,
            reflection_count,
        })
    }

    /// Number of named provider parameters.
    #[must_use]
    pub const fn parameter_count(self) -> usize {
        self.parameter_count
    }

    const fn derivative_index(self, parameter: usize, reflection: usize) -> usize {
        parameter * self.reflection_count + reflection
    }
}

/// Errors while validating or accumulating external CW contributions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CwContributionsError {
    /// One contribution array has an inconsistent length.
    LengthMismatch {
        /// Stable array name.
        name: &'static str,
    },
    /// A per-reflection contribution is non-finite or outside its domain.
    InvalidContribution {
        /// Reflection index.
        reflection: usize,
        /// Stable quantity name.
        quantity: &'static str,
    },
    /// A parameter derivative is non-finite.
    InvalidDerivative {
        /// Provider parameter row.
        parameter: usize,
        /// Reflection index.
        reflection: usize,
        /// Stable derivative-array name.
        quantity: &'static str,
    },
    /// Reflection or instrument input is invalid.
    Cw {
        /// Underlying CW error.
        reason: CwBatchError,
    },
    /// Allocation size arithmetic overflowed.
    AllocationOverflow,
    /// Generic grid, support, or allocation failure.
    Profile {
        /// Underlying profile error.
        reason: ProfileError,
    },
}

impl Display for CwContributionsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { name } => write!(formatter, "{name} has an inconsistent length"),
            Self::InvalidContribution {
                reflection,
                quantity,
            } => write!(
                formatter,
                "reflection {reflection} has an invalid {quantity} contribution"
            ),
            Self::InvalidDerivative {
                parameter,
                reflection,
                quantity,
            } => write!(
                formatter,
                "provider parameter {parameter}, reflection {reflection} has an invalid {quantity} derivative"
            ),
            Self::Cw { reason } => Display::fmt(reason, formatter),
            Self::AllocationOverflow => write!(formatter, "contribution allocation size overflow"),
            Self::Profile { reason } => Display::fmt(reason, formatter),
        }
    }
}

impl Error for CwContributionsError {}

impl From<ProfileError> for CwContributionsError {
    fn from(reason: ProfileError) -> Self {
        Self::Profile { reason }
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedProfile {
    tch: TchShape,
    d_gaussian_d_instrument: [f64; INSTRUMENT_PARAMETER_COUNT],
    d_lorentzian_d_instrument: [f64; INSTRUMENT_PARAMETER_COUNT],
    d_gaussian_d_position: f64,
    d_lorentzian_d_position: f64,
    d_gaussian_d_variance: f64,
}

struct PreparedBatch {
    profiles: Vec<PreparedProfile>,
    starts: Vec<usize>,
    offsets: Vec<usize>,
}

fn prepare_profile(
    reflection: usize,
    position: f64,
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
) -> Result<PreparedProfile, CwContributionsError> {
    let base =
        CwProfileParameters::from_validated_instrument(position, instrument).map_err(|reason| {
            CwContributionsError::Cw {
                reason: CwBatchError::InvalidReflection { reflection, reason },
            }
        })?;
    let variance = base.gaussian_variance_deg2 + contributions.gaussian_variance_deg2[reflection];
    if !variance.is_finite() || variance <= 0.0 {
        return Err(CwContributionsError::Cw {
            reason: CwBatchError::InvalidReflection {
                reflection,
                reason: CwError::NonPositiveGaussianVariance,
            },
        });
    }
    let gaussian = GAUSSIAN_FWHM_PER_SIGMA * variance.sqrt();
    let lorentzian = base.lorentzian_fwhm_deg + contributions.lorentzian_fwhm_deg[reflection];
    let tch = TchShape::from_component_fwhm(TchWidths {
        gaussian_fwhm: gaussian,
        lorentzian_fwhm: lorentzian,
    })
    .map_err(|_| CwContributionsError::Cw {
        reason: CwBatchError::InvalidReflection {
            reflection,
            reason: CwError::InvalidTchTransform,
        },
    })?;
    let instrument_gaussian_scale = base.gaussian_fwhm_deg / gaussian;
    let d_gaussian_d_instrument = base
        .d_gaussian_fwhm_d_instrument
        .map(|value| value * instrument_gaussian_scale);
    let d_gaussian_d_variance = GAUSSIAN_FWHM_PER_SIGMA / (2.0 * variance.sqrt());
    Ok(PreparedProfile {
        tch,
        d_gaussian_d_instrument,
        d_lorentzian_d_instrument: base.d_lorentzian_fwhm_d_instrument,
        d_gaussian_d_position: base.d_gaussian_fwhm_d_two_theta * instrument_gaussian_scale
            + d_gaussian_d_variance * contributions.d_gaussian_variance_d_position[reflection],
        d_lorentzian_d_position: base.d_lorentzian_fwhm_d_two_theta
            + contributions.d_lorentzian_fwhm_d_position[reflection],
        d_gaussian_d_variance,
    })
}

fn prepare_batch(
    x: &[f64],
    positions_deg: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    support: SupportPolicy,
) -> Result<PreparedBatch, CwContributionsError> {
    let reflection_count = positions_deg.len();
    let mut profiles = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    profiles
        .try_reserve_exact(reflection_count)
        .map_err(|_| CwContributionsError::AllocationOverflow)?;
    starts
        .try_reserve_exact(reflection_count)
        .map_err(|_| CwContributionsError::AllocationOverflow)?;
    offsets
        .try_reserve_exact(
            reflection_count
                .checked_add(1)
                .ok_or(CwContributionsError::AllocationOverflow)?,
        )
        .map_err(|_| CwContributionsError::AllocationOverflow)?;
    offsets.push(0);
    for reflection in 0..reflection_count {
        let profile = prepare_profile(
            reflection,
            positions_deg[reflection],
            instrument,
            contributions,
        )?;
        let range = support.range(positions_deg[reflection], profile.tch.total_fwhm);
        let lower = x.partition_point(|value| *value < range.left);
        let upper = x.partition_point(|value| *value <= range.right);
        offsets.push(
            offsets[reflection]
                .checked_add(upper - lower)
                .ok_or(CwContributionsError::AllocationOverflow)?,
        );
        profiles.push(profile);
        starts.push(lower);
    }
    Ok(PreparedBatch {
        profiles,
        starts,
        offsets,
    })
}

/// Accumulate CW reflections with vectorized sample-physics contributions.
///
/// Local derivative order is base integrated intensity and position. Dense
/// global order is U/V/W/X/Y followed by the provider parameter rows.
///
/// # Errors
///
/// Returns [`CwContributionsError`] for invalid inputs or derived profiles.
pub fn accumulate_cw_contributions_batch(
    grid: GridView<'_>,
    positions_deg: &[f64],
    base_intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    support: SupportPolicy,
) -> Result<Accumulation, CwContributionsError> {
    support.validate()?;
    let reflections = crate::cw::CwReflectionBatchView::new(positions_deg, base_intensities)
        .map_err(|reason| CwContributionsError::Cw { reason })?;
    instrument
        .validate()
        .map_err(|reason| CwContributionsError::Cw {
            reason: CwBatchError::InvalidInstrument { reason },
        })?;
    if contributions.reflection_count != reflections.len() {
        return Err(CwContributionsError::LengthMismatch {
            name: "contribution reflection count",
        });
    }
    let x = grid.as_slice();
    let reflection_count = reflections.len();
    let global_parameter_count = INSTRUMENT_PARAMETER_COUNT
        .checked_add(contributions.parameter_count)
        .ok_or(CwContributionsError::AllocationOverflow)?;
    let prepared = prepare_batch(x, positions_deg, instrument, contributions, support)?;

    let active_count = prepared.offsets.last().copied().unwrap_or(0);
    let local_count = active_count
        .checked_mul(LOCAL_PARAMETER_COUNT)
        .ok_or(CwContributionsError::AllocationOverflow)?;
    let global_count = global_parameter_count
        .checked_mul(x.len())
        .ok_or(CwContributionsError::AllocationOverflow)?;
    let mut y = zeroed_f64_vec(x.len())?;
    let mut local_values = zeroed_f64_vec(local_count)?;
    let mut global_values = zeroed_f64_vec(global_count)?;
    for reflection in 0..reflection_count {
        let profile = prepared.profiles[reflection];
        let base_intensity = base_intensities[reflection];
        let multiplier = contributions.intensity_multiplier[reflection];
        let effective_intensity = base_intensity * multiplier;
        if !effective_intensity.is_finite() {
            return Err(CwContributionsError::InvalidContribution {
                reflection,
                quantity: "effective_intensity",
            });
        }
        let begin = prepared.offsets[reflection];
        let end = prepared.offsets[reflection + 1];
        for active in begin..end {
            let sample = prepared.starts[reflection] + active - begin;
            let point = profile.tch.evaluate(x[sample] - positions_deg[reflection]);
            y[sample] += effective_intensity * point.value;
            let local = active * LOCAL_PARAMETER_COUNT;
            local_values[local] = multiplier * point.value;
            local_values[local + 1] = base_intensity
                * (contributions.d_intensity_multiplier_d_position[reflection] * point.value
                    + multiplier
                        * (-point.d_delta
                            + point.d_gaussian_fwhm * profile.d_gaussian_d_position
                            + point.d_lorentzian_fwhm * profile.d_lorentzian_d_position));
            for parameter in 0..INSTRUMENT_PARAMETER_COUNT {
                let derivative = point.d_gaussian_fwhm * profile.d_gaussian_d_instrument[parameter]
                    + point.d_lorentzian_fwhm * profile.d_lorentzian_d_instrument[parameter];
                global_values[parameter * x.len() + sample] += effective_intensity * derivative;
            }
            for parameter in 0..contributions.parameter_count {
                let index = contributions.derivative_index(parameter, reflection);
                let d_multiplier = contributions.d_intensity_multiplier_d_parameters[index];
                let d_gaussian = profile.d_gaussian_d_variance
                    * contributions.d_gaussian_variance_d_parameters[index];
                let d_lorentzian = contributions.d_lorentzian_fwhm_d_parameters[index];
                let derivative = base_intensity
                    * (d_multiplier * point.value
                        + multiplier
                            * (point.d_gaussian_fwhm * d_gaussian
                                + point.d_lorentzian_fwhm * d_lorentzian));
                global_values[(INSTRUMENT_PARAMETER_COUNT + parameter) * x.len() + sample] +=
                    derivative;
            }
        }
    }
    Ok(Accumulation {
        y,
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts: prepared.starts,
                offsets: prepared.offsets,
                values: local_values,
                parameter_count: LOCAL_PARAMETER_COUNT,
            },
            global: Some(DenseJacobian {
                values: global_values,
                parameter_count: global_parameter_count,
                sample_count: x.len(),
            }),
        },
        sample_count: x.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cw::{CwReflectionBatchView, accumulate_cw_batch};

    fn instrument() -> ConstantWavelengthInstrument {
        ConstantWavelengthInstrument {
            wavelength_angstrom: 1.540_56,
            u_deg2: 2.0e-4,
            v_deg2: -1.0e-4,
            w_deg2: 1.2e-4,
            x_deg: 1.5e-3,
            y_deg: 3.0e-3,
        }
    }

    #[test]
    fn neutral_contributions_match_the_instrument_path_exactly() {
        let x: Vec<f64> = (0..=2_000)
            .map(|index| 39.0 + f64::from(index) * 0.001)
            .collect();
        let positions = [39.8, 40.2];
        let intensities = [12.0, 7.0];
        let zeros = [0.0, 0.0];
        let ones = [1.0, 1.0];
        let arrays = CwContributionArrays {
            gaussian_variance_deg2: &zeros,
            lorentzian_fwhm_deg: &zeros,
            intensity_multiplier: &ones,
            d_gaussian_variance_d_position: &zeros,
            d_lorentzian_fwhm_d_position: &zeros,
            d_intensity_multiplier_d_position: &zeros,
            d_gaussian_variance_d_parameters: &[],
            d_lorentzian_fwhm_d_parameters: &[],
            d_intensity_multiplier_d_parameters: &[],
        };
        let contributions = CwContributionsView::new(2, 0, arrays).expect("neutral");
        let grid = GridView::new(&x).expect("grid");
        let support = SupportPolicy::FwhmMultiple(20.0);
        let expected = accumulate_cw_batch(
            grid,
            CwReflectionBatchView::new(&positions, &intensities).expect("reflections"),
            instrument(),
            support,
        )
        .expect("CW");
        let actual = accumulate_cw_contributions_batch(
            grid,
            &positions,
            &intensities,
            instrument(),
            contributions,
            support,
        )
        .expect("contributions");
        assert_eq!(actual, expected);
    }

    #[test]
    fn position_and_provider_derivatives_match_centered_differences() {
        let x: Vec<f64> = (0..=2_000)
            .map(|index| 49.5 + f64::from(index) * 0.000_5)
            .collect();
        let intensity = [8.0];
        let support = SupportPolicy::FwhmMultiple(100.0);
        let calculate = |position: f64, amplitude: f64| {
            let normalized = position / 100.0;
            let variance = [amplitude * normalized * normalized];
            let zeros = [0.0];
            let ones = [1.0];
            let d_variance_d_position = [2.0 * amplitude * normalized / 100.0];
            let d_variance_d_amplitude = [normalized * normalized];
            let arrays = CwContributionArrays {
                gaussian_variance_deg2: &variance,
                lorentzian_fwhm_deg: &zeros,
                intensity_multiplier: &ones,
                d_gaussian_variance_d_position: &d_variance_d_position,
                d_lorentzian_fwhm_d_position: &zeros,
                d_intensity_multiplier_d_position: &zeros,
                d_gaussian_variance_d_parameters: &d_variance_d_amplitude,
                d_lorentzian_fwhm_d_parameters: &zeros,
                d_intensity_multiplier_d_parameters: &zeros,
            };
            accumulate_cw_contributions_batch(
                GridView::new(&x).expect("grid"),
                &[position],
                &intensity,
                instrument(),
                CwContributionsView::new(1, 1, arrays).expect("contributions"),
                support,
            )
            .expect("contribution accumulation")
        };

        let position = 50.0;
        let amplitude = 3.0e-4;
        let baseline = calculate(position, amplitude);
        let position_step = 1.0e-6;
        let position_plus = calculate(position + position_step, amplitude);
        let position_minus = calculate(position - position_step, amplitude);
        let dense = baseline
            .derivatives
            .local
            .to_dense(x.len())
            .expect("dense local derivatives");
        for (sample, (&plus_value, &minus_value)) in
            position_plus.y.iter().zip(&position_minus.y).enumerate()
        {
            let finite_difference = (plus_value - minus_value) / (2.0 * position_step);
            let analytical = dense[x.len() + sample];
            assert!(
                (analytical - finite_difference).abs() < 8.0e-6 * finite_difference.abs().max(1.0)
            );
        }

        let amplitude_step = 1.0e-8;
        let amplitude_plus = calculate(position, amplitude + amplitude_step);
        let amplitude_minus = calculate(position, amplitude - amplitude_step);
        let global = baseline.derivatives.global.as_ref().expect("global");
        for (sample, (&plus_value, &minus_value)) in
            amplitude_plus.y.iter().zip(&amplitude_minus.y).enumerate()
        {
            let finite_difference = (plus_value - minus_value) / (2.0 * amplitude_step);
            let analytical = global.values[INSTRUMENT_PARAMETER_COUNT * x.len() + sample];
            assert!(
                (analytical - finite_difference).abs() < 8.0e-6 * finite_difference.abs().max(1.0)
            );
        }
    }
}
