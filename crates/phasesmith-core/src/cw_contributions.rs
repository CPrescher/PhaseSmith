//! CW accumulation with externally prepared sample-physics contributions.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::ProfileAccuracy;
use crate::cw::{ConstantWavelengthInstrument, CwBatchError, CwError, CwProfileParameters};
use crate::fcj::{FcjError, FcjGeometry, FcjProfile, FcjProfilePoint};
use crate::profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, ProfileError, SupportJacobian,
    SupportPolicy, zeroed_f64_vec,
};
use crate::tch::{TchShape, TchWidths};
use phasesmith_execution::ExecutionContext;

const GAUSSIAN_FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949_3;
const INSTRUMENT_PARAMETER_COUNT: usize = 5;
const FCJ_PARAMETER_COUNT: usize = 2;
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

/// Owned arrays used to construct [`OwnedCwContributions`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OwnedCwContributionArrays {
    /// Additive Gaussian variance for each reflection.
    pub gaussian_variance_deg2: Vec<f64>,
    /// Additive Lorentzian FWHM for each reflection.
    pub lorentzian_fwhm_deg: Vec<f64>,
    /// Multiplicative integrated-intensity correction for each reflection.
    pub intensity_multiplier: Vec<f64>,
    /// Position derivative of the Gaussian-variance contribution.
    pub d_gaussian_variance_d_position: Vec<f64>,
    /// Position derivative of the Lorentzian-FWHM contribution.
    pub d_lorentzian_fwhm_d_position: Vec<f64>,
    /// Position derivative of the intensity multiplier.
    pub d_intensity_multiplier_d_position: Vec<f64>,
    /// Parameter-major Gaussian-variance chains, flattened from `(parameter, reflection)`.
    pub d_gaussian_variance_d_parameters: Vec<f64>,
    /// Parameter-major Lorentzian-FWHM chains, flattened from `(parameter, reflection)`.
    pub d_lorentzian_fwhm_d_parameters: Vec<f64>,
    /// Parameter-major intensity-multiplier chains, flattened from `(parameter, reflection)`.
    pub d_intensity_multiplier_d_parameters: Vec<f64>,
}

impl OwnedCwContributionArrays {
    fn as_borrowed(&self) -> CwContributionArrays<'_> {
        CwContributionArrays {
            gaussian_variance_deg2: &self.gaussian_variance_deg2,
            lorentzian_fwhm_deg: &self.lorentzian_fwhm_deg,
            intensity_multiplier: &self.intensity_multiplier,
            d_gaussian_variance_d_position: &self.d_gaussian_variance_d_position,
            d_lorentzian_fwhm_d_position: &self.d_lorentzian_fwhm_d_position,
            d_intensity_multiplier_d_position: &self.d_intensity_multiplier_d_position,
            d_gaussian_variance_d_parameters: &self.d_gaussian_variance_d_parameters,
            d_lorentzian_fwhm_d_parameters: &self.d_lorentzian_fwhm_d_parameters,
            d_intensity_multiplier_d_parameters: &self.d_intensity_multiplier_d_parameters,
        }
    }
}

/// Validated owned sample-physics contributions for one CW reflection batch.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedCwContributions {
    reflection_count: usize,
    parameter_count: usize,
    arrays: OwnedCwContributionArrays,
}

impl OwnedCwContributions {
    /// Validate and take ownership of one reflection-batch contribution set.
    ///
    /// # Errors
    ///
    /// Returns [`CwContributionsError`] for inconsistent lengths, non-finite
    /// values, negative broadening, or negative intensity multipliers.
    pub fn new(
        reflection_count: usize,
        parameter_count: usize,
        arrays: OwnedCwContributionArrays,
    ) -> Result<Self, CwContributionsError> {
        CwContributionsView::new(reflection_count, parameter_count, arrays.as_borrowed())?;
        Ok(Self {
            reflection_count,
            parameter_count,
            arrays,
        })
    }

    /// Construct neutral contributions with no provider parameters.
    #[must_use]
    pub fn neutral(reflection_count: usize) -> Self {
        Self {
            reflection_count,
            parameter_count: 0,
            arrays: OwnedCwContributionArrays {
                gaussian_variance_deg2: vec![0.0; reflection_count],
                lorentzian_fwhm_deg: vec![0.0; reflection_count],
                intensity_multiplier: vec![1.0; reflection_count],
                d_gaussian_variance_d_position: vec![0.0; reflection_count],
                d_lorentzian_fwhm_d_position: vec![0.0; reflection_count],
                d_intensity_multiplier_d_position: vec![0.0; reflection_count],
                ..OwnedCwContributionArrays::default()
            },
        }
    }

    /// Borrow the owned arrays as a validated kernel input.
    #[must_use]
    pub fn as_view(&self) -> CwContributionsView<'_> {
        CwContributionsView {
            gaussian_variance_deg2: &self.arrays.gaussian_variance_deg2,
            lorentzian_fwhm_deg: &self.arrays.lorentzian_fwhm_deg,
            intensity_multiplier: &self.arrays.intensity_multiplier,
            d_gaussian_variance_d_position: &self.arrays.d_gaussian_variance_d_position,
            d_lorentzian_fwhm_d_position: &self.arrays.d_lorentzian_fwhm_d_position,
            d_intensity_multiplier_d_position: &self.arrays.d_intensity_multiplier_d_position,
            d_gaussian_variance_d_parameters: &self.arrays.d_gaussian_variance_d_parameters,
            d_lorentzian_fwhm_d_parameters: &self.arrays.d_lorentzian_fwhm_d_parameters,
            d_intensity_multiplier_d_parameters: &self.arrays.d_intensity_multiplier_d_parameters,
            parameter_count: self.parameter_count,
            reflection_count: self.reflection_count,
        }
    }

    /// Number of reflections represented by this batch.
    #[must_use]
    pub const fn reflection_count(&self) -> usize {
        self.reflection_count
    }

    /// Number of named provider parameters.
    #[must_use]
    pub const fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Borrow the owned contribution arrays.
    #[must_use]
    pub const fn arrays(&self) -> &OwnedCwContributionArrays {
        &self.arrays
    }
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
    /// One FCJ profile could not be prepared.
    Fcj {
        /// Reflection index.
        reflection: usize,
        /// Underlying FCJ error.
        reason: FcjError,
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
            Self::Fcj { reflection, reason } => {
                write!(
                    formatter,
                    "reflection {reflection} has invalid FCJ geometry: {reason}"
                )
            }
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

#[derive(Clone, Debug)]
struct PreparedProfile {
    tch: TchShape,
    fcj: Option<FcjProfile>,
    support_radius_deg: f64,
    d_gaussian_d_instrument: [f64; INSTRUMENT_PARAMETER_COUNT],
    d_lorentzian_d_instrument: [f64; INSTRUMENT_PARAMETER_COUNT],
    d_gaussian_d_position: f64,
    d_lorentzian_d_position: f64,
    d_gaussian_d_variance: f64,
}

impl PreparedProfile {
    fn evaluate_batch<const AXIAL: bool>(
        &self,
        x_deg: [f64; 4],
        position_deg: f64,
    ) -> [FcjProfilePoint; 4] {
        if let Some(fcj) = &self.fcj {
            return fcj.evaluate_batch::<AXIAL>(x_deg, self.support_radius_deg);
        }
        x_deg.map(|x| self.evaluate::<AXIAL>(x, position_deg))
    }

    fn evaluate<const AXIAL: bool>(&self, x_deg: f64, position_deg: f64) -> FcjProfilePoint {
        if let Some(fcj) = &self.fcj {
            return fcj.evaluate_selected::<AXIAL>(x_deg, self.support_radius_deg);
        }
        let point = self.tch.evaluate(x_deg - position_deg);
        FcjProfilePoint {
            value: point.value,
            d_position: -point.d_delta,
            d_gaussian_fwhm: point.d_gaussian_fwhm,
            d_lorentzian_fwhm: point.d_lorentzian_fwhm,
            d_sample_over_radius: 0.0,
            d_detector_over_radius: 0.0,
        }
    }
}

struct PreparedBatch {
    profiles: Vec<PreparedProfile>,
    starts: Vec<usize>,
    offsets: Vec<usize>,
}

struct ReflectionBlock {
    start: usize,
    y: Vec<f64>,
    local: Vec<f64>,
    global: Vec<f64>,
}

fn prepare_profile(
    reflection: usize,
    position: f64,
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    geometry: Option<FcjGeometry>,
    support: SupportPolicy,
    accuracy: ProfileAccuracy,
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
    let support_radius_deg = accuracy.radius(tch.total_fwhm, tch.eta, support);
    let fcj = geometry
        .map(|geometry| {
            FcjProfile::new_with_accuracy(
                position,
                TchWidths {
                    gaussian_fwhm: gaussian,
                    lorentzian_fwhm: lorentzian,
                },
                geometry,
                accuracy.fast_fcj,
            )
            .map_err(|reason| CwContributionsError::Fcj { reflection, reason })
        })
        .transpose()?;
    Ok(PreparedProfile {
        tch,
        fcj,
        support_radius_deg,
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
    geometry: Option<FcjGeometry>,
    support: SupportPolicy,
    accuracy: ProfileAccuracy,
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
            geometry,
            support,
            accuracy,
        )?;
        let range = match &profile.fcj {
            Some(fcj) => fcj.support_range(profile.support_radius_deg),
            None => crate::SupportRange {
                left: positions_deg[reflection] - profile.support_radius_deg,
                right: positions_deg[reflection] + profile.support_radius_deg,
            },
        };
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
    accumulate_cw_contributions_batch_with_context(
        grid,
        positions_deg,
        base_intensities,
        instrument,
        contributions,
        support,
        &ExecutionContext::serial(),
    )
}

/// Accumulate symmetric CW contributions with a bounded execution context.
///
/// # Errors
///
/// Returns [`CwContributionsError`] for invalid inputs or derived profiles.
pub fn accumulate_cw_contributions_batch_with_context(
    grid: GridView<'_>,
    positions_deg: &[f64],
    base_intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    support: SupportPolicy,
    execution: &ExecutionContext,
) -> Result<Accumulation, CwContributionsError> {
    accumulate_cw_contributions_impl::<true>(
        grid,
        positions_deg,
        base_intensities,
        instrument,
        contributions,
        None,
        support,
        ProfileAccuracy::default(),
        execution,
    )
}

/// Accumulate FCJ-asymmetric CW reflections with vectorized sample physics.
///
/// Local derivative order is base integrated intensity and ideal position.
/// Dense global order is U/V/W/X/Y, the sample and detector axial ratios,
/// followed by provider parameter rows.
///
/// # Errors
///
/// Returns [`CwContributionsError`] for invalid inputs or derived profiles.
pub fn accumulate_cw_fcj_contributions_batch(
    grid: GridView<'_>,
    positions_deg: &[f64],
    base_intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    geometry: FcjGeometry,
    support: SupportPolicy,
) -> Result<Accumulation, CwContributionsError> {
    accumulate_cw_fcj_contributions_batch_with_context(
        grid,
        positions_deg,
        base_intensities,
        instrument,
        contributions,
        geometry,
        support,
        &ExecutionContext::serial(),
    )
}

/// Accumulate FCJ-asymmetric contributions with a bounded execution context.
///
/// # Errors
///
/// Returns [`CwContributionsError`] for invalid inputs or derived profiles.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_cw_fcj_contributions_batch_with_context(
    grid: GridView<'_>,
    positions_deg: &[f64],
    base_intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    geometry: FcjGeometry,
    support: SupportPolicy,
    execution: &ExecutionContext,
) -> Result<Accumulation, CwContributionsError> {
    accumulate_cw_contributions_impl::<true>(
        grid,
        positions_deg,
        base_intensities,
        instrument,
        contributions,
        Some(geometry),
        support,
        ProfileAccuracy::default(),
        execution,
    )
}

/// Accumulate values and required derivatives with fixed axial geometry.
/// The two axial derivative rows are retained as zeros; all other rows and
/// closed finite-support boundaries are identical to the full calculation.
/// # Errors
/// Returns an error for invalid grids, profiles or contribution arrays.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_cw_fixed_axial_contributions_with_context(
    grid: GridView<'_>,
    positions_deg: &[f64],
    base_intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    geometry: FcjGeometry,
    support: SupportPolicy,
    execution: &ExecutionContext,
) -> Result<Accumulation, CwContributionsError> {
    accumulate_cw_contributions_impl::<false>(
        grid,
        positions_deg,
        base_intensities,
        instrument,
        contributions,
        Some(geometry),
        support,
        ProfileAccuracy::default(),
        execution,
    )
}

/// Accumulate structural CW profiles with an explicit numerical accuracy policy.
/// Values and requested derivatives are fused; fixed axial rows can be omitted.
/// # Errors
/// Returns an error for invalid arrays, geometry or accuracy controls.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_cw_contributions_with_accuracy(
    grid: GridView<'_>,
    positions_deg: &[f64],
    base_intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    geometry: Option<FcjGeometry>,
    support: SupportPolicy,
    accuracy: ProfileAccuracy,
    axial_derivatives: bool,
    execution: &ExecutionContext,
) -> Result<Accumulation, CwContributionsError> {
    if axial_derivatives {
        accumulate_cw_contributions_impl::<true>(
            grid,
            positions_deg,
            base_intensities,
            instrument,
            contributions,
            geometry,
            support,
            accuracy,
            execution,
        )
    } else {
        accumulate_cw_contributions_impl::<false>(
            grid,
            positions_deg,
            base_intensities,
            instrument,
            contributions,
            geometry,
            support,
            accuracy,
            execution,
        )
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn accumulate_cw_contributions_impl<const AXIAL: bool>(
    grid: GridView<'_>,
    positions_deg: &[f64],
    base_intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
    contributions: CwContributionsView<'_>,
    geometry: Option<FcjGeometry>,
    support: SupportPolicy,
    accuracy: ProfileAccuracy,
    execution: &ExecutionContext,
) -> Result<Accumulation, CwContributionsError> {
    support.validate()?;
    accuracy.validate()?;
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
    let axial_parameter_count = if geometry.is_some() {
        FCJ_PARAMETER_COUNT
    } else {
        0
    };
    let global_parameter_count = INSTRUMENT_PARAMETER_COUNT
        .checked_add(axial_parameter_count)
        .ok_or(CwContributionsError::AllocationOverflow)?
        .checked_add(contributions.parameter_count)
        .ok_or(CwContributionsError::AllocationOverflow)?;
    let prepared = prepare_batch(
        x,
        positions_deg,
        instrument,
        contributions,
        geometry,
        support,
        accuracy,
    )?;

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
    if execution.threads() == 1 || reflection_count < 16 {
        for reflection in 0..reflection_count {
            let profile = &prepared.profiles[reflection];
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
            for chunk in (begin..end).step_by(4) {
                let count = (end - chunk).min(4);
                let coordinates = std::array::from_fn(|lane| {
                    x[prepared.starts[reflection] + chunk - begin + lane.min(count - 1)]
                });
                let points =
                    profile.evaluate_batch::<AXIAL>(coordinates, positions_deg[reflection]);
                for (lane, point) in points.into_iter().take(count).enumerate() {
                    let active = chunk + lane;
                    let sample = prepared.starts[reflection] + active - begin;
                    y[sample] += effective_intensity * point.value;
                    let local = active * LOCAL_PARAMETER_COUNT;
                    local_values[local] = multiplier * point.value;
                    local_values[local + 1] = base_intensity
                        * (contributions.d_intensity_multiplier_d_position[reflection]
                            * point.value
                            + multiplier
                                * (point.d_position
                                    + point.d_gaussian_fwhm * profile.d_gaussian_d_position
                                    + point.d_lorentzian_fwhm * profile.d_lorentzian_d_position));
                    for parameter in 0..INSTRUMENT_PARAMETER_COUNT {
                        let derivative = point.d_gaussian_fwhm
                            * profile.d_gaussian_d_instrument[parameter]
                            + point.d_lorentzian_fwhm
                                * profile.d_lorentzian_d_instrument[parameter];
                        global_values[parameter * x.len() + sample] +=
                            effective_intensity * derivative;
                    }
                    if geometry.is_some() {
                        global_values[INSTRUMENT_PARAMETER_COUNT * x.len() + sample] +=
                            effective_intensity * point.d_sample_over_radius;
                        global_values[(INSTRUMENT_PARAMETER_COUNT + 1) * x.len() + sample] +=
                            effective_intensity * point.d_detector_over_radius;
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
                        global_values[(INSTRUMENT_PARAMETER_COUNT
                            + axial_parameter_count
                            + parameter)
                            * x.len()
                            + sample] += derivative;
                    }
                }
            }
        }
    } else {
        let blocks = execution.map_ordered(reflection_count, 16, |reflection| {
            let profile = &prepared.profiles[reflection];
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
            let support_count = end - begin;
            let mut block = ReflectionBlock {
                start: prepared.starts[reflection],
                y: zeroed_f64_vec(support_count)?,
                local: zeroed_f64_vec(
                    support_count
                        .checked_mul(LOCAL_PARAMETER_COUNT)
                        .ok_or(CwContributionsError::AllocationOverflow)?,
                )?,
                global: zeroed_f64_vec(
                    support_count
                        .checked_mul(global_parameter_count)
                        .ok_or(CwContributionsError::AllocationOverflow)?,
                )?,
            };
            for chunk in (0..support_count).step_by(4) {
                let count = (support_count - chunk).min(4);
                let coordinates =
                    std::array::from_fn(|lane| x[block.start + chunk + lane.min(count - 1)]);
                let points =
                    profile.evaluate_batch::<AXIAL>(coordinates, positions_deg[reflection]);
                for (lane, point) in points.into_iter().take(count).enumerate() {
                    let support_index = chunk + lane;
                    block.y[support_index] = effective_intensity * point.value;
                    let local = support_index * LOCAL_PARAMETER_COUNT;
                    block.local[local] = multiplier * point.value;
                    block.local[local + 1] = base_intensity
                        * (contributions.d_intensity_multiplier_d_position[reflection]
                            * point.value
                            + multiplier
                                * (point.d_position
                                    + point.d_gaussian_fwhm * profile.d_gaussian_d_position
                                    + point.d_lorentzian_fwhm * profile.d_lorentzian_d_position));
                    for parameter in 0..INSTRUMENT_PARAMETER_COUNT {
                        let derivative = point.d_gaussian_fwhm
                            * profile.d_gaussian_d_instrument[parameter]
                            + point.d_lorentzian_fwhm
                                * profile.d_lorentzian_d_instrument[parameter];
                        block.global[parameter * support_count + support_index] =
                            effective_intensity * derivative;
                    }
                    if geometry.is_some() {
                        block.global[INSTRUMENT_PARAMETER_COUNT * support_count + support_index] =
                            effective_intensity * point.d_sample_over_radius;
                        block.global
                            [(INSTRUMENT_PARAMETER_COUNT + 1) * support_count + support_index] =
                            effective_intensity * point.d_detector_over_radius;
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
                        block.global[(INSTRUMENT_PARAMETER_COUNT
                            + axial_parameter_count
                            + parameter)
                            * support_count
                            + support_index] = derivative;
                    }
                }
            }
            Ok(block)
        });
        for (reflection, block) in blocks.into_iter().enumerate() {
            let block = block?;
            let begin = prepared.offsets[reflection];
            let support_count = block.y.len();
            let local_begin = begin * LOCAL_PARAMETER_COUNT;
            let local_end = local_begin + block.local.len();
            local_values[local_begin..local_end].copy_from_slice(&block.local);
            for support_index in 0..support_count {
                let sample = block.start + support_index;
                y[sample] += block.y[support_index];
                for parameter in 0..global_parameter_count {
                    global_values[parameter * x.len() + sample] +=
                        block.global[parameter * support_count + support_index];
                }
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

    #[test]
    fn accuracy_support_is_closed_and_omitted_axial_rows_preserve_values() {
        let position = 40.0;
        let accuracy = ProfileAccuracy {
            fast_fcj: true,
            tail_area_tolerance: Some(0.01),
        };
        let owned = OwnedCwContributions::neutral(1);
        let contributions = owned.as_view();
        let support = SupportPolicy::FwhmMultiple(20.0);
        let profile = prepare_profile(
            0,
            position,
            instrument(),
            contributions,
            None,
            support,
            accuracy,
        )
        .unwrap();
        let right = position + profile.support_radius_deg;
        let x = [
            position,
            f64::from_bits(right.to_bits() - 1),
            right,
            f64::from_bits(right.to_bits() + 1),
        ];
        let context = ExecutionContext::serial();
        let result = accumulate_cw_contributions_with_accuracy(
            GridView::new(&x).unwrap(),
            &[position],
            &[1.0],
            instrument(),
            contributions,
            None,
            support,
            accuracy,
            true,
            &context,
        )
        .unwrap();
        assert!(result.y[1] > 0.0 && result.y[2] > 0.0);
        assert_eq!(result.y[3].to_bits(), 0.0_f64.to_bits());
        let axial = Some(FcjGeometry {
            sample_over_radius: 0.001,
            detector_over_radius: 0.001,
        });
        let full = accumulate_cw_contributions_with_accuracy(
            GridView::new(&x).unwrap(),
            &[position],
            &[1.0],
            instrument(),
            contributions,
            axial,
            support,
            accuracy,
            true,
            &context,
        )
        .unwrap();
        let selected = accumulate_cw_contributions_with_accuracy(
            GridView::new(&x).unwrap(),
            &[position],
            &[1.0],
            instrument(),
            contributions,
            axial,
            support,
            accuracy,
            false,
            &context,
        )
        .unwrap();
        assert_eq!(full.y, selected.y);
        assert_eq!(full.derivatives.local, selected.derivatives.local);
        assert!(
            selected.derivatives.global.unwrap().values[5 * x.len()..]
                .iter()
                .all(|&v| v == 0.0)
        );
    }

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
    fn symmetric_and_fcj_blocks_are_bitwise_identical_across_worker_counts() {
        let x = (0..=30_000)
            .map(|index| 20.0 + f64::from(index) * 0.003)
            .collect::<Vec<_>>();
        let positions = (0..48)
            .map(|index| 25.0 + f64::from(index) * 1.6)
            .collect::<Vec<_>>();
        let intensities = (0..48)
            .map(|index| 5.0 + 0.2 * f64::from(index))
            .collect::<Vec<_>>();
        let zero = vec![0.0; positions.len()];
        let one = vec![1.0; positions.len()];
        let provider = (0..48)
            .map(|index| 1.0e-5 * f64::from(index + 1))
            .collect::<Vec<_>>();
        let arrays = CwContributionArrays {
            gaussian_variance_deg2: &zero,
            lorentzian_fwhm_deg: &zero,
            intensity_multiplier: &one,
            d_gaussian_variance_d_position: &zero,
            d_lorentzian_fwhm_d_position: &zero,
            d_intensity_multiplier_d_position: &zero,
            d_gaussian_variance_d_parameters: &provider,
            d_lorentzian_fwhm_d_parameters: &zero,
            d_intensity_multiplier_d_parameters: &zero,
        };
        let contributions =
            CwContributionsView::new(positions.len(), 1, arrays).expect("contributions");
        let grid = GridView::new(&x).expect("grid");
        let support = SupportPolicy::FwhmMultiple(20.0);
        let serial = ExecutionContext::serial();
        let two = ExecutionContext::new(2).expect("two threads");
        let three = ExecutionContext::new(3).expect("three threads");

        let expected = accumulate_cw_contributions_batch_with_context(
            grid,
            &positions,
            &intensities,
            instrument(),
            contributions,
            support,
            &serial,
        )
        .expect("serial symmetric");
        let expected_fcj = accumulate_cw_fcj_contributions_batch_with_context(
            grid,
            &positions,
            &intensities,
            instrument(),
            contributions,
            FcjGeometry {
                sample_over_radius: 0.002,
                detector_over_radius: 0.003,
            },
            support,
            &serial,
        )
        .expect("serial FCJ");
        for context in [&two, &three] {
            assert_eq!(
                accumulate_cw_contributions_batch_with_context(
                    grid,
                    &positions,
                    &intensities,
                    instrument(),
                    contributions,
                    support,
                    context,
                )
                .expect("parallel symmetric"),
                expected
            );
            assert_eq!(
                accumulate_cw_fcj_contributions_batch_with_context(
                    grid,
                    &positions,
                    &intensities,
                    instrument(),
                    contributions,
                    FcjGeometry {
                        sample_over_radius: 0.002,
                        detector_over_radius: 0.003,
                    },
                    support,
                    context,
                )
                .expect("parallel FCJ"),
                expected_fcj
            );
        }
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

    #[test]
    fn zero_fcj_geometry_exactly_matches_symmetric_contributions() {
        let x: Vec<f64> = (0..=2_000)
            .map(|index| 39.0 + f64::from(index) * 0.001)
            .collect();
        let positions = [39.8, 40.2];
        let intensities = [12.0, 7.0];
        let variance = [2.0e-5, 3.0e-5];
        let lorentzian = [1.0e-3, 2.0e-3];
        let multiplier = [0.8, 1.2];
        let zeros = [0.0, 0.0];
        let provider = [0.1, 0.2];
        let arrays = CwContributionArrays {
            gaussian_variance_deg2: &variance,
            lorentzian_fwhm_deg: &lorentzian,
            intensity_multiplier: &multiplier,
            d_gaussian_variance_d_position: &zeros,
            d_lorentzian_fwhm_d_position: &zeros,
            d_intensity_multiplier_d_position: &zeros,
            d_gaussian_variance_d_parameters: &zeros,
            d_lorentzian_fwhm_d_parameters: &zeros,
            d_intensity_multiplier_d_parameters: &provider,
        };
        let contributions = CwContributionsView::new(2, 1, arrays).expect("contributions");
        let grid = GridView::new(&x).expect("grid");
        let support = SupportPolicy::FwhmMultiple(20.0);
        let symmetric = accumulate_cw_contributions_batch(
            grid,
            &positions,
            &intensities,
            instrument(),
            contributions,
            support,
        )
        .expect("symmetric");
        let fcj = accumulate_cw_fcj_contributions_batch(
            grid,
            &positions,
            &intensities,
            instrument(),
            contributions,
            FcjGeometry {
                sample_over_radius: 0.0,
                detector_over_radius: 0.0,
            },
            support,
        )
        .expect("FCJ");
        assert_eq!(fcj.y, symmetric.y);
        assert_eq!(fcj.derivatives.local, symmetric.derivatives.local);
        let symmetric_global = symmetric.derivatives.global.expect("symmetric global");
        let fcj_global = fcj.derivatives.global.expect("FCJ global");
        assert_eq!(
            &fcj_global.values[..INSTRUMENT_PARAMETER_COUNT * x.len()],
            &symmetric_global.values[..INSTRUMENT_PARAMETER_COUNT * x.len()]
        );
        assert_eq!(
            &fcj_global.values[(INSTRUMENT_PARAMETER_COUNT + FCJ_PARAMETER_COUNT) * x.len()..],
            &symmetric_global.values[INSTRUMENT_PARAMETER_COUNT * x.len()..]
        );
    }

    #[test]
    fn fcj_geometry_and_provider_derivatives_match_centered_differences() {
        let x: Vec<f64> = (0..=2_000)
            .map(|index| 49.5 + f64::from(index) * 0.000_5)
            .collect();
        let position = [50.0];
        let intensity = [8.0];
        let support = SupportPolicy::FwhmMultiple(100.0);
        let calculate = |sample: f64, detector: f64, amplitude: f64| {
            let variance = [amplitude * 0.25];
            let zeros = [0.0];
            let ones = [1.0];
            let d_variance = [0.25];
            let arrays = CwContributionArrays {
                gaussian_variance_deg2: &variance,
                lorentzian_fwhm_deg: &zeros,
                intensity_multiplier: &ones,
                d_gaussian_variance_d_position: &zeros,
                d_lorentzian_fwhm_d_position: &zeros,
                d_intensity_multiplier_d_position: &zeros,
                d_gaussian_variance_d_parameters: &d_variance,
                d_lorentzian_fwhm_d_parameters: &zeros,
                d_intensity_multiplier_d_parameters: &zeros,
            };
            accumulate_cw_fcj_contributions_batch(
                GridView::new(&x).expect("grid"),
                &position,
                &intensity,
                instrument(),
                CwContributionsView::new(1, 1, arrays).expect("contributions"),
                FcjGeometry {
                    sample_over_radius: sample,
                    detector_over_radius: detector,
                },
                support,
            )
            .expect("FCJ contributions")
        };
        let sample = 0.013;
        let detector = 0.009;
        let amplitude = 3.0e-4;
        let baseline = calculate(sample, detector, amplitude);
        let global = baseline.derivatives.global.as_ref().expect("global");
        for (parameter, step, plus, minus) in [
            (
                INSTRUMENT_PARAMETER_COUNT,
                1.0e-7,
                calculate(sample + 1.0e-7, detector, amplitude),
                calculate(sample - 1.0e-7, detector, amplitude),
            ),
            (
                INSTRUMENT_PARAMETER_COUNT + 1,
                1.0e-7,
                calculate(sample, detector + 1.0e-7, amplitude),
                calculate(sample, detector - 1.0e-7, amplitude),
            ),
            (
                INSTRUMENT_PARAMETER_COUNT + FCJ_PARAMETER_COUNT,
                1.0e-8,
                calculate(sample, detector, amplitude + 1.0e-8),
                calculate(sample, detector, amplitude - 1.0e-8),
            ),
        ] {
            for sample_index in 0..x.len() {
                let finite_difference =
                    (plus.y[sample_index] - minus.y[sample_index]) / (2.0 * step);
                let analytical = global.values[parameter * x.len() + sample_index];
                assert!(
                    (analytical - finite_difference).abs()
                        < 2.0e-4 * finite_difference.abs().max(1.0),
                    "parameter {parameter}, sample {sample_index}: {analytical} != {finite_difference}"
                );
            }
        }
    }

    #[test]
    fn owned_contributions_validate_once_and_reborrow_without_changes() {
        let owned = OwnedCwContributions::new(
            2,
            1,
            OwnedCwContributionArrays {
                gaussian_variance_deg2: vec![0.0, 0.25],
                lorentzian_fwhm_deg: vec![0.1, 0.2],
                intensity_multiplier: vec![1.0, 0.5],
                d_gaussian_variance_d_position: vec![0.0, 0.0],
                d_lorentzian_fwhm_d_position: vec![0.0, 0.0],
                d_intensity_multiplier_d_position: vec![0.0, 0.0],
                d_gaussian_variance_d_parameters: vec![0.3, 0.4],
                d_lorentzian_fwhm_d_parameters: vec![0.0, 0.0],
                d_intensity_multiplier_d_parameters: vec![0.0, 0.0],
            },
        )
        .expect("owned contributions");

        assert_eq!(owned.reflection_count(), 2);
        assert_eq!(owned.parameter_count(), 1);
        assert_eq!(owned.as_view().parameter_count(), 1);
        assert_eq!(owned.arrays().intensity_multiplier, [1.0, 0.5]);
    }

    #[test]
    fn neutral_owned_contributions_have_valid_identity_values() {
        let owned = OwnedCwContributions::neutral(3);
        assert_eq!(owned.reflection_count(), 3);
        assert_eq!(owned.parameter_count(), 0);
        assert_eq!(owned.arrays().gaussian_variance_deg2, [0.0; 3]);
        assert_eq!(owned.arrays().intensity_multiplier, [1.0; 3]);
        assert_eq!(owned.as_view().parameter_count(), 0);
    }

    #[test]
    fn owned_contributions_reject_invalid_arrays_before_storage() {
        assert!(matches!(
            OwnedCwContributions::new(
                1,
                0,
                OwnedCwContributionArrays {
                    gaussian_variance_deg2: vec![-1.0],
                    lorentzian_fwhm_deg: vec![0.0],
                    intensity_multiplier: vec![1.0],
                    d_gaussian_variance_d_position: vec![0.0],
                    d_lorentzian_fwhm_d_position: vec![0.0],
                    d_intensity_multiplier_d_position: vec![0.0],
                    ..OwnedCwContributionArrays::default()
                },
            ),
            Err(CwContributionsError::InvalidContribution {
                quantity: "gaussian_variance_deg2",
                ..
            })
        ));
    }
}
