//! Composition of constant-wavelength broadening and FCJ axial asymmetry.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::cw::{
    ConstantWavelengthInstrument, CwError, CwProfileParameters, CwReflectionBatchView,
};
use crate::fcj::{FcjError, FcjGeometry, FcjProfile};
use crate::profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, ProfileError, SupportJacobian,
    SupportPolicy, zeroed_f64_vec,
};
use crate::tch::TchWidths;

const LOCAL_PARAMETER_COUNT: usize = 2;
const CW_PARAMETER_COUNT: usize = 5;
const GLOBAL_PARAMETER_COUNT: usize = 7;

/// Errors while preparing or accumulating an FCJ-asymmetric CW batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CwFcjBatchError {
    /// The shared constant-wavelength instrument is invalid.
    InvalidInstrument {
        /// Instrument validation failure.
        reason: CwError,
    },
    /// A reflection angle or its derived component widths are invalid.
    InvalidReflectionProfile {
        /// Index of the invalid reflection.
        reflection: usize,
        /// Reflection-specific CW failure.
        reason: CwError,
    },
    /// A reflection cannot be represented by the requested FCJ geometry.
    InvalidReflectionGeometry {
        /// Index of the invalid reflection.
        reflection: usize,
        /// Reflection-specific FCJ failure.
        reason: FcjError,
    },
    /// Grid, support, or allocation validation failed.
    Accumulation {
        /// Underlying generic profile failure.
        reason: ProfileError,
    },
}

impl Display for CwFcjBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInstrument { reason } => {
                write!(
                    formatter,
                    "invalid constant-wavelength instrument: {reason}"
                )
            }
            Self::InvalidReflectionProfile { reflection, reason } => write!(
                formatter,
                "constant-wavelength reflection {reflection} is invalid: {reason}"
            ),
            Self::InvalidReflectionGeometry { reflection, reason } => {
                write!(
                    formatter,
                    "FCJ reflection {reflection} is invalid: {reason}"
                )
            }
            Self::Accumulation { reason } => Display::fmt(reason, formatter),
        }
    }
}

impl Error for CwFcjBatchError {}

impl From<ProfileError> for CwFcjBatchError {
    fn from(reason: ProfileError) -> Self {
        Self::Accumulation { reason }
    }
}

#[derive(Clone, Debug)]
struct PreparedReflection {
    cw: CwProfileParameters,
    fcj: FcjProfile,
    support_radius_deg: f64,
}

type PreparedBatch = (Vec<PreparedReflection>, Vec<usize>, Vec<usize>);

fn prepare_batch(
    x: &[f64],
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    geometry: FcjGeometry,
    support: SupportPolicy,
) -> Result<PreparedBatch, CwFcjBatchError> {
    let reflection_count = reflections.len();
    let mut prepared = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    prepared
        .try_reserve_exact(reflection_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    starts
        .try_reserve_exact(reflection_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets
        .try_reserve_exact(
            reflection_count
                .checked_add(1)
                .ok_or(ProfileError::AllocationOverflow)?,
        )
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets.push(0);

    for reflection in 0..reflection_count {
        let position = reflections.position(reflection);
        let cw = CwProfileParameters::from_validated_instrument(position, instrument)
            .map_err(|reason| CwFcjBatchError::InvalidReflectionProfile { reflection, reason })?;
        let fcj = FcjProfile::new(
            position,
            TchWidths {
                gaussian_fwhm: cw.gaussian_fwhm_deg,
                lorentzian_fwhm: cw.lorentzian_fwhm_deg,
            },
            geometry,
        )
        .map_err(|reason| CwFcjBatchError::InvalidReflectionGeometry { reflection, reason })?;
        let support_radius_deg = support.radius(cw.tch.total_fwhm);
        let range = fcj.support_range(support_radius_deg);
        let lower = x.partition_point(|value| *value < range.left);
        let upper = x.partition_point(|value| *value <= range.right);
        let next_offset = offsets[reflection]
            .checked_add(upper - lower)
            .ok_or(ProfileError::AllocationOverflow)?;
        prepared.push(PreparedReflection {
            cw,
            fcj,
            support_radius_deg,
        });
        starts.push(lower);
        offsets.push(next_offset);
    }
    Ok((prepared, starts, offsets))
}

/// Accumulate FCJ-asymmetric CW reflections in one deterministic native pass.
///
/// Local derivative order is integrated intensity and ideal reflection
/// position. Dense shared derivative order is U, V, W, X, Y,
/// `sample_over_radius`, and `detector_over_radius`.
///
/// # Errors
///
/// Returns [`CwFcjBatchError`] if the instrument, a derived reflection
/// profile, FCJ geometry, support, or an allocation is invalid.
pub fn accumulate_cw_fcj_batch(
    grid: GridView<'_>,
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    geometry: FcjGeometry,
    support: SupportPolicy,
) -> Result<Accumulation, CwFcjBatchError> {
    support.validate()?;
    instrument
        .validate()
        .map_err(|reason| CwFcjBatchError::InvalidInstrument { reason })?;
    let x = grid.as_slice();
    let reflection_count = reflections.len();
    let (prepared, starts, offsets) = prepare_batch(x, reflections, instrument, geometry, support)?;

    let active_sample_count = offsets.last().copied().unwrap_or(0);
    let local_value_count = active_sample_count
        .checked_mul(LOCAL_PARAMETER_COUNT)
        .ok_or(ProfileError::AllocationOverflow)?;
    let global_value_count = GLOBAL_PARAMETER_COUNT
        .checked_mul(x.len())
        .ok_or(ProfileError::AllocationOverflow)?;
    let mut y = zeroed_f64_vec(x.len())?;
    let mut local_values = zeroed_f64_vec(local_value_count)?;
    let mut global_values = zeroed_f64_vec(global_value_count)?;

    for reflection in 0..reflection_count {
        let start = starts[reflection];
        let active_begin = offsets[reflection];
        let active_end = offsets[reflection + 1];
        let profile = &prepared[reflection];
        let intensity = reflections.intensity(reflection);
        for active_index in active_begin..active_end {
            let sample = start + active_index - active_begin;
            let point = profile
                .fcj
                .evaluate_supported(x[sample], profile.support_radius_deg);
            y[sample] += intensity * point.value;
            let local_base = active_index * LOCAL_PARAMETER_COUNT;
            local_values[local_base] = point.value;
            local_values[local_base + 1] = intensity
                * (point.d_position
                    + point.d_gaussian_fwhm * profile.cw.d_gaussian_fwhm_d_two_theta
                    + point.d_lorentzian_fwhm * profile.cw.d_lorentzian_fwhm_d_two_theta);
            for parameter in 0..CW_PARAMETER_COUNT {
                let derivative = point.d_gaussian_fwhm
                    * profile.cw.d_gaussian_fwhm_d_instrument[parameter]
                    + point.d_lorentzian_fwhm
                        * profile.cw.d_lorentzian_fwhm_d_instrument[parameter];
                global_values[parameter * x.len() + sample] += intensity * derivative;
            }
            global_values[CW_PARAMETER_COUNT * x.len() + sample] +=
                intensity * point.d_sample_over_radius;
            global_values[(CW_PARAMETER_COUNT + 1) * x.len() + sample] +=
                intensity * point.d_detector_over_radius;
        }
    }

    Ok(Accumulation {
        y,
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts,
                offsets,
                values: local_values,
                parameter_count: LOCAL_PARAMETER_COUNT,
            },
            global: Some(DenseJacobian {
                values: global_values,
                parameter_count: GLOBAL_PARAMETER_COUNT,
                sample_count: x.len(),
            }),
        },
        sample_count: x.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instrument() -> ConstantWavelengthInstrument {
        ConstantWavelengthInstrument {
            wavelength_angstrom: 1.5406,
            u_deg2: 2.0e-4,
            v_deg2: -1.0e-4,
            w_deg2: 1.0e-4,
            x_deg: 1.0e-3,
            y_deg: 2.0e-3,
        }
    }

    #[test]
    fn zero_geometry_matches_symmetric_cw_values() {
        let x_values: Vec<f64> = (0..=2_000).map(|i| 39.0 + f64::from(i) * 0.001).collect();
        let positions = [39.8, 40.2];
        let intensities = [12.0, 7.0];
        let grid = GridView::new(&x_values).expect("grid");
        let reflections =
            CwReflectionBatchView::new(&positions, &intensities).expect("reflections");
        let support = SupportPolicy::FwhmMultiple(20.0);
        let symmetric = crate::cw::accumulate_cw_batch(grid, reflections, instrument(), support)
            .expect("symmetric");
        let asymmetric = accumulate_cw_fcj_batch(
            grid,
            reflections,
            instrument(),
            FcjGeometry {
                sample_over_radius: 0.0,
                detector_over_radius: 0.0,
            },
            support,
        )
        .expect("zero FCJ");
        assert_eq!(asymmetric.y, symmetric.y);
        assert_eq!(
            asymmetric.derivatives.local.values,
            symmetric.derivatives.local.values
        );
        assert_eq!(
            &asymmetric.derivatives.global.expect("global").values[..5 * x_values.len()],
            symmetric.derivatives.global.expect("global").values
        );
    }
}
