//! Fused CW wavelength-component accumulation with optional FCJ asymmetry.

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
use crate::radiation::{WavelengthComponentsError, WavelengthComponentsView};
use crate::tch::TchWidths;

const DEGREE_HALF_ANGLE_TO_RADIAN: f64 = std::f64::consts::PI / 360.0;
const TWO_THETA_RADIAN_TO_DEGREE: f64 = 360.0 / std::f64::consts::PI;
const LOCAL_PARAMETER_COUNT: usize = 2;
const CW_PARAMETER_COUNT: usize = 5;
const FCJ_PARAMETER_COUNT: usize = 2;

/// Errors while accumulating a CW wavelength-component batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CwComponentsBatchError {
    /// The shared constant-wavelength instrument is invalid.
    InvalidInstrument {
        /// Instrument validation failure.
        reason: CwError,
    },
    /// The wavelength-component arrays are invalid.
    InvalidComponents {
        /// Component validation failure.
        reason: WavelengthComponentsError,
    },
    /// Component zero does not match the instrument reference wavelength.
    ReferenceWavelengthMismatch,
    /// Bragg's law cannot represent a wavelength at one reflection d-spacing.
    InvalidComponentPosition {
        /// Index of the invalid reflection.
        reflection: usize,
        /// Index of the invalid wavelength component.
        component: usize,
    },
    /// A component angle or its derived widths are invalid.
    InvalidComponentProfile {
        /// Index of the invalid reflection.
        reflection: usize,
        /// Index of the invalid wavelength component.
        component: usize,
        /// Component-specific CW failure.
        reason: CwError,
    },
    /// A component cannot be represented by the requested FCJ geometry.
    InvalidComponentGeometry {
        /// Index of the invalid reflection.
        reflection: usize,
        /// Index of the invalid wavelength component.
        component: usize,
        /// Component-specific FCJ failure.
        reason: FcjError,
    },
    /// Grid, support, or allocation validation failed.
    Accumulation {
        /// Underlying generic profile failure.
        reason: ProfileError,
    },
}

impl Display for CwComponentsBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInstrument { reason } => {
                write!(
                    formatter,
                    "invalid constant-wavelength instrument: {reason}"
                )
            }
            Self::InvalidComponents { reason } => {
                write!(formatter, "invalid wavelength components: {reason}")
            }
            Self::ReferenceWavelengthMismatch => write!(
                formatter,
                "component zero wavelength must match the instrument reference wavelength"
            ),
            Self::InvalidComponentPosition {
                reflection,
                component,
            } => write!(
                formatter,
                "wavelength component {component} is outside the Bragg domain for reflection {reflection}"
            ),
            Self::InvalidComponentProfile {
                reflection,
                component,
                reason,
            } => write!(
                formatter,
                "CW component {component} for reflection {reflection} is invalid: {reason}"
            ),
            Self::InvalidComponentGeometry {
                reflection,
                component,
                reason,
            } => write!(
                formatter,
                "FCJ component {component} for reflection {reflection} is invalid: {reason}"
            ),
            Self::Accumulation { reason } => Display::fmt(reason, formatter),
        }
    }
}

impl Error for CwComponentsBatchError {}

impl From<ProfileError> for CwComponentsBatchError {
    fn from(reason: ProfileError) -> Self {
        Self::Accumulation { reason }
    }
}

#[derive(Clone, Debug)]
struct PreparedComponent {
    cw: CwProfileParameters,
    fcj: FcjProfile,
    support_radius_deg: f64,
    d_position_d_base_position: f64,
    d_position_d_wavelength_ratio: f64,
}

struct PreparedBatch {
    components: Vec<PreparedComponent>,
    starts: Vec<usize>,
    offsets: Vec<usize>,
    normalized_weights: Vec<f64>,
}

fn normalized_component_weights(
    components: WavelengthComponentsView<'_>,
) -> Result<Vec<f64>, ProfileError> {
    let maximum = (0..components.len())
        .map(|component| components.relative_intensity(component))
        .fold(0.0, f64::max);
    let scaled_sum: f64 = (0..components.len())
        .map(|component| components.relative_intensity(component) / maximum)
        .sum();
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(components.len())
        .map_err(|_| ProfileError::AllocationOverflow)?;
    for component in 0..components.len() {
        weights.push(components.relative_intensity(component) / maximum / scaled_sum);
    }
    Ok(weights)
}

fn validate_reference_wavelength(
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponentsView<'_>,
) -> Result<(), CwComponentsBatchError> {
    let reference = components.wavelength(0);
    let scale = reference.abs().max(instrument.wavelength_angstrom.abs());
    if (reference - instrument.wavelength_angstrom).abs() > 16.0 * f64::EPSILON * scale {
        return Err(CwComponentsBatchError::ReferenceWavelengthMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_component(
    reflection: usize,
    component: usize,
    base_position_deg: f64,
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponentsView<'_>,
    geometry: FcjGeometry,
    support: SupportPolicy,
) -> Result<PreparedComponent, CwComponentsBatchError> {
    let base_theta = base_position_deg * DEGREE_HALF_ANGLE_TO_RADIAN;
    let wavelength_ratio = components.wavelength(component) / instrument.wavelength_angstrom;
    let component_sine = wavelength_ratio * base_theta.sin();
    if !component_sine.is_finite() || !(0.0..1.0).contains(&component_sine) {
        return Err(CwComponentsBatchError::InvalidComponentPosition {
            reflection,
            component,
        });
    }
    let (position_deg, d_position_d_base_position) = if component == 0 {
        (base_position_deg, 1.0)
    } else {
        let component_theta = component_sine.asin();
        (
            component_theta * TWO_THETA_RADIAN_TO_DEGREE,
            wavelength_ratio * base_theta.cos() / component_theta.cos(),
        )
    };
    let component_theta = position_deg * DEGREE_HALF_ANGLE_TO_RADIAN;
    let d_position_d_wavelength_ratio =
        TWO_THETA_RADIAN_TO_DEGREE * base_theta.sin() / component_theta.cos();
    let cw = CwProfileParameters::from_validated_instrument(position_deg, instrument).map_err(
        |reason| CwComponentsBatchError::InvalidComponentProfile {
            reflection,
            component,
            reason,
        },
    )?;
    let fcj = FcjProfile::new(
        position_deg,
        TchWidths {
            gaussian_fwhm: cw.gaussian_fwhm_deg,
            lorentzian_fwhm: cw.lorentzian_fwhm_deg,
        },
        geometry,
    )
    .map_err(|reason| CwComponentsBatchError::InvalidComponentGeometry {
        reflection,
        component,
        reason,
    })?;
    let support_radius_deg = support.radius(cw.tch.total_fwhm);
    Ok(PreparedComponent {
        cw,
        fcj,
        support_radius_deg,
        d_position_d_base_position,
        d_position_d_wavelength_ratio,
    })
}

fn prepare_batch(
    x: &[f64],
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponentsView<'_>,
    geometry: FcjGeometry,
    support: SupportPolicy,
) -> Result<PreparedBatch, CwComponentsBatchError> {
    let reflection_count = reflections.len();
    let component_count = components.len();
    let prepared_count = reflection_count
        .checked_mul(component_count)
        .ok_or(ProfileError::AllocationOverflow)?;
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(prepared_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    let mut starts = Vec::new();
    starts
        .try_reserve_exact(reflection_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    let mut offsets: Vec<usize> = Vec::new();
    offsets
        .try_reserve_exact(
            reflection_count
                .checked_add(1)
                .ok_or(ProfileError::AllocationOverflow)?,
        )
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets.push(0);

    for reflection in 0..reflection_count {
        let mut support_left = f64::INFINITY;
        let mut support_right = f64::NEG_INFINITY;
        for component in 0..component_count {
            let profile = prepare_component(
                reflection,
                component,
                reflections.position(reflection),
                instrument,
                components,
                geometry,
                support,
            )?;
            let range = profile.fcj.support_range(profile.support_radius_deg);
            support_left = support_left.min(range.left);
            support_right = support_right.max(range.right);
            prepared.push(profile);
        }
        let lower = x.partition_point(|value| *value < support_left);
        let upper = x.partition_point(|value| *value <= support_right);
        let next_offset = offsets[reflection]
            .checked_add(upper - lower)
            .ok_or(ProfileError::AllocationOverflow)?;
        starts.push(lower);
        offsets.push(next_offset);
    }
    Ok(PreparedBatch {
        components: prepared,
        starts,
        offsets,
        normalized_weights: normalized_component_weights(components)?,
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn accumulate_components(
    grid: GridView<'_>,
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponentsView<'_>,
    geometry: FcjGeometry,
    include_fcj_derivatives: bool,
    support: SupportPolicy,
) -> Result<Accumulation, CwComponentsBatchError> {
    support.validate()?;
    instrument
        .validate()
        .map_err(|reason| CwComponentsBatchError::InvalidInstrument { reason })?;
    validate_reference_wavelength(instrument, components)?;
    let x = grid.as_slice();
    let reflection_count = reflections.len();
    let component_count = components.len();
    let secondary_count = component_count - 1;
    let fcj_parameter_count = usize::from(include_fcj_derivatives) * FCJ_PARAMETER_COUNT;
    let secondary_parameter_count = secondary_count
        .checked_mul(2)
        .ok_or(ProfileError::AllocationOverflow)?;
    let global_parameter_count = CW_PARAMETER_COUNT
        .checked_add(fcj_parameter_count)
        .and_then(|count| count.checked_add(secondary_parameter_count))
        .ok_or(ProfileError::AllocationOverflow)?;
    let wavelength_parameter_start = CW_PARAMETER_COUNT + fcj_parameter_count;
    let intensity_parameter_start = wavelength_parameter_start + secondary_count;
    let prepared = prepare_batch(x, reflections, instrument, components, geometry, support)?;

    let active_sample_count = prepared.offsets.last().copied().unwrap_or(0);
    let local_value_count = active_sample_count
        .checked_mul(LOCAL_PARAMETER_COUNT)
        .ok_or(ProfileError::AllocationOverflow)?;
    let global_value_count = global_parameter_count
        .checked_mul(x.len())
        .ok_or(ProfileError::AllocationOverflow)?;
    let mut y = zeroed_f64_vec(x.len())?;
    let mut local_values = zeroed_f64_vec(local_value_count)?;
    let mut global_values = zeroed_f64_vec(global_value_count)?;
    let mut component_values = zeroed_f64_vec(component_count)?;

    for reflection in 0..reflection_count {
        let start = prepared.starts[reflection];
        let active_begin = prepared.offsets[reflection];
        let active_end = prepared.offsets[reflection + 1];
        let intensity = reflections.intensity(reflection);
        for active_index in active_begin..active_end {
            let sample = start + active_index - active_begin;
            let mut mixture_value = 0.0;
            let mut mixture_d_base_position = 0.0;
            for (component, component_value) in component_values.iter_mut().enumerate() {
                let profile = &prepared.components[reflection * component_count + component];
                let weight = prepared.normalized_weights[component];
                let point = profile
                    .fcj
                    .evaluate_supported(x[sample], profile.support_radius_deg);
                *component_value = point.value;
                mixture_value += weight * point.value;
                let d_profile_d_component_position = point.d_position
                    + point.d_gaussian_fwhm * profile.cw.d_gaussian_fwhm_d_two_theta
                    + point.d_lorentzian_fwhm * profile.cw.d_lorentzian_fwhm_d_two_theta;
                mixture_d_base_position +=
                    weight * d_profile_d_component_position * profile.d_position_d_base_position;
                for parameter in 0..CW_PARAMETER_COUNT {
                    let derivative = point.d_gaussian_fwhm
                        * profile.cw.d_gaussian_fwhm_d_instrument[parameter]
                        + point.d_lorentzian_fwhm
                            * profile.cw.d_lorentzian_fwhm_d_instrument[parameter];
                    global_values[parameter * x.len() + sample] += intensity * weight * derivative;
                }
                if include_fcj_derivatives {
                    global_values[CW_PARAMETER_COUNT * x.len() + sample] +=
                        intensity * weight * point.d_sample_over_radius;
                    global_values[(CW_PARAMETER_COUNT + 1) * x.len() + sample] +=
                        intensity * weight * point.d_detector_over_radius;
                }
                if component > 0 {
                    let parameter = wavelength_parameter_start + component - 1;
                    global_values[parameter * x.len() + sample] += intensity
                        * weight
                        * d_profile_d_component_position
                        * profile.d_position_d_wavelength_ratio;
                }
            }
            y[sample] += intensity * mixture_value;
            let local_base = active_index * LOCAL_PARAMETER_COUNT;
            local_values[local_base] = mixture_value;
            local_values[local_base + 1] = intensity * mixture_d_base_position;
            for secondary in 0..secondary_count {
                let parameter = intensity_parameter_start + secondary;
                global_values[parameter * x.len() + sample] += intensity
                    * prepared.normalized_weights[0]
                    * (component_values[secondary + 1] - mixture_value);
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

/// Accumulate a symmetric CW wavelength-component batch.
///
/// Shared derivative order is U, V, W, X, Y, then one wavelength-ratio row
/// and one intensity-ratio row for every secondary component.
///
/// # Errors
///
/// Returns [`CwComponentsBatchError`] for invalid inputs, component Bragg
/// domains, support, or allocation.
pub fn accumulate_cw_components_batch(
    grid: GridView<'_>,
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponentsView<'_>,
    support: SupportPolicy,
) -> Result<Accumulation, CwComponentsBatchError> {
    accumulate_components(
        grid,
        reflections,
        instrument,
        components,
        FcjGeometry {
            sample_over_radius: 0.0,
            detector_over_radius: 0.0,
        },
        false,
        support,
    )
}

/// Accumulate an FCJ-asymmetric CW wavelength-component batch.
///
/// Shared derivative order is U, V, W, X, Y, the two FCJ ratios, then one
/// wavelength-ratio row and one intensity-ratio row for every secondary
/// component.
///
/// # Errors
///
/// Returns [`CwComponentsBatchError`] for invalid inputs, component Bragg/FCJ
/// domains, support, or allocation.
pub fn accumulate_cw_fcj_components_batch(
    grid: GridView<'_>,
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponentsView<'_>,
    geometry: FcjGeometry,
    support: SupportPolicy,
) -> Result<Accumulation, CwComponentsBatchError> {
    accumulate_components(
        grid,
        reflections,
        instrument,
        components,
        geometry,
        true,
        support,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn single_component_matches_cw_values() {
        let x: Vec<f64> = (0..=2_000)
            .map(|index| 39.0 + f64::from(index) * 0.001)
            .collect();
        let positions = [39.8, 40.2];
        let intensities = [12.0, 7.0];
        let wavelengths = [1.540_56];
        let weights = [1.0];
        let grid = GridView::new(&x).expect("grid");
        let reflections =
            CwReflectionBatchView::new(&positions, &intensities).expect("reflections");
        let components = WavelengthComponentsView::new(&wavelengths, &weights).expect("components");
        let support = SupportPolicy::FwhmMultiple(20.0);
        let expected =
            crate::cw::accumulate_cw_batch(grid, reflections, instrument(), support).expect("CW");
        let actual =
            accumulate_cw_components_batch(grid, reflections, instrument(), components, support)
                .expect("components");
        assert_eq!(actual, expected);
    }
}
