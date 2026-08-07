//! Single-wavelength compatibility wrapper for CW plus FCJ accumulation.

use crate::cw::{ConstantWavelengthInstrument, CwReflectionBatchView};
use crate::cw_components::{CwComponentsBatchError, accumulate_cw_fcj_components_batch};
use crate::fcj::FcjGeometry;
use crate::profile::{Accumulation, GridView, SupportPolicy};
use crate::radiation::WavelengthComponentsView;

/// Error type for a single-wavelength FCJ-asymmetric CW batch.
pub type CwFcjBatchError = CwComponentsBatchError;

/// Accumulate FCJ-asymmetric single-wavelength CW reflections.
///
/// This is the one-component convenience surface over the general fused
/// wavelength-component kernel. Local derivative order is integrated intensity
/// and ideal position. Shared order is U, V, W, X, Y, sample/radius, and
/// detector/radius.
///
/// # Errors
///
/// Returns [`CwFcjBatchError`] if the instrument, a reflection profile, FCJ
/// geometry, support, or an allocation is invalid.
pub fn accumulate_cw_fcj_batch(
    grid: GridView<'_>,
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    geometry: FcjGeometry,
    support: SupportPolicy,
) -> Result<Accumulation, CwFcjBatchError> {
    let wavelengths = [instrument.wavelength_angstrom];
    let relative_intensities = [1.0];
    let components = WavelengthComponentsView::new(&wavelengths, &relative_intensities)
        .map_err(|reason| CwComponentsBatchError::InvalidComponents { reason })?;
    accumulate_cw_fcj_components_batch(grid, reflections, instrument, components, geometry, support)
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
        let x_values: Vec<f64> = (0..=2_000)
            .map(|index| 39.0 + f64::from(index) * 0.001)
            .collect();
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
