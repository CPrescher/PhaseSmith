//! Friedel-averaged powder intensities, preserving representative complex F.
//!
//! For real displacements and scattering factors depending on |q|, conjugating
//! the atomic scattering factors gives conj(F(-h)). Average the two squared
//! amplitudes, and their derivatives, before applying the profile kernel.
//! The individual-reflection API remains unchanged. See docs/powder-friedel.md.

use crate::structure_factor::{
    StructureFactorBatchError, StructureFactorBatchView, StructureFactorDenseResult,
    StructureFactorJvpResult, StructureFactorValues, StructureFactorVjpResult,
    calculate_structure_factor_intensity_vjp_with_context,
    calculate_structure_factor_jvp_with_context, calculate_structure_factor_selected_with_context,
    calculate_structure_factor_values_with_context,
};
use crate::{SpaceGroup, UnitCell};
use phasesmith_execution::ExecutionContext;

fn needs_pair(group: &SpaceGroup, batch: StructureFactorBatchView<'_>) -> bool {
    !group
        .rotations()
        .contains(&[[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        && batch
            .scattering_imag
            .iter()
            .chain(batch.d_scattering_imag_d_s)
            .any(|value| *value != 0.0)
}

struct ConjugateScattering {
    imag: Vec<f64>,
    d_imag: Vec<f64>,
}

impl ConjugateScattering {
    fn new(batch: StructureFactorBatchView<'_>) -> Self {
        Self {
            imag: batch.scattering_imag.iter().map(|v| -v).collect(),
            d_imag: batch.d_scattering_imag_d_s.iter().map(|v| -v).collect(),
        }
    }

    fn batch<'a>(&'a self, batch: StructureFactorBatchView<'a>) -> StructureFactorBatchView<'a> {
        StructureFactorBatchView {
            scattering_imag: &self.imag,
            d_scattering_imag_d_s: &self.d_imag,
            ..batch
        }
    }
}

fn average(target: &mut [f64], mate: &[f64]) {
    for (left, right) in target.iter_mut().zip(mate) {
        *left = 0.5 * *left + 0.5 * right;
    }
}

fn average_values(target: &mut StructureFactorValues, mate: &StructureFactorValues) {
    average(&mut target.f_squared, &mate.f_squared);
    average(&mut target.intensity, &mate.intensity);
}

/// Powder-averaged values. Complex F remains the representative's amplitude.
/// # Errors
/// Returns the individual-reflection evaluator's validation/allocation errors.
pub fn calculate_powder_structure_factor_values_with_context(
    cell: UnitCell,
    group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
    execution: &ExecutionContext,
) -> Result<StructureFactorValues, StructureFactorBatchError> {
    let mut result = calculate_structure_factor_values_with_context(cell, group, batch, execution)?;
    if needs_pair(group, batch) {
        let conjugate = ConjugateScattering::new(batch);
        let mate = calculate_structure_factor_values_with_context(
            cell,
            group,
            conjugate.batch(batch),
            execution,
        )?;
        average_values(&mut result, &mate);
    }
    Ok(result)
}

/// Powder-averaged values and selected intensity derivatives; dF stays representative.
/// # Errors
/// Returns the individual-reflection evaluator's validation/allocation errors.
pub fn calculate_powder_structure_factor_selected_with_context(
    cell: UnitCell,
    group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
    execution: &ExecutionContext,
    selected: Option<&[bool]>,
) -> Result<StructureFactorDenseResult, StructureFactorBatchError> {
    let mut result =
        calculate_structure_factor_selected_with_context(cell, group, batch, execution, selected)?;
    if needs_pair(group, batch) {
        let conjugate = ConjugateScattering::new(batch);
        let mate = calculate_structure_factor_selected_with_context(
            cell,
            group,
            conjugate.batch(batch),
            execution,
            selected,
        )?;
        average_values(&mut result.values, &mate.values);
        average(&mut result.d_intensity, &mate.d_intensity);
    }
    Ok(result)
}

/// Powder-averaged values and dense intensity derivatives; dF stays representative.
/// # Errors
/// Returns the individual-reflection evaluator's validation/allocation errors.
pub fn calculate_powder_structure_factor_dense_with_context(
    cell: UnitCell,
    group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
    execution: &ExecutionContext,
) -> Result<StructureFactorDenseResult, StructureFactorBatchError> {
    calculate_powder_structure_factor_selected_with_context(cell, group, batch, execution, None)
}

/// Powder-averaged forward intensity product; F and dF stay representative.
/// # Errors
/// Returns the individual-reflection evaluator's validation/allocation errors.
pub fn calculate_powder_structure_factor_jvp_with_context(
    cell: UnitCell,
    group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
    tangent: &[f64],
    execution: &ExecutionContext,
) -> Result<StructureFactorJvpResult, StructureFactorBatchError> {
    let mut result =
        calculate_structure_factor_jvp_with_context(cell, group, batch, tangent, execution)?;
    if needs_pair(group, batch) {
        let conjugate = ConjugateScattering::new(batch);
        let mate = calculate_structure_factor_jvp_with_context(
            cell,
            group,
            conjugate.batch(batch),
            tangent,
            execution,
        )?;
        average_values(&mut result.values, &mate.values);
        average(&mut result.d_intensity, &mate.d_intensity);
    }
    Ok(result)
}

/// Powder-averaged reverse intensity product; complex F stays representative.
/// # Errors
/// Returns the individual-reflection evaluator's validation/allocation errors.
pub fn calculate_powder_structure_factor_intensity_vjp_with_context(
    cell: UnitCell,
    group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
    weights: &[f64],
    execution: &ExecutionContext,
) -> Result<StructureFactorVjpResult, StructureFactorBatchError> {
    let mut result = calculate_structure_factor_intensity_vjp_with_context(
        cell, group, batch, weights, execution,
    )?;
    if needs_pair(group, batch) {
        let conjugate = ConjugateScattering::new(batch);
        let mate = calculate_structure_factor_intensity_vjp_with_context(
            cell,
            group,
            conjugate.batch(batch),
            weights,
            execution,
        )?;
        average_values(&mut result.values, &mate.values);
        average(&mut result.gradient, &mate.gradient);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure_factor::{
        calculate_structure_factor_dense, calculate_structure_factor_dense_with_context,
    };
    use crate::{Rational, SymmetryOperation};

    fn batch() -> StructureFactorBatchView<'static> {
        StructureFactorBatchView {
            hkl: &[[1, 2, 1], [2, -1, 3]],
            multiplicity: &[2, 4],
            fractional_xyz: &[[0.13, 0.21, 0.07], [0.31, 0.12, 0.42]],
            occupancy: &[0.8, 0.6],
            u_iso_angstrom2: &[0.01, 0.02],
            anisotropic_mask: &[false, false],
            u_aniso_cif_angstrom2: &[[0.0; 6]; 2],
            scattering_real: &[3.0, 2.0, 2.7, 1.9],
            scattering_imag: &[0.3, 0.1, 0.2, 0.05],
            d_scattering_real_d_s: &[-0.2, -0.1, -0.3, -0.15],
            d_scattering_imag_d_s: &[0.1, 0.02, -0.03, 0.07],
            correction: &[1.25, 1.4],
            d_correction_d_q_squared: &[0.1, -0.05],
            scale: 1.4,
            coordinate_tolerance: 1e-10,
        }
    }
    fn cell() -> UnitCell {
        UnitCell {
            a_angstrom: 4.3,
            b_angstrom: 5.1,
            c_angstrom: 6.2,
            alpha_deg: 78.0,
            beta_deg: 83.0,
            gamma_deg: 71.0,
        }
    }
    fn p1() -> SpaceGroup {
        SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap()
    }
    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(
                (a - b).abs() < 2e-12 * a.abs().max(b.abs()).max(1.0),
                "{a} != {b}"
            );
        }
    }

    #[test]
    fn powder_dense_selected_and_products_equal_opposite_reflection_average() {
        let batch = batch();
        let group = p1();
        let context = ExecutionContext::serial();
        let mate_batch = StructureFactorBatchView {
            hkl: &[[-1, -2, -1], [-2, 1, -3]],
            ..batch
        };
        let raw =
            calculate_structure_factor_dense_with_context(cell(), &group, batch, &context).unwrap();
        let mate =
            calculate_structure_factor_dense_with_context(cell(), &group, mate_batch, &context)
                .unwrap();
        let result =
            calculate_powder_structure_factor_dense_with_context(cell(), &group, batch, &context)
                .unwrap();
        assert_eq!(result.d_f_real, raw.d_f_real);
        assert_eq!(result.d_f_imag, raw.d_f_imag);
        let expected: Vec<_> = raw
            .d_intensity
            .iter()
            .zip(&mate.d_intensity)
            .map(|(a, b)| 0.5 * a + 0.5 * b)
            .collect();
        close(&result.d_intensity, &expected);
        assert!(
            raw.values
                .intensity
                .iter()
                .zip(&result.values.intensity)
                .any(|(a, b)| (a - b).abs() > 0.1)
        );
        let values =
            calculate_powder_structure_factor_values_with_context(cell(), &group, batch, &context)
                .unwrap();
        close(&values.intensity, &result.values.intensity);
        let count = result.layout.parameter_count();
        let tangent: Vec<_> = (0..count)
            .map(|i| if i % 2 == 0 { 0.03 } else { -0.02 })
            .collect();
        let jvp = calculate_powder_structure_factor_jvp_with_context(
            cell(),
            &group,
            batch,
            &tangent,
            &context,
        )
        .unwrap();
        let expected_jvp: Vec<_> = (0..2)
            .map(|r| {
                (0..count)
                    .map(|p| result.d_intensity[p * 2 + r] * tangent[p])
                    .sum()
            })
            .collect();
        close(&jvp.d_intensity, &expected_jvp);
        let weights = [0.7, -0.2];
        let vjp = calculate_powder_structure_factor_intensity_vjp_with_context(
            cell(),
            &group,
            batch,
            &weights,
            &context,
        )
        .unwrap();
        let expected_vjp: Vec<_> = (0..count)
            .map(|p| {
                (0..2)
                    .map(|r| result.d_intensity[p * 2 + r] * weights[r])
                    .sum()
            })
            .collect();
        close(&vjp.gradient, &expected_vjp);
        let selected: Vec<_> = (0..count).map(|p| p % 2 == 0).collect();
        let partial = calculate_powder_structure_factor_selected_with_context(
            cell(),
            &group,
            batch,
            &context,
            Some(&selected),
        )
        .unwrap();
        for (p, enabled) in selected.into_iter().enumerate() {
            if enabled {
                close(
                    &partial.d_intensity[p * 2..p * 2 + 2],
                    &result.d_intensity[p * 2..p * 2 + 2],
                );
            } else {
                assert_eq!(partial.d_intensity[p * 2..p * 2 + 2], [0.0, 0.0]);
            }
        }
    }

    #[test]
    fn zero_imaginary_values_still_average_nonzero_imaginary_slopes() {
        let batch = StructureFactorBatchView {
            scattering_imag: &[0.0; 4],
            ..batch()
        };
        let group = p1();
        let context = ExecutionContext::serial();
        assert!(needs_pair(&group, batch));
        let result =
            calculate_powder_structure_factor_dense_with_context(cell(), &group, batch, &context)
                .unwrap();
        let real = StructureFactorBatchView {
            d_scattering_imag_d_s: &[0.0; 4],
            ..batch
        };
        let expected =
            calculate_structure_factor_dense_with_context(cell(), &group, real, &context).unwrap();
        close(&result.d_intensity, &expected.d_intensity);
    }

    #[test]
    fn real_and_centrosymmetric_batches_preserve_existing_arithmetic() {
        let inversion = SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
                .unwrap(),
        ])
        .unwrap();
        let real = StructureFactorBatchView {
            scattering_imag: &[0.0; 4],
            d_scattering_imag_d_s: &[0.0; 4],
            ..batch()
        };
        for (group, batch) in [(p1(), real), (inversion, batch())] {
            assert!(!needs_pair(&group, batch));
            let raw = calculate_structure_factor_dense(cell(), &group, batch).unwrap();
            let powder = calculate_powder_structure_factor_dense_with_context(
                cell(),
                &group,
                batch,
                &ExecutionContext::serial(),
            )
            .unwrap();
            assert_eq!(raw, powder);
        }
    }
}
