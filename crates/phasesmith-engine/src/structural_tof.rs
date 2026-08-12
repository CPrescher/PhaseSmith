//! Fused neutron structural time-of-flight pattern calculation.

use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::{
    Accumulation, GridView, TofBankGeometry, TofError, TofInstrument,
    accumulate_tof_batch_with_context,
};
use phasesmith_crystallography::{
    CELL_PARAMETER_COUNT, CellError, IntegratedIntensityCorrection,
    IntegratedIntensityCorrectionError, IntegratedIntensityCorrectionModel,
    PreparedNeutronScattering, ScatteringBatch, ScatteringError, SpaceGroup,
    StructureFactorBatchError, StructureFactorBatchView, StructureFactorValues, UnitCell,
    calculate_structure_factor_dense_with_context,
    calculate_structure_factor_intensity_vjp_with_context,
    calculate_structure_factor_jvp_with_context, calculate_structure_factor_values_with_context,
};
use phasesmith_execution::ExecutionContext;

/// Borrowed neutron structural and bank inputs for one TOF pattern.
#[derive(Clone, Copy, Debug)]
pub struct StructuralTofInputView<'a> {
    /// Strictly increasing sample centers in microseconds.
    pub tof_us: &'a [f64],
    /// Canonical Miller indices.
    pub hkl: &'a [[i32; 3]],
    /// Powder multiplicities.
    pub multiplicity: &'a [usize],
    /// Asymmetric-unit fractional coordinates.
    pub fractional_xyz: &'a [[f64; 3]],
    /// Asymmetric-site occupancies.
    pub occupancy: &'a [f64],
    /// Isotropic displacement values in square ångströms.
    pub u_iso_angstrom2: &'a [f64],
    /// True for sites described by fixed CIF U tensors.
    pub anisotropic_mask: &'a [bool],
    /// CIF U tensors in component order `11,22,33,23,13,12`.
    pub u_aniso_cif_angstrom2: &'a [[f64; 6]],
    /// Exact bound-coherent neutron table key for every asymmetric site.
    pub scattering_species: &'a [&'a str],
    /// Non-negative structural phase scale.
    pub scale: f64,
    /// Fixed symmetry-expansion deduplication tolerance.
    pub coordinate_tolerance: f64,
    /// Explicit neutral or one-dimensional neutron TOF correction.
    pub correction_model: IntegratedIntensityCorrectionModel,
    /// Fixed focused-bank scattering geometry.
    pub bank_geometry: TofBankGeometry,
    /// Bank-local TOF calibration and profile coefficients.
    pub instrument: TofInstrument,
    /// Inclusive finite support radius in multiples of total FWHM.
    pub support_fwhm: f64,
    /// Exponential quadrature tail cutoff `exp(-tail_log)`.
    pub tail_log: f64,
}

/// Structural reflection values and fused TOF accumulation.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofResult {
    /// Nuclear structure factors and corrected integrated intensities.
    pub structure_factors: StructureFactorValues,
    /// Reflection d-spacings in ångströms.
    pub d_spacing_angstrom: Vec<f64>,
    /// Support-limited TOF values and local/global profile derivatives.
    pub accumulation: Accumulation,
}

/// Values and a parameter-major structural pattern Jacobian.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofDenseResult {
    /// Calculated structural TOF pattern.
    pub result: StructuralTofResult,
    /// Row-major structural Jacobian with shape `(parameter_count, sample_count)`.
    pub d_y: Vec<f64>,
    /// Stable structural parameter count.
    pub parameter_count: usize,
}

/// Values and one structural forward derivative product.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofJvpResult {
    /// Calculated structural TOF pattern.
    pub result: StructuralTofResult,
    /// Directional derivative of pattern samples.
    pub d_y: Vec<f64>,
    /// Directional derivative of integrated reflection intensities.
    pub d_integrated_intensity: Vec<f64>,
    /// Directional derivative of d-spacing.
    pub d_spacing_angstrom: Vec<f64>,
}

/// Values and one structural reverse derivative product.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofVjpResult {
    /// Calculated structural TOF pattern.
    pub result: StructuralTofResult,
    /// Pattern-Jacobian transpose product in structural parameter order.
    pub gradient: Vec<f64>,
}

/// Invalid fused structural TOF request.
#[derive(Debug)]
pub enum StructuralTofError {
    /// The unit cell is invalid.
    Cell(CellError),
    /// Scattering species count does not match the asymmetric-site count.
    SpeciesLengthMismatch,
    /// Built-in neutron scattering preparation or evaluation failed.
    Scattering(ScatteringError),
    /// Integrated-intensity correction evaluation failed.
    Correction(IntegratedIntensityCorrectionError),
    /// Correction family or bank angle is incompatible with this request.
    IncompatibleCorrectionModel,
    /// General-symmetry structural intensity failed.
    StructureFactor(StructureFactorBatchError),
    /// TOF profile or bank geometry evaluation failed.
    Tof(TofError),
    /// A dense output-size calculation overflowed.
    AllocationOverflow,
    /// Pattern reverse weights do not match the sample count.
    PatternWeightLengthMismatch,
    /// A pattern reverse weight is non-finite.
    NonFinitePatternWeight,
}

impl Display for StructuralTofError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cell(error) => Display::fmt(error, formatter),
            Self::SpeciesLengthMismatch => formatter
                .write_str("neutron scattering species must contain one key per asymmetric site"),
            Self::Scattering(error) => Display::fmt(error, formatter),
            Self::Correction(error) => Display::fmt(error, formatter),
            Self::IncompatibleCorrectionModel => formatter.write_str(
                "structural TOF requires neutral correction or a TOF neutron Lorentz angle exactly matching the bank geometry",
            ),
            Self::StructureFactor(error) => Display::fmt(error, formatter),
            Self::Tof(error) => Display::fmt(error, formatter),
            Self::AllocationOverflow => formatter.write_str("structural TOF output allocation overflow"),
            Self::PatternWeightLengthMismatch => {
                formatter.write_str("pattern reverse weights must match the TOF sample count")
            }
            Self::NonFinitePatternWeight => formatter.write_str("pattern reverse weights must be finite"),
        }
    }
}

impl Error for StructuralTofError {}

struct PreparedTofNumerics {
    scattering: ScatteringBatch,
    correction: IntegratedIntensityCorrection,
    d_spacing: Vec<f64>,
    d_spacing_d_cell: Vec<[f64; CELL_PARAMETER_COUNT]>,
}

impl PreparedTofNumerics {
    fn structure_batch<'a>(
        &'a self,
        input: &StructuralTofInputView<'a>,
    ) -> StructureFactorBatchView<'a> {
        StructureFactorBatchView {
            hkl: input.hkl,
            multiplicity: input.multiplicity,
            fractional_xyz: input.fractional_xyz,
            occupancy: input.occupancy,
            u_iso_angstrom2: input.u_iso_angstrom2,
            anisotropic_mask: input.anisotropic_mask,
            u_aniso_cif_angstrom2: input.u_aniso_cif_angstrom2,
            scattering_real: &self.scattering.real,
            scattering_imag: &self.scattering.imag,
            d_scattering_real_d_s: &self.scattering.d_real_d_s,
            d_scattering_imag_d_s: &self.scattering.d_imag_d_s,
            correction: &self.correction.values,
            d_correction_d_q_squared: &self.correction.d_values_d_q_squared,
            scale: input.scale,
            coordinate_tolerance: input.coordinate_tolerance,
        }
    }
}

/// Calculate one neutron structural TOF pattern.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid structural, correction, bank,
/// profile, support, or grid inputs.
pub fn calculate_structural_tof_pattern(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
) -> Result<StructuralTofResult, StructuralTofError> {
    calculate_structural_tof_pattern_with_context(
        cell,
        space_group,
        input,
        &ExecutionContext::serial(),
    )
}

/// Calculate values with an explicit bounded execution context.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid inputs.
pub fn calculate_structural_tof_pattern_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
    execution: &ExecutionContext,
) -> Result<StructuralTofResult, StructuralTofError> {
    let prepared = prepare(cell, input)?;
    let values = calculate_structure_factor_values_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        execution,
    )
    .map_err(StructuralTofError::StructureFactor)?;
    assemble_result(input, prepared, values, execution)
}

/// Calculate values and a dense structural TOF Jacobian.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid inputs or allocation overflow.
pub fn calculate_structural_tof_pattern_dense(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
) -> Result<StructuralTofDenseResult, StructuralTofError> {
    calculate_structural_tof_pattern_dense_with_context(
        cell,
        space_group,
        input,
        &ExecutionContext::serial(),
    )
}

/// Calculate a dense structural TOF Jacobian with a bounded context.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid inputs or allocation overflow.
pub fn calculate_structural_tof_pattern_dense_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
    execution: &ExecutionContext,
) -> Result<StructuralTofDenseResult, StructuralTofError> {
    let prepared = prepare(cell, input)?;
    let structural = calculate_structure_factor_dense_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        execution,
    )
    .map_err(StructuralTofError::StructureFactor)?;
    let parameter_count = structural.layout.parameter_count();
    let sample_count = input.tof_us.len();
    let element_count = parameter_count
        .checked_mul(sample_count)
        .ok_or(StructuralTofError::AllocationOverflow)?;
    let accumulation = accumulate(
        input,
        &prepared.d_spacing,
        &structural.values.intensity,
        execution,
    )?;
    let mut d_y = vec![0.0; element_count];
    let reflection_count = input.hkl.len();
    let local = &accumulation.derivatives.local;
    for reflection in 0..reflection_count {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        for active in begin..end {
            let sample = local.starts[reflection] + active - begin;
            let local_base = 2 * active;
            for parameter in 0..parameter_count {
                let d_spacing = if parameter < CELL_PARAMETER_COUNT {
                    prepared.d_spacing_d_cell[reflection][parameter]
                } else {
                    0.0
                };
                d_y[parameter * sample_count + sample] += local.values[local_base]
                    * structural.d_intensity[parameter * reflection_count + reflection]
                    + local.values[local_base + 1] * d_spacing;
            }
        }
    }
    Ok(StructuralTofDenseResult {
        result: StructuralTofResult {
            structure_factors: structural.values,
            d_spacing_angstrom: prepared.d_spacing,
            accumulation,
        },
        d_y,
        parameter_count,
    })
}

/// Calculate values and one structural forward derivative product.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid inputs or tangent shape.
pub fn calculate_structural_tof_pattern_jvp(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
    tangent: &[f64],
) -> Result<StructuralTofJvpResult, StructuralTofError> {
    calculate_structural_tof_pattern_jvp_with_context(
        cell,
        space_group,
        input,
        tangent,
        &ExecutionContext::serial(),
    )
}

/// Calculate a structural forward product with a bounded context.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid inputs or tangent shape.
pub fn calculate_structural_tof_pattern_jvp_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
    tangent: &[f64],
    execution: &ExecutionContext,
) -> Result<StructuralTofJvpResult, StructuralTofError> {
    let prepared = prepare(cell, input)?;
    let structural = calculate_structure_factor_jvp_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        tangent,
        execution,
    )
    .map_err(StructuralTofError::StructureFactor)?;
    let d_spacing_angstrom = prepared
        .d_spacing_d_cell
        .iter()
        .map(|derivatives| {
            derivatives
                .iter()
                .zip(&tangent[..CELL_PARAMETER_COUNT])
                .map(|(derivative, direction)| derivative * direction)
                .sum()
        })
        .collect::<Vec<_>>();
    let accumulation = accumulate(
        input,
        &prepared.d_spacing,
        &structural.values.intensity,
        execution,
    )?;
    let d_y = chain_jvp(&accumulation, &structural.d_intensity, &d_spacing_angstrom);
    Ok(StructuralTofJvpResult {
        result: StructuralTofResult {
            structure_factors: structural.values,
            d_spacing_angstrom: prepared.d_spacing,
            accumulation,
        },
        d_y,
        d_integrated_intensity: structural.d_intensity,
        d_spacing_angstrom,
    })
}

/// Calculate values and one structural reverse derivative product.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid inputs or sample weights.
pub fn calculate_structural_tof_pattern_vjp(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
    sample_weights: &[f64],
) -> Result<StructuralTofVjpResult, StructuralTofError> {
    calculate_structural_tof_pattern_vjp_with_context(
        cell,
        space_group,
        input,
        sample_weights,
        &ExecutionContext::serial(),
    )
}

/// Calculate a structural reverse product with a bounded context.
///
/// # Errors
///
/// Returns [`StructuralTofError`] for invalid inputs or sample weights.
pub fn calculate_structural_tof_pattern_vjp_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralTofInputView<'_>,
    sample_weights: &[f64],
    execution: &ExecutionContext,
) -> Result<StructuralTofVjpResult, StructuralTofError> {
    if sample_weights.len() != input.tof_us.len() {
        return Err(StructuralTofError::PatternWeightLengthMismatch);
    }
    if sample_weights.iter().any(|value| !value.is_finite()) {
        return Err(StructuralTofError::NonFinitePatternWeight);
    }
    let prepared = prepare(cell, input)?;
    let values = calculate_structure_factor_values_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        execution,
    )
    .map_err(StructuralTofError::StructureFactor)?;
    let accumulation = accumulate(input, &prepared.d_spacing, &values.intensity, execution)?;
    let (intensity_weights, d_spacing_weights) = local_transpose(&accumulation, sample_weights);
    let mut structural = calculate_structure_factor_intensity_vjp_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        &intensity_weights,
        execution,
    )
    .map_err(StructuralTofError::StructureFactor)?;
    for (reflection, weight) in d_spacing_weights.into_iter().enumerate() {
        for parameter in 0..CELL_PARAMETER_COUNT {
            structural.gradient[parameter] +=
                weight * prepared.d_spacing_d_cell[reflection][parameter];
        }
    }
    Ok(StructuralTofVjpResult {
        result: StructuralTofResult {
            structure_factors: values,
            d_spacing_angstrom: prepared.d_spacing,
            accumulation,
        },
        gradient: structural.gradient,
    })
}

fn prepare(
    cell: UnitCell,
    input: &StructuralTofInputView<'_>,
) -> Result<PreparedTofNumerics, StructuralTofError> {
    if input.scattering_species.len() != input.fractional_xyz.len() {
        return Err(StructuralTofError::SpeciesLengthMismatch);
    }
    input
        .bank_geometry
        .validate()
        .map_err(StructuralTofError::Tof)?;
    validate_correction(input.correction_model, input.bank_geometry)?;
    let geometry = cell.geometry().map_err(StructuralTofError::Cell)?;
    let mut q_squared = Vec::with_capacity(input.hkl.len());
    let mut d_spacing = Vec::with_capacity(input.hkl.len());
    let mut d_spacing_d_cell = Vec::with_capacity(input.hkl.len());
    for &hkl in input.hkl {
        let (q, d_q) = geometry.q_squared_and_derivatives(hkl);
        if !q.is_finite() || q <= 0.0 {
            return Err(StructuralTofError::StructureFactor(
                StructureFactorBatchError::ZeroReflection,
            ));
        }
        let d = q.sqrt().recip();
        q_squared.push(q);
        d_spacing.push(d);
        d_spacing_d_cell.push(d_q.map(|derivative| -0.5 * d.powi(3) * derivative));
    }
    let s = q_squared
        .iter()
        .map(|value| 0.5 * value.sqrt())
        .collect::<Vec<_>>();
    let scattering = PreparedNeutronScattering::new(input.scattering_species.iter().copied())
        .and_then(|model| model.evaluate(&s))
        .map_err(StructuralTofError::Scattering)?;
    let correction = input
        .correction_model
        .evaluate(&q_squared)
        .map_err(StructuralTofError::Correction)?;
    Ok(PreparedTofNumerics {
        scattering,
        correction,
        d_spacing,
        d_spacing_d_cell,
    })
}

fn validate_correction(
    correction: IntegratedIntensityCorrectionModel,
    geometry: TofBankGeometry,
) -> Result<(), StructuralTofError> {
    match correction {
        IntegratedIntensityCorrectionModel::Neutral => Ok(()),
        IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg }
            if two_theta_deg.to_bits() == geometry.two_theta_deg.to_bits() =>
        {
            Ok(())
        }
        _ => Err(StructuralTofError::IncompatibleCorrectionModel),
    }
}

fn accumulate(
    input: &StructuralTofInputView<'_>,
    d_spacing: &[f64],
    intensity: &[f64],
    execution: &ExecutionContext,
) -> Result<Accumulation, StructuralTofError> {
    let grid = GridView::new(input.tof_us)
        .map_err(|reason| StructuralTofError::Tof(TofError::Profile { reason }))?;
    accumulate_tof_batch_with_context(
        grid,
        d_spacing,
        intensity,
        input.instrument,
        input.support_fwhm,
        input.tail_log,
        execution,
    )
    .map_err(StructuralTofError::Tof)
}

fn assemble_result(
    input: &StructuralTofInputView<'_>,
    prepared: PreparedTofNumerics,
    values: StructureFactorValues,
    execution: &ExecutionContext,
) -> Result<StructuralTofResult, StructuralTofError> {
    let accumulation = accumulate(input, &prepared.d_spacing, &values.intensity, execution)?;
    Ok(StructuralTofResult {
        structure_factors: values,
        d_spacing_angstrom: prepared.d_spacing,
        accumulation,
    })
}

fn chain_jvp(accumulation: &Accumulation, d_intensity: &[f64], d_spacing: &[f64]) -> Vec<f64> {
    let local = &accumulation.derivatives.local;
    let mut result = vec![0.0; accumulation.sample_count];
    for reflection in 0..local.peak_count() {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        for active in begin..end {
            let sample = local.starts[reflection] + active - begin;
            let base = 2 * active;
            result[sample] += local.values[base] * d_intensity[reflection]
                + local.values[base + 1] * d_spacing[reflection];
        }
    }
    result
}

fn local_transpose(accumulation: &Accumulation, weights: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let local = &accumulation.derivatives.local;
    let mut intensity = vec![0.0; local.peak_count()];
    let mut d_spacing = vec![0.0; local.peak_count()];
    for reflection in 0..local.peak_count() {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        for active in begin..end {
            let sample = local.starts[reflection] + active - begin;
            let base = 2 * active;
            intensity[reflection] += weights[sample] * local.values[base];
            d_spacing[reflection] += weights[sample] * local.values[base + 1];
        }
    }
    (intensity, d_spacing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_crystallography::{Rational, SymmetryOperation};

    const HKL: [[i32; 3]; 3] = [[1, 0, 1], [2, 1, 1], [1, 2, 3]];
    const MULTIPLICITY: [usize; 3] = [2, 4, 2];
    const XYZ: [[f64; 3]; 2] = [[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]];
    const OCCUPANCY: [f64; 2] = [0.82, 0.55];
    const U_ISO: [f64; 2] = [0.012, 0.018];
    const ANISOTROPIC: [bool; 2] = [false, false];
    const U_ANISO: [[f64; 6]; 2] = [[0.0; 6]; 2];
    const SPECIES: [&str; 2] = ["Ni", "O"];

    fn cell() -> UnitCell {
        UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 5.1,
            c_angstrom: 6.2,
            alpha_deg: 82.0,
            beta_deg: 87.0,
            gamma_deg: 74.0,
        }
    }

    fn group() -> SpaceGroup {
        SpaceGroup::new(vec![
            SymmetryOperation::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [Rational::zero(); 3])
                .unwrap(),
        ])
        .unwrap()
    }

    fn instrument() -> TofInstrument {
        TofInstrument {
            zero_us: 1.2,
            difc_us_per_angstrom: 5_000.0,
            difa_us_per_angstrom2: 0.2,
            difb_us_angstrom: 0.0,
            alpha_coefficient: 0.2,
            beta0_per_us: 0.03,
            beta1_angstrom4_per_us: 0.001,
            betaq_angstrom2_per_us: 0.0,
            sigma0_us2: 25.0,
            sigma1_us2_per_angstrom2: 4.0,
            sigma2_us2_per_angstrom4: 0.1,
            sigmaq_us2_per_angstrom: 0.0,
            x_us_per_angstrom: 1.0,
            y_us_per_angstrom2: 0.1,
            z_us: 0.5,
        }
    }

    fn grid() -> Vec<f64> {
        (0..2_401)
            .map(|index| 1_000.0 + f64::from(index) * 10.0)
            .collect()
    }

    fn input<'a>(
        tof_us: &'a [f64],
        xyz: &'a [[f64; 3]],
        occupancy: &'a [f64],
        u_iso: &'a [f64],
        scale: f64,
    ) -> StructuralTofInputView<'a> {
        StructuralTofInputView {
            tof_us,
            hkl: &HKL,
            multiplicity: &MULTIPLICITY,
            fractional_xyz: xyz,
            occupancy,
            u_iso_angstrom2: u_iso,
            anisotropic_mask: &ANISOTROPIC,
            u_aniso_cif_angstrom2: &U_ANISO,
            scattering_species: &SPECIES,
            scale,
            coordinate_tolerance: 1.0e-10,
            correction_model: IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: 88.05,
            },
            bank_geometry: TofBankGeometry {
                two_theta_deg: 88.05,
            },
            instrument: instrument(),
            support_fwhm: 20.0,
            tail_log: 20.0,
        }
    }

    fn values(
        selected_cell: UnitCell,
        tof_us: &[f64],
        xyz: &[[f64; 3]],
        occupancy: &[f64],
        u_iso: &[f64],
        scale: f64,
    ) -> Vec<f64> {
        calculate_structural_tof_pattern(
            selected_cell,
            &group(),
            &input(tof_us, xyz, occupancy, u_iso, scale),
        )
        .unwrap()
        .accumulation
        .y
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn dense_structural_rows_match_centered_pattern_differences() {
        let tof_us = grid();
        let dense = calculate_structural_tof_pattern_dense(
            cell(),
            &group(),
            &input(&tof_us, &XYZ, &OCCUPANCY, &U_ISO, 1.3),
        )
        .unwrap();
        assert_eq!(dense.parameter_count, 17);
        assert_eq!(
            dense
                .result
                .accumulation
                .derivatives
                .global
                .as_ref()
                .unwrap()
                .parameter_count,
            15
        );
        let mut support_boundary = vec![false; tof_us.len()];
        let local = &dense.result.accumulation.derivatives.local;
        for reflection in 0..local.peak_count() {
            let count = local.offsets[reflection + 1] - local.offsets[reflection];
            let start = local.starts[reflection];
            for offset in 0..3.min(count) {
                support_boundary[start + offset] = true;
                support_boundary[start + count - 1 - offset] = true;
            }
        }
        for parameter in 0..dense.parameter_count {
            let step = 1.0e-6;
            let mut plus_cell = cell();
            let mut minus_cell = cell();
            let mut plus_xyz = XYZ;
            let mut minus_xyz = XYZ;
            let mut plus_occupancy = OCCUPANCY;
            let mut minus_occupancy = OCCUPANCY;
            let mut plus_u_iso = U_ISO;
            let mut minus_u_iso = U_ISO;
            let mut plus_scale = 1.3;
            let mut minus_scale = 1.3;
            match parameter {
                0 => {
                    plus_cell.a_angstrom += step;
                    minus_cell.a_angstrom -= step;
                }
                1 => {
                    plus_cell.b_angstrom += step;
                    minus_cell.b_angstrom -= step;
                }
                2 => {
                    plus_cell.c_angstrom += step;
                    minus_cell.c_angstrom -= step;
                }
                3 => {
                    plus_cell.alpha_deg += step;
                    minus_cell.alpha_deg -= step;
                }
                4 => {
                    plus_cell.beta_deg += step;
                    minus_cell.beta_deg -= step;
                }
                5 => {
                    plus_cell.gamma_deg += step;
                    minus_cell.gamma_deg -= step;
                }
                6..=11 => {
                    let local = parameter - CELL_PARAMETER_COUNT;
                    plus_xyz[local / 3][local % 3] += step;
                    minus_xyz[local / 3][local % 3] -= step;
                }
                12..=13 => {
                    plus_occupancy[parameter - 12] += step;
                    minus_occupancy[parameter - 12] -= step;
                }
                14..=15 => {
                    plus_u_iso[parameter - 14] += step;
                    minus_u_iso[parameter - 14] -= step;
                }
                16 => {
                    plus_scale += step;
                    minus_scale -= step;
                }
                _ => unreachable!(),
            }
            let plus = values(
                plus_cell,
                &tof_us,
                &plus_xyz,
                &plus_occupancy,
                &plus_u_iso,
                plus_scale,
            );
            let minus = values(
                minus_cell,
                &tof_us,
                &minus_xyz,
                &minus_occupancy,
                &minus_u_iso,
                minus_scale,
            );
            for sample in 0..tof_us.len() {
                if support_boundary[sample] {
                    continue;
                }
                let finite = (plus[sample] - minus[sample]) / (2.0 * step);
                let analytical = dense.d_y[parameter * tof_us.len() + sample];
                let relative_tolerance = if parameter < CELL_PARAMETER_COUNT {
                    // The cell chain includes the finite-quadrature TOF
                    // d-spacing derivative. Its stable centered-difference
                    // floor is about 5e-5 for this deliberately asymmetric
                    // profile; direct structural rows remain much tighter.
                    2.0e-4
                } else {
                    3.0e-5
                };
                assert!(
                    (analytical - finite).abs() <= relative_tolerance * finite.abs().max(1.0),
                    "parameter={parameter} sample={sample} analytical={analytical} finite={finite} error={}",
                    (analytical - finite).abs(),
                );
            }
        }
    }

    #[test]
    fn jvp_vjp_match_dense_and_are_adjoint_consistent() {
        let tof_us = grid();
        let request = input(&tof_us, &XYZ, &OCCUPANCY, &U_ISO, 1.3);
        let dense = calculate_structural_tof_pattern_dense(cell(), &group(), &request).unwrap();
        let tangent = (0..dense.parameter_count)
            .map(|index| f64::from(u32::try_from(index + 1).unwrap()) * 1.0e-5)
            .collect::<Vec<_>>();
        let jvp =
            calculate_structural_tof_pattern_jvp(cell(), &group(), &request, &tangent).unwrap();
        for sample in 0..tof_us.len() {
            let expected = (0..dense.parameter_count)
                .map(|parameter| dense.d_y[parameter * tof_us.len() + sample] * tangent[parameter])
                .sum::<f64>();
            assert!((jvp.d_y[sample] - expected).abs() < 2.0e-12 * expected.abs().max(1.0));
        }
        let weights = tof_us
            .iter()
            .map(|value| (value * 1.0e-3).sin())
            .collect::<Vec<_>>();
        let vjp =
            calculate_structural_tof_pattern_vjp(cell(), &group(), &request, &weights).unwrap();
        for parameter in 0..dense.parameter_count {
            let expected = (0..tof_us.len())
                .map(|sample| dense.d_y[parameter * tof_us.len() + sample] * weights[sample])
                .sum::<f64>();
            assert!((vjp.gradient[parameter] - expected).abs() < 3.0e-11 * expected.abs().max(1.0));
        }
        let forward = jvp
            .d_y
            .iter()
            .zip(&weights)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let reverse = tangent
            .iter()
            .zip(&vjp.gradient)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        assert!((forward - reverse).abs() < 2.0e-11 * forward.abs().max(1.0));
    }

    #[test]
    fn correction_angle_and_reverse_weight_boundaries_are_explicit() {
        let tof_us = grid();
        let mut request = input(&tof_us, &XYZ, &OCCUPANCY, &U_ISO, 1.3);
        request.correction_model = IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
            two_theta_deg: 90.0,
        };
        assert!(matches!(
            calculate_structural_tof_pattern(cell(), &group(), &request),
            Err(StructuralTofError::IncompatibleCorrectionModel)
        ));
        request.correction_model = IntegratedIntensityCorrectionModel::Neutral;
        assert!(matches!(
            calculate_structural_tof_pattern_vjp(cell(), &group(), &request, &[1.0]),
            Err(StructuralTofError::PatternWeightLengthMismatch)
        ));
    }
}
