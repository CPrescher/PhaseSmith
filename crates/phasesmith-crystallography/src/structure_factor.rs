//! General-symmetry structure factors and integrated intensities.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::cell::{CELL_PARAMETER_COUNT, CellError, CellGeometry, UnitCell};
use crate::p1::P1ParameterLayout;
use crate::symmetry::{ExpandedSites, SpaceGroup, SymmetryError};

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const TWO_PI_SQUARED: f64 = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
const METRIC_TOLERANCE: f64 = 1.0e-10;

/// Borrowed arrays for one general-symmetry structural intensity batch.
#[derive(Clone, Copy, Debug)]
pub struct StructureFactorBatchView<'a> {
    /// Canonical Miller indices, one per powder family.
    pub hkl: &'a [[i32; 3]],
    /// Powder multiplicity for every canonical family.
    pub multiplicity: &'a [usize],
    /// Fractional asymmetric-unit coordinates, one row per independent site.
    pub fractional_xyz: &'a [[f64; 3]],
    /// Fractional occupancy for every independent site.
    pub occupancy: &'a [f64],
    /// Isotropic displacement in square ångströms for every independent site.
    pub u_iso_angstrom2: &'a [f64],
    /// True for sites whose displacement is described by a CIF U tensor.
    pub anisotropic_mask: &'a [bool],
    /// CIF U tensors in component order `11,22,33,23,13,12`.
    pub u_aniso_cif_angstrom2: &'a [[f64; 6]],
    /// Reflection-major real scattering amplitudes, shape `R * S`.
    pub scattering_real: &'a [f64],
    /// Reflection-major imaginary scattering amplitudes, shape `R * S`.
    pub scattering_imag: &'a [f64],
    /// Reflection-major analytical derivatives `d Re(f) / ds`.
    pub d_scattering_real_d_s: &'a [f64],
    /// Reflection-major analytical derivatives `d Im(f) / ds`.
    pub d_scattering_imag_d_s: &'a [f64],
    /// Integrated-intensity correction `C_h`, one per reflection.
    pub correction: &'a [f64],
    /// Analytical `d C_h / d(q²)`, one per reflection.
    pub d_correction_d_q_squared: &'a [f64],
    /// Non-negative structural phase scale.
    pub scale: f64,
    /// Periodic tolerance used only to identify special-position duplicates.
    pub coordinate_tolerance: f64,
}

/// General-symmetry values for one reflection batch.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureFactorValues {
    /// Real part of `F_h`.
    pub f_real: Vec<f64>,
    /// Imaginary part of `F_h`.
    pub f_imag: Vec<f64>,
    /// `|F_h|²` before scale, multiplicity, and correction.
    pub f_squared: Vec<f64>,
    /// Integrated reflection intensity.
    pub intensity: Vec<f64>,
    /// Reciprocal squared length `q² = 1/d²`.
    pub q_squared_inverse_angstrom2: Vec<f64>,
    /// Scattering-vector magnitude `s = sqrt(q²)/2`.
    pub s_inverse_angstrom: Vec<f64>,
}

/// Values and bounded parameter-major analytical derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureFactorDenseResult {
    /// Calculated values.
    pub values: StructureFactorValues,
    /// Parameter-major derivative of real `F`, shape `(P, R)`.
    pub d_f_real: Vec<f64>,
    /// Parameter-major derivative of imaginary `F`, shape `(P, R)`.
    pub d_f_imag: Vec<f64>,
    /// Parameter-major derivative of integrated intensity, shape `(P, R)`.
    pub d_intensity: Vec<f64>,
    /// Stable cell/site/scale parameter layout.
    pub layout: P1ParameterLayout,
}

/// Values and one forward structural derivative product.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureFactorJvpResult {
    /// Calculated values.
    pub values: StructureFactorValues,
    /// Directional derivative of real `F`.
    pub d_f_real: Vec<f64>,
    /// Directional derivative of imaginary `F`.
    pub d_f_imag: Vec<f64>,
    /// Directional derivative of integrated intensity.
    pub d_intensity: Vec<f64>,
}

/// Values and one reverse product for integrated-intensity weights.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureFactorVjpResult {
    /// Calculated values.
    pub values: StructureFactorValues,
    /// `J_intensity^T weights` in stable structural parameter order.
    pub gradient: Vec<f64>,
    /// Stable cell/site/scale parameter layout.
    pub layout: P1ParameterLayout,
}

/// Invalid general-symmetry structure-factor input.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureFactorBatchError {
    /// The unit cell is invalid.
    Cell(CellError),
    /// Symmetry expansion failed.
    Symmetry(SymmetryError),
    /// The cell metric is incompatible with the space-group rotations.
    CellSymmetryMismatch,
    /// Site arrays do not share one site count.
    SiteLengthMismatch,
    /// Reflection arrays do not share one reflection count.
    ReflectionLengthMismatch,
    /// Scattering arrays are not exactly reflection count times site count.
    ScatteringShapeMismatch,
    /// An input scalar or array entry is non-finite.
    NonFiniteInput,
    /// A physical scale, occupancy, displacement, correction, or multiplicity is invalid.
    InvalidPhysicalParameter,
    /// An `hkl = 0` row does not define a structural reflection.
    ZeroReflection,
    /// A requested output allocation overflowed addressable memory.
    AllocationOverflow,
    /// A forward tangent does not match the stable parameter layout.
    TangentLengthMismatch,
    /// Reverse weights do not match the reflection count.
    WeightLengthMismatch,
}

impl Display for StructureFactorBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cell(error) => Display::fmt(error, formatter),
            Self::Symmetry(error) => Display::fmt(error, formatter),
            Self::CellSymmetryMismatch => {
                formatter.write_str("unit-cell metric is incompatible with the space group")
            }
            Self::SiteLengthMismatch => {
                formatter.write_str("structure-factor site arrays must have equal length")
            }
            Self::ReflectionLengthMismatch => formatter
                .write_str("structure-factor reflection arrays must have equal length"),
            Self::ScatteringShapeMismatch => formatter.write_str(
                "structure-factor scattering arrays must have reflection_count * site_count elements",
            ),
            Self::NonFiniteInput => {
                formatter.write_str("structure-factor inputs must contain only finite values")
            }
            Self::InvalidPhysicalParameter => formatter.write_str(
                "scale, occupancy, U_iso, correction, and multiplicity must be physically valid",
            ),
            Self::ZeroReflection => formatter.write_str("hkl = (0, 0, 0) is not a reflection"),
            Self::AllocationOverflow => {
                formatter.write_str("structure-factor output allocation overflow")
            }
            Self::TangentLengthMismatch => formatter
                .write_str("structure-factor tangent length must equal the parameter count"),
            Self::WeightLengthMismatch => formatter
                .write_str("structure-factor reverse weights must match the reflection count"),
        }
    }
}

impl Error for StructureFactorBatchError {}

impl From<CellError> for StructureFactorBatchError {
    fn from(value: CellError) -> Self {
        Self::Cell(value)
    }
}

impl From<SymmetryError> for StructureFactorBatchError {
    fn from(value: SymmetryError) -> Self {
        Self::Symmetry(value)
    }
}

struct ValidatedStructure<'a> {
    batch: StructureFactorBatchView<'a>,
    geometry: CellGeometry,
    expanded: ExpandedSites,
    site_offsets: Vec<usize>,
    layout: P1ParameterLayout,
    reciprocal_axis_lengths: [f64; 3],
    d_reciprocal_axis_lengths: [[f64; CELL_PARAMETER_COUNT]; 3],
}

#[derive(Clone, Copy)]
struct SiteTerms {
    symmetry_real: f64,
    symmetry_imag: f64,
    d_symmetry_real: [f64; 3],
    d_symmetry_imag: [f64; 3],
    d_symmetry_real_d_cell: [f64; CELL_PARAMETER_COUNT],
    d_symmetry_imag_d_cell: [f64; CELL_PARAMETER_COUNT],
}

#[derive(Clone, Copy)]
struct SiteVjpEvaluation {
    terms: SiteTerms,
    scattering: (f64, f64),
    d_scattering: (f64, f64),
    base: (f64, f64),
    contribution: (f64, f64),
}

#[derive(Clone, Copy)]
struct CellReflectionTerms {
    q_squared: f64,
    root_q: f64,
    d_q_squared: [f64; CELL_PARAMETER_COUNT],
    d_q_direction: f64,
}

/// Calculate values without materializing structural derivatives.
///
/// # Errors
///
/// Returns [`StructureFactorBatchError`] for invalid cells, symmetry, shapes,
/// values, physical parameters, or output sizes.
pub fn calculate_structure_factor_values(
    cell: UnitCell,
    space_group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
) -> Result<StructureFactorValues, StructureFactorBatchError> {
    let validated = validate(cell, space_group, batch)?;
    let mut values = empty_values(batch.hkl.len());
    for reflection in 0..batch.hkl.len() {
        evaluate_value_reflection(&validated, reflection, &mut values);
    }
    Ok(values)
}

/// Calculate values and a parameter-major dense structural Jacobian.
///
/// This allocation is intended for tests and small diagnostics. Production
/// composition uses directional derivative products.
///
/// # Errors
///
/// Returns [`StructureFactorBatchError`] for invalid inputs or allocation
/// overflow.
pub fn calculate_structure_factor_dense(
    cell: UnitCell,
    space_group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
) -> Result<StructureFactorDenseResult, StructureFactorBatchError> {
    let validated = validate(cell, space_group, batch)?;
    let reflection_count = batch.hkl.len();
    let parameter_count = validated.layout.parameter_count();
    let element_count = parameter_count
        .checked_mul(reflection_count)
        .ok_or(StructureFactorBatchError::AllocationOverflow)?;
    let mut result = StructureFactorDenseResult {
        values: empty_values(reflection_count),
        d_f_real: vec![0.0; element_count],
        d_f_imag: vec![0.0; element_count],
        d_intensity: vec![0.0; element_count],
        layout: validated.layout,
    };
    for reflection in 0..reflection_count {
        evaluate_dense_reflection(&validated, reflection, &mut result);
    }
    Ok(result)
}

/// Calculate values and one forward derivative without a dense Jacobian.
///
/// # Errors
///
/// Returns [`StructureFactorBatchError`] for invalid batch data or a tangent
/// that does not match the stable parameter layout.
pub fn calculate_structure_factor_jvp(
    cell: UnitCell,
    space_group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
    tangent: &[f64],
) -> Result<StructureFactorJvpResult, StructureFactorBatchError> {
    let validated = validate(cell, space_group, batch)?;
    if tangent.len() != validated.layout.parameter_count() {
        return Err(StructureFactorBatchError::TangentLengthMismatch);
    }
    if tangent.iter().any(|value| !value.is_finite()) {
        return Err(StructureFactorBatchError::NonFiniteInput);
    }
    let reflection_count = batch.hkl.len();
    let mut result = StructureFactorJvpResult {
        values: empty_values(reflection_count),
        d_f_real: vec![0.0; reflection_count],
        d_f_imag: vec![0.0; reflection_count],
        d_intensity: vec![0.0; reflection_count],
    };
    for reflection in 0..reflection_count {
        evaluate_jvp_reflection(&validated, reflection, tangent, &mut result);
    }
    Ok(result)
}

/// Calculate values and `J_intensity^T weights` without a dense Jacobian.
///
/// # Errors
///
/// Returns [`StructureFactorBatchError`] for invalid batch data or reverse
/// weights that do not match the reflection count.
pub fn calculate_structure_factor_intensity_vjp(
    cell: UnitCell,
    space_group: &SpaceGroup,
    batch: StructureFactorBatchView<'_>,
    weights: &[f64],
) -> Result<StructureFactorVjpResult, StructureFactorBatchError> {
    let validated = validate(cell, space_group, batch)?;
    if weights.len() != batch.hkl.len() {
        return Err(StructureFactorBatchError::WeightLengthMismatch);
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(StructureFactorBatchError::NonFiniteInput);
    }
    let mut result = StructureFactorVjpResult {
        values: empty_values(batch.hkl.len()),
        gradient: vec![0.0; validated.layout.parameter_count()],
        layout: validated.layout,
    };
    let mut site_evaluations = Vec::new();
    site_evaluations
        .try_reserve_exact(validated.layout.site_count)
        .map_err(|_| StructureFactorBatchError::AllocationOverflow)?;
    for (reflection, weight) in weights.iter().copied().enumerate() {
        evaluate_vjp_reflection(
            &validated,
            reflection,
            weight,
            &mut site_evaluations,
            &mut result,
        );
    }
    Ok(result)
}

fn validate<'a>(
    cell: UnitCell,
    space_group: &SpaceGroup,
    batch: StructureFactorBatchView<'a>,
) -> Result<ValidatedStructure<'a>, StructureFactorBatchError> {
    let geometry = cell.geometry()?;
    validate_metric_compatibility(
        &geometry,
        space_group.metric_constraints().equations.as_slice(),
    )?;
    let site_count = batch.fractional_xyz.len();
    if batch.occupancy.len() != site_count
        || batch.u_iso_angstrom2.len() != site_count
        || batch.anisotropic_mask.len() != site_count
        || batch.u_aniso_cif_angstrom2.len() != site_count
    {
        return Err(StructureFactorBatchError::SiteLengthMismatch);
    }
    let reflection_count = batch.hkl.len();
    if batch.multiplicity.len() != reflection_count
        || batch.correction.len() != reflection_count
        || batch.d_correction_d_q_squared.len() != reflection_count
    {
        return Err(StructureFactorBatchError::ReflectionLengthMismatch);
    }
    if batch.hkl.contains(&[0, 0, 0]) {
        return Err(StructureFactorBatchError::ZeroReflection);
    }
    let scattering_count = reflection_count
        .checked_mul(site_count)
        .ok_or(StructureFactorBatchError::AllocationOverflow)?;
    if [
        batch.scattering_real.len(),
        batch.scattering_imag.len(),
        batch.d_scattering_real_d_s.len(),
        batch.d_scattering_imag_d_s.len(),
    ]
    .into_iter()
    .any(|count| count != scattering_count)
    {
        return Err(StructureFactorBatchError::ScatteringShapeMismatch);
    }
    if !batch.scale.is_finite()
        || batch
            .fractional_xyz
            .iter()
            .flatten()
            .chain(batch.occupancy)
            .chain(batch.u_iso_angstrom2)
            .chain(batch.u_aniso_cif_angstrom2.iter().flatten())
            .chain(batch.scattering_real)
            .chain(batch.scattering_imag)
            .chain(batch.d_scattering_real_d_s)
            .chain(batch.d_scattering_imag_d_s)
            .chain(batch.correction)
            .chain(batch.d_correction_d_q_squared)
            .any(|value| !value.is_finite())
    {
        return Err(StructureFactorBatchError::NonFiniteInput);
    }
    if batch.scale < 0.0
        || batch.occupancy.iter().any(|value| *value < 0.0)
        || batch
            .u_iso_angstrom2
            .iter()
            .zip(batch.anisotropic_mask)
            .any(|(value, anisotropic)| !anisotropic && *value < 0.0)
        || batch
            .u_aniso_cif_angstrom2
            .iter()
            .zip(batch.anisotropic_mask)
            .any(|(tensor, anisotropic)| *anisotropic && !valid_anisotropic_tensor(*tensor))
        || batch.correction.iter().any(|value| *value < 0.0)
        || batch.multiplicity.contains(&0)
    {
        return Err(StructureFactorBatchError::InvalidPhysicalParameter);
    }
    let expanded = space_group.expand_sites(batch.fractional_xyz, batch.coordinate_tolerance)?;
    let mut site_offsets = vec![0; site_count + 1];
    for source in &expanded.source_site {
        site_offsets[*source + 1] += 1;
    }
    for site in 0..site_count {
        site_offsets[site + 1] += site_offsets[site];
    }
    let reciprocal_axis_lengths = [
        geometry.reciprocal_metric[0][0].sqrt(),
        geometry.reciprocal_metric[1][1].sqrt(),
        geometry.reciprocal_metric[2][2].sqrt(),
    ];
    let d_reciprocal_axis_lengths = if batch.anisotropic_mask.contains(&true) {
        geometry.reciprocal_axis_lengths_and_derivatives().1
    } else {
        [[0.0; CELL_PARAMETER_COUNT]; 3]
    };
    Ok(ValidatedStructure {
        batch,
        geometry,
        expanded,
        site_offsets,
        layout: P1ParameterLayout { site_count },
        reciprocal_axis_lengths,
        d_reciprocal_axis_lengths,
    })
}

fn valid_anisotropic_tensor(tensor: [f64; 6]) -> bool {
    let [u11, u22, u33, u23, u13, u12] = tensor;
    let scale = tensor.iter().copied().map(f64::abs).fold(1.0_f64, f64::max);
    let tolerance = 1.0e-12 * scale;
    let minor_12 = u11 * u22 - u12 * u12;
    let minor_13 = u11 * u33 - u13 * u13;
    let minor_23 = u22 * u33 - u23 * u23;
    let determinant = u11 * u22 * u33 + 2.0 * u12 * u13 * u23
        - u11 * u23 * u23
        - u22 * u13 * u13
        - u33 * u12 * u12;
    u11 >= -tolerance
        && u22 >= -tolerance
        && u33 >= -tolerance
        && minor_12 >= -tolerance * scale
        && minor_13 >= -tolerance * scale
        && minor_23 >= -tolerance * scale
        && determinant >= -tolerance * scale * scale
}

fn empty_values(reflection_count: usize) -> StructureFactorValues {
    StructureFactorValues {
        f_real: vec![0.0; reflection_count],
        f_imag: vec![0.0; reflection_count],
        f_squared: vec![0.0; reflection_count],
        intensity: vec![0.0; reflection_count],
        q_squared_inverse_angstrom2: vec![0.0; reflection_count],
        s_inverse_angstrom: vec![0.0; reflection_count],
    }
}

fn evaluate_value_reflection(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    values: &mut StructureFactorValues,
) {
    let batch = validated.batch;
    let q_squared = validated.geometry.q_squared(batch.hkl[reflection]);
    let s = 0.5 * q_squared.sqrt();
    let mut f_real = 0.0;
    let mut f_imag = 0.0;
    for site in 0..validated.layout.site_count {
        let (base_real, base_imag) = site_value_base(
            validated,
            reflection,
            site,
            batch.hkl[reflection],
            q_squared,
        );
        f_real += batch.occupancy[site] * base_real;
        f_imag += batch.occupancy[site] * base_imag;
    }
    set_values(values, batch, reflection, q_squared, s, f_real, f_imag);
}

fn site_value_base(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    site: usize,
    hkl: [i32; 3],
    q_squared: f64,
) -> (f64, f64) {
    let anisotropic = validated.batch.anisotropic_mask[site];
    let mut symmetry = (0.0, 0.0);
    if anisotropic {
        for expanded_index in validated.site_offsets[site]..validated.site_offsets[site + 1] {
            let position = validated.expanded.fractional_xyz[expanded_index];
            let phase = TWO_PI
                * hkl
                    .iter()
                    .zip(position)
                    .map(|(index, coordinate)| f64::from(*index) * coordinate)
                    .sum::<f64>();
            let (sin_phase, cos_phase) = phase.sin_cos();
            let displacement = anisotropic_displacement_value(
                validated,
                hkl,
                validated.expanded.representative_rotation[expanded_index],
                validated.batch.u_aniso_cif_angstrom2[site],
            );
            symmetry.0 += displacement * cos_phase;
            symmetry.1 += displacement * sin_phase;
        }
    } else {
        for expanded_index in validated.site_offsets[site]..validated.site_offsets[site + 1] {
            let position = validated.expanded.fractional_xyz[expanded_index];
            let phase = TWO_PI
                * hkl
                    .iter()
                    .zip(position)
                    .map(|(index, coordinate)| f64::from(*index) * coordinate)
                    .sum::<f64>();
            let (sin_phase, cos_phase) = phase.sin_cos();
            symmetry.0 += cos_phase;
            symmetry.1 += sin_phase;
        }
        let displacement =
            (-TWO_PI_SQUARED * validated.batch.u_iso_angstrom2[site] * q_squared).exp();
        symmetry.0 *= displacement;
        symmetry.1 *= displacement;
    }
    let index = reflection * validated.layout.site_count + site;
    complex_multiply(
        (
            validated.batch.scattering_real[index],
            validated.batch.scattering_imag[index],
        ),
        symmetry,
    )
}

fn evaluate_dense_reflection(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    result: &mut StructureFactorDenseResult,
) {
    let batch = validated.batch;
    let reflection_count = batch.hkl.len();
    let (q_squared, d_q_squared) = validated
        .geometry
        .q_squared_and_derivatives(batch.hkl[reflection]);
    let root_q = q_squared.sqrt();
    let s = 0.5 * root_q;
    let mut f_real = 0.0;
    let mut f_imag = 0.0;
    for site in 0..validated.layout.site_count {
        let (contribution_real, contribution_imag) = accumulate_dense_site(
            validated,
            reflection,
            site,
            q_squared,
            root_q,
            d_q_squared,
            result,
        );
        f_real += contribution_real;
        f_imag += contribution_imag;
    }
    set_values(
        &mut result.values,
        batch,
        reflection,
        q_squared,
        s,
        f_real,
        f_imag,
    );
    let norm = f_real * f_real + f_imag * f_imag;
    let multiplicity = multiplicity_f64(batch.multiplicity[reflection]);
    let correction = batch.correction[reflection];
    let q_derivatives = d_q_squared
        .into_iter()
        .chain(std::iter::repeat(0.0))
        .take(validated.layout.parameter_count());
    for (parameter, d_q) in q_derivatives.enumerate() {
        let index = parameter * reflection_count + reflection;
        let d_norm = 2.0 * (f_real * result.d_f_real[index] + f_imag * result.d_f_imag[index]);
        let d_correction = batch.d_correction_d_q_squared[reflection] * d_q;
        result.d_intensity[index] =
            multiplicity * batch.scale * (correction * d_norm + d_correction * norm);
    }
    result.d_intensity[validated.layout.scale() * reflection_count + reflection] =
        multiplicity * correction * norm;
}

fn accumulate_dense_site(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    site: usize,
    q_squared: f64,
    root_q: f64,
    d_q_squared: [f64; CELL_PARAMETER_COUNT],
    result: &mut StructureFactorDenseResult,
) -> (f64, f64) {
    let batch = validated.batch;
    let reflection_count = batch.hkl.len();
    let terms = symmetry_terms(
        validated,
        batch.hkl[reflection],
        site,
        q_squared,
        Some(d_q_squared),
    );
    let (base_real, base_imag) = site_base(validated, reflection, site, terms);
    let occupancy = batch.occupancy[site];
    let contribution = (occupancy * base_real, occupancy * base_imag);
    let scattering_index = reflection * validated.layout.site_count + site;
    let scattering = (
        batch.scattering_real[scattering_index],
        batch.scattering_imag[scattering_index],
    );
    let d_scattering = (
        batch.d_scattering_real_d_s[scattering_index],
        batch.d_scattering_imag_d_s[scattering_index],
    );
    for (parameter, d_q) in d_q_squared.into_iter().enumerate() {
        let d_s = d_q / (4.0 * root_q);
        let scattering_derivative = (d_scattering.0 * d_s, d_scattering.1 * d_s);
        let rotated_scattering = complex_multiply(
            scattering_derivative,
            (terms.symmetry_real, terms.symmetry_imag),
        );
        let rotated_displacement = complex_multiply(
            scattering,
            (
                terms.d_symmetry_real_d_cell[parameter],
                terms.d_symmetry_imag_d_cell[parameter],
            ),
        );
        set_f_derivative(
            result,
            parameter,
            reflection,
            reflection_count,
            occupancy * (rotated_scattering.0 + rotated_displacement.0),
            occupancy * (rotated_scattering.1 + rotated_displacement.1),
        );
    }
    for (component, (&d_real, &d_imag)) in terms
        .d_symmetry_real
        .iter()
        .zip(&terms.d_symmetry_imag)
        .enumerate()
    {
        let rotated = complex_multiply(scattering, (d_real, d_imag));
        set_f_derivative(
            result,
            validated.layout.coordinate(site, component),
            reflection,
            reflection_count,
            occupancy * rotated.0,
            occupancy * rotated.1,
        );
    }
    set_f_derivative(
        result,
        validated.layout.occupancy(site),
        reflection,
        reflection_count,
        base_real,
        base_imag,
    );
    if !batch.anisotropic_mask[site] {
        set_f_derivative(
            result,
            validated.layout.u_iso(site),
            reflection,
            reflection_count,
            -TWO_PI_SQUARED * q_squared * contribution.0,
            -TWO_PI_SQUARED * q_squared * contribution.1,
        );
    }
    contribution
}

fn evaluate_jvp_reflection(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    tangent: &[f64],
    result: &mut StructureFactorJvpResult,
) {
    let batch = validated.batch;
    let (q_squared, d_q_squared) = validated
        .geometry
        .q_squared_and_derivatives(batch.hkl[reflection]);
    let root_q = q_squared.sqrt();
    let d_q_direction = d_q_squared
        .iter()
        .zip(&tangent[..CELL_PARAMETER_COUNT])
        .map(|(derivative, direction)| derivative * direction)
        .sum::<f64>();
    let mut f = (0.0, 0.0);
    let mut d_f = (0.0, 0.0);
    let cell_terms = CellReflectionTerms {
        q_squared,
        root_q,
        d_q_squared,
        d_q_direction,
    };
    for site in 0..validated.layout.site_count {
        let (contribution, derivative) = jvp_site(validated, reflection, site, cell_terms, tangent);
        f.0 += contribution.0;
        f.1 += contribution.1;
        d_f.0 += derivative.0;
        d_f.1 += derivative.1;
    }
    let s = 0.5 * root_q;
    set_values(
        &mut result.values,
        batch,
        reflection,
        q_squared,
        s,
        f.0,
        f.1,
    );
    result.d_f_real[reflection] = d_f.0;
    result.d_f_imag[reflection] = d_f.1;
    let norm = f.0 * f.0 + f.1 * f.1;
    let d_norm = 2.0 * (f.0 * d_f.0 + f.1 * d_f.1);
    let correction = batch.correction[reflection];
    let d_correction = batch.d_correction_d_q_squared[reflection] * d_q_direction;
    result.d_intensity[reflection] = multiplicity_f64(batch.multiplicity[reflection])
        * (tangent[validated.layout.scale()] * correction * norm
            + batch.scale * (d_correction * norm + correction * d_norm));
}

fn jvp_site(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    site: usize,
    cell: CellReflectionTerms,
    tangent: &[f64],
) -> ((f64, f64), (f64, f64)) {
    let batch = validated.batch;
    let terms = symmetry_terms(
        validated,
        batch.hkl[reflection],
        site,
        cell.q_squared,
        Some(cell.d_q_squared),
    );
    let scattering_index = reflection * validated.layout.site_count + site;
    let scattering = (
        batch.scattering_real[scattering_index],
        batch.scattering_imag[scattering_index],
    );
    let d_scattering = (
        batch.d_scattering_real_d_s[scattering_index],
        batch.d_scattering_imag_d_s[scattering_index],
    );
    let symmetry = (terms.symmetry_real, terms.symmetry_imag);
    let base_rotated = complex_multiply(scattering, symmetry);
    let base = base_rotated;
    let occupancy = batch.occupancy[site];
    let contribution = (occupancy * base.0, occupancy * base.1);

    let d_s = cell.d_q_direction / (4.0 * cell.root_q);
    let scattering_direction = (d_scattering.0 * d_s, d_scattering.1 * d_s);
    let mut displacement_direction = (0.0, 0.0);
    for (parameter, direction) in tangent[..CELL_PARAMETER_COUNT].iter().copied().enumerate() {
        displacement_direction.0 += direction * terms.d_symmetry_real_d_cell[parameter];
        displacement_direction.1 += direction * terms.d_symmetry_imag_d_cell[parameter];
    }
    let cell_scattering = complex_multiply(scattering_direction, symmetry);
    let cell_displacement = complex_multiply(scattering, displacement_direction);
    let d_symmetry = (0..3).fold((0.0, 0.0), |sum, component| {
        let direction = tangent[validated.layout.coordinate(site, component)];
        (
            sum.0 + direction * terms.d_symmetry_real[component],
            sum.1 + direction * terms.d_symmetry_imag[component],
        )
    });
    let coordinate_rotated = complex_multiply(scattering, d_symmetry);
    let occupancy_direction = tangent[validated.layout.occupancy(site)];
    let u_direction = if batch.anisotropic_mask[site] {
        0.0
    } else {
        tangent[validated.layout.u_iso(site)]
    };
    let derivative = (
        occupancy_direction * base.0
            + occupancy * (cell_scattering.0 + cell_displacement.0 + coordinate_rotated.0)
            - TWO_PI_SQUARED * cell.q_squared * u_direction * contribution.0,
        occupancy_direction * base.1
            + occupancy * (cell_scattering.1 + cell_displacement.1 + coordinate_rotated.1)
            - TWO_PI_SQUARED * cell.q_squared * u_direction * contribution.1,
    );
    (contribution, derivative)
}

fn evaluate_vjp_reflection(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    weight: f64,
    site_evaluations: &mut Vec<SiteVjpEvaluation>,
    result: &mut StructureFactorVjpResult,
) {
    let batch = validated.batch;
    let (q_squared, d_q_squared) = validated
        .geometry
        .q_squared_and_derivatives(batch.hkl[reflection]);
    let root_q = q_squared.sqrt();
    let mut f = (0.0, 0.0);
    site_evaluations.clear();
    for site in 0..validated.layout.site_count {
        let terms = symmetry_terms(
            validated,
            batch.hkl[reflection],
            site,
            q_squared,
            Some(d_q_squared),
        );
        let scattering_index = reflection * validated.layout.site_count + site;
        let scattering = (
            batch.scattering_real[scattering_index],
            batch.scattering_imag[scattering_index],
        );
        let d_scattering = (
            batch.d_scattering_real_d_s[scattering_index],
            batch.d_scattering_imag_d_s[scattering_index],
        );
        let rotated = complex_multiply(scattering, (terms.symmetry_real, terms.symmetry_imag));
        let base = rotated;
        let occupancy = batch.occupancy[site];
        let contribution = (occupancy * base.0, occupancy * base.1);
        f.0 += contribution.0;
        f.1 += contribution.1;
        site_evaluations.push(SiteVjpEvaluation {
            terms,
            scattering,
            d_scattering,
            base,
            contribution,
        });
    }
    set_values(
        &mut result.values,
        batch,
        reflection,
        q_squared,
        0.5 * root_q,
        f.0,
        f.1,
    );
    let norm = f.0 * f.0 + f.1 * f.1;
    let multiplicity = multiplicity_f64(batch.multiplicity[reflection]);
    let correction = batch.correction[reflection];
    let f_weight = 2.0 * weight * multiplicity * batch.scale * correction;
    for (site, evaluation) in site_evaluations.iter().copied().enumerate() {
        accumulate_vjp_site(
            validated,
            site,
            q_squared,
            root_q,
            d_q_squared,
            f,
            f_weight,
            evaluation,
            &mut result.gradient,
        );
    }
    let correction_weight =
        weight * multiplicity * batch.scale * batch.d_correction_d_q_squared[reflection] * norm;
    for (parameter, d_q) in d_q_squared.into_iter().enumerate() {
        result.gradient[parameter] += correction_weight * d_q;
    }
    result.gradient[validated.layout.scale()] += weight * multiplicity * correction * norm;
}

#[allow(clippy::too_many_arguments)]
fn accumulate_vjp_site(
    validated: &ValidatedStructure<'_>,
    site: usize,
    q_squared: f64,
    root_q: f64,
    d_q_squared: [f64; CELL_PARAMETER_COUNT],
    f: (f64, f64),
    f_weight: f64,
    evaluation: SiteVjpEvaluation,
    gradient: &mut [f64],
) {
    let batch = validated.batch;
    let SiteVjpEvaluation {
        terms,
        scattering,
        d_scattering,
        base,
        contribution,
    } = evaluation;
    let symmetry = (terms.symmetry_real, terms.symmetry_imag);
    let occupancy = batch.occupancy[site];
    for (parameter, d_q) in d_q_squared.into_iter().enumerate() {
        let d_s = d_q / (4.0 * root_q);
        let scattering_derivative = (d_scattering.0 * d_s, d_scattering.1 * d_s);
        let rotated_scattering = complex_multiply(scattering_derivative, symmetry);
        let rotated_displacement = complex_multiply(
            scattering,
            (
                terms.d_symmetry_real_d_cell[parameter],
                terms.d_symmetry_imag_d_cell[parameter],
            ),
        );
        gradient[parameter] += f_weight
            * occupancy
            * (f.0 * (rotated_scattering.0 + rotated_displacement.0)
                + f.1 * (rotated_scattering.1 + rotated_displacement.1));
    }
    for (component, (&d_real, &d_imag)) in terms
        .d_symmetry_real
        .iter()
        .zip(&terms.d_symmetry_imag)
        .enumerate()
    {
        let rotated = complex_multiply(scattering, (d_real, d_imag));
        gradient[validated.layout.coordinate(site, component)] +=
            f_weight * occupancy * (f.0 * rotated.0 + f.1 * rotated.1);
    }
    gradient[validated.layout.occupancy(site)] += f_weight * (f.0 * base.0 + f.1 * base.1);
    if !batch.anisotropic_mask[site] {
        gradient[validated.layout.u_iso(site)] +=
            f_weight * -TWO_PI_SQUARED * q_squared * (f.0 * contribution.0 + f.1 * contribution.1);
    }
}

fn symmetry_terms(
    validated: &ValidatedStructure<'_>,
    hkl: [i32; 3],
    site: usize,
    q_squared: f64,
    d_q_squared: Option<[f64; CELL_PARAMETER_COUNT]>,
) -> SiteTerms {
    let mut result = SiteTerms {
        symmetry_real: 0.0,
        symmetry_imag: 0.0,
        d_symmetry_real: [0.0; 3],
        d_symmetry_imag: [0.0; 3],
        d_symmetry_real_d_cell: [0.0; CELL_PARAMETER_COUNT],
        d_symmetry_imag_d_cell: [0.0; CELL_PARAMETER_COUNT],
    };
    let anisotropic = validated.batch.anisotropic_mask[site];
    let tensor = validated.batch.u_aniso_cif_angstrom2[site];
    for expanded_index in validated.site_offsets[site]..validated.site_offsets[site + 1] {
        let position = validated.expanded.fractional_xyz[expanded_index];
        let rotation = validated.expanded.representative_rotation[expanded_index];
        let phase = TWO_PI
            * hkl
                .iter()
                .zip(position)
                .map(|(index, coordinate)| f64::from(*index) * coordinate)
                .sum::<f64>();
        let (sin_phase, cos_phase) = phase.sin_cos();
        let (displacement, d_displacement) = if anisotropic {
            anisotropic_displacement(validated, hkl, rotation, tensor, d_q_squared.is_some())
        } else {
            (1.0, [0.0; CELL_PARAMETER_COUNT])
        };
        result.symmetry_real += displacement * cos_phase;
        result.symmetry_imag += displacement * sin_phase;
        for (component, (d_real, d_imag)) in result
            .d_symmetry_real
            .iter_mut()
            .zip(&mut result.d_symmetry_imag)
            .enumerate()
        {
            let phase_derivative = TWO_PI
                * (0..3)
                    .map(|row| f64::from(hkl[row]) * f64::from(rotation[row][component]))
                    .sum::<f64>();
            *d_real -= displacement * phase_derivative * sin_phase;
            *d_imag += displacement * phase_derivative * cos_phase;
        }
        for (parameter, derivative) in d_displacement.iter().copied().enumerate() {
            result.d_symmetry_real_d_cell[parameter] += derivative * cos_phase;
            result.d_symmetry_imag_d_cell[parameter] += derivative * sin_phase;
        }
    }
    if !anisotropic {
        let displacement =
            (-TWO_PI_SQUARED * validated.batch.u_iso_angstrom2[site] * q_squared).exp();
        let factor = -TWO_PI_SQUARED * validated.batch.u_iso_angstrom2[site];
        if let Some(derivatives) = d_q_squared {
            for (parameter, derivative) in derivatives.iter().copied().enumerate() {
                let d_displacement = factor * derivative * displacement;
                result.d_symmetry_real_d_cell[parameter] = d_displacement * result.symmetry_real;
                result.d_symmetry_imag_d_cell[parameter] = d_displacement * result.symmetry_imag;
            }
        }
        result.symmetry_real *= displacement;
        result.symmetry_imag *= displacement;
        for value in &mut result.d_symmetry_real {
            *value *= displacement;
        }
        for value in &mut result.d_symmetry_imag {
            *value *= displacement;
        }
    }
    result
}

fn anisotropic_displacement(
    validated: &ValidatedStructure<'_>,
    hkl: [i32; 3],
    rotation: [[i32; 3]; 3],
    tensor: [f64; 6],
    calculate_derivatives: bool,
) -> (f64, [f64; CELL_PARAMETER_COUNT]) {
    let (transformed_hkl, vector) = transformed_reciprocal_vector(validated, hkl, rotation);
    let tensor_times_vector = symmetric_tensor_vector(tensor, vector);
    let quadratic = vector
        .iter()
        .zip(tensor_times_vector)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let displacement = (-TWO_PI_SQUARED * quadratic).exp();
    let mut derivatives = [0.0; CELL_PARAMETER_COUNT];
    if !calculate_derivatives {
        return (displacement, derivatives);
    }
    for (parameter, derivative) in derivatives.iter_mut().enumerate() {
        let d_vector = [
            f64::from(transformed_hkl[0]) * validated.d_reciprocal_axis_lengths[0][parameter],
            f64::from(transformed_hkl[1]) * validated.d_reciprocal_axis_lengths[1][parameter],
            f64::from(transformed_hkl[2]) * validated.d_reciprocal_axis_lengths[2][parameter],
        ];
        let d_quadratic = 2.0
            * d_vector
                .iter()
                .zip(tensor_times_vector)
                .map(|(left, right)| left * right)
                .sum::<f64>();
        *derivative = -TWO_PI_SQUARED * d_quadratic * displacement;
    }
    (displacement, derivatives)
}

fn anisotropic_displacement_value(
    validated: &ValidatedStructure<'_>,
    hkl: [i32; 3],
    rotation: [[i32; 3]; 3],
    tensor: [f64; 6],
) -> f64 {
    let (_, vector) = transformed_reciprocal_vector(validated, hkl, rotation);
    let quadratic = vector
        .iter()
        .zip(symmetric_tensor_vector(tensor, vector))
        .map(|(left, right)| left * right)
        .sum::<f64>();
    (-TWO_PI_SQUARED * quadratic).exp()
}

fn transformed_reciprocal_vector(
    validated: &ValidatedStructure<'_>,
    hkl: [i32; 3],
    rotation: [[i32; 3]; 3],
) -> ([i32; 3], [f64; 3]) {
    let transformed_hkl = [0, 1, 2].map(|component| {
        (0..3)
            .map(|row| hkl[row] * rotation[row][component])
            .sum::<i32>()
    });
    let reciprocal = validated.reciprocal_axis_lengths;
    let vector = [
        f64::from(transformed_hkl[0]) * reciprocal[0],
        f64::from(transformed_hkl[1]) * reciprocal[1],
        f64::from(transformed_hkl[2]) * reciprocal[2],
    ];
    (transformed_hkl, vector)
}

fn symmetric_tensor_vector(tensor: [f64; 6], vector: [f64; 3]) -> [f64; 3] {
    let [u11, u22, u33, u23, u13, u12] = tensor;
    [
        u11 * vector[0] + u12 * vector[1] + u13 * vector[2],
        u12 * vector[0] + u22 * vector[1] + u23 * vector[2],
        u13 * vector[0] + u23 * vector[1] + u33 * vector[2],
    ]
}

fn site_base(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    site: usize,
    terms: SiteTerms,
) -> (f64, f64) {
    let index = reflection * validated.layout.site_count + site;
    complex_multiply(
        (
            validated.batch.scattering_real[index],
            validated.batch.scattering_imag[index],
        ),
        (terms.symmetry_real, terms.symmetry_imag),
    )
}

fn set_values(
    values: &mut StructureFactorValues,
    batch: StructureFactorBatchView<'_>,
    reflection: usize,
    q_squared: f64,
    s: f64,
    f_real: f64,
    f_imag: f64,
) {
    let norm = f_real * f_real + f_imag * f_imag;
    values.f_real[reflection] = f_real;
    values.f_imag[reflection] = f_imag;
    values.f_squared[reflection] = norm;
    values.intensity[reflection] = batch.scale
        * multiplicity_f64(batch.multiplicity[reflection])
        * batch.correction[reflection]
        * norm;
    values.q_squared_inverse_angstrom2[reflection] = q_squared;
    values.s_inverse_angstrom[reflection] = s;
}

fn set_f_derivative(
    result: &mut StructureFactorDenseResult,
    parameter: usize,
    reflection: usize,
    reflection_count: usize,
    real: f64,
    imag: f64,
) {
    let index = parameter * reflection_count + reflection;
    result.d_f_real[index] += real;
    result.d_f_imag[index] += imag;
}

fn complex_multiply(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

#[allow(clippy::cast_precision_loss)]
fn multiplicity_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn validate_metric_compatibility(
    geometry: &CellGeometry,
    equations: &[[i64; 6]],
) -> Result<(), StructureFactorBatchError> {
    let metric = geometry.direct_metric;
    let components = [
        metric[0][0],
        metric[1][1],
        metric[2][2],
        metric[1][2],
        metric[0][2],
        metric[0][1],
    ];
    let scale = components
        .iter()
        .copied()
        .map(f64::abs)
        .fold(1.0_f64, f64::max);
    for equation in equations {
        let residual = equation
            .iter()
            .zip(components)
            .map(|(coefficient, value)| *coefficient as f64 * value)
            .sum::<f64>();
        let coefficient_scale = equation.iter().copied().map(i64::unsigned_abs).sum::<u64>() as f64;
        if residual.abs() > METRIC_TOLERANCE * scale * coefficient_scale.max(1.0) {
            return Err(StructureFactorBatchError::CellSymmetryMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry::{Rational, SymmetryOperation};

    fn cubic_cell(a: f64) -> UnitCell {
        UnitCell {
            a_angstrom: a,
            b_angstrom: a,
            c_angstrom: a,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        }
    }

    fn p1() -> SpaceGroup {
        SpaceGroup::new(vec![SymmetryOperation::identity()]).expect("P1")
    }

    fn inversion() -> SpaceGroup {
        SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
                .expect("inversion"),
        ])
        .expect("P-1")
    }

    fn axis_swap_group() -> SpaceGroup {
        SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new([[0, 1, 0], [1, 0, 0], [0, 0, -1]], [Rational::zero(); 3])
                .expect("axis swap"),
        ])
        .expect("closed axis-swap group")
    }

    #[test]
    fn inversion_values_and_special_positions_have_closed_forms() {
        let hkl = [[1, 2, 1]];
        let multiplicity = [2];
        let xyz = [[0.13, 0.21, 0.07], [0.0, 0.0, 0.0]];
        let occupancy = [0.8, 0.5];
        let u_iso = [0.0, 0.0];
        let real = [3.0, 2.0];
        let zero = [0.0, 0.0];
        let correction = [1.25];
        let values = calculate_structure_factor_values(
            cubic_cell(5.0),
            &inversion(),
            StructureFactorBatchView {
                hkl: &hkl,
                multiplicity: &multiplicity,
                fractional_xyz: &xyz,
                occupancy: &occupancy,
                u_iso_angstrom2: &u_iso,
                anisotropic_mask: &[false, false],
                u_aniso_cif_angstrom2: &[[0.0; 6]; 2],
                scattering_real: &real,
                scattering_imag: &zero,
                d_scattering_real_d_s: &zero,
                d_scattering_imag_d_s: &zero,
                correction: &correction,
                d_correction_d_q_squared: &[0.0],
                scale: 1.4,
                coordinate_tolerance: 1.0e-10,
            },
        )
        .expect("structure factors");
        let phase = TWO_PI * (0.13 + 2.0 * 0.21 + 0.07);
        let expected_f = 0.8 * 3.0 * 2.0 * phase.cos() + 0.5 * 2.0;
        assert!((values.f_real[0] - expected_f).abs() < 2.0e-14);
        assert!(values.f_imag[0].abs() < 2.0e-14);
        assert!((values.intensity[0] - 1.4 * 2.0 * 1.25 * expected_f.powi(2)).abs() < 1.0e-12);
    }

    #[test]
    fn anisotropic_symmetry_mates_use_rotated_reciprocal_indices() {
        let hkl = [[2, 1, 1]];
        let xyz = [[0.17, 0.29, 0.11]];
        let tensor = [[0.020, 0.010, 0.030, 0.002, 0.001, 0.003]];
        let values = calculate_structure_factor_values(
            cubic_cell(5.0),
            &axis_swap_group(),
            StructureFactorBatchView {
                hkl: &hkl,
                multiplicity: &[1],
                fractional_xyz: &xyz,
                occupancy: &[0.8],
                u_iso_angstrom2: &[0.0],
                anisotropic_mask: &[true],
                u_aniso_cif_angstrom2: &tensor,
                scattering_real: &[3.0],
                scattering_imag: &[0.0],
                d_scattering_real_d_s: &[0.0],
                d_scattering_imag_d_s: &[0.0],
                correction: &[1.0],
                d_correction_d_q_squared: &[0.0],
                scale: 1.0,
                coordinate_tolerance: 1.0e-10,
            },
        )
        .expect("anisotropic structure factor");
        let reciprocal = 0.2;
        let expected = [[2.0, 1.0, 1.0], [1.0, 2.0, -1.0]]
            .into_iter()
            .zip([[0.17, 0.29, 0.11], [0.29, 0.17, -0.11]])
            .map(|(indices, position)| {
                let vector = indices.map(|value| value * reciprocal);
                let quadratic = vector
                    .iter()
                    .zip(symmetric_tensor_vector(tensor[0], vector))
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                let phase = TWO_PI
                    * hkl[0]
                        .iter()
                        .zip(position)
                        .map(|(index, coordinate)| f64::from(*index) * coordinate)
                        .sum::<f64>();
                let displacement = (-TWO_PI_SQUARED * quadratic).exp();
                (displacement * phase.cos(), displacement * phase.sin())
            })
            .fold((0.0, 0.0), |sum, value| (sum.0 + value.0, sum.1 + value.1));
        assert!((values.f_real[0] - 2.4 * expected.0).abs() < 3.0e-14);
        assert!((values.f_imag[0] - 2.4 * expected.1).abs() < 3.0e-14);
    }

    #[test]
    fn anisotropic_cell_derivatives_match_centered_differences() {
        let hkl = [[2, 1, 3]];
        let xyz = [[0.17, 0.23, 0.31]];
        let tensor = [[0.020, 0.013, 0.027, 0.002, 0.001, 0.003]];
        let evaluate = |cell| {
            calculate_structure_factor_dense(
                cell,
                &p1(),
                StructureFactorBatchView {
                    hkl: &hkl,
                    multiplicity: &[2],
                    fractional_xyz: &xyz,
                    occupancy: &[0.8],
                    u_iso_angstrom2: &[0.37],
                    anisotropic_mask: &[true],
                    u_aniso_cif_angstrom2: &tensor,
                    scattering_real: &[3.0],
                    scattering_imag: &[0.2],
                    d_scattering_real_d_s: &[0.0],
                    d_scattering_imag_d_s: &[0.0],
                    correction: &[1.0],
                    d_correction_d_q_squared: &[0.0],
                    scale: 1.0,
                    coordinate_tolerance: 1.0e-10,
                },
            )
            .expect("anisotropic dense result")
        };
        let cell = UnitCell {
            a_angstrom: 4.3,
            b_angstrom: 5.1,
            c_angstrom: 6.2,
            alpha_deg: 78.0,
            beta_deg: 83.0,
            gamma_deg: 71.0,
        };
        let actual = evaluate(cell);
        for parameter in 0..CELL_PARAMETER_COUNT {
            let step = if parameter < 3 { 1.0e-6 } else { 1.0e-5 };
            let mut plus = cell;
            let mut minus = cell;
            perturb_cell(&mut plus, parameter, step);
            perturb_cell(&mut minus, parameter, -step);
            let plus = evaluate(plus);
            let minus = evaluate(minus);
            let expected_real = (plus.values.f_real[0] - minus.values.f_real[0]) / (2.0 * step);
            let expected_intensity =
                (plus.values.intensity[0] - minus.values.intensity[0]) / (2.0 * step);
            assert!((actual.d_f_real[parameter] - expected_real).abs() < 2.0e-8);
            assert!((actual.d_intensity[parameter] - expected_intensity).abs() < 3.0e-7);
        }
        assert_eq!(actual.d_f_real[actual.layout.u_iso(0)].to_bits(), 0);
        assert_eq!(actual.d_f_imag[actual.layout.u_iso(0)].to_bits(), 0);
        assert_eq!(actual.d_intensity[actual.layout.u_iso(0)].to_bits(), 0);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn dense_derivatives_include_symmetry_scattering_and_correction_chains() {
        let group = inversion();
        let hkl = [[2, 1, 1], [1, 3, 2]];
        let multiplicity = [4, 2];
        let xyz = [[0.17, 0.23, 0.31]];
        let occupancy = [0.72];
        let u_iso = [0.013];
        let scale = 1.6;

        let evaluate =
            |cell: UnitCell, xyz: &[[f64; 3]], occupancy: &[f64], u_iso: &[f64], scale| {
                let geometry = cell.geometry().expect("geometry");
                let q_squared: Vec<f64> = hkl
                    .iter()
                    .map(|&indices| geometry.q_squared(indices))
                    .collect();
                let s: Vec<f64> = q_squared.iter().map(|value| 0.5 * value.sqrt()).collect();
                let real: Vec<f64> = s.iter().map(|value| 4.0 - 0.3 * value).collect();
                let imag: Vec<f64> = s.iter().map(|value| 0.2 + 0.1 * value).collect();
                let d_real = vec![-0.3; hkl.len()];
                let d_imag = vec![0.1; hkl.len()];
                let correction: Vec<f64> =
                    q_squared.iter().map(|value| 1.0 + 0.2 * value).collect();
                let d_correction = vec![0.2; hkl.len()];
                calculate_structure_factor_dense(
                    cell,
                    &group,
                    StructureFactorBatchView {
                        hkl: &hkl,
                        multiplicity: &multiplicity,
                        fractional_xyz: xyz,
                        occupancy,
                        u_iso_angstrom2: u_iso,
                        anisotropic_mask: &[false],
                        u_aniso_cif_angstrom2: &[[0.0; 6]],
                        scattering_real: &real,
                        scattering_imag: &imag,
                        d_scattering_real_d_s: &d_real,
                        d_scattering_imag_d_s: &d_imag,
                        correction: &correction,
                        d_correction_d_q_squared: &d_correction,
                        scale,
                        coordinate_tolerance: 1.0e-10,
                    },
                )
                .expect("dense result")
            };

        let cell = cubic_cell(4.8);
        let actual = evaluate(cell, &xyz, &occupancy, &u_iso, scale);
        let layout = actual.layout;
        let step = 1.0e-6;
        for parameter in 0..layout.parameter_count() {
            let mut plus_cell = cell;
            let mut minus_cell = cell;
            let mut plus_xyz = xyz;
            let mut minus_xyz = xyz;
            let mut plus_occupancy = occupancy;
            let mut minus_occupancy = occupancy;
            let mut plus_u = u_iso;
            let mut minus_u = u_iso;
            let mut plus_scale = scale;
            let mut minus_scale = scale;
            match parameter {
                0..=5 => {
                    perturb_cell(&mut plus_cell, parameter, step);
                    perturb_cell(&mut minus_cell, parameter, -step);
                }
                value if (CELL_PARAMETER_COUNT..CELL_PARAMETER_COUNT + 3).contains(&value) => {
                    let component = value - CELL_PARAMETER_COUNT;
                    plus_xyz[0][component] += step;
                    minus_xyz[0][component] -= step;
                }
                value if value == layout.occupancy(0) => {
                    plus_occupancy[0] += step;
                    minus_occupancy[0] -= step;
                }
                value if value == layout.u_iso(0) => {
                    plus_u[0] += step;
                    minus_u[0] -= step;
                }
                value if value == layout.scale() => {
                    plus_scale += step;
                    minus_scale -= step;
                }
                _ => continue,
            }
            let plus = evaluate(plus_cell, &plus_xyz, &plus_occupancy, &plus_u, plus_scale);
            let minus = evaluate(
                minus_cell,
                &minus_xyz,
                &minus_occupancy,
                &minus_u,
                minus_scale,
            );
            for reflection in 0..hkl.len() {
                let index = parameter * hkl.len() + reflection;
                let expected_real = (plus.values.f_real[reflection]
                    - minus.values.f_real[reflection])
                    / (2.0 * step);
                let expected_imag = (plus.values.f_imag[reflection]
                    - minus.values.f_imag[reflection])
                    / (2.0 * step);
                let expected_intensity = (plus.values.intensity[reflection]
                    - minus.values.intensity[reflection])
                    / (2.0 * step);
                assert!((actual.d_f_real[index] - expected_real).abs() < 3.0e-7);
                assert!((actual.d_f_imag[index] - expected_imag).abs() < 3.0e-7);
                assert!((actual.d_intensity[index] - expected_intensity).abs() < 3.0e-5);
            }
        }
    }

    #[test]
    fn jvp_and_vjp_match_dense_and_are_adjoint_consistent() {
        let hkl = [[1, 2, 1], [2, 1, 3], [3, 2, 1]];
        let multiplicity = [2, 4, 2];
        let xyz = [[0.17, 0.23, 0.31]];
        let occupancy = [0.81];
        let u_iso = [0.014];
        let scattering_real = [3.9, 3.7, 3.5];
        let scattering_imag = [0.1, 0.12, 0.15];
        let d_scattering_real = [-0.2, -0.2, -0.2];
        let d_scattering_imag = [0.05, 0.05, 0.05];
        let correction = [1.1, 1.2, 1.3];
        let d_correction = [0.2, 0.2, 0.2];
        let batch = StructureFactorBatchView {
            hkl: &hkl,
            multiplicity: &multiplicity,
            fractional_xyz: &xyz,
            occupancy: &occupancy,
            u_iso_angstrom2: &u_iso,
            anisotropic_mask: &[false],
            u_aniso_cif_angstrom2: &[[0.0; 6]],
            scattering_real: &scattering_real,
            scattering_imag: &scattering_imag,
            d_scattering_real_d_s: &d_scattering_real,
            d_scattering_imag_d_s: &d_scattering_imag,
            correction: &correction,
            d_correction_d_q_squared: &d_correction,
            scale: 1.4,
            coordinate_tolerance: 1.0e-10,
        };
        let cell = cubic_cell(4.7);
        let group = inversion();
        let dense = calculate_structure_factor_dense(cell, &group, batch).expect("dense");
        let tangent: Vec<f64> = (0..dense.layout.parameter_count())
            .map(|index| f64::from(u32::try_from(index + 1).expect("small index")) * 1.0e-4)
            .collect();
        let weights = [0.7, -0.2, 1.1];
        let jvp = calculate_structure_factor_jvp(cell, &group, batch, &tangent).expect("JVP");
        let vjp =
            calculate_structure_factor_intensity_vjp(cell, &group, batch, &weights).expect("VJP");
        for reflection in 0..hkl.len() {
            let expected_f_real = tangent
                .iter()
                .enumerate()
                .map(|(parameter, value)| {
                    value * dense.d_f_real[parameter * hkl.len() + reflection]
                })
                .sum::<f64>();
            let expected_f_imag = tangent
                .iter()
                .enumerate()
                .map(|(parameter, value)| {
                    value * dense.d_f_imag[parameter * hkl.len() + reflection]
                })
                .sum::<f64>();
            let expected_intensity = tangent
                .iter()
                .enumerate()
                .map(|(parameter, value)| {
                    value * dense.d_intensity[parameter * hkl.len() + reflection]
                })
                .sum::<f64>();
            assert!((jvp.d_f_real[reflection] - expected_f_real).abs() < 2.0e-13);
            assert!((jvp.d_f_imag[reflection] - expected_f_imag).abs() < 2.0e-13);
            assert!((jvp.d_intensity[reflection] - expected_intensity).abs() < 2.0e-11);
        }
        for (parameter, actual) in vjp.gradient.iter().copied().enumerate() {
            let expected = weights
                .iter()
                .enumerate()
                .map(|(reflection, weight)| {
                    weight * dense.d_intensity[parameter * hkl.len() + reflection]
                })
                .sum::<f64>();
            assert!((actual - expected).abs() < 2.0e-10);
        }
        let forward_dot = jvp
            .d_intensity
            .iter()
            .zip(weights)
            .map(|(value, weight)| value * weight)
            .sum::<f64>();
        let reverse_dot = tangent
            .iter()
            .zip(vjp.gradient)
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>();
        assert!((forward_dot - reverse_dot).abs() < 2.0e-12);
    }

    fn perturb_cell(cell: &mut UnitCell, parameter: usize, change: f64) {
        let value = match parameter {
            0 => &mut cell.a_angstrom,
            1 => &mut cell.b_angstrom,
            2 => &mut cell.c_angstrom,
            3 => &mut cell.alpha_deg,
            4 => &mut cell.beta_deg,
            5 => &mut cell.gamma_deg,
            _ => panic!("invalid cell parameter"),
        };
        *value += change;
    }

    #[test]
    fn invalid_shapes_zero_reflections_and_metric_mismatch_are_errors() {
        let base = StructureFactorBatchView {
            hkl: &[[1, 0, 0]],
            multiplicity: &[1],
            fractional_xyz: &[[0.0, 0.0, 0.0]],
            occupancy: &[1.0],
            u_iso_angstrom2: &[0.0],
            anisotropic_mask: &[false],
            u_aniso_cif_angstrom2: &[[0.0; 6]],
            scattering_real: &[1.0],
            scattering_imag: &[0.0],
            d_scattering_real_d_s: &[0.0],
            d_scattering_imag_d_s: &[0.0],
            correction: &[1.0],
            d_correction_d_q_squared: &[0.0],
            scale: 1.0,
            coordinate_tolerance: 1.0e-10,
        };
        let bad_scattering = StructureFactorBatchView {
            scattering_real: &[],
            ..base
        };
        assert_eq!(
            calculate_structure_factor_values(cubic_cell(4.0), &p1(), bad_scattering),
            Err(StructureFactorBatchError::ScatteringShapeMismatch)
        );
        let zero = [[0, 0, 0]];
        let zero_reflection = StructureFactorBatchView { hkl: &zero, ..base };
        assert_eq!(
            calculate_structure_factor_values(cubic_cell(4.0), &p1(), zero_reflection),
            Err(StructureFactorBatchError::ZeroReflection)
        );
        let invalid_tensor = StructureFactorBatchView {
            anisotropic_mask: &[true],
            u_aniso_cif_angstrom2: &[[-0.01, 0.01, 0.01, 0.0, 0.0, 0.0]],
            ..base
        };
        assert_eq!(
            calculate_structure_factor_values(cubic_cell(4.0), &p1(), invalid_tensor),
            Err(StructureFactorBatchError::InvalidPhysicalParameter)
        );
        let incompatible_cell = UnitCell {
            a_angstrom: 4.0,
            b_angstrom: 5.0,
            c_angstrom: 6.0,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        };
        assert_eq!(
            calculate_structure_factor_values(incompatible_cell, &axis_swap_group(), base),
            Err(StructureFactorBatchError::CellSymmetryMismatch)
        );
        assert_eq!(
            calculate_structure_factor_jvp(cubic_cell(4.0), &p1(), base, &[]),
            Err(StructureFactorBatchError::TangentLengthMismatch)
        );
        assert_eq!(
            calculate_structure_factor_intensity_vjp(cubic_cell(4.0), &p1(), base, &[]),
            Err(StructureFactorBatchError::WeightLengthMismatch)
        );
    }
}
