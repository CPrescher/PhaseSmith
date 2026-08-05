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
}

#[derive(Clone, Copy)]
struct SiteTerms {
    symmetry_real: f64,
    symmetry_imag: f64,
    d_symmetry_real: [f64; 3],
    d_symmetry_imag: [f64; 3],
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
    if batch.occupancy.len() != site_count || batch.u_iso_angstrom2.len() != site_count {
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
        || batch.u_iso_angstrom2.iter().any(|value| *value < 0.0)
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
    Ok(ValidatedStructure {
        batch,
        geometry,
        expanded,
        site_offsets,
        layout: P1ParameterLayout { site_count },
    })
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
    let (q_squared, _) = validated
        .geometry
        .q_squared_and_derivatives(batch.hkl[reflection]);
    let s = 0.5 * q_squared.sqrt();
    let mut f_real = 0.0;
    let mut f_imag = 0.0;
    for site in 0..validated.layout.site_count {
        let terms = symmetry_terms(validated, batch.hkl[reflection], site);
        let (base_real, base_imag) = site_base(validated, reflection, site, q_squared, terms);
        f_real += batch.occupancy[site] * base_real;
        f_imag += batch.occupancy[site] * base_imag;
    }
    set_values(values, batch, reflection, q_squared, s, f_real, f_imag);
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
    let terms = symmetry_terms(validated, batch.hkl[reflection], site);
    let (base_real, base_imag) = site_base(validated, reflection, site, q_squared, terms);
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
    let displacement = (-TWO_PI_SQUARED * batch.u_iso_angstrom2[site] * q_squared).exp();
    for (parameter, d_q) in d_q_squared.into_iter().enumerate() {
        let d_s = d_q / (4.0 * root_q);
        let d_amplitude = (
            d_scattering.0 * d_s
                - TWO_PI_SQUARED * batch.u_iso_angstrom2[site] * scattering.0 * d_q,
            d_scattering.1 * d_s
                - TWO_PI_SQUARED * batch.u_iso_angstrom2[site] * scattering.1 * d_q,
        );
        let rotated = complex_multiply(d_amplitude, (terms.symmetry_real, terms.symmetry_imag));
        set_f_derivative(
            result,
            parameter,
            reflection,
            reflection_count,
            occupancy * displacement * rotated.0,
            occupancy * displacement * rotated.1,
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
            occupancy * displacement * rotated.0,
            occupancy * displacement * rotated.1,
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
    set_f_derivative(
        result,
        validated.layout.u_iso(site),
        reflection,
        reflection_count,
        -TWO_PI_SQUARED * q_squared * contribution.0,
        -TWO_PI_SQUARED * q_squared * contribution.1,
    );
    contribution
}

fn symmetry_terms(validated: &ValidatedStructure<'_>, hkl: [i32; 3], site: usize) -> SiteTerms {
    let mut result = SiteTerms {
        symmetry_real: 0.0,
        symmetry_imag: 0.0,
        d_symmetry_real: [0.0; 3],
        d_symmetry_imag: [0.0; 3],
    };
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
        result.symmetry_real += cos_phase;
        result.symmetry_imag += sin_phase;
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
            *d_real -= phase_derivative * sin_phase;
            *d_imag += phase_derivative * cos_phase;
        }
    }
    result
}

fn site_base(
    validated: &ValidatedStructure<'_>,
    reflection: usize,
    site: usize,
    q_squared: f64,
    terms: SiteTerms,
) -> (f64, f64) {
    let index = reflection * validated.layout.site_count + site;
    let rotated = complex_multiply(
        (
            validated.batch.scattering_real[index],
            validated.batch.scattering_imag[index],
        ),
        (terms.symmetry_real, terms.symmetry_imag),
    );
    let displacement = (-TWO_PI_SQUARED * validated.batch.u_iso_angstrom2[site] * q_squared).exp();
    (displacement * rotated.0, displacement * rotated.1)
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
                    .map(|&indices| geometry.q_squared_and_derivatives(indices).0)
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
    }
}
