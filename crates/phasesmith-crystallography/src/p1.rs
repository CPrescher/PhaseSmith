//! P1 structure factors and analytical derivative products.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::cell::{CELL_PARAMETER_COUNT, CellError, CellGeometry, UnitCell};

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const TWO_PI_SQUARED: f64 = 2.0 * std::f64::consts::PI * std::f64::consts::PI;

/// Borrowed reflection/site arrays for one P1 calculation.
#[derive(Clone, Copy, Debug)]
pub struct P1BatchView<'a> {
    /// Miller indices, one row per reflection.
    pub hkl: &'a [[i32; 3]],
    /// Fractional coordinates, one row per atom site.
    pub fractional_xyz: &'a [[f64; 3]],
    /// Fractional site occupancies.
    pub occupancy: &'a [f64],
    /// Isotropic displacement values in square ångströms.
    pub u_iso_angstrom2: &'a [f64],
    /// Reflection-major real scattering amplitudes, length `R * S`.
    pub scattering_real: &'a [f64],
    /// Reflection-major imaginary scattering amplitudes, length `R * S`.
    pub scattering_imag: &'a [f64],
    /// Non-negative phase scale applied to `|F|^2`.
    pub scale: f64,
}

/// Stable native parameter layout for a P1 batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct P1ParameterLayout {
    /// Number of atom sites.
    pub site_count: usize,
}

impl P1ParameterLayout {
    /// Total parameter count: six cell, three coordinates per site, one
    /// occupancy per site, one `U_iso` per site, and one scale.
    #[must_use]
    pub const fn parameter_count(self) -> usize {
        CELL_PARAMETER_COUNT + 5 * self.site_count + 1
    }

    /// Index of fractional coordinate `component` for `site`.
    #[must_use]
    pub const fn coordinate(self, site: usize, component: usize) -> usize {
        CELL_PARAMETER_COUNT + 3 * site + component
    }

    /// Index of occupancy for `site`.
    #[must_use]
    pub const fn occupancy(self, site: usize) -> usize {
        CELL_PARAMETER_COUNT + 3 * self.site_count + site
    }

    /// Index of `U_iso` for `site`.
    #[must_use]
    pub const fn u_iso(self, site: usize) -> usize {
        CELL_PARAMETER_COUNT + 4 * self.site_count + site
    }

    /// Index of the phase scale.
    #[must_use]
    pub const fn scale(self) -> usize {
        CELL_PARAMETER_COUNT + 5 * self.site_count
    }
}

/// Structure-factor values for a reflection batch.
#[derive(Clone, Debug, PartialEq)]
pub struct P1Values {
    /// Real part of `F_h`.
    pub f_real: Vec<f64>,
    /// Imaginary part of `F_h`.
    pub f_imag: Vec<f64>,
    /// `scale * |F_h|^2` with multiplicity and corrections equal to one.
    pub intensity: Vec<f64>,
}

/// Values and dense, parameter-major analytical derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct P1DenseResult {
    /// Calculated values.
    pub values: P1Values,
    /// Parameter-major derivative of real `F`, shape `(P, R)`.
    pub d_f_real: Vec<f64>,
    /// Parameter-major derivative of imaginary `F`, shape `(P, R)`.
    pub d_f_imag: Vec<f64>,
    /// Parameter-major derivative of intensity, shape `(P, R)`.
    pub d_intensity: Vec<f64>,
    /// Parameter layout for derivative rows.
    pub layout: P1ParameterLayout,
}

/// Values and one forward derivative product.
#[derive(Clone, Debug, PartialEq)]
pub struct P1JvpResult {
    /// Calculated values.
    pub values: P1Values,
    /// Directional derivative of real `F`.
    pub d_f_real: Vec<f64>,
    /// Directional derivative of imaginary `F`.
    pub d_f_imag: Vec<f64>,
    /// Directional derivative of intensity.
    pub d_intensity: Vec<f64>,
}

/// Values and one reverse intensity derivative product.
#[derive(Clone, Debug, PartialEq)]
pub struct P1VjpResult {
    /// Calculated values.
    pub values: P1Values,
    /// `J_intensity^T weights` in stable parameter order.
    pub gradient: Vec<f64>,
    /// Parameter layout for the gradient.
    pub layout: P1ParameterLayout,
}

/// Invalid P1 batch or derivative input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum P1BatchError {
    /// The unit cell is invalid.
    Cell(CellError),
    /// Site arrays do not have a common length.
    SiteLengthMismatch,
    /// Scattering arrays are not exactly reflection count times site count.
    ScatteringShapeMismatch,
    /// A coordinate, occupancy, displacement, amplitude, or scale is non-finite.
    NonFiniteInput,
    /// Occupancy, displacement, or scale is negative.
    NegativePhysicalParameter,
    /// A tangent does not match the parameter layout.
    TangentLengthMismatch,
    /// Reverse weights do not match the reflection count.
    WeightLengthMismatch,
    /// A dense derivative allocation would overflow addressable memory.
    AllocationOverflow,
}

impl Display for P1BatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cell(error) => Display::fmt(error, formatter),
            Self::SiteLengthMismatch => {
                formatter.write_str("P1 site arrays must have equal length")
            }
            Self::ScatteringShapeMismatch => formatter
                .write_str("P1 scattering arrays must have reflection_count * site_count elements"),
            Self::NonFiniteInput => {
                formatter.write_str("P1 inputs must contain only finite values")
            }
            Self::NegativePhysicalParameter => {
                formatter.write_str("P1 occupancy, U_iso, and scale must be non-negative")
            }
            Self::TangentLengthMismatch => {
                formatter.write_str("P1 tangent length must equal the parameter count")
            }
            Self::WeightLengthMismatch => {
                formatter.write_str("P1 reverse weights must match the reflection count")
            }
            Self::AllocationOverflow => formatter.write_str("P1 derivative allocation overflow"),
        }
    }
}

impl Error for P1BatchError {}

impl From<CellError> for P1BatchError {
    fn from(value: CellError) -> Self {
        Self::Cell(value)
    }
}

struct ValidatedP1<'a> {
    batch: P1BatchView<'a>,
    geometry: CellGeometry,
    layout: P1ParameterLayout,
}

/// Calculate P1 values without structural derivative storage.
///
/// # Errors
///
/// Returns [`P1BatchError`] for an invalid cell, inconsistent array shapes,
/// non-finite inputs, negative physical parameters, or size overflow.
pub fn calculate_p1_values(
    cell: UnitCell,
    batch: P1BatchView<'_>,
) -> Result<P1Values, P1BatchError> {
    let validated = validate(cell, batch)?;
    let mut values = empty_values(batch.hkl.len());
    for reflection in 0..batch.hkl.len() {
        let (f_real, f_imag) = reflection_value(&validated, reflection);
        values.f_real[reflection] = f_real;
        values.f_imag[reflection] = f_imag;
        values.intensity[reflection] = batch.scale * (f_real * f_real + f_imag * f_imag);
    }
    Ok(values)
}

/// Calculate P1 values and a dense parameter-major Jacobian.
///
/// # Errors
///
/// Returns [`P1BatchError`] for an invalid cell, inconsistent array shapes,
/// non-finite inputs, negative physical parameters, or size overflow.
pub fn calculate_p1_dense(
    cell: UnitCell,
    batch: P1BatchView<'_>,
) -> Result<P1DenseResult, P1BatchError> {
    let validated = validate(cell, batch)?;
    let reflection_count = batch.hkl.len();
    let parameter_count = validated.layout.parameter_count();
    let element_count = parameter_count
        .checked_mul(reflection_count)
        .ok_or(P1BatchError::AllocationOverflow)?;
    let mut result = P1DenseResult {
        values: empty_values(reflection_count),
        d_f_real: vec![0.0; element_count],
        d_f_imag: vec![0.0; element_count],
        d_intensity: vec![0.0; element_count],
        layout: validated.layout,
    };
    for reflection in 0..reflection_count {
        calculate_dense_reflection(&validated, reflection, &mut result);
    }
    Ok(result)
}

/// Calculate values and one forward directional derivative without building a
/// dense Jacobian.
///
/// # Errors
///
/// Returns [`P1BatchError`] for invalid batch data or when `tangent` does not
/// match the stable parameter layout.
pub fn calculate_p1_jvp(
    cell: UnitCell,
    batch: P1BatchView<'_>,
    tangent: &[f64],
) -> Result<P1JvpResult, P1BatchError> {
    let validated = validate(cell, batch)?;
    if tangent.len() != validated.layout.parameter_count() {
        return Err(P1BatchError::TangentLengthMismatch);
    }
    if tangent.iter().any(|value| !value.is_finite()) {
        return Err(P1BatchError::NonFiniteInput);
    }
    let reflection_count = batch.hkl.len();
    let mut result = P1JvpResult {
        values: empty_values(reflection_count),
        d_f_real: vec![0.0; reflection_count],
        d_f_imag: vec![0.0; reflection_count],
        d_intensity: vec![0.0; reflection_count],
    };
    for reflection in 0..reflection_count {
        calculate_jvp_reflection(&validated, reflection, tangent, &mut result);
    }
    Ok(result)
}

/// Calculate values and `J_intensity^T weights` without building a dense
/// Jacobian.
///
/// # Errors
///
/// Returns [`P1BatchError`] for invalid batch data or when `weights` does not
/// match the reflection count.
pub fn calculate_p1_intensity_vjp(
    cell: UnitCell,
    batch: P1BatchView<'_>,
    weights: &[f64],
) -> Result<P1VjpResult, P1BatchError> {
    let validated = validate(cell, batch)?;
    if weights.len() != batch.hkl.len() {
        return Err(P1BatchError::WeightLengthMismatch);
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(P1BatchError::NonFiniteInput);
    }
    let mut result = P1VjpResult {
        values: empty_values(batch.hkl.len()),
        gradient: vec![0.0; validated.layout.parameter_count()],
        layout: validated.layout,
    };
    let mut site_amplitudes = Vec::new();
    site_amplitudes
        .try_reserve_exact(validated.layout.site_count)
        .map_err(|_| P1BatchError::AllocationOverflow)?;
    for (reflection, weight) in weights.iter().copied().enumerate() {
        calculate_vjp_reflection(
            &validated,
            reflection,
            weight,
            &mut site_amplitudes,
            &mut result,
        );
    }
    Ok(result)
}

fn validate(cell: UnitCell, batch: P1BatchView<'_>) -> Result<ValidatedP1<'_>, P1BatchError> {
    let geometry = cell.geometry()?;
    let site_count = batch.fractional_xyz.len();
    if batch.occupancy.len() != site_count || batch.u_iso_angstrom2.len() != site_count {
        return Err(P1BatchError::SiteLengthMismatch);
    }
    let scattering_count = batch
        .hkl
        .len()
        .checked_mul(site_count)
        .ok_or(P1BatchError::AllocationOverflow)?;
    if batch.scattering_real.len() != scattering_count
        || batch.scattering_imag.len() != scattering_count
    {
        return Err(P1BatchError::ScatteringShapeMismatch);
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
            .any(|value| !value.is_finite())
    {
        return Err(P1BatchError::NonFiniteInput);
    }
    if batch.scale < 0.0
        || batch.occupancy.iter().any(|value| *value < 0.0)
        || batch.u_iso_angstrom2.iter().any(|value| *value < 0.0)
    {
        return Err(P1BatchError::NegativePhysicalParameter);
    }
    Ok(ValidatedP1 {
        batch,
        geometry,
        layout: P1ParameterLayout { site_count },
    })
}

fn empty_values(reflection_count: usize) -> P1Values {
    P1Values {
        f_real: vec![0.0; reflection_count],
        f_imag: vec![0.0; reflection_count],
        intensity: vec![0.0; reflection_count],
    }
}

fn atom_amplitude(
    validated: &ValidatedP1<'_>,
    reflection: usize,
    site: usize,
    q_squared: f64,
) -> (f64, f64, f64, f64) {
    let batch = validated.batch;
    let scattering_index = reflection * validated.layout.site_count + site;
    let phase = TWO_PI
        * batch.hkl[reflection]
            .iter()
            .zip(batch.fractional_xyz[site])
            .map(|(index, coordinate)| f64::from(*index) * coordinate)
            .sum::<f64>();
    let (sin_phase, cos_phase) = phase.sin_cos();
    let scattering_real = batch.scattering_real[scattering_index];
    let scattering_imag = batch.scattering_imag[scattering_index];
    let rotated_real = scattering_real * cos_phase - scattering_imag * sin_phase;
    let rotated_imag = scattering_real * sin_phase + scattering_imag * cos_phase;
    let displacement = (-TWO_PI_SQUARED * batch.u_iso_angstrom2[site] * q_squared).exp();
    let base_real = displacement * rotated_real;
    let base_imag = displacement * rotated_imag;
    (
        base_real,
        base_imag,
        batch.occupancy[site] * base_real,
        batch.occupancy[site] * base_imag,
    )
}

fn reflection_value(validated: &ValidatedP1<'_>, reflection: usize) -> (f64, f64) {
    let q_squared = validated
        .geometry
        .q_squared(validated.batch.hkl[reflection]);
    let mut f_real = 0.0;
    let mut f_imag = 0.0;
    for site in 0..validated.layout.site_count {
        let (_, _, contribution_real, contribution_imag) =
            atom_amplitude(validated, reflection, site, q_squared);
        f_real += contribution_real;
        f_imag += contribution_imag;
    }
    (f_real, f_imag)
}

fn calculate_dense_reflection(
    validated: &ValidatedP1<'_>,
    reflection: usize,
    result: &mut P1DenseResult,
) {
    let batch = validated.batch;
    let reflection_count = batch.hkl.len();
    let (q_squared, d_q_squared) = validated
        .geometry
        .q_squared_and_derivatives(batch.hkl[reflection]);
    let mut f_real = 0.0;
    let mut f_imag = 0.0;
    for site in 0..validated.layout.site_count {
        let (base_real, base_imag, contribution_real, contribution_imag) =
            atom_amplitude(validated, reflection, site, q_squared);
        f_real += contribution_real;
        f_imag += contribution_imag;
        for (parameter, d_q) in d_q_squared.iter().copied().enumerate() {
            let factor = -TWO_PI_SQUARED * batch.u_iso_angstrom2[site] * d_q;
            set_derivative(
                result,
                parameter,
                reflection,
                reflection_count,
                factor * contribution_real,
                factor * contribution_imag,
            );
        }
        for component in 0..3 {
            let factor = TWO_PI * f64::from(batch.hkl[reflection][component]);
            set_derivative(
                result,
                validated.layout.coordinate(site, component),
                reflection,
                reflection_count,
                -factor * contribution_imag,
                factor * contribution_real,
            );
        }
        set_derivative(
            result,
            validated.layout.occupancy(site),
            reflection,
            reflection_count,
            base_real,
            base_imag,
        );
        let displacement_factor = -TWO_PI_SQUARED * q_squared;
        set_derivative(
            result,
            validated.layout.u_iso(site),
            reflection,
            reflection_count,
            displacement_factor * contribution_real,
            displacement_factor * contribution_imag,
        );
    }
    let norm = f_real * f_real + f_imag * f_imag;
    result.values.f_real[reflection] = f_real;
    result.values.f_imag[reflection] = f_imag;
    result.values.intensity[reflection] = batch.scale * norm;
    for parameter in 0..validated.layout.parameter_count() {
        let index = parameter * reflection_count + reflection;
        result.d_intensity[index] =
            2.0 * batch.scale * (f_real * result.d_f_real[index] + f_imag * result.d_f_imag[index]);
    }
    result.d_intensity[validated.layout.scale() * reflection_count + reflection] = norm;
}

fn set_derivative(
    result: &mut P1DenseResult,
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

fn calculate_jvp_reflection(
    validated: &ValidatedP1<'_>,
    reflection: usize,
    tangent: &[f64],
    result: &mut P1JvpResult,
) {
    let batch = validated.batch;
    let (q_squared, d_q_squared) = validated
        .geometry
        .q_squared_and_derivatives(batch.hkl[reflection]);
    let d_q_direction = d_q_squared
        .iter()
        .zip(&tangent[..CELL_PARAMETER_COUNT])
        .map(|(derivative, direction)| derivative * direction)
        .sum::<f64>();
    let mut f_real = 0.0;
    let mut f_imag = 0.0;
    let mut d_f_real = 0.0;
    let mut d_f_imag = 0.0;
    for site in 0..validated.layout.site_count {
        let (base_real, base_imag, contribution_real, contribution_imag) =
            atom_amplitude(validated, reflection, site, q_squared);
        f_real += contribution_real;
        f_imag += contribution_imag;
        let phase_direction = TWO_PI
            * (0..3)
                .map(|component| {
                    f64::from(batch.hkl[reflection][component])
                        * tangent[validated.layout.coordinate(site, component)]
                })
                .sum::<f64>();
        let displacement_direction = -TWO_PI_SQUARED
            * (batch.u_iso_angstrom2[site] * d_q_direction
                + q_squared * tangent[validated.layout.u_iso(site)]);
        let occupancy_direction = tangent[validated.layout.occupancy(site)];
        d_f_real += occupancy_direction * base_real + displacement_direction * contribution_real
            - phase_direction * contribution_imag;
        d_f_imag += occupancy_direction * base_imag
            + displacement_direction * contribution_imag
            + phase_direction * contribution_real;
    }
    let norm = f_real * f_real + f_imag * f_imag;
    result.values.f_real[reflection] = f_real;
    result.values.f_imag[reflection] = f_imag;
    result.values.intensity[reflection] = batch.scale * norm;
    result.d_f_real[reflection] = d_f_real;
    result.d_f_imag[reflection] = d_f_imag;
    result.d_intensity[reflection] = 2.0 * batch.scale * (f_real * d_f_real + f_imag * d_f_imag)
        + tangent[validated.layout.scale()] * norm;
}

fn calculate_vjp_reflection(
    validated: &ValidatedP1<'_>,
    reflection: usize,
    weight: f64,
    site_amplitudes: &mut Vec<(f64, f64, f64, f64)>,
    result: &mut P1VjpResult,
) {
    let batch = validated.batch;
    let (q_squared, d_q_squared) = validated
        .geometry
        .q_squared_and_derivatives(batch.hkl[reflection]);
    let mut f_real = 0.0;
    let mut f_imag = 0.0;
    site_amplitudes.clear();
    for site in 0..validated.layout.site_count {
        let amplitude = atom_amplitude(validated, reflection, site, q_squared);
        f_real += amplitude.2;
        f_imag += amplitude.3;
        site_amplitudes.push(amplitude);
    }
    let norm = f_real * f_real + f_imag * f_imag;
    result.values.f_real[reflection] = f_real;
    result.values.f_imag[reflection] = f_imag;
    result.values.intensity[reflection] = batch.scale * norm;
    let intensity_factor = 2.0 * batch.scale * weight;
    for (site, &(base_real, base_imag, contribution_real, contribution_imag)) in
        site_amplitudes.iter().enumerate()
    {
        for (parameter, d_q) in d_q_squared.iter().copied().enumerate() {
            let factor = -TWO_PI_SQUARED * batch.u_iso_angstrom2[site] * d_q;
            result.gradient[parameter] += intensity_factor
                * factor
                * (f_real * contribution_real + f_imag * contribution_imag);
        }
        for component in 0..3 {
            let factor = TWO_PI * f64::from(batch.hkl[reflection][component]);
            let d_real = -factor * contribution_imag;
            let d_imag = factor * contribution_real;
            result.gradient[validated.layout.coordinate(site, component)] +=
                intensity_factor * (f_real * d_real + f_imag * d_imag);
        }
        result.gradient[validated.layout.occupancy(site)] +=
            intensity_factor * (f_real * base_real + f_imag * base_imag);
        let displacement_factor = -TWO_PI_SQUARED * q_squared;
        result.gradient[validated.layout.u_iso(site)] += intensity_factor
            * displacement_factor
            * (f_real * contribution_real + f_imag * contribution_imag);
    }
    result.gradient[validated.layout.scale()] += weight * norm;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> UnitCell {
        UnitCell {
            a_angstrom: 4.2,
            b_angstrom: 5.1,
            c_angstrom: 6.3,
            alpha_deg: 79.0,
            beta_deg: 83.0,
            gamma_deg: 74.0,
        }
    }

    #[test]
    fn one_origin_atom_has_closed_form_value() {
        let hkl = [[1, 2, 3], [0, 0, 0]];
        let xyz = [[0.0, 0.0, 0.0]];
        let occupancy = [0.75];
        let u_iso = [0.0];
        let real = [2.0, 3.0];
        let imag = [-0.5, 0.25];
        let values = calculate_p1_values(
            cell(),
            P1BatchView {
                hkl: &hkl,
                fractional_xyz: &xyz,
                occupancy: &occupancy,
                u_iso_angstrom2: &u_iso,
                scattering_real: &real,
                scattering_imag: &imag,
                scale: 2.0,
            },
        )
        .expect("P1 values");
        assert_eq!(values.f_real, vec![1.5, 2.25]);
        assert_eq!(values.f_imag, vec![-0.375, 0.1875]);
        assert!((values.intensity[0] - 2.0 * (1.5_f64.powi(2) + 0.375_f64.powi(2))).abs() < 1e-15);
    }

    #[test]
    fn lattice_translation_and_origin_shift_preserve_intensity() {
        let hkl = [[1, -2, 3], [2, 1, -1]];
        let occupancy = [0.8, 0.6];
        let u_iso = [0.01, 0.02];
        let real = [2.0, 1.0, 1.5, 0.7];
        let imag = [0.1, -0.2, 0.05, 0.15];
        let original_xyz = [[0.17, 0.29, 0.43], [0.61, 0.11, 0.37]];
        let shifted_xyz = [[1.37, -0.01, 2.83], [1.81, -0.19, 2.77]];
        let original = calculate_p1_values(
            cell(),
            P1BatchView {
                hkl: &hkl,
                fractional_xyz: &original_xyz,
                occupancy: &occupancy,
                u_iso_angstrom2: &u_iso,
                scattering_real: &real,
                scattering_imag: &imag,
                scale: 1.3,
            },
        )
        .expect("original");
        let shifted = calculate_p1_values(
            cell(),
            P1BatchView {
                hkl: &hkl,
                fractional_xyz: &shifted_xyz,
                occupancy: &occupancy,
                u_iso_angstrom2: &u_iso,
                scattering_real: &real,
                scattering_imag: &imag,
                scale: 1.3,
            },
        )
        .expect("shifted");
        for (left, right) in original.intensity.iter().zip(shifted.intensity) {
            assert!((left - right).abs() < 2.0e-13 * left.abs().max(1.0));
        }
    }

    #[test]
    fn jvp_and_vjp_match_dense_jacobian() {
        let hkl = [[1, 0, 1], [2, -1, 3], [-1, 2, 2]];
        let xyz = [[0.17, 0.29, 0.43], [0.61, 0.11, 0.37]];
        let occupancy = [0.8, 0.6];
        let u_iso = [0.01, 0.02];
        let real = [2.0, 1.0, 1.5, 0.7, 0.9, 1.2];
        let imag = [0.1, -0.2, 0.05, 0.15, -0.1, 0.2];
        let batch = P1BatchView {
            hkl: &hkl,
            fractional_xyz: &xyz,
            occupancy: &occupancy,
            u_iso_angstrom2: &u_iso,
            scattering_real: &real,
            scattering_imag: &imag,
            scale: 1.3,
        };
        let dense = calculate_p1_dense(cell(), batch).expect("dense");
        let tangent: Vec<f64> = (0..dense.layout.parameter_count())
            .map(|index| {
                (f64::from(u32::try_from(index).expect("small test parameter count")) + 1.0)
                    * 1.0e-3
            })
            .collect();
        let weights = [0.3, -0.7, 1.1];
        let jvp = calculate_p1_jvp(cell(), batch, &tangent).expect("jvp");
        let vjp = calculate_p1_intensity_vjp(cell(), batch, &weights).expect("vjp");
        for reflection in 0..hkl.len() {
            let expected = (0..dense.layout.parameter_count())
                .map(|parameter| {
                    dense.d_intensity[parameter * hkl.len() + reflection] * tangent[parameter]
                })
                .sum::<f64>();
            assert!((jvp.d_intensity[reflection] - expected).abs() < 2.0e-12);
        }
        for parameter in 0..dense.layout.parameter_count() {
            let expected = (0..hkl.len())
                .map(|reflection| {
                    dense.d_intensity[parameter * hkl.len() + reflection] * weights[reflection]
                })
                .sum::<f64>();
            assert!((vjp.gradient[parameter] - expected).abs() < 2.0e-12);
        }
        let forward_dot = jvp
            .d_intensity
            .iter()
            .zip(weights)
            .map(|(value, weight)| value * weight)
            .sum::<f64>();
        let reverse_dot = vjp
            .gradient
            .iter()
            .zip(tangent)
            .map(|(value, direction)| value * direction)
            .sum::<f64>();
        assert!((forward_dot - reverse_dot).abs() < 2.0e-12);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn dense_derivatives_match_centered_differences() {
        let hkl = [[1, 2, -1]];
        let xyz = [[0.17, 0.29, 0.43]];
        let occupancy = [0.8];
        let u_iso = [0.01];
        let real = [2.0];
        let imag = [0.1];
        let base_cell = cell();
        let layout = P1ParameterLayout { site_count: 1 };
        let evaluate =
            |cell: UnitCell, xyz: &[[f64; 3]], occupancy: &[f64], u_iso: &[f64], scale| {
                calculate_p1_values(
                    cell,
                    P1BatchView {
                        hkl: &hkl,
                        fractional_xyz: xyz,
                        occupancy,
                        u_iso_angstrom2: u_iso,
                        scattering_real: &real,
                        scattering_imag: &imag,
                        scale,
                    },
                )
                .expect("values")
                .intensity[0]
            };
        let batch = P1BatchView {
            hkl: &hkl,
            fractional_xyz: &xyz,
            occupancy: &occupancy,
            u_iso_angstrom2: &u_iso,
            scattering_real: &real,
            scattering_imag: &imag,
            scale: 1.3,
        };
        let dense = calculate_p1_dense(base_cell, batch).expect("dense");
        for parameter in 0..layout.parameter_count() {
            let step = if parameter < 3 { 1e-6 } else { 1e-7 };
            let mut plus_cell = base_cell;
            let mut minus_cell = base_cell;
            let mut plus_xyz = xyz;
            let mut minus_xyz = xyz;
            let mut plus_occupancy = occupancy;
            let mut minus_occupancy = occupancy;
            let mut plus_u = u_iso;
            let mut minus_u = u_iso;
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
                value if value == layout.coordinate(0, 0) => {
                    plus_xyz[0][0] += step;
                    minus_xyz[0][0] -= step;
                }
                value if value == layout.coordinate(0, 1) => {
                    plus_xyz[0][1] += step;
                    minus_xyz[0][1] -= step;
                }
                value if value == layout.coordinate(0, 2) => {
                    plus_xyz[0][2] += step;
                    minus_xyz[0][2] -= step;
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
                _ => unreachable!(),
            }
            let plus = evaluate(plus_cell, &plus_xyz, &plus_occupancy, &plus_u, plus_scale);
            let minus = evaluate(
                minus_cell,
                &minus_xyz,
                &minus_occupancy,
                &minus_u,
                minus_scale,
            );
            let finite = (plus - minus) / (2.0 * step);
            assert!((dense.d_intensity[parameter] - finite).abs() < 3.0e-7 * finite.abs().max(1.0));
        }
    }

    #[test]
    fn invalid_batch_shapes_are_rejected() {
        let hkl = [[1, 0, 0]];
        let xyz = [[0.0, 0.0, 0.0]];
        let occupancy = [1.0];
        let empty = [];
        let error = calculate_p1_values(
            cell(),
            P1BatchView {
                hkl: &hkl,
                fractional_xyz: &xyz,
                occupancy: &occupancy,
                u_iso_angstrom2: &empty,
                scattering_real: &empty,
                scattering_imag: &empty,
                scale: 1.0,
            },
        )
        .expect_err("invalid lengths");
        assert_eq!(error, P1BatchError::SiteLengthMismatch);
    }
}
