//! Unit-cell and reciprocal-metric calculations.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Number and order of direct unit-cell parameters: `a`, `b`, `c`, `alpha`,
/// `beta`, `gamma`.
pub const CELL_PARAMETER_COUNT: usize = 6;

/// Three-by-three row-major matrix.
pub type Matrix3 = [[f64; 3]; 3];

/// Direct unit cell in ångströms and degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitCell {
    /// Direct `a` length in ångströms.
    pub a_angstrom: f64,
    /// Direct `b` length in ångströms.
    pub b_angstrom: f64,
    /// Direct `c` length in ångströms.
    pub c_angstrom: f64,
    /// Angle between `b` and `c`, in degrees.
    pub alpha_deg: f64,
    /// Angle between `a` and `c`, in degrees.
    pub beta_deg: f64,
    /// Angle between `a` and `b`, in degrees.
    pub gamma_deg: f64,
}

/// Validated direct/reciprocal geometry derived from a [`UnitCell`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellGeometry {
    /// Cartesian direct basis vectors as matrix columns, in ångströms.
    pub direct_basis: Matrix3,
    /// Cartesian reciprocal basis vectors as matrix columns, in inverse
    /// ångströms without a `2 pi` factor.
    pub reciprocal_basis: Matrix3,
    /// Direct metric tensor in square ångströms.
    pub direct_metric: Matrix3,
    /// Reciprocal metric tensor in inverse square ångströms.
    pub reciprocal_metric: Matrix3,
    /// Unit-cell volume in cubic ångströms.
    pub volume_angstrom3: f64,
    direct_metric_derivatives: [Matrix3; CELL_PARAMETER_COUNT],
}

/// Unit-cell validation or reciprocal-space evaluation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellError {
    /// A length or angle is not finite.
    NonFiniteParameter,
    /// A direct length is not positive.
    NonPositiveLength,
    /// An angle does not lie strictly within 0 and 180 degrees.
    InvalidAngle,
    /// The six parameters do not define a positive-volume cell.
    DegenerateCell,
    /// The zero Miller index does not define a finite d-spacing.
    ZeroReflection,
}

impl Display for CellError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NonFiniteParameter => "unit-cell parameters must be finite",
            Self::NonPositiveLength => "unit-cell lengths must be positive",
            Self::InvalidAngle => "unit-cell angles must lie strictly within (0, 180) degrees",
            Self::DegenerateCell => "unit-cell parameters must define a positive finite volume",
            Self::ZeroReflection => "hkl = (0, 0, 0) has no finite d-spacing",
        })
    }
}

impl Error for CellError {}

impl UnitCell {
    /// Validate the six parameters and derive direct/reciprocal geometry.
    ///
    /// # Errors
    ///
    /// Returns [`CellError`] when values are non-finite, lengths/angles are
    /// outside their domains, or the resulting metric is degenerate.
    pub fn geometry(self) -> Result<CellGeometry, CellError> {
        let parameters = [
            self.a_angstrom,
            self.b_angstrom,
            self.c_angstrom,
            self.alpha_deg,
            self.beta_deg,
            self.gamma_deg,
        ];
        if parameters.iter().any(|value| !value.is_finite()) {
            return Err(CellError::NonFiniteParameter);
        }
        if parameters[..3].iter().any(|value| *value <= 0.0) {
            return Err(CellError::NonPositiveLength);
        }
        if parameters[3..]
            .iter()
            .any(|value| !(*value > 0.0 && *value < 180.0))
        {
            return Err(CellError::InvalidAngle);
        }

        let radians_per_degree = std::f64::consts::PI / 180.0;
        let alpha = self.alpha_deg * radians_per_degree;
        let beta = self.beta_deg * radians_per_degree;
        let gamma = self.gamma_deg * radians_per_degree;
        let (cos_alpha, cos_beta, cos_gamma) = (alpha.cos(), beta.cos(), gamma.cos());
        let sin_gamma = gamma.sin();
        let direct_metric = [
            [
                self.a_angstrom * self.a_angstrom,
                self.a_angstrom * self.b_angstrom * cos_gamma,
                self.a_angstrom * self.c_angstrom * cos_beta,
            ],
            [
                self.a_angstrom * self.b_angstrom * cos_gamma,
                self.b_angstrom * self.b_angstrom,
                self.b_angstrom * self.c_angstrom * cos_alpha,
            ],
            [
                self.a_angstrom * self.c_angstrom * cos_beta,
                self.b_angstrom * self.c_angstrom * cos_alpha,
                self.c_angstrom * self.c_angstrom,
            ],
        ];
        let determinant = determinant(direct_metric);
        if !determinant.is_finite() || determinant <= 0.0 || sin_gamma <= 0.0 {
            return Err(CellError::DegenerateCell);
        }
        let volume = determinant.sqrt();
        let c_x = self.c_angstrom * cos_beta;
        let c_y = self.c_angstrom * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
        let c_z = volume / (self.a_angstrom * self.b_angstrom * sin_gamma);
        if !c_y.is_finite() || !c_z.is_finite() || c_z <= 0.0 {
            return Err(CellError::DegenerateCell);
        }
        let direct_basis = [
            [self.a_angstrom, self.b_angstrom * cos_gamma, c_x],
            [0.0, self.b_angstrom * sin_gamma, c_y],
            [0.0, 0.0, c_z],
        ];
        let reciprocal_metric = inverse(direct_metric).ok_or(CellError::DegenerateCell)?;
        let reciprocal_basis = transpose(inverse(direct_basis).ok_or(CellError::DegenerateCell)?);
        let direct_metric_derivatives = direct_metric_derivatives(self, alpha, beta, gamma);
        Ok(CellGeometry {
            direct_basis,
            reciprocal_basis,
            direct_metric,
            reciprocal_metric,
            volume_angstrom3: volume,
            direct_metric_derivatives,
        })
    }
}

impl CellGeometry {
    /// Return reciprocal-axis lengths and their direct-cell derivatives.
    ///
    /// The lengths are `a*`, `b*`, and `c*` without a `2 pi` factor. The
    /// derivative rows follow reciprocal-axis order and columns follow direct
    /// cell parameter order.
    #[must_use]
    pub fn reciprocal_axis_lengths_and_derivatives(
        &self,
    ) -> ([f64; 3], [[f64; CELL_PARAMETER_COUNT]; 3]) {
        let lengths = [
            self.reciprocal_metric[0][0].sqrt(),
            self.reciprocal_metric[1][1].sqrt(),
            self.reciprocal_metric[2][2].sqrt(),
        ];
        let mut derivatives = [[0.0; CELL_PARAMETER_COUNT]; 3];
        for axis in 0..3 {
            let reciprocal_column = [
                self.reciprocal_metric[0][axis],
                self.reciprocal_metric[1][axis],
                self.reciprocal_metric[2][axis],
            ];
            for (parameter, derivative) in derivatives[axis].iter_mut().enumerate() {
                let product =
                    matrix_vector(self.direct_metric_derivatives[parameter], reciprocal_column);
                let d_reciprocal_diagonal = -dot(reciprocal_column, product);
                *derivative = 0.5 * d_reciprocal_diagonal / lengths[axis];
            }
        }
        (lengths, derivatives)
    }

    /// Return `|g|² = hᵀ G* h` without constructing cell derivatives.
    #[must_use]
    pub fn q_squared(&self, hkl: [i32; 3]) -> f64 {
        let h = [f64::from(hkl[0]), f64::from(hkl[1]), f64::from(hkl[2])];
        dot(h, matrix_vector(self.reciprocal_metric, h))
    }

    /// Return `|g|^2 = h^T G* h` and its derivatives in direct-cell parameter
    /// order. Unlike d-spacing evaluation, the zero reflection is permitted.
    #[must_use]
    pub fn q_squared_and_derivatives(&self, hkl: [i32; 3]) -> (f64, [f64; CELL_PARAMETER_COUNT]) {
        let h = [f64::from(hkl[0]), f64::from(hkl[1]), f64::from(hkl[2])];
        let reciprocal_h = matrix_vector(self.reciprocal_metric, h);
        let q_squared = dot(h, reciprocal_h);
        let mut derivatives = [0.0; CELL_PARAMETER_COUNT];
        for (parameter, derivative) in derivatives.iter_mut().enumerate() {
            let d_metric_h = matrix_vector(self.direct_metric_derivatives[parameter], reciprocal_h);
            *derivative = -dot(reciprocal_h, d_metric_h);
        }
        (q_squared, derivatives)
    }

    /// Return d-spacing and derivatives in direct-cell parameter order.
    ///
    /// # Errors
    ///
    /// Returns [`CellError::ZeroReflection`] for `hkl = (0, 0, 0)`.
    pub fn d_spacing_and_derivatives(
        &self,
        hkl: [i32; 3],
    ) -> Result<(f64, [f64; CELL_PARAMETER_COUNT]), CellError> {
        let (q_squared, q_derivatives) = self.q_squared_and_derivatives(hkl);
        if q_squared <= 0.0 || !q_squared.is_finite() {
            return Err(CellError::ZeroReflection);
        }
        let d_spacing = q_squared.sqrt().recip();
        let factor = -0.5 * d_spacing.powi(3);
        Ok((d_spacing, q_derivatives.map(|value| factor * value)))
    }

    /// Return volume derivatives in direct-cell parameter order.
    #[must_use]
    pub fn volume_derivatives(&self) -> [f64; CELL_PARAMETER_COUNT] {
        let mut derivatives = [0.0; CELL_PARAMETER_COUNT];
        for (parameter, derivative) in derivatives.iter_mut().enumerate() {
            let product = multiply(
                self.reciprocal_metric,
                self.direct_metric_derivatives[parameter],
            );
            *derivative =
                0.5 * self.volume_angstrom3 * (product[0][0] + product[1][1] + product[2][2]);
        }
        derivatives
    }
}

fn direct_metric_derivatives(
    cell: UnitCell,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> [Matrix3; CELL_PARAMETER_COUNT] {
    let zero = [[0.0; 3]; 3];
    let mut derivatives = [zero; CELL_PARAMETER_COUNT];
    let (ca, cb, cg) = (alpha.cos(), beta.cos(), gamma.cos());
    derivatives[0] = [
        [
            2.0 * cell.a_angstrom,
            cell.b_angstrom * cg,
            cell.c_angstrom * cb,
        ],
        [cell.b_angstrom * cg, 0.0, 0.0],
        [cell.c_angstrom * cb, 0.0, 0.0],
    ];
    derivatives[1] = [
        [0.0, cell.a_angstrom * cg, 0.0],
        [
            cell.a_angstrom * cg,
            2.0 * cell.b_angstrom,
            cell.c_angstrom * ca,
        ],
        [0.0, cell.c_angstrom * ca, 0.0],
    ];
    derivatives[2] = [
        [0.0, 0.0, cell.a_angstrom * cb],
        [0.0, 0.0, cell.b_angstrom * ca],
        [
            cell.a_angstrom * cb,
            cell.b_angstrom * ca,
            2.0 * cell.c_angstrom,
        ],
    ];
    let radians_per_degree = std::f64::consts::PI / 180.0;
    let d_alpha = -cell.b_angstrom * cell.c_angstrom * alpha.sin() * radians_per_degree;
    derivatives[3][1][2] = d_alpha;
    derivatives[3][2][1] = d_alpha;
    let d_beta = -cell.a_angstrom * cell.c_angstrom * beta.sin() * radians_per_degree;
    derivatives[4][0][2] = d_beta;
    derivatives[4][2][0] = d_beta;
    let d_gamma = -cell.a_angstrom * cell.b_angstrom * gamma.sin() * radians_per_degree;
    derivatives[5][0][1] = d_gamma;
    derivatives[5][1][0] = d_gamma;
    derivatives
}

fn determinant(matrix: Matrix3) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn inverse(matrix: Matrix3) -> Option<Matrix3> {
    let det = determinant(matrix);
    if !det.is_finite() || det == 0.0 {
        return None;
    }
    let inverse_det = det.recip();
    Some([
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inverse_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inverse_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inverse_det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inverse_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inverse_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inverse_det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inverse_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inverse_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inverse_det,
        ],
    ])
}

fn transpose(matrix: Matrix3) -> Matrix3 {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn multiply(left: Matrix3, right: Matrix3) -> Matrix3 {
    let mut result = [[0.0; 3]; 3];
    for (row, result_row) in result.iter_mut().enumerate() {
        for (column, value) in result_row.iter_mut().enumerate() {
            *value = (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum();
        }
    }
    result
}

fn matrix_vector(matrix: Matrix3, vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| dot(row, vector))
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triclinic() -> UnitCell {
        UnitCell {
            a_angstrom: 4.3,
            b_angstrom: 5.1,
            c_angstrom: 6.2,
            alpha_deg: 78.0,
            beta_deg: 83.0,
            gamma_deg: 71.0,
        }
    }

    fn changed(mut cell: UnitCell, parameter: usize, delta: f64) -> UnitCell {
        match parameter {
            0 => cell.a_angstrom += delta,
            1 => cell.b_angstrom += delta,
            2 => cell.c_angstrom += delta,
            3 => cell.alpha_deg += delta,
            4 => cell.beta_deg += delta,
            5 => cell.gamma_deg += delta,
            _ => unreachable!(),
        }
        cell
    }

    #[test]
    fn cubic_geometry_is_exact() {
        let geometry = UnitCell {
            a_angstrom: 4.0,
            b_angstrom: 4.0,
            c_angstrom: 4.0,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        }
        .geometry()
        .expect("cubic cell");
        assert!((geometry.volume_angstrom3 - 64.0).abs() < 1.0e-12);
        assert!((geometry.reciprocal_metric[0][0] - 1.0 / 16.0).abs() < 1.0e-15);
        let (d, _) = geometry
            .d_spacing_and_derivatives([1, 1, 0])
            .expect("nonzero reflection");
        assert!((d - 4.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
    }

    #[test]
    fn cell_derivatives_match_centered_differences() {
        let cell = triclinic();
        let geometry = cell.geometry().expect("triclinic cell");
        let (d, derivatives) = geometry
            .d_spacing_and_derivatives([2, -1, 3])
            .expect("reflection");
        let volume_derivatives = geometry.volume_derivatives();
        let (reciprocal_lengths, reciprocal_derivatives) =
            geometry.reciprocal_axis_lengths_and_derivatives();
        for parameter in 0..CELL_PARAMETER_COUNT {
            let step = if parameter < 3 { 1.0e-6 } else { 1.0e-5 };
            let plus = changed(cell, parameter, step)
                .geometry()
                .expect("plus cell");
            let minus = changed(cell, parameter, -step)
                .geometry()
                .expect("minus cell");
            let plus_d = plus
                .d_spacing_and_derivatives([2, -1, 3])
                .expect("plus reflection")
                .0;
            let minus_d = minus
                .d_spacing_and_derivatives([2, -1, 3])
                .expect("minus reflection")
                .0;
            let finite_d = (plus_d - minus_d) / (2.0 * step);
            let finite_volume = (plus.volume_angstrom3 - minus.volume_angstrom3) / (2.0 * step);
            assert!((derivatives[parameter] - finite_d).abs() < 2.0e-8 * d.max(1.0));
            assert!(
                (volume_derivatives[parameter] - finite_volume).abs()
                    < 2.0e-8 * geometry.volume_angstrom3
            );
            let plus_lengths = plus.reciprocal_axis_lengths_and_derivatives().0;
            let minus_lengths = minus.reciprocal_axis_lengths_and_derivatives().0;
            for axis in 0..3 {
                let finite_reciprocal = (plus_lengths[axis] - minus_lengths[axis]) / (2.0 * step);
                assert!(
                    (reciprocal_derivatives[axis][parameter] - finite_reciprocal).abs()
                        < 2.0e-8 * reciprocal_lengths[axis].max(1.0)
                );
            }
        }
    }

    #[test]
    fn invalid_cells_and_zero_reflection_are_rejected() {
        let mut cell = triclinic();
        cell.a_angstrom = 0.0;
        assert_eq!(cell.geometry(), Err(CellError::NonPositiveLength));
        let geometry = triclinic().geometry().expect("cell");
        assert_eq!(
            geometry.d_spacing_and_derivatives([0, 0, 0]),
            Err(CellError::ZeroReflection)
        );
    }
}
