//! Typed analytical background models and row-major derivative bases.

use std::collections::BTreeSet;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter};

use crate::ParameterBounds;

/// Row-major sample-by-parameter analytical background derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundBasis {
    /// Sample row count.
    pub rows: usize,
    /// Parameter column count.
    pub columns: usize,
    /// Row-major values.
    pub values: Vec<f64>,
}

impl BackgroundBasis {
    fn zeros(rows: usize, columns: usize) -> Result<Self, BackgroundError> {
        let count = rows
            .checked_mul(columns)
            .ok_or(BackgroundError::SizeOverflow)?;
        Ok(Self {
            rows,
            columns,
            values: vec![0.0; count],
        })
    }

    /// Borrow one sample row.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.columns)?;
        self.values.get(start..start.checked_add(self.columns)?)
    }

    /// Copy one parameter column.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<Vec<f64>> {
        if index >= self.columns {
            return None;
        }
        Some(
            self.values
                .chunks_exact(self.columns)
                .map(|row| row[index])
                .collect(),
        )
    }
}

/// Shared immutable analytical-background contract.
pub trait DifferentiableBackground {
    /// Stable model identifier.
    fn background_id(&self) -> &str;
    /// Stable parameter names in coefficient order.
    fn parameter_names(&self) -> Vec<String>;
    /// Current physical coefficients in derivative-column order.
    fn coefficients(&self) -> Vec<f64>;
    /// Physical parameter bounds in coefficient order.
    fn parameter_bounds(&self) -> Vec<ParameterBounds>;
    /// Evaluate sample-by-parameter analytical derivatives.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for an invalid grid or allocation overflow.
    fn basis(&self, x_deg: &[f64]) -> Result<BackgroundBasis, BackgroundError>;
    /// Evaluate the background on an ordered finite grid.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for an invalid grid or allocation overflow.
    fn calculate(&self, x_deg: &[f64]) -> Result<Vec<f64>, BackgroundError>;
    /// Return the same model with replacement coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for shape, finiteness, or model-bound failures.
    fn replace_coefficients(&self, coefficients: &[f64]) -> Result<Self, BackgroundError>
    where
        Self: Sized;
    /// Return whether the analytical basis is coefficient-invariant.
    fn basis_is_invariant(&self) -> bool;
}

/// Power series on the normalized input-grid coordinate `[-1, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialBackground {
    background_id: String,
    coefficients: Vec<f64>,
}

impl PolynomialBackground {
    /// Construct a non-empty finite power-series model.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for an invalid ID or coefficients.
    pub fn new(
        background_id: impl Into<String>,
        coefficients: Vec<f64>,
    ) -> Result<Self, BackgroundError> {
        let background_id = validate_id(background_id.into())?;
        validate_coefficients(&coefficients)?;
        Ok(Self {
            background_id,
            coefficients,
        })
    }
}

impl DifferentiableBackground for PolynomialBackground {
    fn background_id(&self) -> &str {
        &self.background_id
    }

    fn parameter_names(&self) -> Vec<String> {
        indexed_names("coefficient", self.coefficients.len())
    }

    fn coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    fn parameter_bounds(&self) -> Vec<ParameterBounds> {
        vec![ParameterBounds::default(); self.coefficients.len()]
    }

    fn basis(&self, x_deg: &[f64]) -> Result<BackgroundBasis, BackgroundError> {
        validate_grid(x_deg)?;
        let mut result = BackgroundBasis::zeros(x_deg.len(), self.coefficients.len())?;
        for (row_index, row) in result.values.chunks_exact_mut(result.columns).enumerate() {
            let normalized = normalized_grid_value(x_deg, row_index);
            let mut power = 1.0;
            for value in row {
                *value = power;
                power *= normalized;
            }
        }
        Ok(result)
    }

    fn calculate(&self, x_deg: &[f64]) -> Result<Vec<f64>, BackgroundError> {
        linear_calculate(&self.basis(x_deg)?, &self.coefficients)
    }

    fn replace_coefficients(&self, coefficients: &[f64]) -> Result<Self, BackgroundError> {
        validate_replacement(coefficients, self.coefficients.len())?;
        Self::new(self.background_id.clone(), coefficients.to_vec())
    }

    fn basis_is_invariant(&self) -> bool {
        true
    }
}

/// Chebyshev series on one explicit closed degree domain.
#[derive(Clone, Debug, PartialEq)]
pub struct ChebyshevBackground {
    background_id: String,
    coefficients: Vec<f64>,
    domain_deg: [f64; 2],
}

impl ChebyshevBackground {
    /// Construct a finite Chebyshev series on an increasing domain.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for invalid ID, coefficients, or domain.
    pub fn new(
        background_id: impl Into<String>,
        coefficients: Vec<f64>,
        domain_deg: [f64; 2],
    ) -> Result<Self, BackgroundError> {
        let background_id = validate_id(background_id.into())?;
        validate_coefficients(&coefficients)?;
        if domain_deg.iter().any(|value| !value.is_finite()) || domain_deg[0] >= domain_deg[1] {
            return Err(BackgroundError::InvalidDomain);
        }
        Ok(Self {
            background_id,
            coefficients,
            domain_deg,
        })
    }

    /// Return the explicit closed domain in degrees.
    #[must_use]
    pub const fn domain_deg(&self) -> [f64; 2] {
        self.domain_deg
    }
}

impl DifferentiableBackground for ChebyshevBackground {
    fn background_id(&self) -> &str {
        &self.background_id
    }

    fn parameter_names(&self) -> Vec<String> {
        indexed_names("coefficient", self.coefficients.len())
    }

    fn coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    fn parameter_bounds(&self) -> Vec<ParameterBounds> {
        vec![ParameterBounds::default(); self.coefficients.len()]
    }

    fn basis(&self, x_deg: &[f64]) -> Result<BackgroundBasis, BackgroundError> {
        validate_grid(x_deg)?;
        let lower = self.domain_deg[0];
        let upper = self.domain_deg[1];
        let tolerance = 64.0 * f64::EPSILON * lower.abs().max(upper.abs()).max(1.0);
        if x_deg
            .iter()
            .any(|value| *value < lower - tolerance || *value > upper + tolerance)
        {
            return Err(BackgroundError::GridOutsideDomain);
        }
        let mut result = BackgroundBasis::zeros(x_deg.len(), self.coefficients.len())?;
        for (x, row) in x_deg
            .iter()
            .zip(result.values.chunks_exact_mut(result.columns))
        {
            let normalized = 2.0 * (x - lower) / (upper - lower) - 1.0;
            row[0] = 1.0;
            if row.len() > 1 {
                row[1] = normalized;
            }
            for order in 2..row.len() {
                row[order] = 2.0 * normalized * row[order - 1] - row[order - 2];
            }
        }
        Ok(result)
    }

    fn calculate(&self, x_deg: &[f64]) -> Result<Vec<f64>, BackgroundError> {
        linear_calculate(&self.basis(x_deg)?, &self.coefficients)
    }

    fn replace_coefficients(&self, coefficients: &[f64]) -> Result<Self, BackgroundError> {
        validate_replacement(coefficients, self.coefficients.len())?;
        Self::new(
            self.background_id.clone(),
            coefficients.to_vec(),
            self.domain_deg,
        )
    }

    fn basis_is_invariant(&self) -> bool {
        true
    }
}

/// Linear interpolation through fixed knots with refinable values.
#[derive(Clone, Debug, PartialEq)]
pub struct PointBackground {
    background_id: String,
    knot_x: Vec<f64>,
    values: Vec<f64>,
}

impl PointBackground {
    /// Construct a fixed-knot piecewise-linear model with constant end extrapolation.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for invalid IDs, knots, or values.
    pub fn new(
        background_id: impl Into<String>,
        knot_x: Vec<f64>,
        values: Vec<f64>,
    ) -> Result<Self, BackgroundError> {
        let background_id = validate_id(background_id.into())?;
        if knot_x.len() < 2 || values.len() != knot_x.len() {
            return Err(BackgroundError::InvalidKnots);
        }
        validate_grid(&knot_x).map_err(|_| BackgroundError::InvalidKnots)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BackgroundError::NonFiniteCoefficients);
        }
        Ok(Self {
            background_id,
            knot_x,
            values,
        })
    }

    /// Borrow fixed interpolation knots.
    #[must_use]
    pub fn knot_x(&self) -> &[f64] {
        &self.knot_x
    }
}

impl DifferentiableBackground for PointBackground {
    fn background_id(&self) -> &str {
        &self.background_id
    }

    fn parameter_names(&self) -> Vec<String> {
        indexed_names("value", self.values.len())
    }

    fn coefficients(&self) -> Vec<f64> {
        self.values.clone()
    }

    fn parameter_bounds(&self) -> Vec<ParameterBounds> {
        vec![ParameterBounds::default(); self.values.len()]
    }

    fn basis(&self, x_deg: &[f64]) -> Result<BackgroundBasis, BackgroundError> {
        validate_grid(x_deg)?;
        let mut result = BackgroundBasis::zeros(x_deg.len(), self.knot_x.len())?;
        for (x, row) in x_deg
            .iter()
            .zip(result.values.chunks_exact_mut(result.columns))
        {
            let right = self.knot_x.partition_point(|knot| knot <= x);
            if right == 0 {
                row[0] = 1.0;
            } else if right == self.knot_x.len() {
                row[right - 1] = 1.0;
            } else {
                let lower = right - 1;
                let fraction = (x - self.knot_x[lower]) / (self.knot_x[right] - self.knot_x[lower]);
                row[lower] = 1.0 - fraction;
                row[right] = fraction;
            }
        }
        Ok(result)
    }

    fn calculate(&self, x_deg: &[f64]) -> Result<Vec<f64>, BackgroundError> {
        linear_calculate(&self.basis(x_deg)?, &self.values)
    }

    fn replace_coefficients(&self, coefficients: &[f64]) -> Result<Self, BackgroundError> {
        validate_replacement(coefficients, self.values.len())?;
        Self::new(
            self.background_id.clone(),
            self.knot_x.clone(),
            coefficients.to_vec(),
        )
    }

    fn basis_is_invariant(&self) -> bool {
        true
    }
}

/// One area-normalized broad Gaussian component in degrees `2theta`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmorphousPeak {
    area: f64,
    center_deg: f64,
    fwhm_deg: f64,
}

impl AmorphousPeak {
    /// Construct one finite non-negative-area, positive-width component.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError::InvalidAmorphousPeak`] for invalid values.
    pub fn new(area: f64, center_deg: f64, fwhm_deg: f64) -> Result<Self, BackgroundError> {
        if !area.is_finite()
            || !center_deg.is_finite()
            || !fwhm_deg.is_finite()
            || area < 0.0
            || fwhm_deg <= 0.0
        {
            return Err(BackgroundError::InvalidAmorphousPeak);
        }
        Ok(Self {
            area,
            center_deg,
            fwhm_deg,
        })
    }

    /// Return integrated area.
    #[must_use]
    pub const fn area(self) -> f64 {
        self.area
    }

    /// Return center in degrees `2theta`.
    #[must_use]
    pub const fn center_deg(self) -> f64 {
        self.center_deg
    }

    /// Return FWHM in degrees `2theta`.
    #[must_use]
    pub const fn fwhm_deg(self) -> f64 {
        self.fwhm_deg
    }
}

/// Sum of broad area-normalized Gaussian amorphous components.
#[derive(Clone, Debug, PartialEq)]
pub struct AmorphousBackground {
    background_id: String,
    peaks: Vec<AmorphousPeak>,
}

impl AmorphousBackground {
    /// Construct a non-empty broad-Gaussian background.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for an invalid ID or empty peak list.
    pub fn new(
        background_id: impl Into<String>,
        peaks: Vec<AmorphousPeak>,
    ) -> Result<Self, BackgroundError> {
        let background_id = validate_id(background_id.into())?;
        if peaks.is_empty() {
            return Err(BackgroundError::EmptyComponents);
        }
        Ok(Self {
            background_id,
            peaks,
        })
    }

    /// Borrow broad components.
    #[must_use]
    pub fn peaks(&self) -> &[AmorphousPeak] {
        &self.peaks
    }
}

impl DifferentiableBackground for AmorphousBackground {
    fn background_id(&self) -> &str {
        &self.background_id
    }

    fn parameter_names(&self) -> Vec<String> {
        self.peaks
            .iter()
            .enumerate()
            .flat_map(|(index, _)| {
                ["area", "center_deg", "fwhm_deg"]
                    .into_iter()
                    .map(move |name| format!("peak_{index}.{name}"))
            })
            .collect()
    }

    fn coefficients(&self) -> Vec<f64> {
        self.peaks
            .iter()
            .flat_map(|peak| [peak.area, peak.center_deg, peak.fwhm_deg])
            .collect()
    }

    fn parameter_bounds(&self) -> Vec<ParameterBounds> {
        self.peaks
            .iter()
            .flat_map(|_| {
                [
                    ParameterBounds::new(0.0, f64::INFINITY).expect("valid area bounds"),
                    ParameterBounds::default(),
                    ParameterBounds::new(f64::MIN_POSITIVE, f64::INFINITY)
                        .expect("valid FWHM bounds"),
                ]
            })
            .collect()
    }

    fn basis(&self, x_deg: &[f64]) -> Result<BackgroundBasis, BackgroundError> {
        validate_grid(x_deg)?;
        let columns = self
            .peaks
            .len()
            .checked_mul(3)
            .ok_or(BackgroundError::SizeOverflow)?;
        let mut result = BackgroundBasis::zeros(x_deg.len(), columns)?;
        let factor = 4.0 * 2.0_f64.ln();
        let normalization = (factor / PI).sqrt();
        for (x, row) in x_deg
            .iter()
            .zip(result.values.chunks_exact_mut(result.columns))
        {
            for (peak_index, peak) in self.peaks.iter().enumerate() {
                let delta = x - peak.center_deg;
                let ratio = delta / peak.fwhm_deg;
                let gaussian = normalization / peak.fwhm_deg * (-factor * ratio * ratio).exp();
                let value = peak.area * gaussian;
                let offset = 3 * peak_index;
                row[offset] = gaussian;
                row[offset + 1] = value * 2.0 * factor * delta / (peak.fwhm_deg * peak.fwhm_deg);
                row[offset + 2] = value
                    * (-1.0 / peak.fwhm_deg
                        + 2.0 * factor * delta * delta
                            / (peak.fwhm_deg * peak.fwhm_deg * peak.fwhm_deg));
            }
        }
        Ok(result)
    }

    fn calculate(&self, x_deg: &[f64]) -> Result<Vec<f64>, BackgroundError> {
        validate_grid(x_deg)?;
        let factor = 4.0 * 2.0_f64.ln();
        let normalization = (factor / PI).sqrt();
        Ok(x_deg
            .iter()
            .map(|x| {
                self.peaks
                    .iter()
                    .map(|peak| {
                        let ratio = (x - peak.center_deg) / peak.fwhm_deg;
                        peak.area * normalization / peak.fwhm_deg * (-factor * ratio * ratio).exp()
                    })
                    .sum()
            })
            .collect())
    }

    fn replace_coefficients(&self, coefficients: &[f64]) -> Result<Self, BackgroundError> {
        let expected = self
            .peaks
            .len()
            .checked_mul(3)
            .ok_or(BackgroundError::SizeOverflow)?;
        validate_replacement(coefficients, expected)?;
        let peaks = coefficients
            .chunks_exact(3)
            .map(|values| AmorphousPeak::new(values[0], values[1], values[2]))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(self.background_id.clone(), peaks)
    }

    fn basis_is_invariant(&self) -> bool {
        false
    }
}

/// Closed native union used by application-neutral request records.
#[derive(Clone, Debug, PartialEq)]
pub enum BackgroundModel {
    /// Normalized-coordinate power series.
    Polynomial(PolynomialBackground),
    /// Explicit-domain Chebyshev series.
    Chebyshev(ChebyshevBackground),
    /// Fixed-knot linear interpolation.
    Point(PointBackground),
    /// Broad Gaussian components.
    Amorphous(AmorphousBackground),
    /// Ordered additive nested composition.
    Composite(CompositeBackground),
}

/// Ordered additive composition of analytical models.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeBackground {
    background_id: String,
    components: Vec<BackgroundModel>,
}

impl CompositeBackground {
    /// Construct a non-empty composition with unique immediate component IDs.
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError`] for an invalid ID, empty components, or
    /// duplicate component IDs.
    pub fn new(
        background_id: impl Into<String>,
        components: Vec<BackgroundModel>,
    ) -> Result<Self, BackgroundError> {
        let background_id = validate_id(background_id.into())?;
        if components.is_empty() {
            return Err(BackgroundError::EmptyComponents);
        }
        let mut ids = BTreeSet::new();
        for component in &components {
            if !ids.insert(component.background_id().to_owned()) {
                return Err(BackgroundError::DuplicateComponentId {
                    background_id: component.background_id().to_owned(),
                });
            }
        }
        Ok(Self {
            background_id,
            components,
        })
    }

    /// Borrow ordered components.
    #[must_use]
    pub fn components(&self) -> &[BackgroundModel] {
        &self.components
    }
}

macro_rules! delegate_background {
    ($self:ident, $method:ident $(, $argument:expr)*) => {
        match $self {
            Self::Polynomial(value) => value.$method($($argument),*),
            Self::Chebyshev(value) => value.$method($($argument),*),
            Self::Point(value) => value.$method($($argument),*),
            Self::Amorphous(value) => value.$method($($argument),*),
            Self::Composite(value) => value.$method($($argument),*),
        }
    };
}

impl DifferentiableBackground for BackgroundModel {
    fn background_id(&self) -> &str {
        delegate_background!(self, background_id)
    }

    fn parameter_names(&self) -> Vec<String> {
        delegate_background!(self, parameter_names)
    }

    fn coefficients(&self) -> Vec<f64> {
        delegate_background!(self, coefficients)
    }

    fn parameter_bounds(&self) -> Vec<ParameterBounds> {
        delegate_background!(self, parameter_bounds)
    }

    fn basis(&self, x_deg: &[f64]) -> Result<BackgroundBasis, BackgroundError> {
        delegate_background!(self, basis, x_deg)
    }

    fn calculate(&self, x_deg: &[f64]) -> Result<Vec<f64>, BackgroundError> {
        delegate_background!(self, calculate, x_deg)
    }

    fn replace_coefficients(&self, coefficients: &[f64]) -> Result<Self, BackgroundError> {
        Ok(match self {
            Self::Polynomial(value) => Self::Polynomial(value.replace_coefficients(coefficients)?),
            Self::Chebyshev(value) => Self::Chebyshev(value.replace_coefficients(coefficients)?),
            Self::Point(value) => Self::Point(value.replace_coefficients(coefficients)?),
            Self::Amorphous(value) => Self::Amorphous(value.replace_coefficients(coefficients)?),
            Self::Composite(value) => Self::Composite(value.replace_coefficients(coefficients)?),
        })
    }

    fn basis_is_invariant(&self) -> bool {
        delegate_background!(self, basis_is_invariant)
    }
}

impl DifferentiableBackground for CompositeBackground {
    fn background_id(&self) -> &str {
        &self.background_id
    }

    fn parameter_names(&self) -> Vec<String> {
        self.components
            .iter()
            .flat_map(|component| {
                component
                    .parameter_names()
                    .into_iter()
                    .map(move |name| format!("{}.{}", component.background_id(), name))
            })
            .collect()
    }

    fn coefficients(&self) -> Vec<f64> {
        self.components
            .iter()
            .flat_map(DifferentiableBackground::coefficients)
            .collect()
    }

    fn parameter_bounds(&self) -> Vec<ParameterBounds> {
        self.components
            .iter()
            .flat_map(DifferentiableBackground::parameter_bounds)
            .collect()
    }

    fn basis(&self, x_deg: &[f64]) -> Result<BackgroundBasis, BackgroundError> {
        validate_grid(x_deg)?;
        let component_bases = self
            .components
            .iter()
            .map(|component| component.basis(x_deg))
            .collect::<Result<Vec<_>, _>>()?;
        let columns = component_bases.iter().try_fold(0_usize, |total, basis| {
            total
                .checked_add(basis.columns)
                .ok_or(BackgroundError::SizeOverflow)
        })?;
        let mut result = BackgroundBasis::zeros(x_deg.len(), columns)?;
        for row_index in 0..x_deg.len() {
            let target = result
                .values
                .get_mut(row_index * columns..(row_index + 1) * columns)
                .ok_or(BackgroundError::InternalInvariant)?;
            let mut offset = 0;
            for basis in &component_bases {
                let source = basis
                    .row(row_index)
                    .ok_or(BackgroundError::InternalInvariant)?;
                let end = offset + source.len();
                target
                    .get_mut(offset..end)
                    .ok_or(BackgroundError::InternalInvariant)?
                    .copy_from_slice(source);
                offset = end;
            }
        }
        Ok(result)
    }

    fn calculate(&self, x_deg: &[f64]) -> Result<Vec<f64>, BackgroundError> {
        validate_grid(x_deg)?;
        let mut result = vec![0.0; x_deg.len()];
        for component in &self.components {
            let values = component.calculate(x_deg)?;
            for (target, value) in result.iter_mut().zip(values) {
                *target += value;
            }
        }
        Ok(result)
    }

    fn replace_coefficients(&self, coefficients: &[f64]) -> Result<Self, BackgroundError> {
        validate_replacement(coefficients, self.coefficients().len())?;
        let mut offset = 0_usize;
        let mut components = Vec::with_capacity(self.components.len());
        for component in &self.components {
            let count = component.coefficients().len();
            let end = offset
                .checked_add(count)
                .ok_or(BackgroundError::SizeOverflow)?;
            components.push(
                component.replace_coefficients(
                    coefficients
                        .get(offset..end)
                        .ok_or(BackgroundError::InternalInvariant)?,
                )?,
            );
            offset = end;
        }
        Self::new(self.background_id.clone(), components)
    }

    fn basis_is_invariant(&self) -> bool {
        self.components
            .iter()
            .all(DifferentiableBackground::basis_is_invariant)
    }
}

/// Invalid analytical background definition, grid, or replacement.
#[derive(Clone, Debug, PartialEq)]
pub enum BackgroundError {
    /// Model ID is empty or untrimmed.
    InvalidId,
    /// Coefficient list is empty.
    EmptyCoefficients,
    /// One coefficient is non-finite.
    NonFiniteCoefficients,
    /// Chebyshev domain is non-finite or not increasing.
    InvalidDomain,
    /// Point knots are invalid or do not match values.
    InvalidKnots,
    /// Grid contains a non-finite sample.
    NonFiniteGrid {
        /// Rejected index.
        index: usize,
    },
    /// Grid is not strictly increasing.
    UnorderedGrid,
    /// Grid lies outside an explicit Chebyshev domain.
    GridOutsideDomain,
    /// Broad Gaussian area/center/FWHM is invalid.
    InvalidAmorphousPeak,
    /// Model composition or broad peak list is empty.
    EmptyComponents,
    /// Composite repeats one immediate component ID.
    DuplicateComponentId {
        /// Repeated ID.
        background_id: String,
    },
    /// Replacement coefficient vector has the wrong length.
    CoefficientLengthMismatch {
        /// Expected count.
        expected: usize,
        /// Received count.
        actual: usize,
    },
    /// A checked matrix/vector size overflowed.
    SizeOverflow,
    /// Private validated state is unexpectedly inconsistent.
    InternalInvariant,
}

impl Display for BackgroundError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidId => formatter.write_str("background_id must be non-empty and trimmed"),
            Self::EmptyCoefficients => {
                formatter.write_str("background coefficients must not be empty")
            }
            Self::NonFiniteCoefficients => {
                formatter.write_str("background coefficients must be finite")
            }
            Self::InvalidDomain => {
                formatter.write_str("background domain must contain two increasing finite values")
            }
            Self::InvalidKnots => formatter.write_str(
                "point background requires matching finite strictly increasing knots and values",
            ),
            Self::NonFiniteGrid { index } => {
                write!(formatter, "background grid is non-finite at index {index}")
            }
            Self::UnorderedGrid => {
                formatter.write_str("background grid must be strictly increasing")
            }
            Self::GridOutsideDomain => {
                formatter.write_str("background grid lies outside the explicit domain")
            }
            Self::InvalidAmorphousPeak => formatter.write_str(
                "amorphous area/center/FWHM must be finite with non-negative area and positive FWHM",
            ),
            Self::EmptyComponents => {
                formatter.write_str("background components must not be empty")
            }
            Self::DuplicateComponentId { background_id } => {
                write!(formatter, "duplicate background component ID {background_id:?}")
            }
            Self::CoefficientLengthMismatch { expected, actual } => write!(
                formatter,
                "replacement coefficient length {actual} does not match {expected}"
            ),
            Self::SizeOverflow => formatter.write_str("background matrix size overflow"),
            Self::InternalInvariant => {
                formatter.write_str("validated background state is inconsistent")
            }
        }
    }
}

impl Error for BackgroundError {}

fn validate_id(value: String) -> Result<String, BackgroundError> {
    if value.is_empty() || value.trim() != value {
        return Err(BackgroundError::InvalidId);
    }
    Ok(value)
}

fn validate_coefficients(values: &[f64]) -> Result<(), BackgroundError> {
    if values.is_empty() {
        return Err(BackgroundError::EmptyCoefficients);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BackgroundError::NonFiniteCoefficients);
    }
    Ok(())
}

fn validate_replacement(values: &[f64], expected: usize) -> Result<(), BackgroundError> {
    if values.len() != expected {
        return Err(BackgroundError::CoefficientLengthMismatch {
            expected,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BackgroundError::NonFiniteCoefficients);
    }
    Ok(())
}

fn validate_grid(x_deg: &[f64]) -> Result<(), BackgroundError> {
    if let Some(index) = x_deg.iter().position(|value| !value.is_finite()) {
        return Err(BackgroundError::NonFiniteGrid { index });
    }
    if x_deg.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(BackgroundError::UnorderedGrid);
    }
    Ok(())
}

fn indexed_names(stem: &str, count: usize) -> Vec<String> {
    (0..count).map(|index| format!("{stem}_{index}")).collect()
}

fn normalized_grid_value(x_deg: &[f64], index: usize) -> f64 {
    if x_deg.len() <= 1 {
        0.0
    } else {
        2.0 * (x_deg[index] - x_deg[0]) / (x_deg[x_deg.len() - 1] - x_deg[0]) - 1.0
    }
}

fn linear_calculate(
    basis: &BackgroundBasis,
    coefficients: &[f64],
) -> Result<Vec<f64>, BackgroundError> {
    if basis.columns != coefficients.len() {
        return Err(BackgroundError::InternalInvariant);
    }
    Ok(basis
        .values
        .chunks_exact(basis.columns)
        .map(|row| row.iter().zip(coefficients).map(|(a, b)| a * b).sum())
        .collect())
}
