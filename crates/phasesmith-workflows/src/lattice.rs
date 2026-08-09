//! Symmetry-aware lattice variables and guarded CW reflection domains.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{Matrix3, SymmetricEigen};
use phasesmith_crystallography::{
    CellError, CrystalSystem, PreparedReflectionGenerator, ReflectionGenerationError,
    ReflectionRange, SpaceGroup, UnitCell,
};

const CELL_NAMES: [&str; 6] = [
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
];

#[derive(Clone, Debug, PartialEq, Eq)]
enum LatticeKind {
    Triclinic,
    Monoclinic { angle: usize },
    Orthorhombic,
    PlaneUnique { plane: [usize; 2], unique: usize },
    Rhombohedral,
    Cubic,
}

/// Independent physical lattice variables for one exact crystallographic setting.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeParameterization {
    space_group: SpaceGroup,
    reference_cell: UnitCell,
    kind: LatticeKind,
    parameter_names: Vec<String>,
}

impl LatticeParameterization {
    /// Derive a setting-aware independent-variable mapping.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] when the cell is incompatible with the group or
    /// the exact metric basis does not describe a supported conventional setting.
    pub fn new(space_group: SpaceGroup, cell: UnitCell) -> Result<Self, LatticeError> {
        validate_metric_compatibility(&space_group, cell)?;
        let basis = &space_group.metric_constraints().parameterization_basis;
        let kind = match space_group.crystal_system() {
            CrystalSystem::Triclinic => LatticeKind::Triclinic,
            CrystalSystem::Monoclinic => {
                let free = (0..3)
                    .filter(|angle| basis.iter().any(|row| row[3 + angle] != 0))
                    .collect::<Vec<_>>();
                if free.len() != 1 {
                    return Err(LatticeError::UnsupportedSetting);
                }
                LatticeKind::Monoclinic { angle: free[0] }
            }
            CrystalSystem::Orthorhombic => LatticeKind::Orthorhombic,
            CrystalSystem::Tetragonal | CrystalSystem::Hexagonal => plane_unique_kind(basis)?,
            CrystalSystem::Trigonal => {
                let diagonal_equal = (1..3).all(|right| metric_columns_equal(basis, 0, right));
                let off_diagonal_equal = (4..6).all(|right| metric_columns_equal(basis, 3, right));
                if diagonal_equal && off_diagonal_equal && basis.iter().any(|row| row[3] != 0) {
                    LatticeKind::Rhombohedral
                } else {
                    plane_unique_kind(basis)?
                }
            }
            CrystalSystem::Cubic => LatticeKind::Cubic,
        };
        let parameter_names = names_for_kind(&kind);
        let result = Self {
            space_group,
            reference_cell: cell,
            kind,
            parameter_names,
        };
        let values = result.values_from_cell(cell)?;
        let rebuilt = result.to_cell(&values)?;
        if !cells_close(rebuilt, cell, 2.0e-10) {
            return Err(LatticeError::UnsupportedSetting);
        }
        Ok(result)
    }

    /// Borrow the exact symmetry group.
    #[must_use]
    pub const fn space_group(&self) -> &SpaceGroup {
        &self.space_group
    }

    /// Borrow independent parameter names in stable order.
    #[must_use]
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Return the reference cell used for fixed components.
    #[must_use]
    pub const fn reference_cell(&self) -> UnitCell {
        self.reference_cell
    }

    /// Extract independent values after checking setting compatibility.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] if the supplied cell is incompatible.
    pub fn values_from_cell(&self, cell: UnitCell) -> Result<Vec<f64>, LatticeError> {
        cell.geometry().map_err(LatticeError::Cell)?;
        let full = cell_values(cell);
        let values = match self.kind {
            LatticeKind::Triclinic => full.to_vec(),
            LatticeKind::Monoclinic { angle } => {
                vec![full[0], full[1], full[2], full[3 + angle]]
            }
            LatticeKind::Orthorhombic => full[..3].to_vec(),
            LatticeKind::PlaneUnique { plane, unique } => {
                let first = plane[0].min(unique);
                let second = plane[0].max(unique);
                vec![full[first], full[second]]
            }
            LatticeKind::Rhombohedral => vec![full[0], full[3]],
            LatticeKind::Cubic => vec![full[0]],
        };
        if !cells_close(self.to_cell(&values)?, cell, 2.0e-9) {
            return Err(LatticeError::IncompatibleCell);
        }
        Ok(values)
    }

    /// Expand independent values into all six direct-cell parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] for shape, finite, or physical-cell failures.
    pub fn to_cell(&self, values: &[f64]) -> Result<UnitCell, LatticeError> {
        if values.len() != self.parameter_names.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(LatticeError::ValueShape);
        }
        let mut full = cell_values(self.reference_cell);
        match self.kind {
            LatticeKind::Triclinic => full.copy_from_slice(values),
            LatticeKind::Monoclinic { angle } => {
                full[..3].copy_from_slice(&values[..3]);
                full[3 + angle] = values[3];
            }
            LatticeKind::Orthorhombic => full[..3].copy_from_slice(values),
            LatticeKind::PlaneUnique { plane, unique } => {
                let representatives = [plane[0].min(unique), plane[0].max(unique)];
                let plane_value = values[usize::from(representatives[1] == plane[0])];
                let unique_value = values[usize::from(representatives[1] == unique)];
                full[plane[0]] = plane_value;
                full[plane[1]] = plane_value;
                full[unique] = unique_value;
            }
            LatticeKind::Rhombohedral => {
                full[..3].fill(values[0]);
                full[3..].fill(values[1]);
            }
            LatticeKind::Cubic => full[..3].fill(values[0]),
        }
        let cell = UnitCell {
            a_angstrom: full[0],
            b_angstrom: full[1],
            c_angstrom: full[2],
            alpha_deg: full[3],
            beta_deg: full[4],
            gamma_deg: full[5],
        };
        cell.geometry().map_err(LatticeError::Cell)?;
        Ok(cell)
    }

    /// Return row-major `d(a,b,c,alpha,beta,gamma)/d(independent)`.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] if values do not describe a physical cell.
    pub fn cell_jacobian(&self, values: &[f64]) -> Result<Vec<f64>, LatticeError> {
        self.to_cell(values)?;
        let columns = values.len();
        let mut matrix = vec![0.0; 6 * columns];
        let mut set = |row: usize, column: usize| matrix[row * columns + column] = 1.0;
        match self.kind {
            LatticeKind::Triclinic => (0..6).for_each(|index| set(index, index)),
            LatticeKind::Monoclinic { angle } => {
                (0..3).for_each(|index| set(index, index));
                set(3 + angle, 3);
            }
            LatticeKind::Orthorhombic => (0..3).for_each(|index| set(index, index)),
            LatticeKind::PlaneUnique { plane, unique } => {
                let representatives = [plane[0].min(unique), plane[0].max(unique)];
                let plane_column = usize::from(representatives[1] == plane[0]);
                let unique_column = usize::from(representatives[1] == unique);
                set(plane[0], plane_column);
                set(plane[1], plane_column);
                set(unique, unique_column);
            }
            LatticeKind::Rhombohedral => {
                (0..3).for_each(|row| set(row, 0));
                (3..6).for_each(|row| set(row, 1));
            }
            LatticeKind::Cubic => (0..3).for_each(|row| set(row, 0)),
        }
        Ok(matrix)
    }
}

/// Finite box bounds in one lattice parameterization.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeBounds {
    parameter_names: Vec<String>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl LatticeBounds {
    /// Validate finite ordered bounds containing the reference lattice.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] for shape/order/physical-corner failures.
    pub fn new(
        parameterization: &LatticeParameterization,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result<Self, LatticeError> {
        let count = parameterization.parameter_names.len();
        if lower.len() != count
            || upper.len() != count
            || lower.iter().chain(&upper).any(|value| !value.is_finite())
            || lower.iter().zip(&upper).any(|(low, high)| low >= high)
        {
            return Err(LatticeError::InvalidBounds);
        }
        let bounds = Self {
            parameter_names: parameterization.parameter_names.clone(),
            lower,
            upper,
        };
        bounds.validate_for(parameterization)?;
        Ok(bounds)
    }

    /// Construct explicit relative-length and absolute-angle bounds.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] for invalid controls or physical corners.
    pub fn around(
        parameterization: &LatticeParameterization,
        relative_length: f64,
        angle_delta_deg: f64,
    ) -> Result<Self, LatticeError> {
        if !relative_length.is_finite()
            || !(0.0..1.0).contains(&relative_length)
            || relative_length == 0.0
            || !angle_delta_deg.is_finite()
            || angle_delta_deg <= 0.0
        {
            return Err(LatticeError::InvalidBounds);
        }
        let values = parameterization.values_from_cell(parameterization.reference_cell)?;
        let mut lower = Vec::with_capacity(values.len());
        let mut upper = Vec::with_capacity(values.len());
        for (name, value) in parameterization.parameter_names.iter().zip(values) {
            if name.ends_with("_angstrom") {
                lower.push(value * (1.0 - relative_length));
                upper.push(value * (1.0 + relative_length));
            } else {
                lower.push((value - angle_delta_deg).max(f64::from_bits(1)));
                upper.push((value + angle_delta_deg).min(f64::from_bits(180_f64.to_bits() - 1)));
            }
        }
        Self::new(parameterization, lower, upper)
    }

    /// Borrow lower bounds.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Borrow upper bounds.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// Borrow the parameter names that define bound order and meaning.
    #[must_use]
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Return deterministic corners of the independent-parameter box.
    #[must_use]
    pub fn corner_values(&self) -> Vec<Vec<f64>> {
        let count = 1_usize << self.lower.len();
        (0..count)
            .map(|mask| {
                (0..self.lower.len())
                    .map(|index| {
                        if mask & (1 << index) == 0 {
                            self.lower[index]
                        } else {
                            self.upper[index]
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn contains(&self, values: &[f64]) -> bool {
        values.len() == self.lower.len()
            && values
                .iter()
                .zip(self.lower.iter().zip(&self.upper))
                .all(|(value, (low, high))| low <= value && value <= high)
    }

    fn validate_for(&self, parameterization: &LatticeParameterization) -> Result<(), LatticeError> {
        if self.parameter_names != parameterization.parameter_names
            || !self.contains(&parameterization.values_from_cell(parameterization.reference_cell)?)
        {
            return Err(LatticeError::InvalidBounds);
        }
        for corner in self.corner_values() {
            parameterization.to_cell(&corner)?;
        }
        Ok(())
    }
}

/// CW reflection geometry and independent lattice derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct CwLatticeGeometry {
    /// D-spacings in reflection order.
    pub d_spacing_angstrom: Vec<f64>,
    /// Two-theta positions in degrees.
    pub two_theta_deg: Vec<f64>,
    /// Row-major reflection-by-parameter d-spacing derivatives.
    pub d_d_spacing_d_parameters: Vec<f64>,
    /// Row-major reflection-by-parameter two-theta derivatives.
    pub d_two_theta_d_parameters: Vec<f64>,
    /// Stable independent parameter names.
    pub parameter_names: Vec<String>,
}

/// Calculate d-spacings, CW positions, and analytical independent-variable chains.
///
/// # Errors
///
/// Returns [`LatticeError`] for invalid cells/reflections or inaccessible peaks.
pub fn cw_lattice_geometry(
    parameterization: &LatticeParameterization,
    cell: UnitCell,
    hkl: &[[i32; 3]],
    wavelength_angstrom: f64,
) -> Result<CwLatticeGeometry, LatticeError> {
    if !wavelength_angstrom.is_finite() || wavelength_angstrom <= 0.0 {
        return Err(LatticeError::InvalidWavelength);
    }
    let values = parameterization.values_from_cell(cell)?;
    let chain = parameterization.cell_jacobian(&values)?;
    let columns = values.len();
    let geometry = cell.geometry().map_err(LatticeError::Cell)?;
    let mut spacing = Vec::with_capacity(hkl.len());
    let mut positions = Vec::with_capacity(hkl.len());
    let mut derivatives = vec![0.0; hkl.len() * columns];
    let mut spacing_derivatives = vec![0.0; hkl.len() * columns];
    for (reflection, hkl) in hkl.iter().copied().enumerate() {
        let (d, d_cell) = geometry
            .d_spacing_and_derivatives(hkl)
            .map_err(LatticeError::Cell)?;
        let argument = wavelength_angstrom / (2.0 * d);
        if argument >= 1.0 {
            return Err(LatticeError::InaccessibleReflection);
        }
        spacing.push(d);
        positions.push(2.0 * argument.asin().to_degrees());
        let per_d = -180.0 / std::f64::consts::PI * wavelength_angstrom
            / (d * d * (1.0 - argument * argument).sqrt());
        for parameter in 0..columns {
            let d_parameter = (0..6)
                .map(|cell_parameter| {
                    d_cell[cell_parameter] * chain[cell_parameter * columns + parameter]
                })
                .sum::<f64>();
            spacing_derivatives[reflection * columns + parameter] = d_parameter;
            derivatives[reflection * columns + parameter] = per_d * d_parameter;
        }
    }
    Ok(CwLatticeGeometry {
        d_spacing_angstrom: spacing,
        two_theta_deg: positions,
        d_d_spacing_d_parameters: spacing_derivatives,
        d_two_theta_d_parameters: derivatives,
        parameter_names: parameterization.parameter_names.clone(),
    })
}

/// Bounded monochromatic reflection topology with stable intensity transfer.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeReflectionDomain {
    parameterization: LatticeParameterization,
    bounds: LatticeBounds,
    wavelength_angstrom: f64,
    visible_two_theta_deg: [f64; 2],
    initial_intensity: f64,
    merge_friedel: bool,
    max_candidates: usize,
    guard_scale: f64,
}

impl LatticeReflectionDomain {
    /// Validate and own one bounded reflection-domain contract.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] for invalid scalar or bound state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        parameterization: LatticeParameterization,
        bounds: LatticeBounds,
        wavelength_angstrom: f64,
        visible_two_theta_deg: [f64; 2],
        initial_intensity: f64,
        merge_friedel: bool,
        max_candidates: usize,
        guard_scale: f64,
    ) -> Result<Self, LatticeError> {
        if !wavelength_angstrom.is_finite()
            || wavelength_angstrom <= 0.0
            || !initial_intensity.is_finite()
            || initial_intensity < 0.0
            || !visible_two_theta_deg.iter().all(|value| value.is_finite())
            || !(0.0 < visible_two_theta_deg[0]
                && visible_two_theta_deg[0] < visible_two_theta_deg[1]
                && visible_two_theta_deg[1] < 180.0)
            || max_candidates == 0
            || !guard_scale.is_finite()
            || guard_scale < 1.0
        {
            return Err(LatticeError::InvalidDomain);
        }
        bounds.validate_for(&parameterization)?;
        let domain = Self {
            parameterization,
            bounds,
            wavelength_angstrom,
            visible_two_theta_deg,
            initial_intensity,
            merge_friedel,
            max_candidates,
            guard_scale,
        };
        domain.guarded_d_range(domain.parameterization.reference_cell)?;
        Ok(domain)
    }

    /// Borrow the parameterization.
    #[must_use]
    pub const fn parameterization(&self) -> &LatticeParameterization {
        &self.parameterization
    }

    /// Borrow finite lattice bounds.
    #[must_use]
    pub const fn bounds(&self) -> &LatticeBounds {
        &self.bounds
    }

    /// Return the monochromatic wavelength.
    #[must_use]
    pub const fn wavelength_angstrom(&self) -> f64 {
        self.wavelength_angstrom
    }

    /// Return the visible two-theta interval in degrees.
    #[must_use]
    pub const fn visible_two_theta_deg(&self) -> [f64; 2] {
        self.visible_two_theta_deg
    }

    /// Return the initial value assigned to newly generated families.
    #[must_use]
    pub const fn initial_intensity(&self) -> f64 {
        self.initial_intensity
    }

    /// Return whether Friedel pairs are merged.
    #[must_use]
    pub const fn merge_friedel(&self) -> bool {
        self.merge_friedel
    }

    /// Return the reflection-candidate resource ceiling.
    #[must_use]
    pub const fn max_candidates(&self) -> usize {
        self.max_candidates
    }

    /// Return the conservative reflection-guard scale.
    #[must_use]
    pub const fn guard_scale(&self) -> f64 {
        self.guard_scale
    }

    /// Clone this guarded contract for another monochromatic wavelength.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] when the wavelength is invalid or the updated
    /// guard cannot cover the declared lattice box.
    pub fn with_wavelength(&self, wavelength_angstrom: f64) -> Result<Self, LatticeError> {
        Self::new(
            self.parameterization.clone(),
            self.bounds.clone(),
            wavelength_angstrom,
            self.visible_two_theta_deg,
            self.initial_intensity,
            self.merge_friedel,
            self.max_candidates,
            self.guard_scale,
        )
    }

    /// Validate a compatible cell inside the finite guard box.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] for an incompatible or out-of-bounds cell.
    pub fn validate_cell(&self, cell: UnitCell) -> Result<Vec<f64>, LatticeError> {
        let values = self.parameterization.values_from_cell(cell)?;
        if !self.bounds.contains(&values) {
            return Err(LatticeError::OutsideBounds);
        }
        Ok(values)
    }

    /// Generate guarded topology and transfer intensities by stable ID.
    ///
    /// # Errors
    ///
    /// Returns [`LatticeError`] for an incompatible cell or generation failure.
    pub fn generate(
        &self,
        cell: UnitCell,
        previous: Option<&BTreeMap<String, f64>>,
    ) -> Result<GeneratedLatticeDomain, LatticeError> {
        self.validate_cell(cell)?;
        if previous.is_some_and(|items| {
            items
                .values()
                .any(|value| !value.is_finite() || *value < 0.0)
        }) {
            return Err(LatticeError::InvalidIntensity);
        }
        let (min_d, max_d) = self.guarded_d_range(cell)?;
        let generator = PreparedReflectionGenerator::new(
            self.parameterization.space_group.clone(),
            self.merge_friedel,
            self.max_candidates,
        )
        .map_err(LatticeError::Generation)?;
        let generated = generator
            .generate(
                cell,
                ReflectionRange::DSpacing {
                    min_angstrom: min_d,
                    max_angstrom: max_d,
                },
            )
            .map_err(LatticeError::Generation)?;
        let physical = generated
            .into_iter()
            .filter(|item| self.wavelength_angstrom < 2.0 * item.d_spacing_angstrom)
            .collect::<Vec<_>>();
        if physical.is_empty() {
            return Err(LatticeError::NoPhysicalReflections);
        }
        let hkl = physical.iter().map(|item| item.hkl).collect::<Vec<_>>();
        let geometry =
            cw_lattice_geometry(&self.parameterization, cell, &hkl, self.wavelength_angstrom)?;
        let reflection_ids = physical
            .iter()
            .map(|item| item.reflection_id.clone())
            .collect::<Vec<_>>();
        let previous_ids = previous
            .map(|values| values.keys().cloned().collect::<BTreeSet<_>>())
            .unwrap_or_default();
        let current_ids = reflection_ids.iter().cloned().collect::<BTreeSet<_>>();
        let preserved_reflection_count = current_ids.intersection(&previous_ids).count();
        let intensities = reflection_ids
            .iter()
            .map(|id| {
                previous
                    .and_then(|values| values.get(id))
                    .copied()
                    .unwrap_or(self.initial_intensity)
            })
            .collect::<Vec<_>>();
        let visible = geometry
            .two_theta_deg
            .iter()
            .map(|value| {
                self.visible_two_theta_deg[0] <= *value && *value <= self.visible_two_theta_deg[1]
            })
            .collect();
        let added_reflection_ids = reflection_ids
            .iter()
            .filter(|id| !previous_ids.contains(*id))
            .cloned()
            .collect();
        let removed_reflection_ids = previous
            .map(|items| {
                items
                    .keys()
                    .filter(|id| !current_ids.contains(*id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        Ok(GeneratedLatticeDomain {
            reflection_ids,
            hkl,
            multiplicity: physical.iter().map(|item| item.multiplicity).collect(),
            d_spacing_angstrom: geometry.d_spacing_angstrom,
            two_theta_deg: geometry.two_theta_deg,
            integrated_intensity: intensities,
            visible,
            guarded_d_min_angstrom: min_d,
            guarded_d_max_angstrom: max_d,
            added_reflection_ids,
            removed_reflection_ids,
            preserved_reflection_count,
        })
    }

    fn guarded_d_range(&self, reference_cell: UnitCell) -> Result<(f64, f64), LatticeError> {
        let cells = self
            .bounds
            .corner_values()
            .into_iter()
            .map(|values| self.parameterization.to_cell(&values).map(cell_values))
            .collect::<Result<Vec<_>, _>>()?;
        let mut physical_lower = [f64::INFINITY; 6];
        let mut physical_upper = [f64::NEG_INFINITY; 6];
        for cell in cells {
            for index in 0..6 {
                physical_lower[index] = physical_lower[index].min(cell[index]);
                physical_upper[index] = physical_upper[index].max(cell[index]);
            }
        }
        let cos_lower = [physical_upper[3], physical_upper[4], physical_upper[5]]
            .map(|value| value.to_radians().cos());
        let cos_upper = [physical_lower[3], physical_lower[4], physical_lower[5]]
            .map(|value| value.to_radians().cos());
        let mut product_lower = f64::INFINITY;
        for mask in 0..8 {
            let product = (0..3)
                .map(|index| {
                    if mask & (1 << index) == 0 {
                        cos_lower[index]
                    } else {
                        cos_upper[index]
                    }
                })
                .product::<f64>();
            product_lower = product_lower.min(product);
        }
        let maximum_squares = (0..3)
            .map(|index| cos_lower[index].powi(2).max(cos_upper[index].powi(2)))
            .sum::<f64>();
        let angular_lower = 1.0 + 2.0 * product_lower - maximum_squares;
        if angular_lower <= 0.0 {
            return Err(LatticeError::UnboundedGuard);
        }
        let length_product = physical_lower[..3].iter().product::<f64>();
        let determinant_lower = length_product * length_product * angular_lower;
        let trace_upper = physical_upper[..3]
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let direct_eigenvalue_lower = 4.0 * determinant_lower / trace_upper.powi(2);
        let reciprocal_lower = trace_upper.recip();
        let reciprocal_upper = direct_eigenvalue_lower.recip();
        let reciprocal = reference_cell
            .geometry()
            .map_err(LatticeError::Cell)?
            .reciprocal_metric;
        let matrix = Matrix3::from_row_slice(&[
            reciprocal[0][0],
            reciprocal[0][1],
            reciprocal[0][2],
            reciprocal[1][0],
            reciprocal[1][1],
            reciprocal[1][2],
            reciprocal[2][0],
            reciprocal[2][1],
            reciprocal[2][2],
        ]);
        let eigenvalues = SymmetricEigen::new(matrix).eigenvalues;
        let minimum_ratio = (eigenvalues.min() / reciprocal_upper).sqrt() / self.guard_scale;
        let maximum_ratio = (eigenvalues.max() / reciprocal_lower).sqrt() * self.guard_scale;
        let theta_min = 0.5 * self.visible_two_theta_deg[0].to_radians();
        let theta_max = 0.5 * self.visible_two_theta_deg[1].to_radians();
        let visible_max = self.wavelength_angstrom / (2.0 * theta_min.sin());
        let visible_min = self.wavelength_angstrom / (2.0 * theta_max.sin());
        Ok((visible_min / maximum_ratio, visible_max / minimum_ratio))
    }
}

/// Generated guarded reflection arrays and topology-transfer diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedLatticeDomain {
    /// Stable reflection IDs.
    pub reflection_ids: Vec<String>,
    /// Canonical Miller indices.
    pub hkl: Vec<[i32; 3]>,
    /// Powder-family multiplicities under the configured Friedel policy.
    pub multiplicity: Vec<usize>,
    /// D-spacings.
    pub d_spacing_angstrom: Vec<f64>,
    /// Two-theta positions.
    pub two_theta_deg: Vec<f64>,
    /// Stable-ID-transferred integrated intensities.
    pub integrated_intensity: Vec<f64>,
    /// Visible-range mask inside guarded topology.
    pub visible: Vec<bool>,
    /// Lower guarded d-spacing limit used for generation.
    pub guarded_d_min_angstrom: f64,
    /// Upper guarded d-spacing limit used for generation.
    pub guarded_d_max_angstrom: f64,
    /// Newly added stable IDs.
    pub added_reflection_ids: Vec<String>,
    /// Removed stable IDs.
    pub removed_reflection_ids: Vec<String>,
    /// Stable IDs preserved from the previous intensity map.
    pub preserved_reflection_count: usize,
}

fn plane_unique_kind(basis: &[[i64; 6]]) -> Result<LatticeKind, LatticeError> {
    let pairs = (0..3)
        .flat_map(|left| (left + 1..3).map(move |right| [left, right]))
        .filter(|pair| metric_columns_equal(basis, pair[0], pair[1]))
        .collect::<Vec<_>>();
    if pairs.len() != 1 {
        return Err(LatticeError::UnsupportedSetting);
    }
    let unique = (0..3)
        .find(|axis| !pairs[0].contains(axis))
        .ok_or(LatticeError::UnsupportedSetting)?;
    Ok(LatticeKind::PlaneUnique {
        plane: pairs[0],
        unique,
    })
}

fn metric_columns_equal(basis: &[[i64; 6]], left: usize, right: usize) -> bool {
    basis.iter().all(|row| row[left] == row[right])
}

fn names_for_kind(kind: &LatticeKind) -> Vec<String> {
    match kind {
        LatticeKind::Triclinic => CELL_NAMES.iter().map(ToString::to_string).collect(),
        LatticeKind::Monoclinic { angle } => [
            CELL_NAMES[0],
            CELL_NAMES[1],
            CELL_NAMES[2],
            CELL_NAMES[3 + angle],
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect(),
        LatticeKind::Orthorhombic => CELL_NAMES[..3].iter().map(ToString::to_string).collect(),
        LatticeKind::PlaneUnique { plane, unique } => {
            let mut axes = [plane[0], *unique];
            axes.sort_unstable();
            axes.into_iter()
                .map(|axis| CELL_NAMES[axis].to_owned())
                .collect()
        }
        LatticeKind::Rhombohedral => vec![CELL_NAMES[0].to_owned(), CELL_NAMES[3].to_owned()],
        LatticeKind::Cubic => vec![CELL_NAMES[0].to_owned()],
    }
}

fn cell_values(cell: UnitCell) -> [f64; 6] {
    [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ]
}

fn cells_close(left: UnitCell, right: UnitCell, tolerance: f64) -> bool {
    cell_values(left)
        .iter()
        .zip(cell_values(right))
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

#[allow(clippy::cast_precision_loss)]
fn validate_metric_compatibility(
    space_group: &SpaceGroup,
    cell: UnitCell,
) -> Result<(), LatticeError> {
    let metric = cell.geometry().map_err(LatticeError::Cell)?.direct_metric;
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
    for equation in &space_group.metric_constraints().equations {
        let residual = equation
            .iter()
            .zip(components)
            .map(|(coefficient, value)| *coefficient as f64 * value)
            .sum::<f64>();
        let coefficient_scale = equation.iter().copied().map(i64::unsigned_abs).sum::<u64>() as f64;
        if residual.abs() > 1.0e-10 * scale * coefficient_scale.max(1.0) {
            return Err(LatticeError::IncompatibleCell);
        }
    }
    Ok(())
}

/// Invalid lattice parameterization, bounds, geometry, or domain generation.
#[derive(Debug)]
pub enum LatticeError {
    /// Unit-cell evaluation failed.
    Cell(CellError),
    /// Reflection generation failed.
    Generation(ReflectionGenerationError),
    /// Exact setting is not represented by the supported conventional mapping.
    UnsupportedSetting,
    /// Supplied cell is incompatible with the parameterization.
    IncompatibleCell,
    /// Independent value vector has the wrong shape or non-finite entries.
    ValueShape,
    /// Bounds are invalid or do not contain the reference.
    InvalidBounds,
    /// A requested cell lies outside the domain's declared finite bounds.
    OutsideBounds,
    /// Domain scalar state is invalid.
    InvalidDomain,
    /// Wavelength is invalid.
    InvalidWavelength,
    /// A reflection is outside the monochromatic Bragg domain.
    InaccessibleReflection,
    /// No physical monochromatic reflection remains after guarded generation.
    NoPhysicalReflections,
    /// A transferred reflection intensity is negative or non-finite.
    InvalidIntensity,
    /// Angle bounds cannot produce a finite guarded search domain.
    UnboundedGuard,
}

impl Display for LatticeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cell(error) => Display::fmt(error, formatter),
            Self::Generation(error) => Display::fmt(error, formatter),
            Self::UnsupportedSetting => formatter.write_str("unsupported lattice setting"),
            Self::IncompatibleCell => {
                formatter.write_str("cell is incompatible with lattice setting")
            }
            Self::ValueShape => formatter
                .write_str("lattice values must match the independent variables and be finite"),
            Self::InvalidBounds => formatter.write_str(
                "lattice bounds must be finite, ordered, physical, and contain the reference",
            ),
            Self::OutsideBounds => {
                formatter.write_str("lattice cell lies outside the reflection-domain bounds")
            }
            Self::InvalidDomain => formatter.write_str("lattice reflection domain is invalid"),
            Self::InvalidWavelength => {
                formatter.write_str("wavelength must be positive and finite")
            }
            Self::InaccessibleReflection => {
                formatter.write_str("reflection lies outside the monochromatic Bragg domain")
            }
            Self::NoPhysicalReflections => {
                formatter.write_str("no physical reflections lie in the guarded CW domain")
            }
            Self::InvalidIntensity => formatter
                .write_str("transferred reflection intensities must be finite and non-negative"),
            Self::UnboundedGuard => formatter
                .write_str("lattice angle bounds are too broad for a finite guarded domain"),
        }
    }
}

impl Error for LatticeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cell(error) => Some(error),
            Self::Generation(error) => Some(error),
            _ => None,
        }
    }
}
