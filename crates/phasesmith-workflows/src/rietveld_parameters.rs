//! Stable Rietveld parameter identities and structural derivative transforms.

use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_crystallography::P1ParameterLayout;
use phasesmith_model::RecordId;

use crate::{
    LatticeBounds, LatticeError, LatticeParameterization, ParameterBounds, ParameterError,
    ParameterKey, ParameterSet, ParameterSpec, RietveldPhase,
};

/// Structural parameter families selected for one native Rietveld layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct RietveldStructuralSelection {
    /// Refine symmetry-independent lattice variables.
    pub lattice: bool,
    /// Refine symmetry-allowed asymmetric-site coordinates.
    pub coordinates: bool,
    /// Refine asymmetric-site occupancies.
    pub occupancy: bool,
    /// Refine isotropic displacement values for isotropic sites.
    pub u_iso: bool,
    /// Refine one structural intensity scale per phase.
    pub phase_scale: bool,
}

/// Symmetry-allowed coordinate tangent basis for one asymmetric site.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteCoordinateModel {
    parameter_names: Vec<String>,
    basis: Vec<f64>,
    special_position: bool,
}

impl SiteCoordinateModel {
    /// Derive the null space of the exact site-stabilizer rotations.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldParameterError::InvalidCoordinateModel`] for a
    /// non-finite coordinate or non-positive/non-finite tolerance.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        space_group: &phasesmith_crystallography::SpaceGroup,
        coordinate: [f64; 3],
        tolerance: f64,
    ) -> Result<Self, RietveldParameterError> {
        if coordinate.iter().any(|value| !value.is_finite())
            || !tolerance.is_finite()
            || tolerance <= 0.0
        {
            return Err(RietveldParameterError::InvalidCoordinateModel);
        }
        let mut equations = Vec::<[f64; 3]>::new();
        for operation in space_group.operations() {
            let rotation = operation.rotation();
            let translation = operation.translation();
            let mut difference = [0.0; 3];
            for row in 0..3 {
                difference[row] = translation[row].as_f64() - coordinate[row];
                for column in 0..3 {
                    difference[row] += f64::from(rotation[row][column]) * coordinate[column];
                }
            }
            if difference
                .iter()
                .all(|value| (value - value.round()).abs() <= tolerance)
            {
                for row in 0..3 {
                    let mut equation = rotation[row].map(f64::from);
                    equation[row] -= 1.0;
                    equations.push(equation);
                }
            }
        }
        let (column_count, mut basis) = deterministic_null_space(equations);
        let special_position = column_count != 3;
        if !special_position {
            basis = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        }
        for value in &mut basis {
            if value.abs() < 1.0e-14 {
                *value = 0.0;
            }
        }
        let parameter_names = if special_position {
            (0..column_count).map(|index| format!("q{index}")).collect()
        } else {
            ["x", "y", "z"].map(str::to_owned).to_vec()
        };
        Ok(Self {
            parameter_names,
            basis,
            special_position,
        })
    }

    /// Borrow stable coordinate parameter names.
    #[must_use]
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Borrow the row-major `3 x parameter_count` tangent basis.
    #[must_use]
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }

    /// Return whether symmetry removes at least one coordinate direction.
    #[must_use]
    pub const fn is_special_position(&self) -> bool {
        self.special_position
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ParameterMapping {
    global_index: usize,
    native_terms: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
struct PhaseDerivativeLayout {
    native_count: usize,
    mappings: Vec<ParameterMapping>,
}

/// Ordered physical parameters and exact transforms to/from engine layouts.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldStructuralLayout {
    parameters: ParameterSet,
    phases: Vec<PhaseDerivativeLayout>,
    coordinate_models: Vec<Vec<SiteCoordinateModel>>,
    phase_ids: Vec<RecordId>,
    site_ids: Vec<Vec<RecordId>>,
    space_groups: Vec<phasesmith_crystallography::SpaceGroup>,
    anisotropic_masks: Vec<Vec<bool>>,
}

impl RietveldStructuralLayout {
    /// Build stable keys, bounds, scaling, and analytical engine transforms.
    ///
    /// `lattice_bounds` must have one entry per phase. A selected lattice
    /// requires bounds; an unselected lattice ignores a missing entry.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldParameterError`] for shape, lattice, identity, or
    /// parameter-domain failures.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        phases: &[RietveldPhase],
        selection: RietveldStructuralSelection,
        lattice_bounds: &[Option<LatticeBounds>],
    ) -> Result<Self, RietveldParameterError> {
        if phases.len() != lattice_bounds.len() {
            return Err(RietveldParameterError::PhaseCountMismatch);
        }
        let mut specs = Vec::new();
        let mut layouts = Vec::with_capacity(phases.len());
        let mut all_coordinate_models = Vec::with_capacity(phases.len());
        for (phase, supplied_bounds) in phases.iter().zip(lattice_bounds) {
            let definition = phase.definition();
            let native = P1ParameterLayout {
                site_count: definition.fractional_xyz.len(),
            };
            let mut mappings = Vec::new();
            if selection.lattice {
                let bounds = supplied_bounds
                    .as_ref()
                    .ok_or(RietveldParameterError::MissingLatticeBounds)?;
                let parameterization =
                    LatticeParameterization::new(definition.space_group.clone(), definition.cell)?;
                if bounds.parameter_names() != parameterization.parameter_names() {
                    return Err(RietveldParameterError::Lattice(LatticeError::InvalidBounds));
                }
                let values = parameterization.values_from_cell(definition.cell)?;
                let jacobian = parameterization.cell_jacobian(&values)?;
                let columns = values.len();
                for column in 0..columns {
                    let name = &parameterization.parameter_names()[column];
                    let unit = if name.ends_with("_angstrom") {
                        "angstrom"
                    } else {
                        "degree"
                    };
                    push_mapping(
                        &mut specs,
                        &mut mappings,
                        ParameterKey::new("lattice", phase.phase_id().as_str(), name)?,
                        values[column],
                        unit,
                        ParameterBounds::new(bounds.lower()[column], bounds.upper()[column])?,
                        values[column].abs().max(1.0),
                        (0..6)
                            .filter_map(|row| {
                                let coefficient = jacobian[row * columns + column];
                                (coefficient != 0.0).then_some((row, coefficient))
                            })
                            .collect(),
                    )?;
                }
            }
            let coordinate_models = definition
                .fractional_xyz
                .iter()
                .map(|coordinate| {
                    SiteCoordinateModel::new(
                        &definition.space_group,
                        *coordinate,
                        definition.coordinate_tolerance,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (site, (site_id, model)) in
                phase.site_ids().iter().zip(&coordinate_models).enumerate()
            {
                let owner = format!("{}/{}", phase.phase_id(), site_id);
                if selection.coordinates {
                    let columns = model.parameter_names.len();
                    for column in 0..columns {
                        let value = if model.special_position {
                            0.0
                        } else {
                            definition.fractional_xyz[site][column]
                        };
                        push_mapping(
                            &mut specs,
                            &mut mappings,
                            ParameterKey::new("site", &owner, &model.parameter_names[column])?,
                            value,
                            "fractional",
                            if model.special_position {
                                ParameterBounds::new(-0.5, 0.5)?
                            } else {
                                ParameterBounds::default()
                            },
                            1.0,
                            (0..3)
                                .filter_map(|row| {
                                    let coefficient = model.basis[row * columns + column];
                                    (coefficient != 0.0)
                                        .then_some((native.coordinate(site, row), coefficient))
                                })
                                .collect(),
                        )?;
                    }
                }
                if selection.occupancy {
                    let value = definition.occupancy[site];
                    push_mapping(
                        &mut specs,
                        &mut mappings,
                        ParameterKey::new("site", &owner, "occupancy")?,
                        value,
                        "fraction",
                        ParameterBounds::new(0.0, (2.0 * value + 0.1).max(1.0))?,
                        value.max(1.0),
                        vec![(native.occupancy(site), 1.0)],
                    )?;
                }
                if selection.u_iso && !definition.anisotropic_mask[site] {
                    let value = definition.u_iso_angstrom2[site];
                    push_mapping(
                        &mut specs,
                        &mut mappings,
                        ParameterKey::new("site", &owner, "u_iso_angstrom2")?,
                        value,
                        "angstrom^2",
                        ParameterBounds::new(0.0, (2.0 * value + 0.05).max(0.5))?,
                        value.max(0.01),
                        vec![(native.u_iso(site), 1.0)],
                    )?;
                }
            }
            if selection.phase_scale {
                push_mapping(
                    &mut specs,
                    &mut mappings,
                    ParameterKey::new("phase", phase.phase_id().as_str(), "scale")?,
                    definition.scale,
                    "relative",
                    ParameterBounds::new(0.0, f64::INFINITY)?,
                    definition.scale.abs().max(1.0),
                    vec![(native.scale(), 1.0)],
                )?;
            }
            layouts.push(PhaseDerivativeLayout {
                native_count: native.parameter_count(),
                mappings,
            });
            all_coordinate_models.push(coordinate_models);
        }
        Ok(Self {
            parameters: ParameterSet::new(specs)?,
            phases: layouts,
            coordinate_models: all_coordinate_models,
            phase_ids: phases
                .iter()
                .map(|phase| phase.phase_id().clone())
                .collect(),
            site_ids: phases
                .iter()
                .map(|phase| phase.site_ids().to_vec())
                .collect(),
            space_groups: phases
                .iter()
                .map(|phase| phase.definition().space_group.clone())
                .collect(),
            anisotropic_masks: phases
                .iter()
                .map(|phase| phase.definition().anisotropic_mask.clone())
                .collect(),
        })
    }

    /// Borrow ordered physical parameter specifications.
    #[must_use]
    pub const fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }

    /// Borrow symmetry-allowed site models in phase/site order.
    #[must_use]
    pub fn coordinate_models(&self) -> &[Vec<SiteCoordinateModel>] {
        &self.coordinate_models
    }

    pub(crate) fn validate_phases(
        &self,
        phases: &[RietveldPhase],
    ) -> Result<(), RietveldParameterError> {
        if phases.len() != self.phase_ids.len()
            || phases.iter().enumerate().any(|(index, phase)| {
                phase.phase_id() != &self.phase_ids[index]
                    || phase.site_ids() != self.site_ids[index]
                    || phase.definition().space_group != self.space_groups[index]
                    || phase.definition().anisotropic_mask != self.anisotropic_masks[index]
                    || 6 + 5 * phase.definition().fractional_xyz.len() + 1
                        != self.phases[index].native_count
            })
        {
            return Err(RietveldParameterError::PhaseIdentityMismatch);
        }
        Ok(())
    }

    /// Expand one physical parameter direction into native per-phase tangents.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldParameterError::DirectionLengthMismatch`] for a wrong
    /// global direction length.
    pub fn native_tangents(
        &self,
        direction: &[f64],
    ) -> Result<Vec<Vec<f64>>, RietveldParameterError> {
        if direction.len() != self.parameters.specs().len() {
            return Err(RietveldParameterError::DirectionLengthMismatch);
        }
        Ok(self
            .phases
            .iter()
            .map(|phase| {
                let mut tangent = vec![0.0; phase.native_count];
                for mapping in &phase.mappings {
                    for &(native_index, coefficient) in &mapping.native_terms {
                        tangent[native_index] += coefficient * direction[mapping.global_index];
                    }
                }
                tangent
            })
            .collect())
    }

    /// Project native per-phase reverse products into physical parameter order.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldParameterError`] for phase or native-gradient shape
    /// mismatches.
    pub fn project_native_gradients(
        &self,
        gradients: &[&[f64]],
    ) -> Result<Vec<f64>, RietveldParameterError> {
        if gradients.len() != self.phases.len() {
            return Err(RietveldParameterError::PhaseCountMismatch);
        }
        let mut projected = vec![0.0; self.parameters.specs().len()];
        for (phase, gradient) in self.phases.iter().zip(gradients) {
            if gradient.len() != phase.native_count {
                return Err(RietveldParameterError::NativeGradientLengthMismatch);
            }
            for mapping in &phase.mappings {
                projected[mapping.global_index] += mapping
                    .native_terms
                    .iter()
                    .map(|(native_index, coefficient)| coefficient * gradient[*native_index])
                    .sum::<f64>();
            }
        }
        Ok(projected)
    }
}

#[allow(clippy::too_many_arguments)]
fn push_mapping(
    specs: &mut Vec<ParameterSpec>,
    mappings: &mut Vec<ParameterMapping>,
    key: ParameterKey,
    value: f64,
    unit: &str,
    bounds: ParameterBounds,
    scale: f64,
    native_terms: Vec<(usize, f64)>,
) -> Result<(), ParameterError> {
    let global_index = specs.len();
    specs.push(ParameterSpec::new(key, value, unit, bounds, scale, true)?);
    mappings.push(ParameterMapping {
        global_index,
        native_terms,
    });
    Ok(())
}

fn deterministic_null_space(mut equations: Vec<[f64; 3]>) -> (usize, Vec<f64>) {
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0;
    for column in 0..3 {
        let Some(row) =
            (pivot_row..equations.len()).find(|row| equations[*row][column].abs() > 1.0e-12)
        else {
            continue;
        };
        equations.swap(pivot_row, row);
        let pivot = equations[pivot_row][column];
        for value in &mut equations[pivot_row] {
            *value /= pivot;
        }
        let pivot_values = equations[pivot_row];
        for (row, equation) in equations.iter_mut().enumerate() {
            if row != pivot_row {
                let factor = equation[column];
                for index in 0..3 {
                    equation[index] -= factor * pivot_values[index];
                }
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
    }
    let free_columns = (0..3)
        .filter(|column| !pivot_columns.contains(column))
        .collect::<Vec<_>>();
    let column_count = free_columns.len();
    let mut basis = vec![0.0; 3 * column_count];
    for (basis_column, free_column) in free_columns.iter().copied().enumerate() {
        let mut vector = [0.0; 3];
        vector[free_column] = 1.0;
        for (row, pivot_column) in pivot_columns.iter().copied().enumerate() {
            vector[pivot_column] = -equations[row][free_column];
        }
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        for row in 0..3 {
            basis[row * column_count + basis_column] = vector[row] / norm;
        }
    }
    (column_count, basis)
}

/// Invalid native Rietveld parameter-layout state.
#[derive(Debug)]
pub enum RietveldParameterError {
    /// Phase-aligned inputs have different lengths.
    PhaseCountMismatch,
    /// Lattice refinement was selected without finite bounds.
    MissingLatticeBounds,
    /// One physical direction has the wrong length.
    DirectionLengthMismatch,
    /// One native reverse product has the wrong length.
    NativeGradientLengthMismatch,
    /// Layout phase/site identities differ from the calculation request.
    PhaseIdentityMismatch,
    /// Site coordinate or stabilizer tolerance is invalid.
    InvalidCoordinateModel,
    /// Stable parameter construction failed.
    Parameter(ParameterError),
    /// Setting-aware lattice construction failed.
    Lattice(LatticeError),
}

impl Display for RietveldParameterError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PhaseCountMismatch => formatter.write_str("Rietveld phase counts must match"),
            Self::MissingLatticeBounds => {
                formatter.write_str("selected Rietveld lattices require finite bounds")
            }
            Self::DirectionLengthMismatch => {
                formatter.write_str("Rietveld parameter direction length mismatch")
            }
            Self::NativeGradientLengthMismatch => {
                formatter.write_str("Rietveld native gradient length mismatch")
            }
            Self::PhaseIdentityMismatch => {
                formatter.write_str("Rietveld parameter layout phase identities differ")
            }
            Self::InvalidCoordinateModel => {
                formatter.write_str("Rietveld site coordinate model is invalid")
            }
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Lattice(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RietveldParameterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parameter(error) => Some(error),
            Self::Lattice(error) => Some(error),
            Self::PhaseCountMismatch
            | Self::MissingLatticeBounds
            | Self::DirectionLengthMismatch
            | Self::NativeGradientLengthMismatch
            | Self::PhaseIdentityMismatch
            | Self::InvalidCoordinateModel => None,
        }
    }
}

impl From<ParameterError> for RietveldParameterError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}

impl From<LatticeError> for RietveldParameterError {
    fn from(value: LatticeError) -> Self {
        Self::Lattice(value)
    }
}
