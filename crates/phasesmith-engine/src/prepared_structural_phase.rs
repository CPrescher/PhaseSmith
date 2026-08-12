//! Owned structural-phase preparation for application-neutral callers.

use phasesmith_core::{
    ConstantWavelengthInstrument, CwContributionsView, FcjGeometry, SupportPolicy,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedNeutronScattering, PreparedXrayScattering,
    SpaceGroup, UnitCell,
};
use phasesmith_execution::ExecutionContext;

use crate::structural_pattern::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPatternDenseResult,
    StructuralPatternError, StructuralPatternInputView, StructuralPatternJvpResult,
    StructuralPatternResult, StructuralPatternVjpResult,
    calculate_structural_pattern_dense_with_context, calculate_structural_pattern_jvp_with_context,
    calculate_structural_pattern_vjp_with_context, calculate_structural_pattern_with_context,
};

/// Owned crystallographic and scattering data for one structural phase.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralPhaseDefinition {
    /// Direct unit cell.
    pub cell: UnitCell,
    /// Validated exact symmetry group.
    pub space_group: SpaceGroup,
    /// Canonical Miller indices.
    pub hkl: Vec<[i32; 3]>,
    /// Powder multiplicity for every reflection.
    pub multiplicity: Vec<usize>,
    /// Asymmetric-unit fractional coordinates.
    pub fractional_xyz: Vec<[f64; 3]>,
    /// Asymmetric-site occupancies.
    pub occupancy: Vec<f64>,
    /// Asymmetric-site isotropic displacement in square ångströms.
    pub u_iso_angstrom2: Vec<f64>,
    /// True for asymmetric sites described by fixed CIF U tensors.
    pub anisotropic_mask: Vec<bool>,
    /// CIF U tensors in component order `11,22,33,23,13,12`.
    pub u_aniso_cif_angstrom2: Vec<[f64; 6]>,
    /// Exact built-in scattering-table key for every asymmetric site.
    pub scattering_species: Vec<String>,
    /// Fixed real X-ray dispersion offset for every site, or empty when absent.
    pub scattering_real_offset: Vec<f64>,
    /// Fixed imaginary X-ray dispersion offset for every site, or empty when absent.
    pub scattering_imag_offset: Vec<f64>,
    /// Structural phase scale.
    pub scale: f64,
    /// Fixed symmetry-expansion deduplication tolerance.
    pub coordinate_tolerance: f64,
    /// Built-in native scattering selection.
    pub scattering_model: BuiltInScatteringModel,
    /// Integrated-intensity correction selection.
    pub correction_model: IntegratedIntensityCorrectionModel,
}

impl StructuralPhaseDefinition {
    /// Validate owned reflection, site, scattering, and offset data.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralPatternError`] when array shapes disagree, offsets
    /// are invalid, or a built-in scattering-table key is unknown.
    pub fn validate(&self) -> Result<(), StructuralPatternError> {
        validate_definition(self)
    }
}

/// Borrowed experiment and sample-physics data for evaluating a prepared phase.
#[derive(Clone, Copy, Debug)]
pub struct PreparedStructuralPatternInputView<'a> {
    /// Sorted pattern grid in degrees `2theta`.
    pub x_deg: &'a [f64],
    /// Monochromatic constant-wavelength instrument parameters.
    pub instrument: ConstantWavelengthInstrument,
    /// Optional Finger--Cox--Jephcoat axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit zero/sample-displacement position correction.
    pub position_correction: MonochromaticPositionCorrection,
    /// Vectorized sample-physics contribution batch.
    pub contributions: CwContributionsView<'a>,
    /// Exact finite profile-support policy.
    pub support: SupportPolicy,
}

/// Reusable, application-neutral structural phase with an owned worker budget.
#[derive(Clone)]
pub struct PreparedStructuralPhase {
    definition: StructuralPhaseDefinition,
    execution: ExecutionContext,
}

impl PreparedStructuralPhase {
    /// Validate and take ownership of one structural phase.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralPatternError`] when reflection/site arrays disagree,
    /// fixed offsets are invalid, or a scattering-table key is unknown.
    pub fn new(
        definition: StructuralPhaseDefinition,
        execution: ExecutionContext,
    ) -> Result<Self, StructuralPatternError> {
        definition.validate()?;
        Ok(Self {
            definition,
            execution,
        })
    }

    /// Return the number of prepared reflections.
    #[must_use]
    pub fn reflection_count(&self) -> usize {
        self.definition.hkl.len()
    }

    /// Return the structural parameter count in the native derivative layout.
    #[must_use]
    pub fn structural_parameter_count(&self) -> usize {
        6 + 5 * self.definition.fractional_xyz.len() + 1
    }

    /// Return the exact worker budget owned by this prepared phase.
    #[must_use]
    pub fn execution_threads(&self) -> usize {
        self.execution.threads()
    }

    /// Borrow the validated owned structural definition.
    #[must_use]
    pub const fn definition(&self) -> &StructuralPhaseDefinition {
        &self.definition
    }

    /// Calculate values for the prepared phase.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralPatternError`] for invalid dynamic or structural
    /// inputs.
    pub fn calculate(
        &self,
        input: &PreparedStructuralPatternInputView<'_>,
    ) -> Result<StructuralPatternResult, StructuralPatternError> {
        self.with_input(input, calculate_structural_pattern_with_context)
    }

    /// Calculate values and a dense structural-pattern linearization.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralPatternError`] for invalid dynamic or structural
    /// inputs.
    pub fn linearize(
        &self,
        input: &PreparedStructuralPatternInputView<'_>,
    ) -> Result<StructuralPatternDenseResult, StructuralPatternError> {
        self.with_input(input, calculate_structural_pattern_dense_with_context)
    }

    /// Calculate values and a structural forward derivative product.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralPatternError`] for invalid inputs or tangent shape.
    pub fn jvp(
        &self,
        input: &PreparedStructuralPatternInputView<'_>,
        tangent: &[f64],
    ) -> Result<StructuralPatternJvpResult, StructuralPatternError> {
        self.with_input(input, |cell, group, structural_input, execution| {
            calculate_structural_pattern_jvp_with_context(
                cell,
                group,
                structural_input,
                tangent,
                execution,
            )
        })
    }

    /// Calculate values and a pattern-Jacobian transpose product.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralPatternError`] for invalid inputs or sample weights.
    pub fn vjp(
        &self,
        input: &PreparedStructuralPatternInputView<'_>,
        sample_weights: &[f64],
    ) -> Result<StructuralPatternVjpResult, StructuralPatternError> {
        self.with_input(input, |cell, group, structural_input, execution| {
            calculate_structural_pattern_vjp_with_context(
                cell,
                group,
                structural_input,
                sample_weights,
                execution,
            )
        })
    }

    fn with_input<R>(
        &self,
        input: &PreparedStructuralPatternInputView<'_>,
        operation: impl FnOnce(
            UnitCell,
            &SpaceGroup,
            &StructuralPatternInputView<'_>,
            &ExecutionContext,
        ) -> Result<R, StructuralPatternError>,
    ) -> Result<R, StructuralPatternError> {
        let definition = &self.definition;
        let species = definition
            .scattering_species
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let structural_input = StructuralPatternInputView {
            x_deg: input.x_deg,
            hkl: &definition.hkl,
            multiplicity: &definition.multiplicity,
            fractional_xyz: &definition.fractional_xyz,
            occupancy: &definition.occupancy,
            u_iso_angstrom2: &definition.u_iso_angstrom2,
            anisotropic_mask: &definition.anisotropic_mask,
            u_aniso_cif_angstrom2: &definition.u_aniso_cif_angstrom2,
            scattering_species: &species,
            scattering_real_offset: &definition.scattering_real_offset,
            scattering_imag_offset: &definition.scattering_imag_offset,
            scale: definition.scale,
            coordinate_tolerance: definition.coordinate_tolerance,
            instrument: input.instrument,
            axial_geometry: input.axial_geometry,
            position_correction: input.position_correction,
            correction_model: correction_for_wavelength(
                definition.correction_model,
                input.instrument.wavelength_angstrom,
            ),
            scattering_model: definition.scattering_model,
            contributions: input.contributions,
            support: input.support,
        };
        operation(
            definition.cell,
            &definition.space_group,
            &structural_input,
            &self.execution,
        )
    }
}

fn validate_definition(
    definition: &StructuralPhaseDefinition,
) -> Result<(), StructuralPatternError> {
    definition
        .cell
        .geometry()
        .map(|_| ())
        .map_err(StructuralPatternError::InvalidCell)?;
    if definition.hkl.len() != definition.multiplicity.len() {
        return Err(StructuralPatternError::ReflectionLengthMismatch);
    }
    let site_count = definition.fractional_xyz.len();
    if site_count != definition.occupancy.len()
        || site_count != definition.u_iso_angstrom2.len()
        || site_count != definition.anisotropic_mask.len()
        || site_count != definition.u_aniso_cif_angstrom2.len()
        || site_count != definition.scattering_species.len()
        || (!definition.scattering_real_offset.is_empty()
            && site_count != definition.scattering_real_offset.len())
        || (!definition.scattering_imag_offset.is_empty()
            && site_count != definition.scattering_imag_offset.len())
        || definition.scattering_real_offset.is_empty()
            != definition.scattering_imag_offset.is_empty()
    {
        return Err(StructuralPatternError::SiteLengthMismatch);
    }
    if definition
        .scattering_real_offset
        .iter()
        .chain(&definition.scattering_imag_offset)
        .any(|value| !value.is_finite())
    {
        return Err(StructuralPatternError::NonFiniteScatteringOffset);
    }
    match definition.scattering_model {
        BuiltInScatteringModel::XrayNonResonant => {
            PreparedXrayScattering::new(definition.scattering_species.iter().map(String::as_str))
                .map(|_| ())
        }
        BuiltInScatteringModel::NeutronNuclear => {
            PreparedNeutronScattering::new(definition.scattering_species.iter().map(String::as_str))
                .map(|_| ())
        }
    }
    .map_err(StructuralPatternError::Scattering)
}

fn correction_for_wavelength(
    correction: IntegratedIntensityCorrectionModel,
    wavelength_angstrom: f64,
) -> IntegratedIntensityCorrectionModel {
    match correction {
        IntegratedIntensityCorrectionModel::Neutral => IntegratedIntensityCorrectionModel::Neutral,
        IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp { .. } => {
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom,
            }
        }
        IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp { polarization, .. } => {
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                wavelength_angstrom,
                polarization,
            }
        }
        IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz { .. } => {
            IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
                wavelength_angstrom,
            }
        }
        IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg } => {
            IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_crystallography::SymmetryOperation;

    fn definition() -> StructuralPhaseDefinition {
        StructuralPhaseDefinition {
            cell: UnitCell {
                a_angstrom: 5.0,
                b_angstrom: 5.0,
                c_angstrom: 5.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).expect("P1"),
            hkl: vec![[1, 0, 0]],
            multiplicity: vec![2],
            fractional_xyz: vec![[0.0, 0.0, 0.0]],
            occupancy: vec![1.0],
            u_iso_angstrom2: vec![0.01],
            anisotropic_mask: vec![false],
            u_aniso_cif_angstrom2: vec![[0.0; 6]],
            scattering_species: vec!["Si".to_owned()],
            scattering_real_offset: Vec::new(),
            scattering_imag_offset: Vec::new(),
            scale: 1.0,
            coordinate_tolerance: 1.0e-10,
            scattering_model: BuiltInScatteringModel::XrayNonResonant,
            correction_model: IntegratedIntensityCorrectionModel::Neutral,
        }
    }

    #[test]
    fn prepared_phase_owns_validated_data_and_execution() {
        let phase = PreparedStructuralPhase::new(
            definition(),
            ExecutionContext::new(2).expect("execution context"),
        )
        .expect("prepared phase");
        assert_eq!(phase.reflection_count(), 1);
        assert_eq!(phase.structural_parameter_count(), 12);
        assert_eq!(phase.execution_threads(), 2);
    }

    #[test]
    fn prepared_phase_rejects_inconsistent_owned_shapes() {
        let mut invalid_reflections = definition();
        invalid_reflections.multiplicity.clear();
        assert!(matches!(
            PreparedStructuralPhase::new(invalid_reflections, ExecutionContext::serial()),
            Err(StructuralPatternError::ReflectionLengthMismatch)
        ));

        let mut invalid_sites = definition();
        invalid_sites.occupancy.clear();
        assert!(matches!(
            PreparedStructuralPhase::new(invalid_sites, ExecutionContext::serial()),
            Err(StructuralPatternError::SiteLengthMismatch)
        ));
    }

    #[test]
    fn prepared_phase_rejects_unknown_scattering_species() {
        let mut invalid = definition();
        invalid.scattering_species[0] = "not-an-element".to_owned();
        assert!(matches!(
            PreparedStructuralPhase::new(invalid, ExecutionContext::serial()),
            Err(StructuralPatternError::Scattering(_))
        ));
    }

    #[test]
    fn dynamic_wavelength_replaces_the_stored_correction_wavelength() {
        let correction = correction_for_wavelength(
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                wavelength_angstrom: 0.5,
                polarization: 0.7,
            },
            1.5406,
        );
        assert_eq!(
            correction,
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                wavelength_angstrom: 1.5406,
                polarization: 0.7,
            }
        );
    }
}
