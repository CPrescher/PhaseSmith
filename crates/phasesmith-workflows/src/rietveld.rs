//! Python-free owned structural-pattern boundary for native Rietveld workflows.

use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::{
    ConstantWavelengthInstrument, CwContributionsError, FcjGeometry, OwnedCwContributionArrays,
    OwnedCwContributions, SupportPolicy,
};
use phasesmith_crystallography::IntegratedIntensityCorrectionModel;
use phasesmith_engine::{
    MonochromaticPositionCorrection, PreparedStructuralModel, PreparedStructuralMultiphase,
    PreparedStructuralPhase, PreparedStructuralSpectrum, StructuralCalculationRequest,
    StructuralModelInput, StructuralMultiphaseError, StructuralPatternError,
    StructuralPatternResult, StructuralPhaseDefinition, StructuralSpectrumError,
    calculate_monochromatic_reflection_geometry,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{DomainError, FixedWavelengthSpectrum, PatternRecord, RecordId};

use crate::{
    BackgroundError, BackgroundModel, DifferentiableBackground, LatticeError,
    LatticeReflectionDomain, ResidualError, ResidualEvaluation, ResidualOptions,
    RietveldSamplePhysicsModel, SamplePhysicsError, evaluate_residuals,
};

/// One owned monochromatic structural phase and its sample-physics inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldPhase {
    phase_id: RecordId,
    name: String,
    site_ids: Vec<RecordId>,
    reflection_ids: Vec<String>,
    definition: StructuralPhaseDefinition,
    contributions: OwnedCwContributions,
    sample_physics: Option<RietveldSamplePhysicsModel>,
    reflection_domain: Option<LatticeReflectionDomain>,
}

impl RietveldPhase {
    /// Validate and own one built-in structural phase.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for an empty name, invalid definition, or a
    /// contribution batch with the wrong reflection count.
    pub fn new(
        phase_id: RecordId,
        name: impl Into<String>,
        definition: StructuralPhaseDefinition,
        contributions: OwnedCwContributions,
    ) -> Result<Self, RietveldError> {
        let site_ids = (0..definition.fractional_xyz.len())
            .map(|index| RecordId::new(format!("site-{index}")))
            .collect::<Result<Vec<_>, _>>()
            .map_err(RietveldError::Pattern)?;
        Self::new_with_site_ids(phase_id, name, site_ids, definition, contributions)
    }

    /// Validate and own one phase with explicit stable asymmetric-site IDs.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] when site IDs are missing or duplicated, or
    /// when another phase invariant is invalid.
    pub fn new_with_site_ids(
        phase_id: RecordId,
        name: impl Into<String>,
        site_ids: Vec<RecordId>,
        definition: StructuralPhaseDefinition,
        contributions: OwnedCwContributions,
    ) -> Result<Self, RietveldError> {
        let reflection_ids = definition
            .hkl
            .iter()
            .map(|hkl| reflection_id(*hkl))
            .collect();
        let phase = Self {
            phase_id,
            name: name.into(),
            site_ids,
            reflection_ids,
            definition,
            contributions,
            sample_physics: None,
            reflection_domain: None,
        };
        phase.validate()?;
        Ok(phase)
    }

    /// Generate a structural phase from one bounded reflection-domain contract.
    ///
    /// Reflection arrays are regenerated from the definition's cell and start
    /// with neutral sample-physics contributions. Subsequent accepted cell
    /// changes transfer contribution arrays by stable reflection ID.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for an invalid domain, definition, or phase.
    pub fn from_lattice_domain(
        phase_id: RecordId,
        name: impl Into<String>,
        site_ids: Vec<RecordId>,
        mut definition: StructuralPhaseDefinition,
        reflection_domain: LatticeReflectionDomain,
    ) -> Result<Self, RietveldError> {
        let generated = reflection_domain
            .generate(definition.cell, None)
            .map_err(RietveldError::Lattice)?;
        definition.hkl = generated.hkl;
        definition.multiplicity = generated.multiplicity;
        let phase = Self {
            phase_id,
            name: name.into(),
            site_ids,
            reflection_ids: generated.reflection_ids,
            contributions: OwnedCwContributions::neutral(definition.hkl.len()),
            sample_physics: None,
            definition,
            reflection_domain: Some(reflection_domain),
        };
        phase.validate()?;
        Ok(phase)
    }

    /// Revalidate adapter-decoded phase identity, structure, and topology.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for invalid phase state.
    pub fn validate(&self) -> Result<(), RietveldError> {
        if self.name.trim().is_empty() {
            return Err(RietveldError::InvalidPhaseName);
        }
        self.definition
            .validate()
            .map_err(RietveldError::StructuralPattern)?;
        if self.site_ids.len() != self.definition.fractional_xyz.len() {
            return Err(RietveldError::SiteIdCountMismatch);
        }
        if self
            .site_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != self.site_ids.len()
        {
            return Err(RietveldError::DuplicateSiteId);
        }
        if self.contributions.reflection_count() != self.definition.hkl.len() {
            return Err(RietveldError::ContributionCountMismatch);
        }
        if self.reflection_ids.len() != self.definition.hkl.len()
            || self
                .reflection_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.reflection_ids.len()
        {
            return Err(RietveldError::ReflectionIdentityMismatch);
        }
        if self
            .reflection_ids
            .iter()
            .zip(&self.definition.hkl)
            .any(|(id, hkl)| id != &reflection_id(*hkl))
        {
            return Err(RietveldError::ReflectionTopologyMismatch);
        }
        if let Some(domain) = &self.reflection_domain {
            domain
                .validate_cell(self.definition.cell)
                .map_err(RietveldError::Lattice)?;
        }
        Ok(())
    }

    /// Borrow the stable phase ID.
    #[must_use]
    pub const fn phase_id(&self) -> &RecordId {
        &self.phase_id
    }

    /// Borrow the human-readable phase name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow stable asymmetric-site IDs in structural-array order.
    #[must_use]
    pub fn site_ids(&self) -> &[RecordId] {
        &self.site_ids
    }

    /// Borrow stable reflection-family IDs in calculation order.
    #[must_use]
    pub fn reflection_ids(&self) -> &[String] {
        &self.reflection_ids
    }

    /// Borrow the complete structural definition.
    #[must_use]
    pub const fn definition(&self) -> &StructuralPhaseDefinition {
        &self.definition
    }

    /// Borrow the owned sample-physics contribution batch.
    #[must_use]
    pub const fn contributions(&self) -> &OwnedCwContributions {
        &self.contributions
    }

    /// Borrow the optional built-in sample-physics model record.
    #[must_use]
    pub const fn sample_physics(&self) -> Option<&RietveldSamplePhysicsModel> {
        self.sample_physics.as_ref()
    }

    /// Attach one built-in sample-physics model to this phase.
    ///
    /// The static contribution batch is retained only as the fixed-provider
    /// fallback and is not composed with the built-in model.
    #[must_use]
    pub fn with_sample_physics(mut self, model: RietveldSamplePhysicsModel) -> Self {
        self.sample_physics = Some(model);
        self
    }

    /// Remove the built-in sample-physics model without changing the phase's
    /// structure, reflection topology, or fixed contribution fallback.
    #[must_use]
    pub fn without_sample_physics(mut self) -> Self {
        self.sample_physics = None;
        self
    }

    pub(crate) fn replace_sample_physics(&self, model: RietveldSamplePhysicsModel) -> Self {
        let mut phase = self.clone();
        phase.sample_physics = Some(model);
        phase
    }

    pub(crate) fn resolved_sample_physics(
        &self,
        instrument: ConstantWavelengthInstrument,
        position_correction: MonochromaticPositionCorrection,
    ) -> Result<(OwnedCwContributions, Vec<String>), RietveldError> {
        let Some(model) = &self.sample_physics else {
            return Ok((self.contributions.clone(), Vec::new()));
        };
        let geometry = calculate_monochromatic_reflection_geometry(
            self.definition.cell,
            &self.definition.hkl,
            instrument,
            position_correction,
        )
        .map_err(RietveldError::StructuralPattern)?;
        let evaluated = model
            .evaluate(
                &self.definition.hkl,
                &geometry.two_theta_deg,
                self.definition.cell,
                instrument.wavelength_angstrom,
            )
            .map_err(RietveldError::SamplePhysics)?;
        Ok((evaluated.contributions, evaluated.parameter_names))
    }

    /// Replace sample-physics contributions without changing phase topology.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] when the contribution reflection count does
    /// not match the current stable reflection list.
    pub fn with_contributions(
        &self,
        contributions: OwnedCwContributions,
    ) -> Result<Self, RietveldError> {
        let mut phase = self.clone();
        phase.contributions = contributions;
        phase.validate()?;
        Ok(phase)
    }

    /// Borrow the guarded reflection domain for a dynamic phase.
    #[must_use]
    pub const fn reflection_domain(&self) -> Option<&LatticeReflectionDomain> {
        self.reflection_domain.as_ref()
    }

    /// Regenerate a dynamic phase at another accepted bounded cell.
    ///
    /// Existing sample-physics values and parameter derivatives transfer by
    /// stable reflection ID. New reflection families receive neutral values.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for a fixed phase, an out-of-domain cell, or
    /// invalid transferred contributions.
    pub fn regenerate_lattice_at_cell(
        &self,
        cell: phasesmith_crystallography::UnitCell,
    ) -> Result<(Self, RietveldTopologyChange), RietveldError> {
        let domain = self
            .reflection_domain
            .as_ref()
            .ok_or(RietveldError::FixedReflectionTopology)?;
        let previous = self
            .reflection_ids
            .iter()
            .cloned()
            .map(|reflection_id| (reflection_id, 1.0))
            .collect::<std::collections::BTreeMap<_, _>>();
        let generated = domain
            .generate(cell, Some(&previous))
            .map_err(RietveldError::Lattice)?;
        let contributions = transfer_contributions(
            &self.reflection_ids,
            &generated.reflection_ids,
            &self.contributions,
        )?;
        let change = RietveldTopologyChange {
            phase_id: self.phase_id.clone(),
            added_reflection_ids: generated.added_reflection_ids.clone(),
            removed_reflection_ids: generated.removed_reflection_ids.clone(),
            preserved_reflection_count: generated.preserved_reflection_count,
        };
        let mut phase = self.clone();
        phase.definition.cell = cell;
        phase.definition.hkl = generated.hkl;
        phase.definition.multiplicity = generated.multiplicity;
        phase.reflection_ids = generated.reflection_ids;
        phase.contributions = contributions;
        phase.validate()?;
        Ok((phase, change))
    }

    pub(crate) fn with_definition(
        &self,
        definition: StructuralPhaseDefinition,
    ) -> Result<Self, RietveldError> {
        if self.reflection_domain.is_some() && definition.cell != self.definition.cell {
            let (mut phase, _) = self.regenerate_lattice_at_cell(definition.cell)?;
            let hkl = std::mem::take(&mut phase.definition.hkl);
            let multiplicity = std::mem::take(&mut phase.definition.multiplicity);
            phase.definition = definition;
            phase.definition.hkl = hkl;
            phase.definition.multiplicity = multiplicity;
            phase.validate()?;
            return Ok(phase);
        }
        let mut phase = self.clone();
        phase.definition = definition;
        phase.validate()?;
        Ok(phase)
    }

    pub(crate) fn restart_compatible(&self, requested: &Self) -> bool {
        self.restart_compatible_with_wavelength(requested, false)
    }

    pub(crate) fn restart_compatible_with_wavelength(
        &self,
        requested: &Self,
        allow_wavelength_change: bool,
    ) -> bool {
        self.phase_id == requested.phase_id
            && self.site_ids == requested.site_ids
            && self.definition.space_group == requested.definition.space_group
            && self.definition.anisotropic_mask == requested.definition.anisotropic_mask
            && self.definition.u_aniso_cif_angstrom2 == requested.definition.u_aniso_cif_angstrom2
            && self.definition.scattering_species == requested.definition.scattering_species
            && self.definition.scattering_real_offset == requested.definition.scattering_real_offset
            && self.definition.scattering_imag_offset == requested.definition.scattering_imag_offset
            && self.definition.coordinate_tolerance.to_bits()
                == requested.definition.coordinate_tolerance.to_bits()
            && self.definition.scattering_model == requested.definition.scattering_model
            && (self.definition.correction_model == requested.definition.correction_model
                || (allow_wavelength_change
                    && correction_identity_matches(
                        self.definition.correction_model,
                        requested.definition.correction_model,
                    )))
            && sample_physics_identity_matches(
                self.sample_physics.as_ref(),
                requested.sample_physics.as_ref(),
            )
            && (self.reflection_domain == requested.reflection_domain
                || (allow_wavelength_change
                    && reflection_domain_identity_matches(
                        self.reflection_domain.as_ref(),
                        requested.reflection_domain.as_ref(),
                    )))
            && (self.reflection_domain.is_some()
                || (self.reflection_ids == requested.reflection_ids
                    && self.definition.hkl == requested.definition.hkl
                    && self.definition.multiplicity == requested.definition.multiplicity))
    }

    pub(crate) fn with_wavelength(&self, wavelength_angstrom: f64) -> Result<Self, RietveldError> {
        let mut phase = self.clone();
        phase.definition.correction_model = match phase.definition.correction_model {
            IntegratedIntensityCorrectionModel::Neutral => {
                IntegratedIntensityCorrectionModel::Neutral
            }
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp { .. } => {
                IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                    wavelength_angstrom,
                }
            }
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                polarization, ..
            } => IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                wavelength_angstrom,
                polarization,
            },
            IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz { .. } => {
                IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
                    wavelength_angstrom,
                }
            }
            IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg } => {
                IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg }
            }
        };
        if let Some(domain) = &self.reflection_domain {
            phase.reflection_domain = Some(
                domain
                    .with_wavelength(wavelength_angstrom)
                    .map_err(RietveldError::Lattice)?,
            );
            phase = phase.regenerate_lattice_at_cell(phase.definition.cell)?.0;
        }
        phase.validate()?;
        Ok(phase)
    }

    fn correction_wavelength(&self) -> Option<f64> {
        match self.definition.correction_model {
            IntegratedIntensityCorrectionModel::Neutral
            | IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { .. } => None,
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom,
            }
            | IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                wavelength_angstrom,
                ..
            }
            | IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
                wavelength_angstrom,
            } => Some(wavelength_angstrom),
        }
    }
}

fn correction_identity_matches(
    left: IntegratedIntensityCorrectionModel,
    right: IntegratedIntensityCorrectionModel,
) -> bool {
    match (left, right) {
        (
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp { .. },
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp { .. },
        )
        | (
            IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz { .. },
            IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz { .. },
        ) => true,
        (
            IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: left,
            },
            IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: right,
            },
        )
        | (
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                polarization: left, ..
            },
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                polarization: right,
                ..
            },
        ) => left.to_bits() == right.to_bits(),
        _ => false,
    }
}

fn reflection_domain_identity_matches(
    left: Option<&LatticeReflectionDomain>,
    right: Option<&LatticeReflectionDomain>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => left
            .with_wavelength(right.wavelength_angstrom())
            .is_ok_and(|updated| updated == *right),
        _ => false,
    }
}

fn sample_physics_identity_matches(
    left: Option<&RietveldSamplePhysicsModel>,
    right: Option<&RietveldSamplePhysicsModel>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => sample_physics_model_identity_matches(left, right),
        _ => false,
    }
}

fn sample_physics_model_identity_matches(
    left: &RietveldSamplePhysicsModel,
    right: &RietveldSamplePhysicsModel,
) -> bool {
    match (left, right) {
        (
            RietveldSamplePhysicsModel::IsotropicSize {
                shape_factor: left, ..
            },
            RietveldSamplePhysicsModel::IsotropicSize {
                shape_factor: right,
                ..
            },
        ) => left.to_bits() == right.to_bits(),
        (
            RietveldSamplePhysicsModel::IsotropicMicrostrain { .. },
            RietveldSamplePhysicsModel::IsotropicMicrostrain { .. },
        )
        | (
            RietveldSamplePhysicsModel::IsotropicLorentzianMicrostrain { .. },
            RietveldSamplePhysicsModel::IsotropicLorentzianMicrostrain { .. },
        ) => true,
        (
            RietveldSamplePhysicsModel::MarchDollase {
                preferred_axis_hkl: left,
                ..
            },
            RietveldSamplePhysicsModel::MarchDollase {
                preferred_axis_hkl: right,
                ..
            },
        ) => left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits()),
        (
            RietveldSamplePhysicsModel::Composite(left),
            RietveldSamplePhysicsModel::Composite(right),
        ) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| sample_physics_model_identity_matches(left, right))
        }
        _ => false,
    }
}

/// Reflection-topology change attached to an accepted structural step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RietveldTopologyChange {
    /// Stable phase identity.
    pub phase_id: RecordId,
    /// Reflection families added at the accepted cell.
    pub added_reflection_ids: Vec<String>,
    /// Reflection families removed at the accepted cell.
    pub removed_reflection_ids: Vec<String>,
    /// Families preserved by stable identity.
    pub preserved_reflection_count: usize,
}

/// Observations, experiment state, and ordered structural phases.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldInput {
    /// Observed pattern and fixed supplied background.
    pub pattern: PatternRecord,
    /// Monochromatic constant-wavelength profile.
    pub instrument: ConstantWavelengthInstrument,
    /// Optional fixed spectrum; absent means a monochromatic calculation.
    pub fixed_spectrum: Option<FixedWavelengthSpectrum>,
    /// Optional Finger--Cox--Jephcoat axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit instrument/sample position correction.
    pub position_correction: MonochromaticPositionCorrection,
    /// Optional differentiable background added to the pattern's fixed values.
    pub background: Option<BackgroundModel>,
    /// Ordered non-empty built-in structural phases.
    pub phases: Vec<RietveldPhase>,
}

impl RietveldInput {
    /// Validate one native monochromatic Rietveld calculation request.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for invalid observations, experiment state,
    /// phase state, or duplicate phase IDs.
    pub fn new(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        axial_geometry: Option<FcjGeometry>,
        position_correction: MonochromaticPositionCorrection,
        phases: Vec<RietveldPhase>,
    ) -> Result<Self, RietveldError> {
        let input = Self {
            pattern,
            instrument,
            fixed_spectrum: None,
            axial_geometry,
            position_correction,
            background: None,
            phases,
        };
        input.validate()?;
        Ok(input)
    }

    /// Validate one native fixed-wavelength-spectrum Rietveld request.
    ///
    /// The instrument wavelength must equal the spectrum's first, reference
    /// component. Structural lattice/topology refinement remains restricted to
    /// monochromatic inputs.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for invalid observations, spectrum,
    /// experiment geometry, or structural phases.
    pub fn new_fixed_spectrum(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        spectrum: FixedWavelengthSpectrum,
        axial_geometry: Option<FcjGeometry>,
        position_correction: MonochromaticPositionCorrection,
        phases: Vec<RietveldPhase>,
    ) -> Result<Self, RietveldError> {
        let input = Self {
            pattern,
            instrument,
            fixed_spectrum: Some(spectrum),
            axial_geometry,
            position_correction,
            background: None,
            phases,
        };
        input.validate()?;
        Ok(input)
    }

    /// Validate a request with an additional native analytical background.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for invalid observations, experiment state,
    /// background state, or phase state.
    pub fn new_with_background(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        axial_geometry: Option<FcjGeometry>,
        position_correction: MonochromaticPositionCorrection,
        background: BackgroundModel,
        phases: Vec<RietveldPhase>,
    ) -> Result<Self, RietveldError> {
        let mut input = Self::new(
            pattern,
            instrument,
            axial_geometry,
            position_correction,
            phases,
        )?;
        input.background = Some(background);
        input.validate()?;
        Ok(input)
    }

    /// Validate a fixed-spectrum request with an analytical background.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for invalid observations, spectrum,
    /// background, experiment geometry, or structural phases.
    pub fn new_fixed_spectrum_with_background(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        spectrum: FixedWavelengthSpectrum,
        axial_geometry: Option<FcjGeometry>,
        position_correction: MonochromaticPositionCorrection,
        background: BackgroundModel,
        phases: Vec<RietveldPhase>,
    ) -> Result<Self, RietveldError> {
        let mut input = Self::new_fixed_spectrum(
            pattern,
            instrument,
            spectrum,
            axial_geometry,
            position_correction,
            phases,
        )?;
        input.background = Some(background);
        input.validate()?;
        Ok(input)
    }

    /// Revalidate an adapter-decoded complete calculation request.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for invalid observations, experiment state,
    /// background state, or phase state.
    pub fn validate(&self) -> Result<(), RietveldError> {
        self.pattern.validate().map_err(RietveldError::Pattern)?;
        if self.pattern.observed_y.is_none() {
            return Err(RietveldError::MissingObservations);
        }
        self.instrument
            .validate()
            .map_err(|_| RietveldError::InvalidInstrument)?;
        if self.fixed_spectrum.as_ref().is_some_and(|spectrum| {
            spectrum.wavelengths_angstrom()[0].to_bits()
                != self.instrument.wavelength_angstrom.to_bits()
        }) {
            return Err(RietveldError::SpectrumReferenceWavelengthMismatch);
        }
        if self.axial_geometry.is_some_and(|geometry| {
            !geometry.sample_over_radius.is_finite()
                || !geometry.detector_over_radius.is_finite()
                || geometry.sample_over_radius < 0.0
                || geometry.detector_over_radius < 0.0
        }) {
            return Err(RietveldError::InvalidAxialGeometry);
        }
        let correction = self.position_correction;
        if !correction.zero_shift_deg.is_finite()
            || (correction.bragg_brentano_mm.is_some()
                && correction.debye_scherrer_micrometre.is_some())
            || correction
                .bragg_brentano_mm
                .is_some_and(|(displacement, radius)| {
                    !displacement.is_finite() || !radius.is_finite() || radius <= 0.0
                })
            || correction
                .debye_scherrer_micrometre
                .is_some_and(|(x, y, radius)| {
                    !x.is_finite() || !y.is_finite() || !radius.is_finite() || radius <= 0.0
                })
        {
            return Err(RietveldError::InvalidPositionCorrection);
        }
        if self.phases.is_empty() {
            return Err(RietveldError::EmptyPhases);
        }
        if let Some(background) = &self.background {
            background
                .basis(&self.pattern.x_deg)
                .map_err(RietveldError::Background)?;
            background
                .calculate(&self.pattern.x_deg)
                .map_err(RietveldError::Background)?;
        }
        let mut identities = std::collections::BTreeSet::new();
        for phase in &self.phases {
            phase.validate()?;
            phase.resolved_sample_physics(self.instrument, self.position_correction)?;
            if self.fixed_spectrum.is_some() && phase.reflection_domain().is_some() {
                return Err(RietveldError::SpectrumReflectionDomain);
            }
            if phase.correction_wavelength().is_some_and(|wavelength| {
                wavelength.to_bits() != self.instrument.wavelength_angstrom.to_bits()
            }) {
                return Err(RietveldError::CorrectionWavelengthMismatch);
            }
            if phase.reflection_domain().is_some_and(|domain| {
                domain.wavelength_angstrom().to_bits()
                    != self.instrument.wavelength_angstrom.to_bits()
            }) {
                return Err(RietveldError::ReflectionWavelengthMismatch);
            }
            if !identities.insert(phase.phase_id.clone()) {
                return Err(RietveldError::DuplicatePhaseId);
            }
        }
        Ok(())
    }
}

pub(crate) fn prepare_phase_model(
    phase: &RietveldPhase,
    spectrum: Option<&FixedWavelengthSpectrum>,
    execution: &ExecutionPolicy,
) -> Result<PreparedStructuralModel, RietveldError> {
    match spectrum {
        None => PreparedStructuralPhase::new(phase.definition.clone(), execution.context().clone())
            .map(PreparedStructuralModel::monochromatic)
            .map_err(RietveldError::StructuralPattern),
        Some(spectrum) => PreparedStructuralSpectrum::new(
            &phase.definition,
            spectrum.wavelengths_angstrom().to_vec(),
            spectrum.relative_intensities(),
            execution.clone(),
        )
        .map(PreparedStructuralModel::fixed_spectrum)
        .map_err(RietveldError::StructuralSpectrum),
    }
}

pub(crate) fn resolve_phase_contributions(
    phase: &RietveldPhase,
    input: &RietveldInput,
) -> Result<Vec<OwnedCwContributions>, RietveldError> {
    let wavelengths = input.fixed_spectrum.as_ref().map_or_else(
        || vec![input.instrument.wavelength_angstrom],
        |spectrum| spectrum.wavelengths_angstrom().to_vec(),
    );
    wavelengths
        .into_iter()
        .map(|wavelength_angstrom| {
            let mut instrument = input.instrument;
            instrument.wavelength_angstrom = wavelength_angstrom;
            phase
                .resolved_sample_physics(instrument, input.position_correction)
                .map(|value| value.0)
        })
        .collect()
}

fn reflection_id(hkl: [i32; 3]) -> String {
    format!("hkl:{},{},{}", hkl[0], hkl[1], hkl[2])
}

fn transfer_contributions(
    previous_ids: &[String],
    current_ids: &[String],
    previous: &OwnedCwContributions,
) -> Result<OwnedCwContributions, RietveldError> {
    let old_count = previous_ids.len();
    let new_count = current_ids.len();
    let parameter_count = previous.parameter_count();
    let derivative_count =
        parameter_count
            .checked_mul(new_count)
            .ok_or(RietveldError::Contributions(
                CwContributionsError::AllocationOverflow,
            ))?;
    let old = previous.arrays();
    let mut arrays = OwnedCwContributionArrays {
        gaussian_variance_deg2: vec![0.0; new_count],
        lorentzian_fwhm_deg: vec![0.0; new_count],
        intensity_multiplier: vec![1.0; new_count],
        d_gaussian_variance_d_position: vec![0.0; new_count],
        d_lorentzian_fwhm_d_position: vec![0.0; new_count],
        d_intensity_multiplier_d_position: vec![0.0; new_count],
        d_gaussian_variance_d_parameters: vec![0.0; derivative_count],
        d_lorentzian_fwhm_d_parameters: vec![0.0; derivative_count],
        d_intensity_multiplier_d_parameters: vec![0.0; derivative_count],
    };
    let previous_index = previous_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    for (new_index, id) in current_ids.iter().enumerate() {
        let Some(&old_index) = previous_index.get(id) else {
            continue;
        };
        for (target, source) in [
            (
                &mut arrays.gaussian_variance_deg2,
                &old.gaussian_variance_deg2,
            ),
            (&mut arrays.lorentzian_fwhm_deg, &old.lorentzian_fwhm_deg),
            (&mut arrays.intensity_multiplier, &old.intensity_multiplier),
            (
                &mut arrays.d_gaussian_variance_d_position,
                &old.d_gaussian_variance_d_position,
            ),
            (
                &mut arrays.d_lorentzian_fwhm_d_position,
                &old.d_lorentzian_fwhm_d_position,
            ),
            (
                &mut arrays.d_intensity_multiplier_d_position,
                &old.d_intensity_multiplier_d_position,
            ),
        ] {
            target[new_index] = source[old_index];
        }
        for parameter in 0..parameter_count {
            let old_offset = parameter * old_count + old_index;
            let new_offset = parameter * new_count + new_index;
            arrays.d_gaussian_variance_d_parameters[new_offset] =
                old.d_gaussian_variance_d_parameters[old_offset];
            arrays.d_lorentzian_fwhm_d_parameters[new_offset] =
                old.d_lorentzian_fwhm_d_parameters[old_offset];
            arrays.d_intensity_multiplier_d_parameters[new_offset] =
                old.d_intensity_multiplier_d_parameters[old_offset];
        }
    }
    OwnedCwContributions::new(new_count, parameter_count, arrays)
        .map_err(RietveldError::Contributions)
}

/// Deterministic calculation controls shared by later native refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldCalculationOptions {
    /// Exact finite profile support in multiples of FWHM.
    pub support_fwhm: f64,
    /// Apply supplied one-sigma uncertainties to residual metrics.
    pub use_uncertainty: bool,
    /// Persistent bounded execution policy.
    pub execution: ExecutionPolicy,
}

impl RietveldCalculationOptions {
    /// Validate explicit support and execution controls.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError::InvalidOptions`] for invalid support.
    pub fn new(
        support_fwhm: f64,
        use_uncertainty: bool,
        execution: ExecutionPolicy,
    ) -> Result<Self, RietveldError> {
        let options = Self {
            support_fwhm,
            use_uncertainty,
            execution,
        };
        options.validate()?;
        Ok(options)
    }

    pub(crate) fn validate(&self) -> Result<(), RietveldError> {
        if !self.support_fwhm.is_finite() || self.support_fwhm <= 0.0 {
            return Err(RietveldError::InvalidOptions);
        }
        Ok(())
    }
}

/// One labeled phase contribution and its crystallographic intermediates.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldPhaseCalculation {
    /// Stable phase ID.
    pub phase_id: RecordId,
    /// Human-readable phase name.
    pub name: String,
    /// Complete structural/profile result from the native engine.
    pub result: StructuralPatternResult,
}

/// Display-ready structural calculation and residual metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldCalculation {
    /// Sum of structural phase profiles before fixed background.
    pub profile_y: Vec<f64>,
    /// Fixed supplied background in sample order.
    pub background_y: Vec<f64>,
    /// Complete calculated pattern (`profile_y + background_y`).
    pub y: Vec<f64>,
    /// Phase calculations in input order.
    pub phases: Vec<RietveldPhaseCalculation>,
    /// Residual arrays and scalar fit metrics.
    pub metrics: ResidualEvaluation,
}

/// Calculate a complete built-in monochromatic structural pattern.
///
/// # Errors
///
/// Returns [`RietveldError`] for invalid phase preparation, calculation, or
/// residual state.
pub fn calculate_rietveld_pattern(
    input: &RietveldInput,
    options: &RietveldCalculationOptions,
) -> Result<RietveldCalculation, RietveldError> {
    input.validate()?;
    options.validate()?;
    let models = input
        .phases
        .iter()
        .map(|phase| prepare_phase_model(phase, input.fixed_spectrum.as_ref(), &options.execution))
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = PreparedStructuralMultiphase::new(models, options.execution.clone())
        .map_err(RietveldError::StructuralMultiphase)?;
    let contributions = input
        .phases
        .iter()
        .map(|phase| resolve_phase_contributions(phase, input))
        .collect::<Result<Vec<_>, _>>()?;
    let request = StructuralCalculationRequest {
        x_deg: input.pattern.x_deg.clone(),
        instrument: input.instrument,
        axial_geometry: input.axial_geometry,
        position_correction: input.position_correction,
        phase_inputs: contributions
            .into_iter()
            .map(|contributions| StructuralModelInput { contributions })
            .collect(),
        support: SupportPolicy::FwhmMultiple(options.support_fwhm),
    };
    let calculated = prepared
        .calculate_request(request)
        .map_err(RietveldError::StructuralMultiphase)?;
    assemble_rietveld_calculation(input, options, calculated.phases)
}

pub(crate) fn assemble_rietveld_calculation(
    input: &RietveldInput,
    options: &RietveldCalculationOptions,
    phase_results: Vec<StructuralPatternResult>,
) -> Result<RietveldCalculation, RietveldError> {
    if phase_results.len() != input.phases.len() {
        return Err(RietveldError::CalculationShapeMismatch);
    }
    let sample_count = input.pattern.sample_count();
    let mut profile_y = vec![0.0; sample_count];
    for result in &phase_results {
        if result.accumulation.sample_count != sample_count
            || result.accumulation.y.len() != sample_count
        {
            return Err(RietveldError::CalculationShapeMismatch);
        }
        for (combined, value) in profile_y.iter_mut().zip(&result.accumulation.y) {
            *combined += value;
        }
    }
    let mut background_y = input.pattern.background_y.clone();
    if let Some(background) = &input.background {
        for (target, value) in background_y.iter_mut().zip(
            background
                .calculate(&input.pattern.x_deg)
                .map_err(RietveldError::Background)?,
        ) {
            *target += value;
        }
    }
    let y = profile_y
        .iter()
        .zip(&background_y)
        .map(|(profile, background)| profile + background)
        .collect::<Vec<_>>();
    if y.iter().any(|value| !value.is_finite()) {
        return Err(RietveldError::NonFiniteCalculation);
    }
    let metrics = evaluate_residuals(
        &input.pattern,
        &y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: 0,
        },
    )
    .map_err(RietveldError::Residual)?;
    let phases = input
        .phases
        .iter()
        .zip(phase_results)
        .map(|(phase, result)| RietveldPhaseCalculation {
            phase_id: phase.phase_id.clone(),
            name: phase.name.clone(),
            result,
        })
        .collect();
    Ok(RietveldCalculation {
        profile_y,
        background_y,
        y,
        phases,
        metrics,
    })
}

/// Invalid owned native Rietveld calculation state.
#[derive(Debug)]
pub enum RietveldError {
    /// Pattern domain state is invalid.
    Pattern(DomainError),
    /// Observations are required for residual-bearing Rietveld requests.
    MissingObservations,
    /// The constant-wavelength instrument is invalid.
    InvalidInstrument,
    /// Finger--Cox--Jephcoat geometry is invalid.
    InvalidAxialGeometry,
    /// Monochromatic position correction is invalid.
    InvalidPositionCorrection,
    /// At least one phase is required.
    EmptyPhases,
    /// Phase names must be non-empty.
    InvalidPhaseName,
    /// Phase IDs must be unique.
    DuplicatePhaseId,
    /// Sample-physics contributions must match the reflection count.
    ContributionCountMismatch,
    /// Stable reflection IDs are missing, duplicated, or mis-sized.
    ReflectionIdentityMismatch,
    /// A dynamic reflection list does not match its current cell/domain.
    ReflectionTopologyMismatch,
    /// A dynamic phase domain must use the experiment wavelength.
    ReflectionWavelengthMismatch,
    /// An integrated-intensity correction must use the experiment wavelength.
    CorrectionWavelengthMismatch,
    /// A spectrum's first component must equal the reference instrument wavelength.
    SpectrumReferenceWavelengthMismatch,
    /// Dynamic lattice/reflection domains are not supported for fixed spectra.
    SpectrumReflectionDomain,
    /// A fixed-reflection phase cannot regenerate lattice topology.
    FixedReflectionTopology,
    /// Stable site IDs must match the asymmetric-site count.
    SiteIdCountMismatch,
    /// Stable site IDs must be unique within a phase.
    DuplicateSiteId,
    /// One structural phase could not be prepared.
    StructuralPattern(StructuralPatternError),
    /// Native fixed-spectrum phase preparation failed.
    StructuralSpectrum(StructuralSpectrumError),
    /// Native multiphase structural calculation failed.
    StructuralMultiphase(StructuralMultiphaseError),
    /// Guarded reflection generation failed.
    Lattice(LatticeError),
    /// Stable-ID contribution transfer produced invalid arrays.
    Contributions(CwContributionsError),
    /// Built-in sample-physics evaluation failed.
    SamplePhysics(SamplePhysicsError),
    /// Residual evaluation failed.
    Residual(ResidualError),
    /// Analytical background evaluation failed.
    Background(BackgroundError),
    /// Calculation controls are invalid.
    InvalidOptions,
    /// Profile/background composition overflowed or became non-finite.
    NonFiniteCalculation,
    /// One internal phase result has incompatible sample dimensions.
    CalculationShapeMismatch,
}

impl Display for RietveldError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::MissingObservations => formatter.write_str("observed_y is required for Rietveld"),
            Self::InvalidInstrument => formatter.write_str("Rietveld instrument is invalid"),
            Self::InvalidAxialGeometry => formatter.write_str("Rietveld axial geometry is invalid"),
            Self::InvalidPositionCorrection => {
                formatter.write_str("Rietveld position correction is invalid")
            }
            Self::EmptyPhases => formatter.write_str("at least one Rietveld phase is required"),
            Self::InvalidPhaseName => formatter.write_str("Rietveld phase names must be non-empty"),
            Self::DuplicatePhaseId => formatter.write_str("Rietveld phase IDs must be unique"),
            Self::ContributionCountMismatch => formatter
                .write_str("sample-physics contributions must match the phase reflection count"),
            Self::ReflectionIdentityMismatch => {
                formatter.write_str("Rietveld reflection identities are invalid")
            }
            Self::ReflectionTopologyMismatch => formatter
                .write_str("Rietveld reflection topology does not match the current cell/domain"),
            Self::ReflectionWavelengthMismatch => formatter
                .write_str("Rietveld reflection domain wavelength differs from the instrument"),
            Self::CorrectionWavelengthMismatch => formatter
                .write_str("Rietveld intensity-correction wavelength differs from the instrument"),
            Self::SpectrumReferenceWavelengthMismatch => formatter.write_str(
                "fixed spectrum reference wavelength differs from the Rietveld instrument",
            ),
            Self::SpectrumReflectionDomain => formatter
                .write_str("fixed-spectrum Rietveld inputs cannot use dynamic reflection domains"),
            Self::FixedReflectionTopology => {
                formatter.write_str("fixed Rietveld phases cannot regenerate topology")
            }
            Self::SiteIdCountMismatch => {
                formatter.write_str("Rietveld site IDs must match the asymmetric-site count")
            }
            Self::DuplicateSiteId => {
                formatter.write_str("Rietveld site IDs must be unique within a phase")
            }
            Self::StructuralPattern(error) => Display::fmt(error, formatter),
            Self::StructuralSpectrum(error) => Display::fmt(error, formatter),
            Self::StructuralMultiphase(error) => Display::fmt(error, formatter),
            Self::Lattice(error) => Display::fmt(error, formatter),
            Self::Contributions(error) => Display::fmt(error, formatter),
            Self::SamplePhysics(error) => Display::fmt(error, formatter),
            Self::Residual(error) => Display::fmt(error, formatter),
            Self::Background(error) => Display::fmt(error, formatter),
            Self::InvalidOptions => formatter.write_str("Rietveld calculation options are invalid"),
            Self::NonFiniteCalculation => {
                formatter.write_str("Rietveld calculated pattern is non-finite")
            }
            Self::CalculationShapeMismatch => {
                formatter.write_str("Rietveld phase calculation shape mismatch")
            }
        }
    }
}

impl Error for RietveldError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pattern(error) => Some(error),
            Self::StructuralPattern(error) => Some(error),
            Self::StructuralSpectrum(error) => Some(error),
            Self::StructuralMultiphase(error) => Some(error),
            Self::Lattice(error) => Some(error),
            Self::Contributions(error) => Some(error),
            Self::SamplePhysics(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::Background(error) => Some(error),
            Self::MissingObservations
            | Self::InvalidInstrument
            | Self::InvalidAxialGeometry
            | Self::InvalidPositionCorrection
            | Self::EmptyPhases
            | Self::InvalidPhaseName
            | Self::DuplicatePhaseId
            | Self::ContributionCountMismatch
            | Self::ReflectionIdentityMismatch
            | Self::ReflectionTopologyMismatch
            | Self::ReflectionWavelengthMismatch
            | Self::CorrectionWavelengthMismatch
            | Self::SpectrumReferenceWavelengthMismatch
            | Self::SpectrumReflectionDomain
            | Self::FixedReflectionTopology
            | Self::SiteIdCountMismatch
            | Self::DuplicateSiteId
            | Self::InvalidOptions
            | Self::NonFiniteCalculation
            | Self::CalculationShapeMismatch => None,
        }
    }
}
