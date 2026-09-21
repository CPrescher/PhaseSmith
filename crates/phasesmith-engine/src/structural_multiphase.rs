//! Native scheduling and ordered composition for multiple structural phases.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::structural_pattern::StructuralPreparationCache;

use phasesmith_core::{
    ConstantWavelengthInstrument, CwContributionsView, FcjGeometry, OwnedCwContributions,
    SupportPolicy,
};
use phasesmith_execution::ExecutionPolicy;

use crate::{
    MonochromaticPositionCorrection, PreparedStructuralPatternInputView, PreparedStructuralPhase,
    PreparedStructuralSpectrum, PreparedStructuralSpectrumInputView, StructuralPatternDenseResult,
    StructuralPatternError, StructuralPatternJvpResult, StructuralPatternResult,
    StructuralPatternVjpResult, StructuralSpectrumError,
};

/// One native built-in structural model, monochromatic or fixed-spectrum.
#[derive(Clone)]
pub enum PreparedStructuralModel {
    /// One monochromatic structural phase.
    Monochromatic(Arc<PreparedStructuralPhase>),
    /// One fixed-wavelength structural spectrum.
    FixedSpectrum(Arc<PreparedStructuralSpectrum>),
}

impl PreparedStructuralModel {
    /// Wrap one prepared monochromatic phase.
    #[must_use]
    pub fn monochromatic(phase: PreparedStructuralPhase) -> Self {
        Self::Monochromatic(Arc::new(phase))
    }

    /// Wrap one prepared fixed-wavelength spectrum.
    #[must_use]
    pub fn fixed_spectrum(spectrum: PreparedStructuralSpectrum) -> Self {
        Self::FixedSpectrum(Arc::new(spectrum))
    }

    /// Return the structural parameter count for this phase model.
    #[must_use]
    pub fn structural_parameter_count(&self) -> usize {
        match self {
            Self::Monochromatic(phase) => phase.structural_parameter_count(),
            Self::FixedSpectrum(spectrum) => spectrum.structural_parameter_count(),
        }
    }

    fn calculate(
        &self,
        input: &PreparedStructuralModelInputView<'_>,
    ) -> Result<StructuralPatternResult, StructuralMultiphaseError> {
        match self {
            Self::Monochromatic(phase) => phase
                .calculate(&input.monochromatic_input()?)
                .map_err(StructuralMultiphaseError::Structural),
            Self::FixedSpectrum(spectrum) => spectrum
                .calculate(&input.spectrum_input())
                .map_err(StructuralMultiphaseError::Spectrum),
        }
    }

    fn linearize(
        &self,
        input: &PreparedStructuralModelInputView<'_>,
    ) -> Result<StructuralPatternDenseResult, StructuralMultiphaseError> {
        match self {
            Self::Monochromatic(phase) => phase
                .linearize(&input.monochromatic_input()?)
                .map_err(StructuralMultiphaseError::Structural),
            Self::FixedSpectrum(spectrum) => spectrum
                .linearize(&input.spectrum_input())
                .map_err(StructuralMultiphaseError::Spectrum),
        }
    }

    fn linearize_selected(
        &self,
        input: &PreparedStructuralModelInputView<'_>,
        selected: &[bool],
        cache: Option<&StructuralPreparationCache>,
    ) -> Result<StructuralPatternDenseResult, StructuralMultiphaseError> {
        match self {
            Self::Monochromatic(phase) => phase
                .linearize_selected(&input.monochromatic_input()?, selected, cache)
                .map_err(StructuralMultiphaseError::Structural),
            Self::FixedSpectrum(spectrum) => spectrum
                .linearize_selected(&input.spectrum_input(), selected, cache)
                .map_err(StructuralMultiphaseError::Spectrum),
        }
    }

    fn jvp(
        &self,
        input: &PreparedStructuralModelInputView<'_>,
        tangent: &[f64],
    ) -> Result<StructuralPatternJvpResult, StructuralMultiphaseError> {
        match self {
            Self::Monochromatic(phase) => phase
                .jvp(&input.monochromatic_input()?, tangent)
                .map_err(StructuralMultiphaseError::Structural),
            Self::FixedSpectrum(spectrum) => spectrum
                .jvp(&input.spectrum_input(), tangent)
                .map_err(StructuralMultiphaseError::Spectrum),
        }
    }

    fn vjp(
        &self,
        input: &PreparedStructuralModelInputView<'_>,
        sample_weights: &[f64],
    ) -> Result<StructuralPatternVjpResult, StructuralMultiphaseError> {
        match self {
            Self::Monochromatic(phase) => phase
                .vjp(&input.monochromatic_input()?, sample_weights)
                .map_err(StructuralMultiphaseError::Structural),
            Self::FixedSpectrum(spectrum) => spectrum
                .vjp(&input.spectrum_input(), sample_weights)
                .map_err(StructuralMultiphaseError::Spectrum),
        }
    }
}

/// Borrowed dynamic data for one phase model in a multiphase operation.
#[derive(Clone, Copy, Debug)]
pub struct PreparedStructuralModelInputView<'a> {
    /// Sorted pattern grid in degrees `2theta`.
    pub x_deg: &'a [f64],
    /// Reference constant-wavelength instrument.
    pub instrument: ConstantWavelengthInstrument,
    /// Optional axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit position correction.
    pub position_correction: MonochromaticPositionCorrection,
    /// One contribution batch for monochromatic models, or one per component.
    pub contributions: &'a [CwContributionsView<'a>],
    /// Exact finite profile-support policy.
    pub support: SupportPolicy,
    /// Explicit CW profile accuracy controls.
    pub profile_accuracy: phasesmith_core::ProfileAccuracy,
    /// Whether calculation includes axial derivative rows; omitted rows are zero.
    pub calculate_axial_derivatives: bool,
}

/// Owned sample-physics inputs for one prepared structural model.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralModelInput {
    /// One contribution batch for monochromatic models, or one per spectrum component.
    pub contributions: Vec<OwnedCwContributions>,
}

/// Owned application-boundary request for one multiphase structural calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralCalculationRequest {
    /// Sorted pattern grid in degrees `2theta`.
    pub x_deg: Vec<f64>,
    /// Reference constant-wavelength instrument.
    pub instrument: ConstantWavelengthInstrument,
    /// Optional axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit position correction.
    pub position_correction: MonochromaticPositionCorrection,
    /// Dynamic inputs in prepared-model order.
    pub phase_inputs: Vec<StructuralModelInput>,
    /// Exact finite profile-support policy.
    pub support: SupportPolicy,
    /// Explicit CW profile accuracy controls.
    pub profile_accuracy: phasesmith_core::ProfileAccuracy,
    /// Whether calculation includes axial derivative rows; omitted rows are zero.
    pub calculate_axial_derivatives: bool,
}

impl<'a> PreparedStructuralModelInputView<'a> {
    fn monochromatic_input(
        &self,
    ) -> Result<PreparedStructuralPatternInputView<'a>, StructuralMultiphaseError> {
        if self.contributions.len() != 1 {
            return Err(StructuralMultiphaseError::ContributionCountMismatch);
        }
        Ok(PreparedStructuralPatternInputView {
            x_deg: self.x_deg,
            instrument: self.instrument,
            axial_geometry: self.axial_geometry,
            position_correction: self.position_correction,
            contributions: self.contributions[0],
            support: self.support,
            profile_accuracy: self.profile_accuracy,
            calculate_axial_derivatives: self.calculate_axial_derivatives,
        })
    }

    fn spectrum_input(&self) -> PreparedStructuralSpectrumInputView<'a> {
        PreparedStructuralSpectrumInputView {
            x_deg: self.x_deg,
            instrument: self.instrument,
            axial_geometry: self.axial_geometry,
            position_correction: self.position_correction,
            contributions: self.contributions,
            support: self.support,
            profile_accuracy: self.profile_accuracy,
            calculate_axial_derivatives: self.calculate_axial_derivatives,
        }
    }
}

/// Ordered multiphase structural values and their combined profile.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralMultiphaseResult {
    /// Sum of phase profile arrays in phase input order.
    pub profile_y: Vec<f64>,
    /// One complete diagnostic result per phase, in input order.
    pub phases: Vec<StructuralPatternResult>,
}

/// Display-ready owned result from a structural calculation request.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralCalculationResult {
    /// Pattern grid copied from the evaluated request.
    pub x_deg: Vec<f64>,
    /// Sum of all phase profiles in prepared-model order.
    pub profile_y: Vec<f64>,
    /// Complete diagnostic result for each phase in prepared-model order.
    pub phases: Vec<StructuralPatternResult>,
}

/// Invalid native multiphase request.
#[derive(Debug)]
pub enum StructuralMultiphaseError {
    /// No structural model was supplied.
    EmptyPhases,
    /// Dynamic phase inputs do not match the prepared phase count.
    PhaseInputCountMismatch,
    /// Tangent vectors do not match the prepared phase count.
    TangentCountMismatch,
    /// A monochromatic model did not receive exactly one contribution batch.
    ContributionCountMismatch,
    /// Phase profile sample counts differ.
    SampleCountMismatch,
    /// A monochromatic phase failed.
    Structural(StructuralPatternError),
    /// A fixed-spectrum phase failed.
    Spectrum(StructuralSpectrumError),
}

impl Display for StructuralMultiphaseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPhases => formatter.write_str("at least one structural phase is required"),
            Self::PhaseInputCountMismatch => {
                formatter.write_str("dynamic phase inputs must match the prepared phase count")
            }
            Self::TangentCountMismatch => {
                formatter.write_str("structural tangents must match the prepared phase count")
            }
            Self::ContributionCountMismatch => {
                formatter.write_str("a monochromatic phase requires exactly one contribution batch")
            }
            Self::SampleCountMismatch => {
                formatter.write_str("structural phases must share one sample grid")
            }
            Self::Structural(error) => Display::fmt(error, formatter),
            Self::Spectrum(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for StructuralMultiphaseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Structural(error) => Some(error),
            Self::Spectrum(error) => Some(error),
            Self::EmptyPhases
            | Self::PhaseInputCountMismatch
            | Self::TangentCountMismatch
            | Self::ContributionCountMismatch
            | Self::SampleCountMismatch => None,
        }
    }
}

/// Reusable native scheduler for ordered structural phase models.
pub struct PreparedStructuralMultiphase {
    models: Vec<PreparedStructuralModel>,
    execution: ExecutionPolicy,
}

impl PreparedStructuralMultiphase {
    /// Prepare a non-empty ordered phase collection.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralMultiphaseError::EmptyPhases`] for an empty model list.
    pub fn new(
        models: Vec<PreparedStructuralModel>,
        execution: ExecutionPolicy,
    ) -> Result<Self, StructuralMultiphaseError> {
        if models.is_empty() {
            return Err(StructuralMultiphaseError::EmptyPhases);
        }
        Ok(Self { models, execution })
    }

    /// Return the number of prepared phases.
    #[must_use]
    pub fn phase_count(&self) -> usize {
        self.models.len()
    }

    /// Return structural parameter counts in phase order.
    #[must_use]
    pub fn structural_parameter_counts(&self) -> Vec<usize> {
        self.models
            .iter()
            .map(PreparedStructuralModel::structural_parameter_count)
            .collect()
    }

    /// Calculate all phases and their ordered sum.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralMultiphaseError`] for invalid inputs or phase failures.
    pub fn calculate(
        &self,
        inputs: &[PreparedStructuralModelInputView<'_>],
    ) -> Result<StructuralMultiphaseResult, StructuralMultiphaseError> {
        let phases = self.map_models(inputs, PreparedStructuralModel::calculate)?;
        let profile_y = sum_phase_profiles(phases.iter().map(|phase| &phase.accumulation.y))?;
        Ok(StructuralMultiphaseResult { profile_y, phases })
    }

    /// Calculate from an entirely owned application-boundary request.
    ///
    /// This is the shared entry point for language and application adapters. The
    /// request owns all dynamic arrays; kernel views exist only for the duration
    /// of this call.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralMultiphaseError`] for invalid inputs or phase failures.
    pub fn calculate_request(
        &self,
        request: StructuralCalculationRequest,
    ) -> Result<StructuralCalculationResult, StructuralMultiphaseError> {
        let contribution_views = request
            .phase_inputs
            .iter()
            .map(|model| {
                model
                    .contributions
                    .iter()
                    .map(OwnedCwContributions::as_view)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let inputs = contribution_views
            .iter()
            .map(|contributions| PreparedStructuralModelInputView {
                x_deg: &request.x_deg,
                instrument: request.instrument,
                axial_geometry: request.axial_geometry,
                position_correction: request.position_correction,
                contributions,
                support: request.support,
                profile_accuracy: request.profile_accuracy,
                calculate_axial_derivatives: request.calculate_axial_derivatives,
            })
            .collect::<Vec<_>>();
        let result = self.calculate(&inputs)?;
        Ok(StructuralCalculationResult {
            x_deg: request.x_deg,
            profile_y: result.profile_y,
            phases: result.phases,
        })
    }

    /// Calculate one dense structural linearization per phase.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralMultiphaseError`] for invalid inputs or phase failures.
    pub fn linearize(
        &self,
        inputs: &[PreparedStructuralModelInputView<'_>],
    ) -> Result<Vec<StructuralPatternDenseResult>, StructuralMultiphaseError> {
        self.map_models(inputs, PreparedStructuralModel::linearize)
    }

    /// Linearize only the selected native rows of each phase.
    /// # Errors
    /// Returns an error for incompatible phase masks or invalid inputs.
    pub fn linearize_selected(
        &self,
        inputs: &[PreparedStructuralModelInputView<'_>],
        selected: &[Vec<bool>],
        cache: Option<&StructuralPreparationCache>,
    ) -> Result<Vec<StructuralPatternDenseResult>, StructuralMultiphaseError> {
        if selected.len() != self.phase_count() {
            return Err(StructuralMultiphaseError::TangentCountMismatch);
        }
        self.map_models_indexed(inputs, |index, model, input| {
            model.linearize_selected(input, &selected[index], cache)
        })
    }

    /// Calculate one structural forward product per phase.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralMultiphaseError`] for invalid inputs or tangents.
    pub fn jvp(
        &self,
        inputs: &[PreparedStructuralModelInputView<'_>],
        tangents: &[&[f64]],
    ) -> Result<Vec<StructuralPatternJvpResult>, StructuralMultiphaseError> {
        if tangents.len() != self.phase_count() {
            return Err(StructuralMultiphaseError::TangentCountMismatch);
        }
        self.map_models_indexed(inputs, |index, model, input| {
            model.jvp(input, tangents[index])
        })
    }

    /// Calculate one structural reverse product per phase.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralMultiphaseError`] for invalid inputs or sample weights.
    pub fn vjp(
        &self,
        inputs: &[PreparedStructuralModelInputView<'_>],
        sample_weights: &[f64],
    ) -> Result<Vec<StructuralPatternVjpResult>, StructuralMultiphaseError> {
        self.map_models(inputs, |model, input| model.vjp(input, sample_weights))
    }

    fn map_models<R: Send>(
        &self,
        inputs: &[PreparedStructuralModelInputView<'_>],
        operation: impl Fn(
            &PreparedStructuralModel,
            &PreparedStructuralModelInputView<'_>,
        ) -> Result<R, StructuralMultiphaseError>
        + Send
        + Sync,
    ) -> Result<Vec<R>, StructuralMultiphaseError> {
        self.map_models_indexed(inputs, |_index, model, input| operation(model, input))
    }

    fn map_models_indexed<R: Send>(
        &self,
        inputs: &[PreparedStructuralModelInputView<'_>],
        operation: impl Fn(
            usize,
            &PreparedStructuralModel,
            &PreparedStructuralModelInputView<'_>,
        ) -> Result<R, StructuralMultiphaseError>
        + Send
        + Sync,
    ) -> Result<Vec<R>, StructuralMultiphaseError> {
        if inputs.len() != self.phase_count() {
            return Err(StructuralMultiphaseError::PhaseInputCountMismatch);
        }
        self.execution
            .context()
            .map_ordered(
                self.phase_count(),
                self.execution.minimum_parallel_tasks(),
                |index| operation(index, &self.models[index], &inputs[index]),
            )
            .into_iter()
            .collect()
    }
}

fn sum_phase_profiles<'a>(
    profiles: impl IntoIterator<Item = &'a Vec<f64>>,
) -> Result<Vec<f64>, StructuralMultiphaseError> {
    let mut iterator = profiles.into_iter();
    let first = iterator
        .next()
        .ok_or(StructuralMultiphaseError::EmptyPhases)?;
    let mut combined = vec![0.0; first.len()];
    for profile in std::iter::once(first).chain(iterator) {
        if profile.len() != combined.len() {
            return Err(StructuralMultiphaseError::SampleCountMismatch);
        }
        for (total, value) in combined.iter_mut().zip(profile) {
            *total += value;
        }
    }
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuiltInScatteringModel, StructuralPhaseDefinition};
    use phasesmith_crystallography::{
        IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
    };

    fn prepared_model() -> PreparedStructuralModel {
        let definition = StructuralPhaseDefinition {
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
        };
        PreparedStructuralModel::monochromatic(
            PreparedStructuralPhase::new(
                definition,
                phasesmith_execution::ExecutionContext::serial(),
            )
            .expect("prepared phase"),
        )
    }

    #[test]
    fn multiphase_requires_at_least_one_model() {
        assert!(matches!(
            PreparedStructuralMultiphase::new(
                Vec::new(),
                ExecutionPolicy::bounded_default().expect("policy"),
            ),
            Err(StructuralMultiphaseError::EmptyPhases)
        ));
    }

    #[test]
    fn profile_sum_preserves_phase_order_and_validates_sample_counts() {
        let first = vec![1.0, 2.0, 3.0];
        let second = vec![0.5, 0.25, 0.125];
        assert_eq!(
            sum_phase_profiles([&first, &second]).expect("sum"),
            [1.5, 2.25, 3.125]
        );
        assert!(matches!(
            sum_phase_profiles([&first, &vec![1.0]]),
            Err(StructuralMultiphaseError::SampleCountMismatch)
        ));
    }

    #[test]
    fn owned_request_validates_phase_input_count() {
        let prepared = PreparedStructuralMultiphase::new(
            vec![prepared_model()],
            ExecutionPolicy::bounded_default().expect("policy"),
        )
        .expect("multiphase");
        let request = StructuralCalculationRequest {
            x_deg: vec![20.0, 21.0],
            instrument: ConstantWavelengthInstrument {
                wavelength_angstrom: 1.5406,
                u_deg2: 0.0,
                v_deg2: 0.0,
                w_deg2: 0.01,
                x_deg: 0.0,
                y_deg: 0.0,
            },
            axial_geometry: None,
            position_correction: MonochromaticPositionCorrection {
                zero_shift_deg: 0.0,
                bragg_brentano_mm: None,
                debye_scherrer_micrometre: None,
            },
            phase_inputs: Vec::new(),
            support: SupportPolicy::FwhmMultiple(8.0),
            profile_accuracy: phasesmith_core::ProfileAccuracy::default(),
            calculate_axial_derivatives: true,
        };
        assert!(matches!(
            prepared.calculate_request(request),
            Err(StructuralMultiphaseError::PhaseInputCountMismatch)
        ));
    }
}
