//! Native fixed-wavelength structural spectrum composition.

use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::{
    Accumulation, ConstantWavelengthInstrument, CwContributionsView, DenseJacobian, FcjGeometry,
    PatternDerivatives, SupportJacobian, SupportPolicy,
};
use phasesmith_crystallography::StructureFactorValues;
use phasesmith_execution::ExecutionPolicy;

use crate::structural_pattern::CW_INSTRUMENT_PARAMETER_COUNT;
use crate::{
    MonochromaticPositionCorrection, PreparedStructuralPatternInputView, PreparedStructuralPhase,
    StructuralPatternDenseResult, StructuralPatternError, StructuralPatternJvpResult,
    StructuralPatternResult, StructuralPatternVjpResult, StructuralPhaseDefinition,
};

const WAVELENGTH_GLOBAL_PARAMETER_INDEX: usize = CW_INSTRUMENT_PARAMETER_COUNT;

/// Borrowed dynamic inputs for one prepared fixed-wavelength spectrum.
#[derive(Clone, Copy, Debug)]
pub struct PreparedStructuralSpectrumInputView<'a> {
    /// Sorted pattern grid in degrees `2theta`.
    pub x_deg: &'a [f64],
    /// Reference instrument; each component replaces only its wavelength.
    pub instrument: ConstantWavelengthInstrument,
    /// Optional Finger--Cox--Jephcoat axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit zero/sample-displacement position correction.
    pub position_correction: MonochromaticPositionCorrection,
    /// One sample-physics contribution batch per wavelength component.
    pub contributions: &'a [CwContributionsView<'a>],
    /// Exact finite profile-support policy.
    pub support: SupportPolicy,
}

/// Invalid fixed-wavelength structural spectrum request.
#[derive(Debug)]
pub enum StructuralSpectrumError {
    /// Wavelength and relative-intensity arrays differ in length.
    ComponentLengthMismatch,
    /// No wavelength component was supplied.
    EmptyComponents,
    /// A wavelength is non-finite or non-positive.
    InvalidWavelength,
    /// A relative intensity is non-finite or negative, or the reference is zero.
    InvalidRelativeIntensity,
    /// Dynamic contribution batches do not match the prepared component count.
    ContributionCountMismatch,
    /// Component calculations do not have compatible result layouts.
    IncompatibleComponentResult,
    /// Component composition allocation arithmetic overflowed.
    AllocationOverflow,
    /// A monochromatic structural calculation failed.
    Structural(StructuralPatternError),
}

impl Display for StructuralSpectrumError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComponentLengthMismatch => formatter.write_str(
                "wavelength and relative-intensity arrays must have the same component count",
            ),
            Self::EmptyComponents => {
                formatter.write_str("at least one wavelength component is required")
            }
            Self::InvalidWavelength => {
                formatter.write_str("component wavelengths must be positive and finite")
            }
            Self::InvalidRelativeIntensity => formatter.write_str(
                "component relative intensities must be finite and non-negative, with a positive reference component",
            ),
            Self::ContributionCountMismatch => formatter.write_str(
                "sample-physics contributions must match the wavelength component count",
            ),
            Self::IncompatibleComponentResult => {
                formatter.write_str("wavelength component result layouts must match")
            }
            Self::AllocationOverflow => {
                formatter.write_str("wavelength component composition allocation overflow")
            }
            Self::Structural(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for StructuralSpectrumError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Structural(error) => Some(error),
            Self::ComponentLengthMismatch
            | Self::EmptyComponents
            | Self::InvalidWavelength
            | Self::InvalidRelativeIntensity
            | Self::ContributionCountMismatch
            | Self::IncompatibleComponentResult
            | Self::AllocationOverflow => None,
        }
    }
}

/// Reusable native fixed-wavelength spectrum for one structural phase.
pub struct PreparedStructuralSpectrum {
    phases: Vec<PreparedStructuralPhase>,
    wavelengths_angstrom: Vec<f64>,
    normalized_weights: Vec<f64>,
    execution: ExecutionPolicy,
}

impl PreparedStructuralSpectrum {
    /// Prepare every wavelength component from one base structural phase.
    ///
    /// Relative intensities are normalized to unit sum and multiply the base
    /// phase scale. All prepared components share the policy's persistent pool.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralSpectrumError`] for invalid component arrays or an
    /// invalid structural phase.
    pub fn new(
        definition: &StructuralPhaseDefinition,
        wavelengths_angstrom: Vec<f64>,
        relative_intensities: &[f64],
        execution: ExecutionPolicy,
    ) -> Result<Self, StructuralSpectrumError> {
        if wavelengths_angstrom.len() != relative_intensities.len() {
            return Err(StructuralSpectrumError::ComponentLengthMismatch);
        }
        if wavelengths_angstrom.is_empty() {
            return Err(StructuralSpectrumError::EmptyComponents);
        }
        if wavelengths_angstrom
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(StructuralSpectrumError::InvalidWavelength);
        }
        if relative_intensities[0] <= 0.0
            || relative_intensities
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralSpectrumError::InvalidRelativeIntensity);
        }
        let intensity_sum = relative_intensities.iter().sum::<f64>();
        if !intensity_sum.is_finite() || intensity_sum <= 0.0 {
            return Err(StructuralSpectrumError::InvalidRelativeIntensity);
        }
        let normalized_weights = relative_intensities
            .iter()
            .map(|value| value / intensity_sum)
            .collect::<Vec<_>>();
        let mut phases = Vec::with_capacity(wavelengths_angstrom.len());
        for &weight in &normalized_weights {
            let mut component = definition.clone();
            component.scale *= weight;
            phases.push(
                PreparedStructuralPhase::new(component, execution.context().clone())
                    .map_err(StructuralSpectrumError::Structural)?,
            );
        }
        Ok(Self {
            phases,
            wavelengths_angstrom,
            normalized_weights,
            execution,
        })
    }

    /// Return the number of prepared wavelength components.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.phases.len()
    }

    /// Return normalized component weights in input order.
    #[must_use]
    pub fn normalized_weights(&self) -> &[f64] {
        &self.normalized_weights
    }

    /// Calculate and combine all fixed wavelength components.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralSpectrumError`] for invalid dynamic inputs or a
    /// failed component calculation.
    pub fn calculate(
        &self,
        input: &PreparedStructuralSpectrumInputView<'_>,
    ) -> Result<StructuralPatternResult, StructuralSpectrumError> {
        let results = self.map_components(input, |phase, component_input| {
            phase.calculate(&component_input)
        })?;
        combine_pattern_results(results)
    }

    /// Calculate and combine dense structural linearizations.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralSpectrumError`] for invalid inputs or incompatible
    /// component layouts.
    pub fn linearize(
        &self,
        input: &PreparedStructuralSpectrumInputView<'_>,
    ) -> Result<StructuralPatternDenseResult, StructuralSpectrumError> {
        let products = self.map_components(input, |phase, component_input| {
            phase.linearize(&component_input)
        })?;
        combine_dense_products(products, &self.normalized_weights)
    }

    /// Calculate and combine structural forward derivative products.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralSpectrumError`] for invalid inputs or tangent shape.
    pub fn jvp(
        &self,
        input: &PreparedStructuralSpectrumInputView<'_>,
        tangent: &[f64],
    ) -> Result<StructuralPatternJvpResult, StructuralSpectrumError> {
        let products =
            self.map_components_indexed(input, |component, phase, component_input| {
                let mut component_tangent = tangent.to_vec();
                let scale_direction = component_tangent
                    .last_mut()
                    .ok_or(StructuralSpectrumError::IncompatibleComponentResult)?;
                *scale_direction *= self.normalized_weights[component];
                phase
                    .jvp(&component_input, &component_tangent)
                    .map_err(StructuralSpectrumError::Structural)
            })?;
        combine_jvp_products(products)
    }

    /// Calculate and combine structural transpose products.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralSpectrumError`] for invalid inputs or sample weights.
    pub fn vjp(
        &self,
        input: &PreparedStructuralSpectrumInputView<'_>,
        sample_weights: &[f64],
    ) -> Result<StructuralPatternVjpResult, StructuralSpectrumError> {
        let products = self.map_components(input, |phase, component_input| {
            phase.vjp(&component_input, sample_weights)
        })?;
        combine_vjp_products(products, &self.normalized_weights)
    }

    fn map_components<R: Send>(
        &self,
        input: &PreparedStructuralSpectrumInputView<'_>,
        operation: impl Fn(
            &PreparedStructuralPhase,
            PreparedStructuralPatternInputView<'_>,
        ) -> Result<R, StructuralPatternError>
        + Send
        + Sync,
    ) -> Result<Vec<R>, StructuralSpectrumError> {
        self.map_components_indexed(input, |_index, phase, component_input| {
            operation(phase, component_input).map_err(StructuralSpectrumError::Structural)
        })
    }

    fn map_components_indexed<R: Send>(
        &self,
        input: &PreparedStructuralSpectrumInputView<'_>,
        operation: impl Fn(
            usize,
            &PreparedStructuralPhase,
            PreparedStructuralPatternInputView<'_>,
        ) -> Result<R, StructuralSpectrumError>
        + Send
        + Sync,
    ) -> Result<Vec<R>, StructuralSpectrumError> {
        if input.contributions.len() != self.component_count() {
            return Err(StructuralSpectrumError::ContributionCountMismatch);
        }
        self.execution
            .context()
            .map_ordered(
                self.component_count(),
                self.execution.minimum_parallel_tasks(),
                |component| {
                    let mut instrument = input.instrument;
                    instrument.wavelength_angstrom = self.wavelengths_angstrom[component];
                    operation(
                        component,
                        &self.phases[component],
                        PreparedStructuralPatternInputView {
                            x_deg: input.x_deg,
                            instrument,
                            axial_geometry: input.axial_geometry,
                            position_correction: input.position_correction,
                            contributions: input.contributions[component],
                            support: input.support,
                        },
                    )
                },
            )
            .into_iter()
            .collect()
    }
}

fn combine_pattern_results(
    results: Vec<StructuralPatternResult>,
) -> Result<StructuralPatternResult, StructuralSpectrumError> {
    let mut iterator = results.into_iter();
    let first = iterator
        .next()
        .ok_or(StructuralSpectrumError::EmptyComponents)?;
    let mut components = vec![first];
    components.extend(iterator);
    let accumulation = combine_accumulations(
        components
            .iter_mut()
            .map(|result| std::mem::replace(&mut result.accumulation, empty_accumulation()))
            .collect(),
    )?;
    let mut structure_factors = StructureFactorValues {
        f_real: Vec::new(),
        f_imag: Vec::new(),
        f_squared: Vec::new(),
        intensity: Vec::new(),
        q_squared_inverse_angstrom2: Vec::new(),
        s_inverse_angstrom: Vec::new(),
    };
    let mut d_spacing_angstrom = Vec::new();
    let mut two_theta_deg = Vec::new();
    for result in components {
        structure_factors
            .f_real
            .extend(result.structure_factors.f_real);
        structure_factors
            .f_imag
            .extend(result.structure_factors.f_imag);
        structure_factors
            .f_squared
            .extend(result.structure_factors.f_squared);
        structure_factors
            .intensity
            .extend(result.structure_factors.intensity);
        structure_factors
            .q_squared_inverse_angstrom2
            .extend(result.structure_factors.q_squared_inverse_angstrom2);
        structure_factors
            .s_inverse_angstrom
            .extend(result.structure_factors.s_inverse_angstrom);
        d_spacing_angstrom.extend(result.d_spacing_angstrom);
        two_theta_deg.extend(result.two_theta_deg);
    }
    Ok(StructuralPatternResult {
        structure_factors,
        d_spacing_angstrom,
        two_theta_deg,
        accumulation,
    })
}

fn combine_accumulations(
    accumulations: Vec<Accumulation>,
) -> Result<Accumulation, StructuralSpectrumError> {
    let first = accumulations
        .first()
        .ok_or(StructuralSpectrumError::EmptyComponents)?;
    let sample_count = first.sample_count;
    let local_parameter_count = first.derivatives.local.parameter_count;
    let global_parameter_count = first
        .derivatives
        .global
        .as_ref()
        .ok_or(StructuralSpectrumError::IncompatibleComponentResult)?
        .parameter_count;
    if global_parameter_count <= WAVELENGTH_GLOBAL_PARAMETER_INDEX {
        return Err(StructuralSpectrumError::IncompatibleComponentResult);
    }
    let combined_global_count = global_parameter_count - 1;
    let global_length = combined_global_count
        .checked_mul(sample_count)
        .ok_or(StructuralSpectrumError::AllocationOverflow)?;
    let mut y = vec![0.0; sample_count];
    let mut starts = Vec::new();
    let mut offsets: Vec<usize> = vec![0];
    let mut local_values = Vec::new();
    let mut global_values = vec![0.0; global_length];
    for accumulation in accumulations {
        let global = accumulation
            .derivatives
            .global
            .ok_or(StructuralSpectrumError::IncompatibleComponentResult)?;
        if accumulation.sample_count != sample_count
            || accumulation.y.len() != sample_count
            || accumulation.derivatives.local.parameter_count != local_parameter_count
            || global.parameter_count != global_parameter_count
            || global.sample_count != sample_count
        {
            return Err(StructuralSpectrumError::IncompatibleComponentResult);
        }
        for (combined, value) in y.iter_mut().zip(accumulation.y) {
            *combined += value;
        }
        let cursor = offsets
            .last()
            .copied()
            .ok_or(StructuralSpectrumError::IncompatibleComponentResult)?;
        starts.extend(accumulation.derivatives.local.starts);
        for offset in accumulation.derivatives.local.offsets.into_iter().skip(1) {
            offsets.push(
                cursor
                    .checked_add(offset)
                    .ok_or(StructuralSpectrumError::AllocationOverflow)?,
            );
        }
        local_values.extend(accumulation.derivatives.local.values);
        let mut combined_row = 0;
        for row in 0..global_parameter_count {
            if row == WAVELENGTH_GLOBAL_PARAMETER_INDEX {
                continue;
            }
            let source = row * sample_count;
            let target = combined_row * sample_count;
            for sample in 0..sample_count {
                global_values[target + sample] += global.values[source + sample];
            }
            combined_row += 1;
        }
    }
    Ok(Accumulation {
        y,
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts,
                offsets,
                values: local_values,
                parameter_count: local_parameter_count,
            },
            global: Some(DenseJacobian {
                values: global_values,
                parameter_count: combined_global_count,
                sample_count,
            }),
        },
        sample_count,
    })
}

fn combine_dense_products(
    products: Vec<StructuralPatternDenseResult>,
    weights: &[f64],
) -> Result<StructuralPatternDenseResult, StructuralSpectrumError> {
    let first = products
        .first()
        .ok_or(StructuralSpectrumError::EmptyComponents)?;
    let parameter_count = first.parameter_count;
    let sample_count = first.result.accumulation.sample_count;
    if parameter_count == 0 || products.len() != weights.len() {
        return Err(StructuralSpectrumError::IncompatibleComponentResult);
    }
    let element_count = parameter_count
        .checked_mul(sample_count)
        .ok_or(StructuralSpectrumError::AllocationOverflow)?;
    let mut d_y = vec![0.0; element_count];
    let mut results = Vec::with_capacity(products.len());
    for (product, &weight) in products.into_iter().zip(weights) {
        if product.parameter_count != parameter_count || product.d_y.len() != element_count {
            return Err(StructuralSpectrumError::IncompatibleComponentResult);
        }
        for parameter in 0..parameter_count {
            let factor = if parameter + 1 == parameter_count {
                weight
            } else {
                1.0
            };
            let row = parameter * sample_count;
            for sample in 0..sample_count {
                d_y[row + sample] += factor * product.d_y[row + sample];
            }
        }
        results.push(product.result);
    }
    Ok(StructuralPatternDenseResult {
        result: combine_pattern_results(results)?,
        d_y,
        parameter_count,
    })
}

fn combine_jvp_products(
    products: Vec<StructuralPatternJvpResult>,
) -> Result<StructuralPatternJvpResult, StructuralSpectrumError> {
    let first = products
        .first()
        .ok_or(StructuralSpectrumError::EmptyComponents)?;
    let sample_count = first.d_y.len();
    let mut d_y = vec![0.0; sample_count];
    let mut d_integrated_intensity = Vec::new();
    let mut d_two_theta_deg = Vec::new();
    let mut results = Vec::with_capacity(products.len());
    for product in products {
        if product.d_y.len() != sample_count {
            return Err(StructuralSpectrumError::IncompatibleComponentResult);
        }
        for (combined, value) in d_y.iter_mut().zip(product.d_y) {
            *combined += value;
        }
        d_integrated_intensity.extend(product.d_integrated_intensity);
        d_two_theta_deg.extend(product.d_two_theta_deg);
        results.push(product.result);
    }
    Ok(StructuralPatternJvpResult {
        result: combine_pattern_results(results)?,
        d_y,
        d_integrated_intensity,
        d_two_theta_deg,
    })
}

fn combine_vjp_products(
    products: Vec<StructuralPatternVjpResult>,
    weights: &[f64],
) -> Result<StructuralPatternVjpResult, StructuralSpectrumError> {
    let first = products
        .first()
        .ok_or(StructuralSpectrumError::EmptyComponents)?;
    let parameter_count = first.gradient.len();
    if parameter_count == 0 || products.len() != weights.len() {
        return Err(StructuralSpectrumError::IncompatibleComponentResult);
    }
    let mut gradient = vec![0.0; parameter_count];
    let mut results = Vec::with_capacity(products.len());
    for (product, &weight) in products.into_iter().zip(weights) {
        if product.gradient.len() != parameter_count {
            return Err(StructuralSpectrumError::IncompatibleComponentResult);
        }
        for (parameter, value) in product.gradient.into_iter().enumerate() {
            let factor = if parameter + 1 == parameter_count {
                weight
            } else {
                1.0
            };
            gradient[parameter] += factor * value;
        }
        results.push(product.result);
    }
    Ok(StructuralPatternVjpResult {
        result: combine_pattern_results(results)?,
        gradient,
    })
}

fn empty_accumulation() -> Accumulation {
    Accumulation {
        y: Vec::new(),
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts: Vec::new(),
                offsets: vec![0],
                values: Vec::new(),
                parameter_count: 0,
            },
            global: None,
        },
        sample_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_core::CwContributionArrays;
    use phasesmith_crystallography::{
        IntegratedIntensityCorrectionModel, Rational, SpaceGroup, SymmetryOperation, UnitCell,
    };
    fn definition() -> StructuralPhaseDefinition {
        StructuralPhaseDefinition {
            cell: UnitCell {
                a_angstrom: 4.7,
                b_angstrom: 5.1,
                c_angstrom: 6.2,
                alpha_deg: 82.0,
                beta_deg: 87.0,
                gamma_deg: 74.0,
            },
            space_group: SpaceGroup::new(vec![
                SymmetryOperation::identity(),
                SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
                    .expect("inversion"),
            ])
            .expect("P-1"),
            hkl: vec![[1, 0, 1], [2, 1, 1], [1, 2, 3]],
            multiplicity: vec![2, 4, 2],
            fractional_xyz: vec![[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]],
            occupancy: vec![0.82, 0.55],
            u_iso_angstrom2: vec![0.012, 0.018],
            anisotropic_mask: vec![false, false],
            u_aniso_cif_angstrom2: vec![[0.0; 6]; 2],
            scattering_species: vec!["Si".to_owned(), "O".to_owned()],
            scattering_real_offset: Vec::new(),
            scattering_imag_offset: Vec::new(),
            scale: 1.4,
            coordinate_tolerance: 1.0e-10,
            scattering_model: crate::BuiltInScatteringModel::XrayNonResonant,
            correction_model: IntegratedIntensityCorrectionModel::Neutral,
        }
    }

    fn instrument() -> ConstantWavelengthInstrument {
        ConstantWavelengthInstrument {
            wavelength_angstrom: 1.5406,
            u_deg2: 2.0e-4,
            v_deg2: -1.0e-4,
            w_deg2: 1.2e-4,
            x_deg: 1.5e-3,
            y_deg: 3.0e-3,
        }
    }

    fn contribution<'a>(zeros: &'a [f64], ones: &'a [f64]) -> CwContributionsView<'a> {
        CwContributionsView::new(
            zeros.len(),
            0,
            CwContributionArrays {
                gaussian_variance_deg2: zeros,
                lorentzian_fwhm_deg: zeros,
                intensity_multiplier: ones,
                d_gaussian_variance_d_position: zeros,
                d_lorentzian_fwhm_d_position: zeros,
                d_intensity_multiplier_d_position: zeros,
                d_gaussian_variance_d_parameters: &[],
                d_lorentzian_fwhm_d_parameters: &[],
                d_intensity_multiplier_d_parameters: &[],
            },
        )
        .expect("contribution")
    }

    #[test]
    fn spectrum_values_and_derivatives_follow_fixed_component_chain_rules() {
        let policy = ExecutionPolicy::new(Some(2), 2).expect("policy");
        let spectrum = PreparedStructuralSpectrum::new(
            &definition(),
            vec![1.5406, 1.54439],
            &[1.0, 0.5],
            policy,
        )
        .expect("spectrum");
        assert_eq!(spectrum.component_count(), 2);
        assert_eq!(spectrum.normalized_weights(), [2.0 / 3.0, 1.0 / 3.0]);

        let x = (0..9_001)
            .map(|index| 10.0 + f64::from(index) * 0.01)
            .collect::<Vec<_>>();
        let zeros = [0.0; 3];
        let ones = [1.0; 3];
        let contributions = [contribution(&zeros, &ones), contribution(&zeros, &ones)];
        let input = PreparedStructuralSpectrumInputView {
            x_deg: &x,
            instrument: instrument(),
            axial_geometry: None,
            position_correction: MonochromaticPositionCorrection {
                zero_shift_deg: 0.0,
                bragg_brentano_mm: None,
                debye_scherrer_micrometre: None,
            },
            contributions: &contributions,
            support: SupportPolicy::FwhmMultiple(20.0),
        };
        let values = spectrum.calculate(&input).expect("values");
        assert_eq!(values.structure_factors.intensity.len(), 6);
        assert_eq!(values.d_spacing_angstrom.len(), 6);
        assert_eq!(values.accumulation.derivatives.local.peak_count(), 6);
        assert_eq!(
            values
                .accumulation
                .derivatives
                .global
                .as_ref()
                .expect("global")
                .parameter_count,
            6
        );

        let parameter_count = spectrum.phases[0].structural_parameter_count();
        let tangent = (0..parameter_count)
            .map(|index| f64::from(u32::try_from(index + 1).expect("small index")) * 2.0e-5)
            .collect::<Vec<_>>();
        let dense = spectrum.linearize(&input).expect("dense");
        let jvp = spectrum.jvp(&input, &tangent).expect("JVP");
        for sample in 0..x.len() {
            let product = tangent
                .iter()
                .enumerate()
                .map(|(parameter, direction)| direction * dense.d_y[parameter * x.len() + sample])
                .sum::<f64>();
            assert!((product - jvp.d_y[sample]).abs() < 3.0e-11 * product.abs().max(1.0));
        }
        let sample_weights = x
            .iter()
            .map(|value| (0.03 * value).sin())
            .collect::<Vec<_>>();
        let vjp = spectrum.vjp(&input, &sample_weights).expect("VJP");
        for (parameter, gradient) in vjp.gradient.iter().enumerate() {
            let expected = dense.d_y[parameter * x.len()..(parameter + 1) * x.len()]
                .iter()
                .zip(&sample_weights)
                .map(|(derivative, weight)| derivative * weight)
                .sum::<f64>();
            assert!((expected - gradient).abs() < 3.0e-10 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn spectrum_validates_component_and_contribution_counts() {
        let definition = definition();
        assert!(matches!(
            PreparedStructuralSpectrum::new(
                &definition,
                vec![1.0],
                &[],
                ExecutionPolicy::bounded_default().expect("policy"),
            ),
            Err(StructuralSpectrumError::ComponentLengthMismatch)
        ));
        let spectrum = PreparedStructuralSpectrum::new(
            &definition,
            vec![1.0],
            &[1.0],
            ExecutionPolicy::bounded_default().expect("policy"),
        )
        .expect("spectrum");
        let input = PreparedStructuralSpectrumInputView {
            x_deg: &[10.0, 11.0],
            instrument: instrument(),
            axial_geometry: None,
            position_correction: MonochromaticPositionCorrection {
                zero_shift_deg: 0.0,
                bragg_brentano_mm: None,
                debye_scherrer_micrometre: None,
            },
            contributions: &[],
            support: SupportPolicy::FwhmMultiple(20.0),
        };
        assert!(matches!(
            spectrum.calculate(&input),
            Err(StructuralSpectrumError::ContributionCountMismatch)
        ));
    }
}
