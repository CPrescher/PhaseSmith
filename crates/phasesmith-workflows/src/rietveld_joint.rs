//! Joint multi-histogram Rietveld parameter and matrix-free objective contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_model::RecordId;

use crate::{
    LatticeBounds, ParameterError, ParameterKey, ParameterSet, ParameterSpec,
    PreparedGeneralRietveldObjective, RietveldCalculationOptions, RietveldGeneralObjectiveError,
    RietveldGeneralParameterError, RietveldInput, RietveldParameterLayout,
    RietveldParameterSelection,
};

/// One experiment in a joint native Rietveld objective.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRietveldHistogram {
    /// Stable experiment identity used to namespace local parameters.
    pub histogram_id: RecordId,
    /// Complete observed pattern, experiment, and phase state.
    pub input: RietveldInput,
    /// Selected complete parameter families for this histogram.
    pub selection: RietveldParameterSelection,
    /// Per-phase lattice bounds in the input phase order.
    pub lattice_bounds: Vec<Option<LatticeBounds>>,
    /// Native calculation controls for this histogram.
    pub calculation: RietveldCalculationOptions,
}

/// One histogram's profile and directional derivative.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRietveldProduct {
    /// Stable histogram identity.
    pub histogram_id: RecordId,
    /// Accepted-state calculated profile.
    pub profile: Vec<f64>,
    /// Profile directional derivative in joint physical coordinates.
    pub derivative: Vec<f64>,
}

/// Complete accepted-state value and gradient of a joint objective.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRietveldGradient {
    /// Calculated profiles in histogram order.
    pub calculated: Vec<Vec<f64>>,
    /// Half the summed weighted squared residual over every histogram.
    pub objective: f64,
    /// Gradient in stable joint physical-parameter order.
    pub gradient: Vec<f64>,
}

/// Stable shared/local parameter packing for a joint objective.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRietveldLayout {
    parameters: ParameterSet,
    histogram_ids: Vec<RecordId>,
    local_layouts: Vec<RietveldParameterLayout>,
    local_to_joint: Vec<Vec<usize>>,
}

impl JointRietveldLayout {
    /// Build one physical parameter set across two or more histograms.
    ///
    /// Lattice, coordinates, occupancies, and atomic displacement parameters
    /// are shared by stable phase/site identity. Instrument, background,
    /// sample-physics, and phase-scale parameters are histogram-local.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldError`] for invalid histogram identity, input,
    /// selection, or incompatible shared structural state.
    pub fn new(histograms: &[JointRietveldHistogram]) -> Result<Self, JointRietveldError> {
        validate_histograms(histograms)?;
        let local_layouts = histograms
            .iter()
            .map(|histogram| {
                RietveldParameterLayout::new(
                    &histogram.input,
                    &histogram.selection,
                    &histogram.lattice_bounds,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut specs = Vec::new();
        let mut shared = BTreeMap::<ParameterKey, usize>::new();
        let mut local_to_joint = Vec::with_capacity(histograms.len());
        for (histogram, layout) in histograms.iter().zip(&local_layouts) {
            let mut mapping = Vec::with_capacity(layout.parameters().specs().len());
            for spec in layout.parameters().specs() {
                let joint_index = if is_shared(spec.key()) {
                    if let Some(index) = shared.get(spec.key()).copied() {
                        if specs[index] != *spec {
                            return Err(JointRietveldError::SharedParameterMismatch {
                                key: spec.key().clone(),
                            });
                        }
                        index
                    } else {
                        let index = specs.len();
                        specs.push(spec.clone());
                        shared.insert(spec.key().clone(), index);
                        index
                    }
                } else {
                    let key = ParameterKey::new(
                        spec.key().module(),
                        format!("{}/{}", histogram.histogram_id, spec.key().owner_id()),
                        spec.key().name(),
                    )?;
                    let index = specs.len();
                    specs.push(ParameterSpec::new(
                        key,
                        spec.value(),
                        spec.unit(),
                        spec.bounds(),
                        spec.scale(),
                        spec.refine(),
                    )?);
                    index
                };
                mapping.push(joint_index);
            }
            local_to_joint.push(mapping);
        }
        Ok(Self {
            parameters: ParameterSet::new(specs)?,
            histogram_ids: histograms
                .iter()
                .map(|histogram| histogram.histogram_id.clone())
                .collect(),
            local_layouts,
            local_to_joint,
        })
    }

    /// Borrow the stable joint physical parameter set.
    #[must_use]
    pub const fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }

    /// Borrow stable histogram identities in packing order.
    #[must_use]
    pub fn histogram_ids(&self) -> &[RecordId] {
        &self.histogram_ids
    }

    /// Install joint physical values into every histogram request.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldError`] for a stale histogram contract or an
    /// invalid joint/local value.
    pub fn apply_values(
        &self,
        histograms: &[JointRietveldHistogram],
        values: &[f64],
    ) -> Result<Vec<JointRietveldHistogram>, JointRietveldError> {
        self.validate_contract(histograms)?;
        if values.len() != self.parameters.specs().len() {
            return Err(JointRietveldError::ValueLengthMismatch);
        }
        histograms
            .iter()
            .enumerate()
            .map(|(histogram_index, histogram)| {
                let local_values = self.local_to_joint[histogram_index]
                    .iter()
                    .map(|index| values[*index])
                    .collect::<Vec<_>>();
                let mut updated = histogram.clone();
                updated.input = self.local_layouts[histogram_index]
                    .apply_values(&histogram.input, &local_values)?;
                Ok(updated)
            })
            .collect()
    }

    fn validate_contract(
        &self,
        histograms: &[JointRietveldHistogram],
    ) -> Result<(), JointRietveldError> {
        if histograms.len() != self.histogram_ids.len()
            || histograms
                .iter()
                .zip(&self.histogram_ids)
                .any(|(histogram, expected)| histogram.histogram_id != *expected)
        {
            return Err(JointRietveldError::HistogramContractMismatch);
        }
        validate_histograms(histograms)?;
        for ((histogram, expected_layout), expected_mapping) in histograms
            .iter()
            .zip(&self.local_layouts)
            .zip(&self.local_to_joint)
        {
            let current_layout = RietveldParameterLayout::new(
                &histogram.input,
                &histogram.selection,
                &histogram.lattice_bounds,
            )?;
            if current_layout != *expected_layout
                || current_layout.parameters().specs().len() != expected_mapping.len()
            {
                return Err(JointRietveldError::HistogramContractMismatch);
            }
        }
        Ok(())
    }

    fn local_direction(
        &self,
        histogram_index: usize,
        direction: &[f64],
    ) -> Result<Vec<f64>, JointRietveldError> {
        if direction.len() != self.parameters.specs().len() {
            return Err(JointRietveldError::ValueLengthMismatch);
        }
        Ok(self.local_to_joint[histogram_index]
            .iter()
            .map(|index| direction[*index])
            .collect())
    }

    fn scatter_add(
        &self,
        histogram_index: usize,
        local: &[f64],
        joint: &mut [f64],
    ) -> Result<(), JointRietveldError> {
        if local.len() != self.local_to_joint[histogram_index].len() {
            return Err(JointRietveldError::LocalProductLengthMismatch);
        }
        for (value, index) in local.iter().zip(&self.local_to_joint[histogram_index]) {
            joint[*index] += value;
        }
        Ok(())
    }
}

/// Prepared matrix-free sum of all histogram objectives.
pub struct PreparedJointRietveldObjective {
    histograms: Vec<JointRietveldHistogram>,
    layout: JointRietveldLayout,
    objectives: Vec<PreparedGeneralRietveldObjective>,
}

impl PreparedJointRietveldObjective {
    /// Prepare all histogram products against one shared/local layout.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldError`] for a stale layout or objective failure.
    pub fn new(
        histograms: Vec<JointRietveldHistogram>,
        layout: JointRietveldLayout,
    ) -> Result<Self, JointRietveldError> {
        layout.validate_contract(&histograms)?;
        let objectives = histograms
            .iter()
            .zip(&layout.local_layouts)
            .map(|(histogram, local_layout)| {
                PreparedGeneralRietveldObjective::new(
                    histogram.input.clone(),
                    histogram.calculation.clone(),
                    local_layout.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            histograms,
            layout,
            objectives,
        })
    }

    /// Borrow the joint parameter layout.
    #[must_use]
    pub const fn layout(&self) -> &JointRietveldLayout {
        &self.layout
    }

    /// Return expensive model products consumed while preparing all gradients.
    #[must_use]
    pub fn preparation_evaluation_count(&self) -> usize {
        self.objectives
            .iter()
            .map(PreparedGeneralRietveldObjective::preparation_evaluation_count)
            .sum()
    }

    /// Return expensive model products consumed by one joint normal product.
    #[must_use]
    pub fn normal_product_evaluation_count(&self) -> usize {
        self.objectives
            .iter()
            .map(PreparedGeneralRietveldObjective::normal_product_evaluation_count)
            .sum()
    }

    /// Apply every histogram Jacobian to one joint physical direction.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldError`] for an invalid direction or product.
    pub fn jvp(&self, direction: &[f64]) -> Result<Vec<JointRietveldProduct>, JointRietveldError> {
        self.objectives
            .iter()
            .enumerate()
            .map(|(index, objective)| {
                let local = self.layout.local_direction(index, direction)?;
                let (profile, derivative) = objective.jvp(&local)?;
                Ok(JointRietveldProduct {
                    histogram_id: self.histograms[index].histogram_id.clone(),
                    profile,
                    derivative,
                })
            })
            .collect()
    }

    /// Apply the transpose of the complete joint Jacobian.
    ///
    /// Shared structural rows receive the sum of all histogram products.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldError`] for histogram/sample shape or product
    /// failures.
    pub fn vjp(&self, sample_weights: &[Vec<f64>]) -> Result<Vec<f64>, JointRietveldError> {
        if sample_weights.len() != self.objectives.len() {
            return Err(JointRietveldError::HistogramProductCountMismatch);
        }
        let mut result = vec![0.0; self.layout.parameters.specs().len()];
        for (index, (objective, weights)) in self.objectives.iter().zip(sample_weights).enumerate()
        {
            self.layout
                .scatter_add(index, &objective.vjp(weights)?, &mut result)?;
        }
        Ok(result)
    }

    /// Apply the summed `J^T W J + damping I` joint normal operator.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldError`] for invalid damping or product state.
    pub fn normal_product(
        &self,
        direction: &[f64],
        damping: f64,
    ) -> Result<Vec<f64>, JointRietveldError> {
        if !damping.is_finite() || damping < 0.0 {
            return Err(JointRietveldError::InvalidDamping);
        }
        let mut result = vec![0.0; self.layout.parameters.specs().len()];
        for (index, objective) in self.objectives.iter().enumerate() {
            let local = self.layout.local_direction(index, direction)?;
            let local_product = objective.normal_product(&local, 0.0)?;
            self.layout
                .scatter_add(index, &local_product, &mut result)?;
        }
        for (value, direction) in result.iter_mut().zip(direction) {
            *value += damping * direction;
        }
        Ok(result)
    }

    /// Evaluate the summed accepted-state objective and physical gradient.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldError`] for residual or reverse-product state.
    pub fn gradient(&self) -> Result<JointRietveldGradient, JointRietveldError> {
        let mut calculated = Vec::with_capacity(self.objectives.len());
        let mut gradient = vec![0.0; self.layout.parameters.specs().len()];
        let mut value = 0.0;
        for (index, objective) in self.objectives.iter().enumerate() {
            let (profile, local_gradient) = objective.gradient()?;
            value += 0.5 * objective.calculation().metrics.chi_square;
            calculated.push(profile);
            self.layout
                .scatter_add(index, &local_gradient, &mut gradient)?;
        }
        Ok(JointRietveldGradient {
            calculated,
            objective: value,
            gradient,
        })
    }
}

fn is_shared(key: &ParameterKey) -> bool {
    matches!(key.module(), "lattice" | "site")
}

fn validate_histograms(histograms: &[JointRietveldHistogram]) -> Result<(), JointRietveldError> {
    if histograms.len() < 2 {
        return Err(JointRietveldError::TooFewHistograms);
    }
    if histograms
        .iter()
        .map(|histogram| &histogram.histogram_id)
        .collect::<BTreeSet<_>>()
        .len()
        != histograms.len()
    {
        return Err(JointRietveldError::DuplicateHistogramId);
    }
    let shared_selection = histograms[0].selection.structural;
    if histograms.iter().skip(1).any(|histogram| {
        let selection = histogram.selection.structural;
        selection.lattice != shared_selection.lattice
            || selection.coordinates != shared_selection.coordinates
            || selection.occupancy != shared_selection.occupancy
            || selection.u_iso != shared_selection.u_iso
    }) {
        return Err(JointRietveldError::SharedSelectionMismatch);
    }
    let mut phases = BTreeMap::new();
    for histogram in histograms {
        histogram.input.validate()?;
        for phase in &histogram.input.phases {
            let definition = phase.definition();
            let contract = (
                phase.site_ids(),
                definition.cell,
                &definition.space_group,
                &definition.fractional_xyz,
                &definition.occupancy,
                &definition.u_iso_angstrom2,
                &definition.anisotropic_mask,
                &definition.u_aniso_cif_angstrom2,
                definition.coordinate_tolerance.to_bits(),
            );
            if let Some(previous) = phases.insert(phase.phase_id().clone(), contract) {
                if previous != contract {
                    return Err(JointRietveldError::SharedPhaseMismatch {
                        phase_id: phase.phase_id().clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Invalid joint native Rietveld parameter or objective state.
#[derive(Debug)]
pub enum JointRietveldError {
    /// A joint objective requires at least two histograms.
    TooFewHistograms,
    /// Stable histogram identities must be unique.
    DuplicateHistogramId,
    /// Shared structural selection differs between histograms.
    SharedSelectionMismatch,
    /// Physical structure differs for one shared stable phase.
    SharedPhaseMismatch {
        /// Stable incompatible phase identity.
        phase_id: RecordId,
    },
    /// One shared scalar differs in value or metadata.
    SharedParameterMismatch {
        /// Stable incompatible parameter identity.
        key: ParameterKey,
    },
    /// A prepared layout was paired with different histograms.
    HistogramContractMismatch,
    /// Joint value/direction length is wrong.
    ValueLengthMismatch,
    /// Number of histogram reverse products is wrong.
    HistogramProductCountMismatch,
    /// A local reverse product has a stale parameter length.
    LocalProductLengthMismatch,
    /// Damping must be finite and non-negative.
    InvalidDamping,
    /// Stable scalar parameter state is invalid.
    Parameter(ParameterError),
    /// Complete single-histogram parameter state is invalid.
    GeneralParameter(RietveldGeneralParameterError),
    /// Complete single-histogram objective state is invalid.
    GeneralObjective(RietveldGeneralObjectiveError),
}

impl Display for JointRietveldError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewHistograms => {
                formatter.write_str("joint Rietveld objective requires at least two histograms")
            }
            Self::DuplicateHistogramId => {
                formatter.write_str("joint Rietveld histogram IDs must be unique")
            }
            Self::SharedSelectionMismatch => {
                formatter.write_str("joint Rietveld shared structural selections must match")
            }
            Self::SharedPhaseMismatch { phase_id } => write!(
                formatter,
                "joint Rietveld phase {phase_id:?} has incompatible shared structural state"
            ),
            Self::SharedParameterMismatch { key } => {
                write!(
                    formatter,
                    "joint Rietveld shared parameter {key} is incompatible"
                )
            }
            Self::HistogramContractMismatch => {
                formatter.write_str("joint Rietveld histogram contract changed under the layout")
            }
            Self::ValueLengthMismatch => {
                formatter.write_str("joint Rietveld value/direction length is wrong")
            }
            Self::HistogramProductCountMismatch => {
                formatter.write_str("joint Rietveld histogram reverse-product count is wrong")
            }
            Self::LocalProductLengthMismatch => {
                formatter.write_str("joint Rietveld local product length is wrong")
            }
            Self::InvalidDamping => {
                formatter.write_str("joint Rietveld damping must be finite and non-negative")
            }
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::GeneralParameter(error) => Display::fmt(error, formatter),
            Self::GeneralObjective(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for JointRietveldError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parameter(error) => Some(error),
            Self::GeneralParameter(error) => Some(error),
            Self::GeneralObjective(error) => Some(error),
            Self::TooFewHistograms
            | Self::DuplicateHistogramId
            | Self::SharedSelectionMismatch
            | Self::SharedPhaseMismatch { .. }
            | Self::SharedParameterMismatch { .. }
            | Self::HistogramContractMismatch
            | Self::ValueLengthMismatch
            | Self::HistogramProductCountMismatch
            | Self::LocalProductLengthMismatch
            | Self::InvalidDamping => None,
        }
    }
}

impl From<ParameterError> for JointRietveldError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}

impl From<RietveldGeneralParameterError> for JointRietveldError {
    fn from(value: RietveldGeneralParameterError) -> Self {
        Self::GeneralParameter(value)
    }
}

impl From<RietveldGeneralObjectiveError> for JointRietveldError {
    fn from(value: RietveldGeneralObjectiveError) -> Self {
        Self::GeneralObjective(value)
    }
}

impl From<crate::RietveldError> for JointRietveldError {
    fn from(value: crate::RietveldError) -> Self {
        Self::GeneralParameter(RietveldGeneralParameterError::Rietveld(value))
    }
}
