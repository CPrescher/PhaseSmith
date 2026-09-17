//! Validated project-level ownership for runnable native Rietveld analyses.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::OwnedCwContributions;
use phasesmith_model::{DomainError, ProjectRecord, RadiationDefinition, RecordId};

use crate::{
    Constraint, ConstraintError, ConstraintTransform, LatticeBounds, RietveldCovarianceOptions,
    RietveldGeneralCheckpoint, RietveldGeneralParameterError, RietveldGeneralRefinementError,
    RietveldInput, RietveldParameterLayout, RietveldParameterSelection, RietveldRefinementError,
    RietveldRefinementOptions,
};

/// One complete runnable single-histogram native Rietveld analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldAnalysis {
    /// Histogram owned by the surrounding project.
    pub histogram_id: RecordId,
    /// Editable calculation state synchronized with the project snapshot.
    pub input: RietveldInput,
    /// Complete selected parameter families.
    pub selection: RietveldParameterSelection,
    /// Optional lattice bounds aligned with `input.phases`.
    pub lattice_bounds: Vec<Option<LatticeBounds>>,
    /// Ordered physical constraint graph.
    pub constraints: Vec<Constraint>,
    /// Numerical and bounded-runtime controls.
    pub options: RietveldRefinementOptions,
    /// Optional final covariance controls.
    pub covariance: RietveldCovarianceOptions,
    /// Optional last accepted continuation state.
    pub checkpoint: Option<RietveldGeneralCheckpoint>,
}

impl RietveldAnalysis {
    /// Validate the complete standalone solver contract.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldProjectError`] for invalid input, selection, bounds,
    /// constraints, options, or checkpoint state.
    pub fn validate(&self) -> Result<(), RietveldProjectError> {
        self.input.validate()?;
        self.selection.validate()?;
        self.options.validate()?;
        RietveldCovarianceOptions::new(
            self.covariance.enabled,
            self.covariance.max_parameters,
            self.covariance.unresolved_correlation,
        )?;
        if self.lattice_bounds.len() != self.input.phases.len() {
            return Err(RietveldProjectError::LatticeBoundCountMismatch);
        }
        let layout =
            RietveldParameterLayout::new(&self.input, &self.selection, &self.lattice_bounds)?;
        let transform =
            ConstraintTransform::new(layout.parameters().clone(), self.constraints.clone())?;
        let constrained = transform.unpack(&transform.pack()?, false)?;
        if layout.parameters().specs().iter().any(|spec| {
            constrained
                .get(spec.key())
                .is_none_or(|value| (value - spec.value()).abs() > 2.0e-12)
        }) {
            return Err(RietveldProjectError::UnsatisfiedConstraint);
        }
        if let Some(checkpoint) = &self.checkpoint {
            if checkpoint.profile_accuracy != self.options.calculation.profile_accuracy {
                return Err(RietveldProjectError::General(
                    RietveldGeneralRefinementError::InvalidCheckpoint {
                        reason: "profile accuracy changed",
                    },
                ));
            }
            if checkpoint.completed_iterations > self.options.limits.max_iterations() {
                return Err(RietveldProjectError::CheckpointExceedsIterationLimit);
            }
            checkpoint.validate_for(
                &self.input,
                &self.selection,
                &self.lattice_bounds,
                &self.constraints,
            )?;
        }
        Ok(())
    }
}

/// Revisioned native project plus zero or one runnable analysis per histogram.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldProjectState {
    /// Application-neutral multi-histogram project snapshot.
    pub project: ProjectRecord,
    /// Runnable native analyses in stable caller-owned order.
    pub analyses: Vec<RietveldAnalysis>,
}

impl RietveldProjectState {
    /// Validate the project and every cross-record Rietveld reference.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldProjectError`] for invalid project state, duplicate or
    /// missing histogram ownership, unsupported radiation/providers, or a
    /// solver request that differs from its project records.
    pub fn validate(&self) -> Result<(), RietveldProjectError> {
        self.project.validate()?;
        let mut histogram_ids = BTreeSet::new();
        for analysis in &self.analyses {
            analysis.validate()?;
            if !histogram_ids.insert(analysis.histogram_id.clone()) {
                return Err(RietveldProjectError::DuplicateAnalysis {
                    histogram_id: analysis.histogram_id.clone(),
                });
            }
            let histogram = self
                .project
                .histograms
                .iter()
                .find(|item| item.histogram_id == analysis.histogram_id)
                .ok_or_else(|| RietveldProjectError::UnknownHistogram {
                    histogram_id: analysis.histogram_id.clone(),
                })?;
            let expected_spectrum = match &histogram.experiment.radiation {
                RadiationDefinition::Monochromatic { .. } => None,
                RadiationDefinition::FixedSpectrum { spectrum, .. } => Some(spectrum),
            };
            if analysis.input.pattern != histogram.pattern
                || analysis.input.instrument != histogram.experiment.instrument
                || analysis.input.fixed_spectrum.as_ref() != expected_spectrum
                || analysis.input.axial_geometry != histogram.experiment.axial_geometry
                || analysis.input.position_correction != histogram.experiment.position_correction
            {
                return Err(RietveldProjectError::HistogramStateMismatch {
                    histogram_id: analysis.histogram_id.clone(),
                });
            }
            let analysis_phase_ids = analysis
                .input
                .phases
                .iter()
                .map(crate::RietveldPhase::phase_id)
                .collect::<Vec<_>>();
            if analysis_phase_ids != histogram.phase_ids.iter().collect::<Vec<_>>() {
                return Err(RietveldProjectError::PhaseOrderMismatch {
                    histogram_id: analysis.histogram_id.clone(),
                });
            }
            for phase in &analysis.input.phases {
                let stored = self
                    .project
                    .phases
                    .iter()
                    .find(|item| &item.phase_id == phase.phase_id())
                    .ok_or_else(|| RietveldProjectError::PhaseOrderMismatch {
                        histogram_id: analysis.histogram_id.clone(),
                    })?;
                if !stored.required_providers.is_empty() {
                    return Err(RietveldProjectError::ExternalProviderRequired {
                        phase_id: stored.phase_id.clone(),
                    });
                }
                if phase.sample_physics().is_none()
                    && phase.contributions()
                        != &OwnedCwContributions::neutral(phase.reflection_ids().len())
                {
                    return Err(RietveldProjectError::OpaqueStaticContributions {
                        phase_id: stored.phase_id.clone(),
                    });
                }
                if stored.name != phase.name() || stored.definition != *phase.definition() {
                    return Err(RietveldProjectError::PhaseStateMismatch {
                        phase_id: stored.phase_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Invalid project-level native Rietveld state.
#[derive(Debug)]
pub enum RietveldProjectError {
    /// Application-neutral project validation failed.
    Domain(DomainError),
    /// Rietveld input validation failed.
    Rietveld(crate::RietveldError),
    /// Selection or parameter-layout validation failed.
    Parameter(RietveldGeneralParameterError),
    /// Constraint graph validation failed.
    Constraint(ConstraintError),
    /// Solver-control validation failed.
    Options(RietveldRefinementError),
    /// Checkpoint or covariance validation failed.
    General(RietveldGeneralRefinementError),
    /// Lattice bounds do not align with phase order.
    LatticeBoundCountMismatch,
    /// Initial physical values do not satisfy the constraint graph.
    UnsatisfiedConstraint,
    /// Saved iteration limits cannot resume the accepted checkpoint.
    CheckpointExceedsIterationLimit,
    /// More than one analysis owns one histogram.
    DuplicateAnalysis {
        /// Duplicated histogram identity.
        histogram_id: RecordId,
    },
    /// Analysis references a missing histogram.
    UnknownHistogram {
        /// Missing histogram identity.
        histogram_id: RecordId,
    },
    /// Pattern or experiment state differs from the project histogram.
    HistogramStateMismatch {
        /// Inconsistent histogram identity.
        histogram_id: RecordId,
    },
    /// Active phase order differs from the histogram references.
    PhaseOrderMismatch {
        /// Inconsistent histogram identity.
        histogram_id: RecordId,
    },
    /// Native analysis cannot execute a project phase requiring an extension.
    ExternalProviderRequired {
        /// Phase requiring the extension.
        phase_id: RecordId,
    },
    /// Phase carries opaque static contribution arrays with no native model.
    OpaqueStaticContributions {
        /// Phase with non-reconstructible contributions.
        phase_id: RecordId,
    },
    /// Analysis phase label or structural definition differs from the project.
    PhaseStateMismatch {
        /// Inconsistent phase identity.
        phase_id: RecordId,
    },
}

impl Display for RietveldProjectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Constraint(error) => Display::fmt(error, formatter),
            Self::Options(error) => Display::fmt(error, formatter),
            Self::General(error) => Display::fmt(error, formatter),
            Self::LatticeBoundCountMismatch => {
                formatter.write_str("Rietveld lattice bounds must align with phase order")
            }
            Self::UnsatisfiedConstraint => {
                formatter.write_str("initial Rietveld values do not satisfy their constraints")
            }
            Self::CheckpointExceedsIterationLimit => {
                formatter.write_str("Rietveld checkpoint exceeds the saved maximum iteration limit")
            }
            Self::DuplicateAnalysis { histogram_id } => {
                write!(
                    formatter,
                    "histogram {histogram_id} has multiple Rietveld analyses"
                )
            }
            Self::UnknownHistogram { histogram_id } => {
                write!(
                    formatter,
                    "Rietveld analysis references unknown histogram {histogram_id}"
                )
            }
            Self::HistogramStateMismatch { histogram_id } => write!(
                formatter,
                "Rietveld analysis state differs from histogram {histogram_id}"
            ),
            Self::PhaseOrderMismatch { histogram_id } => write!(
                formatter,
                "Rietveld phase order differs from histogram {histogram_id} references"
            ),
            Self::ExternalProviderRequired { phase_id } => write!(
                formatter,
                "phase {phase_id} requires an unavailable external provider"
            ),
            Self::OpaqueStaticContributions { phase_id } => write!(
                formatter,
                "phase {phase_id} has opaque static sample-physics contributions"
            ),
            Self::PhaseStateMismatch { phase_id } => {
                write!(
                    formatter,
                    "Rietveld phase {phase_id} differs from project state"
                )
            }
        }
    }
}

impl Error for RietveldProjectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(error) => Some(error),
            Self::Rietveld(error) => Some(error),
            Self::Parameter(error) => Some(error),
            Self::Constraint(error) => Some(error),
            Self::Options(error) => Some(error),
            Self::General(error) => Some(error),
            _ => None,
        }
    }
}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for RietveldProjectError {
            fn from(error: $source) -> Self {
                Self::$variant(error)
            }
        }
    };
}

from_error!(DomainError, Domain);
from_error!(crate::RietveldError, Rietveld);
from_error!(RietveldGeneralParameterError, Parameter);
from_error!(ConstraintError, Constraint);
from_error!(RietveldRefinementError, Options);
from_error!(RietveldGeneralRefinementError, General);
