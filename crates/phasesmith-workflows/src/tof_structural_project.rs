//! Validated project ownership for structural multi-bank neutron TOF analyses.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_model::{DomainError, ProjectRecord, RecordId};

use crate::{
    StructuralTofMultiBankCheckpoint, StructuralTofMultiBankError, StructuralTofMultiBankInput,
    StructuralTofMultiBankRefinementError, StructuralTofMultiBankRefinementOptions,
};

/// One complete runnable structural multi-bank TOF analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankAnalysis {
    /// Stable analysis identity independent of its member histogram IDs.
    pub analysis_id: RecordId,
    /// Editable shared structure and bank-local observation/model state.
    pub input: StructuralTofMultiBankInput,
    /// Numerical and bounded-runtime controls.
    pub options: StructuralTofMultiBankRefinementOptions,
    /// Optional complete last-accepted continuation state.
    pub checkpoint: Option<StructuralTofMultiBankCheckpoint>,
}

impl StructuralTofMultiBankAnalysis {
    /// Validate the workflow and optional checkpoint contract.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofProjectError`] for invalid numerical state.
    pub fn validate(&self) -> Result<(), StructuralTofProjectError> {
        self.input.validate()?;
        self.options.validate()?;
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint.validate_for(&self.input)?;
        }
        Ok(())
    }
}

/// Revisioned project plus disjoint structural multi-bank TOF analyses.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankProjectState {
    /// Application-neutral mixed CW/TOF project snapshot.
    pub project: ProjectRecord,
    /// Runnable structural TOF analyses in stable caller-owned order.
    pub analyses: Vec<StructuralTofMultiBankAnalysis>,
}

impl StructuralTofMultiBankProjectState {
    /// Validate project ownership, exact histogram/phase state, and checkpoints.
    ///
    /// Each TOF histogram may belong to at most one structural analysis. Bank
    /// IDs are the corresponding project histogram IDs, and every member
    /// histogram references exactly the one shared structural phase.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofProjectError`] for invalid or inconsistent state.
    pub fn validate(&self) -> Result<(), StructuralTofProjectError> {
        self.project.validate()?;
        let mut analysis_ids = BTreeSet::new();
        let mut histogram_ids = BTreeSet::new();
        for analysis in &self.analyses {
            analysis.validate()?;
            if !analysis_ids.insert(analysis.analysis_id.clone()) {
                return Err(StructuralTofProjectError::DuplicateAnalysis {
                    analysis_id: analysis.analysis_id.clone(),
                });
            }
            let phase_id = analysis.input.phase.phase_id();
            let stored_phase = self
                .project
                .phases
                .iter()
                .find(|phase| &phase.phase_id == phase_id)
                .ok_or_else(|| StructuralTofProjectError::PhaseStateMismatch {
                    phase_id: phase_id.clone(),
                })?;
            if stored_phase.name != analysis.input.phase.name()
                || stored_phase.definition != *analysis.input.phase.definition()
                || !stored_phase.required_providers.is_empty()
            {
                return Err(StructuralTofProjectError::PhaseStateMismatch {
                    phase_id: phase_id.clone(),
                });
            }
            for bank in &analysis.input.banks {
                if !histogram_ids.insert(bank.bank_id.clone()) {
                    return Err(StructuralTofProjectError::DuplicateHistogramOwnership {
                        histogram_id: bank.bank_id.clone(),
                    });
                }
                let histogram = self
                    .project
                    .tof_histograms
                    .iter()
                    .find(|item| item.histogram_id == bank.bank_id)
                    .ok_or_else(|| StructuralTofProjectError::UnknownHistogram {
                        histogram_id: bank.bank_id.clone(),
                    })?;
                if histogram.pattern != bank.pattern
                    || histogram.experiment.instrument != bank.instrument
                {
                    return Err(StructuralTofProjectError::HistogramStateMismatch {
                        histogram_id: bank.bank_id.clone(),
                    });
                }
                if histogram.phase_ids.as_slice() != std::slice::from_ref(phase_id) {
                    return Err(StructuralTofProjectError::PhaseOrderMismatch {
                        histogram_id: bank.bank_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Invalid project-level structural TOF state.
#[derive(Debug)]
pub enum StructuralTofProjectError {
    /// Application-neutral project validation failed.
    Domain(DomainError),
    /// Structural TOF input failed validation.
    Workflow(StructuralTofMultiBankError),
    /// Structural TOF solver/checkpoint state failed validation.
    Refinement(StructuralTofMultiBankRefinementError),
    /// More than one analysis has the same stable identity.
    DuplicateAnalysis {
        /// Duplicated analysis identity.
        analysis_id: RecordId,
    },
    /// A TOF histogram is assigned to multiple structural analyses.
    DuplicateHistogramOwnership {
        /// Multiply owned histogram identity.
        histogram_id: RecordId,
    },
    /// Analysis references a missing TOF histogram.
    UnknownHistogram {
        /// Missing histogram identity.
        histogram_id: RecordId,
    },
    /// Bank pattern or initial instrument differs from its histogram record.
    HistogramStateMismatch {
        /// Inconsistent histogram identity.
        histogram_id: RecordId,
    },
    /// Histogram phase references are not exactly the shared analysis phase.
    PhaseOrderMismatch {
        /// Inconsistent histogram identity.
        histogram_id: RecordId,
    },
    /// Analysis phase identity, label, definition, or provider contract differs.
    PhaseStateMismatch {
        /// Inconsistent phase identity.
        phase_id: RecordId,
    },
}

impl Display for StructuralTofProjectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain(error) => Display::fmt(error, formatter),
            Self::Workflow(error) => Display::fmt(error, formatter),
            Self::Refinement(error) => Display::fmt(error, formatter),
            Self::DuplicateAnalysis { analysis_id } => {
                write!(formatter, "duplicate structural TOF analysis {analysis_id}")
            }
            Self::DuplicateHistogramOwnership { histogram_id } => write!(
                formatter,
                "TOF histogram {histogram_id} belongs to multiple structural analyses"
            ),
            Self::UnknownHistogram { histogram_id } => {
                write!(formatter, "unknown TOF histogram {histogram_id}")
            }
            Self::HistogramStateMismatch { histogram_id } => write!(
                formatter,
                "structural TOF bank state differs from histogram {histogram_id}"
            ),
            Self::PhaseOrderMismatch { histogram_id } => write!(
                formatter,
                "structural TOF phase order differs from histogram {histogram_id}"
            ),
            Self::PhaseStateMismatch { phase_id } => {
                write!(
                    formatter,
                    "structural TOF phase state differs for {phase_id}"
                )
            }
        }
    }
}

impl Error for StructuralTofProjectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(error) => Some(error),
            Self::Workflow(error) => Some(error),
            Self::Refinement(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DomainError> for StructuralTofProjectError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<StructuralTofMultiBankError> for StructuralTofProjectError {
    fn from(value: StructuralTofMultiBankError) -> Self {
        Self::Workflow(value)
    }
}

impl From<StructuralTofMultiBankRefinementError> for StructuralTofProjectError {
    fn from(value: StructuralTofMultiBankRefinementError) -> Self {
        Self::Refinement(value)
    }
}
