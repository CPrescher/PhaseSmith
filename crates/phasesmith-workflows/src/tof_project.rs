//! Validated project-level ownership for runnable fixed-instrument TOF analyses.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_model::{DomainError, ProjectRecord, RecordId};

use crate::{TofLeBailCheckpoint, TofLeBailError, TofLeBailInput, TofLeBailOptions};

/// One complete runnable fixed-instrument TOF Le Bail analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailAnalysis {
    /// TOF histogram owned by the surrounding project.
    pub histogram_id: RecordId,
    /// Editable fixed-instrument extraction state.
    pub input: TofLeBailInput,
    /// Numerical and execution controls, including total cycle budget.
    pub options: TofLeBailOptions,
    /// Optional last accepted continuation state.
    pub checkpoint: Option<TofLeBailCheckpoint>,
}

impl TofLeBailAnalysis {
    /// Validate the standalone workflow and optional checkpoint contract.
    ///
    /// # Errors
    ///
    /// Returns [`TofProjectError`] for invalid input, options, or continuation.
    pub fn validate(&self) -> Result<(), TofProjectError> {
        self.input.validate()?;
        self.options.validate()?;
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint.validate_for(&self.input, &self.options)?;
        }
        Ok(())
    }
}

/// Revisioned native project plus zero or one TOF analysis per TOF histogram.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailProjectState {
    /// Application-neutral mixed CW/TOF project snapshot.
    pub project: ProjectRecord,
    /// Runnable TOF analyses in stable caller-owned order.
    pub analyses: Vec<TofLeBailAnalysis>,
}

impl TofLeBailProjectState {
    /// Validate project records and every TOF analysis reference.
    ///
    /// # Errors
    ///
    /// Returns [`TofProjectError`] for duplicate/missing ownership, mismatched
    /// pattern/instrument/phase state, or an invalid workflow checkpoint.
    pub fn validate(&self) -> Result<(), TofProjectError> {
        self.project.validate()?;
        let mut histogram_ids = BTreeSet::new();
        for analysis in &self.analyses {
            analysis.validate()?;
            if !histogram_ids.insert(analysis.histogram_id.clone()) {
                return Err(TofProjectError::DuplicateAnalysis {
                    histogram_id: analysis.histogram_id.clone(),
                });
            }
            let histogram = self
                .project
                .tof_histograms
                .iter()
                .find(|item| item.histogram_id == analysis.histogram_id)
                .ok_or_else(|| TofProjectError::UnknownHistogram {
                    histogram_id: analysis.histogram_id.clone(),
                })?;
            if analysis.input.pattern != histogram.pattern
                || analysis.input.instrument != histogram.experiment.instrument
            {
                return Err(TofProjectError::HistogramStateMismatch {
                    histogram_id: analysis.histogram_id.clone(),
                });
            }
            let phase_ids = analysis
                .input
                .phases
                .iter()
                .map(crate::TofLeBailPhase::phase_id)
                .collect::<Vec<_>>();
            if phase_ids != histogram.phase_ids.iter().collect::<Vec<_>>() {
                return Err(TofProjectError::PhaseOrderMismatch {
                    histogram_id: analysis.histogram_id.clone(),
                });
            }
            for phase in &analysis.input.phases {
                let stored = self
                    .project
                    .phases
                    .iter()
                    .find(|item| &item.phase_id == phase.phase_id())
                    .ok_or_else(|| TofProjectError::PhaseOrderMismatch {
                        histogram_id: analysis.histogram_id.clone(),
                    })?;
                if stored.name != phase.name() {
                    return Err(TofProjectError::PhaseStateMismatch {
                        phase_id: stored.phase_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Invalid project-level fixed-instrument TOF state.
#[derive(Debug)]
pub enum TofProjectError {
    /// Application-neutral project validation failed.
    Domain(DomainError),
    /// TOF input, options, or checkpoint validation failed.
    Workflow(TofLeBailError),
    /// More than one analysis owns one TOF histogram.
    DuplicateAnalysis {
        /// Duplicated histogram identity.
        histogram_id: RecordId,
    },
    /// Analysis references a missing TOF histogram.
    UnknownHistogram {
        /// Missing histogram identity.
        histogram_id: RecordId,
    },
    /// Pattern or instrument differs from the project TOF histogram.
    HistogramStateMismatch {
        /// Inconsistent histogram identity.
        histogram_id: RecordId,
    },
    /// Active phase order differs from the histogram references.
    PhaseOrderMismatch {
        /// Inconsistent histogram identity.
        histogram_id: RecordId,
    },
    /// Analysis phase label differs from the project phase record.
    PhaseStateMismatch {
        /// Inconsistent phase identity.
        phase_id: RecordId,
    },
}

impl Display for TofProjectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain(error) => Display::fmt(error, formatter),
            Self::Workflow(error) => Display::fmt(error, formatter),
            Self::DuplicateAnalysis { histogram_id } => {
                write!(
                    formatter,
                    "duplicate TOF analysis for histogram {histogram_id}"
                )
            }
            Self::UnknownHistogram { histogram_id } => {
                write!(formatter, "unknown TOF histogram {histogram_id}")
            }
            Self::HistogramStateMismatch { histogram_id } => write!(
                formatter,
                "TOF analysis state differs from histogram {histogram_id}"
            ),
            Self::PhaseOrderMismatch { histogram_id } => write!(
                formatter,
                "TOF analysis phase order differs from histogram {histogram_id}"
            ),
            Self::PhaseStateMismatch { phase_id } => {
                write!(formatter, "TOF analysis phase state differs for {phase_id}")
            }
        }
    }
}

impl Error for TofProjectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(error) => Some(error),
            Self::Workflow(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DomainError> for TofProjectError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<TofLeBailError> for TofProjectError {
    fn from(value: TofLeBailError) -> Self {
        Self::Workflow(value)
    }
}
