//! Validated project ownership for joint multi-bank TOF geometry analyses.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_model::{DomainError, ProjectRecord, RecordId};

use crate::{
    LatticeParameterization, TofMultiBankGeometryCheckpoint, TofMultiBankGeometryError,
    TofMultiBankGeometryInput, TofMultiBankGeometryOptions,
};

/// One complete runnable joint multi-bank TOF geometry analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryAnalysis {
    /// Stable analysis identity independent of its member histogram IDs.
    pub analysis_id: RecordId,
    /// Editable bank, shared-cell, and bank-local instrument state.
    pub input: TofMultiBankGeometryInput,
    /// Numerical and bounded-runtime controls.
    pub options: TofMultiBankGeometryOptions,
    /// Optional complete last-accepted continuation state.
    pub checkpoint: Option<TofMultiBankGeometryCheckpoint>,
}

impl TofMultiBankGeometryAnalysis {
    /// Validate the workflow and optional checkpoint contract.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankProjectError`] for invalid numerical state.
    pub fn validate(&self) -> Result<(), TofMultiBankProjectError> {
        self.input.validate()?;
        self.options.validate()?;
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint.validate_for(&self.input, &self.options)?;
        }
        Ok(())
    }
}

/// Revisioned project plus disjoint joint multi-bank TOF geometry analyses.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryProjectState {
    /// Application-neutral mixed CW/TOF project snapshot.
    pub project: ProjectRecord,
    /// Runnable joint analyses in stable caller-owned order.
    pub analyses: Vec<TofMultiBankGeometryAnalysis>,
}

impl TofMultiBankGeometryProjectState {
    /// Validate project ownership, exact histogram state, cells, and checkpoints.
    ///
    /// Each TOF histogram may belong to at most one joint analysis. Bank IDs are
    /// the corresponding project histogram IDs; this prevents a detached alias
    /// from silently resolving to the wrong observed data.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankProjectError`] for invalid or inconsistent state.
    pub fn validate(&self) -> Result<(), TofMultiBankProjectError> {
        self.project.validate()?;
        let mut analysis_ids = BTreeSet::new();
        let mut histogram_ids = BTreeSet::new();
        for analysis in &self.analyses {
            analysis.validate()?;
            if !analysis_ids.insert(analysis.analysis_id.clone()) {
                return Err(TofMultiBankProjectError::DuplicateAnalysis {
                    analysis_id: analysis.analysis_id.clone(),
                });
            }
            for bank in &analysis.input.lattice.multibank.banks {
                if !histogram_ids.insert(bank.bank_id.clone()) {
                    return Err(TofMultiBankProjectError::DuplicateHistogramOwnership {
                        histogram_id: bank.bank_id.clone(),
                    });
                }
                let histogram = self
                    .project
                    .tof_histograms
                    .iter()
                    .find(|item| item.histogram_id == bank.bank_id)
                    .ok_or_else(|| TofMultiBankProjectError::UnknownHistogram {
                        histogram_id: bank.bank_id.clone(),
                    })?;
                if bank.input.pattern != histogram.pattern
                    || bank.input.instrument != histogram.experiment.instrument
                {
                    return Err(TofMultiBankProjectError::HistogramStateMismatch {
                        histogram_id: bank.bank_id.clone(),
                    });
                }
                let phase_ids = bank
                    .input
                    .phases
                    .iter()
                    .map(crate::TofLeBailPhase::phase_id)
                    .collect::<Vec<_>>();
                if phase_ids != histogram.phase_ids.iter().collect::<Vec<_>>() {
                    return Err(TofMultiBankProjectError::PhaseOrderMismatch {
                        histogram_id: bank.bank_id.clone(),
                    });
                }
                for phase in &bank.input.phases {
                    let stored = self
                        .project
                        .phases
                        .iter()
                        .find(|item| &item.phase_id == phase.phase_id())
                        .ok_or_else(|| TofMultiBankProjectError::PhaseOrderMismatch {
                            histogram_id: bank.bank_id.clone(),
                        })?;
                    if stored.name != phase.name() {
                        return Err(TofMultiBankProjectError::PhaseStateMismatch {
                            phase_id: stored.phase_id.clone(),
                        });
                    }
                }
            }
            for lattice in &analysis.input.lattice.lattice_phases {
                let stored = self
                    .project
                    .phases
                    .iter()
                    .find(|phase| &phase.phase_id == lattice.phase_id())
                    .ok_or_else(|| TofMultiBankProjectError::PhaseStateMismatch {
                        phase_id: lattice.phase_id().clone(),
                    })?;
                let expected = LatticeParameterization::new(
                    stored.definition.space_group.clone(),
                    stored.definition.cell,
                )?;
                if lattice.initial_cell() != stored.definition.cell
                    || lattice.parameterization() != &expected
                {
                    return Err(TofMultiBankProjectError::LatticeStateMismatch {
                        phase_id: lattice.phase_id().clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Invalid project-level joint multi-bank TOF state.
#[derive(Debug)]
pub enum TofMultiBankProjectError {
    /// Application-neutral project validation failed.
    Domain(DomainError),
    /// Joint geometry input, options, or checkpoint validation failed.
    Workflow(TofMultiBankGeometryError),
    /// More than one analysis has the same stable identity.
    DuplicateAnalysis {
        /// Duplicated analysis identity.
        analysis_id: RecordId,
    },
    /// A TOF histogram is assigned to multiple joint analyses.
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
    /// Active phase order differs from the histogram references.
    PhaseOrderMismatch {
        /// Inconsistent histogram identity.
        histogram_id: RecordId,
    },
    /// Analysis phase label or identity differs from project state.
    PhaseStateMismatch {
        /// Inconsistent phase identity.
        phase_id: RecordId,
    },
    /// Shared initial cell or exact symmetry setting differs from project state.
    LatticeStateMismatch {
        /// Inconsistent phase identity.
        phase_id: RecordId,
    },
}

impl Display for TofMultiBankProjectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain(error) => Display::fmt(error, formatter),
            Self::Workflow(error) => Display::fmt(error, formatter),
            Self::DuplicateAnalysis { analysis_id } => {
                write!(formatter, "duplicate joint TOF analysis {analysis_id}")
            }
            Self::DuplicateHistogramOwnership { histogram_id } => write!(
                formatter,
                "TOF histogram {histogram_id} belongs to multiple joint analyses"
            ),
            Self::UnknownHistogram { histogram_id } => {
                write!(formatter, "unknown TOF histogram {histogram_id}")
            }
            Self::HistogramStateMismatch { histogram_id } => write!(
                formatter,
                "joint TOF bank state differs from histogram {histogram_id}"
            ),
            Self::PhaseOrderMismatch { histogram_id } => write!(
                formatter,
                "joint TOF phase order differs from histogram {histogram_id}"
            ),
            Self::PhaseStateMismatch { phase_id } => {
                write!(formatter, "joint TOF phase state differs for {phase_id}")
            }
            Self::LatticeStateMismatch { phase_id } => write!(
                formatter,
                "joint TOF lattice state differs from project phase {phase_id}"
            ),
        }
    }
}

impl Error for TofMultiBankProjectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(error) => Some(error),
            Self::Workflow(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DomainError> for TofMultiBankProjectError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<TofMultiBankGeometryError> for TofMultiBankProjectError {
    fn from(value: TofMultiBankGeometryError) -> Self {
        Self::Workflow(value)
    }
}

impl From<crate::LatticeError> for TofMultiBankProjectError {
    fn from(value: crate::LatticeError) -> Self {
        Self::Workflow(value.into())
    }
}
