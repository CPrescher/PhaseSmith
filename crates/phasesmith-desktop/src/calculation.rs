//! Retained standalone native calculations for desktop plotting.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use phasesmith_core::OwnedCwContributions;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{PatternRecord, RecordId};
use phasesmith_workflows::{
    RietveldCalculation, RietveldCalculationOptions, RietveldInput, RietveldPhase,
    calculate_rietveld_pattern,
};
use serde::{Deserialize, Serialize};

use crate::{
    BinaryPayload, BinarySeriesDescriptor, DesktopError, DesktopErrorCode, DesktopProjectStore,
    ProjectSnapshot, series,
};

/// Process-local identifier for one retained standalone calculation.
pub type CalculationId = u64;

/// Bounded execution and display-calculation controls.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct CalculationOptionsInput {
    /// Exact finite profile support in multiples of FWHM.
    pub support_fwhm: f64,
    /// Whether supplied one-sigma uncertainty weights residual metrics.
    pub use_uncertainty: bool,
    /// Fixed positive worker count, or automatic when absent.
    pub threads: Option<usize>,
    /// Minimum phase count needed before parallel scheduling.
    pub minimum_parallel_phases: usize,
}

impl Default for CalculationOptionsInput {
    fn default() -> Self {
        Self {
            support_fwhm: 20.0,
            use_uncertainty: true,
            threads: Some(2),
            minimum_parallel_phases: 2,
        }
    }
}

impl CalculationOptionsInput {
    fn native(self) -> Result<RietveldCalculationOptions, DesktopError> {
        let execution = ExecutionPolicy::new(self.threads, self.minimum_parallel_phases)
            .map_err(|error| calculation_error(error.to_string()))?;
        RietveldCalculationOptions::new(self.support_fwhm, self.use_uncertainty, execution)
            .map_err(|error| calculation_error(error.to_string()))
    }
}

/// JSON-safe summary of one retained standalone calculation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CalculationResponse {
    /// Process-local retained-result ID.
    pub calculation_id: CalculationId,
    /// Immutable source project revision.
    pub project_revision: u64,
    /// Calculated histogram.
    pub histogram_id: String,
    /// Number of calculated samples.
    pub sample_count: usize,
    /// Number of structural phases.
    pub phase_count: usize,
    /// Unweighted profile residual when finite.
    pub rp: Option<f64>,
    /// Weighted profile residual when finite.
    pub rwp: Option<f64>,
    /// Chi-square when finite.
    pub chi_square: Option<f64>,
    /// Reduced chi-square when finite.
    pub reduced_chi_square: Option<f64>,
}

impl CalculationResponse {
    fn from_result(
        calculation_id: CalculationId,
        project_revision: u64,
        histogram_id: &str,
        result: &RietveldCalculation,
    ) -> Self {
        let finite = |value: f64| value.is_finite().then_some(value);
        Self {
            calculation_id,
            project_revision,
            histogram_id: histogram_id.to_owned(),
            sample_count: result.y.len(),
            phase_count: result.phases.len(),
            rp: finite(result.metrics.rp),
            rwp: finite(result.metrics.rwp),
            chi_square: finite(result.metrics.chi_square),
            reduced_chi_square: finite(result.metrics.reduced_chi_square),
        }
    }
}

struct CalculationRecord {
    source: ProjectSnapshot,
    histogram_id: RecordId,
    result: Arc<RietveldCalculation>,
}

/// Owns completed standalone calculations independently of project mutation.
#[derive(Clone)]
pub struct CalculationManager {
    projects: DesktopProjectStore,
    calculations: Arc<Mutex<BTreeMap<CalculationId, CalculationRecord>>>,
    next_calculation_id: Arc<AtomicU64>,
}

impl CalculationManager {
    /// Construct an empty retained-calculation manager.
    #[must_use]
    pub fn new(projects: DesktopProjectStore) -> Self {
        Self {
            projects,
            calculations: Arc::new(Mutex::new(BTreeMap::new())),
            next_calculation_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Calculate one histogram without mutating its project snapshot.
    ///
    /// The eventual Tauri command runs this blocking numerical operation on its
    /// native blocking pool. The retained result stays explicitly owned by its
    /// source revision even when later project edits occur.
    ///
    /// # Errors
    ///
    /// Returns a project/revision, phase-capability, execution, calculation, ID,
    /// or shared-state error.
    pub fn calculate_histogram(
        &self,
        expected_revision: u64,
        histogram_id: &str,
        options: CalculationOptionsInput,
    ) -> Result<CalculationResponse, DesktopError> {
        let source = self.projects.snapshot()?;
        if source.revision() != expected_revision {
            return Err(DesktopError::conflict(expected_revision, source.revision()));
        }
        let histogram_id =
            RecordId::new(histogram_id).map_err(|error| calculation_error(error.to_string()))?;
        let input = calculation_input(&source, &histogram_id)?;
        let result = Arc::new(
            calculate_rietveld_pattern(&input, &options.native()?)
                .map_err(|error| calculation_error(error.to_string()))?,
        );
        let calculation_id = self
            .next_calculation_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                DesktopError::simple(
                    DesktopErrorCode::RevisionOverflow,
                    "calculation ID space exhausted",
                )
            })?;
        let response = CalculationResponse::from_result(
            calculation_id,
            source.revision(),
            histogram_id.as_str(),
            &result,
        );
        self.lock()?.insert(
            calculation_id,
            CalculationRecord {
                source,
                histogram_id,
                result,
            },
        );
        Ok(response)
    }

    /// List binary display series for one retained calculation.
    ///
    /// # Errors
    ///
    /// Returns an unknown calculation, invalid retained state, size, or lock error.
    pub fn calculation_series(
        &self,
        calculation_id: CalculationId,
    ) -> Result<Vec<BinarySeriesDescriptor>, DesktopError> {
        let (source, histogram_id, result) = self.retained_result(calculation_id)?;
        let pattern = histogram_pattern(&source, &histogram_id)?;
        series::calculation_descriptors(
            calculation_id,
            source.revision(),
            histogram_id.as_str(),
            pattern,
            &result,
        )
    }

    /// Encode one retained calculation series as an owned binary payload.
    ///
    /// # Errors
    ///
    /// Returns an unknown calculation/series, invalid retained state, size, or lock error.
    pub fn calculation_series_payload(
        &self,
        calculation_id: CalculationId,
        series_id: &str,
    ) -> Result<BinaryPayload, DesktopError> {
        let (source, histogram_id, result) = self.retained_result(calculation_id)?;
        let pattern = histogram_pattern(&source, &histogram_id)?;
        series::calculation_payload(
            calculation_id,
            source.revision(),
            histogram_id.as_str(),
            pattern,
            &result,
            series_id,
        )
    }

    /// Release one retained standalone result and all of its plot arrays.
    ///
    /// # Errors
    ///
    /// Returns an unknown-calculation or shared-state error.
    pub fn discard_calculation(&self, calculation_id: CalculationId) -> Result<(), DesktopError> {
        if self.lock()?.remove(&calculation_id).is_none() {
            return Err(unknown_calculation(calculation_id));
        }
        Ok(())
    }

    fn lock(
        &self,
    ) -> Result<MutexGuard<'_, BTreeMap<CalculationId, CalculationRecord>>, DesktopError> {
        self.calculations.lock().map_err(|_| {
            DesktopError::simple(
                DesktopErrorCode::StateUnavailable,
                "calculation result state lock is poisoned",
            )
        })
    }

    fn retained_result(
        &self,
        calculation_id: CalculationId,
    ) -> Result<(ProjectSnapshot, RecordId, Arc<RietveldCalculation>), DesktopError> {
        let calculations = self.lock()?;
        let record = calculations
            .get(&calculation_id)
            .ok_or_else(|| unknown_calculation(calculation_id))?;
        Ok((
            record.source.clone(),
            record.histogram_id.clone(),
            Arc::clone(&record.result),
        ))
    }
}

fn calculation_input(
    source: &ProjectSnapshot,
    histogram_id: &RecordId,
) -> Result<RietveldInput, DesktopError> {
    let histogram = source
        .state()
        .project
        .histograms
        .iter()
        .find(|histogram| &histogram.histogram_id == histogram_id)
        .ok_or_else(|| calculation_error(format!("unknown histogram {histogram_id}")))?;
    let phases = histogram
        .phase_ids
        .iter()
        .map(|phase_id| {
            let phase = source
                .state()
                .project
                .phases
                .iter()
                .find(|phase| &phase.phase_id == phase_id)
                .ok_or_else(|| calculation_error(format!("unknown phase {phase_id}")))?;
            if !phase.required_providers.is_empty() {
                return Err(DesktopError::simple(
                    DesktopErrorCode::UnsupportedOperation,
                    format!("phase {phase_id} requires an unavailable external provider"),
                ));
            }
            RietveldPhase::new(
                phase.phase_id.clone(),
                phase.name.clone(),
                phase.definition.clone(),
                OwnedCwContributions::neutral(phase.definition.hkl.len()),
            )
            .map_err(|error| calculation_error(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pattern = histogram.pattern.clone();
    let input = RietveldInput::new(
        pattern.clone(),
        histogram.experiment.instrument,
        histogram.experiment.axial_geometry,
        histogram.experiment.position_correction,
        phases,
    )
    .map_err(|error| calculation_error(error.to_string()))?;
    Ok(input)
}

fn histogram_pattern<'a>(
    source: &'a ProjectSnapshot,
    histogram_id: &RecordId,
) -> Result<&'a PatternRecord, DesktopError> {
    source
        .state()
        .project
        .histograms
        .iter()
        .find(|histogram| &histogram.histogram_id == histogram_id)
        .map(|histogram| &histogram.pattern)
        .ok_or_else(|| DesktopError::host_failure("retained calculation lost its source histogram"))
}

fn unknown_calculation(calculation_id: CalculationId) -> DesktopError {
    DesktopError::simple(
        DesktopErrorCode::UnknownCalculation,
        format!("unknown standalone calculation {calculation_id}"),
    )
}

fn calculation_error(message: impl Into<String>) -> DesktopError {
    DesktopError::simple(DesktopErrorCode::Calculation, message)
}
