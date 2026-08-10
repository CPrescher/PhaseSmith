use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

/// Stable outcome used by validation checks and reports.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationStatus {
    /// Every required acceptance condition was satisfied.
    Passed,
    /// At least one acceptance condition was evaluated and failed.
    Failed,
    /// Validation could not complete because a required capability or input was unavailable.
    Blocked,
}

impl ValidationStatus {
    /// Return the stable lowercase wire value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::Blocked => "blocked",
        }
    }
}

/// One machine-readable real-data acceptance check.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ValidationCheck {
    /// Stable identifier within one validation workflow.
    pub check_id: String,
    /// Outcome of this check.
    pub status: ValidationStatus,
    /// Human-readable description of what was checked.
    pub detail: String,
    /// Optional finite measured value.
    pub measured: Option<f64>,
    /// Optional human-readable acceptance criterion.
    pub criterion: Option<String>,
}

impl ValidationCheck {
    /// Construct and validate one check.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationContractError`] for empty text or a non-finite measurement.
    pub fn new(
        check_id: impl Into<String>,
        status: ValidationStatus,
        detail: impl Into<String>,
        measured: Option<f64>,
        criterion: Option<String>,
    ) -> Result<Self, ValidationContractError> {
        let result = Self {
            check_id: check_id.into(),
            status,
            detail: detail.into(),
            measured,
            criterion,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate a decoded or caller-mutated check.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationContractError`] for empty text or a non-finite measurement.
    pub fn validate(&self) -> Result<(), ValidationContractError> {
        if self.check_id.is_empty() || self.detail.is_empty() {
            return Err(ValidationContractError::EmptyCheckText);
        }
        if !self
            .check_id
            .bytes()
            .all(|value| value.is_ascii_lowercase() || value.is_ascii_digit() || value == b'_')
        {
            return Err(ValidationContractError::InvalidCheckId);
        }
        if self.criterion.as_ref().is_none_or(String::is_empty) {
            return Err(ValidationContractError::EmptyCriterion);
        }
        if self.measured.is_some_and(|value| !value.is_finite()) {
            return Err(ValidationContractError::NonFiniteMeasurement);
        }
        Ok(())
    }
}

/// Stable result envelope for one external validation workflow.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RealDataValidationReport {
    /// Stable registered dataset identifier.
    pub dataset_id: String,
    /// Summary derived from all check statuses.
    pub status: ValidationStatus,
    /// Number of selected observed samples.
    pub sample_count: usize,
    /// Optional number of generated reflections.
    pub reflection_count: Option<usize>,
    /// Diagnostic host time in seconds; never an acceptance threshold.
    pub elapsed_seconds: f64,
    /// Non-empty ordered acceptance checks.
    pub checks: Vec<ValidationCheck>,
    /// Ordered non-empty diagnostic notes.
    pub notes: Vec<String>,
}

impl RealDataValidationReport {
    /// Construct a report and derive its summary status from its checks.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationContractError`] for invalid counts, timing, checks, or notes.
    pub fn new(
        dataset_id: impl Into<String>,
        sample_count: usize,
        reflection_count: Option<usize>,
        elapsed_seconds: f64,
        checks: Vec<ValidationCheck>,
        notes: Vec<String>,
    ) -> Result<Self, ValidationContractError> {
        let status = summarize(&checks);
        let result = Self {
            dataset_id: dataset_id.into(),
            status,
            sample_count,
            reflection_count,
            elapsed_seconds,
            checks,
            notes,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate a decoded or caller-mutated report.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationContractError`] when the report violates the stable contract.
    pub fn validate(&self) -> Result<(), ValidationContractError> {
        if self.dataset_id.is_empty() {
            return Err(ValidationContractError::EmptyDatasetId);
        }
        if self.sample_count == 0 {
            return Err(ValidationContractError::ZeroSampleCount);
        }
        if self.reflection_count == Some(0) {
            return Err(ValidationContractError::ZeroReflectionCount);
        }
        if !self.elapsed_seconds.is_finite() || self.elapsed_seconds < 0.0 {
            return Err(ValidationContractError::InvalidElapsedSeconds);
        }
        if self.checks.is_empty() {
            return Err(ValidationContractError::EmptyChecks);
        }
        for check in &self.checks {
            check.validate()?;
        }
        for (index, check) in self.checks.iter().enumerate() {
            if self.checks[..index]
                .iter()
                .any(|previous| previous.check_id == check.check_id)
            {
                return Err(ValidationContractError::DuplicateCheckId);
            }
        }
        if self.notes.iter().any(String::is_empty) {
            return Err(ValidationContractError::EmptyNote);
        }
        if self.status != summarize(&self.checks) {
            return Err(ValidationContractError::InconsistentStatus);
        }
        Ok(())
    }

    /// Encode compact deterministic JSON using round-trip-safe float formatting.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationContractError`] for an invalid report or JSON failure.
    pub fn to_json(&self) -> Result<String, ValidationContractError> {
        self.validate()?;
        serde_json::to_string(self).map_err(Into::into)
    }

    /// Encode pretty deterministic JSON using round-trip-safe float formatting.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationContractError`] for an invalid report or JSON failure.
    pub fn to_pretty_json(&self) -> Result<String, ValidationContractError> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Decode and validate a JSON report.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationContractError`] for malformed JSON or an invalid report.
    pub fn from_json(source: &str) -> Result<Self, ValidationContractError> {
        let result: Self = serde_json::from_str(source)?;
        result.validate()?;
        Ok(result)
    }
}

fn summarize(checks: &[ValidationCheck]) -> ValidationStatus {
    if checks
        .iter()
        .any(|check| check.status == ValidationStatus::Failed)
    {
        ValidationStatus::Failed
    } else if checks
        .iter()
        .any(|check| check.status == ValidationStatus::Blocked)
    {
        ValidationStatus::Blocked
    } else {
        ValidationStatus::Passed
    }
}

/// Invalid validation report/check contract.
#[derive(Debug)]
pub enum ValidationContractError {
    /// A check identifier or detail was empty.
    EmptyCheckText,
    /// A check identifier was not stable lowercase snake case.
    InvalidCheckId,
    /// A supplied acceptance criterion was empty.
    EmptyCriterion,
    /// A measured value was not finite.
    NonFiniteMeasurement,
    /// The dataset identifier was empty.
    EmptyDatasetId,
    /// A report contained no selected samples.
    ZeroSampleCount,
    /// A present reflection count was zero.
    ZeroReflectionCount,
    /// Diagnostic elapsed time was negative or not finite.
    InvalidElapsedSeconds,
    /// A report contained no checks.
    EmptyChecks,
    /// A report contained the same check identifier more than once.
    DuplicateCheckId,
    /// A diagnostic note was empty.
    EmptyNote,
    /// The report summary did not match its check statuses.
    InconsistentStatus,
    /// JSON encoding or decoding failed.
    Json(serde_json::Error),
}

impl Display for ValidationContractError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCheckText => {
                formatter.write_str("validation check ID and detail must not be empty")
            }
            Self::InvalidCheckId => {
                formatter.write_str("validation check ID must be lowercase snake case")
            }
            Self::EmptyCriterion => {
                formatter.write_str("validation criterion must be present and non-empty")
            }
            Self::NonFiniteMeasurement => {
                formatter.write_str("validation measurement must be finite")
            }
            Self::EmptyDatasetId => formatter.write_str("validation dataset ID must not be empty"),
            Self::ZeroSampleCount => formatter.write_str("validation requires observed samples"),
            Self::ZeroReflectionCount => formatter.write_str("reflection count must be positive"),
            Self::InvalidElapsedSeconds => {
                formatter.write_str("elapsed seconds must be finite and nonnegative")
            }
            Self::EmptyChecks => {
                formatter.write_str("validation report requires at least one check")
            }
            Self::DuplicateCheckId => {
                formatter.write_str("validation report check IDs must be unique")
            }
            Self::EmptyNote => formatter.write_str("validation notes must not be empty"),
            Self::InconsistentStatus => {
                formatter.write_str("report status does not summarize its checks")
            }
            Self::Json(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for ValidationContractError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for ValidationContractError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
