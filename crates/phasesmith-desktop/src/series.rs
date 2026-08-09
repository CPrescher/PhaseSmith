//! Typed binary display-series descriptors and little-endian encoding.

use phasesmith_model::HistogramRecord;
use phasesmith_workflows::{RietveldGeneralRefinementResult, RietveldProjectState};
use serde::Serialize;

use crate::{DesktopError, DesktopErrorCode, JobId};

/// Wire dtype for one binary display series.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesDtype {
    /// IEEE-754 binary64 encoded in little-endian byte order.
    Float64Le,
    /// One unsigned byte per value.
    Uint8,
}

impl SeriesDtype {
    const fn byte_width(self) -> usize {
        match self {
            Self::Float64Le => 8,
            Self::Uint8 => 1,
        }
    }
}

/// Immutable source identity for a binary series.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SeriesOwner {
    /// Series belongs to the currently open project snapshot.
    Project {
        /// Project revision that owns the bytes.
        project_revision: u64,
        /// Histogram that owns the series.
        histogram_id: String,
    },
    /// Series belongs to one retained refinement result.
    RefinementJob {
        /// Process-local job ID.
        job_id: JobId,
        /// Immutable project revision evaluated by the job.
        project_revision: u64,
        /// Histogram that was refined.
        histogram_id: String,
    },
}

/// JSON-safe descriptor for a separately transferred numerical series.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BinarySeriesDescriptor {
    /// Stable series ID within its owner.
    pub series_id: String,
    /// Human-readable series label.
    pub label: String,
    /// Semantic role such as coordinate, observed, calculated, or residual.
    pub role: String,
    /// Physical unit carried by each value, or `unitless`.
    pub unit: String,
    /// Binary scalar encoding.
    pub dtype: SeriesDtype,
    /// Number of scalar values.
    pub length: usize,
    /// Exact encoded byte length.
    pub byte_length: usize,
    /// Snapshot/job owner used for cache invalidation.
    pub owner: SeriesOwner,
    /// Optional phase ID for phase-specific sample/reflection series.
    pub phase_id: Option<String>,
}

/// One descriptor plus bytes intended for an IPC binary response body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryPayload {
    descriptor: BinarySeriesDescriptor,
    bytes: Vec<u8>,
}

impl BinaryPayload {
    /// Borrow the JSON-safe descriptor associated with these bytes.
    #[must_use]
    pub const fn descriptor(&self) -> &BinarySeriesDescriptor {
        &self.descriptor
    }

    /// Borrow the exact little-endian payload bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the payload and return bytes for a host-native binary response.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

struct SeriesRef<'a> {
    descriptor: BinarySeriesDescriptor,
    values: SeriesValues<'a>,
}

enum SeriesValues<'a> {
    F64(&'a [f64]),
    Bool(&'a [bool]),
}

impl SeriesRef<'_> {
    fn into_payload(self) -> BinaryPayload {
        let bytes = match self.values {
            SeriesValues::F64(values) => encode_f64(values),
            SeriesValues::Bool(values) => values.iter().map(|value| u8::from(*value)).collect(),
        };
        debug_assert_eq!(bytes.len(), self.descriptor.byte_length);
        BinaryPayload {
            descriptor: self.descriptor,
            bytes,
        }
    }
}

pub(crate) fn project_descriptors(
    state: &RietveldProjectState,
    histogram_id: &str,
) -> Result<Vec<BinarySeriesDescriptor>, DesktopError> {
    Ok(project_series(state, histogram_id)?
        .into_iter()
        .map(|series| series.descriptor)
        .collect())
}

pub(crate) fn project_payload(
    state: &RietveldProjectState,
    histogram_id: &str,
    series_id: &str,
) -> Result<BinaryPayload, DesktopError> {
    project_series(state, histogram_id)?
        .into_iter()
        .find(|series| series.descriptor.series_id == series_id)
        .map(SeriesRef::into_payload)
        .ok_or_else(|| unknown_series(series_id))
}

pub(crate) fn refinement_descriptors(
    job_id: JobId,
    project_revision: u64,
    histogram_id: &str,
    result: &RietveldGeneralRefinementResult,
) -> Result<Vec<BinarySeriesDescriptor>, DesktopError> {
    Ok(
        refinement_series(job_id, project_revision, histogram_id, result)?
            .into_iter()
            .map(|series| series.descriptor)
            .collect(),
    )
}

pub(crate) fn refinement_payload(
    job_id: JobId,
    project_revision: u64,
    histogram_id: &str,
    result: &RietveldGeneralRefinementResult,
    series_id: &str,
) -> Result<BinaryPayload, DesktopError> {
    refinement_series(job_id, project_revision, histogram_id, result)?
        .into_iter()
        .find(|series| series.descriptor.series_id == series_id)
        .map(SeriesRef::into_payload)
        .ok_or_else(|| unknown_series(series_id))
}

fn project_series<'a>(
    state: &'a RietveldProjectState,
    histogram_id: &str,
) -> Result<Vec<SeriesRef<'a>>, DesktopError> {
    let histogram = histogram(state, histogram_id)?;
    let owner = SeriesOwner::Project {
        project_revision: state.project.revision,
        histogram_id: histogram_id.to_owned(),
    };
    let mut result = Vec::with_capacity(5);
    push_f64(
        &mut result,
        owner.clone(),
        "x_deg",
        "2θ",
        "coordinate",
        "degree",
        &histogram.pattern.x_deg,
        None,
    )?;
    if let Some(values) = &histogram.pattern.observed_y {
        push_f64(
            &mut result,
            owner.clone(),
            "observed_y",
            "Observed",
            "observed",
            "counts",
            values,
            None,
        )?;
    }
    if let Some(values) = &histogram.pattern.uncertainty {
        push_f64(
            &mut result,
            owner.clone(),
            "uncertainty",
            "Uncertainty",
            "uncertainty",
            "counts",
            values,
            None,
        )?;
    }
    if let Some(values) = &histogram.pattern.mask {
        push_bool(
            &mut result,
            owner.clone(),
            "included_mask",
            "Included mask",
            "mask",
            values,
        )?;
    }
    push_f64(
        &mut result,
        owner,
        "fixed_background_y",
        "Fixed background",
        "background",
        "counts",
        &histogram.pattern.background_y,
        None,
    )?;
    Ok(result)
}

fn refinement_series<'a>(
    job_id: JobId,
    project_revision: u64,
    histogram_id: &str,
    result: &'a RietveldGeneralRefinementResult,
) -> Result<Vec<SeriesRef<'a>>, DesktopError> {
    let owner = SeriesOwner::RefinementJob {
        job_id,
        project_revision,
        histogram_id: histogram_id.to_owned(),
    };
    let mut series = Vec::with_capacity(8 + result.calculation.phases.len() * 3);
    push_f64(
        &mut series,
        owner.clone(),
        "x_deg",
        "2θ",
        "coordinate",
        "degree",
        &result.input.pattern.x_deg,
        None,
    )?;
    if let Some(values) = &result.input.pattern.observed_y {
        push_f64(
            &mut series,
            owner.clone(),
            "observed_y",
            "Observed",
            "observed",
            "counts",
            values,
            None,
        )?;
    }
    for (series_id, label, role, values) in [
        (
            "calculated_y",
            "Calculated",
            "calculated",
            result.calculation.y.as_slice(),
        ),
        (
            "profile_y",
            "Profile",
            "profile",
            result.calculation.profile_y.as_slice(),
        ),
        (
            "background_y",
            "Background",
            "background",
            result.calculation.background_y.as_slice(),
        ),
        (
            "residual_y",
            "Residual",
            "residual",
            result.calculation.metrics.residual.as_slice(),
        ),
        (
            "weighted_residual_y",
            "Weighted residual",
            "weighted_residual",
            result.calculation.metrics.weighted_residual.as_slice(),
        ),
    ] {
        push_f64(
            &mut series,
            owner.clone(),
            series_id,
            label,
            role,
            "counts",
            values,
            None,
        )?;
    }
    push_bool(
        &mut series,
        owner.clone(),
        "included_mask",
        "Included mask",
        "mask",
        &result.calculation.metrics.included,
    )?;
    for phase in &result.calculation.phases {
        push_phase_series(&mut series, owner.clone(), phase)?;
    }
    Ok(series)
}

fn push_phase_series<'a>(
    series: &mut Vec<SeriesRef<'a>>,
    owner: SeriesOwner,
    phase: &'a phasesmith_workflows::RietveldPhaseCalculation,
) -> Result<(), DesktopError> {
    let phase_id = phase.phase_id.as_str();
    push_f64(
        series,
        owner.clone(),
        &format!("phase/{phase_id}/profile_y"),
        &format!("{} profile", phase.name),
        "phase_profile",
        "counts",
        &phase.result.accumulation.y,
        Some(phase_id),
    )?;
    push_f64(
        series,
        owner.clone(),
        &format!("phase/{phase_id}/two_theta_deg"),
        &format!("{} reflection positions", phase.name),
        "reflection_position",
        "degree",
        &phase.result.two_theta_deg,
        Some(phase_id),
    )?;
    push_f64(
        series,
        owner,
        &format!("phase/{phase_id}/intensity"),
        &format!("{} reflection intensities", phase.name),
        "reflection_intensity",
        "counts",
        &phase.result.structure_factors.intensity,
        Some(phase_id),
    )
}

fn histogram<'a>(
    state: &'a RietveldProjectState,
    histogram_id: &str,
) -> Result<&'a HistogramRecord, DesktopError> {
    state
        .project
        .histograms
        .iter()
        .find(|histogram| histogram.histogram_id.as_str() == histogram_id)
        .ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::UnknownAnalysis,
                format!("project histogram {histogram_id:?} does not exist"),
            )
        })
}

#[allow(clippy::too_many_arguments)]
fn push_f64<'a>(
    series: &mut Vec<SeriesRef<'a>>,
    owner: SeriesOwner,
    series_id: &str,
    label: &str,
    role: &str,
    unit: &str,
    values: &'a [f64],
    phase_id: Option<&str>,
) -> Result<(), DesktopError> {
    let descriptor = descriptor(
        owner,
        series_id,
        label,
        role,
        unit,
        SeriesDtype::Float64Le,
        values.len(),
        phase_id,
    )?;
    series.push(SeriesRef {
        descriptor,
        values: SeriesValues::F64(values),
    });
    Ok(())
}

fn push_bool<'a>(
    series: &mut Vec<SeriesRef<'a>>,
    owner: SeriesOwner,
    series_id: &str,
    label: &str,
    role: &str,
    values: &'a [bool],
) -> Result<(), DesktopError> {
    let descriptor = descriptor(
        owner,
        series_id,
        label,
        role,
        "unitless",
        SeriesDtype::Uint8,
        values.len(),
        None,
    )?;
    series.push(SeriesRef {
        descriptor,
        values: SeriesValues::Bool(values),
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn descriptor(
    owner: SeriesOwner,
    series_id: &str,
    label: &str,
    role: &str,
    unit: &str,
    dtype: SeriesDtype,
    length: usize,
    phase_id: Option<&str>,
) -> Result<BinarySeriesDescriptor, DesktopError> {
    let byte_length = length.checked_mul(dtype.byte_width()).ok_or_else(|| {
        DesktopError::simple(
            DesktopErrorCode::StateUnavailable,
            format!("binary series {series_id:?} byte length overflows"),
        )
    })?;
    Ok(BinarySeriesDescriptor {
        series_id: series_id.to_owned(),
        label: label.to_owned(),
        role: role.to_owned(),
        unit: unit.to_owned(),
        dtype,
        length,
        byte_length,
        owner,
        phase_id: phase_id.map(str::to_owned),
    })
}

fn encode_f64(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn unknown_series(series_id: &str) -> DesktopError {
    DesktopError::simple(
        DesktopErrorCode::UnknownSeries,
        format!("unknown binary display series {series_id:?}"),
    )
}
