//! Stable JSON project summary records for desktop and scripting hosts.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use phasesmith_model::{ProjectRecord, RadiationProbe};
use serde::Serialize;

use crate::{PROJECT_FORMAT_VERSION, PersistenceError};

/// Project-summary report write behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectReportSaveOptions {
    /// Replace an existing report file when true.
    pub overwrite: bool,
}

/// One histogram entry in a project summary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HistogramSummary {
    /// Stable histogram ID.
    pub histogram_id: String,
    /// Human-readable label.
    pub name: String,
    /// Pattern sample count.
    pub sample_count: usize,
    /// Whether observed intensities are present.
    pub has_observations: bool,
    /// Durable coordinate convention: `two_theta_deg` or `tof_us`.
    pub coordinate_kind: String,
    /// X-ray or neutron probe.
    pub probe: String,
    /// Ordered active phase IDs.
    pub phase_ids: Vec<String>,
}

/// One phase entry in a project summary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PhaseSummary {
    /// Stable phase ID.
    pub phase_id: String,
    /// Human-readable label.
    pub name: String,
    /// Stored reflection-family count.
    pub reflection_count: usize,
    /// Asymmetric-site count.
    pub site_count: usize,
    /// Required external provider identifiers with versions.
    pub required_providers: Vec<String>,
}

/// Versioned, display-safe project summary without bulk numerical arrays.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProjectSummaryReport {
    /// Native project wire version summarized by this report.
    pub format_version: u32,
    /// Stable project ID.
    pub project_id: String,
    /// Snapshot revision.
    pub revision: u64,
    /// Human-readable project label.
    pub name: String,
    /// Number of histograms.
    pub histogram_count: usize,
    /// Number of project-owned phases.
    pub phase_count: usize,
    /// Total samples over all histograms.
    pub total_sample_count: usize,
    /// Histogram summaries in project order.
    pub histograms: Vec<HistogramSummary>,
    /// Phase summaries in project order.
    pub phases: Vec<PhaseSummary>,
    /// Project textual metadata.
    pub metadata: BTreeMap<String, String>,
}

impl ProjectSummaryReport {
    /// Build a stable report after recursively validating the project.
    ///
    /// # Errors
    ///
    /// Returns [`PersistenceError::Domain`] for invalid live project state or
    /// [`PersistenceError::InvalidRecord`] if sample counts overflow.
    pub fn from_project(project: &ProjectRecord) -> Result<Self, PersistenceError> {
        project.validate().map_err(PersistenceError::Domain)?;
        let total_sample_count = project
            .histograms
            .iter()
            .map(|histogram| histogram.pattern.sample_count())
            .chain(
                project
                    .tof_histograms
                    .iter()
                    .map(|histogram| histogram.pattern.sample_count()),
            )
            .try_fold(0_usize, |total, sample_count| {
                total
                    .checked_add(sample_count)
                    .ok_or_else(|| PersistenceError::InvalidRecord {
                        message: "project summary sample count overflow".to_owned(),
                    })
            })?;
        Ok(Self {
            format_version: PROJECT_FORMAT_VERSION,
            project_id: project.project_id.as_str().to_owned(),
            revision: project.revision,
            name: project.name.clone(),
            histogram_count: project.histograms.len() + project.tof_histograms.len(),
            phase_count: project.phases.len(),
            total_sample_count,
            histograms: project
                .histograms
                .iter()
                .map(|histogram| HistogramSummary {
                    histogram_id: histogram.histogram_id.as_str().to_owned(),
                    name: histogram.name.clone(),
                    sample_count: histogram.pattern.sample_count(),
                    has_observations: histogram.pattern.observed_y.is_some(),
                    coordinate_kind: "two_theta_deg".to_owned(),
                    probe: match histogram.experiment.radiation.probe() {
                        RadiationProbe::Xray => "xray",
                        RadiationProbe::Neutron => "neutron",
                    }
                    .to_owned(),
                    phase_ids: histogram
                        .phase_ids
                        .iter()
                        .map(|value| value.as_str().to_owned())
                        .collect(),
                })
                .chain(project.tof_histograms.iter().map(|histogram| {
                    HistogramSummary {
                        histogram_id: histogram.histogram_id.as_str().to_owned(),
                        name: histogram.name.clone(),
                        sample_count: histogram.pattern.sample_count(),
                        has_observations: histogram.pattern.observed_y.is_some(),
                        coordinate_kind: "tof_us".to_owned(),
                        probe: "neutron".to_owned(),
                        phase_ids: histogram
                            .phase_ids
                            .iter()
                            .map(|value| value.as_str().to_owned())
                            .collect(),
                    }
                }))
                .collect(),
            phases: project
                .phases
                .iter()
                .map(|phase| PhaseSummary {
                    phase_id: phase.phase_id.as_str().to_owned(),
                    name: phase.name.clone(),
                    reflection_count: phase.definition.hkl.len(),
                    site_count: phase.definition.fractional_xyz.len(),
                    required_providers: phase
                        .required_providers
                        .iter()
                        .map(|requirement| {
                            format!(
                                "{}@{}",
                                requirement.provider_id, requirement.provider_version
                            )
                        })
                        .collect(),
                })
                .collect(),
            metadata: project.metadata.clone(),
        })
    }
}

/// Serialize one validated summary to deterministic pretty JSON.
///
/// # Errors
///
/// Returns [`PersistenceError`] for invalid project state or serialization.
pub fn project_summary_json(project: &ProjectRecord) -> Result<String, PersistenceError> {
    let report = ProjectSummaryReport::from_project(project)?;
    let mut encoded = serde_json::to_string_pretty(&report)?;
    encoded.push('\n');
    Ok(encoded)
}

/// Write one validated project summary as JSON.
///
/// # Errors
///
/// Returns [`PersistenceError`] for validation, serialization, or filesystem
/// failures.
pub fn write_project_summary_json(
    project: &ProjectRecord,
    path: impl AsRef<Path>,
) -> Result<PathBuf, PersistenceError> {
    write_project_summary_json_with_options(
        project,
        path,
        ProjectReportSaveOptions { overwrite: true },
    )
}

/// Write one validated project summary with an explicit overwrite policy.
///
/// # Errors
///
/// Returns [`PersistenceError`] for validation, serialization, destination, or
/// filesystem failures.
pub fn write_project_summary_json_with_options(
    project: &ProjectRecord,
    path: impl AsRef<Path>,
    options: ProjectReportSaveOptions,
) -> Result<PathBuf, PersistenceError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let encoded = project_summary_json(project)?;
    if options.overwrite {
        fs::write(path, encoded)?;
    } else {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    PersistenceError::InvalidDestination {
                        message: format!("report destination already exists: {}", path.display()),
                    }
                } else {
                    PersistenceError::Io(error)
                }
            })?
            .write_all(encoded.as_bytes())?;
    }
    Ok(path.to_owned())
}
