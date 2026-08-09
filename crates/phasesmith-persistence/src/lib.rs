//! Canonical, Python-free project persistence and stable summary reporting.
//!
//! Project bundles are directories containing `manifest.json` plus a numeric
//! `arrays.npz`. Wire records are explicit and versioned; live domain types do
//! not derive serialization directly.

mod arrays;
mod report;
mod rietveld_wire;
mod wire;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arrays::{ArrayDescriptor, read_npz, sha256_hex, write_npz};
use phasesmith_model::{DomainError, ProjectRecord};
use phasesmith_workflows::RietveldProjectState;
use serde::{Deserialize, Serialize};

pub use report::{
    HistogramSummary, PhaseSummary, ProjectSummaryReport, project_summary_json,
    write_project_summary_json,
};

/// Current native project bundle wire version.
pub const PROJECT_FORMAT_VERSION: u32 = 2;
/// Canonical manifest filename within a project directory.
pub const PROJECT_MANIFEST_NAME: &str = "manifest.json";
/// Canonical `NumPy` archive filename within a project directory.
pub const PROJECT_ARRAYS_NAME: &str = "arrays.npz";

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Resource limits applied before allocating project records and arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectReadLimits {
    /// Maximum manifest byte count.
    pub max_manifest_bytes: u64,
    /// Maximum compressed NPZ byte count.
    pub max_archive_bytes: u64,
    /// Maximum number of declared arrays.
    pub max_arrays: usize,
    /// Maximum element count of one array.
    pub max_array_elements: usize,
    /// Maximum combined uncompressed NPY bytes.
    pub max_uncompressed_array_bytes: u64,
    /// Maximum histogram count.
    pub max_histograms: usize,
    /// Maximum structural phase count.
    pub max_phases: usize,
}

impl Default for ProjectReadLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 16 * 1024 * 1024,
            max_archive_bytes: 2 * 1024 * 1024 * 1024,
            max_arrays: 10_000,
            max_array_elements: 100_000_000,
            max_uncompressed_array_bytes: 4 * 1024 * 1024 * 1024,
            max_histograms: 10_000,
            max_phases: 10_000,
        }
    }
}

impl ProjectReadLimits {
    fn validate(self) -> Result<(), PersistenceError> {
        if self.max_manifest_bytes == 0
            || self.max_archive_bytes == 0
            || self.max_arrays == 0
            || self.max_array_elements == 0
            || self.max_uncompressed_array_bytes == 0
            || self.max_histograms == 0
            || self.max_phases == 0
        {
            return Err(PersistenceError::InvalidLimits);
        }
        Ok(())
    }
}

/// Native project save behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectSaveOptions {
    /// Replace only the two library-owned files in an existing directory.
    pub overwrite: bool,
}

/// Structured native persistence failure.
#[derive(Debug)]
pub enum PersistenceError {
    /// A configured read limit is zero.
    InvalidLimits,
    /// Input exceeded an explicit resource limit.
    LimitExceeded {
        /// Stable explanation naming the limit.
        message: String,
    },
    /// A path has the wrong kind or overwrite policy.
    InvalidDestination {
        /// Stable explanation.
        message: String,
    },
    /// Filesystem operation failed.
    Io(std::io::Error),
    /// Manifest JSON syntax or shape is invalid.
    Json(serde_json::Error),
    /// Unsupported project format version.
    UnsupportedVersion {
        /// Rejected version.
        version: u32,
    },
    /// Array descriptor, shape, dtype, or value is invalid.
    InvalidArray {
        /// Stable explanation.
        message: String,
    },
    /// NPZ/NPY archive structure or hashes are invalid.
    InvalidArchive {
        /// Stable explanation.
        message: String,
    },
    /// Wire record is invalid or inconsistent.
    InvalidRecord {
        /// Stable explanation.
        message: String,
    },
    /// Reconstructed domain project failed validation.
    Domain(DomainError),
}

impl Display for PersistenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("all project read limits must be positive"),
            Self::LimitExceeded { message }
            | Self::InvalidDestination { message }
            | Self::InvalidArray { message }
            | Self::InvalidArchive { message }
            | Self::InvalidRecord { message } => formatter.write_str(message),
            Self::Io(error) => Display::fmt(error, formatter),
            Self::Json(error) => Display::fmt(error, formatter),
            Self::UnsupportedVersion { version } => {
                write!(formatter, "unsupported native project format {version}")
            }
            Self::Domain(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for PersistenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Domain(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for PersistenceError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchiveRecord {
    file: String,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectManifest {
    format_version: u32,
    archive: ArchiveRecord,
    arrays: BTreeMap<String, ArrayDescriptor>,
    project: wire::WireProject,
    #[serde(default)]
    rietveld_analyses: Option<Vec<rietveld_wire::WireRietveldAnalysis>>,
}

/// Save one validated native project as canonical JSON plus NPZ.
///
/// Only `manifest.json` and `arrays.npz` are created or replaced; unrelated
/// files in an existing directory remain untouched.
///
/// # Errors
///
/// Returns [`PersistenceError`] for invalid domain state, unsupported values,
/// serialization, archive, filesystem, or overwrite failures.
pub fn save_project(
    path: impl AsRef<Path>,
    project: &ProjectRecord,
    options: ProjectSaveOptions,
) -> Result<PathBuf, PersistenceError> {
    project.validate().map_err(PersistenceError::Domain)?;
    save_project_parts(path.as_ref(), project, Vec::new(), options)
}

/// Save one validated project and all runnable native Rietveld analyses.
///
/// # Errors
///
/// Returns [`PersistenceError`] for invalid cross-record state, serialization,
/// archive, filesystem, or overwrite failures.
pub fn save_rietveld_project(
    path: impl AsRef<Path>,
    state: &RietveldProjectState,
    options: ProjectSaveOptions,
) -> Result<PathBuf, PersistenceError> {
    state
        .validate()
        .map_err(|error| PersistenceError::InvalidRecord {
            message: format!("invalid native Rietveld project state: {error}"),
        })?;
    save_project_parts(
        path.as_ref(),
        &state.project,
        rietveld_wire::encode_analyses(state),
        options,
    )
}

fn save_project_parts(
    path: &Path,
    project: &ProjectRecord,
    rietveld_analyses: Vec<rietveld_wire::WireRietveldAnalysis>,
    options: ProjectSaveOptions,
) -> Result<PathBuf, PersistenceError> {
    let destination = absolute_path(path)?;
    validate_destination(&destination, options)?;
    let (wire_project, arrays) = wire::encode_project(project)?;
    let encoded_archive = write_npz(&arrays)?;
    let descriptors = arrays
        .iter()
        .map(|(name, value)| (name.clone(), value.descriptor()))
        .collect();
    let manifest = ProjectManifest {
        format_version: PROJECT_FORMAT_VERSION,
        archive: ArchiveRecord {
            file: PROJECT_ARRAYS_NAME.to_owned(),
            sha256: sha256_hex(&encoded_archive),
        },
        arrays: descriptors,
        project: wire_project,
        rietveld_analyses: Some(rietveld_analyses),
    };
    let mut encoded_manifest = serde_json::to_string_pretty(&manifest)?;
    encoded_manifest.push('\n');

    let parent = destination
        .parent()
        .ok_or_else(|| PersistenceError::InvalidDestination {
            message: "project destination has no parent directory".to_owned(),
        })?;
    fs::create_dir_all(parent)?;
    let temporary = create_temporary_directory(parent, &destination)?;
    let write_result: Result<(), PersistenceError> = (|| {
        fs::write(temporary.join(PROJECT_ARRAYS_NAME), encoded_archive)?;
        fs::write(temporary.join(PROJECT_MANIFEST_NAME), encoded_manifest)?;
        fs::create_dir_all(&destination)?;
        replace_owned_file(
            &temporary.join(PROJECT_ARRAYS_NAME),
            &destination.join(PROJECT_ARRAYS_NAME),
            options.overwrite,
        )?;
        replace_owned_file(
            &temporary.join(PROJECT_MANIFEST_NAME),
            &destination.join(PROJECT_MANIFEST_NAME),
            options.overwrite,
        )?;
        Ok(())
    })();
    let _ = fs::remove_dir_all(&temporary);
    write_result?;
    Ok(destination)
}

/// Load and fully validate one canonical native project directory.
///
/// # Errors
///
/// Returns [`PersistenceError`] for resource, filesystem, JSON, hash, archive,
/// wire-record, or domain validation failures.
pub fn load_project(
    path: impl AsRef<Path>,
    limits: ProjectReadLimits,
) -> Result<ProjectRecord, PersistenceError> {
    load_rietveld_project(path, limits).map(|state| state.project)
}

/// Load and validate one project plus all native Rietveld analyses.
///
/// Version-1 native projects load with an empty analysis list.
///
/// # Errors
///
/// Returns [`PersistenceError`] for resource, filesystem, JSON, hash, archive,
/// wire-record, domain, or Rietveld validation failures.
pub fn load_rietveld_project(
    path: impl AsRef<Path>,
    limits: ProjectReadLimits,
) -> Result<RietveldProjectState, PersistenceError> {
    limits.validate()?;
    let source = absolute_path(path.as_ref())?;
    let manifest_path = source.join(PROJECT_MANIFEST_NAME);
    let archive_path = source.join(PROJECT_ARRAYS_NAME);
    let manifest_bytes = read_bounded_file(
        &manifest_path,
        limits.max_manifest_bytes,
        "project manifest exceeds max_manifest_bytes",
    )?;
    let manifest: ProjectManifest = serde_json::from_slice(&manifest_bytes)?;
    if !(1..=PROJECT_FORMAT_VERSION).contains(&manifest.format_version) {
        return Err(PersistenceError::UnsupportedVersion {
            version: manifest.format_version,
        });
    }
    let rietveld_analyses = match (manifest.format_version, manifest.rietveld_analyses) {
        (1, None) => Vec::new(),
        (1, Some(_)) => {
            return Err(PersistenceError::InvalidRecord {
                message: "native project format 1 cannot declare Rietveld analyses".to_owned(),
            });
        }
        (_, Some(analyses)) => analyses,
        (_, None) => {
            return Err(PersistenceError::InvalidRecord {
                message: "native project format 2 requires Rietveld analyses".to_owned(),
            });
        }
    };
    if manifest.archive.file != PROJECT_ARRAYS_NAME {
        return Err(PersistenceError::InvalidArchive {
            message: "project archive filename is invalid".to_owned(),
        });
    }
    if manifest.arrays.len() > limits.max_arrays {
        return Err(PersistenceError::LimitExceeded {
            message: "project manifest exceeds max_arrays".to_owned(),
        });
    }
    let archive_bytes = read_bounded_file(
        &archive_path,
        limits.max_archive_bytes,
        "project archive exceeds max_archive_bytes",
    )?;
    if sha256_hex(&archive_bytes) != manifest.archive.sha256 {
        return Err(PersistenceError::InvalidArchive {
            message: "project archive SHA-256 mismatch".to_owned(),
        });
    }
    let arrays = read_npz(&archive_bytes, &manifest.arrays, limits)?;
    let project = wire::decode_project(manifest.project, arrays, limits)?;
    rietveld_wire::decode_state(project, rietveld_analyses, limits)
}

fn read_bounded_file(
    path: &Path,
    maximum_bytes: u64,
    limit_message: &str,
) -> Result<Vec<u8>, PersistenceError> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_or(true, |length| length > maximum_bytes) {
        return Err(PersistenceError::LimitExceeded {
            message: limit_message.to_owned(),
        });
    }
    Ok(bytes)
}

fn absolute_path(path: &Path) -> Result<PathBuf, PersistenceError> {
    if path.is_absolute() {
        return Ok(path.to_owned());
    }
    Ok(std::env::current_dir()?.join(path))
}

fn validate_destination(
    destination: &Path,
    options: ProjectSaveOptions,
) -> Result<(), PersistenceError> {
    if destination.exists() && !destination.is_dir() {
        return Err(PersistenceError::InvalidDestination {
            message: format!(
                "project path exists and is not a directory: {}",
                destination.display()
            ),
        });
    }
    if destination.exists() && !options.overwrite {
        return Err(PersistenceError::InvalidDestination {
            message: format!(
                "project directory already exists: {}",
                destination.display()
            ),
        });
    }
    Ok(())
}

fn create_temporary_directory(
    parent: &Path,
    destination: &Path,
) -> Result<PathBuf, PersistenceError> {
    let stem = destination
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("project");
    for _ in 0..100 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{stem}-{}-{sequence}.tmp", std::process::id()));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(PersistenceError::Io(error)),
        }
    }
    Err(PersistenceError::InvalidDestination {
        message: "could not allocate a temporary project directory".to_owned(),
    })
}

fn replace_owned_file(
    source: &Path,
    destination: &Path,
    overwrite: bool,
) -> Result<(), PersistenceError> {
    if destination.exists() {
        if !overwrite {
            return Err(PersistenceError::InvalidDestination {
                message: format!("project file already exists: {}", destination.display()),
            });
        }
        fs::remove_file(destination)?;
    }
    fs::rename(source, destination)?;
    Ok(())
}
