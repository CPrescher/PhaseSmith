use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One externally hosted file with immutable retrieval provenance.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExternalValidationFile {
    /// Safe basename expected below the dataset directory.
    pub name: String,
    /// Lowercase hexadecimal SHA-256 digest.
    pub sha256: String,
    /// Exact pinned byte length.
    pub size_bytes: u64,
    /// Ordered commit-pinned HTTPS mirrors.
    pub urls: Vec<String>,
}

impl ExternalValidationFile {
    /// Revalidate decoded or caller-mutated file metadata.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetVerificationError::InvalidManifest`] for unsafe or incomplete metadata.
    pub fn validate(&self) -> Result<(), DatasetVerificationError> {
        let path = Path::new(&self.name);
        let mut components = path.components();
        if self.name.is_empty()
            || !matches!(components.next(), Some(Component::Normal(_)))
            || components.next().is_some()
        {
            return Err(DatasetVerificationError::InvalidManifest(
                "external validation filename must be a safe basename",
            ));
        }
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase())
        {
            return Err(DatasetVerificationError::InvalidManifest(
                "external validation SHA-256 must be lowercase hexadecimal",
            ));
        }
        if self.size_bytes == 0 {
            return Err(DatasetVerificationError::InvalidManifest(
                "external validation size must be positive",
            ));
        }
        if self.urls.is_empty() || self.urls.iter().any(|url| !url.starts_with("https://")) {
            return Err(DatasetVerificationError::InvalidManifest(
                "external validation files require HTTPS URLs",
            ));
        }
        Ok(())
    }
}

/// A citable checksum-pinned validation dataset.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ValidationDataset {
    /// Stable alphanumeric/hyphen identifier.
    pub dataset_id: String,
    /// Human-readable title.
    pub title: String,
    /// Primary HTTPS source page.
    pub source_url: String,
    /// Citation text.
    pub citation: String,
    /// Redistribution/license note.
    pub license_note: String,
    /// Ordered required file manifests.
    pub files: Vec<ExternalValidationFile>,
}

impl ValidationDataset {
    /// Revalidate decoded or caller-mutated dataset metadata.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetVerificationError::InvalidManifest`] for invalid metadata.
    pub fn validate(&self) -> Result<(), DatasetVerificationError> {
        if self.dataset_id.is_empty()
            || !self
                .dataset_id
                .bytes()
                .all(|value| value.is_ascii_alphanumeric() || value == b'-')
        {
            return Err(DatasetVerificationError::InvalidManifest(
                "validation dataset ID must be alphanumeric with hyphens",
            ));
        }
        if self.title.is_empty() || self.citation.is_empty() || self.license_note.is_empty() {
            return Err(DatasetVerificationError::InvalidManifest(
                "validation dataset metadata must not be empty",
            ));
        }
        if !self.source_url.starts_with("https://") {
            return Err(DatasetVerificationError::InvalidManifest(
                "validation dataset source URL must use HTTPS",
            ));
        }
        if self.files.is_empty() {
            return Err(DatasetVerificationError::InvalidManifest(
                "validation dataset must contain files",
            ));
        }
        for (index, file) in self.files.iter().enumerate() {
            file.validate()?;
            if self.files[..index]
                .iter()
                .any(|previous| previous.name == file.name)
            {
                return Err(DatasetVerificationError::InvalidManifest(
                    "validation dataset filenames must be unique",
                ));
            }
        }
        Ok(())
    }
}

/// Return all built-in validation manifests in stable identifier order.
#[must_use]
pub fn validation_datasets() -> Vec<ValidationDataset> {
    vec![sucrose_dataset(), pbso4_dataset(), qarr_dataset()]
}

fn sucrose_dataset() -> ValidationDataset {
    let sucrose_source = concat!(
        "https://raw.githubusercontent.com/AdvancedPhotonSource/GSAS-II-Tutorials/",
        "e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/data"
    );
    dataset(
        "aps-sucrose-11bmb",
        "APS 11-BM sucrose Le Bail tutorial pattern",
        concat!(
            "https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/",
            "e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/LeBailSucrose.htm"
        ),
        "Advanced Photon Source, GSAS-II Le Bail fitting tutorial, Sucrose",
        concat!(
            "External tutorial data fetched from the commit-pinned official ",
            "AdvancedPhotonSource/GSAS-II-Tutorials repository; not redistributed."
        ),
        vec![
            file(
                "11bmb_8716.fxye",
                "0971eebded2fdc6d8ae81a228325b2075ac45f6605da67468750e16ec23ecadd",
                1_725_515,
                format!("{sucrose_source}/11bmb_8716.fxye"),
            ),
            file(
                "11bmb_8716.prm",
                "ef606d5620cc8c8d9e7b712e1ba6113c9628e83f085835a2d477d4cd4d0d312c",
                1_134,
                format!("{sucrose_source}/11bmb_8716.prm"),
            ),
        ],
    )
}

fn pbso4_dataset() -> ValidationDataset {
    let pbso4_source = concat!(
        "https://raw.githubusercontent.com/AdvancedPhotonSource/GSAS-II-Tutorials/",
        "e2485148a3d7ee4757239b1ba40653f1f715bba5/PythonScript/data"
    );
    dataset(
        "gsasii-pbso4-cw",
        "GSAS-II PbSO4 combined constant-wavelength refinement tutorial",
        concat!(
            "https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/",
            "e2485148a3d7ee4757239b1ba40653f1f715bba5/",
            "CWCombined/Combined%20refinement.htm"
        ),
        "Advanced Photon Source, GSAS-II combined X-ray/neutron refinement tutorial",
        concat!(
            "External tutorial data fetched from the commit-pinned official ",
            "AdvancedPhotonSource/GSAS-II-Tutorials repository; not redistributed."
        ),
        vec![
            file(
                "PBSO4.XRA",
                "ca2da02fc7e17d2fc912de22a979f8101ba6140f2e8104f9d3aa64f030ca58bb",
                49_445,
                format!("{pbso4_source}/PBSO4.XRA"),
            ),
            file(
                "PBSO4.CWN",
                "59462ba6d7c72c9800b0b6bf41903f9933ed5ac5adab101d27022d5d25cf18ba",
                24_190,
                format!("{pbso4_source}/PBSO4.CWN"),
            ),
            file(
                "PbSO4-Wyckoff.cif",
                "9bc19d0995561afd78a5f0563599c032751621da994e36c7570b4f515d9f0e2f",
                1_516,
                format!("{pbso4_source}/PbSO4-Wyckoff.cif"),
            ),
            file(
                "INST_XRY.PRM",
                "e59413059bc3b12a51f1c6eebc5d581470320d05a31f11a41dbb32e16d1d16b9",
                794,
                format!("{pbso4_source}/INST_XRY.PRM"),
            ),
            file(
                "inst_d1a.prm",
                "a1031174a9b509f889377752f5c095a1dad49a25fb0a72c057f184f7b321fcea",
                971,
                format!("{pbso4_source}/inst_d1a.prm"),
            ),
        ],
    )
}

fn qarr_dataset() -> ValidationDataset {
    let qarr_mirror = concat!(
        "https://raw.githubusercontent.com/EdgarGF93/gsas_tutorial/",
        "78c18d8c2c058067b92e1a8ecb8fa81d32980008"
    );
    dataset(
        "iucr-qarr-1g",
        "IUCr Quantitative Phase Analysis Round Robin sample 1g",
        "https://www.iucr.org/__data/iucr/powder/QARR/data-kit.htm",
        "I. C. Madsen et al., J. Appl. Cryst. 34 (2001) 409-426, doi:10.1107/S0021889801007476",
        concat!(
            "External IUCr round-robin data; fetched from a commit-pinned public mirror. ",
            "COD-derived CIF headers identify their structural data as public domain/CC0."
        ),
        vec![
            file(
                "cpd-1g.prn",
                "afd16d03c8742abf315ab5f94585a6e540ab5c2e2c6df6917ac8dbf6be69d990",
                145_020,
                format!("{qarr_mirror}/cpd-1g.prn"),
            ),
            file(
                "Al2O3.cif",
                "bc8b07c4e27fdb9df562c7b72d181a05394df20274a56b7e1ef50f6ec2c85709",
                3_288,
                format!("{qarr_mirror}/Al2O3.cif"),
            ),
            file(
                "CaF2.cif",
                "b55fd04a3e344f73d4ece983da412056d04874b8ada7dfd4ecb93118e7905cd9",
                4_361,
                format!("{qarr_mirror}/CaF2.cif"),
            ),
            file(
                "ZnO.cif",
                "021db20c9bdabcfd63bc794367fae3536197425ca5826d63541b2417e47d3c1a",
                2_142,
                format!("{qarr_mirror}/ZnO.cif"),
            ),
            file(
                "cuka.instprm",
                "aef315a10622fdb1afae5b264d43e7f9dcc053aba80aa3d9fa05067146922a08",
                215,
                format!("{qarr_mirror}/cuka.instprm"),
            ),
        ],
    )
}

/// Return one built-in validation manifest by stable identifier.
///
/// # Errors
///
/// Returns [`DatasetVerificationError::UnknownDataset`] when the identifier is not registered.
pub fn validation_dataset(dataset_id: &str) -> Result<ValidationDataset, DatasetVerificationError> {
    validation_datasets()
        .into_iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| DatasetVerificationError::UnknownDataset(dataset_id.to_owned()))
}

/// Verify every required local file without network access.
///
/// Returned paths follow the stable manifest order.
///
/// # Errors
///
/// Returns [`DatasetVerificationError`] for unknown datasets, missing files, I/O failures,
/// size mismatches, or digest mismatches.
pub fn verify_validation_dataset(
    dataset_id: &str,
    directory: &Path,
) -> Result<Vec<PathBuf>, DatasetVerificationError> {
    let dataset = validation_dataset(dataset_id)?;
    dataset.validate()?;
    let mut paths = Vec::with_capacity(dataset.files.len());
    for expected in dataset.files {
        let path = directory.join(&expected.name);
        let metadata = path
            .metadata()
            .map_err(|source| DatasetVerificationError::Io {
                path: path.clone(),
                source,
            })?;
        if !metadata.is_file() {
            return Err(DatasetVerificationError::NotAFile(path));
        }
        if metadata.len() != expected.size_bytes {
            return Err(DatasetVerificationError::SizeMismatch {
                path,
                expected: expected.size_bytes,
                actual: metadata.len(),
            });
        }
        let file = File::open(&path).map_err(|source| DatasetVerificationError::Io {
            path: path.clone(),
            source,
        })?;
        let mut reader = BufReader::new(file);
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let count =
                reader
                    .read(&mut buffer)
                    .map_err(|source| DatasetVerificationError::Io {
                        path: path.clone(),
                        source,
                    })?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        let actual = format!("{:x}", digest.finalize());
        if actual != expected.sha256 {
            return Err(DatasetVerificationError::DigestMismatch {
                path,
                expected: expected.sha256,
                actual,
            });
        }
        paths.push(path);
    }
    Ok(paths)
}

fn dataset(
    dataset_id: &str,
    title: &str,
    source_url: &str,
    citation: &str,
    license_note: &str,
    files: Vec<ExternalValidationFile>,
) -> ValidationDataset {
    ValidationDataset {
        dataset_id: dataset_id.to_owned(),
        title: title.to_owned(),
        source_url: source_url.to_owned(),
        citation: citation.to_owned(),
        license_note: license_note.to_owned(),
        files,
    }
}

fn file(name: &str, sha256: &str, size_bytes: u64, url: String) -> ExternalValidationFile {
    ExternalValidationFile {
        name: name.to_owned(),
        sha256: sha256.to_owned(),
        size_bytes,
        urls: vec![url],
    }
}

/// Dataset-manifest or local-file verification failure.
#[derive(Debug)]
pub enum DatasetVerificationError {
    /// The requested stable identifier is not registered.
    UnknownDataset(String),
    /// Built-in or decoded manifest metadata is invalid.
    InvalidManifest(&'static str),
    /// A required path exists but is not a regular file.
    NotAFile(PathBuf),
    /// A local file does not have the pinned byte length.
    SizeMismatch {
        /// Local file path.
        path: PathBuf,
        /// Pinned byte length.
        expected: u64,
        /// Observed byte length.
        actual: u64,
    },
    /// A local file does not have the pinned SHA-256 digest.
    DigestMismatch {
        /// Local file path.
        path: PathBuf,
        /// Pinned lowercase digest.
        expected: String,
        /// Observed lowercase digest.
        actual: String,
    },
    /// Local filesystem access failed.
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying filesystem error.
        source: io::Error,
    },
}

impl Display for DatasetVerificationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownDataset(dataset_id) => {
                write!(formatter, "unknown validation dataset {dataset_id:?}")
            }
            Self::InvalidManifest(message) => formatter.write_str(message),
            Self::NotAFile(path) => write!(
                formatter,
                "validation input is not a file: {}",
                path.display()
            ),
            Self::SizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "size mismatch for {}: expected {expected}, got {actual}",
                path.display()
            ),
            Self::DigestMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "SHA-256 mismatch for {}: expected {expected}, got {actual}",
                path.display()
            ),
            Self::Io { path, source } => {
                write!(formatter, "could not access {}: {source}", path.display())
            }
        }
    }
}

impl Error for DatasetVerificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
