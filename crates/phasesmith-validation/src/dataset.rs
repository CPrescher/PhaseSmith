use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ValidationStatus;

/// Scientific role assigned to one external dataset in the validation matrix.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationPurpose {
    /// Accepted complete `PhaseSmith` workflow.
    Acceptance,
    /// Integrity check for an independently released reference result.
    OracleIntegrity,
    /// Blind transferability case whose reviewed outcome may be a failure.
    Holdout,
    /// Dataset that exposes a deliberately unsupported capability boundary.
    Capability,
}

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
    /// Scientific role of this dataset in the validation matrix.
    pub purpose: ValidationPurpose,
    /// Reviewed outcome expected from the registered runner.
    pub expected_status: ValidationStatus,
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
    vec![
        echidna_dataset(),
        sucrose_dataset(),
        pbso4_dataset(),
        ceria_size_strain_dataset(),
        qarr_dataset(),
        qarr_1h_dataset(),
        nist_srm660c_dataset(),
        lanl_nickel_tof_dataset(),
        powgen_tof_dataset(),
    ]
}

fn ceria_size_strain_dataset() -> ValidationDataset {
    let source = "https://mysite.du.edu/~balzar";
    dataset(
        "iucr-ceria-size-strain-round-robin",
        "IUCr ceria size/strain round-robin laboratory X-ray pair",
        "https://mysite.du.edu/~balzar/s-s_rr.htm",
        concat!(
            "D. Balzar et al., J. Appl. Cryst. 37 (2004) 911-924, ",
            "doi:10.1107/S0021889804022551"
        ),
        concat!(
            "The IUCr-sponsored round-robin page explicitly offers the original ",
            "measurements for download but states no redistribution license. Files are ",
            "therefore checksum-pinned external inputs and are not redistributed."
        ),
        ValidationPurpose::Holdout,
        ValidationStatus::Passed,
        vec![
            file(
                "langfsh1.xy",
                "8b4c0562e3c25f2ed33bade08f03ba990f693c219f991bd408f73b2c6e7c8e9f",
                93_567,
                format!("{source}/langfsh1.xy"),
            ),
            file(
                "langfsh2.xy",
                "87fc1b17a7f17d441333eafea2aec7ad21cff1e25cb89c1bab915585868be963",
                40_017,
                format!("{source}/langfsh2.xy"),
            ),
            file(
                "langfsh3.xy",
                "387563d40cb287b0fc528616e38a65cf383382ed26927aca81cfaa3b277867ab",
                49_992,
                format!("{source}/langfsh3.xy"),
            ),
            file(
                "langfbr1.xy",
                "4c84a0fb8546bc04c12d86316b3eaa00bced244df875ce92af73d400beda6820",
                46_846,
                format!("{source}/langfbr1.xy"),
            ),
            file(
                "langfbr2.xy",
                "fbe42f048ef653731ebc767467d2a0b5fd0ed537c341901ed50b0a0379f812e4",
                20_071,
                format!("{source}/langfbr2.xy"),
            ),
            file(
                "langfbr3.xy",
                "ef3d0f202bf1f88cd6adc7d92cbce56906cb358cca7e1c5b0ed490f785741068",
                20_171,
                format!("{source}/langfbr3.xy"),
            ),
        ],
    )
}

fn echidna_dataset() -> ValidationDataset {
    let source = "https://zenodo.org/records/14286343/files";
    dataset(
        "ansto-echidna-lab6-cw-neutron",
        "ANSTO Echidna LaB6 constant-wavelength neutron calibration pattern",
        "https://doi.org/10.5281/zenodo.14286343",
        concat!(
            "M. Avdeev and J. R. Hester, Echidna Ge(115) monochromator calibration data, ",
            "doi:10.5281/zenodo.14286343"
        ),
        concat!(
            "External ANSTO calibration data from an immutable Zenodo record; the record ",
            "declares Creative Commons Attribution 4.0. Files are not redistributed."
        ),
        ValidationPurpose::Acceptance,
        ValidationStatus::Passed,
        vec![
            file(
                "ECH0034258_LaB6.xyd",
                "09950aaf3596518a40b2c5173ffdfd870043f55dcd3390fb83dc9f64e348e115",
                112_390,
                format!("{source}/ECH0034258_LaB6.xyd?download=1"),
            ),
            file(
                "ECH0034258_LaB6.cif",
                "0fa02945683ea39d56dfb81d0f99de629e4e336abab85c6d92b3689b42d130d1",
                112_390,
                format!("{source}/ECH0034258_LaB6.cif?download=1"),
            ),
        ],
    )
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
        ValidationPurpose::Acceptance,
        ValidationStatus::Passed,
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
        ValidationPurpose::Acceptance,
        ValidationStatus::Passed,
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
        ValidationPurpose::Acceptance,
        ValidationStatus::Passed,
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

fn qarr_1h_dataset() -> ValidationDataset {
    let qarr_mirror = concat!(
        "https://raw.githubusercontent.com/EdgarGF93/gsas_tutorial/",
        "78c18d8c2c058067b92e1a8ecb8fa81d32980008"
    );
    dataset(
        "iucr-qarr-1h",
        "IUCr Quantitative Phase Analysis Round Robin sample 1h holdout",
        "https://www.iucr.org/resources/commissions/powder-diffraction/projects/qarr/data",
        "I. C. Madsen et al., J. Appl. Cryst. 34 (2001) 409-426, doi:10.1107/S0021889801007476",
        concat!(
            "External IUCr round-robin data; fetched from a commit-pinned public mirror. ",
            "COD-derived CIF headers identify their structural data as public domain/CC0."
        ),
        ValidationPurpose::Holdout,
        ValidationStatus::Failed,
        vec![
            file(
                "cpd-1h.prn",
                "0c36af18d7a341f03f4d8f24972608791351c45c951c431dd93fb54eb8e56da3",
                145_020,
                format!("{qarr_mirror}/cpd-1h.prn"),
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

fn nist_srm660c_dataset() -> ValidationDataset {
    let source = "https://data.nist.gov/od/ds/mds2-2315";
    dataset(
        "nist-srm660c-lab6-xray",
        "NIST SRM 660c LaB6 line-position and line-shape certification scans",
        "https://catalog.data.gov/dataset/diffraction-data-for-srm-660c",
        concat!(
            "D. R. Black et al., Powder Diffraction 35 (2020) 17-22, ",
            "doi:10.1017/S0885715620000068"
        ),
        concat!(
            "Public NIST certification data under the NIST open-data license; the original ",
            "archive is checksum-pinned and is not redistributed."
        ),
        ValidationPurpose::OracleIntegrity,
        ValidationStatus::Passed,
        vec![file(
            "srm_660c_cifs_20201029_081700.zip",
            "92034d06498161db365420831fe35d91ec18bad4ed17385329a1761658da2c42",
            1_719_632,
            format!("{source}/srm_660c_cifs_20201029_081700.zip"),
        )],
    )
}

fn powgen_tof_dataset() -> ValidationDataset {
    let source = concat!(
        "https://raw.githubusercontent.com/AdvancedPhotonSource/GSAS-II-Tutorials/",
        "e2485148a3d7ee4757239b1ba40653f1f715bba5/TOF%20Calibration/data"
    );
    dataset(
        "powgen-lab6-tof-calibration",
        "POWGEN NIST LaB6 time-of-flight calibration tutorial",
        concat!(
            "https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/",
            "e2485148a3d7ee4757239b1ba40653f1f715bba5/TOF%20Calibration/",
            "Calibration%20of%20a%20TOF%20powder%20diffractometer.htm"
        ),
        "Advanced Photon Source, GSAS-II TOF powder diffractometer calibration tutorial",
        concat!(
            "External tutorial data fetched from the commit-pinned official ",
            "AdvancedPhotonSource/GSAS-II-Tutorials repository; not redistributed."
        ),
        ValidationPurpose::Acceptance,
        ValidationStatus::Passed,
        vec![
            file(
                "PG3_17541.gsa",
                "ff7a408451e75d23e828ab2bb35a061a53517ff3430331bffeed21fcbc87d69c",
                539_682,
                format!("{source}/PG3_17541.gsa"),
            ),
            file(
                "PGHR_60-2015A.prm",
                "1a098c260555d27642ab0501708c5d9058c5836fb201dec5bc7ab9880cea1cb8",
                7_295,
                format!("{source}/PGHR_60-2015A.prm"),
            ),
        ],
    )
}

fn lanl_nickel_tof_dataset() -> ValidationDataset {
    let source = concat!(
        "https://subversion.xray.aps.anl.gov/EXPGUI/!svn/bc/1253/",
        "tutorials/tutorial1"
    );
    dataset(
        "lanl-nickel-tof",
        "LANL nickel time-of-flight powder refinement tutorial",
        "https://subversion.xray.aps.anl.gov/EXPGUI/tutorials/tutorial1/",
        concat!(
            "A. C. Larson and R. B. Von Dreele, GSAS nickel powder tutorial example; ",
            "EXPGUI adaptation by B. H. Toby"
        ),
        concat!(
            "Official GSAS/EXPGUI example data fetched from immutable APS Subversion ",
            "revision 1253. The tutorial grants public copying and use with its ",
            "authorship notice; files are not redistributed."
        ),
        ValidationPurpose::Acceptance,
        ValidationStatus::Passed,
        vec![
            file(
                "nickel.raw",
                "bfe2afd6843a11dc1935cbbdb3ca05962c3242b9ddc60baa6a4ff6a48491c3e7",
                168_346,
                format!("{source}/nickel.raw"),
            ),
            file(
                "inst_tof.prm",
                "9a4f06cb560c8b7f783fd0773b37814dae1d705042087e66f8cd2be719477038",
                8_610,
                format!("{source}/inst_tof.prm"),
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

#[allow(clippy::too_many_arguments)]
fn dataset(
    dataset_id: &str,
    title: &str,
    source_url: &str,
    citation: &str,
    license_note: &str,
    purpose: ValidationPurpose,
    expected_status: ValidationStatus,
    files: Vec<ExternalValidationFile>,
) -> ValidationDataset {
    ValidationDataset {
        dataset_id: dataset_id.to_owned(),
        title: title.to_owned(),
        source_url: source_url.to_owned(),
        citation: citation.to_owned(),
        license_note: license_note.to_owned(),
        files,
        purpose,
        expected_status,
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
