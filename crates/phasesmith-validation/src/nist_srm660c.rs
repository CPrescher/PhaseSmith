//! NIST SRM 660c archive integrity and published-reference validation.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use zip::ZipArchive;

use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "nist-srm660c-lab6-xray";
const ARCHIVE_NAME: &str = "srm_660c_cifs_20201029_081700.zip";
const CERTIFIED_LATTICE_ANGSTROM: f64 = 4.156_826;
const CERTIFIED_EXPANDED_UNCERTAINTY_ANGSTROM: f64 = 0.000_080;
const EXPECTED_SPECIMENS: usize = 20;
const EXPECTED_POINTS_PER_SPECIMEN: usize = 5_332;
const EXPECTED_REFLECTION_REGIONS: usize = 24;

/// Validate the checksum-pinned NIST certification archive and its supplied reference fits.
///
/// This gate deliberately does not claim pointwise `PhaseSmith` equivalence: the NIST reference
/// curves use a fundamental-parameters source/optics model that `PhaseSmith` does not implement.
/// It establishes an immutable, machine-readable external oracle for the subsequent matched
/// line-position and line-shape checkpoint.
///
/// # Errors
///
/// Returns [`NistSrm660cValidationError`] for missing/corrupt data or malformed pdCIF content.
#[allow(clippy::too_many_lines)]
pub fn run_nist_srm660c_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, NistSrm660cValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let file = File::open(dataset_directory.join(ARCHIVE_NAME))?;
    let mut archive = ZipArchive::new(file)?;
    let mut specimens = 0_usize;
    let mut sample_count = 0_usize;
    let mut lattice_sum = 0.0;
    let mut correlation = Correlation::default();
    let mut weighted_squared_residual = 0.0;
    let mut weighted_squared_observed = 0.0;
    let mut identical_grids = true;
    let mut member_names = BTreeSet::new();
    let mut total_uncompressed_bytes = 0_u64;
    let mut minimum_specimen_correlation = 1.0_f64;
    let mut maximum_specimen_rwp = 0.0_f64;
    let mut maximum_lattice_deviation = 0.0_f64;

    for index in 0..archive.len() {
        let mut member = archive.by_index(index)?;
        let name = member.name().to_owned();
        if name.starts_with("__MACOSX/") || !name.to_ascii_lowercase().ends_with(".cif") {
            continue;
        }
        if !safe_cif_member(&name) || !member_names.insert(name.clone()) {
            return Err(NistSrm660cValidationError::InvalidPdCif(
                "NIST archive contains an unsafe or duplicate pdCIF member".to_owned(),
            ));
        }
        if member.size() > 2 * 1024 * 1024 {
            return Err(NistSrm660cValidationError::InvalidPdCif(
                "NIST pdCIF member exceeds the 2 MiB validation limit".to_owned(),
            ));
        }
        total_uncompressed_bytes = total_uncompressed_bytes.checked_add(member.size()).ok_or(
            NistSrm660cValidationError::Arithmetic("archive size overflow"),
        )?;
        if total_uncompressed_bytes > 40 * 1024 * 1024 {
            return Err(NistSrm660cValidationError::InvalidPdCif(
                "NIST pdCIF members exceed the 40 MiB validation limit".to_owned(),
            ));
        }
        let mut text = String::new();
        member.read_to_string(&mut text)?;
        let parsed = parse_pdcif(&text)?;
        if parsed.measured.len() != EXPECTED_POINTS_PER_SPECIMEN
            || parsed.calculated.len() != EXPECTED_POINTS_PER_SPECIMEN
        {
            return Err(NistSrm660cValidationError::InvalidPdCif(format!(
                "{name} does not contain two {EXPECTED_POINTS_PER_SPECIMEN}-point profiles"
            )));
        }
        specimens += 1;
        sample_count += parsed.measured.len();
        lattice_sum += parsed.lattice_angstrom;
        maximum_lattice_deviation = maximum_lattice_deviation
            .max((parsed.lattice_angstrom - CERTIFIED_LATTICE_ANGSTROM).abs());
        let (specimen_correlation, specimen_rwp) =
            profile_statistics(&parsed.measured, &parsed.calculated)?;
        minimum_specimen_correlation = minimum_specimen_correlation.min(specimen_correlation);
        maximum_specimen_rwp = maximum_specimen_rwp.max(specimen_rwp);
        for (measured, calculated) in parsed.measured.iter().zip(&parsed.calculated) {
            identical_grids &= measured.x.to_bits() == calculated.x.to_bits();
            correlation.push(measured.y, calculated.y);
            let residual = measured.y - calculated.y;
            weighted_squared_residual += measured.weight * residual * residual;
            weighted_squared_observed += measured.weight * measured.y * measured.y;
        }
    }
    if specimens != EXPECTED_SPECIMENS {
        return Err(NistSrm660cValidationError::InvalidPdCif(format!(
            "expected {EXPECTED_SPECIMENS} certification specimens, found {specimens}"
        )));
    }
    let specimen_count = f64::from(
        u32::try_from(specimens)
            .map_err(|_| NistSrm660cValidationError::Arithmetic("specimen count overflow"))?,
    );
    let mean_lattice = lattice_sum / specimen_count;
    let profile_correlation = correlation.finish()?;
    let reference_rwp = (weighted_squared_residual / weighted_squared_observed).sqrt();
    let checks = vec![
        check(
            "archive_specimens",
            specimens == EXPECTED_SPECIMENS,
            "The official archive contains the complete set of certification specimens.",
            Some(specimen_count),
            "20 pdCIF specimens",
        )?,
        check(
            "pdcif_profiles",
            sample_count == EXPECTED_SPECIMENS * EXPECTED_POINTS_PER_SPECIMEN && identical_grids,
            "Every specimen supplies aligned measured and NIST-calculated profile arrays.",
            Some(count_as_f64(sample_count)?),
            "106640 measured points and identical calculated grids",
        )?,
        check(
            "certified_lattice_interval",
            maximum_lattice_deviation <= CERTIFIED_EXPANDED_UNCERTAINTY_ANGSTROM,
            "Every lattice value in the released reference fits lies inside the NIST 95% certified interval.",
            Some(maximum_lattice_deviation),
            "max |a - 4.156826 A| <= 0.000080 A",
        )?,
        check(
            "nist_reference_profile",
            profile_correlation >= 0.999 && reference_rwp <= 0.065,
            "Released NIST reference fits reproduce the raw certification scans; this is an external-oracle integrity gate, not a PhaseSmith equivalence gate.",
            Some(reference_rwp),
            "correlation >= 0.999 and weighted Rwp <= 0.065",
        )?,
        check(
            "nist_specimen_correlation",
            minimum_specimen_correlation >= 0.999,
            "The correlation gate is satisfied independently by every certification specimen.",
            Some(minimum_specimen_correlation),
            "minimum per-specimen correlation >= 0.999",
        )?,
        check(
            "nist_specimen_rwp",
            maximum_specimen_rwp <= 0.065,
            "The weighted residual gate is satisfied independently by every certification specimen.",
            Some(maximum_specimen_rwp),
            "maximum per-specimen weighted Rwp <= 0.065",
        )?,
    ];
    RealDataValidationReport::new(
        DATASET_ID,
        sample_count,
        Some(EXPECTED_REFLECTION_REGIONS),
        started.elapsed().as_secs_f64(),
        checks,
        vec![
            format!(
                "Released-reference correlation={profile_correlation:.9}; weighted Rwp={reference_rwp:.9}."
            ),
            format!(
                "Mean released-fit lattice={mean_lattice:.9} A; maximum certified-value deviation={maximum_lattice_deviation:.9} A."
            ),
            concat!(
                "NIST states that the bundled calculated curves were recomputed for data release; ",
                "the original measured scans are the certification data."
            )
            .to_owned(),
            concat!(
                "A matched PhaseSmith line-shape gate remains separate because the NIST fit uses ",
                "a Cu emission spectrum and fundamental-parameters optics model."
            )
            .to_owned(),
        ],
    )
    .map_err(Into::into)
}

fn safe_cif_member(name: &str) -> bool {
    let path = Path::new(name);
    path.components()
        .all(|component| matches!(component, std::path::Component::Normal(_)))
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("cif"))
}

fn profile_statistics(
    measured: &[ProfilePoint],
    calculated: &[ProfilePoint],
) -> Result<(f64, f64), NistSrm660cValidationError> {
    if measured.len() != calculated.len() || measured.is_empty() {
        return Err(NistSrm660cValidationError::Arithmetic(
            "profile arrays are not aligned",
        ));
    }
    let mut correlation = Correlation::default();
    let mut weighted_squared_residual = 0.0;
    let mut weighted_squared_observed = 0.0;
    for (measured, calculated) in measured.iter().zip(calculated) {
        correlation.push(measured.y, calculated.y);
        weighted_squared_residual +=
            measured.weight * (measured.y - calculated.y) * (measured.y - calculated.y);
        weighted_squared_observed += measured.weight * measured.y * measured.y;
    }
    let rwp = (weighted_squared_residual / weighted_squared_observed).sqrt();
    if !rwp.is_finite() {
        return Err(NistSrm660cValidationError::Arithmetic(
            "profile residual is not finite",
        ));
    }
    Ok((correlation.finish()?, rwp))
}

#[derive(Clone, Copy)]
struct ProfilePoint {
    x: f64,
    y: f64,
    weight: f64,
}

struct ParsedPdCif {
    lattice_angstrom: f64,
    measured: Vec<ProfilePoint>,
    calculated: Vec<ProfilePoint>,
}

fn parse_pdcif(text: &str) -> Result<ParsedPdCif, NistSrm660cValidationError> {
    let lattice_angstrom = tagged_number(text, "_cell_length_a")?;
    let mut profiles = Vec::new();
    let mut active = Vec::new();
    let mut reading_rows = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("data_pd_proc_") {
            if !active.is_empty() {
                profiles.push(std::mem::take(&mut active));
            }
            reading_rows = false;
        } else if trimmed == "_pd_proc_ls_weight" {
            reading_rows = true;
        } else if reading_rows {
            let values = trimmed.split_whitespace().collect::<Vec<_>>();
            if values.len() != 3 {
                if !active.is_empty() {
                    profiles.push(std::mem::take(&mut active));
                }
                reading_rows = false;
                continue;
            }
            let parsed = values
                .iter()
                .map(|value| value.parse::<f64>())
                .collect::<Result<Vec<_>, _>>();
            match parsed {
                Ok(values) => active.push(ProfilePoint {
                    x: values[0],
                    y: values[1],
                    weight: values[2],
                }),
                Err(_) if active.is_empty() => {
                    reading_rows = false;
                }
                Err(_) => {
                    return Err(NistSrm660cValidationError::InvalidPdCif(
                        "non-numeric value inside a NIST profile loop".to_owned(),
                    ));
                }
            }
        }
    }
    if !active.is_empty() {
        profiles.push(active);
    }
    if profiles.len() != 2
        || profiles
            .iter()
            .flatten()
            .any(|point| !point.x.is_finite() || !point.y.is_finite() || point.weight <= 0.0)
    {
        return Err(NistSrm660cValidationError::InvalidPdCif(
            "expected exactly two finite positive-weight profile loops".to_owned(),
        ));
    }
    Ok(ParsedPdCif {
        lattice_angstrom,
        measured: profiles.remove(0),
        calculated: profiles.remove(0),
    })
}

fn tagged_number(text: &str, tag: &str) -> Result<f64, NistSrm660cValidationError> {
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        if fields.next() == Some(tag)
            && let Some(value) = fields.next().and_then(|value| value.parse::<f64>().ok())
            && value.is_finite()
        {
            return Ok(value);
        }
    }
    Err(NistSrm660cValidationError::InvalidPdCif(format!(
        "missing or invalid numeric tag {tag}"
    )))
}

#[derive(Default)]
struct Correlation {
    count: u64,
    left_sum: f64,
    right_sum: f64,
    left_squared_sum: f64,
    right_squared_sum: f64,
    product_sum: f64,
}

impl Correlation {
    fn push(&mut self, left: f64, right: f64) {
        self.count += 1;
        self.left_sum += left;
        self.right_sum += right;
        self.left_squared_sum += left * left;
        self.right_squared_sum += right * right;
        self.product_sum += left * right;
    }

    fn finish(&self) -> Result<f64, NistSrm660cValidationError> {
        let count = f64::from(
            u32::try_from(self.count)
                .map_err(|_| NistSrm660cValidationError::Arithmetic("sample count exceeds u32"))?,
        );
        let numerator = self.product_sum - self.left_sum * self.right_sum / count;
        let left = self.left_squared_sum - self.left_sum * self.left_sum / count;
        let right = self.right_squared_sum - self.right_sum * self.right_sum / count;
        let value = numerator / (left * right).sqrt();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(NistSrm660cValidationError::Arithmetic(
                "profile correlation is not finite",
            ))
        }
    }
}

fn check(
    check_id: &str,
    passed: bool,
    detail: &str,
    measured: Option<f64>,
    criterion: &str,
) -> Result<ValidationCheck, ValidationContractError> {
    ValidationCheck::new(
        check_id,
        if passed {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        },
        detail,
        measured,
        Some(criterion.to_owned()),
    )
}

fn count_as_f64(count: usize) -> Result<f64, NistSrm660cValidationError> {
    u32::try_from(count)
        .map(f64::from)
        .map_err(|_| NistSrm660cValidationError::Arithmetic("sample count exceeds u32"))
}

/// NIST SRM 660c validation setup, archive, or report failure.
#[derive(Debug)]
pub enum NistSrm660cValidationError {
    /// Dataset checksum verification failed.
    Dataset(DatasetVerificationError),
    /// Archive or member I/O failed.
    Io(std::io::Error),
    /// ZIP structure or decompression failed.
    Zip(zip::result::ZipError),
    /// A released pdCIF did not match the bounded validation schema.
    InvalidPdCif(String),
    /// A derived statistic was undefined.
    Arithmetic(&'static str),
    /// Report construction failed.
    Report(ValidationContractError),
}

impl Display for NistSrm660cValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(error) => Display::fmt(error, formatter),
            Self::Io(error) => Display::fmt(error, formatter),
            Self::Zip(error) => Display::fmt(error, formatter),
            Self::InvalidPdCif(message) => formatter.write_str(message),
            Self::Arithmetic(message) => formatter.write_str(message),
            Self::Report(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for NistSrm660cValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dataset(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Zip(error) => Some(error),
            Self::Report(error) => Some(error),
            Self::InvalidPdCif(_) | Self::Arithmetic(_) => None,
        }
    }
}

impl From<DatasetVerificationError> for NistSrm660cValidationError {
    fn from(value: DatasetVerificationError) -> Self {
        Self::Dataset(value)
    }
}

impl From<std::io::Error> for NistSrm660cValidationError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<zip::result::ZipError> for NistSrm660cValidationError {
    fn from(value: zip::result::ZipError) -> Self {
        Self::Zip(value)
    }
}

impl From<ValidationContractError> for NistSrm660cValidationError {
    fn from(value: ValidationContractError) -> Self {
        Self::Report(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pdcif(weight: f64) -> String {
        format!(
            "_cell_length_a 4.1568\n\
             data_pd_proc_measured\n\
             _pd_proc_ls_weight\n\
             1 10 {weight}\n\
             2 20 {weight}\n\
             stop_\n\
             data_pd_proc_calculated\n\
             _pd_proc_ls_weight\n\
             1 11 {weight}\n\
             2 19 {weight}\n\
             stop_\n"
        )
    }

    #[test]
    fn bounded_pdcif_parser_accepts_two_profiles_and_rejects_bad_weights() {
        let parsed = parse_pdcif(&pdcif(1.0)).unwrap();
        assert_eq!(parsed.measured.len(), 2);
        assert_eq!(parsed.calculated.len(), 2);
        assert!(parse_pdcif(&pdcif(0.0)).is_err());
        assert!(parse_pdcif("_cell_length_a nan\n").is_err());
    }

    #[test]
    fn archive_member_policy_rejects_traversal_and_non_cif_files() {
        assert!(safe_cif_member("specimen.cif"));
        assert!(safe_cif_member("nested/specimen.CIF"));
        assert!(!safe_cif_member("../specimen.cif"));
        assert!(!safe_cif_member("/absolute/specimen.cif"));
        assert!(!safe_cif_member("specimen.txt"));
    }
}
