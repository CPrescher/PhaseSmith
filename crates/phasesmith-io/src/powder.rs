//! Deterministic readers for common text powder-pattern formats.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use phasesmith_model::{DomainError, PatternRecord};

type GsasBank = (Vec<String>, Vec<String>);
type GsasBanks = BTreeMap<usize, GsasBank>;

/// Caller-selected or detected powder text format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PowderFormat {
    /// Detect GSAS bank headers or fall back to plain columns.
    #[default]
    Auto,
    /// Two or three whitespace/comma-separated columns.
    Columns,
    /// Unpacked constant-wavelength GSAS FXYE bank.
    GsasFxye,
    /// Packed constant-step GSAS STD bank.
    GsasStd,
}

/// Resource limits checked before or during parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowderReadLimits {
    /// Maximum UTF-8 byte count.
    pub max_bytes: usize,
    /// Maximum number of output samples.
    pub max_rows: usize,
}

impl Default for PowderReadLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            max_rows: 5_000_000,
        }
    }
}

impl PowderReadLimits {
    /// Validate positive byte and row limits.
    ///
    /// # Errors
    ///
    /// Returns [`PowderIoError::InvalidLimits`] when either limit is zero.
    pub fn validate(self) -> Result<(), PowderIoError> {
        if self.max_bytes == 0 || self.max_rows == 0 {
            return Err(PowderIoError::InvalidLimits);
        }
        Ok(())
    }
}

/// One observed powder dataset plus source metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct PowderData {
    /// Validated observed pattern with zero supplied background.
    pub pattern: PatternRecord,
    /// Concrete detected/selected format.
    pub format: PowderFormat,
    /// Source path when read from a file.
    pub source_path: Option<PathBuf>,
    /// Selected GSAS bank, absent for columns.
    pub bank: Option<usize>,
}

/// Native powder input failure with stable categories.
#[derive(Debug)]
pub enum PowderIoError {
    /// A configured resource limit is zero.
    InvalidLimits,
    /// A selected GSAS bank is zero.
    InvalidBank,
    /// Input exceeds the configured byte limit.
    ByteLimitExceeded {
        /// Observed byte count.
        actual: u64,
        /// Configured maximum.
        maximum: usize,
    },
    /// Parsed rows exceed the configured maximum.
    RowLimitExceeded {
        /// Configured maximum.
        maximum: usize,
    },
    /// Filesystem or UTF-8 reading failed.
    Io(std::io::Error),
    /// A line contains invalid syntax.
    Parse {
        /// One-based input line, or zero for bank-wide metadata.
        line: usize,
        /// Stable explanation.
        message: String,
    },
    /// The requested bank is unavailable.
    MissingBank {
        /// Requested positive bank number.
        requested: usize,
        /// Available banks in numeric order.
        available: Vec<usize>,
    },
    /// Parsed numeric arrays violate the domain model.
    Domain(DomainError),
}

impl Display for PowderIoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("powder read limits must be positive"),
            Self::InvalidBank => formatter.write_str("bank must be a positive integer"),
            Self::ByteLimitExceeded { actual, maximum } => {
                write!(
                    formatter,
                    "powder input exceeds max_bytes: {actual} > {maximum}"
                )
            }
            Self::RowLimitExceeded { maximum } => {
                write!(formatter, "powder data exceeds max_rows: {maximum}")
            }
            Self::Io(error) => Display::fmt(error, formatter),
            Self::Parse { line: 0, message } => formatter.write_str(message),
            Self::Parse { line, message } => write!(formatter, "line {line}: {message}"),
            Self::MissingBank {
                requested,
                available,
            } => {
                let values = available
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    formatter,
                    "GSAS bank {requested} not found; available banks: {}",
                    if values.is_empty() { "none" } else { &values }
                )
            }
            Self::Domain(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for PowderIoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Domain(error) => Some(error),
            _ => None,
        }
    }
}

/// Read and parse one bounded UTF-8 powder file.
///
/// # Errors
///
/// Returns [`PowderIoError`] for filesystem, limit, syntax, or domain errors.
pub fn read_powder_file(
    path: impl AsRef<Path>,
    format: PowderFormat,
    bank: usize,
    limits: PowderReadLimits,
) -> Result<PowderData, PowderIoError> {
    limits.validate()?;
    if bank == 0 {
        return Err(PowderIoError::InvalidBank);
    }
    let path = path.as_ref();
    let size = fs::metadata(path).map_err(PowderIoError::Io)?.len();
    if size > u64::try_from(limits.max_bytes).unwrap_or(u64::MAX) {
        return Err(PowderIoError::ByteLimitExceeded {
            actual: size,
            maximum: limits.max_bytes,
        });
    }
    let text = fs::read_to_string(path).map_err(PowderIoError::Io)?;
    parse_powder_text_inner(&text, format, bank, limits, Some(path.to_owned()))
}

/// Parse bounded UTF-8 powder text without filesystem access.
///
/// # Errors
///
/// Returns [`PowderIoError`] for limit, syntax, or domain errors.
pub fn parse_powder_text(
    text: &str,
    format: PowderFormat,
    bank: usize,
    limits: PowderReadLimits,
) -> Result<PowderData, PowderIoError> {
    parse_powder_text_inner(text, format, bank, limits, None)
}

fn parse_powder_text_inner(
    text: &str,
    format: PowderFormat,
    bank: usize,
    limits: PowderReadLimits,
    source_path: Option<PathBuf>,
) -> Result<PowderData, PowderIoError> {
    limits.validate()?;
    if bank == 0 {
        return Err(PowderIoError::InvalidBank);
    }
    if text.len() > limits.max_bytes {
        return Err(PowderIoError::ByteLimitExceeded {
            actual: text.len() as u64,
            maximum: limits.max_bytes,
        });
    }
    let selected = match format {
        PowderFormat::Auto => detect_format(text, source_path.as_deref()),
        selected => selected,
    };
    match selected {
        PowderFormat::Columns => read_columns(text, source_path, limits.max_rows),
        PowderFormat::GsasFxye => read_gsas_fxye(text, source_path, bank, limits.max_rows),
        PowderFormat::GsasStd => read_gsas_std(text, source_path, bank, limits.max_rows),
        PowderFormat::Auto => unreachable!("auto format is resolved above"),
    }
}

fn detect_format(text: &str, source: Option<&Path>) -> PowderFormat {
    let headers = text
        .lines()
        .map(str::trim)
        .filter(|line| line.to_ascii_uppercase().starts_with("BANK "))
        .collect::<Vec<_>>();
    if headers
        .iter()
        .filter_map(|header| header.split_whitespace().next_back())
        .any(|encoding| encoding.eq_ignore_ascii_case("FXYE"))
    {
        return PowderFormat::GsasFxye;
    }
    if !headers.is_empty() {
        return PowderFormat::GsasStd;
    }
    if source
        .and_then(Path::extension)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("fxye"))
    {
        return PowderFormat::GsasFxye;
    }
    PowderFormat::Columns
}

fn numeric_rows(
    text: &str,
    required_columns: Option<usize>,
    max_rows: usize,
) -> Result<Vec<Vec<f64>>, PowderIoError> {
    let mut rows = Vec::new();
    let mut columns = required_columns;
    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let mut line = raw.trim();
        if line.is_empty() || line.starts_with(['#', '!', ';']) {
            continue;
        }
        if let Some(offset) = line.find(['#', '!']) {
            line = line[..offset].trim();
        }
        if line.is_empty() {
            continue;
        }
        let fields = line.replace(',', " ");
        let fields = fields.split_whitespace().collect::<Vec<_>>();
        if columns.is_none() && !matches!(fields.len(), 2 | 3) {
            return Err(parse_error(line_number, "expected two or three columns"));
        }
        let expected = *columns.get_or_insert(fields.len());
        if fields.len() != expected {
            return Err(parse_error(
                line_number,
                format!("expected {expected} columns, got {}", fields.len()),
            ));
        }
        let row = fields
            .iter()
            .map(|field| {
                field
                    .parse::<f64>()
                    .map_err(|_| parse_error(line_number, "non-numeric powder value"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(row);
        if rows.len() > max_rows {
            return Err(PowderIoError::RowLimitExceeded { maximum: max_rows });
        }
    }
    if rows.is_empty() {
        return Err(parse_error(0, "powder data contains no numeric rows"));
    }
    Ok(rows)
}

fn powder_data(
    x_deg: Vec<f64>,
    observed_y: Vec<f64>,
    uncertainty: Option<Vec<f64>>,
    format: PowderFormat,
    source_path: Option<PathBuf>,
    bank: Option<usize>,
) -> Result<PowderData, PowderIoError> {
    if x_deg.is_empty() {
        return Err(parse_error(0, "powder data contains no numeric rows"));
    }
    let pattern = PatternRecord::new(x_deg, Some(observed_y), uncertainty, None, None)
        .map_err(PowderIoError::Domain)?;
    Ok(PowderData {
        pattern,
        format,
        source_path,
        bank,
    })
}

fn read_columns(
    text: &str,
    source: Option<PathBuf>,
    max_rows: usize,
) -> Result<PowderData, PowderIoError> {
    let rows = numeric_rows(text, None, max_rows)?;
    let has_uncertainty = rows[0].len() == 3;
    powder_data(
        rows.iter().map(|row| row[0]).collect(),
        rows.iter().map(|row| row[1]).collect(),
        has_uncertainty.then(|| rows.iter().map(|row| row[2]).collect()),
        PowderFormat::Columns,
        source,
        None,
    )
}

fn gsas_banks(text: &str) -> Result<GsasBanks, PowderIoError> {
    let mut banks = BTreeMap::new();
    let mut active = None;
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.to_ascii_uppercase().starts_with("BANK ") {
            let fields = line
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let bank = fields
                .get(1)
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| parse_error(index + 1, "invalid GSAS bank header"))?;
            if banks.insert(bank, (fields, Vec::new())).is_some() {
                return Err(parse_error(
                    index + 1,
                    format!("duplicate GSAS bank {bank}"),
                ));
            }
            active = Some(bank);
        } else if let Some(bank) = active {
            banks
                .get_mut(&bank)
                .expect("active bank exists")
                .1
                .push(raw.to_owned());
        }
    }
    Ok(banks)
}

fn selected_bank(banks: &GsasBanks, bank: usize) -> Result<&GsasBank, PowderIoError> {
    banks.get(&bank).ok_or_else(|| PowderIoError::MissingBank {
        requested: bank,
        available: banks.keys().copied().collect(),
    })
}

fn read_gsas_fxye(
    text: &str,
    source: Option<PathBuf>,
    bank: usize,
    max_rows: usize,
) -> Result<PowderData, PowderIoError> {
    let banks = gsas_banks(text)?;
    if banks.values().any(|(header, _)| {
        header
            .last()
            .is_none_or(|value| !value.eq_ignore_ascii_case("FXYE"))
    }) {
        return Err(parse_error(
            0,
            "only unpacked GSAS FXYE banks are supported",
        ));
    }
    let (header, lines) = selected_bank(&banks, bank)?;
    debug_assert!(
        header
            .last()
            .is_some_and(|value| value.eq_ignore_ascii_case("FXYE"))
    );
    let rows = numeric_rows(&lines.join("\n"), Some(3), max_rows)?;
    powder_data(
        rows.iter().map(|row| row[0] / 100.0).collect(),
        rows.iter().map(|row| row[1]).collect(),
        Some(rows.iter().map(|row| row[2]).collect()),
        PowderFormat::GsasFxye,
        source,
        Some(bank),
    )
}

fn read_gsas_std(
    text: &str,
    source: Option<PathBuf>,
    bank: usize,
    max_rows: usize,
) -> Result<PowderData, PowderIoError> {
    let banks = gsas_banks(text)?;
    let (header, lines) = selected_bank(&banks, bank)?;
    let encoding = if header.len() >= 10 {
        header.last().map_or("", String::as_str)
    } else {
        "STD"
    };
    if header.len() < 7
        || !header[4].eq_ignore_ascii_case("CONST")
        || !encoding.eq_ignore_ascii_case("STD")
    {
        return Err(parse_error(
            0,
            "only packed constant-step GSAS STD banks are supported",
        ));
    }
    let row_count = header[2]
        .parse::<usize>()
        .map_err(|_| parse_error(0, "invalid packed GSAS STD bank dimensions"))?;
    let start_deg = parse_header_f64(&header[5])? / 100.0;
    let step_deg = parse_header_f64(&header[6])? / 100.0;
    if row_count == 0 || row_count > max_rows || !start_deg.is_finite() || !step_deg.is_finite() {
        return Err(parse_error(
            0,
            "packed GSAS STD bank dimensions exceed limits or are non-finite",
        ));
    }
    if step_deg <= 0.0 {
        return Err(parse_error(0, "packed GSAS STD step must be positive"));
    }
    let mut intensities = Vec::with_capacity(row_count);
    let mut uncertainty = Vec::with_capacity(row_count);
    'lines: for (index, line) in lines.iter().enumerate() {
        for bytes in line.as_bytes().chunks(8) {
            let record = std::str::from_utf8(bytes)
                .map_err(|_| parse_error(index + 1, "invalid UTF-8 fixed-width record"))?;
            if record.trim().is_empty() {
                continue;
            }
            let normalization_field = record.get(..2).unwrap_or(record).trim();
            let normalization = if normalization_field.is_empty() {
                1_u32
            } else {
                normalization_field
                    .parse::<u32>()
                    .map_err(|_| parse_error(index + 1, "invalid fixed-width record"))?
                    .max(1)
            };
            let parsed_intensity = record
                .get(2..)
                .unwrap_or("")
                .trim()
                .parse::<f64>()
                .map_err(|_| parse_error(index + 1, "invalid fixed-width record"))?;
            if !parsed_intensity.is_finite() {
                return Err(parse_error(index + 1, "invalid fixed-width record"));
            }
            let intensity = parsed_intensity.max(0.0);
            intensities.push(intensity);
            uncertainty.push(if intensity > 0.0 {
                (intensity / f64::from(normalization)).sqrt()
            } else {
                1.0
            });
            if intensities.len() == row_count {
                break 'lines;
            }
        }
    }
    if intensities.len() != row_count {
        return Err(parse_error(
            0,
            format!(
                "packed GSAS STD bank contains {} records; expected {row_count}",
                intensities.len()
            ),
        ));
    }
    powder_data(
        coordinate_grid(start_deg, step_deg, row_count)?,
        intensities,
        Some(uncertainty),
        PowderFormat::GsasStd,
        source,
        Some(bank),
    )
}

fn coordinate_grid(
    start_deg: f64,
    step_deg: f64,
    row_count: usize,
) -> Result<Vec<f64>, PowderIoError> {
    (0..row_count)
        .map(|index| {
            let exact_index = u32::try_from(index).map_err(|_| {
                parse_error(
                    0,
                    "packed GSAS STD bank dimensions exceed limits or are non-finite",
                )
            })?;
            Ok(start_deg + step_deg * f64::from(exact_index))
        })
        .collect()
}

fn parse_header_f64(value: &str) -> Result<f64, PowderIoError> {
    value
        .parse::<f64>()
        .map_err(|_| parse_error(0, "invalid packed GSAS STD bank dimensions"))
}

fn parse_error(line: usize, message: impl Into<String>) -> PowderIoError {
    PowderIoError::Parse {
        line,
        message: message.into(),
    }
}
