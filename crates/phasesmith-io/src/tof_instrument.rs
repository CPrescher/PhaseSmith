//! Bounded legacy GSAS TOF instrument-parameter import.
//!
//! Profile functions 1 and 3 are translated into the shared published
//! back-to-back-exponential coefficient law. The adapter emits `PhaseSmith`'s
//! typed 15-coefficient model; legacy records never enter the core.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use phasesmith_core::{TofBankGeometry, TofError, TofInstrument};

/// Resource limit checked before decoding a legacy instrument file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GsasTofInstrumentReadLimits {
    /// Maximum UTF-8 byte count.
    pub max_bytes: usize,
}

impl Default for GsasTofInstrumentReadLimits {
    fn default() -> Self {
        Self {
            max_bytes: 4 * 1024 * 1024,
        }
    }
}

impl GsasTofInstrumentReadLimits {
    /// Reject a zero allocation boundary.
    ///
    /// # Errors
    ///
    /// Returns [`GsasTofInstrumentIoError::InvalidLimits`] for zero bytes.
    pub fn validate(self) -> Result<(), GsasTofInstrumentIoError> {
        if self.max_bytes == 0 {
            return Err(GsasTofInstrumentIoError::InvalidLimits);
        }
        Ok(())
    }
}

/// One translated legacy GSAS TOF bank plus source metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct GsasTofInstrumentData {
    /// `PhaseSmith`'s validated public 15-coefficient model.
    pub instrument: TofInstrument,
    /// Selected positive legacy bank.
    pub bank: usize,
    /// Legacy GSAS profile function number. Currently 1 or 3.
    pub profile_function: usize,
    /// Parsed fixed bank scattering angle when a `BNKPAR` record is present.
    pub bank_geometry: Option<TofBankGeometry>,
    /// Source path when read from a file.
    pub source_path: Option<PathBuf>,
}

/// Stable failure categories for legacy GSAS TOF instrument import.
#[derive(Debug)]
pub enum GsasTofInstrumentIoError {
    /// The configured byte limit is zero.
    InvalidLimits,
    /// The selected bank is zero or too large for the two-column legacy key.
    InvalidBank,
    /// Input exceeds the configured byte limit.
    ByteLimitExceeded {
        /// Observed bytes.
        actual: u64,
        /// Configured maximum.
        maximum: usize,
    },
    /// Filesystem or UTF-8 reading failed.
    Io(std::io::Error),
    /// A required bank record is absent.
    MissingRecord {
        /// Selected bank.
        bank: usize,
        /// Legacy record name.
        record: &'static str,
    },
    /// A record does not contain the required finite values.
    InvalidRecord {
        /// Selected bank.
        bank: usize,
        /// Legacy record name.
        record: &'static str,
    },
    /// Only independently documented legacy translations are supported.
    UnsupportedProfileFunction {
        /// Selected bank.
        bank: usize,
        /// Parsed legacy function number.
        found: usize,
    },
    /// Translated coefficients violate the core model.
    Profile(TofError),
}

impl Display for GsasTofInstrumentIoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLimits => {
                formatter.write_str("GSAS TOF instrument max_bytes must be positive")
            }
            Self::InvalidBank => formatter.write_str("GSAS TOF bank must lie in 1..=99"),
            Self::ByteLimitExceeded { actual, maximum } => write!(
                formatter,
                "GSAS TOF instrument input exceeds max_bytes: {actual} > {maximum}"
            ),
            Self::Io(error) => Display::fmt(error, formatter),
            Self::MissingRecord { bank, record } => {
                write!(formatter, "GSAS TOF bank {bank} has no {record} record")
            }
            Self::InvalidRecord { bank, record } => {
                write!(
                    formatter,
                    "GSAS TOF bank {bank} has an invalid {record} record"
                )
            }
            Self::UnsupportedProfileFunction { bank, found } => write!(
                formatter,
                "GSAS TOF bank {bank} uses unsupported profile function {found}; only functions 1 and 3 are supported"
            ),
            Self::Profile(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for GsasTofInstrumentIoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Profile(error) => Some(error),
            _ => None,
        }
    }
}

/// Read one bounded legacy GSAS TOF profile-function-1 or -3 bank.
///
/// # Errors
///
/// Returns [`GsasTofInstrumentIoError`] for filesystem, limit, bank, syntax,
/// unsupported-profile, or translated-domain failures.
pub fn read_gsas_tof_instrument_file(
    path: impl AsRef<Path>,
    bank: usize,
    limits: GsasTofInstrumentReadLimits,
) -> Result<GsasTofInstrumentData, GsasTofInstrumentIoError> {
    validate_request(bank, limits)?;
    let path = path.as_ref();
    let size = fs::metadata(path)
        .map_err(GsasTofInstrumentIoError::Io)?
        .len();
    if size > u64::try_from(limits.max_bytes).unwrap_or(u64::MAX) {
        return Err(GsasTofInstrumentIoError::ByteLimitExceeded {
            actual: size,
            maximum: limits.max_bytes,
        });
    }
    let text = fs::read_to_string(path).map_err(GsasTofInstrumentIoError::Io)?;
    parse_inner(&text, bank, limits, Some(path.to_owned()))
}

/// Parse one bounded legacy GSAS TOF profile-function-1 or -3 bank from text.
///
/// # Errors
///
/// Returns [`GsasTofInstrumentIoError`] for limit, bank, syntax,
/// unsupported-profile, or translated-domain failures.
pub fn parse_gsas_tof_instrument_text(
    text: &str,
    bank: usize,
    limits: GsasTofInstrumentReadLimits,
) -> Result<GsasTofInstrumentData, GsasTofInstrumentIoError> {
    parse_inner(text, bank, limits, None)
}

fn parse_inner(
    text: &str,
    bank: usize,
    limits: GsasTofInstrumentReadLimits,
    source_path: Option<PathBuf>,
) -> Result<GsasTofInstrumentData, GsasTofInstrumentIoError> {
    validate_request(bank, limits)?;
    if text.len() > limits.max_bytes {
        return Err(GsasTofInstrumentIoError::ByteLimitExceeded {
            actual: u64::try_from(text.len()).unwrap_or(u64::MAX),
            maximum: limits.max_bytes,
        });
    }
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let icons = record_values(text, bank, "ICONS", &format!("INS {bank:>2} ICONS"))?;
    if icons.len() < 4 {
        return Err(invalid(bank, "ICONS"));
    }
    let bank_geometry = parse_optional_bank_geometry(text, bank)?;
    let compact_header = format!("INS {bank:>2}PRCF1 ");
    let spaced_header = format!("INS {bank:>2}PRCF  ");
    let compact = text.lines().any(|line| line.starts_with(&compact_header));
    let function = profile_function(
        text,
        bank,
        if compact {
            &compact_header
        } else {
            &spaced_header
        },
    )?;
    if !matches!(function, 1 | 3) {
        return Err(GsasTofInstrumentIoError::UnsupportedProfileFunction {
            bank,
            found: function,
        });
    }
    let (exponential_prefix, gaussian_prefix) = if compact {
        (
            format!("INS {bank:>2}PRCF11"),
            format!("INS {bank:>2}PRCF12"),
        )
    } else {
        (
            format!("INS {bank:>2}PRCF 1"),
            format!("INS {bank:>2}PRCF 2"),
        )
    };
    let exponential = record_values(text, bank, "PRCF11", &exponential_prefix)?;
    let gaussian = record_values(text, bank, "PRCF12", &gaussian_prefix)?;
    if exponential.len() < if function == 1 { 4 } else { 3 } {
        return Err(invalid(bank, "PRCF11"));
    }
    if gaussian.len() < if function == 1 { 3 } else { 2 } {
        return Err(invalid(bank, "PRCF12"));
    }

    // Legacy ICONS stores DIFC, DIFA, Zero, and an unused fourth field. It
    // does not carry DIFB. Function 1 places an unused coefficient before
    // alpha/beta0/beta1 and before sigma1/sigma2; function 3 starts each
    // group directly. Remaining typed coefficients are explicit zeros.
    let (alpha, beta0, beta1, sigma1, sigma2) = if function == 1 {
        (
            exponential[1],
            exponential[2],
            exponential[3],
            gaussian[1],
            gaussian[2],
        )
    } else {
        (
            exponential[0],
            exponential[1],
            exponential[2],
            gaussian[0],
            gaussian[1],
        )
    };
    let instrument = TofInstrument {
        zero_us: icons[2],
        difc_us_per_angstrom: icons[0],
        difa_us_per_angstrom2: icons[1],
        difb_us_angstrom: 0.0,
        alpha_coefficient: alpha,
        beta0_per_us: beta0,
        beta1_angstrom4_per_us: beta1,
        betaq_angstrom2_per_us: 0.0,
        sigma0_us2: 0.0,
        sigma1_us2_per_angstrom2: sigma1,
        sigma2_us2_per_angstrom4: sigma2,
        sigmaq_us2_per_angstrom: 0.0,
        x_us_per_angstrom: 0.0,
        y_us_per_angstrom2: 0.0,
        z_us: 0.0,
    };
    instrument
        .validate()
        .map_err(GsasTofInstrumentIoError::Profile)?;
    Ok(GsasTofInstrumentData {
        instrument,
        bank,
        profile_function: function,
        bank_geometry,
        source_path,
    })
}

fn validate_request(
    bank: usize,
    limits: GsasTofInstrumentReadLimits,
) -> Result<(), GsasTofInstrumentIoError> {
    limits.validate()?;
    if !(1..=99).contains(&bank) {
        return Err(GsasTofInstrumentIoError::InvalidBank);
    }
    Ok(())
}

fn record_values(
    text: &str,
    bank: usize,
    record: &'static str,
    prefix: &str,
) -> Result<Vec<f64>, GsasTofInstrumentIoError> {
    let line = text
        .lines()
        .find(|line| line.starts_with(prefix))
        .ok_or(GsasTofInstrumentIoError::MissingRecord { bank, record })?;
    let values = line[prefix.len()..]
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid(bank, record))?;
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(bank, record));
    }
    Ok(values)
}

fn optional_record_values(
    text: &str,
    bank: usize,
    record: &'static str,
    prefix: &str,
) -> Result<Option<Vec<f64>>, GsasTofInstrumentIoError> {
    let Some(line) = text.lines().find(|line| line.starts_with(prefix)) else {
        return Ok(None);
    };
    let values = line[prefix.len()..]
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid(bank, record))?;
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(bank, record));
    }
    Ok(Some(values))
}

fn parse_optional_bank_geometry(
    text: &str,
    bank: usize,
) -> Result<Option<TofBankGeometry>, GsasTofInstrumentIoError> {
    optional_record_values(text, bank, "BNKPAR", &format!("INS {bank:>2}BNKPAR"))?
        .map(|values| {
            if values.len() < 2 {
                return Err(invalid(bank, "BNKPAR"));
            }
            let geometry = TofBankGeometry {
                two_theta_deg: values[1],
            };
            geometry.validate().map_err(|_| invalid(bank, "BNKPAR"))?;
            Ok(geometry)
        })
        .transpose()
}

fn profile_function(
    text: &str,
    bank: usize,
    prefix: &str,
) -> Result<usize, GsasTofInstrumentIoError> {
    let line = text.lines().find(|line| line.starts_with(prefix)).ok_or(
        GsasTofInstrumentIoError::MissingRecord {
            bank,
            record: "PRCF1",
        },
    )?;
    let mut tokens = line[prefix.len()..].split_whitespace();
    let function = tokens
        .next()
        .ok_or_else(|| invalid(bank, "PRCF1"))?
        .parse::<usize>()
        .map_err(|_| invalid(bank, "PRCF1"))?;
    Ok(function)
}

const fn invalid(bank: usize, record: &'static str) -> GsasTofInstrumentIoError {
    GsasTofInstrumentIoError::InvalidRecord { bank, record }
}
