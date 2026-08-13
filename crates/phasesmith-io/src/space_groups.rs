//! Native lookup of exact conventional space-group operation sets.

use std::error::Error;
use std::fmt::{Display, Formatter};

use moyo::base::Operation;
use moyo::data::{HallSymbol, HallSymbolEntry, Setting, hall_symbol_entry, operations_from_number};
use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryError, SymmetryOperation};

const HALL_ENTRY_COUNT: i32 = 530;
const TRANSLATION_DENOMINATOR: i64 = 12;
const TRANSLATION_DENOMINATOR_F64: f64 = 12.0;

/// Reviewed source and version for native space-group lookup data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpaceGroupDatabaseProvenance {
    /// Database provider crate.
    pub provider: &'static str,
    /// Exact pinned crate version.
    pub version: &'static str,
    /// Upstream data lineage declared by the provider.
    pub lineage: &'static str,
    /// Number of conventional Hall settings.
    pub hall_setting_count: usize,
}

/// Provenance for the pure-Rust conventional space-group database.
pub const SPACE_GROUP_DATABASE_PROVENANCE: SpaceGroupDatabaseProvenance =
    SpaceGroupDatabaseProvenance {
        provider: "moyo",
        version: "0.15.0",
        lineage: "spglib Hall-symbol database",
        hall_setting_count: 530,
    };

/// Human identifiers plus an exact engine-owned conventional operation set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpaceGroupInfo {
    /// International Tables number in `[1, 230]`.
    pub number: i32,
    /// Short Hermann--Mauguin symbol.
    pub hm_symbol: String,
    /// Hall symbol defining this exact setting and origin.
    pub hall_symbol: String,
    /// Setting qualifier, empty when the group has one reference setting.
    pub setting: String,
    /// Validated exact conventional operation set.
    pub space_group: SpaceGroup,
}

/// Native space-group lookup or database-conversion failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpaceGroupLookupError {
    /// International number lies outside `[1, 230]`.
    InvalidNumber,
    /// No unique supported Hermann--Mauguin or Hall symbol matched.
    UnknownSymbol {
        /// Rejected caller value.
        symbol: String,
    },
    /// A pinned database record could not be resolved consistently.
    InvalidDatabaseEntry,
    /// Generated operations failed `PhaseSmith`'s exact group validation.
    Symmetry(SymmetryError),
}

impl Display for SpaceGroupLookupError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNumber => {
                formatter.write_str("space-group number must be an integer in [1, 230]")
            }
            Self::UnknownSymbol { symbol } => {
                write!(
                    formatter,
                    "unknown or ambiguous space-group symbol {symbol:?}"
                )
            }
            Self::InvalidDatabaseEntry => {
                formatter.write_str("native space-group database entry is invalid")
            }
            Self::Symmetry(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for SpaceGroupLookupError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Symmetry(error) => Some(error),
            _ => None,
        }
    }
}

/// Resolve an International Tables number in the conventional standard setting.
///
/// # Errors
///
/// Returns [`SpaceGroupLookupError`] for an out-of-range number or invalid
/// pinned database record.
pub fn space_group_by_number(number: i32) -> Result<SpaceGroupInfo, SpaceGroupLookupError> {
    if !(1..=230).contains(&number) {
        return Err(SpaceGroupLookupError::InvalidNumber);
    }
    let hall_number = Setting::Standard
        .hall_number(number)
        .ok_or(SpaceGroupLookupError::InvalidDatabaseEntry)?;
    info_from_hall_number(hall_number)
}

/// Resolve an exact Hall symbol, preserving its setting and origin choice.
///
/// Both the conventional CIF quote operator (`"`) and Moyo's internal (`=`)
/// spelling are accepted.
///
/// # Errors
///
/// Returns [`SpaceGroupLookupError::UnknownSymbol`] if no Hall entry matches.
pub fn space_group_by_hall_symbol(symbol: &str) -> Result<SpaceGroupInfo, SpaceGroupLookupError> {
    let requested = symbol.trim();
    let entry = entries()
        .find(|entry| symbol_key(entry.hall_symbol) == symbol_key(requested))
        .ok_or_else(|| unknown_symbol(symbol))?;
    info_from_hall_number(entry.hall_number)
}

/// Parse a general Hall expression into an exact engine-owned operation set.
///
/// Unlike [`space_group_by_hall_symbol`], this function is not limited to the
/// 530 canonical database spellings. It accepts any non-magnetic Hall
/// expression supported by the pinned Moyo parser, including redundant
/// translation spellings and explicit origin shifts. CIF quote syntax (`"`)
/// and underscore component separators are normalized before parsing.
///
/// # Errors
///
/// Returns [`SpaceGroupLookupError::UnknownSymbol`] when the Hall expression
/// is invalid, or a structured database/symmetry error when its generated
/// operations cannot be represented and validated exactly.
pub fn space_group_from_hall_symbol(symbol: &str) -> Result<SpaceGroup, SpaceGroupLookupError> {
    let requested = normalize_hall_expression(symbol);
    if requested.is_empty() {
        return Err(unknown_symbol(symbol));
    }
    let hall_symbol = HallSymbol::new(&requested).ok_or_else(|| unknown_symbol(symbol))?;
    let coset = hall_symbol.traverse();
    let mut operations = Vec::with_capacity(coset.len() * hall_symbol.centering.order());
    for lattice_point in hall_symbol.centering.lattice_points() {
        for operation in &coset {
            let translation =
                (lattice_point + operation.translation).map(|value| value.rem_euclid(1.0));
            operations.push(Operation::new(operation.rotation, translation));
        }
    }
    exact_space_group_from_moyo_operations(&operations)
}

/// Resolve a Hermann--Mauguin, full-setting, or Hall symbol.
///
/// Short Hermann--Mauguin symbols select the conventional standard setting.
/// A full symbol, explicit `:H`/`:R` qualifier, or Hall symbol preserves the
/// requested setting.
///
/// # Errors
///
/// Returns [`SpaceGroupLookupError::UnknownSymbol`] when the value cannot be
/// resolved uniquely, or a structured database/symmetry error.
pub fn space_group_by_symbol(symbol: &str) -> Result<SpaceGroupInfo, SpaceGroupLookupError> {
    let requested = symbol.trim();
    if requested.is_empty() {
        return Err(unknown_symbol(symbol));
    }

    if let Ok(info) = space_group_by_hall_symbol(requested) {
        return Ok(info);
    }

    let (hm, qualifier) = hm_and_qualifier(requested);
    let hm_key = symbol_key(hm);
    let full_matches = entries()
        .filter(|entry| symbol_key(entry.hm_full) == hm_key)
        .filter(|entry| qualifier.is_none_or(|value| entry.setting.eq_ignore_ascii_case(value)))
        .collect::<Vec<_>>();
    if full_matches.len() == 1 {
        return info_from_hall_number(full_matches[0].hall_number);
    }

    let short_numbers = entries()
        .filter(|entry| symbol_key(entry.hm_short) == hm_key)
        .filter(|entry| qualifier.is_none_or(|value| entry.setting.eq_ignore_ascii_case(value)))
        .map(|entry| entry.number)
        .collect::<std::collections::BTreeSet<_>>();
    if short_numbers.len() != 1 {
        return Err(unknown_symbol(symbol));
    }
    let number = *short_numbers
        .first()
        .ok_or_else(|| unknown_symbol(symbol))?;
    if qualifier.is_none() {
        return space_group_by_number(number);
    }
    let matched = entries()
        .find(|entry| {
            entry.number == number
                && symbol_key(entry.hm_short) == hm_key
                && qualifier.is_some_and(|value| entry.setting.eq_ignore_ascii_case(value))
        })
        .ok_or_else(|| unknown_symbol(symbol))?;
    info_from_hall_number(matched.hall_number)
}

fn entries() -> impl Iterator<Item = HallSymbolEntry> {
    (1..=HALL_ENTRY_COUNT).filter_map(hall_symbol_entry)
}

fn info_from_hall_number(hall_number: i32) -> Result<SpaceGroupInfo, SpaceGroupLookupError> {
    let entry =
        hall_symbol_entry(hall_number).ok_or(SpaceGroupLookupError::InvalidDatabaseEntry)?;
    let operations =
        operations_from_number(entry.number, Setting::HallNumber(entry.hall_number), false)
            .map_err(|_| SpaceGroupLookupError::InvalidDatabaseEntry)?;
    let space_group = exact_space_group_from_moyo_operations(&operations)?;
    Ok(SpaceGroupInfo {
        number: entry.number,
        hm_symbol: entry.hm_short.replace('_', ""),
        hall_symbol: entry.hall_symbol.replace('=', "\""),
        setting: entry.setting.to_owned(),
        space_group,
    })
}

fn exact_space_group_from_moyo_operations(
    operations: &[Operation],
) -> Result<SpaceGroup, SpaceGroupLookupError> {
    let operations = operations
        .iter()
        .map(|operation| {
            let rotation = operation.rotation_as_array();
            let translation = operation
                .translation_as_array()
                .map(rational_from_database_translation)
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let translation: [Rational; 3] = translation
                .try_into()
                .map_err(|_| SpaceGroupLookupError::InvalidDatabaseEntry)?;
            SymmetryOperation::new(rotation, translation).map_err(SpaceGroupLookupError::Symmetry)
        })
        .collect::<Result<Vec<_>, _>>()?;
    SpaceGroup::new(operations).map_err(SpaceGroupLookupError::Symmetry)
}

fn rational_from_database_translation(value: f64) -> Result<Rational, SpaceGroupLookupError> {
    let normalized = value.rem_euclid(1.0);
    let scaled = normalized * TRANSLATION_DENOMINATOR_F64;
    let rounded = scaled.round();
    if !rounded.is_finite() || (scaled - rounded).abs() > 1.0e-8 {
        return Err(SpaceGroupLookupError::InvalidDatabaseEntry);
    }
    #[allow(clippy::cast_possible_truncation)]
    let numerator = rounded as i64;
    Rational::new(numerator, TRANSLATION_DENOMINATOR).map_err(SpaceGroupLookupError::Symmetry)
}

fn symbol_key(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && *character != '_')
        // Moyo stores the Hall-symbol quote operator as `=` internally while
        // conventional CIF files use `"`.
        .map(|character| if character == '"' { '=' } else { character })
        .flat_map(char::to_lowercase)
        .collect()
}

fn normalize_hall_expression(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|character| match character {
            '"' => '=',
            '_' => ' ',
            other => other,
        })
        .collect()
}

fn hm_and_qualifier(value: &str) -> (&str, Option<&str>) {
    if let Some((base, qualifier)) = value.rsplit_once(':') {
        let qualifier = qualifier.trim();
        if matches!(qualifier.to_ascii_lowercase().as_str(), "h" | "r") {
            return (base.trim(), Some(qualifier));
        }
    }
    let mut words = value.split_whitespace().collect::<Vec<_>>();
    if words.len() > 1 {
        let last = words.last().copied().unwrap_or_default();
        if matches!(last.to_ascii_lowercase().as_str(), "h" | "r") {
            words.pop();
            let split = value.len() - last.len();
            return (value[..split].trim(), Some(last));
        }
    }
    (value, None)
}

fn unknown_symbol(symbol: &str) -> SpaceGroupLookupError {
    SpaceGroupLookupError::UnknownSymbol {
        symbol: symbol.to_owned(),
    }
}
