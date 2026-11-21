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

/// A CIF number specifies a type. Prefer the declared setting when its exact
/// operations match a tabulated setting of that type. Otherwise return the
/// standard group so the importer's normal conflict policy still applies.
pub(crate) fn space_group_for_cif_number(
    number: i32,
    declared: Option<&SpaceGroup>,
) -> Result<SpaceGroup, SpaceGroupLookupError> {
    let standard = space_group_by_number(number)?;
    if let Some(group) = declared
        && *group != standard.space_group
    {
        for entry in
            entries().filter(|entry| entry.number == number && entry.setting != standard.setting)
        {
            let alternative = info_from_hall_number(entry.hall_number)?.space_group;
            if alternative == *group {
                return Ok(alternative);
            }
        }
    }
    Ok(standard.space_group)
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
    let requested = hall_key(symbol);
    let entry = entries()
        .find(|entry| hall_key(entry.hall_symbol) == requested)
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
    if let Ok(info) = space_group_by_hall_symbol(symbol) {
        return Ok(info.space_group);
    }
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
/// A full symbol, origin choice (`:1`/`:2`), rhombohedral axis qualifier, or Hall symbol preserves the
/// requested setting.
///
/// # Errors
///
/// Returns [`SpaceGroupLookupError::UnknownSymbol`] when the value cannot be
/// resolved uniquely, or a structured database/symmetry error.
pub fn space_group_by_symbol(symbol: &str) -> Result<SpaceGroupInfo, SpaceGroupLookupError> {
    space_group_for_hm(symbol, None).or_else(|error| match error {
        SpaceGroupLookupError::UnknownSymbol { .. } => space_group_by_hall_symbol(symbol),
        other => Err(other),
    })
}

pub(crate) fn space_group_for_hm(
    symbol: &str,
    declared: Option<&SpaceGroup>,
) -> Result<SpaceGroupInfo, SpaceGroupLookupError> {
    let requested = symbol.trim();
    let (hm, qualifier) = hm_and_qualifier(requested);
    let hm_key = symbol_key(hm);
    let mut matches = Vec::new();
    for entry in entries() {
        if qualifier.is_some_and(|q| {
            !(entry.setting.eq_ignore_ascii_case(q)
                || matches!(q, "1" | "2") && entry.setting.starts_with(q))
        }) {
            continue;
        }
        // Moyo's monoclinic short name can describe the reference basis
        // even on another setting's row. Shorten the actual full symbol.
        let short = if (3..=15).contains(&entry.number) {
            entry
                .hm_full
                .split_whitespace()
                .filter(|word| *word != "1")
                .collect::<String>()
        } else {
            entry.hm_short.to_owned()
        };
        if symbol_key(entry.hm_full) == hm_key
            || symbol_key(&short) == hm_key
            || legacy_glide_key(&entry).as_ref() == Some(&hm_key)
        {
            matches.push(entry);
        }
    }
    let first = matches.first().ok_or_else(|| unknown_symbol(symbol))?;
    if matches.iter().any(|m| m.number != first.number) {
        return Err(unknown_symbol(symbol));
    }
    if matches.len() > 1
        && let Some(group) = declared
    {
        for entry in &matches {
            let info = info_from_hall_number(entry.hall_number)?;
            if info.space_group == *group {
                return Ok(info);
            }
        }
    }
    // Use the database default only if it actually matches the spelling.
    // Within an alternate basis the table lists conventional choices first.
    let default = Setting::Standard.hall_number(first.number);
    let chosen = matches
        .iter()
        .find(|entry| Some(entry.hall_number) == default)
        .unwrap_or(first);
    info_from_hall_number(chosen.hall_number)
}

// The five pre-double-glide H-M names (de Wolff et al., Acta Cryst. A48,
// §3) identify a particular glide direction. Carry that direction along
// with the database's axis permutation, including on origin-qualified rows.
fn legacy_glide_key(entry: &HallSymbolEntry) -> Option<String> {
    let old_direction = match entry.number {
        39 | 41 => 'b',      // Abm2, Aba2
        64 | 67 | 68 => 'a', // Cmca, Cmma, Ccca
        _ => return None,
    };
    let axes = entry.setting.trim_start_matches(['1', '2']);
    let axes = if axes.is_empty() { "abc" } else { axes };
    let position = axes
        .chars()
        .filter(char::is_ascii_alphabetic)
        .position(|c| c == old_direction)?;
    let direction = ["a", "b", "c"].get(position)?;
    Some(symbol_key(entry.hm_short).replace('e', direction))
}

/// Enumerate every canonical Hall setting compatible with the supplied cell metric.
/// This tests `R^T G R = G`; it does not infer intensity symmetry, centring or absences.
/// No basis change is performed. All settings in the pinned database are considered.
/// # Errors
/// Invalid cell geometry or an inconsistent database record.
pub fn space_group_settings_for_cell(
    cell: phasesmith_crystallography::UnitCell,
) -> Result<Vec<SpaceGroupInfo>, SpaceGroupLookupError> {
    let g = cell
        .geometry()
        .map_err(|_| SpaceGroupLookupError::InvalidDatabaseEntry)?
        .direct_metric;
    let scale = g.iter().flatten().fold(1.0_f64, |a, b| a.max(b.abs()));
    let mut result = Vec::new();
    let mut unique = std::collections::BTreeMap::new();
    for entry in entries() {
        let operations =
            operations_from_number(entry.number, Setting::HallNumber(entry.hall_number), false)
                .map_err(|_| SpaceGroupLookupError::InvalidDatabaseEntry)?;
        let compatible = operations.iter().all(|op| {
            let r = op.rotation_as_array();
            (0..3).all(|i| {
                (0..3).all(|j| {
                    let transformed: f64 = (0..3)
                        .flat_map(|a| {
                            (0..3).map(move |b| f64::from(r[a][i]) * g[a][b] * f64::from(r[b][j]))
                        })
                        .sum();
                    (transformed - g[i][j]).abs() <= 1e-10 * scale
                })
            })
        });
        if compatible {
            let info = info_from_hall_number(entry.hall_number)?;
            if let Some(previous) = unique.get(&info.hall_symbol) {
                if previous != &info.space_group {
                    return Err(SpaceGroupLookupError::InvalidDatabaseEntry);
                }
            } else {
                unique.insert(info.hall_symbol.clone(), info.space_group.clone());
                result.push(info);
            }
        }
    }
    Ok(result)
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
        .flat_map(char::to_lowercase)
        .collect()
}

fn hall_key(value: &str) -> String {
    // Generator boundaries are meaningful, e.g. P 32 versus P 3 2.
    normalize_hall_expression(value)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
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
        if matches!(
            qualifier.to_ascii_lowercase().as_str(),
            "h" | "r" | "1" | "2"
        ) {
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
