//! CIF-to-domain import policy independent of presentation adapters.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation, UnitCell};

use crate::{space_group_by_hall_symbol, space_group_by_number, space_group_by_symbol};

use super::syntax::{CifBlock, CifValue, parse_document};
use super::{
    CifAnisotropicDisplacement, CifAtomSite, CifDiagnostic, CifIoError, CifReadLimits,
    CifReadResult, CifStructure, CifStructureSource, DisplacementConvention, NATIVE_CIF_BACKEND,
    NATIVE_CIF_BACKEND_VERSION, import_error,
};

const CELL_TAGS: [[&str; 2]; 6] = [
    ["_cell.length_a", "_cell_length_a"],
    ["_cell.length_b", "_cell_length_b"],
    ["_cell.length_c", "_cell_length_c"],
    ["_cell.angle_alpha", "_cell_angle_alpha"],
    ["_cell.angle_beta", "_cell_angle_beta"],
    ["_cell.angle_gamma", "_cell_angle_gamma"],
];
const EXPLICIT_OPERATION_TAGS: [&str; 3] = [
    "_space_group_symop.operation_xyz",
    "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",
];
const HALL_TAGS: [&str; 3] = [
    "_space_group.name_hall",
    "_space_group_name_hall",
    "_symmetry_space_group_name_hall",
];
const HM_TAGS: [&str; 3] = [
    "_space_group.name_h-m_alt",
    "_space_group_name_h-m_alt",
    "_symmetry_space_group_name_h-m",
];
const NUMBER_TAGS: [&str; 3] = [
    "_space_group.it_number",
    "_space_group_it_number",
    "_symmetry_int_tables_number",
];
const ANISO_LABEL_TAGS: [&str; 2] = ["_atom_site_aniso.label", "_atom_site_aniso_label"];
const ANISO_COMPONENTS: [&str; 6] = ["11", "22", "33", "23", "13", "12"];
const B_TO_U: f64 = 8.0 * std::f64::consts::PI * std::f64::consts::PI;

#[derive(Clone, Copy)]
struct ParsedNumber {
    value: f64,
    standard_uncertainty: Option<f64>,
}

#[derive(Clone, Copy)]
struct SiteColumn<'a> {
    tag: Option<&'a str>,
    values: &'a [CifValue],
}

impl SiteColumn<'_> {
    const fn empty() -> Self {
        Self {
            tag: None,
            values: &[],
        }
    }
}

struct SiteColumns<'a> {
    label: SiteColumn<'a>,
    type_symbol: SiteColumn<'a>,
    fract_x: SiteColumn<'a>,
    fract_y: SiteColumn<'a>,
    fract_z: SiteColumn<'a>,
    cart_x: SiteColumn<'a>,
    cart_y: SiteColumn<'a>,
    cart_z: SiteColumn<'a>,
    occupancy: SiteColumn<'a>,
    u_iso: SiteColumn<'a>,
    b_iso: SiteColumn<'a>,
    disorder_group: SiteColumn<'a>,
}

impl<'a> SiteColumns<'a> {
    fn from_block(block: &'a CifBlock) -> Self {
        Self {
            label: column(block, &["_atom_site.label", "_atom_site_label"]),
            type_symbol: column(block, &["_atom_site.type_symbol", "_atom_site_type_symbol"]),
            fract_x: column(block, &["_atom_site.fract_x", "_atom_site_fract_x"]),
            fract_y: column(block, &["_atom_site.fract_y", "_atom_site_fract_y"]),
            fract_z: column(block, &["_atom_site.fract_z", "_atom_site_fract_z"]),
            cart_x: column(block, &["_atom_site.cartn_x", "_atom_site_cartn_x"]),
            cart_y: column(block, &["_atom_site.cartn_y", "_atom_site_cartn_y"]),
            cart_z: column(block, &["_atom_site.cartn_z", "_atom_site_cartn_z"]),
            occupancy: column(block, &["_atom_site.occupancy", "_atom_site_occupancy"]),
            u_iso: column(
                block,
                &["_atom_site.u_iso_or_equiv", "_atom_site_u_iso_or_equiv"],
            ),
            b_iso: column(
                block,
                &["_atom_site.b_iso_or_equiv", "_atom_site_b_iso_or_equiv"],
            ),
            disorder_group: column(
                block,
                &["_atom_site.disorder_group", "_atom_site_disorder_group"],
            ),
        }
    }

    fn all(&self) -> [SiteColumn<'a>; 12] {
        [
            self.label,
            self.type_symbol,
            self.fract_x,
            self.fract_y,
            self.fract_z,
            self.cart_x,
            self.cart_y,
            self.cart_z,
            self.occupancy,
            self.u_iso,
            self.b_iso,
            self.disorder_group,
        ]
    }
}

/// Read and parse one bounded UTF-8 CIF file.
///
/// # Errors
///
/// Returns [`CifIoError`] for filesystem, limit, syntax, lookup, or domain
/// failures.
pub fn read_cif_file(
    path: impl AsRef<Path>,
    block: Option<&str>,
    strict: bool,
    limits: CifReadLimits,
) -> Result<CifReadResult, CifIoError> {
    limits.validate()?;
    let path = path.as_ref();
    let size = fs::metadata(path).map_err(CifIoError::Io)?.len();
    if size > u64::try_from(limits.max_bytes).unwrap_or(u64::MAX) {
        return Err(CifIoError::ByteLimitExceeded {
            actual: size,
            maximum: limits.max_bytes,
        });
    }
    let text = fs::read_to_string(path).map_err(CifIoError::Io)?;
    parse_cif_text_inner(&text, Some(path.to_owned()), block, strict, limits)
}

/// Parse bounded UTF-8 CIF text into one selected native structure.
///
/// # Errors
///
/// Returns [`CifIoError`] for limit, syntax, lookup, or domain failures.
pub fn parse_cif_text(
    text: &str,
    block: Option<&str>,
    strict: bool,
    limits: CifReadLimits,
) -> Result<CifReadResult, CifIoError> {
    parse_cif_text_inner(text, None, block, strict, limits)
}

fn parse_cif_text_inner(
    text: &str,
    source_path: Option<PathBuf>,
    requested_block: Option<&str>,
    strict: bool,
    limits: CifReadLimits,
) -> Result<CifReadResult, CifIoError> {
    limits.validate()?;
    if text.len() > limits.max_bytes {
        return Err(CifIoError::ByteLimitExceeded {
            actual: text.len() as u64,
            maximum: limits.max_bytes,
        });
    }
    let document = parse_document(text, limits)?;
    let available_blocks = document
        .blocks
        .iter()
        .map(|block| block.name.clone())
        .collect::<Vec<_>>();
    let (selected_index, mut diagnostics) =
        select_block(&available_blocks, requested_block, strict)?;
    let selected = &document.blocks[selected_index];
    check_unsupported_features(selected, &mut diagnostics, strict)?;
    let (cell, cell_standard_uncertainties) = parse_cell(selected, &mut diagnostics, strict)?;
    let (space_group, symmetry_metadata) = parse_space_group(selected, &mut diagnostics, strict)?;
    validate_cell_metric(cell, &space_group)?;
    let sites = parse_sites(selected, cell, &mut diagnostics, strict, limits)?;
    let name = first_text(
        selected,
        &[
            "_chemical_name_common",
            "_chemical.name_common",
            "_chemical_formula_structural",
            "_chemical_formula_sum",
            "_pd_phase_name",
        ],
    )
    .unwrap_or_else(|| selected.name.clone());
    let mut metadata = symmetry_metadata;
    for (key, tags) in [
        (
            "chemical_formula_sum",
            &["_chemical_formula_sum", "_chemical.formula_sum"][..],
        ),
        (
            "chemical_formula_structural",
            &[
                "_chemical_formula_structural",
                "_chemical.formula_structural",
            ][..],
        ),
        (
            "radiation_wavelength",
            &[
                "_diffrn_radiation_wavelength",
                "_diffrn_radiation.wavelength",
            ][..],
        ),
    ] {
        if let Some(value) = first_text(selected, tags) {
            metadata.insert(key.to_owned(), value);
        }
    }
    let structure = CifStructure {
        structure_id: stable_structure_id(&selected.name),
        name,
        cell,
        space_group,
        sites,
        source: CifStructureSource {
            format: "CIF".to_owned(),
            block_name: selected.name.clone(),
            backend: NATIVE_CIF_BACKEND.to_owned(),
            backend_version: NATIVE_CIF_BACKEND_VERSION.to_owned(),
            source_path,
        },
        cell_standard_uncertainties,
        diagnostics: diagnostics.clone(),
        metadata,
    };
    Ok(CifReadResult {
        structure,
        diagnostics,
        selected_block: selected.name.clone(),
        available_blocks,
    })
}

fn select_block(
    available: &[String],
    requested: Option<&str>,
    strict: bool,
) -> Result<(usize, Vec<CifDiagnostic>), CifIoError> {
    if let Some(requested) = requested {
        let index = available
            .iter()
            .position(|name| name == requested)
            .ok_or_else(|| import_error(format!("CIF block {requested:?} was not found")))?;
        return Ok((index, Vec::new()));
    }
    if available.len() == 1 {
        return Ok((0, Vec::new()));
    }
    if strict {
        return Err(import_error(
            "multi-block CIF requires an explicit block in strict mode",
        ));
    }
    let diagnostic = CifDiagnostic::warning(
        "multiple_blocks_first_selected",
        format!(
            "selected first of {} CIF blocks: {}",
            available.len(),
            available[0]
        ),
    );
    Ok((0, vec![diagnostic]))
}

fn check_unsupported_features(
    block: &CifBlock,
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
) -> Result<(), CifIoError> {
    let tags = block.all_tags();
    for (feature, prefixes) in [
        (
            "magnetic",
            &[
                "_atom_site_moment",
                "_space_group_symop_magn",
                "_space_group_magn",
            ][..],
        ),
        (
            "modulated",
            &[
                "_cell_wave_vector",
                "_atom_site_fourier",
                "_space_group_symop_ssg",
            ][..],
        ),
        (
            "macromolecular",
            &["_entity_poly", "_pdbx_", "_atom_site.label_asym_id"][..],
        ),
    ] {
        let matches = tags
            .iter()
            .filter(|tag| prefixes.iter().any(|prefix| tag.starts_with(prefix)))
            .take(3)
            .copied()
            .collect::<Vec<_>>();
        if matches.is_empty() {
            continue;
        }
        let message = format!(
            "{feature} CIF features are not supported: {}",
            matches.join(", ")
        );
        if strict {
            return Err(CifIoError::Unsupported {
                feature: feature.to_owned(),
                message,
            });
        }
        diagnostics.push(CifDiagnostic::warning(
            format!("unsupported_{feature}_features"),
            message,
        ));
    }
    Ok(())
}

fn parse_cell(
    block: &CifBlock,
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
) -> Result<(UnitCell, [Option<f64>; 6]), CifIoError> {
    let mut values = [0.0; 6];
    let mut uncertainties = [None; 6];
    for (index, aliases) in CELL_TAGS.iter().enumerate() {
        let definitions = aliases
            .iter()
            .filter_map(|tag| block.first_raw(&[*tag]).map(|(_, value)| (*tag, value)))
            .collect::<Vec<_>>();
        let (tag, raw) = definitions.first().copied().ok_or_else(|| {
            import_error(format!("CIF cell parameter is missing: {}", aliases[0]))
        })?;
        let parsed = parse_number(raw, tag, diagnostics, true, None)?
            .ok_or_else(|| import_error(format!("CIF cell parameter is not numeric: {tag}")))?;
        for (duplicate_tag, duplicate_raw) in definitions.iter().skip(1).copied() {
            if let Some(duplicate) =
                parse_number(duplicate_raw, duplicate_tag, diagnostics, true, None)?
                && !close(parsed.value, duplicate.value, 1.0e-10, 1.0e-12)
            {
                let message =
                    format!("duplicate cell definitions disagree: {tag} and {duplicate_tag}");
                if strict {
                    return Err(import_error(message));
                }
                diagnostics.push(
                    CifDiagnostic::warning("conflicting_cell_definition", message)
                        .with_tag(duplicate_tag),
                );
            }
        }
        values[index] = parsed.value;
        uncertainties[index] = parsed.standard_uncertainty;
    }
    let cell = UnitCell {
        a_angstrom: values[0],
        b_angstrom: values[1],
        c_angstrom: values[2],
        alpha_deg: values[3],
        beta_deg: values[4],
        gamma_deg: values[5],
    };
    cell.geometry().map_err(CifIoError::Cell)?;
    Ok((cell, uncertainties))
}

fn parse_space_group(
    block: &CifBlock,
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
) -> Result<(SpaceGroup, BTreeMap<String, String>), CifIoError> {
    let mut candidates: Vec<(&str, String, SpaceGroup)> = Vec::new();
    let explicit_values = block
        .first_column(&EXPLICIT_OPERATION_TAGS)
        .map(|(tag, values)| (tag, values.iter().collect::<Vec<_>>()))
        .or_else(|| {
            block
                .first_raw(&EXPLICIT_OPERATION_TAGS)
                .map(|(tag, value)| (tag, vec![value]))
        });
    if let Some((tag, values)) = explicit_values {
        let operations = values
            .iter()
            .enumerate()
            .map(|(row, value)| {
                let text = value.text().ok_or_else(|| {
                    import_error(format!("missing explicit symmetry operation at row {row}"))
                })?;
                parse_symmetry_operation(text).map_err(|error| {
                    import_error(format!(
                        "invalid symmetry operation {text:?} at row {row}: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let group = SpaceGroup::new(operations).map_err(CifIoError::Symmetry)?;
        candidates.push(("explicit_operations", tag.to_owned(), group));
    }

    let hall = first_tag_text(block, &HALL_TAGS);
    if let Some((tag, value)) = &hall {
        let info = space_group_by_hall_symbol(value).map_err(CifIoError::SpaceGroup)?;
        candidates.push(("hall", (*tag).to_owned(), info.space_group));
    }

    let hm = first_tag_text(block, &HM_TAGS);
    if let Some((tag, value)) = &hm {
        let info = space_group_by_symbol(value).map_err(CifIoError::SpaceGroup)?;
        candidates.push(("hermann_mauguin", (*tag).to_owned(), info.space_group));
    }

    let number_text = first_tag_text(block, &NUMBER_TAGS);
    if let Some((tag, value)) = &number_text {
        let parsed = value
            .parse::<f64>()
            .map_err(|_| import_error(format!("invalid space-group number {value:?}")))?;
        if !parsed.is_finite() || parsed.fract() != 0.0 || !(1.0..=230.0).contains(&parsed) {
            return Err(import_error(format!(
                "invalid space-group number {value:?}"
            )));
        }
        #[allow(clippy::cast_possible_truncation)]
        let number = parsed as i32;
        let info = space_group_by_number(number).map_err(CifIoError::SpaceGroup)?;
        candidates.push(("international_number", (*tag).to_owned(), info.space_group));
    }

    if candidates.is_empty() {
        diagnostics.push(CifDiagnostic::warning(
            "missing_space_group_assumed_p1",
            "no symmetry identifier was supplied; assumed P1",
        ));
        return Ok((
            SpaceGroup::new(vec![SymmetryOperation::identity()]).map_err(CifIoError::Symmetry)?,
            BTreeMap::from([("symmetry_source".to_owned(), "assumed_p1".to_owned())]),
        ));
    }
    let (source, tag, selected) = candidates.remove(0);
    for (other_source, other_tag, candidate) in candidates {
        if candidate == selected {
            continue;
        }
        let message = format!(
            "space-group definitions disagree: {source} ({tag}) takes precedence over \
             {other_source} ({other_tag})"
        );
        if strict {
            return Err(import_error(message));
        }
        diagnostics.push(
            CifDiagnostic::warning("conflicting_space_group_definition", message)
                .with_tag(other_tag),
        );
    }
    let mut metadata = BTreeMap::from([("symmetry_source".to_owned(), source.to_owned())]);
    if let Some((_, value)) = hall {
        metadata.insert("space_group_hall".to_owned(), value);
    }
    if let Some((_, value)) = hm {
        metadata.insert("space_group_hm".to_owned(), value);
    }
    if let Some((_, value)) = number_text {
        metadata.insert("space_group_number".to_owned(), value);
    }
    Ok((selected, metadata))
}

fn parse_symmetry_operation(text: &str) -> Result<SymmetryOperation, CifIoError> {
    let expressions = text.split(',').map(str::trim).collect::<Vec<_>>();
    if expressions.len() != 3 {
        return Err(import_error(
            "symmetry operation must contain three expressions",
        ));
    }
    let mut rotation = [[0_i32; 3]; 3];
    let mut translation = [Rational::zero(); 3];
    for (row, expression) in expressions.iter().enumerate() {
        let (coefficients, offset) = parse_affine_expression(expression)?;
        rotation[row] = coefficients;
        translation[row] = offset;
    }
    SymmetryOperation::new(rotation, translation).map_err(CifIoError::Symmetry)
}

fn parse_affine_expression(expression: &str) -> Result<([i32; 3], Rational), CifIoError> {
    let compact = expression
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    if compact.is_empty() {
        return Err(import_error("empty symmetry expression"));
    }
    let mut terms = Vec::new();
    let mut start = 0;
    for (index, character) in compact.char_indices() {
        if index > 0 && matches!(character, '+' | '-') {
            terms.push(&compact[start..index]);
            start = index;
        }
    }
    terms.push(&compact[start..]);
    let mut coefficients = [0_i32; 3];
    let mut offsets = Vec::new();
    for term in terms {
        let (sign, body) = match term.as_bytes().first() {
            Some(b'+') => (1_i32, &term[1..]),
            Some(b'-') => (-1_i32, &term[1..]),
            _ => (1_i32, term),
        };
        if body.is_empty() {
            return Err(import_error("invalid empty symmetry term"));
        }
        if let Some(variable) = body
            .chars()
            .last()
            .filter(|value| matches!(value, 'x' | 'y' | 'z'))
        {
            let coefficient_text = &body[..body.len() - 1];
            let magnitude = if coefficient_text.is_empty() {
                1
            } else {
                coefficient_text.parse::<i32>().map_err(|_| {
                    import_error(format!("invalid symmetry coefficient {coefficient_text:?}"))
                })?
            };
            let column = match variable {
                'x' => 0,
                'y' => 1,
                'z' => 2,
                _ => unreachable!(),
            };
            coefficients[column] = coefficients[column]
                .checked_add(
                    sign.checked_mul(magnitude)
                        .ok_or_else(|| import_error("symmetry coefficient overflow"))?,
                )
                .ok_or_else(|| import_error("symmetry coefficient overflow"))?;
        } else {
            offsets.push(parse_rational_term(body, sign)?);
        }
    }
    let (mut numerator, mut denominator) = (0_i64, 1_i64);
    for (term_numerator, term_denominator) in offsets {
        let common = lcm(denominator, term_denominator)?;
        numerator = numerator
            .checked_mul(common / denominator)
            .and_then(|left| {
                term_numerator
                    .checked_mul(common / term_denominator)
                    .and_then(|right| left.checked_add(right))
            })
            .ok_or_else(|| import_error("symmetry translation overflow"))?;
        denominator = common;
    }
    let offset = Rational::new(numerator, denominator).map_err(CifIoError::Symmetry)?;
    Ok((coefficients, offset))
}

fn parse_rational_term(body: &str, sign: i32) -> Result<(i64, i64), CifIoError> {
    let (numerator, denominator) = if let Some((left, right)) = body.split_once('/') {
        (
            left.parse::<i64>()
                .map_err(|_| import_error(format!("invalid symmetry fraction {body:?}")))?,
            right
                .parse::<i64>()
                .map_err(|_| import_error(format!("invalid symmetry fraction {body:?}")))?,
        )
    } else {
        (
            body.parse::<i64>()
                .map_err(|_| import_error(format!("invalid symmetry offset {body:?}")))?,
            1,
        )
    };
    if denominator == 0 {
        return Err(import_error(
            "symmetry fraction denominator must be non-zero",
        ));
    }
    Ok((numerator * i64::from(sign), denominator))
}

// Site import keeps the strict/permissive decisions together so every skipped
// row follows the same diagnostic and duplicate-label policy.
#[allow(clippy::too_many_lines)]
fn parse_sites(
    block: &CifBlock,
    cell: UnitCell,
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
    limits: CifReadLimits,
) -> Result<Vec<CifAtomSite>, CifIoError> {
    let columns = SiteColumns::from_block(block);
    let labels = columns.label.values;
    let coordinate_presence = [
        columns.fract_x,
        columns.fract_y,
        columns.fract_z,
        columns.cart_x,
        columns.cart_y,
        columns.cart_z,
    ]
    .iter()
    .any(|column| !column.values.is_empty());
    if labels.is_empty() {
        if coordinate_presence {
            return Err(import_error(
                "atom-site coordinates require an atom-site label column",
            ));
        }
        if !column(block, &ANISO_LABEL_TAGS).values.is_empty() {
            return Err(import_error(
                "anisotropic displacement rows require atom-site rows",
            ));
        }
        return Ok(Vec::new());
    }
    if labels.len() > limits.max_atom_sites {
        return Err(CifIoError::Limit {
            message: "CIF atom-site loop exceeds max_atom_sites".to_owned(),
        });
    }
    for column in columns.all() {
        if !column.values.is_empty() && column.values.len() != labels.len() {
            return Err(import_error(format!(
                "atom-site column length mismatch for {}",
                column.tag.unwrap_or("atom_site")
            )));
        }
    }
    let fractional_complete = [columns.fract_x, columns.fract_y, columns.fract_z]
        .iter()
        .all(|column| !column.values.is_empty());
    let cartesian_complete = [columns.cart_x, columns.cart_y, columns.cart_z]
        .iter()
        .all(|column| !column.values.is_empty());
    if !fractional_complete && !cartesian_complete {
        return Err(import_error(
            "atom sites require complete fractional or Cartesian coordinates",
        ));
    }
    let anisotropic = parse_anisotropic(block, diagnostics, strict, limits)?;
    let direct_basis = cell.geometry().map_err(CifIoError::Cell)?.direct_basis;
    let mut sites = Vec::new();
    let mut attached_anisotropic = BTreeSet::new();
    let mut used_ids = BTreeMap::<String, usize>::new();
    for (row, raw_label) in labels.iter().enumerate() {
        let Some(label) = raw_label.text().filter(|value| !value.trim().is_empty()) else {
            if strict {
                return Err(import_error(format!(
                    "atom-site label is missing at row {row}"
                )));
            }
            diagnostics.push(
                CifDiagnostic::warning(
                    "skipped_atom_missing_label",
                    "skipped atom with missing label",
                )
                .with_tag(columns.label.tag.unwrap_or("_atom_site_label"))
                .with_row(row),
            );
            continue;
        };
        let parsed = parse_site_row(&columns, row, label, direct_basis, diagnostics, strict);
        let (
            fractional_xyz,
            coordinate_uncertainty,
            type_symbol,
            element_symbol,
            isotope,
            charge,
            occupancy,
            u_iso,
            u_iso_uncertainty,
        ) = match parsed {
            Ok(values) => values,
            Err(error) if strict => {
                return Err(import_error(format!(
                    "invalid atom site {label:?} at row {row}: {error}"
                )));
            }
            Err(error) => {
                diagnostics.push(
                    CifDiagnostic::warning(
                        "skipped_invalid_atom_site",
                        format!("skipped atom {label:?}: {error}"),
                    )
                    .with_row(row),
                );
                continue;
            }
        };
        let occurrence = used_ids.entry(label.to_owned()).or_default();
        *occurrence += 1;
        let site_id = if *occurrence == 1 {
            label.to_owned()
        } else {
            format!("{label}#{occurrence}")
        };
        if *occurrence > 1 {
            let message = format!("duplicate atom label {label:?} was renamed to {site_id:?}");
            if strict {
                return Err(import_error(message));
            }
            diagnostics.push(
                CifDiagnostic::warning("duplicate_atom_label_renamed", message).with_row(row),
            );
        }
        let occupancy_value = occupancy.map_or(1.0, |value| value.value);
        sites.push(CifAtomSite {
            site_id,
            source_label: label.to_owned(),
            type_symbol,
            element_symbol,
            fractional_xyz,
            occupancy: occupancy_value,
            u_iso_angstrom2: u_iso,
            anisotropic_displacement: anisotropic.get(label).cloned(),
            charge,
            isotope,
            disorder_group: site_text(columns.disorder_group, row).map(str::to_owned),
            fractional_xyz_standard_uncertainty: coordinate_uncertainty,
            occupancy_standard_uncertainty: occupancy.and_then(|value| value.standard_uncertainty),
            u_iso_standard_uncertainty: u_iso_uncertainty,
        });
        if anisotropic.contains_key(label) {
            attached_anisotropic.insert(label.to_owned());
        }
    }
    let orphan_labels = anisotropic
        .keys()
        .filter(|label| !attached_anisotropic.contains(*label))
        .cloned()
        .collect::<Vec<_>>();
    if !orphan_labels.is_empty() {
        let message = format!(
            "anisotropic rows have no matching atom sites: {}",
            orphan_labels.join(", ")
        );
        if strict {
            return Err(import_error(message));
        }
        diagnostics.push(CifDiagnostic::warning("orphan_anisotropic_rows", message));
    }
    Ok(sites)
}

#[allow(clippy::type_complexity)]
fn parse_site_row(
    columns: &SiteColumns<'_>,
    row: usize,
    label: &str,
    direct_basis: [[f64; 3]; 3],
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
) -> Result<
    (
        [f64; 3],
        [Option<f64>; 3],
        String,
        String,
        Option<u32>,
        Option<i32>,
        Option<ParsedNumber>,
        Option<f64>,
        Option<f64>,
    ),
    CifIoError,
> {
    let (fractional_xyz, coordinate_uncertainty) =
        site_coordinates(columns, row, direct_basis, diagnostics, strict)?;
    let type_symbol = site_text(columns.type_symbol, row).map_or_else(
        || symbol_from_label(label).unwrap_or_default(),
        str::to_owned,
    );
    if type_symbol.is_empty() {
        return Err(import_error(format!(
            "cannot infer type symbol from atom label {label:?}"
        )));
    }
    let (element_symbol, isotope, charge) = chemical_identity(&type_symbol)?;
    let occupancy = optional_site_number(columns.occupancy, row, diagnostics)?;
    if occupancy.is_some_and(|value| value.value < 0.0) {
        return Err(import_error("occupancy is negative"));
    }
    let (u_iso, u_iso_uncertainty) = site_u_iso(columns, row, diagnostics, strict)?;
    Ok((
        fractional_xyz,
        coordinate_uncertainty,
        type_symbol,
        element_symbol,
        isotope,
        charge,
        occupancy,
        u_iso,
        u_iso_uncertainty,
    ))
}

fn site_coordinates(
    columns: &SiteColumns<'_>,
    row: usize,
    direct_basis: [[f64; 3]; 3],
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
) -> Result<([f64; 3], [Option<f64>; 3]), CifIoError> {
    let fractional_columns = [columns.fract_x, columns.fract_y, columns.fract_z];
    let cartesian_columns = [columns.cart_x, columns.cart_y, columns.cart_z];
    let fractional = if fractional_columns
        .iter()
        .all(|column| !column.values.is_empty())
    {
        Some(
            fractional_columns
                .map(|column| required_site_number(column, row, diagnostics))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let cartesian = if cartesian_columns
        .iter()
        .all(|column| !column.values.is_empty())
    {
        Some(
            cartesian_columns
                .map(|column| required_site_number(column, row, diagnostics))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let converted = cartesian.as_ref().map(|values| {
        solve_upper_triangular(
            direct_basis,
            [values[0].value, values[1].value, values[2].value],
        )
    });
    if let Some(values) = fractional {
        let coordinates = [values[0].value, values[1].value, values[2].value];
        if let Some(converted) = converted
            && periodic_max_difference(coordinates, converted) > 1.0e-8
        {
            let message = "fractional and Cartesian atom coordinates disagree";
            if strict {
                return Err(import_error(message));
            }
            diagnostics.push(
                CifDiagnostic::warning("conflicting_atom_coordinates", message).with_row(row),
            );
        }
        return Ok((
            coordinates,
            [
                values[0].standard_uncertainty,
                values[1].standard_uncertainty,
                values[2].standard_uncertainty,
            ],
        ));
    }
    let cartesian = cartesian.ok_or_else(|| import_error("missing atom coordinates"))?;
    if cartesian
        .iter()
        .any(|value| value.standard_uncertainty.is_some())
    {
        diagnostics.push(
            CifDiagnostic::warning(
                "cartesian_coordinate_uncertainty_not_transformed",
                "Cartesian coordinate standard uncertainties are not transformed because their \
                 covariance is unavailable",
            )
            .with_row(row),
        );
    }
    Ok((
        converted.ok_or_else(|| import_error("Cartesian coordinate conversion failed"))?,
        [None; 3],
    ))
}

fn site_u_iso(
    columns: &SiteColumns<'_>,
    row: usize,
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
) -> Result<(Option<f64>, Option<f64>), CifIoError> {
    let u_value = optional_site_number(columns.u_iso, row, diagnostics)?;
    let b_value = optional_site_number(columns.b_iso, row, diagnostics)?;
    let converted_b = b_value.map(|value| value.value / B_TO_U);
    if let (Some(u), Some(b)) = (u_value, converted_b)
        && !close(u.value, b, 1.0e-8, 1.0e-12)
    {
        let message = "U_iso and B_iso definitions disagree";
        if strict {
            return Err(import_error(message));
        }
        diagnostics.push(
            CifDiagnostic::warning("conflicting_isotropic_displacement", message).with_row(row),
        );
    }
    if let Some(value) = u_value {
        if value.value < 0.0 {
            return Err(import_error("U_iso is negative"));
        }
        return Ok((Some(value.value), value.standard_uncertainty));
    }
    if let Some(value) = b_value {
        if value.value < 0.0 {
            return Err(import_error("B_iso is negative"));
        }
        return Ok((
            converted_b,
            value
                .standard_uncertainty
                .map(|uncertainty| uncertainty / B_TO_U),
        ));
    }
    Ok((None, None))
}

// Keep U/B precedence, conflict reporting, and row construction in one policy
// function; splitting these branches would duplicate strict-mode decisions.
#[allow(clippy::too_many_lines)]
fn parse_anisotropic(
    block: &CifBlock,
    diagnostics: &mut Vec<CifDiagnostic>,
    strict: bool,
    limits: CifReadLimits,
) -> Result<BTreeMap<String, CifAnisotropicDisplacement>, CifIoError> {
    let labels = column(block, &ANISO_LABEL_TAGS);
    if labels.values.is_empty() {
        return Ok(BTreeMap::new());
    }
    if labels.values.len() > limits.max_atom_sites {
        return Err(CifIoError::Limit {
            message: "CIF anisotropic loop exceeds max_atom_sites".to_owned(),
        });
    }
    let u_columns = ANISO_COMPONENTS.map(|component| {
        column(
            block,
            &[
                &format!("_atom_site_aniso.u_{component}"),
                &format!("_atom_site_aniso_u_{component}"),
            ],
        )
    });
    let b_columns = ANISO_COMPONENTS.map(|component| {
        column(
            block,
            &[
                &format!("_atom_site_aniso.b_{component}"),
                &format!("_atom_site_aniso_b_{component}"),
            ],
        )
    });
    let u_complete = u_columns.iter().all(|column| !column.values.is_empty());
    let b_complete = b_columns.iter().all(|column| !column.values.is_empty());
    let u_present = u_columns.iter().any(|column| !column.values.is_empty());
    let b_present = b_columns.iter().any(|column| !column.values.is_empty());
    if !u_complete && !b_complete {
        let message = "anisotropic loop requires all six U or B components";
        if strict {
            return Err(import_error(message));
        }
        diagnostics.push(
            CifDiagnostic::warning("ignored_incomplete_anisotropic_loop", message)
                .with_tag(labels.tag.unwrap_or("_atom_site_aniso_label")),
        );
        return Ok(BTreeMap::new());
    }
    if (u_present && !u_complete) || (b_present && !b_complete) {
        let message = "secondary anisotropic U or B definition is incomplete";
        if strict {
            return Err(import_error(message));
        }
        diagnostics.push(
            CifDiagnostic::warning("ignored_incomplete_anisotropic_definition", message)
                .with_tag(labels.tag.unwrap_or("_atom_site_aniso_label")),
        );
    }
    let (selected, convention) = if u_complete {
        (&u_columns, DisplacementConvention::CifU)
    } else {
        (&b_columns, DisplacementConvention::CifB)
    };
    let mut result = BTreeMap::new();
    for (row, raw_label) in labels.values.iter().enumerate() {
        let label = raw_label
            .text()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| import_error(format!("anisotropic label is missing at row {row}")))?;
        if result.contains_key(label) {
            let message = format!("duplicate anisotropic label {label:?} at row {row}");
            if strict {
                return Err(import_error(message));
            }
            diagnostics.push(
                CifDiagnostic::warning("duplicate_anisotropic_label_ignored", message)
                    .with_row(row),
            );
            continue;
        }
        let parsed = selected
            .iter()
            .map(|column| required_site_number(*column, row, diagnostics))
            .collect::<Result<Vec<_>, _>>()?;
        let mut components: [f64; 6] = parsed
            .iter()
            .map(|value| value.value)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| import_error("invalid anisotropic component count"))?;
        let mut uncertainties: [Option<f64>; 6] = parsed
            .iter()
            .map(|value| value.standard_uncertainty)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| import_error("invalid anisotropic uncertainty count"))?;
        if convention == DisplacementConvention::CifB {
            components = components.map(|value| value / B_TO_U);
            uncertainties = uncertainties.map(|value| value.map(|item| item / B_TO_U));
        }
        if u_complete && b_complete {
            let converted_b = b_columns
                .iter()
                .map(|column| {
                    required_site_number(*column, row, diagnostics)
                        .map(|value| value.value / B_TO_U)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if components
                .iter()
                .zip(converted_b)
                .any(|(left, right)| !close(*left, right, 1.0e-8, 1.0e-12))
            {
                let message = format!("anisotropic U and B definitions disagree for {label:?}");
                if strict {
                    return Err(import_error(message));
                }
                diagnostics.push(
                    CifDiagnostic::warning("conflicting_anisotropic_displacement", message)
                        .with_row(row),
                );
            }
        }
        result.insert(
            label.to_owned(),
            CifAnisotropicDisplacement {
                u_cif_angstrom2: components,
                source_convention: convention,
                standard_uncertainty: uncertainties,
            },
        );
    }
    Ok(result)
}

fn required_site_number(
    column: SiteColumn<'_>,
    row: usize,
    diagnostics: &mut Vec<CifDiagnostic>,
) -> Result<ParsedNumber, CifIoError> {
    let tag = column.tag.unwrap_or("atom_site");
    let value = column
        .values
        .get(row)
        .ok_or_else(|| import_error(format!("required atom-site column is absent: {tag}")))?;
    parse_number(value, tag, diagnostics, true, Some(row))?
        .ok_or_else(|| import_error(format!("required atom-site value is missing: {tag}")))
}

fn optional_site_number(
    column: SiteColumn<'_>,
    row: usize,
    diagnostics: &mut Vec<CifDiagnostic>,
) -> Result<Option<ParsedNumber>, CifIoError> {
    let Some(value) = column.values.get(row) else {
        return Ok(None);
    };
    parse_number(
        value,
        column.tag.unwrap_or("atom_site"),
        diagnostics,
        false,
        Some(row),
    )
}

fn parse_number(
    raw: &CifValue,
    tag: &str,
    diagnostics: &mut Vec<CifDiagnostic>,
    required: bool,
    row: Option<usize>,
) -> Result<Option<ParsedNumber>, CifIoError> {
    let Some(text) = raw.text() else {
        let state = raw.missing_state().unwrap_or("missing");
        let mut diagnostic = CifDiagnostic::warning(
            format!("{state}_cif_value"),
            format!("{state} CIF value for {tag}"),
        )
        .with_tag(tag);
        if let Some(row) = row {
            diagnostic = diagnostic.with_row(row);
        }
        diagnostics.push(diagnostic);
        if required {
            return Err(import_error(format!(
                "required CIF value for {tag} is {state}"
            )));
        }
        return Ok(None);
    };
    let (mantissa_and_uncertainty, exponent) = split_exponent(text)?;
    let (mantissa, uncertainty_digits) = split_uncertainty(mantissa_and_uncertainty)?;
    let value = mantissa
        .parse::<f64>()
        .map_err(|_| import_error(format!("CIF value for {tag} is not a number: {text:?}")))?
        * 10.0_f64.powi(exponent);
    let uncertainty = if let Some(digits) = uncertainty_digits {
        if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(import_error(format!(
                "CIF value for {tag} is not a number: {text:?}"
            )));
        }
        let decimal_places = mantissa
            .split_once('.')
            .map_or(0, |(_, decimals)| decimals.len());
        let decimal_places = i32::try_from(decimal_places)
            .map_err(|_| import_error("CIF numeric precision is too large"))?;
        Some(
            digits.parse::<f64>().map_err(|_| {
                import_error(format!("CIF value for {tag} is not a number: {text:?}"))
            })? * 10.0_f64.powi(exponent - decimal_places),
        )
    } else {
        None
    };
    if !value.is_finite() || uncertainty.is_some_and(|item| !item.is_finite()) {
        return Err(import_error(format!("CIF value for {tag} is not finite")));
    }
    Ok(Some(ParsedNumber {
        value,
        standard_uncertainty: uncertainty,
    }))
}

fn split_exponent(text: &str) -> Result<(&str, i32), CifIoError> {
    if let Some(index) = text
        .char_indices()
        .skip(1)
        .find_map(|(index, character)| matches!(character, 'e' | 'E').then_some(index))
    {
        let exponent = text[index + 1..]
            .parse::<i32>()
            .map_err(|_| import_error(format!("invalid CIF exponent in {text:?}")))?;
        Ok((&text[..index], exponent))
    } else {
        Ok((text, 0))
    }
}

fn split_uncertainty(text: &str) -> Result<(&str, Option<&str>), CifIoError> {
    if !text.ends_with(')') {
        return Ok((text, None));
    }
    let open = text
        .rfind('(')
        .ok_or_else(|| import_error(format!("invalid CIF uncertainty in {text:?}")))?;
    Ok((&text[..open], Some(&text[open + 1..text.len() - 1])))
}

fn first_tag_text<'a>(block: &'a CifBlock, aliases: &[&str]) -> Option<(&'a str, String)> {
    block
        .first_raw(aliases)
        .and_then(|(tag, value)| value.text().map(|text| (tag, text.trim().to_owned())))
        .filter(|(_, value)| !value.is_empty())
}

fn first_text(block: &CifBlock, aliases: &[&str]) -> Option<String> {
    first_tag_text(block, aliases).map(|(_, value)| value)
}

fn column<'a>(block: &'a CifBlock, aliases: &[&str]) -> SiteColumn<'a> {
    block
        .first_column(aliases)
        .map_or_else(SiteColumn::empty, |(tag, values)| SiteColumn {
            tag: Some(tag),
            values,
        })
}

fn site_text(column: SiteColumn<'_>, row: usize) -> Option<&str> {
    column.values.get(row).and_then(CifValue::text)
}

fn symbol_from_label(label: &str) -> Option<String> {
    let mut characters = label.chars();
    let first = characters.next()?;
    if !first.is_ascii_uppercase() {
        return None;
    }
    let mut symbol = first.to_string();
    if let Some(second) = characters.next().filter(char::is_ascii_lowercase) {
        symbol.push(second);
    }
    Some(symbol)
}

fn chemical_identity(type_symbol: &str) -> Result<(String, Option<u32>, Option<i32>), CifIoError> {
    let bytes = type_symbol.as_bytes();
    let isotope_end = bytes
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(bytes.len());
    let isotope = if isotope_end == 0 {
        None
    } else {
        Some(
            type_symbol[..isotope_end]
                .parse::<u32>()
                .map_err(|_| import_error(format!("invalid atom type symbol {type_symbol:?}")))?,
        )
    };
    let rest = &type_symbol[isotope_end..];
    let mut characters = rest.char_indices();
    let (_, first) = characters
        .next()
        .filter(|(_, value)| value.is_ascii_uppercase())
        .ok_or_else(|| import_error(format!("invalid atom type symbol {type_symbol:?}")))?;
    let mut element = first.to_string();
    let mut element_end = first.len_utf8();
    if let Some((index, second)) = characters.next()
        && second.is_ascii_lowercase()
    {
        element.push(second);
        element_end = index + second.len_utf8();
    }
    let mut isotope = isotope;
    if element == "D" {
        "H".clone_into(&mut element);
        isotope = Some(2);
    } else if element == "T" {
        "H".clone_into(&mut element);
        isotope = Some(3);
    }
    let suffix = &rest[element_end..];
    let charge = parse_charge(suffix)?;
    Ok((element, isotope, charge))
}

fn parse_charge(suffix: &str) -> Result<Option<i32>, CifIoError> {
    if suffix.is_empty() {
        return Ok(None);
    }
    let (sign, magnitude) = if let Some(value) = suffix.strip_suffix('+') {
        (1, value)
    } else if let Some(value) = suffix.strip_suffix('-') {
        (-1, value)
    } else if let Some(value) = suffix.strip_prefix('+') {
        (1, value)
    } else if let Some(value) = suffix.strip_prefix('-') {
        (-1, value)
    } else {
        return Ok(None);
    };
    let magnitude = if magnitude.is_empty() {
        1
    } else {
        magnitude
            .parse::<i32>()
            .map_err(|_| import_error(format!("invalid atom charge {suffix:?}")))?
    };
    Ok(Some(sign * magnitude))
}

fn solve_upper_triangular(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    let z = vector[2] / matrix[2][2];
    let y = (vector[1] - matrix[1][2] * z) / matrix[1][1];
    let x = (vector[0] - matrix[0][1] * y - matrix[0][2] * z) / matrix[0][0];
    [x, y, z]
}

fn periodic_max_difference(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = (left - right).abs().rem_euclid(1.0);
            difference.min(1.0 - difference)
        })
        .fold(0.0, f64::max)
}

fn validate_cell_metric(cell: UnitCell, space_group: &SpaceGroup) -> Result<(), CifIoError> {
    let metric = cell.geometry().map_err(CifIoError::Cell)?.direct_metric;
    let components = [
        metric[0][0],
        metric[1][1],
        metric[2][2],
        metric[1][2],
        metric[0][2],
        metric[0][1],
    ];
    let scale = components
        .iter()
        .map(|value| value.abs())
        .fold(1.0, f64::max);
    for equation in &space_group.metric_constraints().equations {
        let mut signed_residual = 0.0;
        let mut coefficient_scale = 0.0;
        for (&coefficient, value) in equation.iter().zip(components) {
            let coefficient = i32::try_from(coefficient)
                .map_err(|_| import_error("space-group metric coefficient is out of range"))?;
            signed_residual += f64::from(coefficient) * value;
            coefficient_scale += f64::from(coefficient.unsigned_abs());
        }
        let residual = signed_residual.abs();
        let coefficient_scale = coefficient_scale.max(1.0);
        if residual > 1.0e-10 * scale * coefficient_scale {
            return Err(import_error(
                "unit-cell metric is incompatible with the structure space group",
            ));
        }
    }
    Ok(())
}

fn stable_structure_id(block_name: &str) -> String {
    let value = block_name
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_owned();
    if value.is_empty() {
        "structure".to_owned()
    } else {
        value
    }
}

fn close(left: f64, right: f64, relative: f64, absolute: f64) -> bool {
    (left - right).abs() <= absolute + relative * right.abs()
}

fn gcd(left: i64, right: i64) -> i64 {
    let mut left = left.unsigned_abs();
    let mut right = right.unsigned_abs();
    while right != 0 {
        (left, right) = (right, left % right);
    }
    i64::try_from(left.max(1)).unwrap_or(i64::MAX)
}

fn lcm(left: i64, right: i64) -> Result<i64, CifIoError> {
    (left / gcd(left, right))
        .checked_mul(right)
        .ok_or_else(|| import_error("symmetry translation overflow"))
}
