//! Assign or recognize redundant tensor rows using independent site-table evidence.

use super::{
    BTreeMap, BTreeSet, CifAnisotropicDisplacement, CifAtomSite, CifBlock, CifDiagnostic,
    CifIoError, CifValue, UnitCell, import_error,
};

type Columns<'a> = BTreeMap<String, &'a [CifValue]>;

fn text<'a>(columns: &Columns<'a>, tag: &str, row: usize) -> Option<&'a str> {
    columns.get(tag)?.get(row)?.text()
}

fn adp_kind<'a>(columns: &Columns<'a>, row: usize) -> Option<&'a str> {
    let modern = text(columns, "_atom_site_adp_type", row);
    let legacy = text(columns, "_atom_site_thermal_displace_type", row);
    match (modern, legacy) {
        (Some(a), Some(b)) if !a.eq_ignore_ascii_case(b) => None,
        _ => modern.or(legacy),
    }
}

// For a derived Ueq compare the nominal value at its printed decimal precision.
fn rounds_to(derived: f64, reported: f64, literal: &str) -> bool {
    let Ok((mantissa, exponent)) = super::split_exponent(literal) else {
        return false;
    };
    let Ok((nominal, _)) = super::split_uncertainty(mantissa) else {
        return false;
    };
    let Some((_, fraction)) = nominal.split_once('.') else {
        return false;
    };
    if fraction.is_empty() || fraction.len() > 15 || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let precision = i32::try_from(fraction.len()).expect("bounded precision");
    exponent.checked_sub(precision).is_some_and(|power| {
        let interval = 0.5 * 10.0_f64.powi(power);
        interval.is_finite() && (derived - reported).abs() < interval
    })
}

pub(super) fn attach_repeated(
    block: &CifBlock,
    cell: UnitCell,
    sites: &mut [CifAtomSite],
    groups: BTreeMap<String, Vec<CifAnisotropicDisplacement>>,
    verified: &mut BTreeSet<String>,
    diagnostics: &mut Vec<CifDiagnostic>,
) -> Result<(), CifIoError> {
    if groups.values().all(Vec::is_empty) {
        return Ok(());
    }
    let invalid = |label: &str| {
        import_error(format!(
            "duplicate atom label {label:?} has an ambiguous displacement tensor"
        ))
    };
    let columns = block.proof_table("_atom_site");
    let tensor_columns = block.proof_table("_atom_site_aniso");
    let geometry = cell.geometry().map_err(CifIoError::Cell)?;
    let reciprocal = [0, 1, 2].map(|i| geometry.reciprocal_metric[i][i].sqrt());
    let ij = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let weights = ij.map(|(i, j)| {
        geometry.direct_metric[i][j]
            * reciprocal[i]
            * reciprocal[j]
            * (if i == j { 1.0 } else { 2.0 })
            / 3.0
    });
    let mut indices = BTreeMap::<String, Vec<usize>>::new();
    for (row, site) in sites.iter().enumerate() {
        if groups.contains_key(&site.source_label) {
            indices
                .entry(site.source_label.clone())
                .or_default()
                .push(row);
        }
    }
    for (label, tensors) in groups.into_iter().filter(|(_, v)| !v.is_empty()) {
        let err = || invalid(&label);
        let columns = columns.as_ref().ok_or_else(err)?;
        if tensor_columns.is_none() {
            return Err(err());
        }
        let rows = &indices[&label];
        // Require an explicit Uani/Uiso declaration for every possible owner.
        if rows.len() != 2 {
            return Err(err());
        }
        let mut anisotropic = Vec::new();
        for &row in rows {
            let kind = adp_kind(columns, row).ok_or_else(err)?;
            if kind.eq_ignore_ascii_case("Uani") {
                anisotropic.push(row);
            } else if !kind.eq_ignore_ascii_case("Uiso") {
                return Err(err());
            }
        }
        if tensors.len() != anisotropic.len() {
            return Err(err());
        }
        for tensor in tensors {
            let equivalent = weights
                .iter()
                .zip(tensor.u_cif_angstrom2)
                .map(|(w, u)| w * u)
                .sum();
            let mut candidates = anisotropic.iter().copied().filter(|&row| {
                sites[row]
                    .u_iso_angstrom2
                    .zip(text(columns, "_atom_site_u_iso_or_equiv", row))
                    .is_some_and(|(u, raw)| rounds_to(equivalent, u, raw))
            });
            let row = candidates.next().ok_or_else(err)?;
            if candidates.next().is_some() || sites[row].anisotropic_displacement.is_some() {
                return Err(err());
            }
            diagnostics.push(CifDiagnostic::warning("matched_tensor_by_u_equiv",
                format!("matched tensor {label:?} to site {:?} using its equivalent isotropic displacement", sites[row].site_id)).with_row(row));
            sites[row].anisotropic_displacement = Some(tensor);
        }
    }
    verified.insert("_atom_site_aniso_label".to_owned());
    Ok(())
}

pub(super) fn remove_redundant(
    block: &CifBlock,
    sites: &[CifAtomSite],
    orphans: &mut Vec<String>,
    diagnostics: &mut Vec<CifDiagnostic>,
) {
    let Some(columns) = block.proof_table("_atom_site") else {
        return;
    };
    let Some(tensors) = block.proof_table("_atom_site_aniso") else {
        return;
    };
    let Some(labels) = tensors.get("_atom_site_aniso_label") else {
        return;
    };
    let rows: BTreeMap<_, _> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, v)| Some((v.text()?, i)))
        .collect();
    if rows.len() != labels.len() {
        return;
    }
    let mut parents = BTreeMap::new();
    let mut assemblies = BTreeMap::<&str, BTreeSet<&str>>::new();
    for (row, site) in sites.iter().enumerate() {
        if parents.insert(site.source_label.as_str(), row).is_some() {
            return;
        }
        if let (Some(a), Some(g)) = (
            text(&columns, "_atom_site_disorder_assembly", row),
            site.disorder_group.as_deref(),
        ) {
            assemblies.entry(a).or_default().insert(g);
        }
    }
    let requested = orphans.iter().map(String::as_str).collect();
    let verified = BTreeSet::from(["_atom_site_aniso_label".to_owned()]);
    let referenced: BTreeSet<String> = block
        .referenced_atom_labels("", &requested, &verified)
        .into_iter()
        .map(str::to_owned)
        .collect();
    orphans.retain(|orphan| {
        let Some(parent) = orphan.strip_suffix('A') else {
            return true;
        };
        let (Some(&row), Some(&parent_tensor), Some(&extra_tensor)) =
            (parents.get(parent), rows.get(parent), rows.get(orphan.as_str()))
        else {
            return true;
        };
        let site = &sites[row];
        let has_disorder = site.disorder_group.is_some()
            && text(&columns, "_atom_site_disorder_assembly", row)
                .and_then(|id| assemblies.get(id))
                .is_some_and(|groups| groups.len() > 1);
        let same_row = tensors.iter().all(|(tag, values)| {
            tag == "_atom_site_aniso_label" || values[parent_tensor] == values[extra_tensor]
        });
        let explicit_full_occupancy = text(&columns, "_atom_site_occupancy", row).is_some()
            && site.occupancy.to_bits() == 1.0_f64.to_bits();
        if referenced.contains(orphan) || site.anisotropic_displacement.is_none()
            || !explicit_full_occupancy || !has_disorder || !same_row
            || !adp_kind(&columns, row).is_some_and(|kind| kind.eq_ignore_ascii_case("Uani"))
        {
            return true;
        }
        let message = format!("anisotropic row {orphan:?} duplicates the full tensor row for fully occupied disorder site {parent:?}");
        diagnostics.push(CifDiagnostic::warning("redundant_disorder_tensor", message).with_row(extra_tensor));
        false
    });
}
