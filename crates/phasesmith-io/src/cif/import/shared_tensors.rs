//! Resolve explicitly identified mixed-site displacement tensors.

use super::{
    BTreeMap, BTreeSet, CifAnisotropicDisplacement, CifBlock, CifDiagnostic, SiteColumns, column,
    site_text,
};

pub(super) fn attach(
    tensors: &mut BTreeMap<String, CifAnisotropicDisplacement>,
    block: &CifBlock,
    columns: &SiteColumns<'_>,
    diagnostics: &mut Vec<CifDiagnostic>,
) {
    if tensors.is_empty() {
        return;
    }
    let kind = column(
        block,
        &[
            "_atom_site_adp_type",
            "_atom_site.adp_type",
            "_atom_site_thermal_displace_type",
            "_atom_site.thermal_displace_type",
        ],
    );
    if kind.values.len() != columns.label.values.len() {
        return;
    }
    let mut rows = BTreeMap::new();
    let mut inconsistent_species = BTreeSet::new();
    let mut parents = BTreeMap::<&str, Vec<usize>>::new();
    for (row, value) in columns.label.values.iter().enumerate() {
        let Some(label) = value.text() else { return };
        if rows.insert(label, row).is_some() {
            return;
        }
        if let Some((parent, species)) = label.strip_suffix(')').and_then(|s| s.rsplit_once('('))
            && !parent.is_empty()
        {
            if site_text(columns.type_symbol, row) != Some(species) {
                inconsistent_species.insert(parent);
            }
            parents.entry(parent).or_default().push(row);
        }
    }
    let mut proposals = Vec::new();
    let mut claims = vec![0_usize; rows.len()];
    for source in tensors
        .keys()
        .filter(|key| !rows.contains_key(key.as_str()))
    {
        let members = if source.contains('/') {
            source
                .split('/')
                .map(|s| rows.get(s).copied())
                .collect::<Option<Vec<_>>>()
        } else if inconsistent_species.contains(source.as_str()) {
            None
        } else {
            parents.get(source.as_str()).cloned()
        };
        let Some(members) = members else { continue };
        let coordinates = [columns.fract_x, columns.fract_y, columns.fract_z];
        let shared = members.iter().all(|&row| {
            site_text(kind, row)
                .is_some_and(|s| s.eq_ignore_ascii_case("Uani") || s.eq_ignore_ascii_case("Bani"))
                && !tensors.contains_key(site_text(columns.label, row).unwrap_or(""))
                && coordinates.iter().all(|&axis| {
                    site_text(axis, row).is_some()
                        && site_text(axis, row) == site_text(axis, members[0])
                })
        });
        if shared {
            for &row in &members {
                claims[row] += 1;
            }
            proposals.push((source.clone(), members));
        }
    }
    for (source, members) in proposals {
        if members.iter().any(|&row| claims[row] != 1) {
            continue;
        }
        let Some(tensor) = tensors.remove(&source) else {
            continue;
        };
        for row in members {
            let label = site_text(columns.label, row).expect("validated site label");
            tensors.insert(label.to_owned(), tensor.clone());
            diagnostics.push(
                CifDiagnostic::warning(
                    "matched_mixed_site_tensor",
                    format!("matched anisotropic label {source:?} to atom-site label {label:?}"),
                )
                .with_row(row),
            );
        }
    }
}
