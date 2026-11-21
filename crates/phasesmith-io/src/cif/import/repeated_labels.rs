//! Check that repeated source labels identify distinct atoms.

use super::{
    BTreeMap, BTreeSet, CifAtomSite, CifIoError, SpaceGroup, import_error, periodic_max_difference,
};

const POSITION_TOLERANCE: f64 = 1.0e-10;
const OCCUPANCY_TOLERANCE: f64 = 1.0e-10;
const BINS: i32 = 1_000_000;

#[allow(clippy::cast_possible_truncation)]
fn bin(wrapped: f64) -> i32 {
    // Inputs are wrapped fractions, plus at most one tolerance at a periodic edge.
    (wrapped * f64::from(BINS)).floor() as i32
}

pub(super) fn validate(
    sites: &[CifAtomSite],
    repeated: &BTreeSet<&str>,
    group: &SpaceGroup,
) -> Result<(), CifIoError> {
    // Index deposited positions so that many independent atoms sharing a
    // label do not require all-pairs comparisons. The final test is periodic
    // distance, independent of which spatial bin contains a position.
    let mut bins = BTreeMap::<(&str, [i32; 3]), Vec<usize>>::new();
    for (index, site) in sites
        .iter()
        .enumerate()
        .filter(|(_, s)| repeated.contains(s.source_label.as_str()))
    {
        let label = site.source_label.as_str();
        if site.anisotropic_displacement.is_some() {
            return Err(import_error(format!(
                "duplicate atom label {label:?} has an ambiguous displacement tensor"
            )));
        }
        let mut coincident = BTreeSet::new();
        for op in group.operations() {
            let image = op.apply_fractional(site.fractional_xyz);
            let [xs, ys, zs] =
                image.map(|v| bin(v - POSITION_TOLERANCE)..=bin(v + POSITION_TOLERANCE));
            for x in xs {
                for y in ys.clone() {
                    for z in zs.clone() {
                        let key = (label, [x, y, z].map(|n| n.rem_euclid(BINS)));
                        if let Some(indices) = bins.get(&key) {
                            coincident.extend(indices.iter().copied().filter(|&i| {
                                periodic_max_difference(image, sites[i].fractional_xyz)
                                    <= POSITION_TOLERANCE
                            }));
                        }
                    }
                }
            }
        }
        let mut occupancy = site.occupancy;
        for i in coincident {
            let other = &sites[i];
            occupancy += other.occupancy;
            if (site.element_symbol == other.element_symbol && site.isotope == other.isotope)
                || occupancy > 1.0 + OCCUPANCY_TOLERANCE
            {
                return Err(import_error(format!(
                    "duplicate atom label {label:?} has indistinguishable or overoccupied symmetry-equivalent sites"
                )));
            }
        }
        let key = site
            .fractional_xyz
            .map(|v| bin(v.rem_euclid(1.0)).rem_euclid(BINS));
        bins.entry((label, key)).or_default().push(index);
    }
    Ok(())
}
