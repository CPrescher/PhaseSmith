//! Resolve geometry references to repeated atom labels in the deposited cell.

use super::{BTreeMap, BTreeSet, CifAtomSite, CifBlock, CifDiagnostic, CifValue, parse_number};

type Edge = [usize; 2];
type Positions = [[f64; 3]];

fn edge(i: usize, j: usize) -> Edge {
    [i.min(j), i.max(j)]
}

fn vector(points: &Positions, i: usize, j: usize) -> [f64; 3] {
    std::array::from_fn(|k| points[i][k] - points[j][k])
}

fn dot(x: [f64; 3], y: [f64; 3]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

struct GeometryTable<'a, const N: usize> {
    labels: [&'a [CifValue]; N],
    symmetry: [Option<&'a [CifValue]>; N],
    values: &'a [CifValue],
}

impl<'a, const N: usize> GeometryTable<'a, N> {
    fn read(block: &'a CifBlock, category: &str, value_tag: &str) -> Option<Self> {
        let columns = block.proof_table(category)?;
        let mut labels = [&[][..]; N];
        let mut symmetry = [None; N];
        for i in 0..N {
            labels[i] = columns.get(&format!("{category}_atom_site_label_{}", i + 1))?;
            symmetry[i] = columns
                .get(&format!("{category}_site_symmetry_{}", i + 1))
                .copied();
        }
        Some(Self {
            labels,
            symmetry,
            values: columns.get(value_tag)?,
        })
    }

    fn candidates<'b>(
        &self,
        row: usize,
        indices: &'b BTreeMap<&str, Vec<usize>>,
    ) -> Option<[&'b [usize]; N]> {
        // Only the as-deposited image is supported. '?' does not establish an image.
        if !self
            .symmetry
            .iter()
            .flatten()
            .all(|column| matches!(column[row], CifValue::Inapplicable))
        {
            return None;
        }
        let mut candidates = [&[][..]; N];
        for (slot, labels) in candidates.iter_mut().zip(self.labels) {
            *slot = indices.get(labels[row].text()?)?;
            // Resolve pairs of like-named sites; bound the number of combinations per row.
            if slot.len() > 2 {
                return None;
            }
        }
        Some(candidates)
    }

    fn touches(&self, row: usize, repeated: &BTreeSet<&str>) -> bool {
        self.labels
            .iter()
            .any(|c| c[row].text().is_some_and(|s| repeated.contains(s)))
    }

    fn measurement(&self, row: usize) -> Option<(f64, f64)> {
        let n = parse_number(
            &self.values[row],
            "geometry",
            &mut Vec::new(),
            true,
            Some(row),
        )
        .ok()??;
        let su = n.standard_uncertainty?;
        (su > 0.0).then_some((n.value, su))
    }
}

fn bond_edges(
    bonds: &GeometryTable<'_, 2>,
    points: &Positions,
    indices: &BTreeMap<&str, Vec<usize>>,
    repeated: &BTreeSet<&str>,
) -> Option<(BTreeSet<Edge>, usize)> {
    let mut edges = BTreeSet::new();
    let mut count = 0;
    for row in 0..bonds.values.len() {
        let affected = bonds.touches(row, repeated);
        let Some([a, b]) = bonds.candidates(row, indices) else {
            if affected {
                return None;
            }
            continue;
        };
        let Some((length, su)) = bonds.measurement(row) else {
            if affected {
                return None;
            }
            continue;
        };
        if !affected {
            // An angle may depend on this uniquely labelled leg. Verify the
            // measured length in the same image before using it as evidence.
            let delta = vector(points, a[0], b[0]);
            if (dot(delta, delta).sqrt() - length).abs() <= su {
                edges.insert(edge(a[0], b[0]));
            }
            continue;
        }
        let mut plausible = a
            .iter()
            .flat_map(|&i| b.iter().map(move |&j| (i, j)))
            .filter(|(i, j)| i != j)
            .filter_map(|(i, j)| {
                let delta = vector(points, i, j);
                let residual = (dot(delta, delta).sqrt() - length).abs();
                (residual <= 3.0 * su).then_some((edge(i, j), residual))
            });
        let (pair, residual) = plausible.next()?;
        if residual > su || plausible.next().is_some() {
            return None;
        }
        edges.insert(pair);
        count += 1;
    }
    Some((edges, count))
}

fn angles_match(
    angles: &GeometryTable<'_, 3>,
    points: &Positions,
    indices: &BTreeMap<&str, Vec<usize>>,
    edges: &BTreeSet<Edge>,
    repeated: &BTreeSet<&str>,
) -> Option<usize> {
    let mut count = 0;
    for row in (0..angles.values.len()).filter(|&r| angles.touches(r, repeated)) {
        let (angle, su) = angles.measurement(row)?;
        let [first, middle, last] = angles.candidates(row, indices)?;
        let mut supported = 0;
        for &j in middle {
            for &i in first.iter().filter(|&&i| edges.contains(&edge(i, j))) {
                for &k in last
                    .iter()
                    .filter(|&&k| k != i && edges.contains(&edge(k, j)))
                {
                    let v = vector(points, i, j);
                    let w = vector(points, k, j);
                    let cosine = dot(v, w) / (dot(v, v) * dot(w, w)).sqrt();
                    let calculated = cosine.clamp(-1.0, 1.0).acos().to_degrees();
                    if !calculated.is_finite() || (calculated - angle).abs() > su {
                        return None;
                    }
                    supported += 1;
                }
            }
        }
        if supported != 1 {
            return None;
        }
        count += 1;
    }
    Some(count)
}

pub(super) fn verify(
    block: &CifBlock,
    sites: &[CifAtomSite],
    repeated: &BTreeSet<&str>,
    basis: [[f64; 3]; 3],
    verified: &mut BTreeSet<String>,
    diagnostics: &mut Vec<CifDiagnostic>,
) {
    if block.proof_table("_atom_site").is_none() {
        return;
    }
    let Some(bonds) = GeometryTable::read(block, "_geom_bond", "_geom_bond_distance") else {
        return;
    };
    let mut indices = BTreeMap::<&str, Vec<usize>>::new();
    let points: Vec<_> = sites
        .iter()
        .enumerate()
        .map(|(row, site)| {
            indices.entry(&site.source_label).or_default().push(row);
            basis.map(|axis| dot(axis, site.fractional_xyz))
        })
        .collect();
    let Some((edges, n_bonds)) = bond_edges(&bonds, &points, &indices, repeated) else {
        return;
    };
    let mut n_angles = 0;
    if let Some(angles) = GeometryTable::read(block, "_geom_angle", "_geom_angle") {
        let Some(n) = angles_match(&angles, &points, &indices, &edges, repeated) else {
            return;
        };
        n_angles = n;
        verified.extend((1..=3).map(|i| format!("_geom_angle_atom_site_label_{i}")));
    }
    verified.extend((1..=2).map(|i| format!("_geom_bond_atom_site_label_{i}")));
    if n_bonds + n_angles > 0 {
        diagnostics.push(CifDiagnostic::warning("resolved_geometry_labels",
            format!("resolved repeated atom-site labels in {n_bonds} bond and {n_angles} angle rows using the deposited geometry")));
    }
}
