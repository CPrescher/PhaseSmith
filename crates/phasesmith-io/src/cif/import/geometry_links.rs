//! Resolve geometry references to repeated atom labels in the deposited cell.

use super::{BTreeMap, BTreeSet, CifAtomSite, CifBlock, CifDiagnostic, CifValue, parse_number};

type Edge = [usize; 2];
// A coarse fractional coordinate cannot establish a riding-hydrogen bond.
// Require sub-percent resolution on each vector used as rounding evidence.
const MAX_RELATIVE_ROUNDING: f64 = 0.01;

struct Positions {
    cartesian: Vec<[f64; 3]>,
    // Cartesian radius enclosing the rounding box of the printed fractional coordinates.
    rounding: Vec<Option<f64>>,
    riding_h: Vec<bool>,
}

fn half_decimal(literal: &str) -> Option<f64> {
    if literal.contains(['e', 'E']) {
        return None;
    }
    let nominal = literal.split('(').next()?;
    let (_, digits) = nominal.split_once('.')?;
    if digits.is_empty() || digits.len() > 15 || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(0.5 * 10_f64.powi(-i32::try_from(digits.len()).ok()?))
}

enum Precision {
    Measured(f64),
    Calculated(f64),
}

impl Precision {
    fn tolerance(&self, rounding: impl FnOnce() -> Option<f64>) -> Option<f64> {
        match self {
            Self::Measured(su) => Some(*su),
            Self::Calculated(half_unit) => Some(half_unit + rounding()?),
        }
    }
}

fn edge(i: usize, j: usize) -> Edge {
    [i.min(j), i.max(j)]
}

fn vector(points: &Positions, i: usize, j: usize) -> [f64; 3] {
    std::array::from_fn(|k| points.cartesian[i][k] - points.cartesian[j][k])
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

    fn measurement(&self, row: usize) -> Option<(f64, Precision)> {
        let n = parse_number(
            &self.values[row],
            "geometry",
            &mut Vec::new(),
            true,
            Some(row),
        )
        .ok()??;
        let precision = match n.standard_uncertainty {
            Some(su) if su > 0.0 => Precision::Measured(su),
            None => {
                let raw = self.values[row].text()?;
                if raw.contains('(') {
                    return None;
                }
                Precision::Calculated(half_decimal(raw)?)
            }
            _ => return None,
        };
        Some((n.value, precision))
    }
}

// Only use geometry without a reported uncertainty for explicitly calculated
// riding hydrogens. Every alternative for the repeated label must qualify.
fn riding_pair(groups: &[&[usize]], points: &Positions) -> bool {
    groups
        .iter()
        .filter(|g| g.len() > 1)
        .all(|g| g.iter().all(|&i| points.riding_h[i]))
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
        let Some((length, precision)) = bonds.measurement(row) else {
            if affected {
                return None;
            }
            continue;
        };
        if !affected {
            let delta = vector(points, a[0], b[0]);
            if let Precision::Measured(su) = precision
                && (dot(delta, delta).sqrt() - length).abs() <= su
            {
                edges.insert(edge(a[0], b[0]));
            }
            continue;
        }
        if matches!(precision, Precision::Calculated(_)) && !riding_pair(&[a, b], points) {
            return None;
        }
        let mut plausible = None;
        for &i in a {
            for &j in b.iter().filter(|&&j| j != i) {
                let tolerance =
                    precision.tolerance(|| Some(points.rounding[i]? + points.rounding[j]?))?;
                if matches!(precision, Precision::Calculated(_))
                    && tolerance >= MAX_RELATIVE_ROUNDING * length
                {
                    return None;
                }
                let delta = vector(points, i, j);
                let residual = (dot(delta, delta).sqrt() - length).abs();
                if residual <= 3.0 * tolerance {
                    if plausible.is_some() || residual > tolerance {
                        return None;
                    }
                    plausible = Some(edge(i, j));
                }
            }
        }
        let pair = plausible?;
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
        let (angle, precision) = angles.measurement(row)?;
        let [first, middle, last] = angles.candidates(row, indices)?;
        if matches!(precision, Precision::Calculated(_))
            && !riding_pair(&[first, middle, last], points)
        {
            return None;
        }
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
                    let tolerance = precision.tolerance(|| {
                        // A vector of length L with endpoint error at most e can
                        // change direction by at most asin(e/L), provided e < L.
                        let left = (points.rounding[i]? + points.rounding[j]?) / dot(v, v).sqrt();
                        let right = (points.rounding[k]? + points.rounding[j]?) / dot(w, w).sqrt();
                        (left < MAX_RELATIVE_ROUNDING && right < MAX_RELATIVE_ROUNDING)
                            .then(|| (left.asin() + right.asin()).to_degrees())
                    })?;
                    if !calculated.is_finite() || (calculated - angle).abs() > tolerance {
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
    let Some(columns) = block.proof_table("_atom_site") else {
        return;
    };
    let Some(bonds) = GeometryTable::read(block, "_geom_bond", "_geom_bond_distance") else {
        return;
    };
    let mut indices = BTreeMap::<&str, Vec<usize>>::new();
    let cartesian: Vec<_> = sites
        .iter()
        .enumerate()
        .map(|(row, site)| {
            indices.entry(&site.source_label).or_default().push(row);
            basis.map(|axis| dot(axis, site.fractional_xyz))
        })
        .collect();
    let axes = [0, 1, 2].map(|k| {
        basis
            .iter()
            .map(|axis| axis[k] * axis[k])
            .sum::<f64>()
            .sqrt()
    });
    let value = |tag: &str, row: usize| columns.get(tag)?.get(row)?.text();
    let rounding = (0..sites.len())
        .map(|row| {
            [
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            ]
            .iter()
            .zip(axes)
            .map(|(tag, length)| Some(half_decimal(value(tag, row)?)? * length))
            .sum::<Option<f64>>()
        })
        .collect();
    let riding_h = sites
        .iter()
        .enumerate()
        .map(|(row, site)| {
            site.element_symbol == "H"
                && value("_atom_site_calc_flag", row) == Some("calc")
                && value("_atom_site_refinement_flags", row) == Some("R")
        })
        .collect();
    let points = Positions {
        cartesian,
        rounding,
        riding_h,
    };
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
