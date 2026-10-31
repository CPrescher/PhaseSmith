//! Recover a supplied symmetry setting from an arbitrary-origin full-cell model.

use crate::math3::{
    IDENTITY, add, dot, matrix_add, multiply, mv, orthogonal_vectors, periodic_difference,
    pseudo_solve, scaled, sub, transpose,
};
use crate::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{
    CellGeometry, Matrix3, SpaceGroup, SymmetryOperation, UnitCell, xray_species_metadata,
};
use std::time::Instant;

/// One unique symmetry image of an asymmetric site.
#[derive(Clone, Debug)]
pub struct SymmetryImage {
    /// Exact fractional operation creating this image.
    pub operation: SymmetryOperation,
    /// Orthogonal Cartesian image rotation.
    pub cartesian_rotation: Matrix3,
    /// Index of the corresponding input full-cell atom.
    pub source_index: usize,
}

/// One projected asymmetric site and its fixed symmetry topology.
#[derive(Clone, Debug)]
pub struct AsymmetricSite {
    /// Neutral element.
    pub species: String,
    /// Coordinate in the caller's space-group setting.
    pub fractional_xyz: [f64; 3],
    /// Independent fractional tangents, normalized to unit Cartesian length.
    pub coordinate_basis: Vec<[f64; 3]>,
    /// Cartesian site-stabilizer rotations.
    pub stabilizer: Vec<Matrix3>,
    /// Unique physical images, with full-cell source identities.
    pub images: Vec<SymmetryImage>,
}

/// Recovered setting, without using any reference atom coordinates.
#[derive(Clone, Debug)]
pub struct SymmetryModel {
    /// Fixed cell geometry.
    pub geometry: CellGeometry,
    /// Supplied exact operation set.
    pub group: SpaceGroup,
    /// Stable asymmetric-site order.
    pub sites: Vec<AsymmetricSite>,
    /// Aligned coordinate = hand * input coordinate + origin shift (modulo 1).
    pub hand: i32,
    /// Shift from the density/P1 frame to the supplied group frame.
    pub origin_shift: [f64; 3],
    /// Maximum group-matching displacement before projection, in angstrom.
    pub maximum_matching_error: f64,
    /// Number of continuously free origin directions removed from refinement.
    pub origin_gauge_dimension: usize,
    /// Input full-cell atom count.
    pub full_count: usize,
}

fn root(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}

fn cart_rotation(g: &CellGeometry, op: SymmetryOperation) -> Matrix3 {
    multiply(
        multiply(g.direct_basis, op.rotation().map(|r| r.map(f64::from))),
        transpose(g.reciprocal_basis),
    )
}

fn nearest(
    g: &CellGeometry,
    point: [f64; 3],
    species: &str,
    points: &[[f64; 3]],
    elements: &[String],
) -> Result<(usize, [f64; 3], f64), SolutionError> {
    let mut best = None;
    for (j, &other) in points.iter().enumerate() {
        if elements[j] != species {
            continue;
        }
        let delta = periodic_difference(g, sub(point, other))?;
        let d2 = dot(delta, mv(g.direct_metric, delta));
        if best.as_ref().is_none_or(|&(_, _, old)| d2 < old) {
            best = Some((j, delta, d2));
        }
    }
    best.ok_or_else(|| error("symmetry matching has no same-element target"))
}

fn origin_operators(group: &SpaceGroup) -> Vec<SymmetryOperation> {
    let mut selected = Vec::new();
    let mut rows = Vec::new();
    let mut rank = 0;
    for &op in group.operations() {
        let rotation = op.rotation();
        let next: Vec<_> = (0..3)
            .map(|i| std::array::from_fn(|j| f64::from(rotation[i][j]) - IDENTITY[i][j]))
            .collect();
        let mut combined = rows.clone();
        combined.extend(next);
        let new_rank = orthogonal_vectors(combined.iter().copied(), IDENTITY).len();
        if new_rank > rank {
            selected.push(op);
            rows = combined;
            rank = new_rank;
        }
        if rank == 3 {
            break;
        }
    }
    selected
}

// Inversion gives -2s = p_j + p_anchor - t (mod integer lattice).
// Enumerate the eight half-cell branches without fixing any coordinate/origin.
fn inversion_origin_seeds(
    group: &SpaceGroup,
    points: &[[f64; 3]],
    elements: &[String],
    anchor: usize,
) -> Vec<[f64; 3]> {
    let inversion = group
        .operations()
        .iter()
        .find(|op| op.rotation() == [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]);
    let Some(op) = inversion else {
        return Vec::new();
    };
    let translation = op.apply_fractional([0.; 3]);
    let mut seeds = Vec::new();
    for (j, point) in points.iter().enumerate() {
        if elements[j] != elements[anchor] {
            continue;
        }
        for branch in 0..8 {
            let seed = std::array::from_fn(|axis| {
                (-0.5 * (points[anchor][axis] + point[axis] - translation[axis])
                    + 0.5 * f64::from((branch >> axis) & 1))
                .rem_euclid(1.)
            });
            seeds.push(seed);
        }
    }
    seeds
}

fn fit_origin(
    g: &CellGeometry,
    ops: &[SymmetryOperation],
    points: &[[f64; 3]],
    elements: &[String],
    anchors: &[usize],
    mut shift: [f64; 3],
) -> Result<[f64; 3], SolutionError> {
    for _ in 0..12 {
        let aligned: Vec<_> = points
            .iter()
            .map(|&p| add(p, shift).map(|v| v.rem_euclid(1.)))
            .collect();
        let mut normal = [[0.; 3]; 3];
        let mut gradient = [0.; 3];
        for &op in ops {
            let delta = std::array::from_fn(|i| {
                std::array::from_fn(|j| f64::from(op.rotation()[i][j]) - IDENTITY[i][j])
            });
            let jac = multiply(g.direct_basis, delta);
            for &i in anchors {
                let (_, difference, _) = nearest(
                    g,
                    op.apply_fractional(aligned[i]),
                    &elements[i],
                    &aligned,
                    elements,
                )?;
                gradient = add(gradient, mv(transpose(jac), mv(g.direct_basis, difference)));
                matrix_add(&mut normal, multiply(transpose(jac), jac));
            }
        }
        let step = pseudo_solve(normal, gradient.map(|v| -v));
        shift = add(shift, step).map(|v| v.rem_euclid(1.));
        if dot(step, mv(g.direct_metric, step)) < 1e-18 {
            break;
        }
    }
    Ok(shift)
}

fn match_operations(
    g: &CellGeometry,
    group: &SpaceGroup,
    points: &[[f64; 3]],
    elements: &[String],
    tolerance: f64,
) -> Result<(Vec<Vec<usize>>, f64, f64), SolutionError> {
    let mut permutations = Vec::new();
    let mut maximum = 0.0_f64;
    let mut sum = 0.;
    for &op in group.operations() {
        let mut permutation = Vec::new();
        let mut occupied = vec![false; points.len()];
        for (i, &point) in points.iter().enumerate() {
            let (j, _, d2) = nearest(
                g,
                op.apply_fractional(point),
                &elements[i],
                points,
                elements,
            )?;
            if d2 > tolerance * tolerance || occupied[j] {
                return Err(error(
                    "candidate does not admit a complete same-element symmetry bijection",
                ));
            }
            occupied[j] = true;
            permutation.push(j);
            maximum = maximum.max(d2.sqrt());
            sum += d2;
        }
        permutations.push(permutation);
    }
    Ok((permutations, maximum, sum))
}

/// Identify the origin/hand and project complete same-element symmetry orbits.
///
/// # Errors
/// Rejects malformed models, incompatible topology, ambiguous/poor matching,
/// a periodic-distance resource overflow, cancellation or runtime exhaustion.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
fn recover_models(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    points: &[[f64; 3]],
    tolerance: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
    maximum_models: usize,
) -> Result<Vec<SymmetryModel>, SolutionError> {
    if points.is_empty()
        || points.len() > 128
        || points.len() != elements.len()
        || points.iter().flatten().any(|v| !v.is_finite())
        || !tolerance.is_finite()
        || !(1e-5..=0.5).contains(&tolerance)
    {
        return Err(error(
            "invalid full-cell model or symmetry matching tolerance",
        ));
    }
    let g = cell.geometry().map_err(|e| error(e.to_string()))?;
    if group.operations().len() > 192 {
        return Err(error("symmetry operation count exceeds 192"));
    }
    for &op in group.operations() {
        let q = cart_rotation(&g, op);
        let metric_check = multiply(transpose(q), q);
        if metric_check
            .iter()
            .flatten()
            .zip(IDENTITY.iter().flatten())
            .any(|(a, b)| (a - b).abs() > 1e-8)
        {
            return Err(error(
                "cell metric is incompatible with the supplied symmetry",
            ));
        }
    }
    let ops = origin_operators(group);
    let mut anchors: Vec<_> = (0..points.len()).collect();
    anchors.sort_by_key(|&i| {
        (
            std::cmp::Reverse(xray_species_metadata(&elements[i]).map_or(0, |m| m.atomic_number)),
            i,
        )
    });
    anchors.truncate(12);
    let mut best = None;
    let mut alternatives: Vec<OriginMatch> = Vec::new();
    for pair_seeds in [false, true] {
        if pair_seeds && best.is_some() {
            break;
        }
        for hand in [1, -1] {
            let reflected: Vec<_> = points
                .iter()
                .map(|&p| p.map(|v| (v * f64::from(hand)).rem_euclid(1.)))
                .collect();
            let starts = if pair_seeds {
                inversion_origin_seeds(group, &reflected, elements, anchors[0])
            } else {
                let grid = if ops.is_empty() { 1 } else { 4 };
                let mut starts = Vec::new();
                for a in 0..grid {
                    for b in 0..grid {
                        for c in 0..grid {
                            starts.push([
                                f64::from(a) / f64::from(grid),
                                f64::from(b) / f64::from(grid),
                                f64::from(c) / f64::from(grid),
                            ]);
                        }
                    }
                }
                starts
            };
            for start in starts {
                if crate::stopped(deadline, cancellation) {
                    return Err(error(
                        "symmetry recovery interrupted by cancellation or runtime limit",
                    ));
                }
                let shift = fit_origin(&g, &ops, &reflected, elements, &anchors, start)?;
                let aligned: Vec<_> = reflected
                    .iter()
                    .map(|&p| add(p, shift).map(|v| v.rem_euclid(1.)))
                    .collect();
                if let Ok((permutations, maximum, score)) =
                    match_operations(&g, group, &aligned, elements, tolerance)
                {
                    if maximum_models > 1 {
                        alternatives.push((
                            hand,
                            shift,
                            aligned.clone(),
                            permutations.clone(),
                            maximum,
                            score,
                        ));
                    }
                    if best
                        .as_ref()
                        .is_none_or(|&(_, _, _, _, _, old)| score < old - 1e-12)
                    {
                        best = Some((hand, shift, aligned, permutations, maximum, score));
                    }
                }
            }
        }
    }
    let best =
        best.ok_or_else(|| error("no origin/hand matches the supplied symmetry within tolerance"))?;
    alternatives.sort_by(|a, b| a.5.total_cmp(&b.5));
    let mut raw = vec![best];
    raw.extend(alternatives);
    let mut models = Vec::new();
    for matched in raw {
        if crate::stopped(deadline, cancellation) {
            return Err(error(
                "symmetry recovery interrupted by cancellation or runtime limit",
            ));
        }
        let model = build_model(&g, group, elements, matched)?;
        let mut duplicate = false;
        for previous in &models {
            if equivalent_framework(&model, previous)? {
                duplicate = true;
                break;
            }
        }
        if !duplicate {
            models.push(model);
        }
        if models.len() >= maximum_models {
            break;
        }
    }
    Ok(models)
}

type OriginMatch = (i32, [f64; 3], Vec<[f64; 3]>, Vec<Vec<usize>>, f64, f64);

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn build_model(
    g: &CellGeometry,
    group: &SpaceGroup,
    elements: &[String],
    matched: OriginMatch,
) -> Result<SymmetryModel, SolutionError> {
    let (hand, shift, aligned, permutations, maximum, _) = matched;
    let mut parent: Vec<_> = (0..elements.len()).collect();
    for permutation in &permutations {
        for (i, &j) in permutation.iter().enumerate() {
            let a = root(&parent, i);
            let b = root(&parent, j);
            parent[a.max(b)] = a.min(b);
        }
    }
    let free_projector = group.operations().iter().fold([[0.; 3]; 3], |mut p, op| {
        matrix_add(
            &mut p,
            op.rotation()
                .map(|r| r.map(|v| f64::from(v) / group.operations().len() as f64)),
        );
        p
    });
    let free_basis = orthogonal_vectors(transpose(free_projector), g.direct_metric);
    let mut sites = Vec::new();
    for rep in 0..elements.len() {
        if root(&parent, rep) != rep {
            continue;
        }
        let orbit: Vec<_> = (0..elements.len())
            .filter(|&j| root(&parent, j) == rep)
            .collect();
        let mut adjustment = [0.; 3];
        for (op_index, &op) in group.operations().iter().enumerate() {
            let j = permutations[op_index][rep];
            let difference =
                periodic_difference(g, sub(op.apply_fractional(aligned[rep]), aligned[j]))?;
            let inverse = multiply(
                multiply(
                    transpose(g.reciprocal_basis),
                    transpose(cart_rotation(g, op)),
                ),
                g.direct_basis,
            );
            adjustment = add(adjustment, mv(inverse, difference));
        }
        let mut point = sub(
            aligned[rep],
            scaled(adjustment, 1. / group.operations().len() as f64),
        );
        let stabilizing: Vec<_> = group
            .operations()
            .iter()
            .copied()
            .enumerate()
            .filter(|(op, _)| permutations[*op][rep] == rep)
            .map(|(_, op)| op)
            .collect();
        let mut projector = [[0.; 3]; 3];
        let mut projection = [0.; 3];
        for &op in &stabilizing {
            matrix_add(
                &mut projector,
                op.rotation()
                    .map(|r| r.map(|v| f64::from(v) / stabilizing.len() as f64)),
            );
            let difference = periodic_difference(g, sub(op.apply_fractional(point), point))?;
            projection = add(projection, difference);
        }
        point =
            add(point, scaled(projection, 1. / stabilizing.len() as f64)).map(|v| v.rem_euclid(1.));
        for &op in &stabilizing {
            let d = periodic_difference(g, sub(op.apply_fractional(point), point))?;
            if dot(d, mv(g.direct_metric, d)) > 1e-16 {
                return Err(error("site-stabilizer projection is inconsistent"));
            }
        }
        let expanded = group
            .expand_sites(&[point], 1e-8)
            .map_err(|e| error(e.to_string()))?;
        if expanded.fractional_xyz.len() != orbit.len() {
            return Err(error(
                "projected special-position multiplicity differs from the recovered orbit",
            ));
        }
        let mut basis = orthogonal_vectors(transpose(projector), g.direct_metric);
        if sites.is_empty() {
            let projected: Vec<_> = basis
                .iter()
                .map(|&b| {
                    free_basis
                        .iter()
                        .fold(b, |v, &t| sub(v, scaled(t, dot(t, mv(g.direct_metric, v)))))
                })
                .collect();
            basis = orthogonal_vectors(projected, g.direct_metric);
        }
        let images = orbit
            .iter()
            .map(|&j| {
                let index = permutations
                    .iter()
                    .position(|p| p[rep] == j)
                    .ok_or_else(|| error("incomplete recovered orbit"))?;
                let op = group.operations()[index];
                Ok(SymmetryImage {
                    operation: op,
                    cartesian_rotation: cart_rotation(g, op),
                    source_index: j,
                })
            })
            .collect::<Result<_, SolutionError>>()?;
        sites.push(AsymmetricSite {
            species: elements[rep].clone(),
            fractional_xyz: point,
            coordinate_basis: basis,
            stabilizer: stabilizing.iter().map(|&op| cart_rotation(g, op)).collect(),
            images,
        });
    }
    Ok(SymmetryModel {
        geometry: *g,
        group: group.clone(),
        sites,
        hand,
        origin_shift: shift,
        maximum_matching_error: maximum,
        origin_gauge_dimension: free_basis.len(),
        full_count: elements.len(),
    })
}

/// Build an already aligned orbit model without an origin/hand search.
pub(crate) fn model_in_setting(
    cell: UnitCell,
    group: &SpaceGroup,
    sites: &[(String, [f64; 3])],
) -> Result<SymmetryModel, SolutionError> {
    if sites.is_empty() {
        return Err(error("empty aligned peak model"));
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let mut points = Vec::new();
    let mut species = Vec::new();
    for (label, point) in sites {
        let images = group
            .expand_sites(&[*point], 1e-8)
            .map_err(|e| error(e.to_string()))?;
        species.extend(std::iter::repeat_n(
            label.clone(),
            images.fractional_xyz.len(),
        ));
        points.extend(images.fractional_xyz);
    }
    if points.len() > 128 {
        return Err(error("aligned model exceeds 128 atoms"));
    }
    let (permutations, maximum, score) =
        match_operations(&geometry, group, &points, &species, 1e-6)?;
    build_model(
        &geometry,
        group,
        &species,
        (1, [0.; 3], points, permutations, maximum, score),
    )
}

/// Retain distinct partial-framework settings; origin shifts normalizing the full
/// supplied group are deduplicated, while inequivalent heavy-site settings survive.
///
/// # Errors
/// The same boundary as `recover_symmetry`, with a hypothesis budget in `[1,16]`.
#[allow(clippy::too_many_arguments)]
pub fn recover_symmetry_hypotheses(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    points: &[[f64; 3]],
    tolerance: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
    maximum_models: usize,
) -> Result<Vec<SymmetryModel>, SolutionError> {
    if !(1..=16).contains(&maximum_models) {
        return Err(error("symmetry hypothesis budget must lie in [1,16]"));
    }
    recover_models(
        cell,
        group,
        elements,
        points,
        tolerance,
        deadline,
        cancellation,
        maximum_models,
    )
}

/// Identify one deterministic origin/hand and project complete symmetry orbits.
///
/// # Errors
/// Rejects malformed/incompatible models, poor matching or exhausted resources.
#[allow(clippy::too_many_arguments)]
pub fn recover_symmetry(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    points: &[[f64; 3]],
    tolerance: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<SymmetryModel, SolutionError> {
    recover_models(
        cell,
        group,
        elements,
        points,
        tolerance,
        deadline,
        cancellation,
        1,
    )?
    .pop()
    .ok_or_else(|| error("no symmetry model"))
}

fn equivalent_framework(a: &SymmetryModel, b: &SymmetryModel) -> Result<bool, SolutionError> {
    fn expand(model: &SymmetryModel) -> Vec<(&String, [f64; 3])> {
        model
            .sites
            .iter()
            .flat_map(|s| {
                s.images.iter().map(|image| {
                    (
                        &s.species,
                        image.operation.apply_fractional(s.fractional_xyz),
                    )
                })
            })
            .collect::<Vec<_>>()
    }
    let left = expand(a);
    let right = expand(b);
    for &(element, anchor) in &right {
        if *element != a.sites[0].species {
            continue;
        }
        let shift = sub(left[0].1, anchor);
        let mut normalizes = true;
        for op in a.group.operations() {
            let d = periodic_difference(
                &a.geometry,
                sub(mv(op.rotation().map(|r| r.map(f64::from)), shift), shift),
            )?;
            if dot(d, mv(a.geometry.direct_metric, d)) > 1e-14 {
                normalizes = false;
                break;
            }
        }
        if !normalizes {
            continue;
        }
        let mut all = true;
        for &(element, point) in &left {
            let mut found = false;
            for &(other, target) in &right {
                if element != other {
                    continue;
                }
                let d = periodic_difference(&a.geometry, sub(point, add(target, shift)))?;
                if dot(d, mv(a.geometry.direct_metric, d)) < 1e-14 {
                    found = true;
                    break;
                }
            }
            if !found {
                all = false;
                break;
            }
        }
        if all {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod inversion_seed_tests {
    use super::*;
    use phasesmith_crystallography::Rational;

    #[test]
    fn inversion_seeds_satisfy_correspondence_with_translated_center() {
        let inversion = SymmetryOperation::new(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [
                Rational::new(1, 3).unwrap(),
                Rational::new(1, 4).unwrap(),
                Rational::new(1, 2).unwrap(),
            ],
        )
        .unwrap();
        let group = SpaceGroup::new(vec![SymmetryOperation::identity(), inversion]).unwrap();
        let points = [[0.137, 0.259, 0.382], [0.641, 0.314, 0.728]];
        let labels = vec!["Si".into(), "Si".into()];
        let seeds = inversion_origin_seeds(&group, &points, &labels, 0);
        assert_eq!(seeds.len(), 16);
        for (j, branch) in seeds.as_chunks::<8>().0.iter().enumerate() {
            for &shift in branch {
                let image = inversion.apply_fractional(add(points[0], shift));
                for axis in 0..3 {
                    let d = image[axis] - points[j][axis] - shift[axis];
                    assert!((d - d.round()).abs() < 1e-14);
                }
            }
        }
        let labels = vec!["Si".into(), "O".into()];
        assert_eq!(inversion_origin_seeds(&group, &points, &labels, 0).len(), 8);
    }

    #[test]
    fn noncentrosymmetric_group_has_no_inversion_seeds() {
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        assert_eq!(
            inversion_origin_seeds(&group, &[[0.1; 3]], &["Si".into()], 0),
            [] as [[f64; 3]; 0]
        );
    }
}
