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
pub fn recover_symmetry(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    points: &[[f64; 3]],
    tolerance: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<SymmetryModel, SolutionError> {
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
    for hand in [1, -1] {
        let reflected: Vec<_> = points
            .iter()
            .map(|&p| p.map(|v| (v * f64::from(hand)).rem_euclid(1.)))
            .collect();
        let grid = if ops.is_empty() { 1 } else { 4 };
        for a in 0..grid {
            for b in 0..grid {
                for c in 0..grid {
                    if crate::stopped(deadline, cancellation) {
                        return Err(error(
                            "symmetry recovery interrupted by cancellation or runtime limit",
                        ));
                    }
                    let start = [
                        f64::from(a) / f64::from(grid),
                        f64::from(b) / f64::from(grid),
                        f64::from(c) / f64::from(grid),
                    ];
                    let shift = fit_origin(&g, &ops, &reflected, elements, &anchors, start)?;
                    let aligned: Vec<_> = reflected
                        .iter()
                        .map(|&p| add(p, shift).map(|v| v.rem_euclid(1.)))
                        .collect();
                    if let Ok((permutations, maximum, score)) =
                        match_operations(&g, group, &aligned, elements, tolerance)
                    {
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
    }
    let (hand, shift, aligned, permutations, maximum, _) =
        best.ok_or_else(|| error("no origin/hand matches the supplied symmetry within tolerance"))?;
    let mut parent: Vec<_> = (0..points.len()).collect();
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
    for rep in 0..points.len() {
        if root(&parent, rep) != rep {
            continue;
        }
        let orbit: Vec<_> = (0..points.len())
            .filter(|&j| root(&parent, j) == rep)
            .collect();
        let mut adjustment = [0.; 3];
        for (op_index, &op) in group.operations().iter().enumerate() {
            let j = permutations[op_index][rep];
            let difference =
                periodic_difference(&g, sub(op.apply_fractional(aligned[rep]), aligned[j]))?;
            let inverse = multiply(
                multiply(
                    transpose(g.reciprocal_basis),
                    transpose(cart_rotation(&g, op)),
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
            let difference = periodic_difference(&g, sub(op.apply_fractional(point), point))?;
            projection = add(projection, difference);
        }
        point =
            add(point, scaled(projection, 1. / stabilizing.len() as f64)).map(|v| v.rem_euclid(1.));
        for &op in &stabilizing {
            let d = periodic_difference(&g, sub(op.apply_fractional(point), point))?;
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
                    cartesian_rotation: cart_rotation(&g, op),
                    source_index: j,
                })
            })
            .collect::<Result<_, SolutionError>>()?;
        sites.push(AsymmetricSite {
            species: elements[rep].clone(),
            fractional_xyz: point,
            coordinate_basis: basis,
            stabilizer: stabilizing
                .iter()
                .map(|&op| cart_rotation(&g, op))
                .collect(),
            images,
        });
    }
    Ok(SymmetryModel {
        geometry: g,
        group: group.clone(),
        sites,
        hand,
        origin_shift: shift,
        maximum_matching_error: maximum,
        origin_gauge_dimension: free_basis.len(),
        full_count: points.len(),
    })
}
