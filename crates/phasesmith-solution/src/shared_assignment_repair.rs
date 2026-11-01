//! Post-search tensor-informed count/element repair; original winner remains available.
use super::{
    FitPass, Instant, PhaseSearchContext, PreparedPeakCompletion, SharedSearchOptions,
    SharedSearchSummary, Solution, SolutionError, SolveOptions, SpaceGroup, State, UnitCell, error,
    extract_peaks, record, refine, xray_species_metadata,
};
use crate::math3::{conjugate, eigen, multiply, transpose};
use phasesmith_crystallography::Matrix3;

pub(in crate::shared_search) fn transport_tensors(
    cell: UnitCell,
    group: &SpaceGroup,
    sites: &[(String, [f64; 3])],
    tensors: &[Matrix3],
    model: &crate::SymmetryModel,
) -> Result<Vec<Matrix3>, SolutionError> {
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let mut full = Vec::new();
    for ((_, point), &tensor) in sites.iter().zip(tensors) {
        let expansion = group
            .expand_sites(&[*point], 1e-8)
            .map_err(|e| error(e.to_string()))?;
        for rotation in expansion.representative_rotation {
            let q = multiply(
                multiply(geometry.direct_basis, rotation.map(|r| r.map(f64::from))),
                transpose(geometry.reciprocal_basis),
            );
            full.push(conjugate(q, tensor));
        }
    }
    if full.len() != model.full_count {
        return Err(error("repair tensor transport topology differs"));
    }
    Ok(model
        .sites
        .iter()
        .map(|s| {
            let image = &s.images[0];
            conjugate(
                transpose(image.cartesian_rotation),
                full[image.source_index],
            )
        })
        .collect())
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn repair_round(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    search: &SharedSearchOptions,
    shape: [usize; 3],
    context: &PhaseSearchContext,
    winners: &mut Vec<State>,
    result: &mut Solution,
    audit: &mut SharedSearchSummary,
    deadline: Instant,
    r1_floor: f64,
) -> Result<(), SolutionError> {
    let Some(parent) = winners.first().cloned() else {
        return Ok(());
    };
    let Some(summary) = parent.candidate.refinement.as_ref() else {
        return Ok(());
    };
    if !super::complete(&parent, elements)
        || crate::stopped(Some(deadline), options.cancellation.as_ref())
        || !(r1_floor..=0.18).contains(&parent.candidate.r1)
        || parent.assignment_disfavoured
    {
        return Ok(());
    }
    let numbers: Vec<_> = elements
        .iter()
        .map(|e| {
            xray_species_metadata(e)
                .expect("validated elements")
                .atomic_number
        })
        .collect();
    if !numbers
        .iter()
        .enumerate()
        .any(|(i, a)| numbers[..i].iter().any(|b| a.abs_diff(*b) <= 2))
    {
        return Ok(());
    }
    let local = deadline.min(Instant::now() + std::time::Duration::from_secs(2));
    let mut sites: Vec<_> = summary
        .species
        .iter()
        .cloned()
        .zip(summary.fractional_xyz.iter().copied())
        .collect();
    let base = sites.len();
    let mut tensors = summary.u_cartesian.clone();
    // The residual map proposes locations only; tensor-aware full-data fits decide acceptance.
    if parent.candidate.species.len() < search.maximum_atoms
        && let Ok(pool) = crate::PreparedElementPool::new(
            cell,
            group,
            &parent.candidate.species,
            &parent.aligned,
            elements,
            &result.data.hkl,
            &result.data.amplitudes,
            parent.candidate.u_iso.clamp(0., 0.2),
        )
        && let Ok(map) = pool.parent.difference_map(shape)
        && let Ok(seeds) = extract_peaks(&map.density, shape, 16)
        && let Ok(points) = super::super::proposals(
            group,
            &cell.geometry().map_err(|e| error(e.to_string()))?,
            &seeds,
            &parent.aligned,
            search.maximum_atoms,
            options.symmetry_tolerance_angstrom,
            options,
            local,
        )
    {
        for point in points.into_iter().take(2) {
            sites.push((elements.last().expect("elements").clone(), point));
            let u = parent.candidate.u_iso.clamp(1e-5, 0.2);
            tensors.push([[u, 0., 0.], [0., u, 0.], [0., 0., u]]);
        }
    }
    let Ok(model) = crate::symmetry_model::model_in_setting(cell, group, &sites) else {
        return Ok(());
    };
    let tensors = transport_tensors(cell, group, &sites, &tensors, &model)?;
    let points: Vec<_> = model.sites.iter().map(|s| s.fractional_xyz).collect();
    // Source indices identify which input orbit supplied each recovered site.
    let mut source_orbits = Vec::new();
    for (orbit, (_, point)) in sites.iter().enumerate() {
        let m = group
            .expand_sites(&[*point], 1e-8)
            .map_err(|e| error(e.to_string()))?
            .fractional_xyz
            .len();
        source_orbits.extend(std::iter::repeat_n(orbit, m));
    }
    let labels: Vec<_> = model
        .sites
        .iter()
        .map(|s| {
            let source = source_orbits[s.images[0].source_index];
            (source < base).then(|| {
                elements
                    .iter()
                    .position(|e| *e == sites[source].0)
                    .expect("required species")
            })
        })
        .collect();
    let limits: Vec<_> = tensors
        .iter()
        .map(|&u| {
            let (v, _) = eigen(u);
            [(1e-5 - v[0]).min(0.), (0.2 - v[2]).max(0.)]
        })
        .collect();
    let Ok(plan) = PreparedPeakCompletion::new(
        cell,
        group,
        &points,
        elements,
        &context.data.hkl,
        &context.data.amplitudes,
        search.maximum_atoms,
    )
    .and_then(|p| p.with_tensor_model(&model, &tensors, &context.data.hkl)) else {
        return Ok(());
    };
    let mut proposals = plan.joint_neighbors_with_cache(
        &labels,
        &limits,
        2,
        Some(local),
        options.cancellation.as_ref(),
        search.cached_repair_evaluation,
    )?;
    audit.joint_assignment_field_builds += proposals.profile_builds;
    audit.joint_assignment_field_reuses += proposals.profile_reuses;
    audit.joint_assignment_score_only += proposals.score_only_requests;
    audit.joint_assignment_workspace_bytes = audit
        .joint_assignment_workspace_bytes
        .max(proposals.workspace_bytes);
    let mut coherent = Vec::new();
    let expanded = search
        .cached_repair_evaluation
        .then(|| {
            points
                .iter()
                .map(|&p| {
                    group
                        .expand_sites(&[p], 1e-8)
                        .map(|v| v.fractional_xyz)
                        .map_err(|e| error(e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let mut neighbor_medians: Vec<Option<f64>> = vec![None; points.len()];
    for (strength_index, strength) in [0.5, 1.].into_iter().enumerate() {
        if crate::stopped(Some(local), options.cancellation.as_ref()) {
            break;
        }
        let mut shifts = vec![0.; points.len()];
        for i in 0..points.len() {
            if crate::stopped(Some(local), options.cancellation.as_ref()) {
                break;
            }
            let Some(label) = labels[i] else { continue };
            if search.cached_repair_evaluation && strength_index == 1 {
                if let Some(median) = neighbor_medians[i] {
                    let own = (tensors[i][0][0] + tensors[i][1][1] + tensors[i][2][2]) / 3.;
                    shifts[i] = (strength * (median - own))
                        .clamp(-0.02, 0.02)
                        .clamp(limits[i][0], limits[i][1]);
                }
                continue;
            }
            let mut neighbors = Vec::new();
            for j in 0..points.len() {
                if crate::stopped(Some(local), options.cancellation.as_ref()) {
                    break;
                }
                let Some(other) = labels[j] else { continue };
                if numbers[label].abs_diff(numbers[other]) > 3 {
                    continue;
                }
                let direct;
                let images = if let Some(cache) = &expanded {
                    &cache[j]
                } else {
                    direct = group
                        .expand_sites(&[points[j]], 1e-8)
                        .map_err(|e| error(e.to_string()))?
                        .fractional_xyz;
                    &direct
                };
                for &image in images {
                    let d = super::distance2(&model.geometry, points[i], image)?;
                    if (0.81..=5.29).contains(&d) {
                        neighbors
                            .push((tensors[j][0][0] + tensors[j][1][1] + tensors[j][2][2]) / 3.);
                    }
                }
            }
            if !neighbors.is_empty() {
                neighbors.sort_by(f64::total_cmp);
                neighbor_medians[i] = Some(neighbors[neighbors.len() / 2]);
                let own = (tensors[i][0][0] + tensors[i][1][1] + tensors[i][2][2]) / 3.;
                shifts[i] = (strength * (neighbors[neighbors.len() / 2] - own))
                    .clamp(-0.02, 0.02)
                    .clamp(limits[i][0], limits[i][1]);
            }
        }
        let found = plan.coherent_reassign(
            &labels,
            &shifts,
            &limits,
            Some(local),
            options.cancellation.as_ref(),
        )?;
        proposals.evaluations += found.evaluations;
        proposals.depths += found.depths;
        for candidate in found.candidates {
            if !coherent
                .iter()
                .any(|p: &crate::JointAssignment| p.labels == candidate.labels)
            {
                coherent.push(candidate);
            }
        }
    }
    coherent.sort_by(|a, b| a.score.total_cmp(&b.score));
    if let Some(challenger) = coherent.into_iter().find(|c| {
        !proposals
            .candidates
            .iter()
            .take(1)
            .any(|p| p.labels == c.labels)
    }) {
        proposals.candidates.truncate(1);
        proposals.candidates.push(challenger);
    }
    audit.joint_assignment_scores += proposals.evaluations;
    audit.joint_assignment_depths += proposals.depths;
    let mut best = parent.clone();
    for proposal in proposals.candidates {
        if crate::stopped(Some(local), options.cancellation.as_ref()) {
            break;
        }
        let mut chosen = Vec::new();
        let mut seeds = Vec::new();
        for (i, label) in proposal.labels.iter().enumerate() {
            if let Some(label) = label {
                chosen.push((elements[*label].clone(), points[i]));
                let mut tensor = tensors[i];
                for (axis, row) in tensor.iter_mut().enumerate() {
                    row[axis] += proposal.shifts[i];
                }
                seeds.push(tensor);
            }
        }
        let Ok(candidate_model) = crate::symmetry_model::model_in_setting(cell, group, &chosen)
        else {
            continue;
        };
        let seeds = transport_tensors(cell, group, &chosen, &seeds, &candidate_model)?;
        audit.joint_assignment_fits += 1;
        if let Ok(mut candidate) = refine(
            candidate_model,
            &result.data,
            options,
            local,
            (summary.hand, summary.origin_shift),
            parent.candidate.u_iso,
            parent.lineage,
            &parent.moves,
            "joint_assignment_repair",
            String::new(),
            audit,
            FitPass::TensorSeed(&seeds),
        ) {
            candidate.recycling_validated = parent.recycling_validated;
            candidate.initial_prefix = parent.initial_prefix;
            if crate::peak_completion::ambiguity::replacement_supported(
                candidate.score,
                candidate.candidate.r1,
                best.score,
                best.candidate.r1,
            ) && candidate.score < best.score - 1e-5
                && candidate.candidate.objective <= 0.99 * best.candidate.objective
            {
                best = candidate;
            }
        }
    }
    if best.score < parent.score - 1e-5 {
        audit.joint_assignment_accepted += 1;
        let density = result.density.clone();
        record(
            winners,
            &[best],
            elements,
            search.beam_width,
            parent.phase_start,
            &density,
            result,
        );
        result.termination = "joint_assignment_repair";
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn repair_with_floor(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    search: &SharedSearchOptions,
    shape: [usize; 3],
    context: &PhaseSearchContext,
    winners: &mut Vec<State>,
    result: &mut Solution,
    audit: &mut SharedSearchSummary,
    deadline: Instant,
    r1_floor: f64,
) -> Result<(), SolutionError> {
    let local = deadline.min(Instant::now() + std::time::Duration::from_secs(2));
    for _ in 0..3 {
        let accepted = audit.joint_assignment_accepted;
        repair_round(
            cell, group, elements, options, search, shape, context, winners, result, audit, local,
            r1_floor,
        )?;
        if audit.joint_assignment_accepted == accepted
            || crate::stopped(Some(local), options.cancellation.as_ref())
        {
            break;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::shared_search) fn repair(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    search: &SharedSearchOptions,
    shape: [usize; 3],
    context: &PhaseSearchContext,
    winners: &mut Vec<State>,
    result: &mut Solution,
    audit: &mut SharedSearchSummary,
    deadline: Instant,
) -> Result<(), SolutionError> {
    repair_with_floor(
        cell, group, elements, options, search, shape, context, winners, result, audit, deadline,
        0.04,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::shared_search) fn repair_polished(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    search: &SharedSearchOptions,
    shape: [usize; 3],
    context: &PhaseSearchContext,
    winners: &mut Vec<State>,
    result: &mut Solution,
    audit: &mut SharedSearchSummary,
    deadline: Instant,
) -> Result<(), SolutionError> {
    repair_with_floor(
        cell, group, elements, options, search, shape, context, winners, result, audit, deadline,
        0.,
    )
}
