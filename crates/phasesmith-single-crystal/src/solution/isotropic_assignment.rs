//! Bounded isotropic relaxation of competing labels, using independent observed families.
use super::{
    FitPass, Instant, PhaseOriginAudit, PhaseSearchContext, PreparedData, PreparedPeakCompletion,
    SharedSearchOptions, SharedSearchSummary, SolutionError, SolveOptions, SpaceGroup, State,
    UnitCell, distance2, error, refine,
};

/// Parent candidates remain in the competition even when a neighbor looks better cheaply.
/// All candidates use the same deterministic independent-family screen.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn relax(
    context: &PhaseSearchContext,
    cell: UnitCell,
    group: &SpaceGroup,
    options: &SolveOptions,
    search: &SharedSearchOptions,
    elements: &[String],
    data: &PreparedData,
    deadline: Instant,
    screened: Vec<(State, usize)>,
    trace: &mut PhaseOriginAudit,
    audit: &mut SharedSearchSummary,
) -> Result<Vec<(State, usize)>, SolutionError> {
    let mut relaxed = Vec::new();
    for (state, setting) in screened
        .iter()
        .filter(|(s, _)| s.candidate.r1 <= 0.3)
        .take(2)
    {
        if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let seed = state.promotion.as_ref().expect("isotropic seed");
        trace.assignment_parent_fits += 1;
        if let Ok(next) = refine(
            seed.model.clone(),
            data,
            options,
            deadline,
            (1, [0.; 3]),
            state.candidate.u_iso,
            state.lineage,
            &state.moves,
            "isotropic_assignment_parent",
            String::new(),
            audit,
            FitPass::ScreenSeed(&seed.site_u, 12),
        ) {
            relaxed.push((next, *setting));
        }
    }
    if relaxed.is_empty() {
        return Ok(screened);
    }
    // Generate neighbors at relaxed coordinates/U on the existing bounded family cache.
    // Select two alternatives globally by their relative cheap objective, not by
    // mixing cheap and independent-family BIC ranks.
    let mut proposals = Vec::new();
    for (index, (state, _)) in relaxed.iter().enumerate() {
        if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let seed = state.promotion.as_ref().expect("isotropic seed");
        let points: Vec<_> = seed.model.sites.iter().map(|s| s.fractional_xyz).collect();
        let labels: Vec<_> = seed
            .model
            .sites
            .iter()
            .map(|s| elements.iter().position(|e| *e == s.species))
            .collect();
        let cache = PreparedPeakCompletion::new(
            cell,
            group,
            &points,
            elements,
            &context.data.hkl,
            &context.data.amplitudes,
            search.maximum_atoms,
        )?
        .with_site_displacements(&seed.site_u)?;
        let parent_objective = cache.evaluate(&labels, 0)?.0;
        let neighbors = cache.assignment_neighbors(
            &labels,
            2,
            Some(deadline),
            options.cancellation.as_ref(),
        )?;
        trace.assignment_neighbor_scores += neighbors.evaluations;
        for proposal in neighbors.candidates {
            // Avoid spending a fit on an already retained identical labeling/topology.
            if relaxed.iter().any(|(s, _)| {
                let other = s.promotion.as_ref().expect("isotropic seed");
                other.model.sites.len() == seed.model.sites.len()
                    && other
                        .model
                        .sites
                        .iter()
                        .zip(&proposal.labels)
                        .zip(&points)
                        .all(|((site, label), p)| {
                            label.is_some_and(|e| site.species == elements[e])
                                && distance2(&seed.model.geometry, site.fractional_xyz, *p)
                                    .is_ok_and(|d| d < 1e-8)
                        })
            }) {
                continue;
            }
            let merit = (proposal.objective + 1e-4) / (parent_objective + 1e-4);
            proposals.push((merit, index, proposal));
        }
    }
    proposals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut alternatives = Vec::new();
    for (_, index, proposal) in proposals.into_iter().take(2) {
        if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let (parent, setting) = &relaxed[index];
        let seed = parent.promotion.as_ref().expect("isotropic seed");
        let mut model = seed.model.clone();
        for (site, label) in model.sites.iter_mut().zip(proposal.labels) {
            site.species
                .clone_from(&elements[label.ok_or_else(|| error("omitted assignment orbit"))?]);
        }
        trace.assignment_neighbor_fits += 1;
        if let Ok(next) = refine(
            model,
            data,
            options,
            deadline,
            (1, [0.; 3]),
            parent.candidate.u_iso,
            parent.lineage,
            &parent.moves,
            "isotropic_assignment_neighbor",
            String::new(),
            audit,
            FitPass::ScreenSeed(&seed.site_u, 8),
        ) {
            alternatives.push((next, *setting));
        }
    }
    relaxed.extend(alternatives);
    relaxed.sort_by(|a, b| a.0.score.total_cmp(&b.0.score));
    if search.ambiguous_assignments {
        if let Some((parent, _)) = relaxed.first_mut() {
            parent.assignment_reference = true;
        }
        explore(
            context,
            cell,
            group,
            options,
            search,
            elements,
            data,
            deadline,
            &mut relaxed,
            trace,
            audit,
        )?;
    }
    trace.assignment_families = data.family_count;
    trace.assignment_retained = relaxed.len();
    trace.assignment_scores = relaxed.iter().map(|(s, _)| s.score).collect();
    Ok(relaxed)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn explore(
    context: &PhaseSearchContext,
    cell: UnitCell,
    group: &SpaceGroup,
    options: &SolveOptions,
    search: &SharedSearchOptions,
    elements: &[String],
    data: &PreparedData,
    deadline: Instant,
    retained: &mut Vec<(State, usize)>,
    trace: &mut PhaseOriginAudit,
    audit: &mut SharedSearchSummary,
) -> Result<(), SolutionError> {
    for round in 0..2 {
        if crate::solution::stopped(Some(deadline), options.cancellation.as_ref())
            || retained.is_empty()
        {
            break;
        }
        let (parent, setting) = retained[0].clone();
        if parent.candidate.r1 > 0.15
            && parent.candidate.species.len() * 10 >= search.maximum_atoms * 9
        {
            break;
        }
        let seed = parent.promotion.as_ref().expect("isotropic seed");
        let points: Vec<_> = seed.model.sites.iter().map(|s| s.fractional_xyz).collect();
        let labels: Vec<_> = seed
            .model
            .sites
            .iter()
            .map(|s| elements.iter().position(|e| *e == s.species))
            .collect();
        let cache = PreparedPeakCompletion::new(
            cell,
            group,
            &points,
            elements,
            &context.data.hkl,
            &context.data.amplitudes,
            search.maximum_atoms,
        )?;
        let proposals = cache.ambiguity_neighbors(
            &labels,
            &seed.site_u,
            if round == 0 { 4 } else { 2 },
            Some(deadline),
            options.cancellation.as_ref(),
        )?;
        trace.ambiguity_scores += proposals.evaluations;
        trace.ambiguity_rounds += usize::from(proposals.evaluations > 0);
        for proposal in proposals.candidates {
            if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
                break;
            }
            if retained.iter().any(|(s, _)| {
                let other = s.promotion.as_ref().expect("isotropic seed");
                other.model.sites.len() == points.len()
                    && other
                        .model
                        .sites
                        .iter()
                        .zip(&points)
                        .zip(&proposal.labels)
                        .all(|((s, p), label)| {
                            label.is_some_and(|e| s.species == elements[e])
                                && distance2(&seed.model.geometry, s.fractional_xyz, *p)
                                    .is_ok_and(|v| v < 1e-6)
                        })
            }) {
                continue;
            }
            let mut model = seed.model.clone();
            for (site, label) in model.sites.iter_mut().zip(&proposal.labels) {
                site.species
                    .clone_from(&elements[label.expect("complete assignment")]);
            }
            trace.ambiguity_relaxations += 1;
            if let Ok(next) = refine(
                model,
                data,
                options,
                deadline,
                (1, [0.; 3]),
                parent.candidate.u_iso,
                parent.lineage,
                &parent.moves,
                "ambiguous_assignment",
                String::new(),
                audit,
                FitPass::ScreenSeed(&proposal.site_u, 8),
            ) {
                retained.push((next, setting));
                trace.ambiguity_retained += 1;
            }
        }
        retained.sort_by(|a, b| a.0.score.total_cmp(&b.0.score));
        if retained[0].0.score >= parent.score - 1e-6 {
            break;
        }
    }
    Ok(())
}
