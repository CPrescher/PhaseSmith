//! Bounded final tensor/element recovery, using the existing fused ADP evaluator.
use super::{
    FitPass, Instant, SharedSearchOptions, SharedSearchSummary, Solution, SolveOptions, SpaceGroup,
    State, UnitCell, complete, phase_path, record, refine,
};
use std::time::Duration;

fn promising(r1: f64) -> bool {
    r1.is_finite() && (0. ..=0.18).contains(&r1)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn polish(
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    search: &SharedSearchOptions,
    shape: [usize; 3],
    context: &phase_path::PhaseSearchContext,
    winners: &mut Vec<State>,
    result: &mut Solution,
    audit: &mut SharedSearchSummary,
    deadline: Instant,
) {
    let Some(parent) = winners.first().cloned() else {
        return;
    };
    if options.refinement != Some(crate::solution::AdpMode::Isotropic)
        || !complete(&parent, elements)
        || !parent.recycling_validated
        || parent.assignment_disfavoured
        || !promising(parent.candidate.r1)
        || crate::solution::stopped(Some(deadline), options.cancellation.as_ref())
    {
        return;
    }
    let started = Instant::now();
    let local = deadline.min(started + Duration::from_secs(2));
    audit.candidate_polishing_attempts += 1;
    let outcome = promote(cell, group, options, &parent, result, audit, local);
    match outcome {
        Ok(mut promoted) => {
            promoted.recycling_validated = parent.recycling_validated;
            promoted.initial_prefix = parent.initial_prefix;
            if let Some(summary) = &promoted.candidate.refinement {
                let boundary_sites = summary
                    .u_cartesian
                    .iter()
                    .filter(|&&u| {
                        let (values, _) = crate::solution::math3::eigen(u);
                        values
                            .iter()
                            .any(|&v| v <= 1e-5 + 1e-12 || v >= 0.2 - 1e-12)
                    })
                    .count();
                audit
                    .candidate_polishing_trials
                    .push(super::CandidatePolishingTrial {
                        parent_r1: parent.candidate.r1,
                        parent_score: parent.score,
                        fitted_r1: promoted.candidate.r1,
                        fitted_score: promoted.score,
                        parameters: summary.parameter_count,
                        accepted_steps: summary.accepted_steps,
                        termination: summary.termination,
                        boundary_sites,
                    });
            }

            let mut alternatives = vec![promoted];
            let mut requested = options.clone();
            requested.refinement = Some(crate::solution::AdpMode::Anisotropic);
            requested.symmetry_iterations = requested.symmetry_iterations.min(20);
            let termination = result.termination;
            if let Err(error) = phase_path::repair_polished(
                cell,
                group,
                elements,
                &requested,
                search,
                shape,
                context,
                &mut alternatives,
                result,
                audit,
                local,
            ) {
                audit.candidate_polishing_diagnostic = Some(error.to_string());
            }
            result.termination = termination;
            alternatives.sort_by(|a, b| super::ordering(a, b, elements));
            if let Some(best) = alternatives.first()
                && best.score < parent.score - 1e-5
                && crate::solution::peak_completion::ambiguity::replacement_supported(
                    best.score,
                    best.candidate.r1,
                    parent.score,
                    parent.candidate.r1,
                )
            {
                audit.candidate_polishing_adopted += 1;
                let density = result.density.clone();
                record(
                    winners,
                    &alternatives,
                    elements,
                    search.beam_width,
                    parent.phase_start,
                    &density,
                    result,
                );
            }
        }
        Err(error) => audit.candidate_polishing_diagnostic = Some(error.to_string()),
    }
    audit.candidate_polishing_seconds += started.elapsed().as_secs_f64();
}

fn promote(
    cell: UnitCell,
    group: &SpaceGroup,
    options: &SolveOptions,
    parent: &State,
    result: &Solution,
    audit: &mut SharedSearchSummary,
    deadline: Instant,
) -> Result<State, crate::solution::SolutionError> {
    let summary =
        parent.candidate.refinement.as_ref().ok_or_else(|| {
            crate::solution::error("candidate polishing requires a refined parent")
        })?;
    let sites: Vec<_> = summary
        .species
        .iter()
        .cloned()
        .zip(summary.fractional_xyz.iter().copied())
        .collect();
    let model = crate::solution::symmetry_model::model_in_setting(cell, group, &sites)?;
    let tensors = phase_path::transport_tensors(cell, group, &sites, &summary.u_cartesian, &model)?;
    let mut requested = options.clone();
    requested.refinement = Some(crate::solution::AdpMode::Anisotropic);
    requested.symmetry_iterations = requested.symmetry_iterations.min(20);
    refine(
        model,
        &result.data,
        &requested,
        deadline,
        (summary.hand, summary.origin_shift),
        parent.candidate.u_iso,
        parent.lineage,
        &parent.moves,
        "anisotropic_candidate_polishing",
        String::new(),
        audit,
        FitPass::TensorSeed(&tensors),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn promising_trigger_rejects_nonfinite_and_implausible_models() {
        assert!(promising(0.06));
        assert!(promising(0.05));
        assert!(promising(0.09));
        assert!(promising(0.18));
        assert!(!promising(0.180_000_1));
        assert!(!promising(f64::NAN));
        assert!(!promising(f64::INFINITY));
        assert!(!promising(-0.01));
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Independent structure-factor fixture and its two fitted models.
    fn anisotropic_promotion_fits_independent_structure_factors_and_counts_parameters() {
        use crate::solution::math3::{dot, mv};
        use phasesmith_crystallography::{PreparedXrayScattering, Rational, SymmetryOperation};
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 6.,
            c_angstrom: 7.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let group = SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
                .unwrap(),
        ])
        .unwrap();
        let sites = vec![
            ("Ni".into(), [0.13, 0.26, 0.37]),
            ("C".into(), [0.22, 0.34, 0.14]),
            ("O".into(), [0.36, 0.13, 0.42]),
        ];
        let model =
            crate::solution::symmetry_model::model_in_setting(cell, &group, &sites).unwrap();
        let mut points = Vec::new();
        let mut species = Vec::new();
        for (label, point) in &sites {
            for p in group.expand_sites(&[*point], 1e-8).unwrap().fractional_xyz {
                points.push(p);
                species.push(label.clone());
            }
        }
        let hkl: Vec<_> = (-3i32..=3)
            .flat_map(|h| {
                (-3..=3).flat_map(move |k| {
                    (-3..=3).filter_map(move |l| ([h, k, l] != [0; 3]).then_some([h, k, l]))
                })
            })
            .collect();
        let g = cell.geometry().unwrap();
        let s: Vec<_> = hkl
            .iter()
            .map(|&h| 0.5 / g.d_spacing_and_derivatives(h).unwrap().0)
            .collect();
        let factors = PreparedXrayScattering::new(&species)
            .unwrap()
            .evaluate(&s)
            .unwrap()
            .real;
        let mut rows = Vec::new();
        for (r, &h) in hkl.iter().enumerate() {
            let q = mv(g.reciprocal_basis, h.map(f64::from));
            let d = (-2.
                * std::f64::consts::PI.powi(2)
                * (0.012 * q[0] * q[0] + 0.045 * q[1] * q[1] + 0.025 * q[2] * q[2]))
                .exp();
            let mut re = 0.;
            let mut im = 0.;
            for (a, &p) in points.iter().enumerate() {
                let (sin, cos) = (std::f64::consts::TAU * dot(h.map(f64::from), p)).sin_cos();
                re += factors[r * points.len() + a] * d * cos;
                im += factors[r * points.len() + a] * d * sin;
            }
            rows.push(crate::solution::Observation {
                hkl: h,
                intensity: re * re + im * im,
                sigma: 1.,
            });
        }
        let data = crate::solution::prepare_data(&rows, cell, &group).unwrap();
        let options = SolveOptions {
            refinement: Some(crate::solution::AdpMode::Isotropic),
            symmetry_iterations: 20,
            r1_target: 1e-4,
            ..SolveOptions::default()
        };
        let mut audit = SharedSearchSummary {
            beam_width: 4,
            cached_refinement: true,
            ..SharedSearchSummary::default()
        };
        let parent = refine(
            model,
            &data,
            &options,
            Instant::now() + Duration::from_secs(10),
            (1, [0.; 3]),
            0.025,
            0,
            &[],
            "initial",
            String::new(),
            &mut audit,
            FitPass::Full,
        )
        .unwrap();
        assert!(promising(parent.candidate.r1));
        let result = Solution {
            data,
            grid_shape: [16; 3],
            candidates: vec![parent.candidate.clone()],
            density: vec![],
            histories: vec![],
            phasing_trials: vec![],
            rejected_trials: vec![],
            termination: "trial_budget",
            target_met: false,
            shared_search: None,
        };
        let promoted = promote(
            cell,
            &group,
            &options,
            &parent,
            &result,
            &mut audit,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(
            promoted.candidate.refinement.as_ref().unwrap().mode,
            "anisotropic"
        );
        assert!(promoted.candidate.objective < 0.1 * parent.candidate.objective);
        assert!(promoted.score < parent.score);
        assert!(
            promoted
                .candidate
                .refinement
                .as_ref()
                .unwrap()
                .parameter_count
                > parent
                    .candidate
                    .refinement
                    .as_ref()
                    .unwrap()
                    .parameter_count
        );
    }
}
