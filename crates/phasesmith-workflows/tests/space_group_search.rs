//! Measured-only hypothesis evidence, missing coverage and conventional setting checks.
#![cfg(feature = "single-crystal")]
#![allow(clippy::float_cmp)] // Exact zero evidence and discrete identity are tested deliberately.
use phasesmith_crystallography::UnitCell;
use phasesmith_io::{space_group_by_number, space_group_settings_for_cell};
use phasesmith_solution::Observation;
use phasesmith_workflows::space_group_search::{AssessmentOptions, Hypothesis, assess};
fn cell() -> UnitCell {
    UnitCell {
        a_angstrom: 7.1,
        b_angstrom: 8.3,
        c_angstrom: 9.7,
        alpha_deg: 90.,
        beta_deg: 90.,
        gamma_deg: 90.,
    }
}
fn hypothesis(number: i32) -> Hypothesis {
    let h = space_group_by_number(number).unwrap();
    Hypothesis {
        number,
        symbol: h.hm_symbol,
        hall: h.hall_symbol,
        group: h.space_group,
    }
}
fn measurements(number: i32) -> Vec<Observation> {
    let group = hypothesis(number).group;
    let mut rows = Vec::new();
    for h in -4i32..=4 {
        for k in -4i32..=4 {
            for l in -4i32..=4 {
                if [h, k, l] == [0; 3] {
                    continue;
                }
                let absent = group.is_systematically_absent([h, k, l]).unwrap();
                let intensity = if absent {
                    -0.5
                } else {
                    50. + f64::from(3 * h * h + 7 * k * k + 11 * l * l)
                };
                rows.push(Observation {
                    hkl: [h, k, l],
                    intensity,
                    sigma: 2.,
                });
            }
        }
    }
    rows
}
#[test]
fn metric_permission_does_not_establish_intensity_symmetry() {
    let candidates = [hypothesis(1), hypothesis(47)];
    let rows: Vec<_> = measurements(1)
        .into_iter()
        .map(|mut r| {
            r.intensity *= if r.hkl[0] * r.hkl[1] > 0 { 10. } else { 1. };
            r
        })
        .collect();
    let a = assess(&rows, cell(), &candidates, &AssessmentOptions::default()).unwrap();
    assert_eq!(a.laue[a.evidence[1].laue].status, "contradicted");
    assert_eq!(a.laue[a.processing_laue].rotations.len(), 2);
}
#[test]
fn measured_glides_rank_pbca_without_a_space_group_prior() {
    let candidates = [
        hypothesis(1),
        hypothesis(19),
        hypothesis(47),
        hypothesis(61),
        hypothesis(62),
    ];
    let a = assess(
        &measurements(61),
        cell(),
        &candidates,
        &AssessmentOptions::default(),
    )
    .unwrap();
    assert_eq!(a.ranking[0], 3);
    assert!(a.evidence[3].absence_log_evidence > 0.);
    assert_eq!(a.evidence[3].strong_violations, 0);
    assert_eq!(a.laue[a.processing_laue].rotations.len(), 8);
}
#[test]
fn unmeasured_forbidden_indices_never_become_absence_evidence() {
    let candidates = [hypothesis(1), hypothesis(61)];
    let group = &candidates[1].group;
    let rows: Vec<_> = measurements(61)
        .into_iter()
        .filter(|r| !group.is_systematically_absent(r.hkl).unwrap())
        .collect();
    let a = assess(&rows, cell(), &candidates, &AssessmentOptions::default()).unwrap();
    assert_eq!(a.evidence[1].absent_families, 0);
    assert_eq!(a.evidence[1].absence_log_evidence, 0.);
    assert_eq!(
        a.evidence[0].absence_log_evidence,
        a.evidence[1].absence_log_evidence
    );
}
#[test]
fn all_nonpositive_data_are_evidence_not_a_positive_intensity_requirement() {
    let mut rows = measurements(61);
    for r in &mut rows {
        r.intensity = -1.;
    }
    let a = assess(
        &rows,
        cell(),
        &[hypothesis(1), hypothesis(61)],
        &AssessmentOptions::default(),
    )
    .unwrap();
    assert_eq!(a.evidence[1].strong_violations, 0);
    assert!(a.evidence[1].absent_families > 0);
}
#[test]
fn same_extinction_symbols_remain_ambiguous_and_strong_violations_are_retained() {
    let candidates = [hypothesis(16), hypothesis(47)];
    let a = assess(
        &measurements(61),
        cell(),
        &candidates,
        &AssessmentOptions::default(),
    )
    .unwrap();
    assert_eq!(a.evidence[0].absent_families, a.evidence[1].absent_families);
    assert_eq!(
        a.evidence[0].absence_log_evidence,
        a.evidence[1].absence_log_evidence
    );
    let a = assess(
        &measurements(1),
        cell(),
        &[hypothesis(61)],
        &AssessmentOptions::default(),
    )
    .unwrap();
    assert!(a.evidence[0].strong_violations > 0);
    assert!(a.evidence[0].absence_log_evidence < 0.);
}
#[test]
fn hall_settings_preserve_axis_choices_and_metric_filter_does_not_select_pbca() {
    let catalogue = space_group_settings_for_cell(cell()).unwrap();
    assert!(catalogue.iter().any(|h| h.number == 61));
    assert!(catalogue.iter().any(|h| h.number == 1));
    assert!(catalogue.iter().filter(|h| h.number == 14).count() >= 3);
    assert!(catalogue.iter().all(|h| h.number <= 74));
    let mut cubic = cell();
    cubic.b_angstrom = cubic.a_angstrom;
    cubic.c_angstrom = cubic.a_angstrom;
    assert!(
        space_group_settings_for_cell(cubic)
            .unwrap()
            .iter()
            .any(|h| h.number == 225)
    );
}
#[test]
fn invalid_catalogues_observations_and_metric_are_rejected() {
    let rows = measurements(61);
    let group = hypothesis(61);
    let o = AssessmentOptions::default();
    assert!(assess(&rows, cell(), &[], &o).is_err());
    assert!(assess(&rows, cell(), &[group.clone(), group.clone()], &o).is_err());
    for row in [
        Observation {
            hkl: [0; 3],
            intensity: 1.,
            sigma: 1.,
        },
        Observation {
            hkl: [1, 0, 0],
            intensity: f64::NAN,
            sigma: 1.,
        },
        Observation {
            hkl: [1, 0, 0],
            intensity: 1.,
            sigma: 0.,
        },
    ] {
        assert!(assess(&[row], cell(), std::slice::from_ref(&group), &o).is_err());
    }
    assert!(assess(&rows, cell(), &[hypothesis(225)], &o).is_err());
}
