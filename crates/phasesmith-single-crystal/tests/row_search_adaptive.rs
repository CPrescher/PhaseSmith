// Retain the original standalone row-search test lint policy.
#![allow(missing_docs, clippy::all, clippy::pedantic)]
#![cfg(feature = "row-search")]

use phasesmith_single_crystal::row_search::{Options, Point, ResultData, run};
use std::sync::atomic::AtomicBool;

fn cloud() -> Vec<Point> {
    (0..4)
        .flat_map(|i| {
            (0..3).flat_map(move |j| {
                (0..3).map(move |k| {
                    let jitter = ((i * 7 + j * 11 + k * 13) % 5) as f64 * 1e-6;
                    [i as f64 * 0.19, j as f64 * 0.23 + jitter, k as f64 * 0.31]
                })
            })
        })
        .collect()
}
fn options() -> Options {
    Options {
        projection_grids: 4,
        phase_min_row: 3,
        relative_gap: true,
        directions: 512,
        groups: 6,
        ..Options::default()
    }
}
fn compute(p: &[Point], o: &Options) -> ResultData {
    run(p, o, &AtomicBool::new(false)).unwrap()
}
fn same(a: &ResultData, b: &ResultData) {
    assert_eq!(a.labels, b.labels);
    assert_eq!(a.groups, b.groups);
    assert_eq!(a.centers_completed, b.centers_completed);
    assert_eq!(a.occupied_directions, b.occupied_directions);
    assert_eq!(a.evaluations_completed, b.evaluations_completed);
    for k in 0..3 {
        assert_eq!(a.mean[k].to_bits(), b.mean[k].to_bits());
        assert_eq!(a.scale[k].to_bits(), b.scale[k].to_bits());
    }
    assert!(!a.budget_exhausted && !a.cancelled && !b.budget_exhausted && !b.cancelled);
}
#[test]
fn crossover_matches_the_two_explicit_published_methods() {
    let p = cloud();
    let base = options();
    let smart = compute(&p, &base);
    let naive = compute(
        &p,
        &Options {
            smart: false,
            ..base.clone()
        },
    );
    let automatic = compute(
        &p,
        &Options {
            naive_max_points: p.len(),
            ..base.clone()
        },
    );
    same(&automatic, &naive);
    assert_eq!(automatic.effective_smart, Some(false));
    assert!(automatic.options.smart); // requested policy remains observable
    assert!(automatic.labels.iter().filter(|&&g| g > 0).count() >= 20);
    let above = compute(
        &p,
        &Options {
            naive_max_points: p.len() - 1,
            ..base.clone()
        },
    );
    same(&above, &smart);
    assert_eq!(above.effective_smart, Some(true));
    let explicit = compute(
        &p,
        &Options {
            smart: false,
            naive_max_points: 1,
            ..base
        },
    );
    same(&explicit, &naive);
}
#[test]
fn duplicate_padding_cannot_change_the_method_or_unique_assignments() {
    let p = cloud();
    let o = Options {
        naive_max_points: p.len(),
        ..options()
    };
    let expected = compute(&p, &o);
    let mut padded = p.clone();
    padded.extend(std::iter::repeat_n(p[0], 50));
    let actual = compute(&padded, &o);
    assert_eq!(actual.effective_smart, Some(false));
    assert_eq!(actual.duplicate_observations, 50);
    assert_eq!(&actual.labels[..p.len()], expected.labels);
    assert!(actual.labels[p.len()..].iter().all(|&g| g == 0));
}
#[test]
fn automatic_mode_is_worker_cache_and_permutation_deterministic() {
    let p = cloud();
    let o = Options {
        naive_max_points: p.len(),
        ..options()
    };
    let expected = compute(&p, &o);
    for workers in [1, 2, 4, 8] {
        for (cache_megabytes, cache_unlimited) in [(0, false), (512, false), (0, true)] {
            let actual = compute(
                &p,
                &Options {
                    workers,
                    cache_megabytes,
                    cache_unlimited,
                    ..o.clone()
                },
            );
            same(&actual, &expected);
        }
    }
    let reversed: Vec<_> = p.iter().copied().rev().collect();
    let actual = compute(&reversed, &o);
    assert_eq!(
        actual.labels.into_iter().rev().collect::<Vec<_>>(),
        expected.labels
    );
}
#[test]
fn legacy_options_and_results_remain_deserializable() {
    let o: Options = serde_json::from_str("{\"smart\":true}").unwrap();
    assert_eq!(o.naive_max_points, 0);
    let result = compute(&cloud(), &options());
    let mut json = serde_json::to_value(&result).unwrap();
    json.as_object_mut().unwrap().remove("effective_smart");
    let old: ResultData = serde_json::from_value(json).unwrap();
    assert_eq!(old.effective_smart, None);
    same(&old, &result);
}
