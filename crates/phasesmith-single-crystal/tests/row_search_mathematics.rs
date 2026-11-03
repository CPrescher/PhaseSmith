// Retain the original standalone row-search test lint policy.
#![allow(missing_docs, clippy::all, clippy::pedantic)]
#![cfg(feature = "row-search")]

use phasesmith_single_crystal::row_search::{
    Options, direction_key, interval_support, phase_subset, run,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
#[test]
fn paper_direction_example() {
    let eps = 0.01;
    let side = (2.0_f64.sqrt() / eps).floor() as usize + 1;
    assert_eq!(
        direction_key([-0.36, -0.8, 0.48], eps),
        Some((side + 106) * side + 22)
    );
}
#[test]
fn interval_sweep_matches_exhaustive_breakpoints() {
    let o = Options::default();
    let mut state = 471u64;
    for n in 1..12 {
        let ds: Vec<_> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                0.01 + (state >> 32) as f64 / (u32::MAX as f64) * 2.
            })
            .collect();
        let ranges: Vec<Vec<_>> = ds
            .iter()
            .map(|&d| {
                (1..=o.kmax)
                    .map(|k| {
                        (
                            d / (k as f64 + o.relative_epsilon),
                            d / (k as f64 - o.relative_epsilon),
                        )
                    })
                    .collect()
            })
            .collect();
        let brute = ranges
            .iter()
            .flatten()
            .map(|&(x, _)| {
                ranges
                    .iter()
                    .filter(|rs| rs.iter().any(|&(l, r)| l <= x && x <= r))
                    .count()
            })
            .max()
            .unwrap();
        assert_eq!(interval_support(&ds, &o), brute);
    }
}
#[test]
fn circular_distinct_sites_matches_brute() {
    let mut state = 2005u64;
    for n in 1..40 {
        let row: Vec<_> = (0..n)
            .map(|id| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let t = (state >> 32) as f64 / (u32::MAX as f64) * 8. - 4.;
                (t, id)
            })
            .collect();
        let d = 0.4;
        let eps = 0.2;
        let mut brute = 0;
        for &(anchor, _) in &row {
            let phase = anchor.rem_euclid(d);
            let mut sites = std::collections::HashSet::new();
            for &(t, _) in &row {
                let r = t.rem_euclid(d);
                let wrap = if r < phase { 1 } else { 0 };
                if r + wrap as f64 * d - phase <= d * eps {
                    sites.insert((t / d).floor() as i64 - wrap);
                }
            }
            brute = brute.max(sites.len());
        }
        assert_eq!(phase_subset(&row, d, eps).len(), brute);
    }
}
#[test]
fn asynchronous_cancellation_stops_and_conserves() {
    let q: Vec<_> = (0..10000)
        .map(|i| {
            [
                i as f64 / 10000.,
                ((i * 37) % 10000) as f64 / 10000.,
                ((i * 1237) % 10000) as f64 / 10000.,
            ]
        })
        .collect();
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(10));
            cancel.store(true, Ordering::Relaxed);
        });
        run(&q, &Options::default(), &cancel).unwrap()
    });
    assert!(result.cancelled);
    assert_eq!(result.labels.len(), q.len());
    assert!(start.elapsed() < Duration::from_secs(5));
}
#[test]
fn cache_cannot_change_grouping() {
    let q: Vec<_> = (-2..=2)
        .flat_map(|i| {
            (-2..=2).flat_map(move |j| {
                (-2..=2).map(move |k| [i as f64 * 0.19, j as f64 * 0.23, k as f64 * 0.31])
            })
        })
        .collect();
    let cancel = AtomicBool::new(false);
    let o = Options::default();
    let a = run(&q, &o, &cancel).unwrap();
    let mut uncached = o.clone();
    uncached.cache_megabytes = 0;
    let b = run(&q, &uncached, &cancel).unwrap();
    assert_eq!(a.labels, b.labels);
}
#[test]
fn exact_closed_interval_boundary() {
    let o = Options {
        kmax: 1,
        relative_epsilon: 0.25,
        ..Options::default()
    };
    // [3/1.25,3/.75] and [5/1.25,5/.75] touch exactly at4.
    assert_eq!(interval_support(&[3., 5.], &o), 2);
}
#[test]
fn exact_phase_width_boundary() {
    assert_eq!(phase_subset(&[(0., 0), (1.25, 1)], 1., 0.25).len(), 2);
    assert_eq!(phase_subset(&[(0., 0), (1.2500001, 1)], 1., 0.25).len(), 1);
}
#[test]
fn mixed_projection_boundaries_split_valid_row_in_two_grid_version() {
    use phasesmith_single_crystal::row_search::group_at_distance;
    let q = vec![
        [0., 0.0039, 0.0019],
        [0.2, 0.0041, 0.0021],
        [0.4, 0.0039, 0.0021],
        [0.6, 0.0041, 0.0019],
    ];
    let ids = vec![0, 1, 2, 3];
    let o = Options::default();
    assert!(group_at_distance(&q, &ids, [1., 0., 0.], 0.2, &o).is_empty());
    let four = Options {
        projection_grids: 4,
        ..o
    };
    assert_eq!(
        group_at_distance(&q, &ids, [1., 0., 0.], 0.2, &four).len(),
        4
    );
}
#[test]
fn local_grid_boundaries_do_not_require_global_parity() {
    use phasesmith_single_crystal::row_search::group_at_distance;
    let mut q = Vec::new();
    for (y, z) in [
        (0.0039, 0.0019),
        (0.0041, 0.0021),
        (0.0039, 0.0021),
        (0.0041, 0.0019),
    ] {
        let i = q.len();
        q.push([i as f64 * 0.2, y, z]);
    }
    for (y, z) in [
        (0.0219, 0.0239),
        (0.0221, 0.0241),
        (0.0219, 0.0241),
        (0.0221, 0.0239),
    ] {
        let i = q.len() - 4;
        q.push([i as f64 * 0.2, y, z]);
    }
    let ids: Vec<_> = (0..q.len()).collect();
    let four = Options {
        projection_grids: 4,
        ..Options::default()
    };
    let adaptive = Options {
        projection_grids: 0,
        ..four.clone()
    };
    assert!(group_at_distance(&q, &ids, [1., 0., 0.], 0.2, &four).len() < 8);
    assert_eq!(
        group_at_distance(&q, &ids, [1., 0., 0.], 0.2, &adaptive).len(),
        8
    );
}
#[test]
fn phase_support_can_collect_triples_after_a_four_point_seed() {
    use phasesmith_single_crystal::row_search::group_for_direction;
    let mut q = Vec::new();
    for i in 0..4 {
        q.push([i as f64 * 0.2, 0.101, 0.101]);
    }
    for i in 0..3 {
        q.push([i as f64 * 0.2, 0.201, 0.201]);
    }
    let ids: Vec<_> = (0..q.len()).collect();
    let seed = Options {
        projection_grids: 0,
        relative_gap: true,
        ..Options::default()
    };
    let support = Options {
        phase_min_row: 3,
        ..seed.clone()
    };
    assert_eq!(group_for_direction(&q, &ids, [1., 0., 0.], &seed).len(), 4);
    assert_eq!(
        group_for_direction(&q, &ids, [1., 0., 0.], &support).len(),
        7
    );
}
#[test]
fn adjacent_cells_do_not_transitively_bridge_a_long_chain() {
    use phasesmith_single_crystal::row_search::group_at_distance;
    let q: Vec<_> = (0..6)
        .map(|i| [i as f64 * 0.05, 0.001 + (i / 2) as f64 * 0.002, 0.001])
        .collect();
    let ids: Vec<_> = (0..6).collect();
    let o = Options {
        projection_grids: 0,
        ..Options::default()
    };
    assert_eq!(group_at_distance(&q, &ids, [1., 0., 0.], 0.05, &o).len(), 4);
}
#[test]
fn eligibility_pruning_is_exact_on_every_four_by_three_cell_pattern() {
    use phasesmith_single_crystal::row_search::group_at_distance;
    for phase_min in [0, 3] {
        let o = Options {
            projection_grids: 0,
            phase_min_row: phase_min,
            ..Options::default()
        };
        let no = Options {
            component_pruning: false,
            ..o.clone()
        };
        for mask in 0..4096u32 {
            let q: Vec<_> = (0..12)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| {
                    [
                        i as f64 * 0.05,
                        0.001 + (i % 4) as f64 * 0.002,
                        0.001 + (i / 4) as f64 * 0.002,
                    ]
                })
                .collect();
            let ids: Vec<_> = (0..q.len()).collect();
            assert_eq!(
                group_at_distance(&q, &ids, [1., 0., 0.], 0.05, &o),
                group_at_distance(&q, &ids, [1., 0., 0.], 0.05, &no),
                "mask {mask}, phase minimum {phase_min}"
            );
        }
    }
}
#[test]
fn adaptive_cache_and_workers_preserve_complete_partitions() {
    let q: Vec<_> = (-2..=2)
        .flat_map(|i| {
            (-2..=2).flat_map(move |j| {
                (-2..=2).map(move |k| [i as f64 * 0.19, j as f64 * 0.23, k as f64 * 0.31])
            })
        })
        .enumerate()
        .filter(|(i, _)| i % 7 != 0)
        .map(|(_, v)| v)
        .collect();
    let cancel = AtomicBool::new(false);
    for phase in [0, 3] {
        let o = Options {
            projection_grids: 0,
            phase_min_row: phase,
            ..Options::default()
        };
        let a = run(&q, &o, &cancel).unwrap();
        let no = Options {
            cache_megabytes: 0,
            component_pruning: false,
            workers: 1,
            ..o
        };
        let b = run(&q, &no, &cancel).unwrap();
        assert_eq!(a.labels, b.labels);
    }
}
#[test]
fn short_row_stack_sweep_matches_generic_exactly() {
    use phasesmith_single_crystal::row_search::phase_subset_generic;
    let mut state = 821u64;
    for n in 1..=20 {
        for _ in 0..100 {
            let row: Vec<_> = (0..n)
                .map(|id| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let t = (state >> 32) as f64 / (u32::MAX as f64) * 4. - 2.;
                    (t, id)
                })
                .collect();
            for eps in [0.02, 0.04, 0.2] {
                assert_eq!(
                    phase_subset(&row, 0.2, eps),
                    phase_subset_generic(&row, 0.2, eps)
                );
            }
        }
    }
}
#[test]
fn phase_radius_is_a_declared_error_convention_not_a_hidden_looser_pairwise_test() {
    let row = vec![(0.0015, 0), (0.0985, 1), (0.2015, 2), (0.2985, 3)];
    assert_eq!(phase_subset(&row, 0.1, 0.02).len(), 2);
    assert_eq!(phase_subset(&row, 0.1, 0.04).len(), 4);
    assert!(
        row.iter()
            .enumerate()
            .all(|(i, &(t, _))| (t - i as f64 * 0.1).abs() <= 0.02 * 0.1)
    );
}

#[test]
fn four_and_eight_workers_preserve_complete_grouping_and_work_counts() {
    let mut q = Vec::new();
    for i in -2..=2 {
        for j in -2..=2 {
            for k in -1..=1 {
                q.push([
                    i as f64 * 0.19 + j as f64 * 0.013,
                    j as f64 * 0.23 - k as f64 * 0.017,
                    k as f64 * 0.31 + i as f64 * 0.011,
                ]);
            }
        }
    }
    for i in 0..23 {
        q.push([
            ((i * 37) % 101) as f64 / 51. - 1.,
            ((i * 19) % 103) as f64 / 52. - 1.,
            ((i * 61) % 107) as f64 / 54. - 1.,
        ]);
    }
    let base_options = Options {
        relative_gap: true,
        projection_grids: 4,
        phase_min_row: 3,
        cache_megabytes: 0,
        workers: 1,
        ..Options::default()
    };
    let reference = run(&q, &base_options, &AtomicBool::new(false)).unwrap();
    assert!(!reference.groups.is_empty());
    for (cache, unlimited) in [
        (0, false),
        (1, false),
        (128, false),
        (512, false),
        (2048, false),
        (0, true),
    ] {
        let o = Options {
            cache_megabytes: cache,
            cache_unlimited: unlimited,
            ..base_options.clone()
        };
        for workers in [1, 2, 4, 8] {
            let mut next = o.clone();
            next.workers = workers;
            let out = run(&q, &next, &AtomicBool::new(false)).unwrap();
            assert_eq!(out.labels, reference.labels);
            assert_eq!(out.groups, reference.groups);
            assert_eq!(out.mean.map(f64::to_bits), reference.mean.map(f64::to_bits));
            assert_eq!(
                out.scale.map(f64::to_bits),
                reference.scale.map(f64::to_bits)
            );
            assert_eq!(out.centers_completed, reference.centers_completed);
            assert_eq!(out.occupied_directions, reference.occupied_directions);
            assert_eq!(out.evaluations_completed, reference.evaluations_completed);
            assert!(!out.cancelled && !out.budget_exhausted);
            if unlimited {
                assert!(out.cache_payload_bytes > 0);
                assert!(out.cached_directions > 0);
                assert_eq!(out.cached_candidate_directions, 0);
            } else {
                assert!(out.cache_payload_bytes <= cache * 1024 * 1024);
            }
        }
    }
}
#[test]
fn worker_limit_accepts_requested_counts_and_rejects_unbounded_counts() {
    for workers in [1, 2, 4, 8] {
        assert!(
            Options {
                workers,
                ..Options::default()
            }
            .validate()
            .is_ok()
        );
    }
    for workers in [0, 9, usize::MAX] {
        assert!(
            Options {
                workers,
                ..Options::default()
            }
            .validate()
            .is_err()
        );
    }
    assert_eq!(Options::default().workers, 2);
}

#[test]
fn cache_json_defaults_and_legacy_integer_settings_round_trip() {
    let omitted: Options = serde_json::from_str("{}").unwrap();
    assert_eq!(omitted.cache_megabytes, 512);
    assert!(!omitted.cache_unlimited);
    for mib in [0, 128, 512, 1024, 2048, 65536] {
        let old_json = serde_json::json!({"cache_megabytes": mib});
        let o: Options = serde_json::from_value(old_json).unwrap();
        assert_eq!(o.cache_megabytes, mib);
        assert!(!o.cache_unlimited);
        o.validate().unwrap();
        let encoded = serde_json::to_value(&o).unwrap();
        let decoded: Options = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.cache_megabytes, mib);
        assert!(!decoded.cache_unlimited);
    }
    let o: Options = serde_json::from_value(serde_json::json!({
        "cache_megabytes": 0, "cache_unlimited": true
    }))
    .unwrap();
    assert!(o.cache_unlimited);
    o.validate().unwrap();
    let again: Options = serde_json::from_str(&serde_json::to_string(&o).unwrap()).unwrap();
    assert!(again.cache_unlimited);
    assert_eq!(again.cache_megabytes, 0);
}

#[test]
fn finite_cache_conversion_rejects_only_platform_overflow() {
    let max_mib = usize::MAX / (1024 * 1024);
    assert!(
        Options {
            cache_megabytes: max_mib,
            ..Options::default()
        }
        .validate()
        .is_ok()
    );
    for mib in [max_mib + 1, usize::MAX] {
        let finite = Options {
            cache_megabytes: mib,
            ..Options::default()
        };
        assert!(finite.validate().unwrap_err().contains("overflows"));
        let unlimited = Options {
            cache_unlimited: true,
            ..finite
        };
        unlimited.validate().unwrap();
    }
}

#[test]
fn malformed_cache_json_is_rejected() {
    for value in [
        serde_json::json!(-1),
        serde_json::json!(1.5),
        serde_json::Value::Null,
        serde_json::json!("unlimited"),
    ] {
        assert!(
            serde_json::from_value::<Options>(serde_json::json!({"cache_megabytes":value}))
                .is_err()
        );
    }
    assert!(
        serde_json::from_value::<Options>(serde_json::json!({"cache_unlimited":"true"})).is_err()
    );
}
