use super::*;
// Moved with the code from phasesmith-workflows (single_crystal/grains.rs).
#[test]
fn refinement_reuse_keys_preserve_signed_zero_lengths_and_order() {
    let fit = RefinementFit {
        h: vec![[1., 0., 0.], [0., 1., 0.]],
        points: vec![[0.25, 0., 0.], [0., 0.2, 0.]],
        updated: [[4., 0., 0.], [0., 5., 0.], [0., 0., 6.]],
    };
    assert!(fit.matches(&fit.h, &fit.points));
    assert!(!fit.matches(&fit.h[..1], &fit.points));
    let mut h = fit.h.clone();
    h[0][1] = -0.;
    assert!(!fit.matches(&h, &fit.points));
    let mut q = fit.points.clone();
    q[0][1] = -0.;
    assert!(!fit.matches(&fit.h, &q));
    q = fit.points.clone();
    q[0][0] = f64::from_bits(q[0][0].to_bits() + 1);
    assert!(!fit.matches(&fit.h, &q));
    q.swap(0, 1);
    h = fit.h.clone();
    h.swap(0, 1);
    assert!(!fit.matches(&h, &q));
}
#[test]
fn primitive_shortcut_matches_independent_exact_minor_index() {
    fn gcd128(mut a: i128, mut b: i128) -> i128 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a.abs()
    }
    let mut seed = 83_u64;
    for scale in [1, 2, 3] {
        for _ in 0..100 {
            let rows: Vec<[i128; 3]> = (0..8)
                .map(|_| {
                    std::array::from_fn(|_| {
                        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        (i128::from((seed >> 32) % 13) - 6) * scale
                    })
                })
                .collect();
            let mut index = 0;
            for i in 0..rows.len() {
                for j in i + 1..rows.len() {
                    for k in j + 1..rows.len() {
                        let [a, b, c] = [rows[i], rows[j], rows[k]];
                        let minor = a[0] * (b[1] * c[2] - b[2] * c[1])
                            - a[1] * (b[0] * c[2] - b[2] * c[0])
                            + a[2] * (b[0] * c[1] - b[1] * c[0]);
                        index = gcd128(index, minor);
                    }
                }
            }
            let h: Vec<_> = rows.iter().map(|h| h.map(|x| x as f64)).collect();
            for shortcut in [false, true] {
                let result = integer_basis(&h, shortcut);
                if index == 0 {
                    assert!(result.is_none());
                } else {
                    assert_eq!(det(result.unwrap()).abs().round() as i128, index);
                }
            }
        }
    }
}
#[test]
fn primitive_shortcut_validates_rows_after_unit_pivots() {
    for invalid in [f64::NAN, f64::INFINITY, 1_000_001., 0.5] {
        let h = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [invalid, 0., 0.]];
        assert!(integer_basis(&h, true).is_none());
        assert!(integer_basis(&h, false).is_none());
    }
    assert!(integer_basis(&[[1., 0., 0.], [0., 1., 0.]], true).is_none());
}
#[test]
fn retained_alias_scores_equal_a_fresh_score_bit_for_bit() {
    let q: Vec<_> = (-3..=3)
        .flat_map(|x| {
            (-3..=3).flat_map(move |y| {
                (-3..=3).map(move |z| [f64::from(x) / 4., f64::from(y) / 5., f64::from(z) / 6.])
            })
        })
        .collect();
    for workers in [1, 4] {
        let e = Engine::new(workers).unwrap();
        let o = Options::default();
        let search = Search { e, o: &o };
        let result = search
            .refine(&q, [[8., 0., 0.], [0., 10., 0.], [0., 0., 12.]])
            .unwrap()
            .unwrap();
        assert!((det(result.direct).abs() - 120.).abs() < 1e-9);
        let fresh = search.score(&q, result.direct).unwrap();
        for ((h, d), (old_h, old_d)) in result.scores.iter().zip(fresh) {
            assert_eq!(h.map(f64::to_bits), old_h.map(f64::to_bits));
            assert_eq!(d.to_bits(), old_d.to_bits());
        }
    }
}
#[test]
fn inferred_integer_lattice_preserves_the_index_and_rejects_rank_deficiency() {
    let h = vec![[2., 0., 0.], [0., 3., 0.], [0., 0., 4.], [2., 3., 4.]];
    let a = integer_basis(&h, false).unwrap();
    assert!((det(a).abs() - 24.).abs() < 1e-12);
    let mut primitive = h;
    primitive.push([1., 1., 1.]);
    let a = integer_basis(&primitive, false).unwrap();
    assert!((det(a).abs() - 2.).abs() < 1e-12);
    assert!(integer_basis(&[[1., 0., 0.], [0., 1., 0.]], false).is_none());
    assert!(integer_basis(&[[1e7, 0., 0.]], false).is_none());
}

/// Deterministic synthetic grains: every node within `qmax` outside a
/// DAC-like cone (`|q_z| < 0.6|q|`), 90 % kept, ±0.001 Å⁻¹ scatter.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn rotation(x: f64, y: f64, z: f64) -> M3 {
    let (sx, cx, sy, cy, sz, cz) = (x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos());
    [
        [cy * cz, -cy * sz, sy],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
    ]
}
fn monoclinic(cell: V3, beta: f64) -> M3 {
    let b = beta.to_radians();
    [
        [cell[0], 0., 0.],
        [0., cell[1], 0.],
        [cell[2] * b.cos(), 0., cell[2] * b.sin()],
    ]
}
fn grain(direct: M3, r: M3, qmax: f64, rng: &mut Lcg) -> Vec<V3> {
    let reciprocal = inv_t(direct).unwrap();
    let mut out = vec![];
    for h in -14..=14 {
        for k in -14..=14 {
            for l in -14..=14 {
                if (h, k, l) == (0, 0, 0) {
                    continue;
                }
                let q = row(
                    row([f64::from(h), f64::from(k), f64::from(l)], reciprocal),
                    r,
                );
                let n = norm(q);
                if n > qmax || q[2].abs() > 0.6 * n || rng.next() > 0.9 {
                    continue;
                }
                out.push(std::array::from_fn(|j| q[j] + 0.002 * (rng.next() - 0.5)));
            }
        }
    }
    out
}
fn noise(n: usize, rng: &mut Lcg) -> Vec<V3> {
    (0..n)
        .map(|_| std::array::from_fn(|_| 1.8 * rng.next() - 0.9))
        .filter(|v: &V3| norm(*v) < 0.9)
        .collect()
}
fn shuffle(mut q: Vec<V3>, rng: &mut Lcg) -> Vec<V3> {
    for i in (1..q.len()).rev() {
        q.swap(i, (rng.next() * (i + 1) as f64) as usize);
    }
    q
}
fn controls() -> Vec<M3> {
    (0..64)
        .map(|i| {
            let i = f64::from(i);
            rotation(0.3 + 0.7 * i, 1.1 * i, -0.5 + 0.37 * i)
        })
        .collect()
}
const ENSTATITE: (V3, f64) = ([5.18, 8.81, 18.23], 90.);
const MONOCLINIC: (V3, f64) = ([7.5, 9.2, 12.4], 104.);
const SMALL: (V3, f64) = ([4.9, 6.1, 7.3], 97.);
/// Two grains of similar size, and a third with about a tenth of the
/// positions (a smaller cell measured to lower q), plus noise.
fn three_grains() -> Vec<V3> {
    let rng = &mut Lcg(7);
    let g1 = grain(
        monoclinic(ENSTATITE.0, ENSTATITE.1),
        rotation(0.4, 1.1, -0.7),
        0.9,
        rng,
    );
    let g2 = grain(
        monoclinic(MONOCLINIC.0, MONOCLINIC.1),
        rotation(-1.3, 0.2, 2.1),
        0.9,
        rng,
    );
    let g3 = grain(
        monoclinic(SMALL.0, SMALL.1),
        rotation(2.2, -0.9, 0.5),
        0.7,
        rng,
    );
    let n = noise(300, rng);
    shuffle([g1, g2, g3, n].concat(), rng)
}
fn training(n: usize) -> Vec<bool> {
    (0..n).map(|i| i % 5 != 0).collect()
}
/// Accepted grains matching this cell: sorted lengths within 0.5 %, volume 1.5 %.
fn accepted(grains: &[Grain], cell: (V3, f64)) -> usize {
    let want = monoclinic(cell.0, cell.1);
    let lengths = |a: M3| {
        let mut l = reduce(a).map(norm);
        l.sort_by(f64::total_cmp);
        l
    };
    let (wl, wv) = (lengths(want), det(want).abs());
    grains
        .iter()
        .filter(|g| g.accepted)
        .filter(|g| {
            let l = lengths(g.direct);
            l.iter().zip(&wl).all(|(a, b)| (a / b - 1.).abs() < 0.005)
                && (det(g.direct).abs() / wv - 1.).abs() < 0.015
        })
        .count()
}
fn run(q: &[V3], options: &Options, method: &Method) -> Vec<Grain> {
    find(
        q,
        &training(q.len()),
        &controls(),
        options,
        method,
        &mut |_, _| Ok(()),
    )
    .unwrap()
}

#[test]
fn defaults_report_two_of_three_grains_as_supercells() {
    // The workflow's round selection (largest support beyond the uniform
    // expectation): a supercell catches its grain's positions plus chance
    // matches and wins the round; the sublattice check keeps it because the
    // chance matches are more than a tenth of its support.
    let q = three_grains();
    let grains = run(&q, &Options::default(), &Method::Fft);
    assert_eq!(accepted(&grains, ENSTATITE), 1);
    assert_eq!(accepted(&grains, MONOCLINIC), 0);
    assert_eq!(accepted(&grains, SMALL), 0);
    let volumes: Vec<f64> = grains
        .iter()
        .filter(|g| g.accepted)
        .map(|g| det(g.direct).abs())
        .collect();
    assert_eq!(volumes.len(), 3);
    assert!(
        volumes.iter().filter(|&&v| v > 1500.).count() >= 1,
        "{volumes:?}"
    );
    assert!(
        volumes
            .iter()
            .any(|&v| v > 2.5 * det(monoclinic(SMALL.0, SMALL.1)).abs() && v < 1500.),
        "{volumes:?}"
    );
}

#[test]
fn round_controls_recover_the_small_grain_and_tight_matches_keep_spurious_cells_out() {
    let q = three_grains();
    let options = Options {
        round_controls: 16,
        min_tight_fraction: 0.75,
        ..Options::default()
    };
    let grains = run(&q, &options, &Method::Fft);
    for cell in [ENSTATITE, MONOCLINIC, SMALL] {
        assert_eq!(accepted(&grains, cell), 1, "{cell:?}");
    }
    assert_eq!(grains.iter().filter(|g| g.accepted).count(), 3);
}

#[test]
fn row_search_and_both_methods_find_the_grains() {
    let rng = &mut Lcg(11);
    let g1 = grain(
        monoclinic(ENSTATITE.0, ENSTATITE.1),
        rotation(0.4, 1.1, -0.7),
        0.9,
        rng,
    );
    let g2 = grain(
        monoclinic(MONOCLINIC.0, MONOCLINIC.1),
        rotation(-1.3, 0.2, 2.1),
        0.9,
        rng,
    );
    let q = shuffle([g1, g2, noise(400, rng)].concat(), rng);
    let rows = crate::row_search::Options {
        projection_grids: 4,
        phase_min_row: 3,
        relative_gap: true,
        naive_max_points: 2048,
        workers: 4,
        seconds: 120.,
        ..crate::row_search::Options::default()
    };
    for method in [Method::Rows(rows.clone()), Method::Both(rows)] {
        let grains = run(&q, &Options::default(), &method);
        assert_eq!(accepted(&grains, ENSTATITE), 1, "{method:?}");
        assert_eq!(accepted(&grains, MONOCLINIC), 1, "{method:?}");
    }
}

#[test]
fn max_grains_bounds_the_rounds_and_random_positions_give_no_grain() {
    let q = three_grains();
    let one = Options {
        max_grains: 1,
        ..Options::default()
    };
    assert_eq!(run(&q, &one, &Method::Fft).len(), 1);
    let rng = &mut Lcg(3);
    let random = noise(2600, rng);
    let grains = run(&random, &Options::default(), &Method::Fft);
    assert!(grains.iter().all(|g| !g.accepted));
}

#[test]
fn progress_cancels_and_options_are_checked() {
    let q = three_grains();
    let stop = find(
        &q,
        &training(q.len()),
        &controls(),
        &Options::default(),
        &Method::Fft,
        &mut |_, _| Err(SpotError("cancelled".into())),
    );
    assert_eq!(stop.err().map(|e| e.0), Some("cancelled".into()));
    for bad in [
        Options {
            max_grains: 0,
            ..Options::default()
        },
        Options {
            grid_size: 255,
            ..Options::default()
        },
        Options {
            min_tight_fraction: 1.5,
            ..Options::default()
        },
        Options {
            reciprocal_tolerance: f64::NAN,
            ..Options::default()
        },
        Options {
            round_controls: 200,
            ..Options::default()
        },
    ] {
        assert!(bad.check().is_err(), "{bad:?}");
    }
    assert!(Options::default().check().is_ok());
    // More round controls than supplied rotations.
    let too_many = Options {
        round_controls: 65,
        ..Options::default()
    };
    assert!(discover(&q, &too_many, &controls(), &Method::Fft, &mut |_, _| Ok(())).is_err());
}

/// Rotation by `degrees` about the unit vector along (1, 2, 3).
fn tilt(degrees: f64) -> M3 {
    let n = [1., 2., 3.].map(|x: f64| x / 14_f64.sqrt());
    let (s, c) = degrees.to_radians().sin_cos();
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let cross = match (i, j) {
                (0, 1) => -n[2],
                (0, 2) => n[1],
                (1, 0) => n[2],
                (1, 2) => -n[0],
                (2, 0) => -n[1],
                (2, 1) => n[0],
                _ => 0.,
            };
            c * f64::from(u8::from(i == j)) + s * cross + (1. - c) * n[i] * n[j]
        })
    })
}
/// One crystal whose reflections come from two orientations `degrees`
/// apart, each reflection from one of them at random, plus noise. Returns
/// the positions and the two direct bases.
fn two_domains(degrees: f64) -> (Vec<V3>, M3, M3) {
    let rng = &mut Lcg(11);
    let direct = monoclinic(ENSTATITE.0, ENSTATITE.1);
    let first = rotation(0.4, 1.1, -0.7);
    let second = mul(first, tilt(degrees));
    let reciprocal = inv_t(direct).unwrap();
    let mut q = vec![];
    for h in -14..=14 {
        for k in -14..=14 {
            for l in -14..=14 {
                let node = row([f64::from(h), f64::from(k), f64::from(l)], reciprocal);
                let n = norm(node);
                if (h, k, l) == (0, 0, 0) || n > 0.9 || rng.next() > 0.9 {
                    continue;
                }
                let p = row(node, if rng.next() < 0.5 { first } else { second });
                q.push(std::array::from_fn(|j| p[j] + 0.002 * (rng.next() - 0.5)));
            }
        }
    }
    q.extend(noise(200, rng));
    (shuffle(q, rng), mul(direct, first), mul(direct, second))
}

#[test]
fn misorientation_is_the_rotation_between_settings_of_one_cell() {
    let a = mul(
        monoclinic(MONOCLINIC.0, MONOCLINIC.1),
        rotation(0.4, 1.1, -0.7),
    );
    // Another setting of the same lattice, turned by 0.26°.
    let setting = [[1., 0., 0.], [1., 1., 0.], [0., -1., -1.]];
    let b = mul(mul(setting, a), tilt(0.26));
    let angle = misorientation(a, b).unwrap();
    assert!((angle - 0.26).abs() < 1e-6, "{angle}");
    assert!((misorientation(b, a).unwrap() - 0.26).abs() < 1e-6);
    assert!(misorientation(a, a).unwrap() < 1e-6);
    // Inverted basis: the lattice is the same.
    let inverted = a.map(|r| r.map(|x| -x));
    assert!(misorientation(a, inverted).unwrap() < 1e-6);
    // A 1 % strain along c is another cell; 0.1 % is not and adds no angle.
    let strained = |f: f64| mul([[1., 0., 0.], [0., 1., 0.], [0., 0., f]], a);
    assert!(misorientation(a, strained(1.01)).is_none());
    assert!(misorientation(a, strained(1.001)).unwrap() < 0.05);
    // A different crystal, a supercell and a singular basis.
    let other = mul(monoclinic(SMALL.0, SMALL.1), rotation(0.4, 1.1, -0.7));
    assert!(misorientation(a, other).is_none());
    let doubled = mul([[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]], a);
    assert!(misorientation(a, doubled).is_none());
    assert!(misorientation(a, [[0.; 3]; 3]).is_none());
}

#[test]
fn domains_of_one_crystal_merge_and_separate_grains_stay_apart() {
    let with = |merge_angle: f64| Options {
        merge_angle,
        ..Options::default()
    };
    let owned = |q: &[V3], models: &[M3], o: &Options| {
        let search = Search {
            e: Engine::new(o.workers).unwrap(),
            o,
        };
        let labels = search.assign(q, models).unwrap();
        (1..=models.len() as i32)
            .map(|i| labels.iter().filter(|&&l| l == i).count())
            .collect::<Vec<_>>()
    };

    // 0.26° apart: one lattice that owns nearly all reflections.
    let (q, a, b) = two_domains(0.26);
    let reflections = q.len() - 200;
    let merged = merge_domains(&q, &[a, b], &with(0.5)).unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].domains, 2);
    assert!((merged[0].angle - 0.26).abs() < 0.01, "{}", merged[0].angle);
    assert!(misorientation(merged[0].direct, a).unwrap() < 0.26);
    let count = owned(&q, &[merged[0].direct], &with(0.5))[0];
    assert!(count as f64 > 0.95 * reflections as f64, "{count}");
    // Off, or a limit below the angle: both come back untouched.
    for limit in [0., 0.2] {
        let kept = merge_domains(&q, &[a, b], &with(limit)).unwrap();
        assert_eq!(
            kept.iter().map(|m| m.direct).collect::<Vec<_>>(),
            vec![a, b]
        );
        assert!(kept.iter().all(|m| m.domains == 1 && m.angle == 0.));
    }

    // 3° apart: within a 5° limit, but one lattice cannot hold both, so the
    // merge is refused and the grains stay as they were.
    let (q, a, b) = two_domains(3.);
    assert!((misorientation(a, b).unwrap() - 3.).abs() < 1e-6);
    let apart = merge_domains(&q, &[a, b], &with(5.)).unwrap();
    assert_eq!(
        apart.iter().map(|m| m.direct).collect::<Vec<_>>(),
        vec![a, b]
    );

    // Three lattices: the two domains merge, the other crystal is refit only.
    let (mut q, a, b) = two_domains(0.26);
    let other = monoclinic(SMALL.0, SMALL.1);
    let turn = rotation(2.2, -0.9, 0.5);
    q.extend(grain(other, turn, 0.7, &mut Lcg(9)));
    let third = mul(other, turn);
    let merged = merge_domains(&q, &[a, third, b], &with(0.5)).unwrap();
    // The merged lattice takes the place of its larger part.
    assert_eq!(merged.len(), 2);
    let joined = merged.iter().find(|m| m.domains == 2).unwrap();
    let single = merged.iter().find(|m| m.domains == 1).unwrap();
    assert!(misorientation(joined.direct, a).unwrap() < 0.26);
    assert!(misorientation(single.direct, third).unwrap() < 0.05);
    assert_eq!(single.angle, 0.);
}

#[test]
fn merging_leaves_distinct_grains_untouched_and_checks_its_angle() {
    let q = three_grains();
    let controlled = |merge_angle: f64| Options {
        round_controls: 16,
        min_tight_fraction: 0.75,
        merge_angle,
        ..Options::default()
    };
    let bases = |o: &Options| {
        run(&q, o, &Method::Fft)
            .iter()
            .map(|g| g.direct)
            .collect::<Vec<_>>()
    };
    assert_eq!(bases(&controlled(0.)), bases(&controlled(0.5)));
    assert_eq!(
        bases(&Options::default()),
        bases(&Options {
            merge_angle: 0.5,
            ..Options::default()
        })
    );
    for bad in [-0.1, 10.5, f64::NAN] {
        let o = Options {
            merge_angle: bad,
            ..Options::default()
        };
        assert!(o.check().is_err(), "{bad}");
        assert!(merge_domains(&q, &[], &o).is_err());
    }
    assert_eq!(merge_domains(&q, &[], &controlled(0.5)).unwrap(), vec![]);
}
