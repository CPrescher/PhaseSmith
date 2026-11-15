use super::*;

/// Deterministic generator, as in the grain tests.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2. * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() * (hi - lo + 1) as f64) as i64
    }
}
fn rotation(rng: &mut Lcg) -> Matrix3<f64> {
    // Uniform random rotation from a unit quaternion.
    let q = [rng.gauss(), rng.gauss(), rng.gauss(), rng.gauss()];
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    let [w, x, y, z] = q.map(|v| v / n);
    Matrix3::new(
        1. - 2. * (y * y + z * z),
        2. * (x * y - z * w),
        2. * (x * z + y * w),
        2. * (x * y + z * w),
        1. - 2. * (x * x + z * z),
        2. * (y * z - x * w),
        2. * (x * z - y * w),
        2. * (y * z + x * w),
        1. - 2. * (x * x + y * y),
    )
}
/// Random unimodular, determinant +1, moderate coefficients.
fn unimodular(rng: &mut Lcg) -> Transform {
    let mut t: Transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for _ in 0..6 {
        let i = usize::try_from(rng.int(0, 2)).unwrap();
        let j = (i + usize::try_from(rng.int(1, 2)).unwrap()) % 3;
        let k = rng.int(-2, 2);
        for c in 0..3 {
            t[i][c] += k * t[j][c];
        }
        if rng.next() < 0.3 {
            t.swap(i, j);
            t[i] = t[i].map(|x| -x);
        }
    }
    assert_eq!(det_i(t), 1);
    t
}
fn primitive(bravais: Bravais, conventional: M3) -> M3 {
    let c: [[f64; 3]; 3] = match bravais.centring() {
        Centring::P => [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        Centring::C => [[0.5, 0.5, 0.], [-0.5, 0.5, 0.], [0., 0., 1.]],
        Centring::I => [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
        Centring::F => [[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]],
        Centring::R => [
            [2. / 3., 1. / 3., 1. / 3.],
            [-1. / 3., 1. / 3., 1. / 3.],
            [-1. / 3., -2. / 3., 1. / 3.],
        ],
    };
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| c[i][k] * conventional[k][j]).sum())
    })
}
/// Scrambled, rotated primitive basis of a conventional cell, with a
/// symmetric metric strain of relative size `strain`.
fn scrambled(bravais: Bravais, cell: [f64; 6], strain: f64, rng: &mut Lcg) -> M3 {
    let a = primitive(bravais, cell_basis(cell).unwrap());
    let e: M3 = {
        let mut e = [[0.; 3]; 3];
        for i in 0..3 {
            for j in i..3 {
                let v = strain * rng.gauss();
                e[i][j] = v;
                e[j][i] = v;
            }
        }
        e
    };
    let strained: M3 =
        a.map(|v| std::array::from_fn(|j| v[j] + (0..3).map(|k| v[k] * e[k][j]).sum::<f64>()));
    let r = rotation(rng);
    rotate_rows(apply(unimodular(rng), strained), &r)
}

/// Generic cells of the 14 types (no accidental higher symmetry), and the
/// types each lattice must be reported with (its lattice subgroups).
fn cases() -> Vec<(Bravais, [f64; 6], &'static [&'static str])> {
    vec![
        (
            Bravais::TriclinicP,
            [5.1, 6.2, 7.3, 78.3, 83.1, 71.4],
            &["aP"],
        ),
        (
            Bravais::MonoclinicP,
            [5.3, 7.1, 8.4, 90., 104.2, 90.],
            &["mP", "aP"],
        ),
        (
            Bravais::MonoclinicC,
            [9.6, 5.7, 6.1, 90., 103.5, 90.],
            &["mC", "aP"],
        ),
        (
            Bravais::OrthorhombicP,
            [4.1, 5.7, 7.9, 90., 90., 90.],
            &["oP", "mP", "aP"],
        ),
        (
            Bravais::OrthorhombicC,
            [5.0, 8.3, 6.4, 90., 90., 90.],
            &["oC", "mP", "mC", "aP"],
        ),
        (
            Bravais::OrthorhombicI,
            [4.5, 6.2, 7.9, 90., 90., 90.],
            &["oI", "mC", "aP"],
        ),
        (
            Bravais::OrthorhombicF,
            [5.5, 7.0, 9.0, 90., 90., 90.],
            &["oF", "mC", "aP"],
        ),
        (
            Bravais::TetragonalP,
            [4.4, 4.4, 6.7, 90., 90., 90.],
            &["tP", "oP", "oC", "mP", "mC", "aP"],
        ),
        (
            Bravais::TetragonalI,
            [3.9, 3.9, 9.8, 90., 90., 90.],
            &["tI", "oI", "oF", "mC", "aP"],
        ),
        (
            Bravais::HexagonalP,
            [3.2, 3.2, 5.2, 90., 90., 120.],
            &["hP", "oC", "mP", "mC", "aP"],
        ),
        (
            Bravais::RhombohedralR,
            [5.0, 5.0, 14.0, 90., 90., 120.],
            &["hR", "mC", "aP"],
        ),
        (
            Bravais::CubicP,
            [4.0, 4.0, 4.0, 90., 90., 90.],
            &["cP", "tP", "hR", "oP", "oC", "mP", "mC", "aP"],
        ),
        (
            Bravais::CubicI,
            [3.3, 3.3, 3.3, 90., 90., 90.],
            &["cI", "tI", "hR", "oI", "oF", "mC", "aP"],
        ),
        (
            Bravais::CubicF,
            [5.6, 5.6, 5.6, 90., 90., 90.],
            &["cF", "tI", "hR", "oI", "oF", "mC", "aP"],
        ),
    ]
}
fn symbols(matches: &[BravaisMatch]) -> Vec<&'static str> {
    matches.iter().map(|m| m.bravais.symbol()).collect()
}
fn same_cell(a: [f64; 6], b: [f64; 6], length: f64, angle: f64) -> bool {
    (0..3).all(|k| (a[k] / b[k] - 1.).abs() <= length)
        && (3..6).all(|k| (a[k] - b[k]).abs() <= angle)
}
fn sorted(set: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = set.iter().map(|s| (*s).to_string()).collect();
    v.sort();
    v
}

#[test]
fn all_fourteen_types_are_recognised_from_scrambled_rotated_primitive_bases() {
    let rng = &mut Lcg(17);
    for (bravais, cell, expected) in cases() {
        for _ in 0..4 {
            let input = scrambled(bravais, cell, 0., rng);
            let found = search(input, &SearchOptions::default()).unwrap();
            assert_eq!(
                found[0].bravais,
                bravais,
                "{}: {:?}",
                bravais.symbol(),
                symbols(&found)
            );
            assert_eq!(found.last().unwrap().bravais, Bravais::TriclinicP);
            assert_eq!(
                sorted(&symbols(&found)),
                sorted(expected),
                "{}",
                bravais.symbol()
            );
            for m in &found {
                assert_eq!(det_i(m.transform), m.bravais.centring().multiplicity());
                let exact = apply(m.transform, input);
                assert!(
                    exact
                        .iter()
                        .flatten()
                        .zip(m.conventional.iter().flatten())
                        .all(|(x, y)| (x - y).abs() < 1e-9)
                );
                assert!(det(m.conventional) > 0.);
                assert_eq!(m.cell, cell_parameters(m.conventional));
                assert!(m.length_deviation < 1e-9 && m.angle_deviation < 1e-7);
                assert!(same_cell(
                    m.idealized_cell(),
                    m.constrained_cell,
                    1e-9,
                    1e-7
                ));
            }
            let top = &found[0];
            if bravais != Bravais::TriclinicP {
                assert!(
                    same_cell(top.constrained_cell, cell, 1e-9, 1e-7),
                    "{}: {:?} vs {:?}",
                    bravais.symbol(),
                    top.constrained_cell,
                    cell
                );
            }
            // Volume per lattice point is that of the input.
            assert!(
                (det(top.conventional) / det(input) - top.transform_index() as f64).abs() < 1e-9
            );
        }
    }
}

impl BravaisMatch {
    fn idealized_cell(&self) -> [f64; 6] {
        cell_parameters(self.idealized)
    }
    fn transform_index(&self) -> i64 {
        det_i(self.transform)
    }
}

#[test]
fn settings_follow_the_documented_conventions() {
    let rng = &mut Lcg(5);
    for (bravais, cell, _) in cases() {
        let input = scrambled(bravais, cell, 2e-4, rng);
        for m in search(input, &SearchOptions::default()).unwrap() {
            let [a, b, c, al, be, ga] = m.cell;
            match m.bravais.system() {
                LatticeSystem::Monoclinic => {
                    assert!(be >= 90. && al >= 90.);
                    if m.bravais == Bravais::MonoclinicP {
                        assert!(a <= c);
                    }
                }
                LatticeSystem::Orthorhombic | LatticeSystem::Cubic => {
                    assert!(a <= b && al >= 90. && be >= 90.);
                    if m.bravais.centring() != Centring::C {
                        assert!(b <= c);
                    }
                }
                LatticeSystem::Tetragonal => assert!(a <= b),
                LatticeSystem::Hexagonal | LatticeSystem::Rhombohedral => {
                    assert!(a <= b && ga > 100.);
                }
                LatticeSystem::Triclinic => {
                    // Niggli: a ≤ b ≤ c, all angles acute or all non-acute.
                    assert!(a <= b + 1e-9 && b <= c + 1e-9);
                    assert!(
                        (al < 90. && be < 90. && ga < 90.)
                            || (al >= 90. - 1e-6 && be >= 90. - 1e-6 && ga >= 90. - 1e-6)
                    );
                }
            }
        }
    }
}

#[test]
fn result_does_not_depend_on_the_input_basis() {
    let rng = &mut Lcg(29);
    for (bravais, cell, _) in cases() {
        // Fixed strained lattice, then two independent bases and rotations.
        let reference = scrambled(bravais, cell, 3e-4, rng);
        let first = search(reference, &SearchOptions::default()).unwrap();
        for _ in 0..3 {
            let r = rotation(rng);
            let other = rotate_rows(apply(unimodular(rng), reference), &r);
            let second = search(other, &SearchOptions::default()).unwrap();
            assert_eq!(symbols(&first), symbols(&second), "{}", bravais.symbol());
            for (x, y) in first.iter().zip(&second) {
                assert!(
                    same_cell(x.cell, y.cell, 1e-9, 1e-7),
                    "{} {}: {:?} vs {:?}",
                    bravais.symbol(),
                    x.bravais.symbol(),
                    x.cell,
                    y.cell
                );
                // The same lattice vectors: rotating the first conventional
                // basis gives the second.
                let turned = rotate_rows(x.conventional, &r);
                assert!(
                    turned
                        .iter()
                        .flatten()
                        .zip(y.conventional.iter().flatten())
                        .all(|(p, q)| (p - q).abs() < 1e-8),
                    "{} {}",
                    bravais.symbol(),
                    x.bravais.symbol()
                );
            }
        }
    }
}

#[test]
fn small_metric_noise_keeps_the_type_and_large_deviations_drop_it() {
    let rng = &mut Lcg(41);
    for (bravais, cell, _) in cases() {
        for _ in 0..3 {
            let found = search(
                scrambled(bravais, cell, 1e-3, rng),
                &SearchOptions::default(),
            )
            .unwrap();
            assert_eq!(found[0].bravais, bravais, "{}", bravais.symbol());
            assert!(
                found[0].length_deviation > 0.
                    || found[0].angle_deviation > 0.
                    || bravais == Bravais::TriclinicP
            );
        }
    }
    let options = SearchOptions::default();
    // Tetragonal with a/b off by 1.5 %: oP stays, tP goes. The diagonal
    // oC cell (a ± b, c) still fits: its angle is off by only 0.85°.
    let near = search(
        scrambled(
            Bravais::OrthorhombicP,
            [4.4, 4.466, 6.7, 90., 90., 90.],
            0.,
            rng,
        ),
        &options,
    )
    .unwrap();
    assert_eq!(symbols(&near), ["oC", "oP", "mC", "mP", "aP"]);
    assert!((near[0].angle_deviation - 0.85).abs() < 0.01);
    // Within 1 %: tP reported, with its measured mismatch.
    let inside = search(
        scrambled(
            Bravais::OrthorhombicP,
            [4.4, 4.44, 6.7, 90., 90., 90.],
            0.,
            rng,
        ),
        &options,
    )
    .unwrap();
    assert_eq!(inside[0].bravais, Bravais::TetragonalP);
    assert!((inside[0].length_deviation - 0.04 / 4.42).abs() < 1e-9);
    // Orthorhombic with γ = 91.5°: monoclinic only (unique axis c becomes b).
    let skew = search(
        scrambled(
            Bravais::MonoclinicP,
            [4.1, 7.9, 5.7, 90., 91.5, 90.],
            0.,
            rng,
        ),
        &options,
    )
    .unwrap();
    assert_eq!(symbols(&skew), ["mP", "aP"]);
    assert!((skew[0].cell[4] - 91.5).abs() < 1e-7);
    // The same at 0.6°: oP with that deviation.
    let slight = search(
        scrambled(
            Bravais::MonoclinicP,
            [4.1, 7.9, 5.7, 90., 90.6, 90.],
            0.,
            rng,
        ),
        &options,
    )
    .unwrap();
    assert_eq!(slight[0].bravais, Bravais::OrthorhombicP);
    assert!((slight[0].angle_deviation - 0.6).abs() < 1e-7);
    // A wider tolerance admits the 91.5° cell; a tighter one rejects 0.6°.
    let wide = SearchOptions {
        angle_tolerance: 2.,
        ..options
    };
    assert_eq!(
        search(skew[0].conventional, &wide).unwrap()[0].bravais,
        Bravais::OrthorhombicP
    );
    let tight = SearchOptions {
        angle_tolerance: 0.5,
        ..options
    };
    assert_eq!(
        search(slight[0].conventional, &tight).unwrap()[0].bravais,
        Bravais::MonoclinicP
    );
    // Rhombohedral at 61° is not cubic F (60°); at 60.4° it is.
    let rhombohedron = |alpha: f64| {
        let r = cell_basis([5., 5., 5., alpha, alpha, alpha]).unwrap();
        search(r, &options).unwrap()
    };
    assert_eq!(rhombohedron(61.)[0].bravais, Bravais::RhombohedralR);
    assert_eq!(rhombohedron(60.4)[0].bravais, Bravais::CubicF);
    assert_eq!(rhombohedron(90.)[0].bravais, Bravais::CubicP);
    assert_eq!(rhombohedron(109.471_220_6)[0].bravais, Bravais::CubicI);
}

#[test]
fn lower_symmetry_is_never_promoted() {
    let rng = &mut Lcg(43);
    let options = SearchOptions::default();
    // Triclinic close to monoclinic only within 3°: aP alone.
    let a = search(
        scrambled(
            Bravais::TriclinicP,
            [5.3, 7.1, 8.4, 93., 104.2, 87.],
            0.,
            rng,
        ),
        &options,
    )
    .unwrap();
    assert_eq!(symbols(&a), ["aP"]);
    // Monoclinic with β = 90.0 but a ≈ c within 3 %: no tetragonal.
    let m = search(
        scrambled(
            Bravais::OrthorhombicP,
            [5.0, 7.1, 5.15, 90., 90., 90.],
            0.,
            rng,
        ),
        &options,
    )
    .unwrap();
    assert!(
        m.iter().all(|x| x.bravais.holohedry_order() <= 8),
        "{:?}",
        symbols(&m)
    );
    // Hexagonal metric with γ = 118.5°: no hP.
    let h = search(
        scrambled(
            Bravais::MonoclinicP,
            [3.2, 5.2, 3.2, 90., 118.5, 90.],
            0.,
            rng,
        ),
        &options,
    )
    .unwrap();
    assert!(!symbols(&h).contains(&"hP"));
}

#[test]
fn a_supercell_with_a_higher_metric_is_not_reported() {
    let rng = &mut Lcg(47);
    let options = SearchOptions::default();
    // Doubling a gives an 8 × 8.03 × 6 tetragonal metric; tripling a and
    // doubling c of 2.68 × 8.02 × 4 a cubic one. Neither cell is a centred
    // cell of the lattice, so oP stays the highest type, primitive.
    for cell in [
        [4.0, 8.03, 6.0, 90., 90., 90.],
        [2.68, 8.02, 4.0, 90., 90., 90.],
    ] {
        let found = search(scrambled(Bravais::OrthorhombicP, cell, 0., rng), &options).unwrap();
        assert_eq!(
            found[0].bravais,
            Bravais::OrthorhombicP,
            "{:?}",
            symbols(&found)
        );
        assert_eq!(det_i(found[0].transform), 1);
        assert!(
            found
                .iter()
                .all(|m| det_i(m.transform) == m.bravais.centring().multiplicity())
        );
    }
    // Centring sets: a doubled cell has the coset (0 0 ½), not a centring.
    assert_eq!(centring([[2, 0, 0], [0, 1, 0], [0, 0, 1]]), None);
    assert_eq!(
        centring([[1, 1, 0], [-1, 1, 0], [0, 0, 1]]),
        Some(Centring::C)
    );
    assert_eq!(centring([[1, 0, 0], [0, 1, 0], [0, 0, 2]]), None);
    assert_eq!(
        centring([[1, 0, 0], [0, 1, 0], [1, 1, 2]]),
        Some(Centring::I)
    );
    assert_eq!(
        centring([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        Some(Centring::I)
    );
    assert_eq!(
        centring([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
        Some(Centring::F)
    );
    assert_eq!(
        centring([[1, -1, 0], [0, 1, -1], [1, 1, 1]]),
        Some(Centring::R)
    );
    // Reverse setting of the same rhombohedron.
    assert_eq!(centring([[-1, 1, 0], [0, -1, 1], [1, 1, 1]]), None);
}

#[test]
fn orthoenstatite_reduced_cell_gives_orthorhombic_primitive() {
    let rng = &mut Lcg(53);
    // MgSiO3, Pbca, a 18.2269, b 8.8152, c 5.1800 Å: from the reduced
    // primitive cell, as the grain finder reports it, with 0.05 % strain.
    let cell = [18.2269, 8.8152, 5.18, 90., 90., 90.];
    for _ in 0..5 {
        let raw = scrambled(Bravais::OrthorhombicP, cell, 5e-4, rng);
        let reduced = crate::grains::reduce(raw);
        let found = search(reduced, &SearchOptions::default()).unwrap();
        assert_eq!(symbols(&found), ["oP", "mP", "aP"]);
        let c = found[0].constrained_cell;
        assert!(same_cell(
            c,
            [5.18, 8.8152, 18.2269, 90., 90., 90.],
            3e-3,
            0.
        ));
    }
}

#[test]
fn niggli_cell_is_reduced_and_shared_by_equivalent_bases() {
    let rng = &mut Lcg(59);
    let a = cell_basis([5.1, 6.2, 7.3, 78.3, 83.1, 71.4]).unwrap();
    let (n0, t0) = niggli(a).unwrap();
    let c0 = cell_parameters(n0);
    for _ in 0..20 {
        let u = unimodular(rng);
        let b = apply(u, a);
        let s = crate::spots::reduction::reduce(b, 1e-12, 10_000).unwrap();
        let (n, t) = niggli(s.basis).unwrap();
        assert!(same_cell(cell_parameters(n), c0, 1e-12, 1e-9));
        assert_eq!(det_i(t).abs(), 1);
        assert_eq!(det_i(t0).abs(), 1);
    }
    assert!(search([[0.; 3]; 3], &SearchOptions::default()).is_err());
    let bad = SearchOptions {
        angle_tolerance: 0.,
        ..SearchOptions::default()
    };
    assert!(search(a, &bad).is_err());
}

#[test]
fn symbols_and_metadata_are_consistent() {
    for b in Bravais::ALL {
        assert_eq!(Bravais::from_symbol(b.symbol()), Some(b));
    }
    assert_eq!(Bravais::from_symbol("xx"), None);
    let free: Vec<usize> = Bravais::ALL
        .iter()
        .map(|b| b.system().free_parameters())
        .collect();
    assert_eq!(free, [6, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1]);
    assert!(Centring::C.allows([1, 1, 0]) && !Centring::C.allows([1, 0, 0]));
    assert!(Centring::F.allows([1, 1, 1]) && !Centring::F.allows([1, 1, 0]));
    assert!(Centring::R.allows([1, 0, 1]) && !Centring::R.allows([1, 0, 0]));
}

// ---- Constrained refinement ----

/// Reciprocal nodes of a conventional cell within `qmax`, allowed by the
/// centring, with Gaussian position noise `sigma` per component.
fn positions(
    bravais: Bravais,
    direct: M3,
    qmax: f64,
    sigma: f64,
    rng: &mut Lcg,
) -> (Vec<V3>, Vec<[i64; 3]>) {
    let b = inverse_transpose_for_test(direct);
    let mut q = Vec::new();
    let mut hkl = Vec::new();
    for h in -12_i64..=12 {
        for k in -12_i64..=12 {
            for l in -12_i64..=12 {
                if [h, k, l] == [0; 3] || !bravais.centring().allows([h, k, l]) {
                    continue;
                }
                let p: V3 = std::array::from_fn(|j| {
                    h as f64 * b[0][j] + k as f64 * b[1][j] + l as f64 * b[2][j]
                });
                if dot(p, p).sqrt() > qmax {
                    continue;
                }
                q.push(p.map(|x| x + sigma * rng.gauss()));
                hkl.push([h, k, l]);
            }
        }
    }
    (q, hkl)
}
fn inverse_transpose_for_test(a: M3) -> M3 {
    let m = Matrix3::from_fn(|i, j| a[i][j])
        .try_inverse()
        .unwrap()
        .transpose();
    std::array::from_fn(|i| std::array::from_fn(|j| m[(i, j)]))
}
fn strained(a: M3, s: f64) -> M3 {
    let e = [
        [1. + s, 0.3 * s, -0.2 * s],
        [0.3 * s, 1. - 0.5 * s, 0.1 * s],
        [-0.2 * s, 0.1 * s, 1. + 0.7 * s],
    ];
    a.map(|v| std::array::from_fn(|j| (0..3).map(|k| v[k] * e[k][j]).sum()))
}

#[test]
fn constrained_fit_recovers_exact_cells_for_every_system() {
    let rng = &mut Lcg(61);
    for (bravais, cell, _) in cases() {
        let r = rotation(rng);
        let truth = rotate_rows(cell_basis(cell).unwrap(), &r);
        let (q, hkl) = positions(bravais, truth, 0.7, 0., rng);
        let fit = refine_cell(
            &q,
            &hkl,
            strained(truth, 0.01),
            bravais,
            &RefineOptions::default(),
        )
        .unwrap();
        assert_eq!(fit.status, Convergence::Converged);
        assert!(
            same_cell(fit.cell, cell, 1e-9, 1e-7),
            "{}: {:?}",
            bravais.symbol(),
            fit.cell
        );
        assert!(fit.rms < 1e-10, "{} {}", bravais.symbol(), fit.rms);
        assert_eq!(fit.parameters, bravais.system().free_parameters());
        assert_eq!(fit.used, q.len());
        assert!(
            fit.conventional
                .iter()
                .flatten()
                .zip(truth.iter().flatten())
                .all(|(x, y)| (x - y).abs() < 1e-8)
        );
        assert!((fit.volume - det(truth)).abs() < 1e-8 * det(truth));
        // Indexing the noise-free positions against the start reproduces hkl.
        let indexed = index(&q, strained(truth, 0.002), bravais, 0.02).unwrap();
        assert!(indexed.iter().zip(&hkl).all(|(i, h)| *i == Some(*h)));
    }
}

#[test]
fn uncertainties_match_the_scatter_over_repeated_noise() {
    let rng = &mut Lcg(67);
    for (bravais, cell) in [
        (Bravais::MonoclinicP, [5.3, 7.1, 8.4, 90., 104.2, 90.]),
        (Bravais::RhombohedralR, [5.0, 5.0, 14.0, 90., 90., 120.]),
        (
            Bravais::OrthorhombicP,
            [18.2269, 8.8152, 5.18, 90., 90., 90.],
        ),
    ] {
        let sigma = 0.002;
        let trials = 150;
        let free: Vec<usize> = match bravais.system() {
            LatticeSystem::Monoclinic => vec![0, 1, 2, 4],
            LatticeSystem::Orthorhombic => vec![0, 1, 2],
            _ => vec![0, 2],
        };
        let mut pulls = vec![Vec::new(); 7];
        for _ in 0..trials {
            let truth = rotate_rows(cell_basis(cell).unwrap(), &rotation(rng));
            let (q, hkl) = positions(bravais, truth, 0.6, sigma, rng);
            let fit = refine_cell(
                &q,
                &hkl,
                strained(truth, 0.005),
                bravais,
                &RefineOptions::default(),
            )
            .unwrap();
            assert_eq!(fit.status, Convergence::Converged);
            for (slot, &k) in free.iter().enumerate() {
                pulls[slot].push((fit.cell[k] - cell[k]) / fit.cell_su[k]);
            }
            let v = det(cell_basis(cell).unwrap());
            pulls[6].push((fit.volume - v) / fit.volume_su);
            // The scatter estimate: rms per position ≈ sqrt(3) sigma.
            assert!((fit.rms / (3f64.sqrt() * sigma) - 1.).abs() < 0.15);
        }
        for p in pulls.iter().filter(|p| !p.is_empty()) {
            let n = p.len() as f64;
            let mean = p.iter().sum::<f64>() / n;
            let sd = (p.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.)).sqrt();
            assert!(mean.abs() < 0.3, "{}: mean pull {mean}", bravais.symbol());
            assert!(
                (0.8..1.25).contains(&sd),
                "{}: pull sd {sd}",
                bravais.symbol()
            );
        }
    }
}

#[test]
fn too_high_a_symmetry_fits_visibly_worse() {
    let rng = &mut Lcg(71);
    // Monoclinic β = 91.5°, fitted as monoclinic and as orthorhombic.
    let cell = [5.3, 7.1, 8.4, 90., 91.5, 90.];
    let truth = rotate_rows(cell_basis(cell).unwrap(), &rotation(rng));
    let (q, hkl) = positions(Bravais::MonoclinicP, truth, 0.7, 0.002, rng);
    let right = refine_cell(
        &q,
        &hkl,
        truth,
        Bravais::MonoclinicP,
        &RefineOptions::default(),
    )
    .unwrap();
    let wrong = refine_cell(
        &q,
        &hkl,
        truth,
        Bravais::OrthorhombicP,
        &RefineOptions::default(),
    )
    .unwrap();
    assert!(
        wrong.rms > 1.5 * right.rms,
        "{} vs {}",
        wrong.rms,
        right.rms
    );
    assert!((right.cell[4] - 91.5).abs() < 4. * right.cell_su[4]);
    assert_eq!(wrong.cell[4], 90.);
    // Tetragonal a/b = 1.02 forced to tP.
    let cell = [4.4, 4.488, 6.7, 90., 90., 90.];
    let truth = rotate_rows(cell_basis(cell).unwrap(), &rotation(rng));
    let (q, hkl) = positions(Bravais::OrthorhombicP, truth, 0.8, 0.002, rng);
    let right = refine_cell(
        &q,
        &hkl,
        truth,
        Bravais::OrthorhombicP,
        &RefineOptions::default(),
    )
    .unwrap();
    let wrong = refine_cell(
        &q,
        &hkl,
        truth,
        Bravais::TetragonalP,
        &RefineOptions::default(),
    )
    .unwrap();
    assert!(
        wrong.rms > 1.5 * right.rms,
        "{} vs {}",
        wrong.rms,
        right.rms
    );
}

#[test]
fn search_then_refine_from_a_reduced_primitive_cell() {
    let rng = &mut Lcg(73);
    let cell = [18.2269, 8.8152, 5.18, 90., 90., 90.];
    let truth = rotate_rows(cell_basis(cell).unwrap(), &rotation(rng));
    let (q, _) = positions(Bravais::OrthorhombicP, truth, 0.6, 0.001, rng);
    let reduced = crate::grains::reduce(strained(apply(unimodular(rng), truth), 0.002));
    let top = search(reduced, &SearchOptions::default())
        .unwrap()
        .remove(0);
    assert_eq!(top.bravais, Bravais::OrthorhombicP);
    let indexed = index(&q, top.idealized, top.bravais, 0.03).unwrap();
    let (mut qs, mut hs) = (Vec::new(), Vec::new());
    for (p, h) in q.iter().zip(&indexed) {
        if let Some(h) = h {
            qs.push(*p);
            hs.push(*h);
        }
    }
    assert!(qs.len() * 10 > q.len() * 9);
    let fit = refine_cell(
        &qs,
        &hs,
        top.idealized,
        top.bravais,
        &RefineOptions::default(),
    )
    .unwrap();
    assert!(same_cell(
        fit.cell,
        [5.18, 8.8152, 18.2269, 90., 90., 90.],
        1e-3,
        0.
    ));
    for k in 0..3 {
        assert!(fit.cell_su[k] > 0. && fit.cell_su[k] < 2e-3 * fit.cell[k]);
    }
    assert!(fit.volume_su > 0.);
}

#[test]
fn cell_jacobian_matches_full_model_differences() {
    let rng = &mut Lcg(79);
    for (bravais, cell, _) in cases() {
        let system = bravais.system();
        let truth = cell_basis(cell).unwrap();
        let (q, hkl) = positions(bravais, truth, 0.5, 0.001, rng);
        let h: Vec<V3> = hkl.iter().map(|h| h.map(|x| x as f64)).collect();
        let p = refinement::free_for_test(system, constrain_cell(system, cell));
        let (analytic, numeric) =
            refinement::jacobian_for_test(system, &q, &h, &p, [0.3, -0.2, 0.5]).unwrap();
        let scale = analytic.abs().max();
        assert!(
            (analytic - numeric).abs().max() < 1e-7 * scale,
            "{}",
            bravais.symbol()
        );
    }
}

#[test]
fn refinement_inputs_are_checked() {
    let a = cell_basis([5., 6., 7., 90., 90., 90.]).unwrap();
    let options = RefineOptions::default();
    let q = vec![[0.2, 0., 0.]; 3];
    assert!(refine_cell(&q, &[[1, 0, 0]; 2], a, Bravais::OrthorhombicP, &options).is_err());
    assert!(refine_cell(&q, &[[0, 0, 0]; 3], a, Bravais::OrthorhombicP, &options).is_err());
    assert!(refine_cell(&q[..1], &[[1, 0, 0]], a, Bravais::OrthorhombicP, &options).is_err());
    // Rank deficient: all indices along one row.
    let q: Vec<V3> = (1..10).map(|k| [0.2 * f64::from(k), 0., 0.]).collect();
    let h: Vec<[i64; 3]> = (1..10).map(|k| [k, 0, 0]).collect();
    assert!(refine_cell(&q, &h, a, Bravais::OrthorhombicP, &options).is_err());
    let bad = RefineOptions {
        max_iterations: 0,
        ..options
    };
    assert!(refine_cell(&q, &h, a, Bravais::OrthorhombicP, &bad).is_err());
    assert!(index(&q, a, Bravais::OrthorhombicP, 0.).is_err());
    assert!(index(&q, [[0.; 3]; 3], Bravais::OrthorhombicP, 0.01).is_err());
}
