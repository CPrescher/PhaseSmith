#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use phasesmith_crystallography::{
    PreparedReflectionGenerator, Rational, ReflectionRange, SpaceGroup, SymmetryOperation, UnitCell,
};

fn c_centred_orthorhombic_group() -> SpaceGroup {
    let rotations = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    ];
    let half = Rational::new(1, 2).expect("half");
    let translations = [[Rational::zero(); 3], [half, half, Rational::zero()]];
    let operations = rotations
        .into_iter()
        .flat_map(|rotation| {
            translations.map(|translation| {
                SymmetryOperation::new(rotation, translation).expect("benchmark operation")
            })
        })
        .collect();
    SpaceGroup::new(operations).expect("benchmark group")
}

fn face_centred_cubic_group() -> SpaceGroup {
    let zero = Rational::zero();
    let half = Rational::new(1, 2).expect("half");
    let translations = [
        [zero; 3],
        [zero, half, half],
        [half, zero, half],
        [half, half, zero],
    ];
    let mut operations = Vec::new();
    for axes in [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 2, 1],
        [2, 1, 0],
    ] {
        for signs in 0..8 {
            let mut rotation = [[0; 3]; 3];
            for (row, column) in axes.into_iter().enumerate() {
                rotation[row][column] = if signs & (1 << row) == 0 { 1 } else { -1 };
            }
            for translation in translations {
                operations
                    .push(SymmetryOperation::new(rotation, translation).expect("cubic operation"));
            }
        }
    }
    SpaceGroup::new(operations).expect("Fm-3m")
}

fn reflection_generation(criterion: &mut Criterion) {
    let cell = UnitCell {
        a_angstrom: 9.1,
        b_angstrom: 11.2,
        c_angstrom: 13.4,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let range = ReflectionRange::DSpacing {
        min_angstrom: 0.5,
        max_angstrom: 14.0,
    };
    let cubic_cell = UnitCell {
        b_angstrom: cell.a_angstrom,
        c_angstrom: cell.a_angstrom,
        ..cell
    };
    for (name, group, cell) in [
        (
            "prepared_c_orthorhombic",
            c_centred_orthorhombic_group(),
            cell,
        ),
        ("prepared_f_cubic", face_centred_cubic_group(), cubic_cell),
    ] {
        let generator =
            PreparedReflectionGenerator::new(group, true, 5_000_000).expect("benchmark generator");
        let reflection_count = generator
            .generate(cell, range)
            .expect("benchmark preflight")
            .len();
        criterion.bench_with_input(
            BenchmarkId::new(name, reflection_count),
            &reflection_count,
            |bencher, _| {
                bencher.iter(|| {
                    generator
                        .generate(black_box(cell), black_box(range))
                        .expect("benchmark reflection generation")
                });
            },
        );
    }
}

criterion_group!(benches, reflection_generation);
criterion_main!(benches);
