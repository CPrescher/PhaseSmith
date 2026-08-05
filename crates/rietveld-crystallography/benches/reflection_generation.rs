#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rietveld_crystallography::{
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

fn reflection_generation(criterion: &mut Criterion) {
    let generator =
        PreparedReflectionGenerator::new(c_centred_orthorhombic_group(), true, 5_000_000)
            .expect("benchmark generator");
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
    let reflection_count = generator
        .generate(cell, range)
        .expect("benchmark preflight")
        .len();
    criterion.bench_with_input(
        BenchmarkId::new("prepared_c_orthorhombic", reflection_count),
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

criterion_group!(benches, reflection_generation);
criterion_main!(benches);
