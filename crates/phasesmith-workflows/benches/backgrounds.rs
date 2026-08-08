#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use phasesmith_workflows::{
    AmorphousBackground, AmorphousPeak, BackgroundModel, ChebyshevBackground, CompositeBackground,
    DifferentiableBackground, PointBackground,
};

fn analytical_backgrounds(criterion: &mut Criterion) {
    let grid = linspace(10.0, 120.0, 20_001);
    let model = CompositeBackground::new(
        "realistic",
        vec![
            BackgroundModel::Chebyshev(
                ChebyshevBackground::new(
                    "chebyshev",
                    vec![100.0, -8.0, 3.0, -1.0, 0.4, -0.1],
                    [10.0, 120.0],
                )
                .unwrap(),
            ),
            BackgroundModel::Point(
                PointBackground::new(
                    "points",
                    vec![10.0, 35.0, 65.0, 95.0, 120.0],
                    vec![2.0, -1.0, 0.5, -0.3, 1.0],
                )
                .unwrap(),
            ),
            BackgroundModel::Amorphous(
                AmorphousBackground::new(
                    "glass",
                    vec![
                        AmorphousPeak::new(1500.0, 35.0, 18.0).unwrap(),
                        AmorphousPeak::new(900.0, 72.0, 25.0).unwrap(),
                    ],
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let mut group = criterion.benchmark_group("analytical_background");
    group.throughput(Throughput::Elements(grid.len() as u64));
    group.bench_function("calculate_20001_samples", |bencher| {
        bencher.iter(|| model.calculate(black_box(&grid)).unwrap());
    });
    group.bench_function("basis_20001_samples", |bencher| {
        bencher.iter(|| model.basis(black_box(&grid)).unwrap());
    });
    group.finish();
}

#[allow(clippy::cast_precision_loss)]
fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    let denominator = (count - 1) as f64;
    (0..count)
        .map(|index| start + (end - start) * index as f64 / denominator)
        .collect()
}

criterion_group!(benches, analytical_backgrounds);
criterion_main!(benches);
