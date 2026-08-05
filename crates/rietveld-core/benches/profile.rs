#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rietveld_core::{
    GridView, PeakBatchView, SupportPolicy, accumulate_batch, accumulate_values_batch,
    symmetric_pseudo_voigt,
};

fn scalar_profile(criterion: &mut Criterion) {
    criterion.bench_function("symmetric_pseudo_voigt/value_and_derivatives", |bencher| {
        bencher.iter(|| symmetric_pseudo_voigt(black_box(0.17), black_box(0.08), black_box(0.43)));
    });
}

fn fused_accumulator(criterion: &mut Criterion) {
    let x: Vec<f64> = (0..5_001)
        .map(|index| 10.0 + f64::from(index) * 0.02)
        .collect();
    let positions: Vec<f64> = (0..200)
        .map(|index| 10.1 + f64::from(index) * 0.49)
        .collect();
    let intensities: Vec<f64> = (0..200)
        .map(|index| 100.0 + f64::from(index % 31))
        .collect();
    let fwhms: Vec<f64> = (0..200)
        .map(|index| 0.03 + f64::from(index % 7) * 0.002)
        .collect();
    let etas: Vec<f64> = (0..200)
        .map(|index| 0.2 + f64::from(index % 5) * 0.1)
        .collect();
    let grid = GridView::new(&x).expect("benchmark grid");
    let peaks =
        PeakBatchView::new(&positions, &intensities, &fwhms, &etas).expect("benchmark peaks");
    let support = SupportPolicy::FwhmMultiple(20.0);

    let mut group = criterion.benchmark_group("fused_accumulator");
    group.throughput(Throughput::Elements(positions.len() as u64));
    group.bench_function(BenchmarkId::new("values_only", x.len()), |bencher| {
        bencher.iter(|| {
            accumulate_values_batch(black_box(grid), black_box(peaks), black_box(support))
                .expect("valid benchmark")
        });
    });
    group.bench_function(BenchmarkId::new("support_jacobian", x.len()), |bencher| {
        bencher.iter(|| {
            accumulate_batch(black_box(grid), black_box(peaks), black_box(support))
                .expect("valid benchmark")
        });
    });
    group.bench_function(BenchmarkId::new("dense_jacobian", x.len()), |bencher| {
        bencher.iter(|| {
            let accumulation =
                accumulate_batch(black_box(grid), black_box(peaks), black_box(support))
                    .expect("valid benchmark");
            accumulation
                .dense_local_jacobian()
                .expect("valid dense compatibility allocation")
        });
    });
    group.finish();
}

criterion_group!(benches, scalar_profile, fused_accumulator);
criterion_main!(benches);
