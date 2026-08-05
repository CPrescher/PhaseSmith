#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rietveld_core::{Peak, accumulate_peaks, symmetric_pseudo_voigt};

fn scalar_profile(criterion: &mut Criterion) {
    criterion.bench_function("symmetric_pseudo_voigt/value_and_derivatives", |bencher| {
        bencher.iter(|| symmetric_pseudo_voigt(black_box(0.17), black_box(0.08), black_box(0.43)));
    });
}

fn fused_accumulator(criterion: &mut Criterion) {
    let x: Vec<f64> = (0..5_001)
        .map(|index| 10.0 + f64::from(index) * 0.02)
        .collect();
    let peaks: Vec<Peak> = (0..200)
        .map(|index| Peak {
            position: 10.1 + f64::from(index) * 0.49,
            intensity: 100.0 + f64::from(index % 31),
            fwhm: 0.03 + f64::from(index % 7) * 0.002,
            eta: 0.2 + f64::from(index % 5) * 0.1,
        })
        .collect();

    let mut group = criterion.benchmark_group("fused_accumulator");
    group.throughput(Throughput::Elements(peaks.len() as u64));
    group.bench_with_input(BenchmarkId::new("grid", x.len()), &x, |bencher, grid| {
        bencher.iter(|| accumulate_peaks(black_box(grid), black_box(&peaks), black_box(20.0)));
    });
    group.finish();
}

criterion_group!(benches, scalar_profile, fused_accumulator);
criterion_main!(benches);
