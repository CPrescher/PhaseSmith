#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rietveld_core::{
    ConstantWavelengthInstrument, CwReflectionBatchView, FcjGeometry, GridView, PeakBatchView,
    SupportPolicy, TchPeakBatchView, TchShape, TchWidths, WavelengthComponentsView,
    accumulate_batch, accumulate_cw_batch, accumulate_cw_fcj_batch,
    accumulate_cw_fcj_components_batch, accumulate_tch_batch, accumulate_values_batch,
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

fn tch_accumulator(criterion: &mut Criterion) {
    criterion.bench_function("tch/width_transform", |bencher| {
        bencher.iter(|| {
            TchShape::from_component_fwhm(TchWidths {
                gaussian_fwhm: black_box(0.071),
                lorentzian_fwhm: black_box(0.023),
            })
            .expect("valid benchmark widths")
        });
    });

    let x: Vec<f64> = (0..5_001)
        .map(|index| 10.0 + f64::from(index) * 0.02)
        .collect();
    let positions: Vec<f64> = (0..200)
        .map(|index| 10.1 + f64::from(index) * 0.49)
        .collect();
    let intensities: Vec<f64> = (0..200)
        .map(|index| 100.0 + f64::from(index % 31))
        .collect();
    let gaussian_fwhms: Vec<f64> = (0..200)
        .map(|index| 0.02 + f64::from(index % 7) * 0.002)
        .collect();
    let lorentzian_fwhms: Vec<f64> = (0..200)
        .map(|index| 0.01 + f64::from(index % 5) * 0.002)
        .collect();
    let grid = GridView::new(&x).expect("benchmark grid");
    let peaks = TchPeakBatchView::new(&positions, &intensities, &gaussian_fwhms, &lorentzian_fwhms)
        .expect("benchmark TCH peaks");
    let support = SupportPolicy::FwhmMultiple(20.0);
    let mut group = criterion.benchmark_group("tch_accumulator");
    group.throughput(Throughput::Elements(positions.len() as u64));
    group.bench_function(BenchmarkId::new("support_jacobian", x.len()), |bencher| {
        bencher.iter(|| {
            accumulate_tch_batch(black_box(grid), black_box(peaks), black_box(support))
                .expect("valid benchmark")
        });
    });
    group.finish();
}

fn cw_accumulator(criterion: &mut Criterion) {
    let x: Vec<f64> = (0..5_001)
        .map(|index| 10.0 + f64::from(index) * 0.02)
        .collect();
    let positions: Vec<f64> = (0..200)
        .map(|index| 10.1 + f64::from(index) * 0.49)
        .collect();
    let intensities: Vec<f64> = (0..200)
        .map(|index| 100.0 + f64::from(index % 31))
        .collect();
    let grid = GridView::new(&x).expect("benchmark grid");
    let reflections =
        CwReflectionBatchView::new(&positions, &intensities).expect("benchmark reflections");
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 1.2e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    };
    let support = SupportPolicy::FwhmMultiple(20.0);
    let mut group = criterion.benchmark_group("cw_accumulator");
    group.throughput(Throughput::Elements(positions.len() as u64));
    group.bench_function(
        BenchmarkId::new("local_and_global_jacobian", x.len()),
        |bencher| {
            bencher.iter(|| {
                accumulate_cw_batch(
                    black_box(grid),
                    black_box(reflections),
                    black_box(instrument),
                    black_box(support),
                )
                .expect("valid benchmark")
            });
        },
    );
    group.finish();
}

fn cw_fcj_accumulator(criterion: &mut Criterion) {
    let x: Vec<f64> = (0..5_001)
        .map(|index| 10.0 + f64::from(index) * 0.02)
        .collect();
    let positions: Vec<f64> = (0..200)
        .map(|index| 10.1 + f64::from(index) * 0.49)
        .collect();
    let intensities: Vec<f64> = (0..200)
        .map(|index| 100.0 + f64::from(index % 31))
        .collect();
    let grid = GridView::new(&x).expect("benchmark grid");
    let reflections =
        CwReflectionBatchView::new(&positions, &intensities).expect("benchmark reflections");
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 1.2e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    };
    let geometry = FcjGeometry {
        sample_over_radius: 0.012,
        detector_over_radius: 0.012,
    };
    let support = SupportPolicy::FwhmMultiple(20.0);
    let wavelengths = [instrument.wavelength_angstrom, 1.544_43];
    let relative_intensities = [1.0, 0.5];
    let components = WavelengthComponentsView::new(&wavelengths, &relative_intensities)
        .expect("benchmark components");
    let mut group = criterion.benchmark_group("cw_fcj_accumulator");
    group.throughput(Throughput::Elements(positions.len() as u64));
    group.bench_function(
        BenchmarkId::new("order_48_local_and_global_jacobian", x.len()),
        |bencher| {
            bencher.iter(|| {
                accumulate_cw_fcj_batch(
                    black_box(grid),
                    black_box(reflections),
                    black_box(instrument),
                    black_box(geometry),
                    black_box(support),
                )
                .expect("valid benchmark")
            });
        },
    );
    group.bench_function(
        BenchmarkId::new("order_48_doublet_local_and_global_jacobian", x.len()),
        |bencher| {
            bencher.iter(|| {
                accumulate_cw_fcj_components_batch(
                    black_box(grid),
                    black_box(reflections),
                    black_box(instrument),
                    black_box(components),
                    black_box(geometry),
                    black_box(support),
                )
                .expect("valid doublet benchmark")
            });
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    scalar_profile,
    fused_accumulator,
    tch_accumulator,
    cw_accumulator,
    cw_fcj_accumulator
);
criterion_main!(benches);
