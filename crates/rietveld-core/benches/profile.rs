#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rietveld_core::{
    ConstantWavelengthInstrument, CwContributionArrays, CwContributionsView, CwReflectionBatchView,
    FcjGeometry, GridView, PeakBatchView, SupportPolicy, TchPeakBatchView, TchShape, TchWidths,
    TofInstrument, WavelengthComponentsView, accumulate_batch, accumulate_cw_batch,
    accumulate_cw_contributions_batch, accumulate_cw_fcj_batch, accumulate_cw_fcj_components_batch,
    accumulate_tch_batch, accumulate_tof_batch, accumulate_values_batch, symmetric_pseudo_voigt,
};

struct OwnedContributions {
    gaussian_variance: Vec<f64>,
    lorentzian: Vec<f64>,
    multiplier: Vec<f64>,
    d_gaussian_position: Vec<f64>,
    d_lorentzian_position: Vec<f64>,
    d_multiplier_position: Vec<f64>,
    d_gaussian_parameters: Vec<f64>,
    d_lorentzian_parameters: Vec<f64>,
    d_multiplier_parameters: Vec<f64>,
}

impl OwnedContributions {
    fn arrays(&self) -> CwContributionArrays<'_> {
        CwContributionArrays {
            gaussian_variance_deg2: &self.gaussian_variance,
            lorentzian_fwhm_deg: &self.lorentzian,
            intensity_multiplier: &self.multiplier,
            d_gaussian_variance_d_position: &self.d_gaussian_position,
            d_lorentzian_fwhm_d_position: &self.d_lorentzian_position,
            d_intensity_multiplier_d_position: &self.d_multiplier_position,
            d_gaussian_variance_d_parameters: &self.d_gaussian_parameters,
            d_lorentzian_fwhm_d_parameters: &self.d_lorentzian_parameters,
            d_intensity_multiplier_d_parameters: &self.d_multiplier_parameters,
        }
    }
}

fn sample_contributions(
    positions: &[f64],
    instrument: ConstantWavelengthInstrument,
) -> OwnedContributions {
    let theta: Vec<f64> = positions
        .iter()
        .map(|position| position * std::f64::consts::PI / 360.0)
        .collect();
    let size_nm = 50.0;
    let strain = 5.0e-4;
    let size_scale =
        180.0 / std::f64::consts::PI * 0.9 * instrument.wavelength_angstrom / (10.0 * size_nm);
    let lorentzian: Vec<f64> = theta.iter().map(|angle| size_scale / angle.cos()).collect();
    let gaussian_variance: Vec<f64> = theta
        .iter()
        .map(|angle| (2.0 * 180.0 / std::f64::consts::PI * strain * angle.tan()).powi(2))
        .collect();
    let d_gaussian_position: Vec<f64> = theta
        .iter()
        .map(|angle| {
            let coefficient = (2.0 * 180.0 / std::f64::consts::PI).powi(2);
            2.0 * coefficient * strain.powi(2) * angle.tan() / angle.cos().powi(2)
                * std::f64::consts::PI
                / 360.0
        })
        .collect();
    let d_lorentzian_position: Vec<f64> = theta
        .iter()
        .zip(&lorentzian)
        .map(|(angle, width)| width * std::f64::consts::PI / 360.0 * angle.tan())
        .collect();
    let mut d_gaussian_parameters = vec![0.0; 2 * positions.len()];
    let mut d_lorentzian_parameters = vec![0.0; 2 * positions.len()];
    for reflection in 0..positions.len() {
        d_lorentzian_parameters[reflection] = -lorentzian[reflection] / size_nm;
        d_gaussian_parameters[positions.len() + reflection] =
            2.0 * gaussian_variance[reflection] / strain;
    }
    OwnedContributions {
        gaussian_variance,
        lorentzian,
        multiplier: vec![1.0; positions.len()],
        d_gaussian_position,
        d_lorentzian_position,
        d_multiplier_position: vec![0.0; positions.len()],
        d_gaussian_parameters,
        d_lorentzian_parameters,
        d_multiplier_parameters: vec![0.0; 2 * positions.len()],
    }
}

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
    let owned_contributions = sample_contributions(&positions, instrument);
    let contributions = CwContributionsView::new(positions.len(), 2, owned_contributions.arrays())
        .expect("benchmark contributions");
    group.bench_function(
        BenchmarkId::new("size_strain_local_and_global_jacobian", x.len()),
        |bencher| {
            bencher.iter(|| {
                accumulate_cw_contributions_batch(
                    black_box(grid),
                    black_box(&positions),
                    black_box(&intensities),
                    black_box(instrument),
                    black_box(contributions),
                    black_box(support),
                )
                .expect("valid sample benchmark")
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

fn tof_accumulator(criterion: &mut Criterion) {
    let x: Vec<f64> = (0..5_001)
        .map(|index| 2_000.0 + f64::from(index) * 3.6)
        .collect();
    let d_spacings: Vec<f64> = (0..200)
        .map(|index| 0.42 + f64::from(index) * 0.017)
        .collect();
    let intensities: Vec<f64> = (0..200)
        .map(|index| 100.0 + f64::from(index % 31))
        .collect();
    let instrument = TofInstrument {
        zero_us: -0.773_346_536_757,
        difc_us_per_angstrom: 5_084.827_630_65,
        difa_us_per_angstrom2: -2.630_417_748_6,
        difb_us_angstrom: 1.25,
        alpha_coefficient: 5.0,
        beta0_per_us: 0.028,
        beta1_angstrom4_per_us: 0.0012,
        betaq_angstrom2_per_us: 0.003,
        sigma0_us2: 1.5,
        sigma1_us2_per_angstrom2: 15.140_286_726_8,
        sigma2_us2_per_angstrom4: 0.08,
        sigmaq_us2_per_angstrom: 0.7,
        x_us_per_angstrom: 0.8,
        y_us_per_angstrom2: 0.15,
        z_us: 1.2,
    };
    let grid = GridView::new(&x).expect("benchmark grid");
    let mut group = criterion.benchmark_group("tof_accumulator");
    group.throughput(Throughput::Elements(d_spacings.len() as u64));
    for tail_log in [8.0, 20.0] {
        group.bench_function(
            BenchmarkId::new(
                format!("tail_log_{tail_log:.0}_order_192_local_and_global_jacobian"),
                x.len(),
            ),
            |bencher| {
                bencher.iter(|| {
                    accumulate_tof_batch(
                        black_box(grid),
                        black_box(&d_spacings),
                        black_box(&intensities),
                        black_box(instrument),
                        black_box(20.0),
                        black_box(tail_log),
                    )
                    .expect("valid TOF benchmark")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    scalar_profile,
    fused_accumulator,
    tch_accumulator,
    cw_accumulator,
    cw_fcj_accumulator,
    tof_accumulator
);
criterion_main!(benches);
