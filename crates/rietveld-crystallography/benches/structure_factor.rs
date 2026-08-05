#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rietveld_crystallography::{
    P1BatchView, P1ParameterLayout, UnitCell, calculate_p1_intensity_vjp, calculate_p1_jvp,
    calculate_p1_values,
};

fn index_i32(index: usize) -> i32 {
    i32::try_from(index).expect("benchmark index fits i32")
}

fn index_f64(index: usize) -> f64 {
    f64::from(u32::try_from(index).expect("benchmark index fits u32"))
}

fn p1_structure_factors(criterion: &mut Criterion) {
    let reflection_count = 5_000;
    let site_count = 64;
    let hkl: Vec<[i32; 3]> = (0..reflection_count)
        .map(|index| {
            let value = index_i32(index) + 1;
            [
                value % 17 - 8,
                (value / 17) % 19 - 9,
                (value / 323) % 23 + 1,
            ]
        })
        .collect();
    let xyz: Vec<[f64; 3]> = (0..site_count)
        .map(|index| {
            let value = index_f64(index);
            [
                (0.137 * value).fract(),
                (0.271 * value).fract(),
                (0.419 * value).fract(),
            ]
        })
        .collect();
    let occupancy: Vec<f64> = (0..site_count)
        .map(|index| 0.7 + 0.3 * index_f64(index % 4) / 3.0)
        .collect();
    let u_iso: Vec<f64> = (0..site_count)
        .map(|index| 0.003 + 0.001 * index_f64(index % 7))
        .collect();
    let scattering_real: Vec<f64> = (0..reflection_count * site_count)
        .map(|index| 2.0 + index_f64(index % site_count) * 0.03)
        .collect();
    let scattering_imag = vec![0.0; reflection_count * site_count];
    let batch = P1BatchView {
        hkl: &hkl,
        fractional_xyz: &xyz,
        occupancy: &occupancy,
        u_iso_angstrom2: &u_iso,
        scattering_real: &scattering_real,
        scattering_imag: &scattering_imag,
        scale: 1.0,
    };
    let cell = UnitCell {
        a_angstrom: 8.1,
        b_angstrom: 9.2,
        c_angstrom: 10.3,
        alpha_deg: 88.0,
        beta_deg: 91.0,
        gamma_deg: 94.0,
    };
    let layout = P1ParameterLayout { site_count };
    let tangent: Vec<f64> = (0..layout.parameter_count())
        .map(|index| index_f64(index + 1) * 1.0e-6)
        .collect();
    let weights: Vec<f64> = (0..reflection_count)
        .map(|index| 0.5 + index_f64(index % 11) * 0.01)
        .collect();

    let mut group = criterion.benchmark_group("p1_structure_factor");
    group.throughput(Throughput::Elements((reflection_count * site_count) as u64));
    group.bench_function(BenchmarkId::new("values", reflection_count), |bencher| {
        bencher.iter(|| {
            calculate_p1_values(black_box(cell), black_box(batch)).expect("benchmark values")
        });
    });
    group.bench_function(BenchmarkId::new("jvp", reflection_count), |bencher| {
        bencher.iter(|| {
            calculate_p1_jvp(black_box(cell), black_box(batch), black_box(&tangent))
                .expect("benchmark JVP")
        });
    });
    group.bench_function(BenchmarkId::new("vjp", reflection_count), |bencher| {
        bencher.iter(|| {
            calculate_p1_intensity_vjp(black_box(cell), black_box(batch), black_box(&weights))
                .expect("benchmark VJP")
        });
    });
    group.finish();
}

criterion_group!(benches, p1_structure_factors);
criterion_main!(benches);
