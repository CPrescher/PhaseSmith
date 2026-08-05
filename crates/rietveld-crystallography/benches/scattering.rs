#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rietveld_crystallography::{PreparedNeutronScattering, PreparedXrayScattering};

fn index_f64(index: usize) -> f64 {
    f64::from(u32::try_from(index).expect("benchmark index fits u32"))
}

fn scattering_batches(criterion: &mut Criterion) {
    let reflection_count = 20_000;
    let species = ["Si", "O", "O", "Na", "Al", "O", "O", "Fe3+"];
    let neutron_species = ["Si", "O", "O", "Na", "Al", "O", "O", "Fe"];
    let s: Vec<f64> = (0..reflection_count)
        .map(|index| 5.9 * index_f64(index) / index_f64(reflection_count - 1))
        .collect();
    let xray = PreparedXrayScattering::new(species).expect("valid X-ray species");
    let neutron = PreparedNeutronScattering::new(neutron_species).expect("valid neutron species");

    let mut group = criterion.benchmark_group("prepared_scattering");
    group.throughput(Throughput::Elements(
        (reflection_count * species.len()) as u64,
    ));
    group.bench_function(
        BenchmarkId::new("xray_value_and_derivative", reflection_count),
        |bencher| {
            bencher.iter(|| xray.evaluate(black_box(&s)).expect("valid benchmark batch"));
        },
    );
    group.bench_function(
        BenchmarkId::new("neutron_value_and_derivative", reflection_count),
        |bencher| {
            bencher.iter(|| {
                neutron
                    .evaluate(black_box(&s))
                    .expect("valid benchmark batch")
            });
        },
    );
    group.finish();
}

criterion_group!(benches, scattering_batches);
criterion_main!(benches);
