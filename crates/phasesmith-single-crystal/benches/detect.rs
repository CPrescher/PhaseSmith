#![allow(
    missing_docs,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)] // Small synthetic coordinates.
//! Spot detection on a realistic detector frame (Eiger2 4M shape): integer
//! Poisson-like background, 150 spots of varied brightness and width, 3 %
//! masked pixels. `detect` runs the local pass two (exact integer sums);
//! `detect_two_pass` rebuilds the whole frame for pass two.
use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_single_crystal::spots::{Image, Options, detect, detect_two_pass};

fn frame(width: usize, height: usize, spots: usize) -> (Vec<f64>, Vec<bool>) {
    let mut state = 8_675_309_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut values: Vec<f64> = (0..width * height)
        .map(|_| (3.0 + 4.0 * next() + 3.0 * next()).floor())
        .collect();
    for _ in 0..spots {
        let (cx, cy) = (next() * width as f64, next() * height as f64);
        let (peak, sigma) = (20.0 + 3000.0 * next() * next(), 0.6 + 2.0 * next());
        let r = (4.0 * sigma).ceil() as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (cx as isize + dx, cy as isize + dy);
                if x >= 0 && y >= 0 && x < width as isize && y < height as isize {
                    let d2 = (dx * dx + dy * dy) as f64;
                    values[y as usize * width + x as usize] +=
                        (peak * (-d2 / (2.0 * sigma * sigma)).exp()).floor();
                }
            }
        }
    }
    let mask = (0..width * height).map(|_| next() < 0.03).collect();
    (values, mask)
}

fn benchmark(c: &mut Criterion) {
    let (width, height) = (2068, 2162);
    let (values, mask) = frame(width, height, 150);
    let image = Image {
        values: &values,
        height,
        width,
        mask: &mask,
        variance: None,
        gain: 1.0,
        read_noise: 0.0,
        saturation: None,
    };
    let options = Options::default();
    assert_eq!(
        detect(&image, options).unwrap().spots.len(),
        detect_two_pass(&image, options).unwrap().spots.len()
    );
    let mut group = c.benchmark_group("spot_detection");
    group.sample_size(20);
    group.bench_function("local_pass_two", |b| {
        b.iter(|| detect(&image, options).unwrap());
    });
    group.bench_function("full_pass_two", |b| {
        b.iter(|| detect_two_pass(&image, options).unwrap());
    });
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
