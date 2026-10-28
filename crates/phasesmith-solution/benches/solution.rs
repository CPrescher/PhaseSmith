#![allow(missing_docs)]
use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_solution::{FlipOptions, charge_flip};

fn benchmark(c: &mut Criterion) {
    let shape = [32; 3];
    let mut hkl = Vec::new();
    for h in -8_i32..=8 {
        for k in -8_i32..=8 {
            for l in -8_i32..=8 {
                if h * h + k * k + l * l <= 64 && [h, k, l] != [0; 3] {
                    hkl.push([h, k, l]);
                }
            }
        }
    }
    let amplitudes: Vec<_> = hkl
        .iter()
        .map(|h| {
            // Four independent point atoms produce an actual multi-peak Fourier field.
            let xyz = [
                [0., 0., 0.],
                [0.17, 0.29, 0.41],
                [0.36, 0.62, 0.13],
                [0.73, 0.21, 0.84],
            ];
            let mut re = 0.;
            let mut im = 0.;
            for p in xyz {
                let phase = std::f64::consts::TAU
                    * (f64::from(h[0]) * p[0] + f64::from(h[1]) * p[1] + f64::from(h[2]) * p[2]);
                let (s, c) = phase.sin_cos();
                re += c;
                im += s;
            }
            re.hypot(im)
        })
        .collect();
    let phases: Vec<_> = hkl
        .iter()
        .map(|h| f64::from(h[0] + 3 * h[1] + 7 * h[2]) * 0.137)
        .collect();
    let mut support = vec![0; 32 * 32 * 32];
    support[0] = 1;
    for h in &hkl {
        let v = h.map(|x| usize::try_from(x.rem_euclid(32)).unwrap());
        support[(v[0] * 32 + v[1]) * 32 + v[2]] = 2;
    }
    c.bench_function(
        "charge_flip_32cubed_four_atoms_2108_reflections_50_cycles",
        |b| {
            b.iter(|| {
                charge_flip(
                    shape,
                    &hkl,
                    &amplitudes,
                    &phases,
                    &support,
                    &FlipOptions {
                        cycles: 50,
                        delta_sigma: 1.,
                        weak_fraction: 0.2,
                        cleanup_cycles: 0,
                        cancellation: None,
                    },
                    None,
                )
                .unwrap()
            });
        },
    );
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
