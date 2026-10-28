#![allow(missing_docs)]
use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_crystallography::{SpaceGroup, SymmetryOperation, UnitCell};
use phasesmith_solution::{
    AdpMode, DualSpaceOptions, FlipOptions, PreparedSymmetricModel, charge_flip, dual_space,
    recover_symmetry,
};

#[allow(clippy::too_many_lines)]
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
    let cell = UnitCell {
        a_angstrom: 8.,
        b_angstrom: 9.,
        c_angstrom: 10.,
        alpha_deg: 90.,
        beta_deg: 90.,
        gamma_deg: 90.,
    };
    let points = [
        [0.03, 0.07, 0.11],
        [0.17, 0.29, 0.41],
        [0.36, 0.62, 0.13],
        [0.73, 0.21, 0.84],
        [0.25, 0.75, 0.31],
        [0.43, 0.14, 0.68],
        [0.82, 0.56, 0.27],
        [0.61, 0.87, 0.49],
    ];
    let elements = ["Si", "Si", "O", "O", "O", "O", "O", "O"].map(str::to_owned);
    let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
    let symmetric = recover_symmetry(cell, &group, &elements, &points, 0.01, None, None).unwrap();
    let prepared =
        PreparedSymmetricModel::new(symmetric, &hkl, &amplitudes, AdpMode::Anisotropic).unwrap();
    let parameters = prepared.initial_parameters(0.015).unwrap();
    c.bench_function(
        "symmetric_adp_evaluate_8_atoms_2108_reflections_69_parameters",
        |b| {
            b.iter(|| {
                prepared
                    .evaluate(std::hint::black_box(&parameters))
                    .unwrap()
            });
        },
    );
    c.bench_function(
        "dual_space_32cubed_four_atoms_2108_reflections_50_cycles",
        |b| {
            b.iter(|| {
                dual_space(
                    shape,
                    cell,
                    &hkl,
                    &amplitudes,
                    &phases,
                    &support,
                    &DualSpaceOptions {
                        cycles: 50,
                        atom_count: 4,
                        mask_sigma: 0.3,
                        hybrid_m: 3.,
                        omit_fraction: 0.1,
                        seed: 0,
                        patterson_rank: Some(0),
                        cancellation: None,
                    },
                    None,
                )
                .unwrap()
            });
        },
    );
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
