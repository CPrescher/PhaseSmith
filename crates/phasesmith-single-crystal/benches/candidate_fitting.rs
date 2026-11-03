#![allow(missing_docs)]
use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation, UnitCell};
use phasesmith_single_crystal::solution::{AdpMode, PreparedSymmetricModel, recover_symmetry};

#[allow(clippy::too_many_lines)] // One shared realistic fixture and paired full-grid workloads.
fn benchmark(c: &mut Criterion) {
    let cell = UnitCell {
        a_angstrom: 8.,
        b_angstrom: 9.,
        c_angstrom: 10.,
        alpha_deg: 90.,
        beta_deg: 90.,
        gamma_deg: 90.,
    };
    let group = SpaceGroup::new(vec![
        SymmetryOperation::identity(),
        SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
            .unwrap(),
    ])
    .unwrap();
    let mut points = vec![[0., 0., 0.], [0.5, 0.5, 0.5]];
    for p in [
        [0.13, 0.26, 0.37],
        [0.22, 0.17, 0.41],
        [0.32, 0.45, 0.18],
        [0.44, 0.31, 0.09],
    ] {
        points.push(p);
        points.push(p.map(|v| 1. - v));
    }
    let mut labels = vec!["Si".to_owned(), "Mg".to_owned()];
    labels.extend(vec!["O".to_owned(); 8]);
    let mut hkl = Vec::new();
    let mut observed = Vec::new();
    for h in -10i32..=10 {
        for k in -10..=10 {
            for l in -10..=10 {
                if [h, k, l] == [0; 3] || h * h + k * k + l * l > 100 {
                    continue;
                }
                let mut re = 0.;
                let mut im = 0.;
                for (p, w) in points
                    .iter()
                    .zip([14., 12., 8., 8., 8., 8., 8., 8., 8., 8.])
                {
                    let phase = std::f64::consts::TAU
                        * (f64::from(h) * p[0] + f64::from(k) * p[1] + f64::from(l) * p[2]);
                    let (s, c) = phase.sin_cos();
                    re += w * c;
                    im += w * s;
                }
                hkl.push([h, k, l]);
                observed.push(re.hypot(im));
            }
        }
    }
    let symmetry = recover_symmetry(cell, &group, &labels, &points, 0.01, None, None).unwrap();
    let prepare = || {
        PreparedSymmetricModel::new(symmetry.clone(), &hkl, &observed, AdpMode::Isotropic)
            .unwrap()
            .with_specialized_isotropic_evaluation(true)
    };
    let model = prepare();
    let start = model.initial_parameters(0.015).unwrap();
    c.bench_function("candidate_iso_10_atoms_4168_rows_evaluate", |b| {
        b.iter(|| model.evaluate(std::hint::black_box(&start)).unwrap());
    });
    c.bench_function("candidate_iso_10_atoms_4168_rows_prepare_fit8", |b| {
        b.iter(|| {
            let model = prepare();
            let start = model.initial_parameters(0.015).unwrap();
            model
                .refine_with_normal_reuse(&start, 8, None, None, true, true, true)
                .unwrap()
        });
    });
    for (name, stored_frame) in [
        (
            "candidate_common_10_atoms_4168_rows_arbitrary_origin",
            false,
        ),
        ("candidate_common_10_atoms_4168_rows_validated_frame", true),
    ] {
        c.bench_function(name, |b| {
            b.iter(|| {
                let model = if stored_frame {
                    phasesmith_single_crystal::solution::recover_symmetry_in_frame(
                        cell,
                        &group,
                        &labels,
                        &points,
                        0.01,
                        (1, [0.; 3]),
                        None,
                        None,
                    )
                } else {
                    recover_symmetry(cell, &group, &labels, &points, 0.01, None, None)
                }
                .unwrap();
                let prepared =
                    PreparedSymmetricModel::new(model, &hkl, &observed, AdpMode::Isotropic)
                        .unwrap();
                prepared
                    .evaluate(&prepared.initial_parameters(0.015).unwrap())
                    .unwrap()
            });
        });
    }
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
