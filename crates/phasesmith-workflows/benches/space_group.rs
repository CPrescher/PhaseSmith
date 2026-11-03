#![allow(missing_docs)]
use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_crystallography::UnitCell;
use phasesmith_single_crystal::solution::Observation;
use phasesmith_workflows::space_group_search::{AssessmentOptions, Hypothesis, assess};
fn benchmark(c: &mut Criterion) {
    let cell = UnitCell {
        a_angstrom: 7.,
        b_angstrom: 8.,
        c_angstrom: 9.,
        alpha_deg: 90.,
        beta_deg: 90.,
        gamma_deg: 90.,
    };
    let hypotheses = phasesmith_io::space_group_settings_for_cell(cell)
        .unwrap()
        .into_iter()
        .map(|h| Hypothesis {
            number: h.number,
            symbol: h.hm_symbol,
            hall: h.hall_symbol,
            group: h.space_group,
        })
        .collect::<Vec<_>>();
    let points = [
        [0.13, 0.27, 0.41],
        [0.71, 0.39, 0.11],
        [0.37, 0.63, 0.83],
        [0.21, 0.73, 0.49],
    ];
    let mut rows = Vec::new();
    for h in -10i32..=10 {
        for k in -10i32..=10 {
            for l in -10i32..=10 {
                if h * h + k * k + l * l > 100 || [h, k, l] == [0; 3] {
                    continue;
                }
                let mut re = 0.;
                let mut im = 0.;
                for (p, w) in points.iter().zip([8., 8., 14., 12.]) {
                    let phase = std::f64::consts::TAU
                        * (f64::from(h) * p[0] + f64::from(k) * p[1] + f64::from(l) * p[2]);
                    let (s, c) = phase.sin_cos();
                    re += w * c;
                    im += w * s;
                }
                let intensity = re * re + im * im;
                rows.push(Observation {
                    hkl: [h, k, l],
                    intensity,
                    sigma: 2. + 0.05 * intensity,
                });
            }
        }
    }
    c.bench_function(
        "space_group_assessment_345_settings_4168_rows_4_peak_field",
        |b| {
            b.iter(|| {
                assess(
                    std::hint::black_box(&rows),
                    cell,
                    &hypotheses,
                    &AssessmentOptions::default(),
                )
                .unwrap()
            });
        },
    );
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
