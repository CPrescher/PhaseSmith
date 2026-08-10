#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_crystallography::UnitCell;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    BackgroundModel, ChebyshevBackground, Constraint, FixedConstraint, LatticeBounds,
    LatticeParameterization, LatticeReflectionDomain, LeBailInput, LeBailOptions, LeBailPhase,
    build_lebail_parameter_set_with_lattice, calculate_lebail_pattern,
    calculate_lebail_pattern_with_background, lebail_lattice_parameter_key, refine_lebail,
};

fn benchmark_fixed_lebail(criterion: &mut Criterion) {
    let execution = ExecutionPolicy::new(Some(1), 2).unwrap();
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 2.0e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    };
    let x = linspace(20.0, 100.0, 3_001);
    let positions = linspace(22.0, 98.0, 85);
    let truth_intensities = (0..85)
        .map(|index| 2.0 + f64::from(u32::try_from(index % 11).unwrap()))
        .collect::<Vec<_>>();
    let truth = phase("alpha", &positions, &truth_intensities, instrument);
    let fixed_background = x
        .iter()
        .map(|position| 4.0 + 0.005 * (position - 60.0).powi(2))
        .collect::<Vec<_>>();
    let blank =
        PatternRecord::new(x.clone(), None, None, None, Some(fixed_background.clone())).unwrap();
    let truth_background = BackgroundModel::Chebyshev(
        ChebyshevBackground::new(
            "residual",
            vec![0.5, -0.2, 0.08, -0.03, 0.01, -0.004],
            [20.0, 100.0],
        )
        .unwrap(),
    );
    let observed = calculate_lebail_pattern_with_background(
        &blank,
        instrument,
        std::slice::from_ref(&truth),
        Some(&truth_background),
        20.0,
        &execution,
    )
    .unwrap()
    .y;
    let pattern =
        PatternRecord::new(x, Some(observed), None, None, Some(fixed_background)).unwrap();
    let starting = phase("alpha", &positions, &vec![1.0; 85], instrument);
    let starting_background = BackgroundModel::Chebyshev(
        ChebyshevBackground::new("residual", vec![0.0; 6], [20.0, 100.0]).unwrap(),
    );
    let input = LeBailInput::new(pattern, instrument, vec![starting])
        .unwrap()
        .with_refinable_background(starting_background)
        .unwrap();
    let options = LeBailOptions::new(
        8,
        2,
        1.0e-12,
        1.0e-14,
        1.0,
        1.0e-15,
        1.0e-12,
        true,
        1.0 - 1.0e-10,
        false,
        20.0,
        execution,
    )
    .unwrap();
    criterion.bench_function(
        "lebail_85_reflections_3001_samples_6_term_residual_background",
        |bencher| {
            bencher.iter(|| refine_lebail(black_box(&input), black_box(&options), None).unwrap());
        },
    );
}

fn benchmark_lattice_lebail(criterion: &mut Criterion) {
    let execution = ExecutionPolicy::new(Some(1), 2).unwrap();
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 2.0e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    };
    let starting_cell = tetragonal_cell(3.995, 6.0);
    let group = space_group_by_number(123).unwrap().space_group;
    let parameterization = LatticeParameterization::new(group, starting_cell).unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.04, 5.0).unwrap();
    let domain = LatticeReflectionDomain::new(
        parameterization,
        bounds,
        instrument.wavelength_angstrom,
        [20.0, 90.0],
        1.0,
        true,
        50_000_000,
        1.001,
    )
    .unwrap();
    let starting =
        LeBailPhase::from_lattice_domain("alpha", "benchmark phase", starting_cell, 1.0, domain)
            .unwrap();
    let truth = starting
        .regenerate_lattice_at_cell(tetragonal_cell(4.0, 6.0))
        .unwrap();
    let x = linspace(20.0, 90.0, 3_001);
    let blank = PatternRecord::new(x.clone(), None, None, None, None).unwrap();
    let observed = calculate_lebail_pattern(
        &blank,
        instrument,
        std::slice::from_ref(&truth),
        20.0,
        &execution,
    )
    .unwrap()
    .y;
    let pattern = PatternRecord::new(x, Some(observed), None, None, None).unwrap();
    let parameters = build_lebail_parameter_set_with_lattice(
        instrument,
        std::slice::from_ref(&starting),
        &[],
        false,
        false,
        true,
    )
    .unwrap();
    let constraints = vec![Constraint::Fixed(
        FixedConstraint::new(
            lebail_lattice_parameter_key("alpha", "c_angstrom").unwrap(),
            6.0,
        )
        .unwrap(),
    )];
    let input = LeBailInput::new_with_parameters(
        pattern,
        instrument,
        vec![starting],
        parameters,
        constraints,
    )
    .unwrap();
    let options = LeBailOptions::new(
        1,
        1,
        1.0e-12,
        1.0e-14,
        1.0,
        1.0e-15,
        1.0e-12,
        true,
        1.0 - 1.0e-10,
        false,
        20.0,
        execution,
    )
    .unwrap()
    .with_profile_controls(1.0e-10, 0.05, 8)
    .unwrap();
    criterion.bench_function("lattice_lebail_one_iteration_3001_samples", |bencher| {
        bencher.iter(|| refine_lebail(black_box(&input), black_box(&options), None).unwrap());
    });
}

fn phase(
    phase_id: &str,
    positions: &[f64],
    intensities: &[f64],
    instrument: ConstantWavelengthInstrument,
) -> LeBailPhase {
    LeBailPhase::new(
        phase_id,
        "benchmark phase",
        (0..positions.len())
            .map(|index| format!("reflection-{index}"))
            .collect(),
        (0..positions.len())
            .map(|index| [i32::try_from(index + 1).unwrap(), 1, 0])
            .collect(),
        positions
            .iter()
            .map(|position| {
                instrument.wavelength_angstrom / (2.0 * (0.5 * position.to_radians()).sin())
            })
            .collect(),
        positions.to_vec(),
        intensities.to_vec(),
        1.0,
        Vec::new(),
    )
    .unwrap()
}

fn tetragonal_cell(a_angstrom: f64, c_angstrom: f64) -> UnitCell {
    UnitCell {
        a_angstrom,
        b_angstrom: a_angstrom,
        c_angstrom,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

#[allow(clippy::cast_precision_loss)]
fn linspace(start: f64, endpoint: f64, count: usize) -> Vec<f64> {
    let spacing = (endpoint - start) / (count - 1) as f64;
    (0..count)
        .map(|index| {
            if index + 1 == count {
                endpoint
            } else {
                start + index as f64 * spacing
            }
        })
        .collect()
}

criterion_group!(benches, benchmark_fixed_lebail, benchmark_lattice_lebail);
criterion_main!(benches);
