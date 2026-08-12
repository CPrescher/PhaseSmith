#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_core::{ConstantWavelengthInstrument, TofInstrument, TofInstrumentParameter};
use phasesmith_crystallography::UnitCell;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{PatternRecord, RecordId, TofPatternRecord};
use phasesmith_workflows::{
    BackgroundModel, ChebyshevBackground, Constraint, FixedConstraint, LatticeBounds,
    LatticeParameterization, LatticeReflectionDomain, LeBailInput, LeBailOptions, LeBailPhase,
    TofBankInstrumentModel, TofInstrumentParameterBound, TofLeBailBank, TofLeBailInput,
    TofLeBailOptions, TofLeBailPhase, TofMultiBankGeometryInput, TofMultiBankGeometryOptions,
    TofMultiBankInput, TofMultiBankInstrumentInput, TofMultiBankInstrumentOptions,
    TofMultiBankLatticeInput, TofSharedLatticePhase, build_lebail_parameter_set_with_lattice,
    calculate_lebail_pattern, calculate_lebail_pattern_with_background,
    calculate_tof_lebail_pattern, lebail_lattice_parameter_key, refine_lebail,
    refine_tof_multibank_geometry, refine_tof_multibank_instrument, tof_lattice_geometry,
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

fn benchmark_tof_instrument_lebail(criterion: &mut Criterion) {
    let execution = ExecutionPolicy::new(Some(1), 2).unwrap();
    let d_spacings = linspace(0.65, 2.9, 80);
    let intensities = (0..d_spacings.len())
        .map(|index| 20.0 + f64::from(u32::try_from(index % 23).unwrap()))
        .collect::<Vec<_>>();
    let grid = linspace(3_000.0, 16_000.0, 4_001);
    let truth_instruments = [tof_instrument(5_000.0, -0.7), tof_instrument(4_600.0, 1.2)];
    let starting_instruments = [tof_instrument(5_000.0, 0.8), tof_instrument(4_600.0, -0.4)];
    let calculation_options =
        TofLeBailOptions::new(1, 1.0, 1.0e-12, 1.0e-15, 20.0, 20.0, true, execution).unwrap();
    let mut banks = Vec::new();
    let mut models = Vec::new();
    for index in 0..2 {
        let bank_id = RecordId::new(format!("bank-{}", index + 1)).unwrap();
        let blank = TofPatternRecord::new(
            grid.clone(),
            Some(vec![0.0; grid.len()]),
            Some(vec![1.0; grid.len()]),
            None,
            None,
        )
        .unwrap();
        let truth = TofLeBailInput::new(
            blank,
            truth_instruments[index],
            vec![tof_phase(&d_spacings, &intensities)],
        )
        .unwrap();
        let observed = calculate_tof_lebail_pattern(&truth, &calculation_options)
            .unwrap()
            .y;
        let pattern = TofPatternRecord::new(
            grid.clone(),
            Some(observed),
            Some(vec![1.0; grid.len()]),
            None,
            None,
        )
        .unwrap();
        banks.push(TofLeBailBank {
            bank_id: bank_id.clone(),
            input: TofLeBailInput::new(
                pattern,
                starting_instruments[index],
                vec![tof_phase(&d_spacings, &vec![1.0; d_spacings.len()])],
            )
            .unwrap(),
        });
        models.push(
            TofBankInstrumentModel::new(
                bank_id,
                vec![
                    TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0)
                        .unwrap(),
                ],
            )
            .unwrap(),
        );
    }
    let input = TofMultiBankInstrumentInput {
        multibank: TofMultiBankInput { banks },
        instrument_models: models,
    };
    let options =
        TofMultiBankInstrumentOptions::new(calculation_options, 1.0e-10, 0.2, 8, 0.999_999)
            .unwrap();
    criterion.bench_function(
        "tof_multibank_instrument_one_iteration_2x80_reflections_4001_samples",
        |bencher| {
            bencher.iter(|| {
                refine_tof_multibank_instrument(black_box(&input), black_box(&options)).unwrap()
            });
        },
    );
}

fn benchmark_tof_joint_geometry(criterion: &mut Criterion) {
    let execution = ExecutionPolicy::new(Some(1), 2).unwrap();
    let initial_cell = cubic_cell(3.995);
    let truth_cell = cubic_cell(4.0);
    let group = space_group_by_number(221).unwrap().space_group;
    let parameterization = LatticeParameterization::new(group, initial_cell).unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.02, 1.0).unwrap();
    let hkl = (1..=5)
        .flat_map(|h| (0..=5).flat_map(move |k| (0..=5).map(move |l| [h, k, l])))
        .take(80)
        .collect::<Vec<_>>();
    let intensities = (0..hkl.len())
        .map(|index| 20.0 + f64::from(u32::try_from(index % 23).unwrap()))
        .collect::<Vec<_>>();
    let grid = linspace(2_000.0, 21_000.0, 4_001);
    let truth_instruments = [tof_instrument(5_000.0, -0.7), tof_instrument(4_600.0, 1.2)];
    let starting_instruments = [tof_instrument(5_000.0, 0.8), tof_instrument(4_600.0, -0.4)];
    let calculation_options =
        TofLeBailOptions::new(1, 1.0, 1.0e-12, 1.0e-15, 20.0, 20.0, true, execution).unwrap();
    let mut banks = Vec::new();
    let mut models = Vec::new();
    for index in 0..2 {
        let bank_id = RecordId::new(format!("joint-bank-{}", index + 1)).unwrap();
        let blank = TofPatternRecord::new(
            grid.clone(),
            Some(vec![0.0; grid.len()]),
            Some(vec![1.0; grid.len()]),
            None,
            None,
        )
        .unwrap();
        let truth_phase = tof_lattice_phase(
            &parameterization,
            truth_cell,
            truth_instruments[index],
            &hkl,
            &intensities,
        );
        let truth =
            TofLeBailInput::new(blank, truth_instruments[index], vec![truth_phase]).unwrap();
        let observed = calculate_tof_lebail_pattern(&truth, &calculation_options)
            .unwrap()
            .y;
        let pattern = TofPatternRecord::new(
            grid.clone(),
            Some(observed),
            Some(vec![1.0; grid.len()]),
            None,
            None,
        )
        .unwrap();
        banks.push(TofLeBailBank {
            bank_id: bank_id.clone(),
            input: TofLeBailInput::new(
                pattern,
                starting_instruments[index],
                vec![tof_lattice_phase(
                    &parameterization,
                    initial_cell,
                    starting_instruments[index],
                    &hkl,
                    &vec![1.0; hkl.len()],
                )],
            )
            .unwrap(),
        });
        models.push(
            TofBankInstrumentModel::new(
                bank_id,
                vec![
                    TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0)
                        .unwrap(),
                ],
            )
            .unwrap(),
        );
    }
    let lattice_phase = TofSharedLatticePhase::new(
        RecordId::new("alpha").unwrap(),
        parameterization,
        bounds,
        initial_cell,
    )
    .unwrap();
    let input = TofMultiBankGeometryInput {
        lattice: TofMultiBankLatticeInput {
            multibank: TofMultiBankInput { banks },
            lattice_phases: vec![lattice_phase],
        },
        instrument_models: models,
    };
    let options =
        TofMultiBankGeometryOptions::new(calculation_options, 1.0e-10, 0.1, 8, 0.999_999).unwrap();
    criterion.bench_function(
        "tof_multibank_joint_geometry_one_iteration_2x80_reflections_4001_samples",
        |bencher| {
            bencher.iter(|| {
                refine_tof_multibank_geometry(black_box(&input), black_box(&options)).unwrap()
            });
        },
    );
}

fn cubic_cell(a_angstrom: f64) -> UnitCell {
    UnitCell {
        a_angstrom,
        b_angstrom: a_angstrom,
        c_angstrom: a_angstrom,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

fn tof_lattice_phase(
    parameterization: &LatticeParameterization,
    cell: UnitCell,
    instrument: TofInstrument,
    hkl: &[[i32; 3]],
    intensities: &[f64],
) -> TofLeBailPhase {
    let geometry = tof_lattice_geometry(parameterization, cell, hkl, instrument).unwrap();
    TofLeBailPhase::new(
        RecordId::new("alpha").unwrap(),
        "joint benchmark phase",
        (0..hkl.len())
            .map(|index| format!("reflection-{index}"))
            .collect(),
        hkl.to_vec(),
        geometry.d_spacing_angstrom,
        intensities.to_vec(),
        1.0,
    )
    .unwrap()
}

fn tof_instrument(difc: f64, zero: f64) -> TofInstrument {
    TofInstrument {
        zero_us: zero,
        difc_us_per_angstrom: difc,
        difa_us_per_angstrom2: -0.2,
        difb_us_angstrom: 0.3,
        alpha_coefficient: 0.18,
        beta0_per_us: 0.04,
        beta1_angstrom4_per_us: 0.000_5,
        betaq_angstrom2_per_us: 0.001,
        sigma0_us2: 1.0,
        sigma1_us2_per_angstrom2: 10.0,
        sigma2_us2_per_angstrom4: 0.05,
        sigmaq_us2_per_angstrom: 0.2,
        x_us_per_angstrom: 0.3,
        y_us_per_angstrom2: 0.05,
        z_us: 0.4,
    }
}

fn tof_phase(d_spacings: &[f64], intensities: &[f64]) -> TofLeBailPhase {
    TofLeBailPhase::new(
        RecordId::new("alpha").unwrap(),
        "benchmark TOF phase",
        (0..d_spacings.len())
            .map(|index| format!("reflection-{index}"))
            .collect(),
        (0..d_spacings.len())
            .map(|index| [i32::try_from(index + 1).unwrap(), 1, 0])
            .collect(),
        d_spacings.to_vec(),
        intensities.to_vec(),
        1.0,
    )
    .unwrap()
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

criterion_group!(
    benches,
    benchmark_fixed_lebail,
    benchmark_lattice_lebail,
    benchmark_tof_instrument_lebail,
    benchmark_tof_joint_geometry
);
criterion_main!(benches);
