#![allow(missing_docs)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_core::{ConstantWavelengthInstrument, FcjGeometry, OwnedCwContributions};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{FixedWavelengthSpectrum, PatternRecord, RecordId};
use phasesmith_workflows::{
    BackgroundModel, PolynomialBackground, PreparedGeneralRietveldObjective,
    RietveldCalculationOptions, RietveldInput, RietveldInstrumentParameter,
    RietveldParameterLayout, RietveldParameterSelection, RietveldPhase,
    RietveldStructuralSelection,
};

#[allow(clippy::too_many_lines)]
fn fixed_doublet_fixture() -> (
    RietveldInput,
    RietveldCalculationOptions,
    RietveldParameterLayout,
    Vec<f64>,
) {
    let cell = UnitCell {
        a_angstrom: 12.0,
        b_angstrom: 12.0,
        c_angstrom: 12.0,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let wavelength = 1.5405;
    let mut hkl = Vec::new();
    for h in 0..=10 {
        for k in 0..=10 {
            for l in 0..=10 {
                let squared = h * h + k * k + l * l;
                if squared == 0 {
                    continue;
                }
                let sine = wavelength * f64::from(squared).sqrt() / (2.0 * cell.a_angstrom);
                if sine < 0.95 {
                    hkl.push([h, k, l]);
                }
                if hkl.len() == 256 {
                    break;
                }
            }
            if hkl.len() == 256 {
                break;
            }
        }
        if hkl.len() == 256 {
            break;
        }
    }
    assert_eq!(hkl.len(), 256);
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [0.13, 0.21, 0.34],
        [0.27, 0.42, 0.18],
        [0.39, 0.17, 0.46],
        [0.51, 0.32, 0.23],
        [0.62, 0.48, 0.37],
        [0.74, 0.29, 0.11],
        [0.86, 0.63, 0.55],
    ];
    let definition = StructuralPhaseDefinition {
        cell,
        space_group: space_group_by_number(1).unwrap().space_group,
        multiplicity: vec![1; hkl.len()],
        hkl,
        occupancy: vec![1.0; coordinates.len()],
        u_iso_angstrom2: vec![0.008; coordinates.len()],
        anisotropic_mask: vec![false; coordinates.len()],
        u_aniso_cif_angstrom2: vec![[0.0; 6]; coordinates.len()],
        scattering_species: (0..coordinates.len())
            .map(|index| if index % 2 == 0 { "Si" } else { "O" }.to_owned())
            .collect(),
        fractional_xyz: coordinates,
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom: wavelength,
            polarization: 0.7,
        },
    };
    let phase = RietveldPhase::new(
        RecordId::new("benchmark").unwrap(),
        "fixed-doublet benchmark",
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap();
    let x_deg = (0..10_001)
        .map(|index| 10.0 + f64::from(index) * 0.014)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![1.0; x_deg.len()]),
        Some(vec![1.0; x_deg.len()]),
        None,
        Some(vec![0.0; x_deg.len()]),
    )
    .unwrap();
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: wavelength,
        u_deg2: 4.0e-3,
        v_deg2: -2.0e-3,
        w_deg2: 2.5e-3,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    };
    let input = RietveldInput::new_fixed_spectrum_with_background(
        pattern,
        instrument,
        FixedWavelengthSpectrum::new(vec![wavelength, 1.5443], vec![1.0, 0.5]).unwrap(),
        Some(FcjGeometry {
            sample_over_radius: 0.0075,
            detector_over_radius: 0.0075,
        }),
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        BackgroundModel::Polynomial(
            PolynomialBackground::new("benchmark", vec![0.0, 0.0, 0.0]).unwrap(),
        ),
        vec![phase],
    )
    .unwrap();
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            coordinates: true,
            u_iso: true,
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        vec![
            RietveldInstrumentParameter::UDeg2,
            RietveldInstrumentParameter::VDeg2,
            RietveldInstrumentParameter::WDeg2,
            RietveldInstrumentParameter::ZeroShiftDeg,
        ],
        true,
        false,
    )
    .unwrap();
    let layout = RietveldParameterLayout::new(&input, &selection, &[None]).unwrap();
    let direction = vec![0.01; layout.parameters().specs().len()];
    let options =
        RietveldCalculationOptions::new(30.0, true, ExecutionPolicy::new(Some(2), 2).unwrap())
            .unwrap();
    (input, options, layout, direction)
}

fn benchmark_fixed_doublet_objective(criterion: &mut Criterion) {
    let (input, options, layout, direction) = fixed_doublet_fixture();
    let mut group = criterion.benchmark_group("fixed_doublet_256_reflections_10001_samples");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    group.bench_function("prepare_bounded_dense_linearization", |bencher| {
        bencher.iter(|| {
            PreparedGeneralRietveldObjective::new(
                black_box(input.clone()),
                black_box(options.clone()),
                black_box(layout.clone()),
            )
            .unwrap()
        });
    });
    let dense =
        PreparedGeneralRietveldObjective::new(input.clone(), options.clone(), layout.clone())
            .unwrap();
    group.bench_function("dense_normal_product", |bencher| {
        bencher.iter(|| {
            dense
                .normal_product(black_box(&direction), black_box(1.0e-6))
                .unwrap()
        });
    });
    let matrix_free = PreparedGeneralRietveldObjective::new_with_max_linearization_elements(
        input, options, layout, 0,
    )
    .unwrap();
    group.bench_function("matrix_free_normal_product", |bencher| {
        bencher.iter(|| {
            matrix_free
                .normal_product(black_box(&direction), black_box(1.0e-6))
                .unwrap()
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_fixed_doublet_objective);
criterion_main!(benches);
