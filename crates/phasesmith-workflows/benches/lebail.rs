#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    LeBailInput, LeBailOptions, LeBailPhase, calculate_lebail_pattern, refine_lebail,
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
    let starting = phase("alpha", &positions, &vec![1.0; 85], instrument);
    let input = LeBailInput::new(pattern, instrument, vec![starting]).unwrap();
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
    criterion.bench_function("fixed_lebail_85_reflections_3001_samples", |bencher| {
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

criterion_group!(benches, benchmark_fixed_lebail);
criterion_main!(benches);
