//! Complete two-bank TOF Pawley extraction without Python or atomic coordinates.
use phasesmith_core::TofInstrument;
use phasesmith_model::TofPatternRecord;
use phasesmith_workflows::{
    ConstraintTransform, PawleyOptions, PawleySolver, TofPawleyBank, TofPawleyInput,
    TofPawleyPhase, evaluate_tof_pawley, refine_tof_pawley, tof_pawley_parameters,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instrument = TofInstrument::from_values([
        0.0, 5000.0, 0.0, 0.0, 0.18, 0.04, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.3, 0.0, 0.4,
    ])?;
    let mut banks = Vec::new();
    for index in 0..2 {
        let x = (0..1001)
            .map(|i| 7000.0 + f64::from(i) * 5.0)
            .collect::<Vec<_>>();
        banks.push(TofPawleyBank {
            id: format!("bank{index}"),
            pattern: TofPatternRecord::new(x, Some(vec![0.0; 1001]), None, None, None)?,
            instrument,
            phases: vec![TofPawleyPhase {
                id: "sample".into(),
                reflection_ids: vec!["family-a".into(), "family-b".into()],
                hkl: Vec::new(),
                d_spacing_angstrom: vec![1.7, 2.0],
                intensities: vec![30.0, 70.0],
            }],
            background: None,
            normalization: "Synthetic density per microsecond".into(),
        });
    }
    let mut input = TofPawleyInput::new(banks, Vec::new(), false, 20.0)?;
    let options = PawleyOptions {
        solver: PawleySolver::MatrixFree,
        ..PawleyOptions::default()
    };
    let transform = ConstraintTransform::new(input.parameters.clone(), Vec::new())?;
    let truth = evaluate_tof_pawley(&input, &transform.pack()?, &options)?;
    for (bank, values) in input
        .banks
        .iter_mut()
        .zip(truth.calculated_y.chunks_exact(1001))
    {
        bank.pattern.observed_y = Some(values.to_vec());
        bank.phases[0].intensities.fill(0.0);
    }
    input.parameters = tof_pawley_parameters(&input.banks, &input.shared_lattice, false)?;
    let result = refine_tof_pawley(&input, &options)?;
    assert_eq!(
        result.termination_reason,
        phasesmith_workflows::TerminationReason::Converged
    );
    for (actual, expected) in result
        .evaluation
        .intensities
        .iter()
        .zip([30.0, 70.0, 30.0, 70.0])
    {
        assert!((actual - expected).abs() < 1e-7);
    }
    println!(
        "{}; Rwp={:.8}; bank-local areas={:?}",
        result.termination_reason.as_str(),
        result.evaluation.residuals.rwp,
        result.evaluation.intensities
    );
    Ok(())
}
