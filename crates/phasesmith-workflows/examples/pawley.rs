//! Run a complete CW Pawley refinement without Python or a structure-factor model.
use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    ConstraintTransform, PawleyInput, PawleyOptions, PawleyPhase, evaluate_pawley,
    pawley_parameters, refine_pawley,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 0.0,
        v_deg2: 0.0,
        w_deg2: 0.001,
        x_deg: 0.002,
        y_deg: 0.0,
    };
    let phases = vec![PawleyPhase {
        id: "sample".into(),
        reflection_ids: vec!["100".into(), "110".into()],
        two_theta_deg: vec![40.0, 40.06],
        intensities: vec![3.0, 7.0],
        hkl: vec![],
        lattice: None,
    }];
    let x: Vec<f64> = (0..1001).map(|i| 39.0 + f64::from(i) * 0.002).collect();
    let parameters = pawley_parameters(&phases, instrument, None, false)?;
    let mut input = PawleyInput {
        fixed_spectrum: None,
        pattern: PatternRecord::new(x, Some(vec![0.0; 1001]), None, None, None)?,
        instrument,
        axial: None,
        phases,
        background: None,
        signed_intensities: false,
        parameters,
        constraints: vec![],
    };
    let transform = ConstraintTransform::new(input.parameters.clone(), Vec::new())?;
    let truth = evaluate_pawley(&input, &transform.pack()?, 20.0, true, 50_000_000)?;
    input.pattern.observed_y = Some(truth.calculated_y);
    input.phases[0].intensities.fill(0.0);
    input.parameters = pawley_parameters(&input.phases, instrument, None, false)?;
    let fit = refine_pawley(&input, &PawleyOptions::default())?;
    println!(
        "{}: areas={:?}, Rwp={:.8}, rank={:?}",
        fit.termination_reason.as_str(),
        fit.evaluation.intensities,
        fit.evaluation.residuals.rwp,
        fit.rank
    );
    Ok(())
}
