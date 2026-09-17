//! Native Pawley wire round trips and corruption boundaries.
use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_model::PatternRecord;
use phasesmith_persistence::{PawleyProject, decode_pawley_project, encode_pawley_project};
use phasesmith_workflows::{
    PawleyInput, PawleyOptions, PawleyPhase, pawley_parameters, refine_pawley,
};

#[test]
fn native_roundtrip_and_digest_checks() {
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.54,
        u_deg2: 0.0,
        v_deg2: 0.0,
        w_deg2: 0.001,
        x_deg: 0.002,
        y_deg: 0.0,
    };
    let phases = vec![PawleyPhase {
        id: "phase".into(),
        reflection_ids: vec!["100".into()],
        two_theta_deg: vec![40.0],
        intensities: vec![1.0],
        hkl: vec![],
        lattice: None,
    }];
    let parameters = pawley_parameters(&phases, instrument, None, false).unwrap();
    let x: Vec<f64> = (0..1001).map(|i| 39.0 + f64::from(i) * 0.002).collect();
    let pattern = PatternRecord::new(x, Some(vec![0.0; 1001]), None, None, None).unwrap();
    let input = PawleyInput {
        pattern,
        instrument,
        phases,
        background: None,
        axial: None,
        signed_intensities: false,
        parameters,
        constraints: vec![],
    };
    let options = PawleyOptions::default();
    let fit = refine_pawley(&input, &options).unwrap();
    let project = PawleyProject {
        input,
        options,
        checkpoint: Some(fit.checkpoint),
    };
    let encoded = encode_pawley_project(&project).unwrap();
    let decoded = decode_pawley_project(&encoded, 10_000_000).unwrap();
    assert_eq!(decoded, project);
    assert_eq!(encode_pawley_project(&decoded).unwrap(), encoded);
    assert!(decode_pawley_project(&encoded, 10).is_err());
    let mut wire: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    wire["input"]["observed_y"][0] = serde_json::json!(123.0);
    assert!(
        decode_pawley_project(&wire.to_string(), 10_000_000)
            .unwrap_err()
            .to_string()
            .contains("digest")
    );
    let mut invalid = project;
    invalid.checkpoint.as_mut().unwrap().free.clear();
    assert!(encode_pawley_project(&invalid).is_err());
}
