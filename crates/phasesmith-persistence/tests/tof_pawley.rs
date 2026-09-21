//! TOF Pawley native codec and exclusive atomic file checks (including Windows CI).
use phasesmith_core::TofInstrument;
use phasesmith_model::TofPatternRecord;
use phasesmith_persistence::{
    TofPawleyProject, decode_tof_pawley_project, encode_tof_pawley_project,
    load_tof_pawley_project, save_tof_pawley_project,
};
use phasesmith_workflows::{
    PawleyOptions, TofPawleyBank, TofPawleyInput, TofPawleyPhase, refine_tof_pawley,
};

#[test]
fn native_joint_roundtrip_digest_and_exclusive_save() {
    let instrument = TofInstrument::from_values([
        0.0, 5000.0, 0.0, 0.0, 0.18, 0.04, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.3, 0.0, 0.4,
    ])
    .unwrap();
    let banks = (0..2)
        .map(|bank| TofPawleyBank {
            id: format!("bank{bank}"),
            pattern: TofPatternRecord::new(
                (0..201).map(|i| 9900.0 + f64::from(i)).collect(),
                Some(vec![0.0; 201]),
                None,
                None,
                None,
            )
            .unwrap(),
            instrument,
            phases: vec![TofPawleyPhase {
                id: "sample".into(),
                reflection_ids: vec!["family".into()],
                hkl: vec![],
                d_spacing_angstrom: vec![2.0],
                intensities: vec![1.0],
            }],
            background: None,
            normalization: "Synthetic density per microsecond".into(),
        })
        .collect();
    let input = TofPawleyInput::new(banks, vec![], false, 20.0).unwrap();
    let options = PawleyOptions::default();
    let fit = refine_tof_pawley(&input, &options).unwrap();
    assert!(fit.evaluation.intensities.iter().all(|a| a.abs() < 1e-10));
    let project = TofPawleyProject {
        input,
        options,
        checkpoint: Some(fit.checkpoint),
    };
    let text = encode_tof_pawley_project(&project).unwrap();
    assert_eq!(
        decode_tof_pawley_project(&text, text.len()).unwrap(),
        project
    );
    assert!(decode_tof_pawley_project(&text, 10).is_err());
    let mut wire: serde_json::Value = serde_json::from_str(&text).unwrap();
    wire["input"]["banks"][1]["normalization"] = serde_json::json!("Changed normalization");
    assert!(
        decode_tof_pawley_project(&wire.to_string(), 10_000_000)
            .unwrap_err()
            .to_string()
            .contains("digest")
    );
    let mut wire: serde_json::Value = serde_json::from_str(&text).unwrap();
    wire["version"] = serde_json::json!(2);
    assert!(decode_tof_pawley_project(&wire.to_string(), 10_000_000).is_err());
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("phasesmith-tof-pawley-{nonce}"));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("joint.json");
    save_tof_pawley_project(&path, &project).unwrap();
    assert!(save_tof_pawley_project(&path, &project).is_err());
    assert_eq!(load_tof_pawley_project(&path, 10_000_000).unwrap(), project);
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
    std::fs::remove_dir_all(directory).unwrap();
}
