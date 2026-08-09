//! Native Hill--Howard quantitative phase-analysis contracts.

use phasesmith_workflows::{QuantitativeError, QuantitativePhase, quantitative_phase_analysis};

#[test]
fn hill_howard_relation_preserves_order_and_normalizes_exactly() {
    let phases = [
        QuantitativePhase::new("Al2O3", 0.25, 6.0, 101.961_276, 254.0).unwrap(),
        QuantitativePhase::new("ZnO", 0.75, 2.0, 81.38, 47.6).unwrap(),
    ];
    let result = quantitative_phase_analysis(&phases).unwrap();
    let first = 0.25 * 6.0 * 101.961_276 * 254.0;
    let second = 0.75 * 2.0 * 81.38 * 47.6;
    assert_eq!(result[0].phase_id, "Al2O3");
    assert_eq!(result[1].phase_id, "ZnO");
    assert!((result[0].weight_fraction - first / (first + second)).abs() < 1.0e-15);
    assert_eq!(
        (result[0].weight_fraction + result[1].weight_fraction).to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn invalid_quantitative_inputs_are_structured() {
    assert_eq!(
        quantitative_phase_analysis(&[]),
        Err(QuantitativeError::EmptyPhases)
    );
    assert!(QuantitativePhase::new("", 1.0, 1.0, 1.0, 1.0).is_err());
    assert!(QuantitativePhase::new("a", -1.0, 1.0, 1.0, 1.0).is_err());
    assert!(QuantitativePhase::new("a", 1.0, 0.0, 1.0, 1.0).is_err());
    let duplicate = [
        QuantitativePhase::new("a", 1.0, 1.0, 1.0, 1.0).unwrap(),
        QuantitativePhase::new("a", 1.0, 1.0, 1.0, 1.0).unwrap(),
    ];
    assert_eq!(
        quantitative_phase_analysis(&duplicate),
        Err(QuantitativeError::DuplicatePhaseId)
    );
    let zero = [QuantitativePhase::new("a", 0.0, 1.0, 1.0, 1.0).unwrap()];
    assert_eq!(
        quantitative_phase_analysis(&zero),
        Err(QuantitativeError::ZeroTotal)
    );
}
