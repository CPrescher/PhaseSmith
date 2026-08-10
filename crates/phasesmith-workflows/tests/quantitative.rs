//! Native Hill--Howard quantitative phase-analysis contracts.

use phasesmith_workflows::{
    QuantitativeError, QuantitativePhase, quantitative_phase_analysis,
    quantitative_phase_analysis_with_covariance,
};

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

#[test]
fn analytical_fraction_covariance_matches_centered_scale_differences() {
    let phases = vec![
        QuantitativePhase::new("a", 2.0, 1.0, 3.0, 4.0).unwrap(),
        QuantitativePhase::new("b", 3.0, 2.0, 5.0, 6.0).unwrap(),
    ];
    let scale_covariance = [0.04, 0.01, 0.01, 0.09];
    let analysis = quantitative_phase_analysis_with_covariance(&phases, &scale_covariance).unwrap();
    let step = 1.0e-6;
    let mut jacobian = [[0.0; 2]; 2];
    for column in 0..2 {
        let mut plus = phases.clone();
        let mut minus = phases.clone();
        plus[column].scale += step;
        minus[column].scale -= step;
        let high = quantitative_phase_analysis(&plus).unwrap();
        let low = quantitative_phase_analysis(&minus).unwrap();
        for row in 0..2 {
            jacobian[row][column] =
                (high[row].weight_fraction - low[row].weight_fraction) / (2.0 * step);
        }
    }
    for row in 0..2 {
        for column in 0..2 {
            let mut expected = 0.0;
            for left in 0..2 {
                for right in 0..2 {
                    expected += jacobian[row][left]
                        * scale_covariance[left * 2 + right]
                        * jacobian[column][right];
                }
            }
            assert!((analysis.covariance[row * 2 + column] - expected).abs() < 1.0e-12);
        }
    }
    assert!((analysis.covariance[0] - analysis.covariance[3]).abs() < 1.0e-12);
    assert!((analysis.covariance[1] + analysis.covariance[0]).abs() < 1.0e-12);
    assert_eq!(
        quantitative_phase_analysis_with_covariance(&phases, &[1.0]),
        Err(QuantitativeError::CovarianceShape)
    );
}
