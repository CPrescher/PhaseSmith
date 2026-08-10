//! Validation-only anchor initialization for linear backgrounds.

use nalgebra::{DMatrix, DVector};
use phasesmith_workflows::{
    BackgroundModel, ChebyshevBackground, DifferentiableBackground, LeBailError,
};

/// Select deterministic, evenly spaced background anchors, including both ends.
pub(crate) fn regular_background_anchors(
    x_deg: &[f64],
    baseline: &[f64],
    anchor_count: usize,
) -> Result<(Vec<f64>, Vec<f64>), LeBailError> {
    if x_deg.len() != baseline.len() || anchor_count < 2 || anchor_count > x_deg.len() {
        return Err(LeBailError::LinearSolve);
    }
    let last = x_deg.len() - 1;
    let denominator = anchor_count - 1;
    let indices = (0..anchor_count)
        .map(|anchor| anchor * last / denominator)
        .collect::<Vec<_>>();
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(LeBailError::LinearSolve);
    }
    Ok((
        indices.iter().map(|index| x_deg[*index]).collect(),
        indices.iter().map(|index| baseline[*index]).collect(),
    ))
}

/// Fit a Chebyshev background through caller-owned anchor points.
///
/// Anchors may come from deterministic preprocessing or manual GUI picks. Only
/// the fitted Chebyshev model enters the subsequent refinement.
pub(crate) fn fitted_chebyshev_from_anchors(
    background_id: &str,
    domain: [f64; 2],
    anchor_x_deg: &[f64],
    anchor_y: &[f64],
    terms: usize,
) -> Result<BackgroundModel, LeBailError> {
    let initial = ChebyshevBackground::new(background_id, vec![0.0; terms], domain)
        .map_err(LeBailError::Background)?;
    let basis = initial
        .basis(anchor_x_deg)
        .map_err(LeBailError::Background)?;
    if anchor_y.len() != anchor_x_deg.len() || anchor_x_deg.len() < terms {
        return Err(LeBailError::LinearSolve);
    }
    let matrix = DMatrix::from_row_slice(basis.rows, basis.columns, &basis.values);
    let target = DVector::from_vec(anchor_y.to_vec());
    let coefficients = matrix
        .svd(true, true)
        .solve(&target, 1.0e-12)
        .map_err(|_| LeBailError::LinearSolve)?;
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(LeBailError::LinearSolve);
    }
    initial
        .replace_coefficients(coefficients.as_slice())
        .map(BackgroundModel::Chebyshev)
        .map_err(LeBailError::Background)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_anchors_include_both_domain_ends() {
        let x = (0..10).map(f64::from).collect::<Vec<_>>();
        let baseline = x.iter().map(|value| value * value).collect::<Vec<_>>();

        let (anchor_x, anchor_y) =
            regular_background_anchors(&x, &baseline, 4).expect("regular anchors");

        assert_eq!(anchor_x, vec![0.0, 3.0, 6.0, 9.0]);
        assert_eq!(anchor_y, vec![0.0, 9.0, 36.0, 81.0]);
    }

    #[test]
    fn anchor_fit_recovers_a_ten_term_chebyshev_model() {
        let x = (0..20).map(f64::from).collect::<Vec<_>>();
        let expected = ChebyshevBackground::new(
            "expected",
            vec![8.0, -2.0, 1.0, 0.5, -0.2, 0.1, -0.05, 0.02, -0.01, 0.005],
            [0.0, 19.0],
        )
        .expect("expected background");
        let y = expected.calculate(&x).expect("expected values");

        let actual =
            fitted_chebyshev_from_anchors("actual", [0.0, 19.0], &x, &y, 10).expect("anchor fit");
        let actual_y = actual.calculate(&x).expect("actual values");

        assert!(
            actual_y
                .iter()
                .zip(y)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12)
        );
    }
}
