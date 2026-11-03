//! Deterministic conservative screening selection, separate from final fit ranks.
use crate::solution::{SolutionError, error};

/// Select the best scores, near ties, and one birth for each missing element.
/// Returned indices are ordered by score then original input order.
/// # Errors
/// Rejects invalid limits, arrays, nonfinite scores or excessive candidates.
pub fn screening_shortlist(
    scores: &[f64],
    birth_elements: &[Option<String>],
    missing: &[String],
    limit: usize,
) -> Result<Vec<usize>, SolutionError> {
    if !(1..=16).contains(&limit)
        || scores.len() != birth_elements.len()
        || scores.len() > 2048
        || scores.iter().any(|s| !s.is_finite())
    {
        return Err(error("invalid screening shortlist scores, labels or limit"));
    }
    let mut order: Vec<_> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if order.is_empty() {
        return Ok(order);
    }
    let cutoff = scores[order[limit.min(order.len()) - 1]];
    let mut selected = vec![false; scores.len()];
    for &index in &order {
        if scores[index] <= cutoff + 2. {
            selected[index] = true;
        }
    }
    for element in missing {
        if let Some(&index) = order
            .iter()
            .find(|&&i| birth_elements[i].as_ref() == Some(element))
        {
            selected[index] = true;
        }
    }
    Ok(order.into_iter().filter(|&i| selected[i]).collect())
}
