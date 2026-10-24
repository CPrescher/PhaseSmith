//! Deterministic adjacent-frame linking and detected-position moments.
use crate::{MomentSum, SpotError, invalid};

/// Link finite x/y observations in consecutive frames of one contiguous scan.
/// Edges include the exact distance boundary. Components are numbered in input
/// order. Adjacent frames separated by an angular gap must have distinct segment IDs.
///
/// # Errors
/// Rejects inconsistent/unsorted inputs, nonfinite positions and excessive candidates.
pub fn link(
    frames: &[u32],
    xy: &[[f64; 2]],
    segments: &[u32],
    radius: f64,
    max_candidates: usize,
) -> Result<Vec<u32>, SpotError> {
    if frames.len() != xy.len() || !radius.is_finite() || radius <= 0.0 || radius > 1e6 {
        return Err(invalid("invalid linking arrays or radius"));
    }
    if frames.len() > u32::MAX as usize
        || frames.windows(2).any(|w| w[0] > w[1])
        || frames.iter().any(|&f| f as usize >= segments.len())
        || xy.iter().flatten().any(|v| !v.is_finite() || v.abs() > 1e9)
    {
        return Err(invalid(
            "link coordinates must be finite and frames sorted/in range",
        ));
    }
    let mut parent: Vec<usize> = (0..frames.len()).collect();
    let mut ranks = vec![0_u32; frames.len()];
    let mut previous: Vec<usize> = Vec::new();
    let mut last = None;
    let mut first = 0;
    let mut candidates = 0usize;
    while first < frames.len() {
        let frame = frames[first];
        let end = first + frames[first..].partition_point(|&f| f == frame);
        if last.is_some_and(|f: u32| {
            f + 1 == frame && segments[f as usize] == segments[frame as usize]
        }) {
            for j in first..end {
                let low = previous.partition_point(|&i| xy[i][0] < xy[j][0] - radius);
                let high = previous.partition_point(|&i| xy[i][0] <= xy[j][0] + radius);
                for &i in &previous[low..high] {
                    candidates += 1;
                    if candidates > max_candidates {
                        return Err(invalid("rotation linking candidate budget exceeded"));
                    }
                    let dx = xy[i][0] - xy[j][0];
                    let dy = xy[i][1] - xy[j][1];
                    if dx * dx + dy * dy <= radius * radius {
                        let a = root(&mut parent, i);
                        let b = root(&mut parent, j);
                        if a != b {
                            match ranks[a].cmp(&ranks[b]) {
                                std::cmp::Ordering::Less => parent[a] = b,
                                std::cmp::Ordering::Greater => parent[b] = a,
                                std::cmp::Ordering::Equal => {
                                    let r = a.min(b);
                                    parent[a.max(b)] = r;
                                    ranks[r] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        previous = (first..end).collect();
        previous.sort_by(|&a, &b| xy[a][0].total_cmp(&xy[b][0]).then(a.cmp(&b)));
        last = Some(frame);
        first = end;
    }
    let mut numbers = vec![u32::MAX; frames.len()];
    let mut next = 0;
    let mut labels = Vec::with_capacity(frames.len());
    for i in 0..frames.len() {
        let r = root(&mut parent, i);
        if numbers[r] == u32::MAX {
            numbers[r] = next;
            next += 1;
        }
        labels.push(numbers[r]);
    }
    Ok(labels)
}
fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Summarize detected positions `[x,y,omega,weight]` using positive weights.
/// Returns `[x,y,omega,omega_rms,n_observations]`. This is a centroid of
/// detected observations, not an integrated intensity or angular uncertainty.
///
/// # Errors
/// Rejects empty supports and nonfinite omega. Invalid position/weight rows
/// do not contribute to moments; zero total weight produces NaN coordinates.
pub fn summarize(samples: &[[f64; 4]]) -> Result<[f64; 5], SpotError> {
    if samples.is_empty() || samples.iter().any(|r| !r[2].is_finite()) {
        return Err(invalid("invalid rotation samples"));
    }
    let mut sums: [MomentSum; 4] = std::array::from_fn(|_| MomentSum::default());
    for r in samples {
        if r.iter().all(|v| v.is_finite()) {
            let w = r[3].max(0.0);
            for (sum, value) in sums.iter_mut().zip([w, w * r[0], w * r[1], w * r[2]]) {
                sum.add(value);
            }
        }
    }
    let w = sums[0].value;
    let x = sums[1].value / w;
    let y = sums[2].value / w;
    let omega = sums[3].value / w;
    let mut centered = MomentSum::default();
    for r in samples {
        if r.iter().all(|v| v.is_finite()) {
            centered.add(r[3].max(0.0) * (r[2] - omega).powi(2));
        }
    }
    #[allow(clippy::cast_precision_loss)] // At most u32 observations.
    Ok([
        x,
        y,
        omega,
        (centered.value / w).sqrt(),
        samples.len() as f64,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries_and_segments() {
        let p = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]];
        assert_eq!(
            link(&[0, 1, 2], &p, &[0, 0, 1], 5.0, 100).unwrap(),
            [0, 0, 1]
        );
        assert_eq!(
            link(&[0, 1, 2], &p, &[0, 0, 0], 4.999, 100).unwrap(),
            [0, 1, 2]
        );
        assert!(link(&[0, 1], &p[..2], &[0, 0], 5.0, 0).is_err());
    }
    #[test]
    fn positive_weight_moments() {
        let v = summarize(&[[2.0, 3.0, 0.0, 10.0], [2.0, 3.0, 1.0, -2.0]]).unwrap();
        assert_eq!(
            v.map(f64::to_bits),
            [2.0_f64, 3.0, 0.0, 0.0, 2.0].map(f64::to_bits)
        );
    }
}
