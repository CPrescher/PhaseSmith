//! Ordered reduction of independent fixed-aperture rotation measurements.
use crate::{SpotError, invalid};

/// No exposure contributes to this predicted crossing.
pub const NO_DATA: u32 = 1024;
/// Sum signed I, variance, background counts, signal pixels, frames and background pixels.
///
/// # Errors
/// Rejects inconsistent arrays, excessive budgets and nonfinite accumulation.
pub fn accumulate(
    count: usize,
    ids: &[usize],
    values: &[[f64; 11]],
    flags: &[u32],
) -> Result<(Vec<[f64; 6]>, Vec<u32>), SpotError> {
    if count > 1_000_000
        || ids.len() > 1_000_000
        || ids.len() != values.len()
        || ids.len() != flags.len()
    {
        return Err(invalid(
            "inconsistent integration arrays or budget exceeded",
        ));
    }
    let mut out = vec![[0.0; 6]; count];
    let mut quality = vec![0; count];
    let mut undefined = vec![false; count];
    for ((&id, v), &f) in ids.iter().zip(values).zip(flags) {
        if id >= count || !v[9].is_finite() || v[9] < 0.0 || !v[10].is_finite() || v[10] < 0.0 {
            return Err(invalid("invalid reflection ID or pixel counts"));
        }
        quality[id] |= f & !crate::NO_CENTROID;
        let r = &mut out[id];
        if v[2].is_nan() || v[3].is_nan() || v[4].is_nan() {
            undefined[id] = true;
        } else {
            if !v[2].is_finite() || !v[3].is_finite() || v[3] < 0.0 || !v[4].is_finite() {
                return Err(invalid("invalid frame intensity, error or background"));
            }
            r[0] += v[2];
            r[1] += v[3] * v[3];
            r[2] += v[4] * v[9];
        }
        r[3] += v[9];
        r[4] += 1.0;
        r[5] += v[10];
    }
    for (id, r) in out.iter_mut().enumerate() {
        if !r.iter().all(|v| v.is_finite()) {
            return Err(invalid("integration accumulation overflow"));
        }
        if r[4] == 0.0 {
            quality[id] |= NO_DATA;
        }
        if undefined[id] || r[4] == 0.0 {
            r[..3].fill(f64::NAN);
        } else {
            r[1] = r[1].sqrt();
        }
    }
    Ok((out, quality))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(clippy::float_cmp)] // Exact arithmetic and deterministic reduction are the contract.
    fn signed_ordered_sum_and_missing() {
        let mut v = [0.0; 11];
        v[2] = -3.0;
        v[3] = 2.0;
        v[4] = 5.0;
        v[9] = 9.0;
        v[10] = 16.0;
        let (a, f) = accumulate(2, &[0, 0], &[v, v], &[0, crate::MASKED]).unwrap();
        assert_eq!(a[0], [-6.0, 8_f64.sqrt(), 90.0, 18.0, 2.0, 32.0]);
        assert_eq!(f, [crate::MASKED, NO_DATA]);
        assert!(a[1][0].is_nan());
        v[2] = f64::NAN;
        assert!(accumulate(1, &[0], &[v], &[crate::BACKGROUND]).unwrap().0[0][0].is_nan());
        assert!(accumulate(1, &[1], &[v], &[0]).is_err());
    }
}
