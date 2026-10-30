//! Small blocked layout permutations; no arithmetic on complex transform values.
use rustfft::num_complex::Complex;

pub(super) const BATCH_LINES: usize = 8;

/// Pack adjacent strided lines as contiguous FFTs, copying small row/line tiles.
pub(super) fn pack(
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    offset: usize,
    stride: usize,
    length: usize,
    lanes: usize,
) {
    debug_assert_eq!(output.len(), length * lanes);
    for first in (0..length).step_by(BATCH_LINES) {
        for row in first..(first + BATCH_LINES).min(length) {
            let start = offset + row * stride;
            for (lane, &value) in input[start..start + lanes].iter().enumerate() {
                output[lane * length + row] = value;
            }
        }
    }
}

/// Restore each packed line to its original XYZ positions in the caller's buffer.
pub(super) fn unpack(
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    offset: usize,
    stride: usize,
    length: usize,
    lanes: usize,
) {
    debug_assert_eq!(input.len(), length * lanes);
    for first in (0..length).step_by(BATCH_LINES) {
        for row in first..(first + BATCH_LINES).min(length) {
            let start = offset + row * stride;
            for (lane, value) in output[start..start + lanes].iter_mut().enumerate() {
                *value = input[lane * length + row];
            }
        }
    }
}
