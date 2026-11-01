//! Periodic real-density interpolation by Hermitian Fourier zero padding.
use crate::fourier::{Transform, grid_index, grid_len};
use crate::{SolutionError, error};
use rustfft::num_complex::Complex;

/// Interpolate a periodic real map onto an equal or finer even grid.
/// Split an upsampled source Nyquist coefficient equally between its positive
/// and negative frequencies, independently per axis. No density clipping or
/// invented observed coefficients is performed. Preserve the density scale.
/// # Errors
/// Invalid grid/length, decreasing axes, or nonfinite/excessive source values.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)] // Admitted axes are 8..128.
pub fn interpolate_phase_density(
    density: &[f64],
    source: [usize; 3],
    target: [usize; 3],
) -> Result<Vec<f64>, SolutionError> {
    let n = grid_len(source)?;
    let m = grid_len(target)?;
    if density.len() != n
        || (0..3).any(|a| target[a] < source[a])
        || density.iter().any(|v| !v.is_finite() || v.abs() > 1e100)
    {
        return Err(error("invalid phase-density interpolation input"));
    }
    if source == target {
        return Ok(density.to_vec());
    }
    let mut spectrum: Vec<_> = density.iter().map(|&v| Complex::new(v, 0.)).collect();
    Transform::batched(source).apply(&mut spectrum, true);
    let mut padded = vec![Complex::default(); m];
    for (i, &value) in spectrum.iter().enumerate() {
        let xyz = [
            i / (source[1] * source[2]),
            i / source[2] % source[1],
            i % source[2],
        ];
        let mut frequencies = [[0i32; 2]; 3];
        let mut counts = [1; 3];
        for a in 0..3 {
            let frequency = if xyz[a] <= source[a] / 2 {
                xyz[a] as i32
            } else {
                xyz[a] as i32 - source[a] as i32
            };
            frequencies[a][0] = frequency;
            if xyz[a] == source[a] / 2 && target[a] > source[a] {
                frequencies[a][1] = -frequency;
                counts[a] = 2;
            }
        }
        let value = value / (counts.iter().product::<usize>() as f64 * n as f64);
        for a in 0..counts[0] {
            for b in 0..counts[1] {
                for c in 0..counts[2] {
                    padded[grid_index(
                        [frequencies[0][a], frequencies[1][b], frequencies[2][c]],
                        target,
                    )] += value;
                }
            }
        }
    }
    Transform::batched(target).apply(&mut padded, false);
    Ok(padded.into_iter().map(|v| v.re).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_nyquist_planes_preserve_samples_mean_and_analytic_periodic_field() {
        let source = [8, 10, 12];
        let target = [16, 20, 24];
        let field = |x: [f64; 3]| {
            let t = std::f64::consts::TAU;
            1. + 0.7 * (t * 4. * x[0]).cos()
                + 0.3 * (t * 4. * x[0]).cos() * (t * 5. * x[1]).cos()
                + 0.2 * (t * (x[1] + 2. * x[2])).sin()
        };
        let density: Vec<_> = (0..960)
            .map(|i| {
                field([
                    f64::from(i / 120) / 8.,
                    f64::from(i / 12 % 10) / 10.,
                    f64::from(i % 12) / 12.,
                ])
            })
            .collect();
        let expanded = interpolate_phase_density(&density, source, target).unwrap();
        for (i, &v) in expanded.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let x = [
                (i / 480) as f64 / 16.,
                (i / 24 % 20) as f64 / 20.,
                (i % 24) as f64 / 24.,
            ];
            assert!((v - field(x)).abs() < 2e-14, "{i}: {v} vs {}", field(x));
        }
        assert!((expanded.iter().sum::<f64>() / 7680. - 1.).abs() < 2e-14);
        assert_eq!(
            interpolate_phase_density(&density, source, source).unwrap(),
            density
        );
        assert!(interpolate_phase_density(&density, source, [8, 8, 12]).is_err());
        assert!(interpolate_phase_density(&density[..959], source, target).is_err());
        let mut bad = density;
        bad[0] = f64::NAN;
        assert!(interpolate_phase_density(&bad, source, target).is_err());
    }
}
