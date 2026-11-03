//! Measured-envelope bounded prediction, kept separate from observation arrays.
use crate::solution::fourier::{grid_index, grid_len};
use crate::solution::math3::{dot, mv};
use crate::solution::{PreparedData, SolutionError, WilsonNormalization, error};
use phasesmith_crystallography::{SpaceGroup, UnitCell};
use rustfft::num_complex::Complex;

/// Static prediction provenance. Counts refer to Friedel pairs, not observations.
#[derive(Clone, Debug)]
pub struct PredictionAudit {
    /// Actual Fourier grid.
    pub shape: [usize; 3],
    /// Original measured row count.
    pub measured_rows: usize,
    /// Missing pairs inside the observed sphere that receive bounded prediction.
    pub missing_pairs: usize,
    /// Newly supported allowed pairs beyond the observed sphere.
    pub extended_pairs: usize,
    /// Original measured resolution in angstrom.
    pub observed_d_min: f64,
    /// Requested reciprocal-radius multiplier.
    pub requested_extension: f64,
    /// Effective multiplier after explicit Nyquist bounds.
    pub effective_extension: f64,
    /// Effective support resolution in angstrom; never measured resolution.
    pub effective_d_min: f64,
}
/// One actual prediction step, including its bounded work.
#[derive(Clone, Debug)]
pub struct PredictionStep {
    /// Nonzero predicted pairs updated.
    pub updated: usize,
    /// Pairs at the normalized magnitude cap.
    pub clipped: usize,
    /// Average measured-family intensity correlation across populated bins.
    pub reliability: f64,
}
/// Local complex prediction and analytical 2x2 coefficient Jacobian.
#[derive(Clone, Debug)]
pub struct PredictedCoefficient {
    /// Predicted real and imaginary components.
    pub value: [f64; 2],
    /// Cartesian derivative with shell statistics held fixed.
    pub jacobian: [[f64; 2]; 2],
    /// Whether the squared normalized amplitude was capped at 16.
    pub clipped: bool,
}
/// Bounded conditional-second-moment prediction with soft phase regularization.
/// # Errors
/// Rejects nonfinite coefficients/statistics or invalid positive scales/gains.
pub fn prediction_coefficient(
    z: [f64; 2],
    expected: f64,
    moment: f64,
    correlation: f64,
    gain: f64,
) -> Result<PredictedCoefficient, SolutionError> {
    if z.iter().any(|v| !v.is_finite() || v.abs() > 1e100)
        || !expected.is_finite()
        || !(1e-100..=1e100).contains(&expected)
        || !moment.is_finite()
        || !(1e-12..=1e12).contains(&moment)
        || !correlation.is_finite()
        || !(0. ..=1.).contains(&correlation)
        || !gain.is_finite()
        || !(0. ..=1.).contains(&gain)
    {
        return Err(error(
            "invalid reflection prediction coefficient/statistics",
        ));
    }
    Ok(coefficient(z, expected, moment, correlation, gain))
}
fn coefficient(
    z: [f64; 2],
    expected: f64,
    moment: f64,
    correlation: f64,
    gain: f64,
) -> PredictedCoefficient {
    let radius = z[0].hypot(z[1]);
    let width = expected * moment.sqrt();
    let relative = radius / width;
    let clipped = relative >= 4.;
    let root = (1. - correlation + correlation * relative.min(4.).powi(2))
        .max(0.)
        .sqrt();
    let amplitude = gain * correlation * expected * root;
    let denominator = radius.hypot(0.1 * width);
    let factor = amplitude / denominator;
    let unit = if radius > 0. {
        z.map(|x| x / radius)
    } else {
        [0.; 2]
    };
    let radial = if !clipped && root > 1e-100 {
        gain * correlation * correlation * expected / denominator * relative.powi(2) / root
    } else {
        0.
    } - factor * (radius / denominator).powi(2);
    PredictedCoefficient {
        value: z.map(|x| amplitude * (x / denominator)),
        jacobian: std::array::from_fn(|i| {
            std::array::from_fn(|j| factor * f64::from(i == j) + radial * unit[i] * unit[j])
        }),
        clipped,
    }
}
#[derive(Clone, Debug)]
struct Entry {
    index: usize,
    mate: usize,
    bin: usize,
    expected: f64,
    taper: f64,
    extended: bool,
}
#[derive(Clone, Debug)]
struct Measured {
    index: usize,
    bin: usize,
    expected: f64,
    observed: f64,
}
/// Frozen support and observed-only normalization for one amplitude convention.
#[derive(Clone, Debug)]
pub struct PreparedReflectionPrediction {
    /// Static coefficient provenance; does not modify the original data.
    pub audit: PredictionAudit,
    /// Existing support convention: 0 fixed zero, 1 free/predicted, 2 measured.
    pub support: Vec<u8>,
    /// RMS used to normalize the supplied measured amplitudes.
    pub observed_rms: f64,
    entries: Vec<Entry>,
    measured: Vec<Measured>,
}
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn signed(index: usize, shape: [usize; 3]) -> [i32; 3] {
    let p = [
        index / (shape[1] * shape[2]),
        index / shape[2] % shape[1],
        index % shape[2],
    ];
    std::array::from_fn(|a| {
        if p[a] < shape[a] / 2 {
            p[a] as i32
        } else {
            p[a] as i32 - shape[a] as i32
        }
    })
}
impl PreparedReflectionPrediction {
    /// Diagnostic reciprocal columns (index,mate,bin,expected,taper), never observations.
    #[must_use]
    pub fn predicted_columns(&self) -> Vec<(usize, usize, usize, f64, f64, bool)> {
        self.entries
            .iter()
            .map(|r| (r.index, r.mate, r.bin, r.expected, r.taper, r.extended))
            .collect()
    }
    /// Independent measured columns (index,bin,expected,observed).
    #[must_use]
    pub fn measured_columns(&self) -> Vec<(usize, usize, f64, f64)> {
        self.measured
            .iter()
            .map(|r| (r.index, r.bin, r.expected, r.observed))
            .collect()
    }

    /// Prepare predicted roles and the original measured Wilson envelope.
    /// `power` is 1 for Fo maps and 0.5 for mixed sqrt(E*F) phasing amplitudes.
    /// # Errors
    /// Rejects incompatible arrays, controls, aliasing, normalization or grid budgets.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn new(
        shape: [usize; 3],
        cell: UnitCell,
        group: &SpaceGroup,
        data: &PreparedData,
        norm: &WilsonNormalization,
        elements: &[String],
        amplitudes: &[f64],
        power: f64,
        missing: bool,
        extension: f64,
    ) -> Result<Self, SolutionError> {
        grid_len(shape)?;
        if !matches!(power, 0.5 | 1.)
            || !extension.is_finite()
            || !(1. ..=1.5).contains(&extension)
            || amplitudes.len() != data.hkl.len()
            || norm.expected_intensity.len() != data.hkl.len()
            || amplitudes
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 1e100)
        {
            return Err(error("invalid reflection prediction controls/arrays"));
        }
        let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
        let lengths = [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom];
        let effective = (0..3)
            .map(|a| (shape[a] as f64 / 2. - 1e-6) * data.d_min / lengths[a])
            .fold(extension, f64::min);
        if effective < 1. {
            return Err(error("prediction grid cannot contain the measured sphere"));
        }
        let mut band = data.clone();
        band.d_min /= effective;
        let support = crate::solution::solver::reciprocal_support(shape, cell, group, &band)?;
        let max = amplitudes.iter().copied().fold(0., f64::max);
        if max <= 0. {
            return Err(error("prediction needs positive measured amplitudes"));
        }
        let rms = max
            * (amplitudes.iter().map(|v| (v / max).powi(2)).sum::<f64>() / amplitudes.len() as f64)
                .sqrt();
        let qobs2 = data.d_min.recip().powi(2);
        let qext2 = band.d_min.recip().powi(2);
        let bin = |q2: f64| ((q2 / qobs2 * 8. + 1e-10).floor() as usize).min(7);
        let mut measured = Vec::new();
        for &row in &norm.representatives {
            let h = data.hkl[row];
            let q = mv(geometry.reciprocal_basis, h.map(f64::from));
            measured.push(Measured {
                index: grid_index(h, shape),
                bin: bin(dot(q, q)),
                expected: norm.expected_intensity[row].powf(0.5 * power) / rms,
                observed: amplitudes[row] / rms,
            });
        }
        let mut hkl = Vec::new();
        let mut roles = Vec::new();
        let mut missing_pairs = 0;
        let mut extended_pairs = 0;
        for (index, &code) in support.iter().enumerate().skip(1) {
            if code != 1 {
                continue;
            }
            let h = signed(index, shape);
            if h > h.map(|v| -v) {
                continue;
            }
            let q = mv(geometry.reciprocal_basis, h.map(f64::from));
            let q2 = dot(q, q);
            let extended = q2 > qobs2 * (1. + 1e-12);
            if !extended && !missing {
                continue;
            }
            let taper = if extended {
                f64::midpoint(
                    1.,
                    (std::f64::consts::PI * ((q2 - qobs2) / (qext2 - qobs2)).clamp(0., 1.)).cos(),
                )
            } else {
                1.
            };
            hkl.push(h);
            roles.push((
                index,
                grid_index(h.map(|v| -v), shape),
                bin(q2),
                taper,
                extended,
            ));
            if extended {
                extended_pairs += 1;
            } else {
                missing_pairs += 1;
            }
        }
        let mut expectations = Vec::with_capacity(hkl.len());
        for block in hkl.chunks(4096) {
            expectations.extend(norm.expected_for_hkl(cell, group, elements, block)?);
        }
        let entries = roles
            .into_iter()
            .zip(expectations)
            .map(|((index, mate, bin, taper, extended), w)| Entry {
                index,
                mate,
                bin,
                expected: w.powf(0.5 * power) / rms,
                taper,
                extended,
            })
            .collect();
        Ok(Self {
            audit: PredictionAudit {
                shape,
                measured_rows: data.hkl.len(),
                missing_pairs,
                extended_pairs,
                observed_d_min: data.d_min,
                requested_extension: extension,
                effective_extension: effective,
                effective_d_min: band.d_min,
            },
            support,
            observed_rms: rms,
            entries,
            measured,
        })
    }
    /// Estimate model moments/correlations from independent measured families only.
    /// # Errors
    /// Rejects wrong grids, nonfinite coefficients or nonpositive input unit scale.
    pub fn statistics(
        &self,
        values: &[Complex<f64>],
        unit_scale: f64,
    ) -> Result<[(f64, f64); 8], SolutionError> {
        if values.len() != self.support.len() || !unit_scale.is_finite() || unit_scale <= 0. {
            return Err(error("invalid prediction coefficient grid/scale"));
        }
        let mut sums = [[0.; 6]; 9];
        for row in &self.measured {
            let a = (row.observed / row.expected).powi(2);
            let b = (values[row.index].norm() * unit_scale / row.expected).powi(2);
            if !a.is_finite() || !b.is_finite() {
                return Err(error("nonfinite prediction shell moments"));
            }
            for index in [row.bin, 8] {
                let s = &mut sums[index];
                s[0] += 1.;
                s[1] += a;
                s[2] += b;
                s[3] += a * a;
                s[4] += b * b;
                s[5] += a * b;
            }
        }
        Ok(std::array::from_fn(|i| {
            let s = if sums[i][0] >= 12. { sums[i] } else { sums[8] };
            let v = (s[2] / s[0].max(1.)).clamp(1e-12, 1e12);
            let denominator =
                ((s[0] * s[3] - s[1] * s[1]).max(0.) * (s[0] * s[4] - s[2] * s[2]).max(0.)).sqrt();
            let c = if denominator > 1e-100 {
                ((s[0] * s[5] - s[1] * s[2]) / denominator).clamp(0., 1.)
            } else {
                0.
            };
            (v, c)
        }))
    }
    /// Apply a prediction step, preserving measured, origin, absent and free-control entries.
    /// # Errors
    /// Rejects invalid grids or nonfinite measured moments.
    pub fn apply(
        &self,
        values: &mut [Complex<f64>],
        unit_scale: f64,
        cycle: usize,
    ) -> Result<PredictionStep, SolutionError> {
        let statistics = self.statistics(values, unit_scale)?;
        self.apply_with_statistics(values, unit_scale, cycle, &statistics)
    }
    /// Apply fixed measured-family statistics to a symmetry-projected coefficient grid.
    /// # Errors
    /// Rejects malformed grids, scales or statistics.
    pub fn apply_with_statistics(
        &self,
        values: &mut [Complex<f64>],
        unit_scale: f64,
        cycle: usize,
        statistics: &[(f64, f64); 8],
    ) -> Result<PredictionStep, SolutionError> {
        if values.len() != self.support.len()
            || !unit_scale.is_finite()
            || unit_scale <= 0.
            || statistics.iter().any(|(v, c)| {
                !v.is_finite()
                    || !(1e-12..=1e12).contains(v)
                    || !c.is_finite()
                    || !(0. ..=1.).contains(c)
            })
        {
            return Err(error("invalid prediction grid/statistics"));
        }
        let ramp = if cycle < 8 {
            0.
        } else {
            (f64::from(u32::try_from(cycle.saturating_sub(7).min(16)).unwrap_or(16)) / 16.).min(1.)
        };
        let mut step = PredictionStep {
            updated: 0,
            clipped: 0,
            reliability: statistics.iter().map(|v| v.1).sum::<f64>() / 8.,
        };
        for entry in &self.entries {
            let z = values[entry.index] * unit_scale;
            if !z.re.is_finite() || !z.im.is_finite() {
                return Err(error("nonfinite predicted coefficient"));
            }
            let (v, c) = statistics[entry.bin];
            let prediction = coefficient([z.re, z.im], entry.expected, v, c, ramp * entry.taper);
            let conditional = Complex::new(prediction.value[0], prediction.value[1]);
            let result = (conditional
                + if entry.extended {
                    Complex::default()
                } else {
                    z * (1. - ramp)
                })
                / unit_scale;
            values[entry.index] = result;
            values[entry.mate] = result.conj();
            step.updated += usize::from(ramp > 0. && result.norm_sqr() > 0.);
            step.clipped += usize::from(ramp > 0. && prediction.clipped);
        }
        Ok(step)
    }
}
