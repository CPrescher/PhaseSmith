//! Reciprocal symmetry/Friedel phase projection and positive-density recycling.
use crate::fourier::{Transform, grid_index, grid_len, prepare_fourier};
use crate::{Cancellation, SolutionError, error};
use phasesmith_crystallography::SpaceGroup;
use rustfft::num_complex::Complex;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Member {
    index: usize,
    factor: Complex<f64>,
    conjugate: bool,
}
#[derive(Clone, Debug)]
struct Orbit {
    terms: Vec<Member>,
    members: Vec<Member>,
    target: Option<f64>,
    fallback: Complex<f64>,
}

/// Actual bounded recycling result, in the supplied symmetry frame.
#[derive(Clone, Debug)]
pub struct RecycledDensity {
    /// Band-limited, final symmetry-projected map; negative ripples are allowed.
    pub density: Vec<f64>,
    /// Completed positivity cycles.
    pub cycles: usize,
    /// Allowed unmeasured coefficients, including F000.
    pub predicted_coefficients: usize,
    /// Negative-node fraction before each positivity projection.
    pub negative_fractions: Vec<f64>,
    /// Degenerate phasor averages using the documented deterministic fallback.
    pub phase_fallbacks: usize,
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn signed_index(i: usize, shape: [usize; 3]) -> [i32; 3] {
    let p = [
        i / (shape[1] * shape[2]),
        (i / shape[2]) % shape[1],
        i % shape[2],
    ];
    std::array::from_fn(|a| {
        if p[a] < shape[a] / 2 {
            p[a] as i32
        } else {
            p[a] as i32 - shape[a] as i32
        }
    })
}
fn transformed(
    h: [i32; 3],
    rotation: [[i32; 3]; 3],
    shape: [usize; 3],
) -> Result<[i32; 3], SolutionError> {
    let mut result = [0; 3];
    for a in 0..3 {
        let v = (0..3)
            .map(|i| i64::from(h[i]) * i64::from(rotation[i][a]))
            .sum::<i64>();
        if 2 * v.abs() >= i64::try_from(shape[a]).unwrap_or(0) {
            return Err(error("symmetry recycling support aliases Nyquist boundary"));
        }
        result[a] = i32::try_from(v).map_err(|_| error("symmetry reciprocal index overflow"))?;
    }
    Ok(result)
}
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn plan(
    shape: [usize; 3],
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    support: &[u8],
    group: &SpaceGroup,
) -> Result<Vec<Orbit>, SolutionError> {
    if group.operations().len() > 192 {
        return Err(error("recycling symmetry operation budget exceeds 192"));
    }
    let mut target = vec![None; support.len()];
    for (&h, &a) in hkl.iter().zip(amplitudes) {
        target[grid_index(h, shape)] = Some(a);
    }
    let mut seen = vec![false; support.len()];
    let mut orbits = Vec::new();
    let mut bytes = 0;
    for index in 0..support.len() {
        if support[index] == 0 || seen[index] {
            continue;
        }
        let h = signed_index(index, shape);
        if (0..3).any(|a| i64::from(h[a]).abs() * 2 >= i64::try_from(shape[a]).unwrap_or(0)) {
            return Err(error("recycling active Nyquist coefficient"));
        }
        if group
            .is_systematically_absent(h)
            .map_err(|e| error(e.to_string()))?
        {
            return Err(error("recycling support contains systematic absence"));
        }
        let mut terms = Vec::new();
        let mut members = Vec::new();
        for &operation in group.operations() {
            let mapped = transformed(h, operation.rotation(), shape)?;
            let translation = operation
                .translation()
                .map(phasesmith_crystallography::Rational::as_f64);
            let angle = std::f64::consts::TAU
                * (0..3)
                    .map(|a| f64::from(h[a]) * translation[a])
                    .sum::<f64>();
            let factor = Complex::from_polar(1., -angle);
            for conjugate in [false, true] {
                let idx = grid_index(
                    if conjugate {
                        mapped.map(|x| -x)
                    } else {
                        mapped
                    },
                    shape,
                );
                if support[idx] != support[index] {
                    return Err(error(
                        "recycling reciprocal support/families are not symmetry closed",
                    ));
                }
                if let (Some(a), Some(b)) = (target[index], target[idx]) {
                    if (a - b).abs() > 1e-10 * (1. + a) {
                        return Err(error("inconsistent symmetry-equivalent amplitudes"));
                    }
                }
                let term = Member {
                    index: idx,
                    factor: if conjugate { factor.conj() } else { factor },
                    conjugate,
                };
                if !seen[idx] {
                    seen[idx] = true;
                    members.push(term.clone());
                }
                terms.push(term);
            }
        }
        bytes += (terms.len() + members.len()) * std::mem::size_of::<Member>();
        if bytes > 64 * 1024 * 1024 {
            return Err(error("symmetry recycling projection plan exceeds 64 MiB"));
        }
        let fallback = terms
            .iter()
            .find(|term| term.index == index && term.conjugate)
            .map_or(Complex::new(1., 0.), |term| {
                Complex::from_polar(1., term.factor.arg() / 2.)
            });
        orbits.push(Orbit {
            terms,
            members,
            target: target[index],
            fallback,
        });
    }
    Ok(orbits)
}
fn project(input: &[Complex<f64>], output: &mut [Complex<f64>], orbits: &[Orbit]) -> usize {
    output.fill(Complex::default());
    let mut fallbacks = 0;
    for orbit in orbits {
        let mut sum: Complex<f64> = Complex::default();
        let mut magnitude = 0.;
        for term in &orbit.terms {
            let value = input[term.index];
            let norm = value.norm();
            let value = if norm > 1e-100 {
                value / norm
            } else {
                Complex::default()
            };
            let value = value * term.factor.conj();
            sum += if term.conjugate { value.conj() } else { value };
            magnitude += norm;
        }
        let raw_norm = sum.norm();
        #[allow(clippy::cast_precision_loss)]
        let norm = raw_norm / orbit.terms.len() as f64;
        let phase = if norm > 1e-12 {
            sum / raw_norm
        } else {
            fallbacks += 1;
            orbit.fallback
        };
        #[allow(clippy::cast_precision_loss)]
        let amplitude = orbit.target.unwrap_or(magnitude / orbit.terms.len() as f64);
        for member in &orbit.members {
            output[member.index] = member.factor
                * if member.conjugate {
                    phase.conj()
                } else {
                    phase
                }
                * amplitude;
        }
    }
    fallbacks
}

/// Recycle a phase map in one supplied origin/hand before weak-peak assignment.
/// # Errors
/// Rejects malformed/nonfinite arrays, non-closed support, aliases, inconsistent
/// amplitudes, invalid transforms/cycles and the explicit projection-plan budget.
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
pub fn recycle_density(
    shape: [usize; 3],
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    support: &[u8],
    group: &SpaceGroup,
    source: &[f64],
    hand: i32,
    shift: [f64; 3],
    cycles: usize,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<RecycledDensity, SolutionError> {
    let n = grid_len(shape)?;
    if source.len() != n
        || source.iter().any(|v| !v.is_finite() || v.abs() > 1e100)
        || ![-1, 1].contains(&hand)
        || shift.iter().any(|v| !v.is_finite())
        || cycles > 50
    {
        return Err(error("invalid symmetry recycling map/transform/cycles"));
    }
    prepare_fourier(shape, hkl, amplitudes, &vec![0.; hkl.len()], support)?;
    let orbits = plan(shape, hkl, amplitudes, support, group)?;
    let mut fft = Transform::new(shape);
    let mut original: Vec<_> = source.iter().map(|&v| Complex::new(v, 0.)).collect();
    fft.apply(&mut original, true);
    let shift = shift.map(|v| v.rem_euclid(1.));
    let mut coefficients = vec![Complex::default(); n];
    for (i, &code) in support.iter().enumerate() {
        if code == 0 {
            continue;
        }
        let h = signed_index(i, shape);
        let angle = std::f64::consts::TAU * (0..3).map(|a| f64::from(h[a]) * shift[a]).sum::<f64>();
        coefficients[i] =
            original[grid_index(h.map(|v| hand * v), shape)] * Complex::from_polar(1., angle);
    }
    drop(original);
    let mut projected = vec![Complex::default(); n];
    let mut negative_fractions = Vec::new();
    let mut fallbacks = 0;
    for _ in 0..cycles {
        if crate::stopped(deadline, cancellation) {
            break;
        }
        fallbacks += project(&coefficients, &mut projected, &orbits);
        fft.apply(&mut projected, false);
        let negative = projected.iter().filter(|v| v.re < 0.).count();
        negative_fractions.push(negative as f64 / n as f64);
        for value in &mut projected {
            *value = Complex::new((value.re / n as f64).max(0.), 0.);
        }
        fft.apply(&mut projected, true);
        std::mem::swap(&mut coefficients, &mut projected);
    }
    fallbacks += project(&coefficients, &mut projected, &orbits);
    fft.apply(&mut projected, false);
    let density: Vec<_> = projected.iter().map(|v| v.re / n as f64).collect();
    if density.iter().any(|v| !v.is_finite()) {
        return Err(error("nonfinite recycled density"));
    }
    #[allow(clippy::naive_bytecount)] // Avoid a new dependency for this once-per-map audit.
    let predicted_coefficients = support.iter().filter(|&&v| v == 1).count();
    Ok(RecycledDensity {
        density,
        cycles: negative_fractions.len(),
        predicted_coefficients,
        negative_fractions,
        phase_fallbacks: fallbacks,
    })
}
