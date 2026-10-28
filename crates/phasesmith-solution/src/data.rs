//! Symmetry/Friedel merging without changing the retained source observations.

use std::collections::BTreeMap;

use phasesmith_crystallography::{SpaceGroup, UnitCell};

use crate::{SolutionError, error};

/// One corrected integrated intensity; negative intensities are valid.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    /// Integer reflection index, excluding 000.
    pub hkl: [i32; 3],
    /// Signed intensity in a common arbitrary scale.
    pub intensity: f64,
    /// Strictly positive standard uncertainty of intensity.
    pub sigma: f64,
}

/// Intensity means expanded to P1 including Friedel mates, in sorted HKL order.
#[derive(Clone, Debug)]
pub struct PreparedData {
    /// Expanded reciprocal indices.
    pub hkl: Vec<[i32; 3]>,
    /// Signed inverse-variance family means, repeated only for symmetry expansion.
    pub intensity: Vec<f64>,
    /// Standard errors of family means, without empirical error inflation.
    pub sigma: Vec<f64>,
    /// sqrt(max(mean intensity, 0)); separate from source observations.
    pub amplitudes: Vec<f64>,
    /// Independent merged symmetry/Friedel family count.
    pub family_count: usize,
    /// Number of source observations excluded by exact systematic absences.
    pub absent_observations: usize,
    /// Number of independent families with nonpositive mean intensity.
    pub nonpositive_families: usize,
    /// Minimum spacing of the retained measured set, in angstrom.
    pub d_min: f64,
}

/// Validate, inverse-variance merge, and expand corrected X-ray observations.
///
/// # Errors
/// Rejects empty/oversized/nonfinite data, 000, nonpositive sigma, incompatible
/// cell/symmetry, or an empty/all-nonpositive allowed set.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)] // Small exact metric-equation coefficients.
pub fn prepare_data(
    observations: &[Observation],
    cell: UnitCell,
    group: &SpaceGroup,
) -> Result<PreparedData, SolutionError> {
    if observations.is_empty() || observations.len() > 100_000 {
        return Err(error("observation count must lie in [1,100000]"));
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let metric = geometry.direct_metric;
    let components = [
        metric[0][0],
        metric[1][1],
        metric[2][2],
        metric[1][2],
        metric[0][2],
        metric[0][1],
    ];
    let scale = components.iter().fold(1.0_f64, |a, b| a.max(b.abs()));
    for equation in &group.metric_constraints().equations {
        let residual: f64 = equation
            .iter()
            .zip(components)
            .map(|(&a, b)| a as f64 * b)
            .sum();
        let coefficients: f64 = equation.iter().map(|a| a.unsigned_abs() as f64).sum();
        if residual.abs() > 1e-10 * scale * coefficients.max(1.0) {
            return Err(error(
                "cell metric is incompatible with the supplied space group",
            ));
        }
    }
    let mut families: BTreeMap<[i32; 3], Vec<&Observation>> = BTreeMap::new();
    let mut absent_observations = 0;
    for observation in observations {
        if observation.hkl == [0; 3]
            || observation.hkl.iter().any(|h| h.unsigned_abs() > 1024)
            || !observation.intensity.is_finite()
            || observation.intensity.abs() > 1.0e100
            || !observation.sigma.is_finite()
            || observation.sigma <= 0.0
        {
            return Err(error(
                "HKLs must be nonzero and bounded by 1024; intensities finite and <=1e100; sigma positive and finite",
            ));
        }
        if group
            .is_systematically_absent(observation.hkl)
            .map_err(|e| error(e.to_string()))?
        {
            absent_observations += 1;
            continue;
        }
        let orbit = group
            .reflection_orbit(observation.hkl, true)
            .map_err(|e| error(e.to_string()))?;
        let (d, _) = geometry
            .d_spacing_and_derivatives(observation.hkl)
            .map_err(|e| error(e.to_string()))?;
        for &hkl in &orbit {
            let (other, _) = geometry
                .d_spacing_and_derivatives(hkl)
                .map_err(|e| error(e.to_string()))?;
            if (d / other - 1.0).abs() > 1.0e-7 {
                return Err(error(
                    "cell metric is incompatible with the supplied space group",
                ));
            }
        }
        families.entry(orbit[0]).or_default().push(observation);
    }
    let mut expanded = BTreeMap::new();
    let mut nonpositive_families = 0;
    let mut d_min = f64::INFINITY;
    for (&canonical, rows) in &families {
        // Normalize the weights before squaring to avoid sigma^-2 overflow.
        let minimum_sigma = rows.iter().map(|o| o.sigma).fold(f64::INFINITY, f64::min);
        let weights: Vec<_> = rows
            .iter()
            .map(|o| (minimum_sigma / o.sigma).powi(2))
            .collect();
        let total: f64 = weights.iter().sum();
        let intensity: f64 = rows
            .iter()
            .zip(&weights)
            .map(|(o, w)| o.intensity * (w / total))
            .sum();
        let sigma = minimum_sigma / total.sqrt();
        if sigma == 0.0 || !intensity.is_finite() {
            return Err(error("merged observations exceed finite numerical range"));
        }
        nonpositive_families += usize::from(intensity <= 0.0);
        for hkl in group
            .reflection_orbit(canonical, true)
            .map_err(|e| error(e.to_string()))?
        {
            let (d, _) = geometry
                .d_spacing_and_derivatives(hkl)
                .map_err(|e| error(e.to_string()))?;
            d_min = d_min.min(d);
            expanded.insert(hkl, (intensity, sigma));
        }
    }
    if expanded.is_empty() || nonpositive_families == families.len() {
        return Err(error(
            "at least one positive, symmetry-allowed intensity family is required",
        ));
    }
    if expanded.len() > 100_000 {
        return Err(error("symmetry expansion exceeds 100000 reflections"));
    }
    let mut result = PreparedData {
        hkl: Vec::with_capacity(expanded.len()),
        intensity: Vec::with_capacity(expanded.len()),
        sigma: Vec::with_capacity(expanded.len()),
        amplitudes: Vec::with_capacity(expanded.len()),
        family_count: families.len(),
        absent_observations,
        nonpositive_families,
        d_min,
    };
    for (hkl, (intensity, sigma)) in expanded {
        result.hkl.push(hkl);
        result.intensity.push(intensity);
        result.sigma.push(sigma);
        result.amplitudes.push(intensity.max(0.0).sqrt());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_crystallography::SymmetryOperation;

    #[test]
    fn signed_means_and_friedel_mates() {
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 5.,
            c_angstrom: 5.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let rows = [
            Observation {
                hkl: [1, 0, 0],
                intensity: -4.,
                sigma: 2.,
            },
            Observation {
                hkl: [-1, 0, 0],
                intensity: 8.,
                sigma: 1.,
            },
            Observation {
                hkl: [0, 1, 0],
                intensity: -1.,
                sigma: 1.,
            },
        ];
        let data = prepare_data(&rows, cell, &group).unwrap();
        assert_eq!(data.family_count, 2);
        assert_eq!(data.nonpositive_families, 1);
        assert_eq!(data.hkl.len(), 4);
        assert!((data.intensity[0] - 5.6).abs() < 1e-14);
        assert!((data.sigma[0] - (0.8_f64).sqrt()).abs() < 1e-14);
        assert!(data.intensity.contains(&-1.));
        assert!(
            prepare_data(
                &[Observation {
                    sigma: 0.,
                    ..rows[0]
                }],
                cell,
                &group
            )
            .is_err()
        );
    }
}
