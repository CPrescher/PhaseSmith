//! Read-only residual localization. No physical cause is inferred from residuals.

/// Residual evidence in one equal-coordinate interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualRegion {
    /// Lower coordinate edge (included).
    pub lower: f64,
    /// Upper coordinate edge (excluded except in the last region).
    pub upper: f64,
    /// Number of included observations.
    pub count: usize,
    /// Sum of squared weighted residuals.
    pub chi_square: f64,
    /// Fraction of total chi-square; zero for an exact fit.
    pub chi_square_fraction: f64,
    /// Largest absolute weighted residual, or zero for an empty region.
    pub maximum_absolute_weighted_residual: f64,
}

/// Mask-aware residual evidence, independent of a model attribution policy.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualDiagnostics {
    /// Number of included observations.
    pub included_count: usize,
    /// Number of pairs adjacent in the original grid and both included.
    pub adjacent_pair_count: usize,
    /// Sum of squared weighted residuals.
    pub chi_square: f64,
    /// Unweighted residual mean.
    pub mean: f64,
    /// Weighted residual root-mean-square.
    pub weighted_rms: f64,
    /// Sum of adjacent squared differences divided by total chi-square.
    /// Undefined for zero chi-square or no adjacent included pair.
    pub durbin_watson: Option<f64>,
    /// Equal-coordinate regions in coordinate order (including empty ones).
    pub regions: Vec<ResidualRegion>,
}

/// Localize residual error without crossing excluded intervals.
///
/// `residual` uses calculated-minus-observed convention. `weighted` is that
/// residual divided by the actual fit uncertainty, or unchanged for unit
/// weighting. Regions are left-closed/right-open except for the final edge.
/// Squared residual fractions describe where error lies, never its cause.
///
/// # Errors
/// Returns an error for invalid shapes, nonfinite input or arithmetic, an
/// unsorted grid, an empty selection, or a region count outside `1..=4096`.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn diagnose_residuals(
    x: &[f64],
    residual: &[f64],
    weighted: &[f64],
    included: &[bool],
    region_count: usize,
) -> Result<ResidualDiagnostics, &'static str> {
    if x.len() < 2
        || residual.len() != x.len()
        || weighted.len() != x.len()
        || included.len() != x.len()
    {
        return Err("diagnostic arrays must have matching lengths of at least two");
    }
    if !(1..=4096).contains(&region_count) {
        return Err("region count must be between 1 and 4096");
    }
    if x.iter()
        .chain(residual)
        .chain(weighted)
        .any(|v| !v.is_finite())
        || x.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("diagnostic arrays must be finite and coordinates strictly increasing");
    }
    let span = x[x.len() - 1] - x[0];
    if !span.is_finite() {
        return Err("diagnostic coordinate span overflowed");
    }
    let edges = (0..=region_count)
        .map(|i| {
            if i == region_count {
                x[x.len() - 1]
            } else {
                x[0] + span * (i as f64 / region_count as f64)
            }
        })
        .collect::<Vec<_>>();
    let mut regions = edges
        .windows(2)
        .map(|pair| ResidualRegion {
            lower: pair[0],
            upper: pair[1],
            count: 0,
            chi_square: 0.0,
            chi_square_fraction: 0.0,
            maximum_absolute_weighted_residual: 0.0,
        })
        .collect::<Vec<_>>();
    let mut count = 0;
    let mut adjacent_pair_count = 0;
    let mut residual_sum = 0.0;
    let mut chi_square = 0.0;
    let mut difference_square_sum = 0.0;
    for i in 0..x.len() {
        if !included[i] {
            continue;
        }
        count += 1;
        residual_sum += residual[i];
        let square = weighted[i] * weighted[i];
        chi_square += square;
        let region = edges
            .partition_point(|edge| *edge <= x[i])
            .saturating_sub(1)
            .min(region_count - 1);
        let bucket = &mut regions[region];
        bucket.count += 1;
        bucket.chi_square += square;
        bucket.maximum_absolute_weighted_residual = bucket
            .maximum_absolute_weighted_residual
            .max(weighted[i].abs());
        if i > 0 && included[i - 1] {
            adjacent_pair_count += 1;
            difference_square_sum += (weighted[i] - weighted[i - 1]).powi(2);
        }
    }
    if count == 0 {
        return Err("diagnostics require an included observation");
    }
    if !chi_square.is_finite() || !residual_sum.is_finite() || !difference_square_sum.is_finite() {
        return Err("diagnostic accumulation overflowed");
    }
    if chi_square > 0.0 {
        for region in &mut regions {
            region.chi_square_fraction = region.chi_square / chi_square;
        }
    }
    Ok(ResidualDiagnostics {
        included_count: count,
        adjacent_pair_count,
        chi_square,
        mean: residual_sum / count as f64,
        weighted_rms: (chi_square / count as f64).sqrt(),
        durbin_watson: (chi_square > 0.0 && adjacent_pair_count > 0)
            .then(|| difference_square_sum / chi_square),
        regions,
    })
}

#[cfg(test)]
mod tests {
    use super::diagnose_residuals;

    #[test]
    fn masks_do_not_create_adjacency_and_edges_are_explicit() {
        let r = diagnose_residuals(
            &[0., 1., 2., 3., 4.],
            &[1., 99., -1., 2., 3.],
            &[1., 99., -1., 2., 3.],
            &[true, false, true, true, true],
            2,
        )
        .unwrap();
        assert_eq!(r.included_count, 4);
        assert_eq!(r.adjacent_pair_count, 2);
        assert_eq!(r.regions[0].count, 1);
        assert_eq!(r.regions[1].count, 3);
        assert!((r.chi_square - 15.).abs() < 1e-14);
        assert!((r.durbin_watson.unwrap() - 10. / 15.).abs() < 1e-14);
    }

    #[test]
    fn zero_residual_is_not_a_defined_serial_correlation_statistic() {
        let r = diagnose_residuals(&[0., 1.], &[0., 0.], &[0., 0.], &[true; 2], 3).unwrap();
        assert_eq!(r.durbin_watson, None);
        assert!(r.regions.iter().all(|r| r.chi_square_fraction == 0.));
        assert!(diagnose_residuals(&[0., 1.], &[0.; 2], &[0.; 2], &[false; 2], 2).is_err());
        assert!(diagnose_residuals(&[1., 0.], &[0.; 2], &[0.; 2], &[true; 2], 2).is_err());
        assert!(diagnose_residuals(&[0., 1.], &[1e308; 2], &[1e308; 2], &[true; 2], 2).is_err());
    }
}
