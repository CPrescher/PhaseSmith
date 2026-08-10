//! Quantitative crystalline phase fractions from compatible Rietveld scales.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Metadata required by the Hill--Howard scale relation.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantitativePhase {
    /// Stable phase identity.
    pub phase_id: String,
    /// Compatible non-negative refined phase scale.
    pub scale: f64,
    /// Formula units per crystallographic unit cell.
    pub formula_units_per_cell: f64,
    /// Formula mass in grams per mole.
    pub formula_mass_g_mol: f64,
    /// Unit-cell volume in cubic ångströms.
    pub cell_volume_angstrom3: f64,
}

impl QuantitativePhase {
    /// Validate one quantitative phase record.
    ///
    /// # Errors
    ///
    /// Returns [`QuantitativeError`] for empty identity, negative scale, or
    /// non-positive/non-finite physical metadata.
    pub fn new(
        phase_id: impl Into<String>,
        scale: f64,
        formula_units_per_cell: f64,
        formula_mass_g_mol: f64,
        cell_volume_angstrom3: f64,
    ) -> Result<Self, QuantitativeError> {
        let result = Self {
            phase_id: phase_id.into(),
            scale,
            formula_units_per_cell,
            formula_mass_g_mol,
            cell_volume_angstrom3,
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<(), QuantitativeError> {
        if self.phase_id.is_empty() {
            return Err(QuantitativeError::EmptyPhaseId);
        }
        if !self.scale.is_finite() || self.scale < 0.0 {
            return Err(QuantitativeError::InvalidScale);
        }
        if [
            self.formula_units_per_cell,
            self.formula_mass_g_mol,
            self.cell_volume_angstrom3,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(QuantitativeError::InvalidMetadata);
        }
        Ok(())
    }
}

/// One normalized crystalline weight fraction.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseWeightFraction {
    /// Stable phase identity.
    pub phase_id: String,
    /// Fraction of the supplied crystalline phases in `[0, 1]`.
    pub weight_fraction: f64,
}

/// Weight fractions and their row-major covariance propagated from phase scales.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantitativePhaseAnalysis {
    /// Normalized crystalline weight fractions in input order.
    pub phases: Vec<PhaseWeightFraction>,
    /// Row-major fraction covariance with dimension `phases.len()`.
    pub covariance: Vec<f64>,
}

/// Calculate Hill--Howard crystalline weight fractions in input order.
///
/// `W_p = S_p (Z M V)_p / sum_i S_i (Z M V)_i`.
///
/// # Errors
///
/// Returns [`QuantitativeError`] for empty, duplicate, invalid, overflowing,
/// or all-zero phase records.
pub fn quantitative_phase_analysis(
    phases: &[QuantitativePhase],
) -> Result<Vec<PhaseWeightFraction>, QuantitativeError> {
    if phases.is_empty() {
        return Err(QuantitativeError::EmptyPhases);
    }
    let mut identities = std::collections::BTreeSet::new();
    let mut contributions = Vec::with_capacity(phases.len());
    for phase in phases {
        phase.validate()?;
        if !identities.insert(&phase.phase_id) {
            return Err(QuantitativeError::DuplicatePhaseId);
        }
        let contribution = phase.scale
            * phase.formula_units_per_cell
            * phase.formula_mass_g_mol
            * phase.cell_volume_angstrom3;
        if !contribution.is_finite() {
            return Err(QuantitativeError::ContributionOverflow);
        }
        contributions.push(contribution);
    }
    let total = contributions.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(QuantitativeError::ZeroTotal);
    }
    Ok(phases
        .iter()
        .zip(contributions)
        .map(|(phase, contribution)| PhaseWeightFraction {
            phase_id: phase.phase_id.clone(),
            weight_fraction: contribution / total,
        })
        .collect())
}

/// Calculate weight fractions and analytically propagate a phase-scale covariance.
///
/// For `c_i = S_i (Z M V)_i`, `W_i = c_i / sum(c)`, the scale derivative is
/// `dW_i/dS_j = (delta_ij k_i - W_i k_j) / sum(c)`, where `k_i = (Z M V)_i`.
///
/// # Errors
///
/// Returns [`QuantitativeError`] for invalid phases or covariance shape/values.
pub fn quantitative_phase_analysis_with_covariance(
    phases: &[QuantitativePhase],
    scale_covariance: &[f64],
) -> Result<QuantitativePhaseAnalysis, QuantitativeError> {
    let fractions = quantitative_phase_analysis(phases)?;
    let count = phases.len();
    if scale_covariance.len()
        != count
            .checked_mul(count)
            .ok_or(QuantitativeError::CovarianceShape)?
    {
        return Err(QuantitativeError::CovarianceShape);
    }
    if scale_covariance.iter().any(|value| !value.is_finite()) {
        return Err(QuantitativeError::NonFiniteCovariance);
    }
    let factors = phases
        .iter()
        .map(|phase| {
            phase.formula_units_per_cell * phase.formula_mass_g_mol * phase.cell_volume_angstrom3
        })
        .collect::<Vec<_>>();
    let total = phases
        .iter()
        .zip(&factors)
        .map(|(phase, factor)| phase.scale * factor)
        .sum::<f64>();
    let mut jacobian = vec![0.0; count * count];
    for row in 0..count {
        for column in 0..count {
            jacobian[row * count + column] = (if row == column { factors[row] } else { 0.0 }
                - fractions[row].weight_fraction * factors[column])
                / total;
        }
    }
    let mut covariance = vec![0.0; count * count];
    for row in 0..count {
        for column in 0..count {
            let mut value = 0.0;
            for left in 0..count {
                for right in 0..count {
                    value += jacobian[row * count + left]
                        * scale_covariance[left * count + right]
                        * jacobian[column * count + right];
                }
            }
            if !value.is_finite() {
                return Err(QuantitativeError::NonFiniteCovariance);
            }
            covariance[row * count + column] = value;
        }
    }
    Ok(QuantitativePhaseAnalysis {
        phases: fractions,
        covariance,
    })
}

/// Invalid quantitative-phase input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantitativeError {
    /// At least one phase is required.
    EmptyPhases,
    /// Phase identity must not be empty.
    EmptyPhaseId,
    /// Phase identities must be unique.
    DuplicatePhaseId,
    /// Scale must be finite and non-negative.
    InvalidScale,
    /// Z, mass, and cell volume must be positive and finite.
    InvalidMetadata,
    /// A scale-times-metadata product overflowed.
    ContributionOverflow,
    /// At least one phase scale must be positive.
    ZeroTotal,
    /// Phase-scale covariance did not have square phase dimension.
    CovarianceShape,
    /// Phase-scale covariance or propagated covariance was not finite.
    NonFiniteCovariance,
}

impl Display for QuantitativeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::EmptyPhases => "quantitative phase analysis requires at least one phase",
            Self::EmptyPhaseId => "quantitative phase IDs must not be empty",
            Self::DuplicatePhaseId => "quantitative phase IDs must be unique",
            Self::InvalidScale => "quantitative phase scales must be finite and non-negative",
            Self::InvalidMetadata => "quantitative Z, mass, and volume must be positive and finite",
            Self::ContributionOverflow => "quantitative phase contribution overflowed",
            Self::ZeroTotal => "at least one quantitative phase scale must be positive",
            Self::CovarianceShape => "scale covariance must be square with one row per phase",
            Self::NonFiniteCovariance => {
                "scale covariance and propagated covariance must be finite"
            }
        })
    }
}

impl Error for QuantitativeError {}
