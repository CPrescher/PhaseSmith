//! Explicit intensity symmetry and translation-lattice priors, never an exact space group.
use super::{DiscoveryOptions, Error, Result};
use crate::space_group_search::{self as sg, Assessment, Hypothesis, Rotation};
use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation};
use std::collections::BTreeSet;

const IDENTITY: Rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/// Supplied Laue class and centring in the original index basis. Neither screw/glide
/// translations, inversion of the atomic structure, nor atom counts are specified.
#[derive(Clone, Debug)]
pub struct SymmetryPrior {
    /// Complete closed Laue rotations, including identity and inversion, without duplicates.
    pub laue_rotations: Vec<Rotation>,
    /// Complete pure-translation subgroup, including zero, as exact fractional translations.
    pub centring_translations: Vec<[Rational; 3]>,
    /// Nonempty source/assumption description; measured evidence remains independently reported.
    pub provenance: String,
}

impl SymmetryPrior {
    /// Validate exact closure, basis bounds and compatibility between rotations and centring.
    /// # Errors
    /// Invalid, duplicate, nonclosed or unsupported operations, or missing provenance.
    pub fn processing_group(&self) -> Result<SpaceGroup> {
        let rotations = &self.laue_rotations;
        let translations = &self.centring_translations;
        if self.provenance.trim().is_empty()
            || rotations.is_empty()
            || rotations.len() > 48
            || rotations
                .iter()
                .flatten()
                .flatten()
                .any(|r| r.unsigned_abs() > 8)
            || rotations.iter().copied().collect::<BTreeSet<_>>().len() != rotations.len()
            || !rotations.contains(&IDENTITY)
            || !rotations.contains(&IDENTITY.map(|r| r.map(|v| -v)))
            || translations.is_empty()
            || translations.len() > 4
            || !translations.contains(&[Rational::zero(); 3])
            || translations
                .iter()
                .flatten()
                .any(|t| t.denominator() > 24 || t.modulo_one() != *t || 24 % t.denominator() != 0)
            || translations.iter().collect::<BTreeSet<_>>().len() != translations.len()
        {
            return Err(Error("invalid Laue/centring prior".into()));
        }
        SpaceGroup::new(
            rotations
                .iter()
                .flat_map(|&r| translations.iter().map(move |&t| (r, t)))
                .map(|(r, t)| SymmetryOperation::new(r, t))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error(e.to_string()))?,
        )
        .map_err(|e| Error(format!("invalid Laue/centring prior: {e}")))
    }

    /// Whether a conventional hypothesis has exactly this Laue class and translation lattice.
    /// No proper/acentric point-group choice or nonsymmorphic translation is imposed.
    #[must_use]
    pub fn admits(&self, hypothesis: &Hypothesis) -> bool {
        let mut rotations = self.laue_rotations.clone();
        rotations.sort_unstable();
        sg::laue_rotations(&hypothesis.group) == rotations
            && centring(&hypothesis.group) == self.centring_translations.iter().copied().collect()
    }

    /// Select the supplied processing Laue class while retaining measured evidence.
    /// # Errors
    /// Invalid prior, no matching metric-compatible Laue/centring setting, or
    /// an assessment whose catalogue does not match the supplied hypotheses.
    pub fn apply(&self, assessment: &mut Assessment, hypotheses: &[Hypothesis]) -> Result<()> {
        self.processing_group()?;
        let mut rotations = self.laue_rotations.clone();
        rotations.sort_unstable();
        if assessment.evidence.len() != hypotheses.len()
            || !hypotheses.iter().any(|h| self.admits(h))
            || assessment.evidence.iter().zip(hypotheses).any(|(e, h)| {
                assessment
                    .laue
                    .get(e.laue)
                    .is_none_or(|l| l.rotations != sg::laue_rotations(&h.group))
            })
        {
            return Err(Error(
                "no catalogue setting matches supplied Laue/centring prior".into(),
            ));
        }
        assessment.processing_laue = assessment
            .laue
            .iter()
            .position(|l| l.rotations == rotations)
            .ok_or_else(|| {
                Error("supplied Laue prior is incompatible with the metric/catalogue".into())
            })?;
        Ok(())
    }
}

fn centring(group: &SpaceGroup) -> BTreeSet<[Rational; 3]> {
    group
        .operations()
        .iter()
        .filter(|o| o.rotation() == IDENTITY)
        .map(|o| o.translation())
        .collect()
}

pub(super) fn apply(assessment: &mut Assessment, policy: &DiscoveryOptions) -> Result<()> {
    if let Some(prior) = &policy.symmetry_prior {
        prior.apply(assessment, &policy.hypotheses)?;
        // Preserve the actual supported/insufficient/contradicted evidence status.
        // Supplied priors do not manufacture measured constraints.
    }
    Ok(())
}

pub(super) fn processing_group(
    assessment: &Assessment,
    policy: &DiscoveryOptions,
) -> Result<SpaceGroup> {
    policy.symmetry_prior.as_ref().map_or_else(
        || sg::rotational_group(&assessment.laue[assessment.processing_laue].rotations),
        SymmetryPrior::processing_group,
    )
}

/// Check unresolved higher Laue hypotheses using exact rotational containment.
/// Metric permission and insufficient data must not be treated as rejection.
pub(super) fn unresolved_laue(report: &super::DiscoveryReport, index: usize) -> bool {
    let selected = &report.assessment.laue[report.assessment.evidence[index].laue].rotations;
    report.assessment.laue.iter().any(|l| {
        l.status != "contradicted"
            && l.rotations.len() > selected.len()
            && selected.iter().all(|r| l.rotations.contains(r))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hypothesis(number: i32) -> Hypothesis {
        let h = phasesmith_io::space_group_by_number(number).unwrap();
        Hypothesis {
            number,
            symbol: h.hm_symbol,
            hall: h.hall_symbol,
            group: h.space_group,
        }
    }
    #[test]
    fn prior_does_not_supply_exact_group_or_centrosymmetry_and_preserves_centering() {
        let h = hypothesis(139);
        let p = SymmetryPrior {
            laue_rotations: sg::laue_rotations(&h.group),
            centring_translations: centring(&h.group).into_iter().collect(),
            provenance: "test prior".into(),
        };
        p.processing_group().unwrap();
        assert!(p.admits(&h));
        assert!(p.admits(&hypothesis(107))); // I4mm: acentric, same Laue and I lattice.
        assert!(p.admits(&hypothesis(141))); // I41/amd: a different nonsymmorphic type.
        assert!(!p.admits(&hypothesis(123))); // Primitive 4/mmm.
        assert!(!p.admits(&hypothesis(87))); // I4/m: lower Laue class.
    }
    #[test]
    fn conventional_prior_roundtrips_and_rejects_noncanonical_or_nonclosed_sets() {
        for number in 1..=230 {
            let h = hypothesis(number);
            let p = SymmetryPrior {
                laue_rotations: sg::laue_rotations(&h.group),
                centring_translations: centring(&h.group).into_iter().collect(),
                provenance: "conventional setting".into(),
            };
            p.processing_group().unwrap();
            assert!(p.admits(&h));
        }
        let h = hypothesis(139);
        let mut p = SymmetryPrior {
            laue_rotations: sg::laue_rotations(&h.group),
            centring_translations: centring(&h.group).into_iter().collect(),
            provenance: "test".into(),
        };
        p.laue_rotations.push(IDENTITY);
        assert!(p.processing_group().is_err());
        p.laue_rotations = sg::laue_rotations(&h.group);
        p.centring_translations.push([
            Rational::new(1, 2).unwrap(),
            Rational::zero(),
            Rational::zero(),
        ]);
        assert!(p.processing_group().is_err());
    }
}
