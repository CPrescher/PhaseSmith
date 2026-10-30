//! Actual supplied symmetry/Friedel rows, retaining their multiplicities.
use crate::{SolutionError, error};
use phasesmith_crystallography::SpaceGroup;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(super) struct ReflectionFamilies {
    pub representatives: Vec<usize>,
    pub multiplicities: Vec<usize>,
    pub square_roots: Vec<f64>,
    pub input_rows: usize,
    pub group: SpaceGroup,
}

impl ReflectionFamilies {
    #[allow(clippy::cast_precision_loss)] // Counts never exceed the original 100,000-row bound.
    pub fn new(
        group: &SpaceGroup,
        hkl: &[[i32; 3]],
        observed: &[f64],
    ) -> Result<Self, SolutionError> {
        let mut remaining: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for (i, &h) in hkl.iter().enumerate() {
            remaining.entry(h).or_default().push(i);
        }
        let mut result = Self {
            representatives: Vec::new(),
            multiplicities: Vec::new(),
            square_roots: Vec::new(),
            input_rows: hkl.len(),
            group: group.clone(),
        };
        for (i, &h) in hkl.iter().enumerate() {
            if !remaining.contains_key(&h) {
                continue;
            }
            let mut count = 0;
            for mate in group
                .reflection_orbit(h, true)
                .map_err(|e| error(e.to_string()))?
            {
                if let Some(indices) = remaining.remove(&mate) {
                    for index in indices {
                        if observed[index].to_bits() != observed[i].to_bits() {
                            return Err(error(
                                "multiplicity refinement requires identical amplitudes within each family",
                            ));
                        }
                        count += 1;
                    }
                }
            }
            result.representatives.push(i);
            result.multiplicities.push(count);
            result.square_roots.push((count as f64).sqrt());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_crystallography::SymmetryOperation;
    #[test]
    fn counts_only_supplied_rows_and_preserves_first_member_order() {
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let rows = [[-2, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0]];
        let plan = ReflectionFamilies::new(&group, &rows, &[2., 1., 1., 1., 3.]).unwrap();
        assert_eq!(plan.representatives, vec![0, 1, 4]);
        assert_eq!(plan.multiplicities, vec![1, 3, 1]);
        assert_eq!(plan.input_rows, 5);
    }
    #[test]
    fn unequal_observations_are_not_silently_merged() {
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        assert!(ReflectionFamilies::new(&group, &[[1, 0, 0], [-1, 0, 0]], &[1., 2.]).is_err());
    }
}
