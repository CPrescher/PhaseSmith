//! Explicit admission bounds for dense f64 numerical buffers, separate from caches.

use phasesmith_crystallography::{SpaceGroup, UnitCell};

use crate::{Observation, SolutionError, error, prepare_data};

/// Historical dense-buffer ceiling, in bytes (two million f64 elements).
pub const DEFAULT_DENSE_MEMORY_BUDGET_BYTES: usize = 16_000_000;
/// Largest supported explicit dense-buffer ceiling (one GiB per buffer).
pub const MAX_DENSE_MEMORY_BUDGET_BYTES: usize = 1_073_741_824;

/// Checked per-buffer dense allocation policy; not a total process-memory limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseMemoryBudget(usize);

impl Default for DenseMemoryBudget {
    fn default() -> Self {
        Self(DEFAULT_DENSE_MEMORY_BUDGET_BYTES)
    }
}

impl DenseMemoryBudget {
    /// Construct a policy with a byte limit in [8, 1 GiB].
    /// # Errors
    /// Rejects zero/sub-element and unsupported budgets.
    pub fn new(bytes: usize) -> Result<Self, SolutionError> {
        if !(size_of::<f64>()..=MAX_DENSE_MEMORY_BUDGET_BYTES).contains(&bytes) {
            return Err(error(
                "dense_memory_budget_bytes must be in [8, 1073741824]",
            ));
        }
        Ok(Self(bytes))
    }

    /// Maximum admitted bytes for each controlled dense f64 buffer.
    #[must_use]
    pub fn bytes(self) -> usize {
        self.0
    }

    /// Check a dense shape before allocation; equality to the byte limit is valid.
    /// # Errors
    /// Rejects overflowing dimensions or a request exceeding the explicit budget.
    pub fn check(self, rows: usize, columns: usize) -> Result<(), SolutionError> {
        let required = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(size_of::<f64>()))
            .ok_or_else(|| error("dense memory shape overflows the allocation size"))?;
        if required > self.0 {
            return Err(error(format!(
                "dense memory budget exceeded: requires {required} bytes, configured {} bytes",
                self.0
            )));
        }
        Ok(())
    }

    /// Conservative P1 derivative/scattering envelope: `(5*N + 7)*M` f64 values.
    /// # Errors
    /// Rejects overflow or an envelope beyond the configured limit.
    pub fn check_model(self, atoms: usize, reflections: usize) -> Result<(), SolutionError> {
        let rows = atoms
            .checked_mul(5)
            .and_then(|n| n.checked_add(7))
            .ok_or_else(|| error("dense model atom count overflows"))?;
        self.check(rows, reflections)
    }
}

/// Input-only full-cell search admission estimate; no structure/count prior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolutionMemoryEstimate {
    /// Allowed symmetry/Friedel-expanded unique measured reciprocal rows.
    pub expanded_reflection_count: usize,
    /// Existing volume-derived cap, clipped to the supported 128-atom ceiling.
    pub volume_atom_cap: usize,
    /// Cap from the conservative dense buffer envelope, clipped to 128 atoms.
    pub memory_atom_cap: usize,
    /// Minimum of volume, memory and supported atom caps; zero means no atom fits.
    pub maximum_atoms: usize,
    /// Configured per-buffer byte ceiling.
    pub dense_memory_budget_bytes: usize,
}

/// Estimate admissible search capacity from observations, cell and supplied symmetry.
/// # Errors
/// Rejects invalid input data, cell/symmetry or a malformed byte budget.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn estimate_solution_memory(
    observations: &[Observation],
    cell: UnitCell,
    group: &SpaceGroup,
    budget: DenseMemoryBudget,
) -> Result<SolutionMemoryEstimate, SolutionError> {
    let data = prepare_data(observations, cell, group)?;
    let reflections = data.hkl.len();
    let elements = budget.bytes() / size_of::<f64>();
    let memory_atom_cap = (elements / reflections).saturating_sub(7) / 5;
    let volume_atom_cap = (cell
        .geometry()
        .map_err(|e| error(e.to_string()))?
        .volume_angstrom3
        / 8.)
        .ceil()
        .clamp(1., 128.) as usize;
    Ok(SolutionMemoryEstimate {
        expanded_reflection_count: reflections,
        volume_atom_cap,
        memory_atom_cap: memory_atom_cap.min(128),
        maximum_atoms: memory_atom_cap.min(volume_atom_cap).min(128),
        dense_memory_budget_bytes: budget.bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inclusive_bytes_partial_elements_and_checked_shapes() {
        let budget = DenseMemoryBudget::new(17).unwrap();
        assert!(budget.check(1, 2).is_ok());
        assert!(DenseMemoryBudget::new(16).unwrap().check(1, 2).is_ok());
        assert!(
            budget
                .check(1, 3)
                .unwrap_err()
                .0
                .contains("requires 24 bytes")
        );
        assert!(budget.check(usize::MAX, 2).is_err());
        assert!(budget.check_model(usize::MAX, 1).is_err());
        for bytes in [0, 7, MAX_DENSE_MEMORY_BUDGET_BYTES + 1] {
            assert!(DenseMemoryBudget::new(bytes).is_err());
        }
    }
}
