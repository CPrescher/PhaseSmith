//! Support-block analytical Pawley Jacobian products without an observation-by-area matrix.
use crate::PawleyError;
use crate::pawley::err;
use nalgebra::DMatrix;

/// One contiguous derivative support. Global parameters use the entire grid.
#[derive(Clone, Debug)]
pub(crate) struct PawleyColumn {
    pub start: usize,
    pub values: Vec<f64>,
}
impl PawleyColumn {
    pub fn empty() -> Self {
        Self {
            start: 0,
            values: Vec::new(),
        }
    }
    pub fn add_global(&mut self, sample: usize, value: f64, samples: usize) {
        if self.values.is_empty() {
            self.values.resize(samples, 0.0);
        }
        self.values[sample] += value;
    }
}

/// Analytical unweighted products in scaled-free coordinates.
///
/// Physical columns retain native finite support, while a sparse constraint
/// chain maps free coordinates to physical parameters. Values and derivatives
/// come from the same native profile pass; products do not reevaluate profiles.
#[derive(Clone, Debug)]
pub struct PawleyJacobian {
    samples: usize,
    free: usize,
    physical: Vec<PawleyColumn>,
    // Each entry is (physical index, free index, coefficient), in physical order.
    chain: Vec<(usize, usize, f64)>,
}
impl PawleyJacobian {
    pub(crate) fn new(
        samples: usize,
        free: usize,
        physical: Vec<PawleyColumn>,
        chain: &[f64],
    ) -> Result<Self, PawleyError> {
        if chain.len()
            != physical
                .len()
                .checked_mul(free)
                .ok_or_else(|| err("size overflow"))?
            || physical.iter().any(|c| {
                c.start
                    .checked_add(c.values.len())
                    .is_none_or(|end| end > samples)
                    || c.values.iter().any(|v| !v.is_finite())
            })
            || chain.iter().any(|v| !v.is_finite())
        {
            return Err(err("invalid Pawley derivative storage"));
        }
        let chain = chain
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| *v != 0.0)
            .map(|(i, v)| (i / free, i % free, v))
            .collect();
        Ok(Self {
            samples,
            free,
            physical,
            chain,
        })
    }
    /// Number of observations, including excluded rows.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.samples
    }
    /// Number of scaled free coordinates.
    #[must_use]
    pub const fn free_count(&self) -> usize {
        self.free
    }
    /// Stored derivative and nonzero constraint coefficients (excluding integer indices).
    #[must_use]
    pub fn storage_elements(&self) -> usize {
        self.physical.iter().map(|c| c.values.len()).sum::<usize>() + self.chain.len()
    }
    /// Apply the full-grid analytical Jacobian to a finite free-coordinate vector.
    ///
    /// # Errors
    /// Rejects vector shape, nonfinite values and nonfinite products.
    pub fn jvp(&self, vector: &[f64]) -> Result<Vec<f64>, PawleyError> {
        if vector.len() != self.free || vector.iter().any(|v| !v.is_finite()) {
            return Err(err("Pawley JVP requires a finite free-coordinate vector"));
        }
        let mut amplitudes = vec![0.0; self.physical.len()];
        for &(physical, free, coefficient) in &self.chain {
            amplitudes[physical] += coefficient * vector[free];
        }
        let mut result = vec![0.0; self.samples];
        for (column, amplitude) in self.physical.iter().zip(amplitudes) {
            if amplitude != 0.0 {
                for (i, &v) in column.values.iter().enumerate() {
                    result[column.start + i] += amplitude * v;
                }
            }
        }
        finite(result)
    }
    /// Apply the transpose to a finite full-grid vector.
    ///
    /// # Errors
    /// Rejects vector shape, nonfinite values and nonfinite products.
    pub fn vjp(&self, vector: &[f64]) -> Result<Vec<f64>, PawleyError> {
        if vector.len() != self.samples || vector.iter().any(|v| !v.is_finite()) {
            return Err(err("Pawley VJP requires a finite sample vector"));
        }
        let physical: Vec<f64> = self
            .physical
            .iter()
            .map(|column| {
                column
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * vector[column.start + i])
                    .sum()
            })
            .collect();
        let mut result = vec![0.0; self.free];
        for &(p, f, coefficient) in &self.chain {
            result[f] += coefficient * physical[p];
        }
        finite(result)
    }
    /// Materialize the unweighted Jacobian only within an explicit allocation limit.
    ///
    /// # Errors
    /// Rejects element-count overflow and allocations above the limit.
    pub fn materialize(&self, max_elements: usize) -> Result<DMatrix<f64>, PawleyError> {
        if self
            .samples
            .checked_mul(self.free)
            .is_none_or(|n| n > max_elements)
        {
            return Err(err("Pawley dense Jacobian element limit exceeded"));
        }
        let mut result = DMatrix::zeros(self.samples, self.free);
        for &(p, f, coefficient) in &self.chain {
            let column = &self.physical[p];
            for (i, v) in column.values.iter().enumerate() {
                result[(column.start + i, f)] += coefficient * v;
            }
        }
        Ok(result)
    }
    /// Small diagonal Gram blocks used only as an iterative preconditioner.
    /// Does not allocate a sample-by-free-parameter matrix.
    pub(crate) fn gram_blocks(
        &self,
        weights: &[f64],
        norms: &[f64],
        frozen: &[usize],
        block_size: usize,
    ) -> Vec<(usize, DMatrix<f64>)> {
        let mut terms = vec![Vec::new(); self.free];
        for &(p, f, c) in &self.chain {
            terms[f].push((p, c));
        }
        let mut result = Vec::new();
        for start in (0..self.free).step_by(block_size) {
            let size = block_size.min(self.free - start);
            let mut block = DMatrix::zeros(size, size);
            for i in 0..size {
                if frozen.contains(&(start + i)) {
                    continue;
                }
                for j in 0..=i {
                    if frozen.contains(&(start + j)) {
                        continue;
                    }
                    let mut value = 0.0;
                    for &(p, cp) in &terms[start + i] {
                        for &(q, cq) in &terms[start + j] {
                            let a = &self.physical[p];
                            let b = &self.physical[q];
                            let lo = a.start.max(b.start);
                            let hi = (a.start + a.values.len()).min(b.start + b.values.len());
                            let dot: f64 = (lo..hi)
                                .map(|k| {
                                    (a.values[k - a.start] * weights[k])
                                        * (b.values[k - b.start] * weights[k])
                                })
                                .sum();
                            value += cp * cq * dot;
                        }
                    }
                    value /= norms[start + i] * norms[start + j];
                    block[(i, j)] = value;
                    block[(j, i)] = value;
                }
            }
            result.push((start, block));
        }
        result
    }
    /// Exact weighted column norms without retaining a dense Jacobian.
    ///
    /// # Errors
    /// Rejects invalid weights or nonfinite derivative combinations.
    pub fn column_norms(&self, weights: &[f64]) -> Result<Vec<f64>, PawleyError> {
        if weights.len() != self.samples || weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(err("invalid Pawley row weights"));
        }
        let mut result = Vec::with_capacity(self.free);
        let mut column = vec![0.0; self.samples];
        for free in 0..self.free {
            column.fill(0.0);
            for &(p, _, coefficient) in self.chain.iter().filter(|(_, f, _)| *f == free) {
                let support = &self.physical[p];
                for (i, v) in support.values.iter().enumerate() {
                    column[support.start + i] += coefficient * v;
                }
            }
            result.push(
                column
                    .iter()
                    .zip(weights)
                    .map(|(v, w)| (v * w).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
        }
        finite(result)
    }
}
fn finite(values: Vec<f64>) -> Result<Vec<f64>, PawleyError> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(err("nonfinite Pawley derivative product"))
    } else {
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_gram_blocks_match_dense_with_overlapping_ties_and_masks() {
        let operator = PawleyJacobian::new(
            5,
            3,
            vec![
                PawleyColumn {
                    start: 0,
                    values: vec![1.0, 2.0, -1.0],
                },
                PawleyColumn {
                    start: 2,
                    values: vec![3.0, -2.0, 1.0],
                },
                PawleyColumn {
                    start: 0,
                    values: vec![2.0; 5],
                },
            ],
            &[1.0, 0.0, 0.5, -2.0, 1.0, 0.0, 0.0, 0.5, 1.0],
        )
        .unwrap();
        let weights = [1.0, 0.0, 2.0, 0.5, 1.5];
        let norms = operator.column_norms(&weights).unwrap();
        let mut dense = operator.materialize(15).unwrap();
        for i in 0..5 {
            for j in 0..3 {
                dense[(i, j)] *= weights[i] / norms[j];
            }
        }
        dense.column_mut(1).fill(0.0);
        let gram = dense.transpose() * dense;
        for (start, block) in operator.gram_blocks(&weights, &norms, &[1], 2) {
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    assert!((block[(i, j)] - gram[(start + i, start + j)]).abs() < 2e-15);
                }
            }
        }
    }
}
