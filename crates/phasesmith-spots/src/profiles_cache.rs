//! Bounded exact sparse pixel/voxel stencils, local to one immutable learning call.
use super::profiles::{Grid, Observations};

pub(super) struct LearningGeometry {
    samples: Vec<usize>,
    ids: Vec<usize>,
    offsets: Vec<usize>,
    voxels: Vec<u32>,
    weights: Vec<f64>,
}
impl LearningGeometry {
    pub(super) fn bytes(&self) -> usize {
        (self.samples.capacity() + self.ids.capacity() + self.offsets.capacity())
            * size_of::<usize>()
            + self.voxels.capacity() * size_of::<u32>()
            + self.weights.capacity() * size_of::<f64>()
    }
    pub(super) fn entries(&self) -> usize {
        self.weights.len()
    }
    pub(super) fn rows(&self) -> usize {
        self.samples.len()
    }
    // Count first, so Vec growth cannot silently exceed the explicit cache cap.
    // Allocation/admission failure selects the unchanged streaming evaluator.
    pub(super) fn new(
        o: &Observations<'_>,
        grid: Grid,
        centres: &[[f64; 3]],
        used: &[bool],
        budget: usize,
    ) -> Option<Self> {
        if budget == 0 {
            return None;
        }
        let rows: usize = (0..o.ids.len())
            .filter(|&g| used[o.ids[g]])
            .map(|g| o.offsets[g + 1] - o.offsets[g])
            .sum();
        let base = rows
            .checked_mul(3 * size_of::<usize>())?
            .checked_add(size_of::<usize>())?;
        if base > budget {
            return None;
        }
        let maximum = (budget - base) / (size_of::<u32>() + size_of::<f64>());
        let mut entries = 0_usize;
        for g in 0..o.ids.len() {
            let id = o.ids[g];
            if !used[id] {
                continue;
            }
            for i in o.offsets[g]..o.offsets[g + 1] {
                o.weights(grid, o.bounds(g, i), centres[id], id, |_, _, _| {
                    entries = entries.saturating_add(1);
                });
                if entries > maximum {
                    return None;
                }
            }
        }
        let mut cache = Self {
            samples: Vec::new(),
            ids: Vec::new(),
            offsets: Vec::new(),
            voxels: Vec::new(),
            weights: Vec::new(),
        };
        cache.samples.try_reserve_exact(rows).ok()?;
        cache.ids.try_reserve_exact(rows).ok()?;
        cache.offsets.try_reserve_exact(rows + 1).ok()?;
        cache.voxels.try_reserve_exact(entries).ok()?;
        cache.weights.try_reserve_exact(entries).ok()?;
        if cache.bytes() > budget {
            return None;
        }
        cache.offsets.push(0);
        for g in 0..o.ids.len() {
            let id = o.ids[g];
            if !used[id] {
                continue;
            }
            for i in o.offsets[g]..o.offsets[g + 1] {
                cache.samples.push(i);
                cache.ids.push(id);
                o.weights(grid, o.bounds(g, i), centres[id], id, |j, w, _| {
                    // The validated finite grid has at most 300,000 voxels.
                    cache
                        .voxels
                        .push(u32::try_from(j).expect("validated finite voxel index"));
                    cache.weights.push(w);
                });
                cache.offsets.push(cache.weights.len());
            }
        }
        Some(cache)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn initialize(
        &self,
        o: &Observations<'_>,
        regions: &[usize],
        total: &[f64],
        reference_weights: &[f64],
        k: usize,
        profile: &mut [f64],
        denominator: &mut [f64],
    ) {
        for r in 0..self.rows() {
            let id = self.ids[r];
            let value = o.samples[self.samples[r]][0] / total[id];
            let range = self.offsets[r]..self.offsets[r + 1];
            let mut row = 0.;
            for &w in &self.weights[range.clone()] {
                row += w;
            }
            for e in range {
                let j = self.voxels[e] as usize;
                let w = self.weights[e];
                for m in [0, regions[id] + 1] {
                    profile[m * k + j] += reference_weights[id] * w * value;
                    denominator[m * k + j] += reference_weights[id] * w * row;
                }
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gradient(
        &self,
        o: &Observations<'_>,
        regions: &[usize],
        total: &[f64],
        reference_weights: &[f64],
        k: usize,
        profile: &[f64],
        gradient: &mut [f64],
    ) {
        for r in 0..self.rows() {
            let id = self.ids[r];
            let value = o.samples[self.samples[r]][0] / total[id];
            let range = self.offsets[r]..self.offsets[r + 1];
            let mut expected = [0.; 2];
            for e in range.clone() {
                let j = self.voxels[e] as usize;
                let w = self.weights[e];
                expected[0] += w * profile[j];
                expected[1] += w * profile[(regions[id] + 1) * k + j];
            }
            for e in range {
                let j = self.voxels[e] as usize;
                let w = self.weights[e];
                for (slot, m) in [0, regions[id] + 1].into_iter().enumerate() {
                    gradient[m * k + j] += reference_weights[id] * w * (value - expected[slot]);
                }
            }
        }
    }
}
