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

impl LearningGeometry {
    // Banks have no cross-bank terms or per-iteration joint normalization. Each
    // complete bank solve can run independently while retaining its exact source
    // order for every scalar sum, across both iterations and worker counts.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn iterate_parallel(
        &self,
        o: &Observations<'_>,
        regions: &[usize],
        total: &[f64],
        reference_weights: &[f64],
        k: usize,
        profile: &mut [f64],
        denominator: &[f64],
        iterations: usize,
        workers: usize,
        budget: usize,
    ) -> Option<(usize, usize)> {
        use rayon::prelude::*;
        if workers <= 1 {
            return None;
        }
        let models = profile.len() / k;
        let extra = (self.rows() + models + 1).checked_mul(size_of::<usize>())?;
        if self.bytes().checked_add(extra)? > budget {
            return None;
        }
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(models + 1).ok()?;
        offsets.resize(models + 1, 0_usize);
        for &id in &self.ids {
            offsets[regions[id] + 2] += 1;
        }
        for i in 0..models {
            offsets[i + 1] += offsets[i];
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.rows()).ok()?;
        rows.resize(self.rows(), 0_usize);
        let extra = (offsets.capacity() + rows.capacity()) * size_of::<usize>();
        if self.bytes().checked_add(extra)? > budget {
            return None;
        }
        let mut cursor = [0_usize; 65];
        cursor[..models].copy_from_slice(&offsets[..models]);
        for (r, &id) in self.ids.iter().enumerate() {
            let bank = regions[id] + 1;
            rows[cursor[bank]] = r;
            cursor[bank] += 1;
        }
        let workers = workers.min(models);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .ok()?;
        pool.install(|| {
            profile.par_chunks_mut(k).enumerate().for_each(|(bank, p)| {
                let count = if bank == 0 {
                    self.rows()
                } else {
                    offsets[bank + 1] - offsets[bank]
                };
                let mut gradient = vec![0.; k];
                for _ in 0..iterations {
                    gradient.fill(0.);
                    for index in 0..count {
                        let r = if bank == 0 {
                            index
                        } else {
                            rows[offsets[bank] + index]
                        };
                        let id = self.ids[r];
                        let value = o.samples[self.samples[r]][0] / total[id];
                        let range = self.offsets[r]..self.offsets[r + 1];
                        let mut expected = 0.;
                        for e in range.clone() {
                            expected += self.weights[e] * p[self.voxels[e] as usize];
                        }
                        for e in range {
                            gradient[self.voxels[e] as usize] +=
                                reference_weights[id] * self.weights[e] * (value - expected);
                        }
                    }
                    for ((p, &d), &g) in p
                        .iter_mut()
                        .zip(&denominator[bank * k..(bank + 1) * k])
                        .zip(&gradient)
                    {
                        if d > 0. {
                            *p = (*p + 0.9 * g / d).max(0.);
                        }
                    }
                }
            });
        });
        Some((workers, extra))
    }
}
