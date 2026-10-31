//! Bounded original-u32 frame preparation, with exact source hashes and masks.
use crate::{
    SpotError,
    frame_validity::{self, DacOptions, Shape},
    invalid,
    reciprocal::Geometry,
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

/// One immutable mask sequence expanded with whole-block exclusions.
pub struct BlockMasks {
    /// Block side in detector pixels.
    pub block: usize,
    /// Frame-major row-major coarse booleans.
    pub excluded: Vec<bool>,
    /// Optional full-resolution fixed mask.
    pub fixed: Vec<bool>,
}
/// Explicit affine full-exposure DAC mask sequence.
pub struct DacMasks {
    /// Detector geometry, unchanged from the ordinary mask kernel.
    pub geometry: Geometry,
    /// Ordered angular intervals, degrees.
    pub bounds: Vec<[f64; 2]>,
    /// Pixel block side.
    pub block: usize,
    /// Conical aperture parameters.
    pub options: DacOptions,
    /// Optional full-resolution fixed mask.
    pub fixed: Vec<bool>,
}
/// Immutable scan-specific source/mask plan, independent of execution resources.
pub struct Plan {
    /// Detector rows and columns.
    pub shape: [usize; 2],
    /// Selected frame count.
    pub frames: usize,
    /// Explicit source orientation.
    pub vertical_flip: bool,
    /// Inclusive detector-reserved value cutoff.
    pub reserved_minimum: u32,
    /// Expected image SHA256 per selected frame, one sequence per source provider.
    pub expected_hashes: Vec<Vec<String>>,
    /// Inferred masks, including any fixed exclusions.
    pub blocks: Vec<BlockMasks>,
    /// Explicit geometric masks.
    pub dacs: Vec<DacMasks>,
}
/// Owned original decoder output, in native detector orientation.
pub struct Input {
    /// Zero-based index in the selected scan.
    pub index: usize,
    /// Original integer counts, including detector-reserved values.
    pub pixels: Vec<u32>,
}
/// Oriented original counts and one composed unavailable-pixel mask.
pub struct Prepared {
    /// Selected scan index.
    pub index: usize,
    /// Original integer counts; no interpolation or normalization.
    pub pixels: Vec<u32>,
    /// True excludes a pixel.
    pub excluded: Vec<bool>,
    /// Digest computed on this oriented native-endian u32 payload.
    pub digest: String,
}

fn apply_mask(output: &mut [bool], width: usize, block: usize, coarse: &[bool], fixed: &[bool]) {
    let cw = width.div_ceil(block);
    for (y, row) in output.chunks_exact_mut(width).enumerate() {
        let first = (y / block) * cw;
        for (pixels, &excluded) in row.chunks_mut(block).zip(&coarse[first..first + cw]) {
            if excluded {
                pixels.fill(true);
            }
        }
    }
    if !fixed.is_empty() {
        for (value, &excluded) in output.iter_mut().zip(fixed) {
            *value |= excluded;
        }
    }
}

impl Plan {
    /// Validate shapes, mask/identity sequences and explicit geometry.
    /// # Errors
    /// Rejects inconsistent or excessive input.
    pub fn validate(&self) -> Result<(), SpotError> {
        let n = self.shape[0]
            .checked_mul(self.shape[1])
            .filter(|&n| n > 0 && n <= 100_000_000)
            .ok_or_else(|| invalid("invalid prepared-frame dimensions"))?;
        if self.frames == 0
            || self.frames > 1_000_000
            || self.blocks.len() + self.dacs.len() > 32
            || self.expected_hashes.len() > 32
        {
            return Err(invalid("invalid frame/mask count"));
        }
        for hashes in &self.expected_hashes {
            if hashes.len() != self.frames
                || hashes
                    .iter()
                    .any(|s| s.len() != 64 || !s.bytes().all(|v| v.is_ascii_hexdigit()))
            {
                return Err(invalid("invalid frame source-hash sequence"));
            }
        }
        for mask in &self.blocks {
            let [h, w] = Shape {
                height: self.shape[0],
                width: self.shape[1],
                block: mask.block,
            }
            .coarse()
            .map_err(|e| invalid(&e.to_string()))?;
            if h.checked_mul(w)
                .and_then(|n| n.checked_mul(self.frames))
                .is_none_or(|n| n > frame_validity::MAX_CELLS || n != mask.excluded.len())
                || (!mask.fixed.is_empty() && mask.fixed.len() != n)
            {
                return Err(invalid("invalid prepared-frame block mask"));
            }
        }
        for dac in &self.dacs {
            if dac.bounds.len() != self.frames || (!dac.fixed.is_empty() && dac.fixed.len() != n) {
                return Err(invalid("invalid prepared-frame DAC sequence"));
            }
            // Validate the same kernel/closed support before any frame is admitted.
            frame_validity::dac_mask(
                dac.geometry,
                dac.bounds[0],
                Shape {
                    height: self.shape[0],
                    width: self.shape[1],
                    block: dac.block,
                },
                dac.options,
            )
            .map_err(|e| invalid(&e.to_string()))?;
        }
        Ok(())
    }
    fn prepare(&self, mut input: Input) -> Result<Prepared, SpotError> {
        let [height, width] = self.shape;
        if input.index >= self.frames || input.pixels.len() != height * width {
            return Err(invalid("prepared source frame shape/index differs"));
        }
        if self.vertical_flip {
            for row in 0..height / 2 {
                let opposite = (height - 1 - row) * width;
                let (before, after) = input.pixels.split_at_mut(opposite);
                before[row * width..(row + 1) * width].swap_with_slice(&mut after[..width]);
            }
        }
        let mut hash = Sha256::new();
        let mut bytes = [0_u8; 4096];
        for chunk in input.pixels.chunks(1024) {
            for (value, out) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
                out.copy_from_slice(&value.to_ne_bytes());
            }
            hash.update(&bytes[..chunk.len() * 4]);
        }
        let digest = format!("{:x}", hash.finalize());
        if self
            .expected_hashes
            .iter()
            .any(|hashes| hashes[input.index] != digest)
        {
            return Err(invalid("source frame hash differs from validity snapshot"));
        }
        let mut excluded: Vec<_> = input
            .pixels
            .iter()
            .map(|&v| v >= self.reserved_minimum)
            .collect();
        for mask in &self.blocks {
            let w = width.div_ceil(mask.block);
            let frame_cells = height.div_ceil(mask.block) * w;
            apply_mask(
                &mut excluded,
                width,
                mask.block,
                &mask.excluded[input.index * frame_cells..(input.index + 1) * frame_cells],
                &mask.fixed,
            );
        }
        for dac in &self.dacs {
            let mask = frame_validity::dac_mask(
                dac.geometry,
                dac.bounds[input.index],
                Shape {
                    height,
                    width,
                    block: dac.block,
                },
                dac.options,
            )
            .map_err(|e| invalid(&e.to_string()))?;
            apply_mask(&mut excluded, width, dac.block, &mask.excluded, &dac.fixed);
        }
        Ok(Prepared {
            index: input.index,
            pixels: input.pixels,
            excluded,
            digest,
        })
    }
    /// Prepare independent owned frames in a private native pool, source order.
    ///
    /// The cap covers source snapshots, owned counts, output masks, a temporary
    /// DAC mask and 4096 digest scratch bytes per active worker. Plan storage,
    /// downstream detector workspace and thread stacks are separate.
    /// # Errors
    /// Invalid controls, allocation budget, source identity or mask geometry.
    pub fn prepare_batch(
        &self,
        frames: Vec<Input>,
        workers: usize,
        budget: usize,
    ) -> Result<Vec<Result<Prepared, SpotError>>, SpotError> {
        self.validate()?;
        self.admit(frames.len(), workers, budget)?;
        let count = frames.len();
        if workers == 1 || count <= 1 {
            return Ok(frames.into_iter().map(|f| self.prepare(f)).collect());
        }
        Ok(rayon::ThreadPoolBuilder::new()
            .num_threads(workers.min(count))
            .build()
            .map_err(|_| invalid("could not create frame preparation pool"))?
            .install(|| frames.into_par_iter().map(|f| self.prepare(f)).collect()))
    }
    /// Check the bounded batch payload before copying input arrays.
    /// # Errors
    /// Invalid execution controls or a batch exceeding its declared payload cap.
    pub fn admit(&self, count: usize, workers: usize, budget: usize) -> Result<(), SpotError> {
        if !(1..=32).contains(&workers) || budget > 1024 * 1024 * 1024 {
            return Err(invalid("invalid frame preparation workers/payload budget"));
        }
        let bytes = self.shape[0]
            .checked_mul(self.shape[1])
            .and_then(|n| n.checked_mul(10))
            .and_then(|n| n.checked_mul(count))
            .and_then(|n| n.checked_add(workers.min(count) * 4096));
        if bytes.is_none_or(|n| n > budget) {
            return Err(invalid("prepared-frame payload budget exceeded"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_preparation_has_ordered_identity_errors_and_explicit_budget() {
        // SHA256 of four zero bytes, independent of machine byte order.
        let expected = "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119";
        let plan = Plan {
            shape: [1, 1],
            frames: 3,
            vertical_flip: false,
            reserved_minimum: 0,
            expected_hashes: vec![vec![expected.into(); 3]],
            blocks: vec![],
            dacs: vec![],
        };
        for workers in [1, 2, 4, 10] {
            let frames = (0..3)
                .map(|index| Input {
                    index,
                    pixels: vec![u32::from(index == 2)],
                })
                .collect();
            let rows = plan.prepare_batch(frames, workers, 1024 * 1024).unwrap();
            for (i, row) in rows.iter().enumerate().take(2) {
                let row = row.as_ref().unwrap();
                assert_eq!(row.index, i);
                assert_eq!(row.pixels, [0]);
                assert_eq!(row.excluded, [true]);
                assert_eq!(row.digest, expected);
            }
            assert!(
                rows[2]
                    .as_ref()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("hash differs")
            );
        }
        assert!(plan.admit(4, 4, 1).is_err());
        assert!(plan.admit(4, 0, 1024 * 1024).is_err());
    }
}
