//! Packed even/odd real transforms; preserve full XYZ spectra at caller boundaries.
use super::layout;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub(super) struct RealTransform {
    shape: [usize; 3],
    work: Vec<Complex<f64>>,
    line: Vec<Complex<f64>>,
    batch: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
    twiddle: Vec<Complex<f64>>,
    opposite_row: Vec<usize>,
}
impl RealTransform {
    pub(super) fn new(
        shape: [usize; 3],
        original_forward: &[Arc<dyn Fft<f64>>; 3],
        original_inverse: &[Arc<dyn Fft<f64>>; 3],
    ) -> Self {
        let [nx, ny, nz] = shape;
        let half = nz / 2;
        let mut planner = FftPlanner::new();
        let forward = [
            original_forward[0].clone(),
            original_forward[1].clone(),
            planner.plan_fft_forward(half),
        ];
        let inverse = [
            original_inverse[0].clone(),
            original_inverse[1].clone(),
            planner.plan_fft_inverse(half),
        ];
        let scratch_size = forward
            .iter()
            .chain(&inverse)
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        #[allow(clippy::cast_precision_loss)]
        let twiddle = (0..=half)
            .map(|k| Complex::from_polar(1., std::f64::consts::TAU * k as f64 / nz as f64))
            .collect();
        Self {
            shape,
            work: vec![Complex::default(); nx * ny * (half + 1)],
            line: vec![Complex::default(); half],
            batch: vec![Complex::default(); layout::BATCH_LINES * nx.max(ny)],
            scratch: vec![Complex::default(); scratch_size],
            forward,
            inverse,
            twiddle,
            opposite_row: (0..nx * ny)
                .map(|row| ((nx - row / ny) % nx) * ny + (ny - row % ny) % ny)
                .collect(),
        }
    }
    fn xy(&mut self, inverse: bool) {
        let [nx, ny, nz] = self.shape;
        let reduced = [nx, ny, nz / 2 + 1];
        // Real-to-complex follows z/y/x. Complex-to-real reverses the axes.
        for axis in if inverse { [1, 0] } else { [0, 1] } {
            let length = reduced[axis];
            let stride: usize = reduced[axis + 1..].iter().product();
            let block = length * stride;
            let plan = if inverse {
                &self.inverse[axis]
            } else {
                &self.forward[axis]
            };
            for outer in 0..self.work.len() / block {
                for inner in (0..stride).step_by(layout::BATCH_LINES) {
                    let lanes = layout::BATCH_LINES.min(stride - inner);
                    let offset = outer * block + inner;
                    let batch = &mut self.batch[..lanes * length];
                    layout::pack(&self.work, batch, offset, stride, length, lanes);
                    plan.process_with_scratch(batch, &mut self.scratch);
                    layout::unpack(batch, &mut self.work, offset, stride, length, lanes);
                }
            }
        }
    }
    // Positive-exponent unnormalized DFT of the real density.
    pub(super) fn density_to_spectrum(&mut self, values: &mut [Complex<f64>]) {
        self.prepare_density(|i| values[i].re);
        self.restore_spectrum(values);
    }
    pub(super) fn density_slice_to_spectrum(
        &mut self,
        density: &[f64],
        values: &mut [Complex<f64>],
    ) {
        self.prepare_density(|i| density[i]);
        self.restore_spectrum(values);
    }
    fn prepare_density(&mut self, read: impl Fn(usize) -> f64) {
        let [nx, ny, nz] = self.shape;
        let half = nz / 2;
        let width = half + 1;
        for row in 0..nx * ny {
            for (k, value) in self.line.iter_mut().enumerate() {
                *value = Complex::new(read(row * nz + 2 * k), read(row * nz + 2 * k + 1));
            }
            self.inverse[2].process_with_scratch(&mut self.line, &mut self.scratch);
            let target = &mut self.work[row * width..(row + 1) * width];
            target[0] = Complex::new(self.line[0].re + self.line[0].im, 0.);
            target[half] = Complex::new(self.line[0].re - self.line[0].im, 0.);
            for (k, out) in target.iter_mut().enumerate().take(half).skip(1) {
                let a = self.line[k];
                let b = self.line[half - k].conj();
                *out = (a + b) * 0.5 + Complex::new(0., -0.5) * self.twiddle[k] * (a - b);
            }
        }
        self.xy(true);
    }
    fn restore_spectrum(&self, values: &mut [Complex<f64>]) {
        let [nx, ny, nz] = self.shape;
        let width = nz / 2 + 1;
        for row in 0..nx * ny {
            let target = &mut values[row * nz..(row + 1) * nz];
            target[..width].copy_from_slice(&self.work[row * width..(row + 1) * width]);
            for (k, out) in target.iter_mut().enumerate().skip(width) {
                *out = self.work[self.opposite_row[row] * width + nz - k].conj();
            }
        }
    }
    // Real part of the negative-exponent DFT, valid even for non-Hermitian input.
    pub(super) fn spectrum_to_density(&mut self, values: &mut [Complex<f64>]) {
        self.prepare_spectrum(values);
        self.write_density(|i, v| values[i] = Complex::new(v, 0.));
    }
    #[allow(clippy::cast_precision_loss)] // Original grid bound makes this conversion exact.
    pub(super) fn spectrum_to_density_slice(
        &mut self,
        values: &[Complex<f64>],
        density: &mut [f64],
    ) {
        self.prepare_spectrum(values);
        let norm = values.len() as f64;
        self.write_density(|i, v| density[i] = v / norm);
    }
    fn prepare_spectrum(&mut self, values: &[Complex<f64>]) {
        let [nx, ny, nz] = self.shape;
        let half = nz / 2;
        let width = half + 1;
        for row in 0..nx * ny {
            for k in 0..width {
                let a = values[row * nz + k];
                let b = values[self.opposite_row[row] * nz + (nz - k) % nz].conj();
                self.work[row * width + k] = a * 0.5 + b * 0.5;
            }
        }
        self.xy(false);
    }
    fn write_density(&mut self, mut write: impl FnMut(usize, f64)) {
        let [nx, ny, nz] = self.shape;
        let half = nz / 2;
        let width = half + 1;
        for row in 0..nx * ny {
            let source = &self.work[row * width..(row + 1) * width];
            self.line[0] = Complex::new(
                source[0].re.midpoint(source[half].re),
                source[0].re.midpoint(-source[half].re),
            );
            for (k, out) in self.line.iter_mut().enumerate().skip(1) {
                let a = source[k];
                let b = source[half - k].conj();
                *out = (a + b) * 0.5 + Complex::new(0., 0.5) * self.twiddle[k].conj() * (a - b);
            }
            self.forward[2].process_with_scratch(&mut self.line, &mut self.scratch);
            for (k, value) in self.line.iter().enumerate() {
                write(row * nz + 2 * k, 2. * value.re);
                write(row * nz + 2 * k + 1, 2. * value.im);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn direct_density_transfers_preserve_every_float_bit() {
        use crate::fourier::Transform;
        for shape in [[8; 3], [10, 14, 18], [32, 16, 64]] {
            let count = shape.iter().product::<usize>();
            let density: Vec<_> = (0..count).map(|i| (i as f64 * 0.137).sin()).collect();
            let mut old: Vec<_> = density.iter().map(|&v| Complex::new(v, 0.)).collect();
            let mut fused = vec![Complex::default(); count];
            let mut first = Transform::real(shape);
            let mut second = Transform::real(shape);
            first.density_to_fourier(&mut old);
            second.density_slice_to_fourier(&density, &mut fused);
            assert_eq!(
                old.iter()
                    .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                    .collect::<Vec<_>>(),
                fused
                    .iter()
                    .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                    .collect::<Vec<_>>()
            );
            // General complex input ensures no implicit Hermitian assumption appears.
            let spectrum: Vec<_> = (0..count)
                .map(|i| Complex::new((i as f64 * 0.219).sin(), (i as f64 * 0.331).cos()))
                .collect();
            let mut control = spectrum.clone();
            let mut direct = vec![0.; count];
            first.fourier_to_density(&mut control);
            second.fourier_to_density_slice(&spectrum, &mut direct);
            assert_eq!(
                control
                    .iter()
                    .map(|v| (v.re / count as f64).to_bits())
                    .collect::<Vec<_>>(),
                direct.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn maximum_grid_workspace_is_bounded_by_half_spectrum() {
        let mut planner = FftPlanner::new();
        let forward = [128; 3].map(|n| planner.plan_fft_forward(n));
        let inverse = [128; 3].map(|n| planner.plan_fft_inverse(n));
        let plan = RealTransform::new([128; 3], &forward, &inverse);
        assert_eq!(plan.work.len(), 128 * 128 * 65);
        let bytes = (plan.work.len()
            + plan.line.len()
            + plan.batch.len()
            + plan.scratch.len()
            + plan.twiddle.len())
            * size_of::<Complex<f64>>()
            + plan.opposite_row.len() * size_of::<usize>();
        assert!(bytes < 18 * 1024 * 1024);
    }
}
