//! Call-local reuse of exact categorical fields and fused shift derivatives.
use super::super::PreparedPeakCompletion;
use rustfft::num_complex::Complex;
use std::mem::size_of;

pub(super) struct Columns {
    values: Vec<Complex<f64>>,
    rows: usize,
    width: usize,
}
impl Columns {
    pub(super) fn new(plan: &PreparedPeakCompletion) -> Self {
        Self::with_limit(plan, 4 * 1024 * 1024)
    }
    pub(super) fn with_limit(plan: &PreparedPeakCompletion, bytes: usize) -> Self {
        let rows = plan.observed.len();
        let width = plan.elements.len();
        let length = plan.points.len() * width * rows;
        let mut values = Vec::new();
        if length <= bytes / size_of::<Complex<f64>>() {
            values.reserve_exact(length);
            for phase in &plan.phase {
                for e in 0..width {
                    for (h, &p) in phase.iter().enumerate() {
                        values.push(p * plan.factors[h * width + e]);
                    }
                }
            }
        }
        Self {
            values,
            rows,
            width,
        }
    }
    pub(super) fn value(
        &self,
        plan: &PreparedPeakCompletion,
        site: usize,
        label: usize,
        h: usize,
    ) -> Complex<f64> {
        if self.values.is_empty() {
            plan.phase[site][h] * plan.factors[h * self.width + label]
        } else {
            self.values[(site * self.width + label) * self.rows + h]
        }
    }
    pub(super) fn bytes(&self) -> usize {
        self.values.len() * size_of::<Complex<f64>>()
    }
}
struct Field {
    key: Option<u64>,
    stamp: usize,
    total: Vec<Complex<f64>>,
    amplitude: Vec<f64>,
    derivative: Vec<f64>,
    square: f64,
    scale: f64,
    value: f64,
    full: Option<(f64, f64, f64)>,
}
impl Field {
    fn new(rows: usize) -> Self {
        Self {
            key: None,
            stamp: 0,
            total: vec![Complex::default(); rows],
            amplitude: vec![0.; rows],
            derivative: vec![0.; rows],
            square: 0.,
            scale: 0.,
            value: 0.,
            full: None,
        }
    }
}
pub(super) struct Workspace {
    fields: [Field; 2],
    trial: Vec<Complex<f64>>,
    amplitude: Vec<f64>,
    norm: f64,
    tick: usize,
    pub(super) builds: usize,
    pub(super) reuses: usize,
    pub(super) score_only: usize,
}
impl Workspace {
    pub(super) fn new(plan: &PreparedPeakCompletion) -> Self {
        let rows = plan.observed.len();
        Self {
            fields: std::array::from_fn(|_| Field::new(rows)),
            trial: vec![Complex::default(); rows],
            amplitude: vec![0.; rows],
            norm: plan.observed.iter().map(|o| o * o).sum(),
            tick: 0,
            builds: 0,
            reuses: 0,
            score_only: 0,
        }
    }
    pub(super) fn bytes(&self) -> usize {
        self.trial.len() * (3 * size_of::<Complex<f64>>() + 5 * size_of::<f64>())
    }
    pub(super) fn begin(&mut self) {
        for field in &mut self.fields {
            field.key = None;
            field.full = None;
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn profile(
        &mut self,
        plan: &PreparedPeakCompletion,
        columns: &Columns,
        rest: &[Complex<f64>],
        site: usize,
        label: Option<usize>,
        shift: f64,
        full: bool,
    ) -> (f64, f64, f64) {
        if !full {
            self.score_only += 1;
        }
        self.tick += 1;
        let index = if let Some(i) = self
            .fields
            .iter()
            .position(|f| f.key == Some(shift.to_bits()))
        {
            self.reuses += 1;
            i
        } else {
            self.builds += 1;
            let i = self
                .fields
                .iter()
                .position(|f| f.key.is_none())
                .unwrap_or_else(|| usize::from(self.fields[1].stamp < self.fields[0].stamp));
            let field = &mut self.fields[i];
            let beta = 8. * std::f64::consts::PI.powi(2);
            for (h, &background) in rest.iter().enumerate() {
                let column = label.map_or(Complex::default(), |e| {
                    columns.value(plan, site, e, h) * (-beta * shift * plan.s_squared[h]).exp()
                });
                let total = background + column;
                let a = total.norm();
                let da = if a > 1e-100 {
                    (total.conj() * column).re * (-beta * plan.s_squared[h]) / a
                } else {
                    0.
                };
                field.total[h] = total;
                field.amplitude[h] = a;
                field.derivative[h] = da;
            }
            field.square = field.amplitude.iter().map(|a| a * a).sum();
            field.value = f64::INFINITY;
            field.full = None;
            if field.square > 1e-250 {
                field.scale = field
                    .amplitude
                    .iter()
                    .zip(&plan.observed)
                    .map(|(a, o)| a * o)
                    .sum::<f64>()
                    / field.square;
                field.value = 0.;
                for (&a, &o) in field.amplitude.iter().zip(&plan.observed) {
                    let r = field.scale * a - o;
                    field.value += r * r / self.norm;
                }
            }
            field.key = Some(shift.to_bits());
            i
        };
        let field = &mut self.fields[index];
        field.stamp = self.tick;
        if !full {
            return (field.value, 0., 0.);
        }
        if let Some(value) = field.full {
            return value;
        }
        if field.square <= 1e-250 {
            return (f64::INFINITY, 0., 0.);
        }
        let scale_derivative = field
            .derivative
            .iter()
            .zip(&plan.observed)
            .zip(&field.amplitude)
            .map(|((da, o), a)| da * (o - 2. * field.scale * a))
            .sum::<f64>()
            / field.square;
        let (mut gradient, mut curvature) = (0., 0.);
        for ((&a, &da), &o) in field
            .amplitude
            .iter()
            .zip(&field.derivative)
            .zip(&plan.observed)
        {
            let r = field.scale * a - o;
            let dr = field.scale * da + scale_derivative * a;
            gradient += 2. * r * dr / self.norm;
            curvature += 2. * dr * dr / self.norm;
        }
        let result = (field.value, gradient, curvature);
        field.full = Some(result);
        result
    }
    // Preserve the categorical scorer's norm-squared denominator and final division.
    fn fixed(total: &[Complex<f64>], amplitude: &[f64], observed: &[f64], norm: f64) -> f64 {
        let square: f64 = total.iter().map(Complex::norm_sqr).sum();
        if square <= 1e-250 {
            return f64::INFINITY;
        }
        let scale = amplitude
            .iter()
            .zip(observed)
            .map(|(a, o)| a * o)
            .sum::<f64>()
            / square;
        amplitude
            .iter()
            .zip(observed)
            .map(|(a, o)| (scale * a - o).powi(2))
            .sum::<f64>()
            / norm
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fixed_trial(
        &mut self,
        plan: &PreparedPeakCompletion,
        columns: &Columns,
        rest: &[Complex<f64>],
        site: usize,
        label: Option<usize>,
        shift: f64,
    ) -> f64 {
        if let Some(field) = self.fields.iter().find(|f| f.key == Some(shift.to_bits())) {
            self.reuses += 1;
            return Self::fixed(&field.total, &field.amplitude, &plan.observed, self.norm);
        }
        self.builds += 1;
        let beta = 8. * std::f64::consts::PI.powi(2);
        for (h, &background) in rest.iter().enumerate() {
            let column = label.map_or(Complex::default(), |e| {
                columns.value(plan, site, e, h)
                    * (-beta * shift * plan.s_squared[h]).clamp(-600., 600.).exp()
            });
            self.trial[h] = background + column;
            self.amplitude[h] = self.trial[h].norm();
        }
        Self::fixed(&self.trial, &self.amplitude, &plan.observed, self.norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> PreparedPeakCompletion {
        let cell = phasesmith_crystallography::UnitCell {
            a_angstrom: 6.,
            b_angstrom: 7.,
            c_angstrom: 8.,
            alpha_deg: 90.,
            beta_deg: 100.,
            gamma_deg: 90.,
        };
        let group = phasesmith_crystallography::SpaceGroup::new(vec![
            phasesmith_crystallography::SymmetryOperation::identity(),
        ])
        .unwrap();
        let mut hkl = Vec::new();
        for h in -3..=3 {
            for k in -3..=3 {
                for l in -3..=3 {
                    if [h, k, l] != [0; 3] {
                        hkl.push([h, k, l]);
                    }
                }
            }
        }
        let observed: Vec<_> = (0..hkl.len())
            .map(|i| 1. + f64::from(u32::try_from(i % 23).unwrap()) / 13.)
            .collect();
        PreparedPeakCompletion::new(
            cell,
            &group,
            &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.3, 0.1, 0.7]],
            &["C".into(), "N".into()],
            &hkl,
            &observed,
            8,
        )
        .unwrap()
        .with_site_displacements(&[0.02; 3])
        .unwrap()
    }
    fn bits(a: (f64, f64, f64), b: (f64, f64, f64)) {
        assert_eq!(
            [a.0.to_bits(), a.1.to_bits(), a.2.to_bits()],
            [b.0.to_bits(), b.1.to_bits(), b.2.to_bits()]
        );
    }
    #[test]
    fn fields_derivatives_and_categorical_scores_preserve_all_bits() {
        let plan = plan();
        for cap in [0, 4 * 1024 * 1024] {
            let columns = Columns::with_limit(&plan, cap);
            assert!(columns.bytes() <= cap);
            let mut ws = Workspace::new(&plan);
            for background in [0.13, 0.47] {
                let rest: Vec<_> = (0..plan.observed.len())
                    .map(|i| {
                        Complex::new(
                            background + f64::from(u32::try_from(i % 11).unwrap()) / 13.,
                            -0.31,
                        )
                    })
                    .collect();
                for label in [None, Some(0), Some(1)] {
                    ws.begin();
                    for shift in [-0.019, 0., 0.011, 0.16] {
                        let expected = plan.shift_profile(&rest, 1, label, shift);
                        assert_eq!(
                            ws.profile(&plan, &columns, &rest, 1, label, shift, false)
                                .0
                                .to_bits(),
                            expected.0.to_bits()
                        );
                        bits(
                            ws.profile(&plan, &columns, &rest, 1, label, shift, true),
                            expected,
                        );
                        bits(
                            ws.profile(&plan, &columns, &rest, 1, label, shift, true),
                            expected,
                        );
                        let beta = 8. * std::f64::consts::PI.powi(2);
                        let total: Vec<_> = rest
                            .iter()
                            .enumerate()
                            .map(|(h, &r)| {
                                r + label.map_or(Complex::default(), |e| {
                                    plan.phase[1][h]
                                        * plan.factors[h * plan.elements.len() + e]
                                        * (-beta * shift * plan.s_squared[h])
                                            .clamp(-600., 600.)
                                            .exp()
                                })
                            })
                            .collect();
                        let value = plan.fixed_objective(&total);
                        assert_eq!(
                            ws.fixed_trial(&plan, &columns, &rest, 1, label, shift)
                                .to_bits(),
                            value.to_bits()
                        );
                    }
                }
            }
            assert!(ws.reuses > ws.builds && ws.score_only > 0);
        }
    }
}
