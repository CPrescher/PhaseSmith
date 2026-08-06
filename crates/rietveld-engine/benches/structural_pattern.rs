#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rietveld_core::{
    ConstantWavelengthInstrument, CwContributionArrays, CwContributionsView, GridView,
    SupportPolicy, accumulate_cw_contributions_batch,
};
use rietveld_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedXrayScattering, SpaceGroup,
    StructureFactorBatchView, SymmetryOperation, UnitCell, calculate_structure_factor_values,
};
use rietveld_engine::{
    BuiltInScatteringModel, StructuralPatternInputView, calculate_structural_pattern,
};

struct BenchmarkCase {
    x: Vec<f64>,
    hkl: Vec<[i32; 3]>,
    multiplicity: Vec<usize>,
    xyz: Vec<[f64; 3]>,
    occupancy: Vec<f64>,
    u_iso: Vec<f64>,
    species: Vec<&'static str>,
    zeros: Vec<f64>,
    ones: Vec<f64>,
    cell: UnitCell,
    group: SpaceGroup,
    instrument: ConstantWavelengthInstrument,
}

fn index_f64(index: usize) -> f64 {
    f64::from(u32::try_from(index).expect("benchmark index fits u32"))
}

impl BenchmarkCase {
    fn new(reflection_count: usize, site_count: usize) -> Self {
        let cell = UnitCell {
            a_angstrom: 15.0,
            b_angstrom: 15.0,
            c_angstrom: 15.0,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        };
        let mut hkl = Vec::with_capacity(reflection_count);
        'outer: for h in 0..20 {
            for k in 0..20 {
                for ell in 0..20 {
                    if (h, k, ell) != (0, 0, 0) && h * h + k * k + ell * ell < 340 {
                        hkl.push([h, k, ell]);
                        if hkl.len() == reflection_count {
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert_eq!(hkl.len(), reflection_count);
        let species_keys = ["C", "O", "Si", "Fe"];
        let xyz = (0..site_count)
            .map(|site| {
                let value = index_f64(site);
                [
                    (0.137 * value).fract(),
                    (0.271 * value + 0.11).fract(),
                    (0.419 * value + 0.23).fract(),
                ]
            })
            .collect();
        Self {
            x: (0..20_001)
                .map(|sample| 5.0 + f64::from(sample) * 0.006)
                .collect(),
            hkl,
            multiplicity: vec![1; reflection_count],
            xyz,
            occupancy: vec![1.0; site_count],
            u_iso: (0..site_count)
                .map(|site| 0.005 + 0.0002 * index_f64(site))
                .collect(),
            species: (0..site_count)
                .map(|site| species_keys[site % species_keys.len()])
                .collect(),
            zeros: vec![0.0; reflection_count],
            ones: vec![1.0; reflection_count],
            cell,
            group: SpaceGroup::new(vec![SymmetryOperation::identity()]).expect("P1"),
            instrument: ConstantWavelengthInstrument {
                wavelength_angstrom: 1.5406,
                u_deg2: 2.0e-4,
                v_deg2: -1.0e-4,
                w_deg2: 1.2e-4,
                x_deg: 1.5e-3,
                y_deg: 3.0e-3,
            },
        }
    }

    fn contributions(&self) -> CwContributionsView<'_> {
        CwContributionsView::new(
            self.hkl.len(),
            0,
            CwContributionArrays {
                gaussian_variance_deg2: &self.zeros,
                lorentzian_fwhm_deg: &self.zeros,
                intensity_multiplier: &self.ones,
                d_gaussian_variance_d_position: &self.zeros,
                d_lorentzian_fwhm_d_position: &self.zeros,
                d_intensity_multiplier_d_position: &self.zeros,
                d_gaussian_variance_d_parameters: &[],
                d_lorentzian_fwhm_d_parameters: &[],
                d_intensity_multiplier_d_parameters: &[],
            },
        )
        .expect("contributions")
    }

    fn fused(&self) -> usize {
        calculate_structural_pattern(
            self.cell,
            &self.group,
            &StructuralPatternInputView {
                x_deg: &self.x,
                hkl: &self.hkl,
                multiplicity: &self.multiplicity,
                fractional_xyz: &self.xyz,
                occupancy: &self.occupancy,
                u_iso_angstrom2: &self.u_iso,
                scattering_species: &self.species,
                scale: 1.3,
                coordinate_tolerance: 1.0e-10,
                instrument: self.instrument,
                correction_model: IntegratedIntensityCorrectionModel::Neutral,
                scattering_model: BuiltInScatteringModel::XrayNonResonant,
                contributions: self.contributions(),
                support: SupportPolicy::FwhmMultiple(20.0),
            },
        )
        .expect("fused structural pattern")
        .accumulation
        .derivatives
        .local
        .active_sample_count()
    }

    fn separate(&self) -> usize {
        let geometry = self.cell.geometry().expect("cell geometry");
        let q_squared = self
            .hkl
            .iter()
            .map(|&hkl| geometry.q_squared_and_derivatives(hkl).0)
            .collect::<Vec<_>>();
        let s = q_squared
            .iter()
            .map(|value| 0.5 * value.sqrt())
            .collect::<Vec<_>>();
        let d_spacing = q_squared
            .iter()
            .map(|value| value.sqrt().recip())
            .collect::<Vec<_>>();
        let positions = d_spacing
            .iter()
            .map(|d| {
                (self.instrument.wavelength_angstrom / (2.0 * d))
                    .asin()
                    .to_degrees()
                    * 2.0
            })
            .collect::<Vec<_>>();
        let scattering = PreparedXrayScattering::new(self.species.iter().copied())
            .and_then(|model| model.evaluate(&s))
            .expect("scattering");
        let correction = IntegratedIntensityCorrectionModel::Neutral
            .evaluate(&q_squared)
            .expect("correction");
        let structural = calculate_structure_factor_values(
            self.cell,
            &self.group,
            StructureFactorBatchView {
                hkl: &self.hkl,
                multiplicity: &self.multiplicity,
                fractional_xyz: &self.xyz,
                occupancy: &self.occupancy,
                u_iso_angstrom2: &self.u_iso,
                scattering_real: &scattering.real,
                scattering_imag: &scattering.imag,
                d_scattering_real_d_s: &scattering.d_real_d_s,
                d_scattering_imag_d_s: &scattering.d_imag_d_s,
                correction: &correction.values,
                d_correction_d_q_squared: &correction.d_values_d_q_squared,
                scale: 1.3,
                coordinate_tolerance: 1.0e-10,
            },
        )
        .expect("structure factors");
        accumulate_cw_contributions_batch(
            GridView::new(&self.x).expect("grid"),
            &positions,
            &structural.intensity,
            self.instrument,
            self.contributions(),
            SupportPolicy::FwhmMultiple(20.0),
        )
        .expect("profile")
        .derivatives
        .local
        .active_sample_count()
    }
}

fn structural_pattern_benchmark(criterion: &mut Criterion) {
    let case = BenchmarkCase::new(256, 32);
    assert_eq!(case.fused(), case.separate());
    let mut group = criterion.benchmark_group("structural_pattern_256_reflections_32_sites");
    group.throughput(Throughput::Elements(case.hkl.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("fused", "values+derivatives"),
        &case,
        |bench, case| {
            bench.iter(|| black_box(case.fused()));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("separate", "values+derivatives"),
        &case,
        |bench, case| {
            bench.iter(|| black_box(case.separate()));
        },
    );
    group.finish();
}

criterion_group!(benches, structural_pattern_benchmark);
criterion_main!(benches);
