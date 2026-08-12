#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use phasesmith_core::{TofBankGeometry, TofInstrument};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::{
    StructuralTofInputView, calculate_structural_tof_pattern,
    calculate_structural_tof_pattern_dense, calculate_structural_tof_pattern_jvp,
};

struct BenchmarkCase {
    tof_us: Vec<f64>,
    hkl: Vec<[i32; 3]>,
    multiplicity: Vec<usize>,
    xyz: Vec<[f64; 3]>,
    occupancy: Vec<f64>,
    u_iso: Vec<f64>,
    anisotropic: Vec<bool>,
    u_aniso: Vec<[f64; 6]>,
    species: Vec<&'static str>,
    tangent: Vec<f64>,
    cell: UnitCell,
    group: SpaceGroup,
}

impl BenchmarkCase {
    fn new(reflection_count: usize, site_count: usize) -> Self {
        let mut hkl = Vec::with_capacity(reflection_count);
        'outer: for h in 0..18 {
            for k in 0..18 {
                for ell in 0..18 {
                    let norm = h * h + k * k + ell * ell;
                    if (9..=225).contains(&norm) {
                        hkl.push([h, k, ell]);
                        if hkl.len() == reflection_count {
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert_eq!(hkl.len(), reflection_count);
        let index =
            |value: usize| f64::from(u32::try_from(value).expect("benchmark index fits in u32"));
        let species_keys = ["C", "O", "Si", "Fe"];
        let parameter_count = 6 + 5 * site_count + 1;
        Self {
            tof_us: (0..14_501)
                .map(|sample| 1_000.0 + 2.0 * f64::from(sample))
                .collect(),
            hkl,
            multiplicity: vec![2; reflection_count],
            xyz: (0..site_count)
                .map(|site| {
                    let value = index(site);
                    [
                        (0.137 * value).fract(),
                        (0.271 * value + 0.11).fract(),
                        (0.419 * value + 0.23).fract(),
                    ]
                })
                .collect(),
            occupancy: vec![1.0; site_count],
            u_iso: (0..site_count)
                .map(|site| 0.005 + 0.0002 * index(site))
                .collect(),
            anisotropic: vec![false; site_count],
            u_aniso: vec![[0.0; 6]; site_count],
            species: (0..site_count)
                .map(|site| species_keys[site % species_keys.len()])
                .collect(),
            tangent: (0..parameter_count)
                .map(|parameter| 1.0e-5 * index(parameter + 1))
                .collect(),
            cell: UnitCell {
                a_angstrom: 15.0,
                b_angstrom: 15.0,
                c_angstrom: 15.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            group: SpaceGroup::new(vec![SymmetryOperation::identity()]).expect("P1"),
        }
    }

    fn input(&self) -> StructuralTofInputView<'_> {
        StructuralTofInputView {
            tof_us: &self.tof_us,
            hkl: &self.hkl,
            multiplicity: &self.multiplicity,
            fractional_xyz: &self.xyz,
            occupancy: &self.occupancy,
            u_iso_angstrom2: &self.u_iso,
            anisotropic_mask: &self.anisotropic,
            u_aniso_cif_angstrom2: &self.u_aniso,
            scattering_species: &self.species,
            scale: 1.3,
            coordinate_tolerance: 1.0e-10,
            correction_model: IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: 90.0,
            },
            bank_geometry: TofBankGeometry {
                two_theta_deg: 90.0,
            },
            instrument: TofInstrument {
                zero_us: 0.0,
                difc_us_per_angstrom: 5_000.0,
                difa_us_per_angstrom2: 0.0,
                difb_us_angstrom: 0.0,
                alpha_coefficient: 0.2,
                beta0_per_us: 0.03,
                beta1_angstrom4_per_us: 0.001,
                betaq_angstrom2_per_us: 0.0,
                sigma0_us2: 25.0,
                sigma1_us2_per_angstrom2: 4.0,
                sigma2_us2_per_angstrom4: 0.1,
                sigmaq_us2_per_angstrom: 0.0,
                x_us_per_angstrom: 1.0,
                y_us_per_angstrom2: 0.1,
                z_us: 0.5,
            },
            support_fwhm: 20.0,
            tail_log: 20.0,
        }
    }
}

fn structural_tof_benchmark(criterion: &mut Criterion) {
    let case = BenchmarkCase::new(128, 16);
    let mut group = criterion.benchmark_group("structural_tof_128_reflections_16_sites");
    group.throughput(Throughput::Elements(case.hkl.len() as u64));
    group.bench_with_input(BenchmarkId::new("values", "fused"), &case, |bench, case| {
        bench.iter(|| {
            black_box(calculate_structural_tof_pattern(
                case.cell,
                &case.group,
                &case.input(),
            ))
        });
    });
    group.bench_with_input(BenchmarkId::new("dense", "fused"), &case, |bench, case| {
        bench.iter(|| {
            black_box(calculate_structural_tof_pattern_dense(
                case.cell,
                &case.group,
                &case.input(),
            ))
        });
    });
    group.bench_with_input(BenchmarkId::new("jvp", "fused"), &case, |bench, case| {
        bench.iter(|| {
            black_box(calculate_structural_tof_pattern_jvp(
                case.cell,
                &case.group,
                &case.input(),
                &case.tangent,
            ))
        });
    });
    group.finish();
}

criterion_group!(benches, structural_tof_benchmark);
criterion_main!(benches);
