#![allow(missing_docs)]
use criterion::{Criterion, criterion_group, criterion_main};
use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation, UnitCell};
use phasesmith_solution::{
    AdpMode, DualSpaceOptions, FlipOptions, IntegratedPeakDensity, Observation, PeakSelection,
    PhasingMethod, PreparedElementPool, PreparedOrbitSearch, PreparedPeakCompletion,
    PreparedPhaseOrigin, PreparedSymmetricModel, PreparedTangent, SharedSearchOptions,
    SolveOptions, charge_flip, dual_space, recover_symmetry, recycle_density,
    recycle_density_guided,
};

#[allow(clippy::too_many_lines)]
fn benchmark(c: &mut Criterion) {
    let shape = [32; 3];
    let mut hkl = Vec::new();
    for h in -8_i32..=8 {
        for k in -8_i32..=8 {
            for l in -8_i32..=8 {
                if h * h + k * k + l * l <= 64 && [h, k, l] != [0; 3] {
                    hkl.push([h, k, l]);
                }
            }
        }
    }
    let amplitudes: Vec<_> = hkl
        .iter()
        .map(|h| {
            // Four independent point atoms produce an actual multi-peak Fourier field.
            let xyz = [
                [0., 0., 0.],
                [0.17, 0.29, 0.41],
                [0.36, 0.62, 0.13],
                [0.73, 0.21, 0.84],
            ];
            let mut re = 0.;
            let mut im = 0.;
            for p in xyz {
                let phase = std::f64::consts::TAU
                    * (f64::from(h[0]) * p[0] + f64::from(h[1]) * p[1] + f64::from(h[2]) * p[2]);
                let (s, c) = phase.sin_cos();
                re += c;
                im += s;
            }
            re.hypot(im)
        })
        .collect();
    let phases: Vec<_> = hkl
        .iter()
        .map(|h| f64::from(h[0] + 3 * h[1] + 7 * h[2]) * 0.137)
        .collect();
    let mut support = vec![0; 32 * 32 * 32];
    support[0] = 1;
    for h in &hkl {
        let v = h.map(|x| usize::try_from(x.rem_euclid(32)).unwrap());
        support[(v[0] * 32 + v[1]) * 32 + v[2]] = 2;
    }
    let cell = UnitCell {
        a_angstrom: 8.,
        b_angstrom: 9.,
        c_angstrom: 10.,
        alpha_deg: 90.,
        beta_deg: 90.,
        gamma_deg: 90.,
    };
    let source = charge_flip(
        shape,
        &hkl,
        &amplitudes,
        &phases,
        &support,
        &FlipOptions {
            cycles: 50,
            delta_sigma: 1.,
            weak_fraction: 0.2,
            cleanup_cycles: 0,
            cancellation: None,
        },
        None,
    )
    .unwrap()
    .density;
    let integrated =
        IntegratedPeakDensity::new(cell, shape, &hkl, &amplitudes, &source, 0.7).unwrap();
    c.bench_function("integrated_peak_map_32cubed_2108_reflections", |b| {
        b.iter(|| {
            IntegratedPeakDensity::new(cell, shape, &hkl, &amplitudes, &source, 0.7).unwrap()
        });
    });
    let centers: Vec<_> = (0..32)
        .map(|i| [f64::from(i) / 32. + 0.013, 0.271, 0.413])
        .collect();
    c.bench_function(
        "integrated_peak_32_centers_fused_value_gradient_radius",
        |b| {
            b.iter(|| {
                centers
                    .iter()
                    .map(|&p| integrated.evaluate(std::hint::black_box(p)).unwrap())
                    .collect::<Vec<_>>()
            });
        },
    );
    let points = [
        [0.03, 0.07, 0.11],
        [0.17, 0.29, 0.41],
        [0.36, 0.62, 0.13],
        [0.73, 0.21, 0.84],
        [0.25, 0.75, 0.31],
        [0.43, 0.14, 0.68],
        [0.82, 0.56, 0.27],
        [0.61, 0.87, 0.49],
    ];
    let elements = ["Si", "Si", "O", "O", "O", "O", "O", "O"].map(str::to_owned);
    let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
    let symmetric = recover_symmetry(cell, &group, &elements, &points, 0.01, None, None).unwrap();
    let prepared =
        PreparedSymmetricModel::new(symmetric, &hkl, &amplitudes, AdpMode::Anisotropic).unwrap();
    let parameters = prepared.initial_parameters(0.015).unwrap();
    c.bench_function(
        "symmetric_adp_evaluate_8_atoms_2108_reflections_69_parameters",
        |b| {
            b.iter(|| {
                prepared
                    .evaluate(std::hint::black_box(&parameters))
                    .unwrap()
            });
        },
    );
    for iterative in [false, true] {
        c.bench_function(
            if iterative {
                "refinement_cg_8_atoms_2108_rows_69_parameters_3_steps"
            } else {
                "refinement_dense_8_atoms_2108_rows_69_parameters_3_steps"
            },
            |b| {
                b.iter(|| {
                    prepared
                        .clone()
                        .refine_with_solver(&parameters, 3, None, None, iterative)
                        .unwrap()
                });
            },
        );
    }
    let mirror_group = SpaceGroup::new(vec![
        SymmetryOperation::identity(),
        SymmetryOperation::new(
            [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [Rational::new(0, 1).unwrap(); 3],
        )
        .unwrap(),
    ])
    .unwrap();
    let mut mirror_points = Vec::new();
    let mut mirror_elements = Vec::new();
    for (i, point) in points[..4].iter().enumerate() {
        let images = mirror_group.expand_sites(&[*point], 1e-8).unwrap();
        mirror_elements.extend(std::iter::repeat_n(
            if i < 2 {
                "Si".to_owned()
            } else {
                "O".to_owned()
            },
            images.fractional_xyz.len(),
        ));
        mirror_points.extend(images.fractional_xyz);
    }
    let recycling_amplitudes: Vec<_> = hkl
        .iter()
        .map(|h| {
            let mut re = 0.;
            let mut im = 0.;
            for (point, species) in mirror_points.iter().zip(&mirror_elements) {
                let angle =
                    std::f64::consts::TAU * (0..3).map(|a| f64::from(h[a]) * point[a]).sum::<f64>();
                let (sin, cos) = angle.sin_cos();
                let weight = if species == "Si" { 14. } else { 8. };
                re += weight * cos;
                im += weight * sin;
            }
            re.hypot(im)
        })
        .collect();
    // Eight separated periodic peaks on a realistic multi-reflection grid.
    let shifted: Vec<_> = mirror_points
        .iter()
        .map(|p| {
            std::array::from_fn::<_, 3, _>(|a| (p[a] + [0.137, 0.271, 0.419][a]).rem_euclid(1.))
        })
        .collect();
    let mut origin_density = vec![0.; 32_usize.pow(3)];
    for (index, value) in origin_density.iter_mut().enumerate() {
        let node = [index / 1024, index / 32 % 32, index % 32];
        for (p, species) in shifted.iter().zip(&mirror_elements) {
            let d2: f64 = (0..3)
                .map(|a| {
                    let d = (f64::from(u32::try_from(node[a]).unwrap()) / 32. - p[a] + 0.5)
                        .rem_euclid(1.)
                        - 0.5;
                    d * d
                })
                .sum();
            *value += if species == "Si" { 14. } else { 8. } * (-d2 / 0.0008).exp();
        }
    }
    let origin_plan = PreparedPhaseOrigin::new(&mirror_group, &hkl, &recycling_amplitudes).unwrap();
    let origin_map = origin_plan.map(&origin_density, shape, 1).unwrap();
    c.bench_function(
        "phase_origin_fused_loss_gradient_hessian_8_peaks_2108_rows",
        |b| {
            b.iter(|| {
                origin_map
                    .evaluate(std::hint::black_box([0.137, 0.271, 0.419]))
                    .unwrap()
            });
        },
    );
    c.bench_function("phase_origin_search_32cubed_8_peaks_2108_rows", |b| {
        b.iter(|| {
            origin_plan
                .search(&origin_density, shape, 4, None, None)
                .unwrap()
        });
    });
    c.bench_function("phase_origin_independent_32cubed_8_peaks_2108_rows", |b| {
        b.iter(|| {
            origin_plan
                .search_sampled(&origin_density, shape, 4, None, None, true)
                .unwrap()
        });
    });
    c.bench_function("phase_checkpoint_32cubed_8_peaks", |b| {
        let mut selector = phasesmith_solution::PhaseCheckpointSelector::default();
        selector
            .consider(&origin_density, 16, 0.7, 0.1, 0.7, false)
            .unwrap();
        b.iter(|| {
            selector
                .consider(
                    std::hint::black_box(&origin_density),
                    64,
                    0.7,
                    0.1,
                    0.7,
                    false,
                )
                .unwrap()
        });
    });
    let cache_rows: Vec<_> = hkl
        .iter()
        .zip(&recycling_amplitudes)
        .filter(|(h, _)| **h > [0; 3])
        .collect();
    let cache_hkl: Vec<_> = cache_rows.iter().map(|(h, _)| **h).collect();
    let cache_observed: Vec<_> = cache_rows.iter().map(|(_, a)| **a).collect();
    let cache_elements = ["Si".to_owned(), "O".to_owned()];
    let make_cache = || {
        PreparedPeakCompletion::new(
            cell,
            &mirror_group,
            &points[..4],
            &cache_elements,
            &cache_hkl,
            &cache_observed,
            32,
        )
        .unwrap()
    };
    let cache = make_cache();
    let labels = [Some(0), Some(0), Some(1), Some(1)];
    let displaced_cache = make_cache()
        .with_site_displacements(&[0.01, 0.02, 0.015, 0.03])
        .unwrap();
    c.bench_function("chemical_reassignment_8_atoms_1054_rows", |b| {
        b.iter(|| {
            displaced_cache
                .reassign(&[Some(1), Some(0), Some(1), Some(0)], None, None)
                .unwrap()
        });
    });
    let prediction_observations: Vec<_> = hkl
        .iter()
        .zip(&amplitudes)
        .map(|(&h, &a)| Observation {
            hkl: h,
            intensity: a * a,
            sigma: (a * a * 0.02).max(0.1),
        })
        .collect();
    let prediction_data =
        phasesmith_solution::prepare_data(&prediction_observations, cell, &group).unwrap();
    let prediction_norm = phasesmith_solution::normalize_wilson(
        cell,
        &group,
        &prediction_data.hkl,
        &prediction_data.intensity,
        &cache_elements,
    )
    .unwrap();
    let make_prediction = || {
        phasesmith_solution::PreparedReflectionPrediction::new(
            shape,
            cell,
            &group,
            &prediction_data,
            &prediction_norm,
            &cache_elements,
            &prediction_data.amplitudes,
            1.,
            true,
            1.2,
        )
        .unwrap()
    };
    let prediction = make_prediction();
    let mut prediction_values = vec![rustfft::num_complex::Complex::default(); 32_usize.pow(3)];
    for (index, value) in prediction_values.iter_mut().enumerate() {
        if prediction.support[index] == 0 {
            continue;
        }
        let node = [index / 1024, index / 32 % 32, index % 32];
        let h = node.map(|i| f64::from(i32::try_from(i).unwrap()) - if i >= 16 { 32. } else { 0. });
        for (site, p) in points.iter().enumerate() {
            let angle = std::f64::consts::TAU * (0..3).map(|a| h[a] * p[a]).sum::<f64>();
            *value +=
                rustfft::num_complex::Complex::from_polar(if site < 2 { 14. } else { 8. }, angle);
        }
        *value /= prediction.observed_rms;
    }
    c.bench_function(
        "reflection_prediction_plan_32cubed_2108_measured_rows",
        |b| {
            b.iter(make_prediction);
        },
    );
    c.bench_function(
        "reflection_prediction_update_32cubed_2108_measured_rows",
        |b| {
            b.iter(|| {
                prediction
                    .apply(&mut prediction_values.clone(), 1., 24)
                    .unwrap()
            });
        },
    );
    c.bench_function("peak_completion_cache_8_atoms_1054_rows", |b| {
        b.iter(make_cache);
    });
    c.bench_function(
        "peak_completion_fused_value_gradient_8_atoms_1054_rows",
        |b| {
            b.iter(|| cache.evaluate(std::hint::black_box(&labels), 2).unwrap());
        },
    );
    c.bench_function("peak_completion_counts_labels_8_atoms_1054_rows", |b| {
        b.iter(|| {
            cache
                .search(&[28., 28., 16., 16.], &[1.7, 2., 2.3], 4, None, None)
                .unwrap()
        });
    });
    c.bench_function(
        "symmetry_recycling_10_cycles_32cubed_8_peaks_2108_reflections",
        |b| {
            b.iter(|| {
                recycle_density(
                    shape,
                    &hkl,
                    &recycling_amplitudes,
                    &support,
                    &mirror_group,
                    &source,
                    1,
                    [0.137, 0.271, 0.419],
                    10,
                    None,
                    None,
                )
                .unwrap()
            });
        },
    );
    let phase_intensity: Vec<_> = recycling_amplitudes.iter().map(|a| a * a).collect();
    let phase_sigma: Vec<_> = phase_intensity.iter().map(|i| 0.01 + 0.02 * i).collect();
    let tangent = PreparedTangent::new(
        cell,
        &mirror_group,
        &hkl,
        &phase_intensity,
        &phase_sigma,
        None,
        None,
    )
    .unwrap();
    let phase_values: Vec<_> = tangent
        .independent_hkl()
        .iter()
        .map(|h| f64::from(h[0] + 3 * h[1] + 7 * h[2]) * 0.137)
        .collect();
    c.bench_function(
        "symmetry_tangent_recycling_10_cycles_32cubed_8_peaks_2108_reflections",
        |b| {
            b.iter(|| {
                recycle_density_guided(
                    shape,
                    &hkl,
                    &recycling_amplitudes,
                    &support,
                    &mirror_group,
                    &source,
                    1,
                    [0.137, 0.271, 0.419],
                    10,
                    None,
                    None,
                    Some(&tangent),
                    471,
                )
                .unwrap()
            });
        },
    );
    let phase_magnitudes = tangent.normalized_values().0.to_vec();
    c.bench_function("tangent_plan_8_atoms_2108_reflections", |b| {
        b.iter(|| {
            PreparedTangent::new(
                cell,
                &mirror_group,
                &hkl,
                &phase_intensity,
                &phase_sigma,
                None,
                None,
            )
            .unwrap()
        });
    });
    c.bench_function("tangent_fused_values_gradients_2108_reflections", |b| {
        b.iter(|| {
            tangent
                .evaluate(std::hint::black_box(&phase_values))
                .unwrap()
        });
    });
    c.bench_function("tangent_complete_annealed_step_2108_reflections", |b| {
        b.iter(|| {
            tangent
                .step(
                    std::hint::black_box(&phase_values),
                    &phase_magnitudes,
                    3,
                    471,
                )
                .unwrap()
        });
    });
    let mirror = recover_symmetry(
        cell,
        &mirror_group,
        &mirror_elements,
        &mirror_points,
        0.01,
        None,
        None,
    )
    .unwrap();
    for (mode_name, mode) in [
        ("isotropic", AdpMode::Isotropic),
        ("anisotropic", AdpMode::Anisotropic),
    ] {
        for (cache_name, enabled) in [("direct", false), ("cached", true)] {
            let model = PreparedSymmetricModel::new_with_geometry_cache(
                mirror.clone(),
                &hkl,
                &amplitudes,
                mode,
                enabled,
            )
            .unwrap();
            let seed = model.initial_parameters(0.015).unwrap();
            c.bench_function(&format!("refinement_geometry_{mode_name}_{cache_name}_evaluate_8_images_2108_reflections"), |b| {
                b.iter(|| model.evaluate(std::hint::black_box(&seed)).unwrap());
            });
            c.bench_function(&format!("refinement_geometry_{mode_name}_{cache_name}_prepare_fit8_8_images_2108_reflections"), |b| {
                b.iter(|| {
                    let prepared=PreparedSymmetricModel::new_with_geometry_cache(mirror.clone(),&hkl,&amplitudes,mode,enabled).unwrap();
                    let start=prepared.initial_parameters(0.015).unwrap();
                    prepared.refine(&start,8,None,None).unwrap()
                });
            });
        }
    }
    c.bench_function(
        "dual_space_32cubed_four_atoms_2108_reflections_50_cycles",
        |b| {
            b.iter(|| {
                dual_space(
                    shape,
                    cell,
                    &hkl,
                    &amplitudes,
                    &phases,
                    &support,
                    &DualSpaceOptions {
                        cycles: 50,
                        peak_selection: phasesmith_solution::PeakSelection::Fixed(4),
                        mask_sigma: 0.3,
                        hybrid_m: 3.,
                        omit_fraction: 0.1,
                        seed: 0,
                        patterson_rank: Some(0),
                        cancellation: None,
                    },
                    None,
                )
                .unwrap()
            });
        },
    );
    c.bench_function(
        "count_free_dual_space_32cubed_2108_reflections_50_cycles",
        |b| {
            b.iter(|| {
                dual_space(
                    shape,
                    cell,
                    &hkl,
                    &amplitudes,
                    &phases,
                    &support,
                    &DualSpaceOptions {
                        cycles: 50,
                        peak_selection: phasesmith_solution::PeakSelection::Adaptive {
                            sigma: 2.,
                            cap: 128,
                        },
                        mask_sigma: 0.3,
                        hybrid_m: 3.,
                        omit_fraction: 0.1,
                        seed: 0,
                        patterson_rank: Some(0),
                        cancellation: None,
                    },
                    None,
                )
                .unwrap()
            });
        },
    );
    c.bench_function(
        "monitored_dual_space_32cubed_2108_reflections_50_cycles",
        |b| {
            b.iter(|| {
                let mut decline = |_: &[f64], _: usize, _: usize| Ok(false);
                phasesmith_solution::dual_space_monitored(
                    shape,
                    cell,
                    &hkl,
                    &amplitudes,
                    &phases,
                    &support,
                    &DualSpaceOptions {
                        cycles: 50,
                        peak_selection: phasesmith_solution::PeakSelection::Adaptive {
                            sigma: 2.,
                            cap: 128,
                        },
                        mask_sigma: 0.3,
                        hybrid_m: 3.,
                        omit_fraction: 0.1,
                        seed: 0,
                        patterson_rank: Some(0),
                        cancellation: None,
                    },
                    None,
                    Some(&mut decline),
                )
                .unwrap()
            });
        },
    );
    let mut operations = Vec::new();
    for centre in [0, 1] {
        for x in [-1, 1] {
            for y in [-1, 1] {
                for z in [-1, 1] {
                    operations.push(
                        SymmetryOperation::new(
                            [[x, 0, 0], [0, y, 0], [0, 0, z]],
                            [Rational::new(centre, 2).unwrap(); 3],
                        )
                        .unwrap(),
                    );
                }
            }
        }
    }
    let orbit_group = SpaceGroup::new(operations).unwrap();
    let framework = orbit_group
        .expand_sites(&[[0.08, 0.17, 0.29]], 1e-8)
        .unwrap()
        .fractional_xyz;
    let indices = (0..hkl.len())
        .filter(|&i| (hkl[i][0] + hkl[i][1] + hkl[i][2]) % 2 == 0)
        .collect::<Vec<_>>();
    let orbit_hkl = indices.iter().map(|&i| hkl[i]).collect::<Vec<_>>();
    let orbit_observed = indices.iter().map(|&i| amplitudes[i]).collect::<Vec<_>>();
    let orbit = PreparedOrbitSearch::new(
        cell,
        &orbit_group,
        &vec!["Si".to_owned(); framework.len()],
        &framework,
        "O",
        &orbit_hkl,
        &orbit_observed,
        0.01,
    )
    .unwrap();
    let shared = PreparedElementPool::new(
        cell,
        &orbit_group,
        &vec!["Si".to_owned(); framework.len()],
        &framework,
        &["Si".to_owned(), "O".to_owned(), "Fe".to_owned()],
        &orbit_hkl,
        &orbit_observed,
        0.01,
    )
    .unwrap();
    c.bench_function(
        "shared_orbit_16_images_1060_reflections_three_labels_fused_gradient",
        |b| {
            b.iter(|| {
                shared
                    .evaluate(std::hint::black_box([0.31, 0.43, 0.67]))
                    .unwrap()
            });
        },
    );
    let separate: Vec<_> = ["Si", "O", "Fe"]
        .into_iter()
        .map(|element| {
            PreparedOrbitSearch::new(
                cell,
                &orbit_group,
                &vec!["Si".to_owned(); framework.len()],
                &framework,
                element,
                &orbit_hkl,
                &orbit_observed,
                0.01,
            )
            .unwrap()
        })
        .collect();
    c.bench_function(
        "independent_orbit_16_images_1060_reflections_three_labels_fused_gradient",
        |b| {
            b.iter(|| {
                separate
                    .iter()
                    .map(|orbit| {
                        orbit
                            .evaluate(std::hint::black_box([0.31, 0.43, 0.67]))
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            });
        },
    );
    c.bench_function(
        "completion_orbit_16_images_multi_reflection_fused_gradient",
        |b| {
            b.iter(|| {
                orbit
                    .evaluate(std::hint::black_box([0.31, 0.43, 0.67]))
                    .unwrap()
            });
        },
    );
    // Actual multi-atom/count search over 2,108 measured reflection rows,
    // including density recycling, candidate generation and constrained fits.
    let calibration = phasesmith_solution::evaluate_model(
        cell,
        &hkl,
        &vec![1.; hkl.len()],
        &elements,
        &points,
        0.012,
    )
    .unwrap();
    #[allow(clippy::cast_precision_loss)]
    let reference_norm = (hkl.len() as f64).sqrt();
    let count_observations: Vec<_> = hkl
        .iter()
        .zip(&calibration.residual)
        .map(|(&hkl, &r)| {
            let amplitude = (r * reference_norm + 1.) / calibration.amplitude_scale;
            Observation {
                hkl,
                intensity: amplitude * amplitude,
                sigma: 1.,
            }
        })
        .collect();
    let quality_data =
        phasesmith_solution::prepare_data(&count_observations, cell, &group).unwrap();
    let quality_elements = ["Si".to_owned(), "O".to_owned()];
    let quality_norm = phasesmith_solution::normalize_wilson(
        cell,
        &group,
        &quality_data.hkl,
        &quality_data.intensity,
        &quality_elements,
    )
    .unwrap();
    let quality_plan = phasesmith_solution::PreparedPhaseQuality::new(
        cell,
        &group,
        &quality_data,
        &count_observations,
        &quality_elements,
        &quality_norm,
        true,
        true,
    )
    .unwrap();
    c.bench_function("wilson_normalization_8_atoms_2108_reflections", |b| {
        b.iter(|| {
            phasesmith_solution::normalize_wilson(
                cell,
                &group,
                &quality_data.hkl,
                &quality_data.intensity,
                &quality_elements,
            )
            .unwrap()
        });
    });
    c.bench_function(
        "phase_quality_fused_value_gradient_8_atoms_1054_families",
        |b| {
            b.iter(|| {
                quality_plan
                    .evaluate(std::hint::black_box(&quality_plan.target))
                    .unwrap()
            });
        },
    );
    c.bench_function("phase_quality_values_only_8_atoms_1054_families", |b| {
        let mut scorer = phasesmith_solution::PreparedPhaseScore::new(&quality_plan).unwrap();
        b.iter(|| {
            scorer
                .evaluate(std::hint::black_box(&quality_plan.target))
                .unwrap()
        });
    });
    for (label, score_only, reuse) in [
        ("control", false, false),
        ("score", true, false),
        ("workspace", false, true),
        ("combined", true, true),
    ] {
        c.bench_function(
            &format!("phase_execution_32cubed_8_atoms_2108_rows_64_cycle_cap_{label}"),
            |b| {
                let execution = phasesmith_solution::PhasingExecution {
                    score_only_evaluation: score_only,
                    reusable_fourier_workspaces: reuse,
                    filtered_peak_search: false,
                    batched_fft: false,
                };
                let options = DualSpaceOptions {
                    cycles: 64,
                    peak_selection: PeakSelection::Adaptive {
                        sigma: 2.,
                        cap: 128,
                    },
                    mask_sigma: 0.25,
                    hybrid_m: 3.,
                    omit_fraction: 0.1,
                    seed: 1034,
                    patterson_rank: Some(0),
                    cancellation: None,
                };
                let sample = phasesmith_solution::dual_space_with_execution(
                    shape,
                    cell,
                    &quality_data.hkl,
                    &quality_data.amplitudes,
                    &phases,
                    &support,
                    &options,
                    None,
                    Some(&quality_plan),
                    execution,
                )
                .unwrap();
                eprintln!(
                    "execution_sample {label} completed_cycles {}",
                    sample.trial.history.len()
                );
                drop(sample);
                b.iter(|| {
                    phasesmith_solution::dual_space_with_execution(
                        shape,
                        cell,
                        &quality_data.hkl,
                        &quality_data.amplitudes,
                        &phases,
                        &support,
                        &options,
                        None,
                        Some(&quality_plan),
                        execution,
                    )
                    .unwrap()
                });
            },
        );
    }
    for (label, filtered) in [("control", false), ("filtered", true)] {
        c.bench_function(
            &format!("filtered_peak_phase_32cubed_8_atoms_2108_rows_64_cycle_cap_{label}"),
            |b| {
                let execution = phasesmith_solution::PhasingExecution {
                    score_only_evaluation: true,
                    reusable_fourier_workspaces: true,
                    filtered_peak_search: filtered,
                    batched_fft: false,
                };
                let options = DualSpaceOptions {
                    cycles: 64,
                    peak_selection: PeakSelection::Adaptive {
                        sigma: 2.,
                        cap: 128,
                    },
                    mask_sigma: 0.25,
                    hybrid_m: 3.,
                    omit_fraction: 0.1,
                    seed: 1034,
                    patterson_rank: Some(0),
                    cancellation: None,
                };
                let sample = phasesmith_solution::dual_space_with_execution(
                    shape,
                    cell,
                    &quality_data.hkl,
                    &quality_data.amplitudes,
                    &phases,
                    &support,
                    &options,
                    None,
                    Some(&quality_plan),
                    execution,
                )
                .unwrap();
                eprintln!(
                    "execution_sample {label} completed_cycles {}",
                    sample.trial.history.len()
                );
                drop(sample);
                b.iter(|| {
                    phasesmith_solution::dual_space_with_execution(
                        shape,
                        cell,
                        &quality_data.hkl,
                        &quality_data.amplitudes,
                        &phases,
                        &support,
                        &options,
                        None,
                        Some(&quality_plan),
                        execution,
                    )
                    .unwrap()
                });
            },
        );
    }
    for (label, batched) in [("control", false), ("batched", true)] {
        c.bench_function(
            &format!("batched_fft_phase_32cubed_8_atoms_2108_rows_64_cycle_cap_{label}"),
            |b| {
                let execution = phasesmith_solution::PhasingExecution {
                    score_only_evaluation: true,
                    reusable_fourier_workspaces: true,
                    filtered_peak_search: true,
                    batched_fft: batched,
                };
                let options = DualSpaceOptions {
                    cycles: 64,
                    peak_selection: PeakSelection::Adaptive {
                        sigma: 2.,
                        cap: 128,
                    },
                    mask_sigma: 0.25,
                    hybrid_m: 3.,
                    omit_fraction: 0.1,
                    seed: 1034,
                    patterson_rank: Some(0),
                    cancellation: None,
                };
                let sample = phasesmith_solution::dual_space_with_execution(
                    shape,
                    cell,
                    &quality_data.hkl,
                    &quality_data.amplitudes,
                    &phases,
                    &support,
                    &options,
                    None,
                    Some(&quality_plan),
                    execution,
                )
                .unwrap();
                eprintln!(
                    "execution_sample {label} completed_cycles {}",
                    sample.trial.history.len()
                );
                drop(sample);
                b.iter(|| {
                    phasesmith_solution::dual_space_with_execution(
                        shape,
                        cell,
                        &quality_data.hkl,
                        &quality_data.amplitudes,
                        &phases,
                        &support,
                        &options,
                        None,
                        Some(&quality_plan),
                        execution,
                    )
                    .unwrap()
                });
            },
        );
    }
    for filtered in [false, true] {
        let mut plan = phasesmith_solution::PreparedAdaptivePeaks::new(shape).unwrap();
        let label = if filtered { "filtered" } else { "control" };
        c.bench_function(
            &format!("filtered_peak_scan_32cubed_8_atoms_{label}"),
            |b| {
                b.iter(|| {
                    if filtered {
                        plan.select(std::hint::black_box(&source), 2., 128).unwrap()
                    } else {
                        phasesmith_solution::adaptive_peaks(
                            std::hint::black_box(&source),
                            shape,
                            2.,
                            128,
                        )
                        .unwrap()
                    }
                });
            },
        );
    }
    for work_sliced in [false, true] {
        let options = SolveOptions {
            phasing: PhasingMethod::DualSpace,
            trials: 2,
            cycles: 128,
            max_seconds: 60.,
            r1_target: 1e-6,
            ..SolveOptions::default()
        };
        let search = SharedSearchOptions {
            maximum_atoms: 32,
            symmetry_recycling: true,
            tangent_refinement: true,
            wilson_normalization: true,
            phase_quality_selection: true,
            phase_origin_selection: true,
            chemical_assignment: true,
            iterative_refinement: true,
            score_only_evaluation: true,
            reusable_fourier_workspaces: true,
            work_sliced_phasing: work_sliced,
            ..SharedSearchOptions::default()
        };
        let label = if work_sliced { "work" } else { "control" };
        c.bench_function(
            &format!("work_slice_search_32cubed_8_atoms_2108_rows_{label}"),
            |b| {
                b.iter(|| {
                    phasesmith_solution::solve_elements(
                        &count_observations,
                        cell,
                        &group,
                        &quality_elements,
                        &options,
                        &search,
                    )
                    .unwrap()
                });
            },
        );
    }
    for assignment in [false, true] {
        let options = SolveOptions {
            phasing: PhasingMethod::DualSpace,
            trials: 2,
            cycles: 128,
            max_seconds: 60.,
            r1_target: 1e-6,
            ..SolveOptions::default()
        };
        let search = SharedSearchOptions {
            maximum_atoms: 32,
            symmetry_recycling: true,
            tangent_refinement: true,
            wilson_normalization: true,
            phase_quality_selection: true,
            phase_origin_selection: true,
            chemical_assignment: true,
            iterative_refinement: true,
            score_only_evaluation: true,
            reusable_fourier_workspaces: true,
            work_sliced_phasing: true,
            isotropic_assignment_refinement: assignment,
            ..SharedSearchOptions::default()
        };
        let label = if assignment { "assignment" } else { "control" };
        c.bench_function(
            &format!("isotropic_assignment_search_32cubed_8_atoms_2108_rows_{label}"),
            |b| {
                b.iter(|| {
                    phasesmith_solution::solve_elements(
                        &count_observations,
                        cell,
                        &group,
                        &quality_elements,
                        &options,
                        &search,
                    )
                    .unwrap()
                });
            },
        );
    }
    for filtered in [false, true] {
        let options = SolveOptions {
            phasing: PhasingMethod::DualSpace,
            trials: 2,
            cycles: 128,
            max_seconds: 60.,
            r1_target: 1e-6,
            ..SolveOptions::default()
        };
        let search = SharedSearchOptions {
            maximum_atoms: 32,
            symmetry_recycling: true,
            tangent_refinement: true,
            wilson_normalization: true,
            phase_quality_selection: true,
            phase_origin_selection: true,
            chemical_assignment: true,
            iterative_refinement: true,
            score_only_evaluation: true,
            reusable_fourier_workspaces: true,
            work_sliced_phasing: true,
            isotropic_assignment_refinement: true,
            filtered_peak_search: filtered,
            ..SharedSearchOptions::default()
        };
        let label = if filtered { "filtered" } else { "control" };
        c.bench_function(
            &format!("filtered_peak_search_32cubed_8_atoms_2108_rows_{label}"),
            |b| {
                b.iter(|| {
                    phasesmith_solution::solve_elements(
                        &count_observations,
                        cell,
                        &group,
                        &quality_elements,
                        &options,
                        &search,
                    )
                    .unwrap()
                });
            },
        );
    }
    for batched in [false, true] {
        let options = SolveOptions {
            phasing: PhasingMethod::DualSpace,
            trials: 2,
            cycles: 128,
            max_seconds: 60.,
            r1_target: 1e-6,
            ..SolveOptions::default()
        };
        let search = SharedSearchOptions {
            maximum_atoms: 32,
            symmetry_recycling: true,
            tangent_refinement: true,
            wilson_normalization: true,
            phase_quality_selection: true,
            phase_origin_selection: true,
            chemical_assignment: true,
            iterative_refinement: true,
            score_only_evaluation: true,
            reusable_fourier_workspaces: true,
            work_sliced_phasing: true,
            isotropic_assignment_refinement: true,
            filtered_peak_search: true,
            batched_fft: batched,
            ..SharedSearchOptions::default()
        };
        let label = if batched { "batched" } else { "control" };
        c.bench_function(
            &format!("batched_fft_search_32cubed_8_atoms_2108_rows_{label}"),
            |b| {
                b.iter(|| {
                    phasesmith_solution::solve_elements(
                        &count_observations,
                        cell,
                        &group,
                        &quality_elements,
                        &options,
                        &search,
                    )
                    .unwrap()
                });
            },
        );
    }
    let mut count_benches = c.benchmark_group("count_selection_8_atoms_2108_reflections");
    count_benches
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    for adaptive in [false, true] {
        count_benches.bench_function(if adaptive { "adaptive" } else { "legacy" }, |b| {
            b.iter(|| {
                phasesmith_solution::solve_elements(
                    &count_observations,
                    cell,
                    &group,
                    &["Si".to_owned(), "O".to_owned()],
                    &SolveOptions {
                        trials: 1,
                        cycles: 64,
                        max_seconds: 2.,
                        r1_target: 0.06,
                        phasing: PhasingMethod::DualSpace,
                        ..SolveOptions::default()
                    },
                    &SharedSearchOptions {
                        maximum_atoms: 32,
                        symmetry_recycling: true,
                        adaptive_counts: adaptive,
                        ..SharedSearchOptions::default()
                    },
                )
                .unwrap()
            });
        });
    }
    count_benches.finish();
    // Fixed cycle/round work, with a generous deadline, isolates orchestration
    // instead of measuring the wall-clock cap on a scalar statistic call.
    let mut schedules = c.benchmark_group("phase_schedule_8_atoms_2108_reflections");
    schedules
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    for phase_first in [true, false] {
        schedules.bench_function(
            if phase_first {
                "phase_first"
            } else {
                "checkpoint"
            },
            |b| {
                b.iter(|| {
                    phasesmith_solution::solve_elements(
                        &count_observations,
                        cell,
                        &group,
                        &quality_elements,
                        &SolveOptions {
                            trials: 1,
                            cycles: 16,
                            max_seconds: 30.,
                            r1_target: 0.06,
                            phasing: PhasingMethod::DualSpace,
                            ..SolveOptions::default()
                        },
                        &SharedSearchOptions {
                            maximum_atoms: 16,
                            beam_width: 1,
                            rounds: 1,
                            symmetry_recycling: true,
                            wilson_normalization: true,
                            phase_quality_selection: true,
                            phase_first_selection: phase_first,
                            ..SharedSearchOptions::default()
                        },
                    )
                    .unwrap()
                });
            },
        );
    }
    schedules.finish();
    c.bench_function(
        "charge_flip_32cubed_four_atoms_2108_reflections_50_cycles",
        |b| {
            b.iter(|| {
                charge_flip(
                    shape,
                    &hkl,
                    &amplitudes,
                    &phases,
                    &support,
                    &FlipOptions {
                        cycles: 50,
                        delta_sigma: 1.,
                        weak_fraction: 0.2,
                        cleanup_cycles: 0,
                        cancellation: None,
                    },
                    None,
                )
                .unwrap()
            });
        },
    );
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
