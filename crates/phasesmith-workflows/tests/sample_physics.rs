//! Native built-in sample-physics value and derivative contracts.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use phasesmith_crystallography::UnitCell;
use phasesmith_workflows::RietveldSamplePhysicsModel;

fn cell() -> UnitCell {
    UnitCell {
        a_angstrom: 4.0,
        b_angstrom: 5.0,
        c_angstrom: 6.0,
        alpha_deg: 90.0,
        beta_deg: 100.0,
        gamma_deg: 90.0,
    }
}

fn perturb_cell(cell: UnitCell, parameter: usize, delta: f64) -> UnitCell {
    let mut values = [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ];
    values[parameter] += delta;
    UnitCell {
        a_angstrom: values[0],
        b_angstrom: values[1],
        c_angstrom: values[2],
        alpha_deg: values[3],
        beta_deg: values[4],
        gamma_deg: values[5],
    }
}

fn assert_close(actual: &[f64], expected: &[f64], relative: f64, absolute: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute.max(relative * expected.abs()),
            "value {index}: actual={actual:.15e}, expected={expected:.15e}"
        );
    }
}

#[test]
fn size_and_microstrain_parameter_and_position_derivatives_are_analytical() {
    let hkl = [[1, 0, 0], [1, 1, 0], [2, 1, 1]];
    let positions = [25.0, 47.0, 83.0];
    let step = 1.0e-5;
    for (model, plus, minus, gaussian) in [
        (
            RietveldSamplePhysicsModel::IsotropicSize {
                crystallite_size_nm: 75.0,
                shape_factor: 0.9,
            },
            RietveldSamplePhysicsModel::IsotropicSize {
                crystallite_size_nm: 75.0 + step,
                shape_factor: 0.9,
            },
            RietveldSamplePhysicsModel::IsotropicSize {
                crystallite_size_nm: 75.0 - step,
                shape_factor: 0.9,
            },
            false,
        ),
        (
            RietveldSamplePhysicsModel::IsotropicMicrostrain {
                rms_microstrain: 8.0e-4,
            },
            RietveldSamplePhysicsModel::IsotropicMicrostrain {
                rms_microstrain: 8.0e-4 + step,
            },
            RietveldSamplePhysicsModel::IsotropicMicrostrain {
                rms_microstrain: 8.0e-4 - step,
            },
            true,
        ),
    ] {
        let base = model.evaluate(&hkl, &positions, cell(), 1.5406).unwrap();
        let high = plus.evaluate(&hkl, &positions, cell(), 1.5406).unwrap();
        let low = minus.evaluate(&hkl, &positions, cell(), 1.5406).unwrap();
        let arrays = base.contributions.arrays();
        let values = if gaussian {
            &arrays.gaussian_variance_deg2
        } else {
            &arrays.lorentzian_fwhm_deg
        };
        let derivatives = if gaussian {
            &arrays.d_gaussian_variance_d_parameters
        } else {
            &arrays.d_lorentzian_fwhm_d_parameters
        };
        let high_values = if gaussian {
            &high.contributions.arrays().gaussian_variance_deg2
        } else {
            &high.contributions.arrays().lorentzian_fwhm_deg
        };
        let low_values = if gaussian {
            &low.contributions.arrays().gaussian_variance_deg2
        } else {
            &low.contributions.arrays().lorentzian_fwhm_deg
        };
        for index in 0..values.len() {
            let numerical = (high_values[index] - low_values[index]) / (2.0 * step);
            assert!((numerical - derivatives[index]).abs() < 2.0e-7);

            let mut high_positions = positions;
            let mut low_positions = positions;
            high_positions[index] += step;
            low_positions[index] -= step;
            let high = model
                .evaluate(&hkl, &high_positions, cell(), 1.5406)
                .unwrap();
            let low = model
                .evaluate(&hkl, &low_positions, cell(), 1.5406)
                .unwrap();
            let high_value = if gaussian {
                high.contributions.arrays().gaussian_variance_deg2[index]
            } else {
                high.contributions.arrays().lorentzian_fwhm_deg[index]
            };
            let low_value = if gaussian {
                low.contributions.arrays().gaussian_variance_deg2[index]
            } else {
                low.contributions.arrays().lorentzian_fwhm_deg[index]
            };
            let position_derivative = if gaussian {
                arrays.d_gaussian_variance_d_position[index]
            } else {
                arrays.d_lorentzian_fwhm_d_position[index]
            };
            let numerical = (high_value - low_value) / (2.0 * step);
            assert!((numerical - position_derivative).abs() < 2.0e-9);
        }
    }
}

#[test]
fn march_ratio_and_all_cell_derivatives_match_centered_differences() {
    let hkl = [[1, 0, 0], [1, 1, 0], [2, 1, 1], [-1, 2, 1]];
    let positions = [25.0, 47.0, 83.0, 106.0];
    let ratio = 0.78;
    let axis = [1.0, 0.3, -0.2];
    let model = RietveldSamplePhysicsModel::MarchDollase {
        ratio,
        preferred_axis_hkl: axis,
    };
    let actual = model.evaluate(&hkl, &positions, cell(), 1.5406).unwrap();
    let arrays = actual.contributions.arrays();
    let ratio_step = 1.0e-6;
    let plus = RietveldSamplePhysicsModel::MarchDollase {
        ratio: ratio + ratio_step,
        preferred_axis_hkl: axis,
    }
    .evaluate(&hkl, &positions, cell(), 1.5406)
    .unwrap();
    let minus = RietveldSamplePhysicsModel::MarchDollase {
        ratio: ratio - ratio_step,
        preferred_axis_hkl: axis,
    }
    .evaluate(&hkl, &positions, cell(), 1.5406)
    .unwrap();
    let numerical = plus
        .contributions
        .arrays()
        .intensity_multiplier
        .iter()
        .zip(&minus.contributions.arrays().intensity_multiplier)
        .map(|(plus, minus)| (plus - minus) / (2.0 * ratio_step))
        .collect::<Vec<_>>();
    assert_close(
        &arrays.d_intensity_multiplier_d_parameters[..hkl.len()],
        &numerical,
        3.0e-9,
        2.0e-10,
    );

    for parameter in 0..6 {
        let step = if parameter < 3 { 1.0e-5 } else { 1.0e-4 };
        let plus = model
            .evaluate(
                &hkl,
                &positions,
                perturb_cell(cell(), parameter, step),
                1.5406,
            )
            .unwrap();
        let minus = model
            .evaluate(
                &hkl,
                &positions,
                perturb_cell(cell(), parameter, -step),
                1.5406,
            )
            .unwrap();
        let numerical = plus
            .contributions
            .arrays()
            .intensity_multiplier
            .iter()
            .zip(&minus.contributions.arrays().intensity_multiplier)
            .map(|(plus, minus)| (plus - minus) / (2.0 * step))
            .collect::<Vec<_>>();
        let start = (parameter + 1) * hkl.len();
        assert_close(
            &arrays.d_intensity_multiplier_d_parameters[start..start + hkl.len()],
            &numerical,
            3.0e-8,
            3.0e-10,
        );
    }
}

#[test]
fn march_dollase_and_composition_have_stable_rows_and_product_rule() {
    let hkl = [[1, 0, 0], [0, 1, 0], [1, 1, 0]];
    let positions = [30.0, 45.0, 60.0];
    let march = RietveldSamplePhysicsModel::MarchDollase {
        ratio: 0.8,
        preferred_axis_hkl: [1.0, 0.0, 0.0],
    };
    let composite = RietveldSamplePhysicsModel::Composite(vec![
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 5.0e-4,
        },
        march.clone(),
    ]);
    let march_values = march.evaluate(&hkl, &positions, cell(), 1.5406).unwrap();
    let combined = composite
        .evaluate(&hkl, &positions, cell(), 1.5406)
        .unwrap();
    assert_eq!(
        combined.parameter_names,
        [
            "isotropic_microstrain.rms",
            "march_dollase.ratio",
            "march_dollase.cell.a_angstrom",
            "march_dollase.cell.b_angstrom",
            "march_dollase.cell.c_angstrom",
            "march_dollase.cell.alpha_deg",
            "march_dollase.cell.beta_deg",
            "march_dollase.cell.gamma_deg",
        ]
    );
    assert_eq!(combined.contributions.parameter_count(), 8);
    assert_eq!(
        combined.contributions.arrays().intensity_multiplier,
        march_values.contributions.arrays().intensity_multiplier
    );
    assert!(
        combined
            .contributions
            .arrays()
            .gaussian_variance_deg2
            .iter()
            .all(|value| *value > 0.0)
    );
}

#[test]
fn invalid_models_and_duplicate_composite_parameters_are_rejected() {
    let hkl = [[1, 0, 0]];
    let positions = [30.0];
    assert!(
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: -1.0,
        }
        .evaluate(&hkl, &positions, cell(), 1.5406)
        .is_err()
    );
    let duplicate = RietveldSamplePhysicsModel::Composite(vec![
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 1.0e-4,
        },
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 2.0e-4,
        },
    ]);
    assert!(
        duplicate
            .evaluate(&hkl, &positions, cell(), 1.5406)
            .is_err()
    );
}

#[test]
fn refinable_records_replace_exactly_and_disabled_size_stays_calculable() {
    let model = RietveldSamplePhysicsModel::Composite(vec![
        RietveldSamplePhysicsModel::IsotropicSize {
            crystallite_size_nm: 60.0,
            shape_factor: 0.88,
        },
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 4.0e-4,
        },
    ]);
    let parameters = model.parameters().unwrap();
    assert_eq!(
        parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect::<Vec<_>>(),
        [
            "isotropic_size.crystallite_size_nm",
            "isotropic_microstrain.rms"
        ]
    );
    let replacements = BTreeMap::from([
        ("isotropic_size.crystallite_size_nm".to_owned(), 80.0),
        ("isotropic_microstrain.rms".to_owned(), 7.0e-4),
    ]);
    let replaced = model.replace_parameters(&replacements).unwrap();
    assert_eq!(
        replaced
            .parameters()
            .unwrap()
            .iter()
            .map(|parameter| parameter.value)
            .collect::<Vec<_>>(),
        [80.0, 7.0e-4]
    );
    assert!(
        model
            .replace_parameters(&BTreeMap::from([(
                "isotropic_size.crystallite_size_nm".to_owned(),
                80.0,
            )]))
            .is_err()
    );

    let disabled = RietveldSamplePhysicsModel::IsotropicSize {
        crystallite_size_nm: f64::INFINITY,
        shape_factor: 0.9,
    };
    assert!(disabled.parameters().is_err());
    let evaluated = disabled
        .evaluate(&[[1, 0, 0]], &[30.0], cell(), 1.5406)
        .unwrap();
    assert_eq!(evaluated.contributions.arrays().lorentzian_fwhm_deg, [0.0]);
    assert!(
        model
            .evaluate(&[[1, 0, 0]], &[180.0], cell(), 1.5406)
            .is_err()
    );
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn native_models_match_python_values_and_derivatives_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let hkl = [[1, 0, 0], [1, 1, 0], [2, 1, 1], [-1, 2, 1]];
    let positions = [25.0, 47.0, 83.0, 106.0];
    let model = RietveldSamplePhysicsModel::Composite(vec![
        RietveldSamplePhysicsModel::IsotropicSize {
            crystallite_size_nm: 75.0,
            shape_factor: 0.9,
        },
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 8.0e-4,
        },
        RietveldSamplePhysicsModel::MarchDollase {
            ratio: 0.78,
            preferred_axis_hkl: [1.0, 0.3, -0.2],
        },
    ]);
    let native = model.evaluate(&hkl, &positions, cell(), 1.5406).unwrap();
    let arrays = native.contributions.arrays();
    let expected = [
        arrays.gaussian_variance_deg2.as_slice(),
        arrays.lorentzian_fwhm_deg.as_slice(),
        arrays.intensity_multiplier.as_slice(),
        arrays.d_gaussian_variance_d_position.as_slice(),
        arrays.d_lorentzian_fwhm_d_position.as_slice(),
        arrays.d_intensity_multiplier_d_position.as_slice(),
        arrays.d_gaussian_variance_d_parameters.as_slice(),
        arrays.d_lorentzian_fwhm_d_parameters.as_slice(),
        arrays.d_intensity_multiplier_d_parameters.as_slice(),
    ];
    let script = r#"
import numpy as np
import phasesmith

cell = phasesmith.UnitCell(4.0, 5.0, 6.0, 90.0, 100.0, 90.0)
hkl = np.asarray([[1, 0, 0], [1, 1, 0], [2, 1, 1], [-1, 2, 1]], dtype=np.int64)
positions = np.asarray([25.0, 47.0, 83.0, 106.0])
wavelength = 1.5406
d = wavelength / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
batch = phasesmith.ReflectionGeometryBatch(hkl, d, positions, np.ones(4))
instrument = phasesmith.ConstantWavelengthInstrument(wavelength, 2e-4, -1e-4, 1.2e-4, 1.5e-3, 3e-3)
metric = phasesmith.ReciprocalMetric(cell.geometry().reciprocal_metric)
provider = phasesmith.CompositePhysicsProvider((
    phasesmith.IsotropicSizeBroadening(75.0, 0.9),
    phasesmith.IsotropicMicrostrainBroadening(8e-4),
    phasesmith.MarchDollasePreferredOrientation(0.78, (1.0, 0.3, -0.2), metric),
))
value = provider.evaluate(phasesmith.PhysicsContext(batch, instrument, cell))
print("|".join(value.parameter_names))
for name in (
    "gaussian_variance_deg2", "lorentzian_fwhm_deg", "intensity_multiplier",
    "d_gaussian_variance_d_position", "d_lorentzian_fwhm_d_position",
    "d_intensity_multiplier_d_position", "d_gaussian_variance_d_parameters",
    "d_lorentzian_fwhm_d_parameters", "d_intensity_multiplier_d_parameters",
):
    print(" ".join(format(float(item), ".17g") for item in getattr(value, name).ravel()))
"#;
    let python_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python");
    let output = Command::new(python)
        .args(["-c", script])
        .env("PYTHONPATH", python_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python sample-physics oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next().unwrap().split('|').collect::<Vec<_>>(),
        native.parameter_names
    );
    let actual = lines.collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        let actual = actual
            .split_whitespace()
            .map(|value| value.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert_close(&actual, expected, 2.0e-13, 2.0e-14);
    }
}
