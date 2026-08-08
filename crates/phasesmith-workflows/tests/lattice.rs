//! Native lattice parameterization, derivative, and guarded-domain contracts.

use std::collections::{BTreeMap, BTreeSet};

use phasesmith_crystallography::{PreparedReflectionGenerator, ReflectionRange, UnitCell};
use phasesmith_io::{space_group_by_number, space_group_by_symbol};
use phasesmith_workflows::{
    LatticeBounds, LatticeError, LatticeParameterization, LatticeReflectionDomain,
    cw_lattice_geometry,
};

fn lattice_cases() -> Vec<(i32, UnitCell, &'static [&'static str])> {
    vec![
        (
            1,
            UnitCell {
                a_angstrom: 4.1,
                b_angstrom: 5.2,
                c_angstrom: 6.3,
                alpha_deg: 77.0,
                beta_deg: 83.0,
                gamma_deg: 72.0,
            },
            &[
                "a_angstrom",
                "b_angstrom",
                "c_angstrom",
                "alpha_deg",
                "beta_deg",
                "gamma_deg",
            ],
        ),
        (
            10,
            UnitCell {
                a_angstrom: 4.0,
                b_angstrom: 5.0,
                c_angstrom: 6.0,
                alpha_deg: 90.0,
                beta_deg: 103.0,
                gamma_deg: 90.0,
            },
            &["a_angstrom", "b_angstrom", "c_angstrom", "beta_deg"],
        ),
        (
            47,
            UnitCell {
                a_angstrom: 4.0,
                b_angstrom: 5.0,
                c_angstrom: 6.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            &["a_angstrom", "b_angstrom", "c_angstrom"],
        ),
        (
            123,
            UnitCell {
                a_angstrom: 4.0,
                b_angstrom: 4.0,
                c_angstrom: 6.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            &["a_angstrom", "c_angstrom"],
        ),
        (
            164,
            UnitCell {
                a_angstrom: 4.0,
                b_angstrom: 4.0,
                c_angstrom: 6.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 120.0,
            },
            &["a_angstrom", "c_angstrom"],
        ),
        (
            191,
            UnitCell {
                a_angstrom: 4.0,
                b_angstrom: 4.0,
                c_angstrom: 6.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 120.0,
            },
            &["a_angstrom", "c_angstrom"],
        ),
        (
            221,
            UnitCell {
                a_angstrom: 4.0,
                b_angstrom: 4.0,
                c_angstrom: 4.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            &["a_angstrom"],
        ),
    ]
}

fn rhombohedral_case() -> (phasesmith_crystallography::SpaceGroup, UnitCell) {
    let group = space_group_by_symbol("R -3 m :R")
        .expect("rhombohedral setting")
        .space_group;
    let cell = UnitCell {
        a_angstrom: 5.0,
        b_angstrom: 5.0,
        c_angstrom: 5.0,
        alpha_deg: 75.0,
        beta_deg: 75.0,
        gamma_deg: 75.0,
    };
    (group, cell)
}

#[test]
fn parameterization_covers_every_crystal_system_and_rhombohedral_setting() {
    for (number, cell, expected_names) in lattice_cases() {
        let group = space_group_by_number(number)
            .expect("space group")
            .space_group;
        let parameterization =
            LatticeParameterization::new(group.clone(), cell).unwrap_or_else(|error| {
                panic!(
                    "space group {number} {:?} {:?}: {error}",
                    group.crystal_system(),
                    group.metric_constraints().parameterization_basis
                )
            });
        assert_eq!(parameterization.parameter_names(), expected_names);
        assert_eq!(
            parameterization.parameter_names().len(),
            group.metric_constraints().independent_parameter_count
        );
        let values = parameterization.values_from_cell(cell).expect("values");
        assert_eq!(parameterization.to_cell(&values).expect("cell"), cell);
        assert_cell_jacobian_matches_differences(&parameterization, &values);
    }

    let (group, cell) = rhombohedral_case();
    let system = group.crystal_system();
    let basis = group.metric_constraints().parameterization_basis.clone();
    let parameterization = LatticeParameterization::new(group, cell)
        .unwrap_or_else(|error| panic!("rhombohedral {system:?} {basis:?}: {error}"));
    assert_eq!(
        parameterization.parameter_names(),
        ["a_angstrom", "alpha_deg"]
    );
    let values = parameterization.values_from_cell(cell).expect("values");
    assert_cell_jacobian_matches_differences(&parameterization, &values);
}

fn assert_cell_jacobian_matches_differences(
    parameterization: &LatticeParameterization,
    values: &[f64],
) {
    let jacobian = parameterization.cell_jacobian(values).expect("jacobian");
    for parameter in 0..values.len() {
        let step = 1.0e-6;
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[parameter] += step;
        minus[parameter] -= step;
        let plus = cell_values(parameterization.to_cell(&plus).expect("plus cell"));
        let minus = cell_values(parameterization.to_cell(&minus).expect("minus cell"));
        for row in 0..6 {
            let difference = (plus[row] - minus[row]) / (2.0 * step);
            assert!((jacobian[row * values.len() + parameter] - difference).abs() < 4.0e-9);
        }
    }
}

#[test]
fn cw_geometry_derivatives_match_centered_differences() {
    let mut cases = lattice_cases()
        .into_iter()
        .map(|(number, cell, _)| {
            let group = space_group_by_number(number)
                .expect("space group")
                .space_group;
            (number.to_string(), group, cell)
        })
        .collect::<Vec<_>>();
    let (rhombohedral, rhombohedral_cell) = rhombohedral_case();
    cases.push(("rhombohedral".to_owned(), rhombohedral, rhombohedral_cell));
    let hkl = [[1, 0, 0], [1, 1, 0], [1, 1, 1]];
    for (label, group, cell) in cases {
        let parameterization = LatticeParameterization::new(group, cell)
            .unwrap_or_else(|error| panic!("space group {label}: {error}"));
        let values = parameterization.values_from_cell(cell).expect("values");
        let analytical = cw_lattice_geometry(&parameterization, cell, &hkl, 1.0).expect("geometry");
        for parameter in 0..values.len() {
            let step = 1.0e-6;
            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[parameter] += step;
            minus[parameter] -= step;
            let plus = cw_lattice_geometry(
                &parameterization,
                parameterization.to_cell(&plus).expect("plus cell"),
                &hkl,
                1.0,
            )
            .expect("plus geometry");
            let minus = cw_lattice_geometry(
                &parameterization,
                parameterization.to_cell(&minus).expect("minus cell"),
                &hkl,
                1.0,
            )
            .expect("minus geometry");
            for reflection in 0..hkl.len() {
                let index = reflection * values.len() + parameter;
                let spacing_difference = (plus.d_spacing_angstrom[reflection]
                    - minus.d_spacing_angstrom[reflection])
                    / (2.0 * step);
                let position_difference = (plus.two_theta_deg[reflection]
                    - minus.two_theta_deg[reflection])
                    / (2.0 * step);
                assert_relative_close(
                    analytical.d_d_spacing_d_parameters[index],
                    spacing_difference,
                    3.0e-8,
                    3.0e-9,
                );
                assert_relative_close(
                    analytical.d_two_theta_d_parameters[index],
                    position_difference,
                    4.0e-8,
                    4.0e-8,
                );
            }
        }
    }
}

#[test]
fn guarded_domain_contains_visible_families_and_transfers_intensities() {
    let group = space_group_by_number(123).expect("space group").space_group;
    let cell = UnitCell {
        a_angstrom: 4.0,
        b_angstrom: 4.0,
        c_angstrom: 6.0,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let parameterization = LatticeParameterization::new(group.clone(), cell).expect("parameters");
    let bounds = LatticeBounds::around(&parameterization, 0.1, 5.0).expect("bounds");
    let domain = LatticeReflectionDomain::new(
        parameterization.clone(),
        bounds.clone(),
        1.5406,
        [20.0, 70.0],
        0.25,
        true,
        50_000_000,
        1.001,
    )
    .expect("domain");
    let initial = domain.generate(cell, None).expect("initial domain");
    assert!(initial.reflection_ids.len() > initial.visible.iter().filter(|value| **value).count());
    assert_eq!(initial.reflection_ids.len(), initial.multiplicity.len());
    assert!(initial.multiplicity.iter().all(|value| *value > 0));
    assert_eq!(initial.preserved_reflection_count, 0);
    assert_eq!(initial.added_reflection_ids, initial.reflection_ids);

    let guarded_ids = initial
        .reflection_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let visible_generator =
        PreparedReflectionGenerator::new(group, true, 50_000_000).expect("generator");
    for corner in bounds.corner_values() {
        let corner_cell = parameterization.to_cell(&corner).expect("corner cell");
        let visible = visible_generator
            .generate(
                corner_cell,
                ReflectionRange::CwTwoTheta {
                    min_deg: 20.0,
                    max_deg: 70.0,
                    wavelength_angstrom: 1.5406,
                },
            )
            .expect("visible reflections");
        assert!(
            visible
                .iter()
                .all(|item| guarded_ids.contains(&item.reflection_id))
        );
    }

    let previous = initial
        .reflection_ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            let seed = u32::try_from(index).expect("small reflection test domain");
            (id.clone(), f64::from(seed) + 1.0)
        })
        .collect::<BTreeMap<_, _>>();
    let moved_cell = parameterization.to_cell(&[4.32, 5.52]).expect("moved cell");
    let moved = domain
        .generate(moved_cell, Some(&previous))
        .expect("moved domain");
    for (id, intensity) in moved.reflection_ids.iter().zip(&moved.integrated_intensity) {
        assert!((*intensity - previous.get(id).copied().unwrap_or(0.25)).abs() <= f64::EPSILON);
    }
    assert_eq!(
        moved.preserved_reflection_count,
        moved
            .reflection_ids
            .iter()
            .filter(|id| previous.contains_key(*id))
            .count()
    );

    let outside_cell = parameterization.to_cell(&[4.5, 6.0]).expect("outside cell");
    assert!(matches!(
        domain.generate(outside_cell, None),
        Err(LatticeError::OutsideBounds)
    ));
    let invalid_previous = BTreeMap::from([("hkl:1,0,0".to_owned(), f64::NAN)]);
    assert!(matches!(
        domain.generate(cell, Some(&invalid_previous)),
        Err(LatticeError::InvalidIntensity)
    ));
}

#[test]
fn reflection_domain_revalidates_bounds_against_its_parameterization() {
    let group = space_group_by_number(123).expect("space group").space_group;
    let original_cell = UnitCell {
        a_angstrom: 4.0,
        b_angstrom: 4.0,
        c_angstrom: 6.0,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let foreign_cell = UnitCell {
        a_angstrom: 8.0,
        b_angstrom: 8.0,
        c_angstrom: 9.0,
        ..original_cell
    };
    let original = LatticeParameterization::new(group.clone(), original_cell).expect("original");
    let foreign = LatticeParameterization::new(group, foreign_cell).expect("foreign");
    let foreign_bounds = LatticeBounds::around(&foreign, 0.05, 5.0).expect("foreign bounds");
    assert!(matches!(
        LatticeReflectionDomain::new(
            original,
            foreign_bounds,
            1.5406,
            [20.0, 70.0],
            1.0,
            true,
            50_000_000,
            1.001,
        ),
        Err(LatticeError::InvalidBounds)
    ));
}

#[test]
fn incompatible_cell_is_rejected_before_parameterization() {
    let cubic = space_group_by_number(221).expect("space group").space_group;
    let incompatible = UnitCell {
        a_angstrom: 4.0,
        b_angstrom: 4.1,
        c_angstrom: 4.0,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    assert!(matches!(
        LatticeParameterization::new(cubic, incompatible),
        Err(LatticeError::IncompatibleCell)
    ));
}

fn cell_values(cell: UnitCell) -> [f64; 6] {
    [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ]
}

fn assert_relative_close(left: f64, right: f64, relative: f64, absolute: f64) {
    assert!(
        (left - right).abs() <= absolute + relative * left.abs().max(right.abs()),
        "{left} != {right}"
    );
}
