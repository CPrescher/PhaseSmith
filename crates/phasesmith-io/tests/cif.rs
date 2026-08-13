//! Native, parser-independent CIF import contract tests.

use std::path::PathBuf;

use phasesmith_io::{
    CifIoError, CifReadLimits, DisplacementConvention, NATIVE_CIF_BACKEND, parse_cif_text,
    read_cif_file,
};

const P21_CIF: &str = r"
data_demo
_chemical_name_common 'Authored monoclinic test'
_chemical_formula_sum 'C Fe'
_cell_formula_units_Z 4
_chemical_formula_weight 123.45
_cell_length_a 4.200(5)
_cell_length_b 5.100(6)
_cell_length_c 6.300(7)
_cell_angle_alpha 90
_cell_angle_beta 101.00(2)
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1 21/c 1'
_space_group_IT_number 14
loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,y+1/2,-z+1/2'
'-x,-y,-z'
'x,-y+1/2,z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_B_iso_or_equiv
C1 13C 0.100(2) 0.200(3) 0.300(4) 0.50(2) 0.7895683521
Fe1 Fe3+ 0.4 0.5 0.6 ? .
loop_
_atom_site_aniso_label
_atom_site_aniso_B_11
_atom_site_aniso_B_22
_atom_site_aniso_B_33
_atom_site_aniso_B_23
_atom_site_aniso_B_13
_atom_site_aniso_B_12
C1 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0
";

const CELL_ONLY_CIF: &str = r"
data_cell_only
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'I m -3 m'
";

#[test]
fn imports_typed_sites_uncertainties_and_exact_symmetry() {
    let result = parse(P21_CIF);
    let structure = &result.structure;
    assert_eq!(result.selected_block, "demo");
    assert_eq!(structure.name, "Authored monoclinic test");
    assert_eq!(structure.source.backend, NATIVE_CIF_BACKEND);
    assert_eq!(structure.metadata["chemical_formula_sum"], "C Fe");
    assert_eq!(structure.metadata["formula_units_per_cell"], "4");
    assert_eq!(structure.metadata["formula_mass_g_mol"], "123.45");
    assert_eq!(structure.space_group.operations().len(), 4);
    assert_close(structure.cell_standard_uncertainties[0].unwrap(), 0.005);
    assert_close(structure.cell_standard_uncertainties[4].unwrap(), 0.02);
    assert_eq!(structure.sites.len(), 2);

    let carbon = &structure.sites[0];
    assert_eq!(carbon.type_symbol, "13C");
    assert_eq!(carbon.element_symbol, "C");
    assert_eq!(carbon.isotope, Some(13));
    assert_close(carbon.occupancy, 0.5);
    assert_close(carbon.occupancy_standard_uncertainty.unwrap(), 0.02);
    assert_eq!(
        carbon.fractional_xyz_standard_uncertainty,
        [Some(0.002), Some(0.003), Some(0.004)]
    );
    assert_close(carbon.u_iso_angstrom2.unwrap(), 0.01);
    let anisotropic = carbon.anisotropic_displacement.as_ref().unwrap();
    assert_eq!(anisotropic.source_convention, DisplacementConvention::CifB);
    for (actual, expected) in anisotropic.u_cif_angstrom2[..3]
        .iter()
        .zip([0.01, 0.02, 0.03])
    {
        assert_close(*actual, expected);
    }

    let iron = &structure.sites[1];
    assert_eq!(iron.charge, Some(3));
    assert_close(iron.occupancy, 1.0);
    assert!(iron.u_iso_angstrom2.is_none());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|item| item.code == "unknown_cif_value")
    );
}

#[test]
fn converts_cartesian_coordinates_and_preserves_visible_warning() {
    let text = r"
data_cart
_cell_length_a 4
_cell_length_b 5
_cell_length_c 6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_Cartn_x
_atom_site_Cartn_y
_atom_site_Cartn_z
X1 Si4+ 2.00(1) 1 3
";
    let result = parse(text);
    let site = &result.structure.sites[0];
    for (actual, expected) in site.fractional_xyz.iter().zip([0.5, 0.2, 0.5]) {
        assert_close(*actual, expected);
    }
    assert_eq!(site.element_symbol, "Si");
    assert_eq!(site.charge, Some(4));
    assert!(
        result
            .diagnostics
            .iter()
            .any(|item| { item.code == "cartesian_coordinate_uncertainty_not_transformed" })
    );
}

#[test]
fn selection_conflicts_and_limits_follow_strict_policy() {
    let second = CELL_ONLY_CIF
        .replace("cell_only", "second")
        .replace("_cell_length_a 4", "_cell_length_a 5")
        .replace("_cell_length_b 4", "_cell_length_b 5")
        .replace("_cell_length_c 4", "_cell_length_c 5");
    let document = format!("{CELL_ONLY_CIF}{second}");
    assert!(parse_cif_text(&document, None, true, CifReadLimits::default()).is_err());
    let selected =
        parse_cif_text(&document, Some("second"), true, CifReadLimits::default()).unwrap();
    assert_close(selected.structure.cell.a_angstrom, 5.0);
    let permissive = parse_cif_text(&document, None, false, CifReadLimits::default()).unwrap();
    assert_eq!(permissive.selected_block, "cell_only");
    assert_eq!(
        permissive.diagnostics[0].code,
        "multiple_blocks_first_selected"
    );

    let conflict = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'",
        "_space_group_name_H-M_alt 'P 1'\nloop_\n_space_group_symop_operation_xyz\n'x,y,z'\n'x+1/2,y+1/2,z+1/2'",
    );
    let strict_error = parse_cif_text(&conflict, None, true, CifReadLimits::default())
        .expect_err("strict conflicting symmetry must fail");
    assert!(strict_error.to_string().contains("disagree"));
    let permissive = parse_cif_text(&conflict, None, false, CifReadLimits::default()).unwrap();
    assert_eq!(permissive.structure.space_group.operations().len(), 2);
    assert!(
        permissive
            .diagnostics
            .iter()
            .any(|item| { item.code == "conflicting_space_group_definition" })
    );

    let limits = CifReadLimits {
        max_atom_sites: 1,
        ..CifReadLimits::default()
    };
    assert!(matches!(
        parse_cif_text(P21_CIF, None, true, limits),
        Err(CifIoError::Limit { .. })
    ));
}

#[test]
fn accepts_a_single_explicit_operation_stored_as_a_tag_value() {
    let text = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'",
        "_space_group_symop_operation_xyz 'x,y,z'",
    );
    let imported = parse(&text);
    assert_eq!(imported.structure.space_group.operations().len(), 1);
    assert_eq!(
        imported.structure.metadata["symmetry_source"],
        "explicit_operations"
    );
}

#[test]
fn imports_general_hall_expressions_outside_canonical_database_spellings() {
    let text = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'",
        "_symmetry_space_group_name_Hall 'A 2 -2ac'",
    );
    let imported = parse(&text);
    assert_eq!(imported.structure.space_group.operations().len(), 8);
    assert_eq!(imported.structure.metadata["symmetry_source"], "hall");
    assert_eq!(imported.structure.metadata["space_group_hall"], "A 2 -2ac");
}

#[test]
fn permissive_import_uses_valid_operations_without_guessing_malformed_hm() {
    let text = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'",
        "_space_group_name_H-M_alt 'F m 3 m'\n\
         _space_group_symop_operation_xyz 'x,y,z'",
    );
    assert!(parse_cif_text(&text, None, true, CifReadLimits::default()).is_err());

    let imported = parse_cif_text(&text, None, false, CifReadLimits::default()).unwrap();
    assert_eq!(imported.structure.space_group.operations().len(), 1);
    assert_eq!(
        imported.structure.metadata["symmetry_source"],
        "explicit_operations"
    );
    let diagnostic = imported
        .diagnostics
        .iter()
        .find(|item| item.code == "invalid_space_group_definition_ignored")
        .expect("malformed secondary HM tag must remain visible");
    assert_eq!(diagnostic.tag.as_deref(), Some("_space_group_name_h-m_alt"));
    assert!(diagnostic.message.contains("F m 3 m"));
}

#[test]
fn unsupported_features_and_duplicate_labels_are_explicit() {
    let magnetic = format!("{CELL_ONLY_CIF}\n_atom_site_moment.label M1\n");
    assert!(matches!(
        parse_cif_text(&magnetic, None, true, CifReadLimits::default()),
        Err(CifIoError::Unsupported { .. })
    ));
    let permissive = parse_cif_text(&magnetic, None, false, CifReadLimits::default()).unwrap();
    assert_eq!(
        permissive.diagnostics[0].code,
        "unsupported_magnetic_features"
    );

    let duplicate = r"
data_duplicate
_cell_length_a 3
_cell_length_b 3
_cell_length_c 3
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0
C1 D 0.5 0.5 0.5
";
    assert!(parse_cif_text(duplicate, None, true, CifReadLimits::default()).is_err());
    let imported = parse_cif_text(duplicate, None, false, CifReadLimits::default()).unwrap();
    assert_eq!(imported.structure.sites[0].site_id, "C1");
    assert_eq!(imported.structure.sites[1].site_id, "C1#2");
    assert_eq!(imported.structure.sites[1].element_symbol, "H");
    assert_eq!(imported.structure.sites[1].isotope, Some(2));
}

#[test]
#[ignore = "requires checksum-pinned external validation data"]
fn imports_pinned_real_small_structure_cifs_without_python() {
    let fixtures = [
        ("iucr-qarr-1g/Al2O3.cif", 36_usize),
        ("iucr-qarr-1g/CaF2.cif", 192),
        ("iucr-qarr-1g/ZnO.cif", 12),
        ("gsasii-pbso4-cw/PbSO4-Wyckoff.cif", 8),
    ];
    for (relative, operation_count) in fixtures {
        let path = validation_data().join(relative);
        let result = read_cif_file(&path, None, true, CifReadLimits::default())
            .unwrap_or_else(|error| panic!("{} failed native import: {error}", path.display()));
        assert!(
            !result.structure.sites.is_empty(),
            "{} has no sites",
            path.display()
        );
        assert_eq!(
            result.structure.space_group.operations().len(),
            operation_count,
            "unexpected operation count for {}",
            path.display()
        );
    }
}

fn parse(text: &str) -> phasesmith_io::CifReadResult {
    parse_cif_text(text, None, true, CifReadLimits::default()).unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-10,
        "{actual} != {expected}"
    );
}

fn validation_data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../validation/data")
}
