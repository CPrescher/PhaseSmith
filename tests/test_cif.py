from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import phasesmith
import pytest
from phasesmith.io.cif import CifReadLimits, CifReadResult, read_cif
from phasesmith.io.space_groups import space_group_by_number, space_group_from_hall_symbol
from phasesmith.refinement import lebail
from phasesmith.refinement.lebail import LeBailInput, LeBailPhase

P21_CIF = """
data_demo
_chemical_name_common 'Authored monoclinic test'
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
"""


CELL_ONLY_CIF = """
data_cell_only
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'I m -3 m'
"""


def test_script_first_cif_lebail_input_builds_bounded_lattice_parameters() -> None:
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    x = np.linspace(20.0, 90.0, 7_001)
    truth = LeBailPhase.from_cif(
        CELL_ONLY_CIF,
        phase_id="alpha",
        wavelength_angstrom=instrument.wavelength_angstrom,
        two_theta_min_deg=float(x[0]),
        two_theta_max_deg=float(x[-1]),
    )
    calculated = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument, (truth,))
    request = LeBailInput.from_cif(
        phasesmith.PowderPattern(x, observed_y=calculated.y),
        instrument,
        CELL_ONLY_CIF,
        phase_id="alpha",
    )

    phase = request.phases[0]
    assert isinstance(phase, LeBailPhase)
    assert phase.reflection_domain is not None
    assert phase.reflections_generated
    assert request.parameters is not None
    assert tuple(spec.key.name for spec in request.parameters.specs) == ("a_angstrom",)
    assert phase.structure.source is not None
    assert phase.structure.source.backend == "phasesmith-native"
    result = lebail.iterate_once(request)
    assert result.checkpoint.completed_iterations == 1


def test_cif_element_identity_reaches_scattering_tables() -> None:
    from phasesmith.scattering import species_from_structure

    header = CELL_ONLY_CIF + "\nloop_\n_atom_site_label\n"
    xyz = "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    inferred = read_cif(header + xyz + "FE1 0 0 0\nCL1 .1 .2 .3\nD1 .3 .2 .1").structure
    species = species_from_structure(inferred)
    assert [s.element_symbol for s in species] == ["Fe", "Cl", "H"]
    assert [s.xray_key for s in species] == ["Fe", "Cl", "H"]
    assert species[2].neutron_key == "H-2"
    phasesmith.XrayNonResonant().prepare(species)

    typed = read_cif(
        header
        + "_atom_site_type_symbol\n"
        + xyz
        + "C1 TI+4 0 0 0\nC2 57FE3+ 0 0 0\nC3 n 0 0 0\nC4 Cval 0 0 0"
        + "\nD2 D1+ 0 0 0\nT1 TA 0 0 0"
    ).structure
    species = species_from_structure(typed)
    assert [s.type_symbol for s in typed.sites] == ["TI+4", "57FE3+", "n", "Cval", "D1+", "TA"]
    assert [s.xray_key for s in species] == ["Ti4+", "Fe3+", "N", "Cval", "H1+", "Ta"]
    assert [s.neutron_key for s in species] == ["Ti", "Fe-57", "N", "C", "H-2", "Ta"]


def test_native_adapter_extracts_plain_typed_values_and_uncertainties() -> None:
    result = read_cif(P21_CIF)
    structure = result.structure
    assert result.selected_block == "demo"
    assert structure.name == "Authored monoclinic test"
    assert structure.source is not None
    assert structure.source.backend == "phasesmith-native"
    assert structure.space_group.crystal_system == "monoclinic"
    assert len(structure.space_group.operations) == 4
    assert structure.cell_standard_uncertainties[:3] == pytest.approx((0.005, 0.006, 0.007))
    assert structure.cell_standard_uncertainties[3] is None
    assert structure.cell_standard_uncertainties[4] == pytest.approx(0.02)
    assert structure.cell_standard_uncertainties[5] is None
    carbon, iron = structure.sites
    assert carbon.type_symbol == "13C"
    assert carbon.element_symbol == "C"
    assert carbon.isotope == 13
    assert carbon.occupancy == pytest.approx(0.5)
    assert carbon.occupancy_standard_uncertainty == pytest.approx(0.02)
    assert carbon.fractional_xyz_standard_uncertainty == pytest.approx((0.002, 0.003, 0.004))
    assert carbon.u_iso_angstrom2 == pytest.approx(0.01)
    assert carbon.anisotropic_displacement is not None
    assert carbon.anisotropic_displacement.source_convention == "B_cif"
    assert carbon.anisotropic_displacement.u_cif_angstrom2[:3] == pytest.approx((0.01, 0.02, 0.03))
    assert carbon.anisotropic_displacement.standard_uncertainty[0] == pytest.approx(
        8e-10 / (8.0 * np.pi**2)
    )
    assert iron.charge == 3
    assert iron.occupancy == 1.0
    assert iron.u_iso_angstrom2 is None
    assert any(item.code == "unknown_cif_value" for item in result.diagnostics)
    assert all("gemmi" not in type(value).__module__ for value in (structure, *structure.sites))


def test_cartesian_coordinates_and_legacy_tags_are_converted_explicitly() -> None:
    text = """
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
"""
    result = read_cif(text)
    site = result.structure.sites[0]
    assert site.fractional_xyz == pytest.approx((0.5, 0.2, 0.5), abs=2e-16)
    assert site.element_symbol == "Si"
    assert site.charge == 4
    assert site.fractional_xyz_standard_uncertainty == (None, None, None)
    assert any(
        item.code == "cartesian_coordinate_uncertainty_not_transformed"
        for item in result.diagnostics
    )


def test_multiple_blocks_require_selection_and_permissive_mode_is_visible() -> None:
    second = CELL_ONLY_CIF.replace("cell_only", "second").replace(
        "_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4",
        "_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5",
    )
    text = CELL_ONLY_CIF + second
    with pytest.raises(ValueError, match="explicit block"):
        read_cif(text)
    selected = read_cif(text, block="second")
    assert selected.selected_block == "second"
    permissive = read_cif(text, strict=False)
    assert permissive.selected_block == "cell_only"
    assert permissive.diagnostics[0].code == "multiple_blocks_first_selected"


def test_symmetry_precedence_and_duplicate_cell_conflicts_are_strict() -> None:
    conflicting_symmetry = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'",
        """_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
'x+1/2,y+1/2,z+1/2'""",
    )
    with pytest.raises(ValueError, match="disagree"):
        read_cif(conflicting_symmetry)
    permissive = read_cif(conflicting_symmetry, strict=False)
    assert len(permissive.structure.space_group.operations) == 2
    assert any(item.code == "conflicting_space_group_definition" for item in permissive.diagnostics)

    conflicting_cell = CELL_ONLY_CIF.replace(
        "_cell_length_a 4", "_cell.length_a 5\n_cell_length_a 4"
    ).replace(
        "_cell_length_b 4\n_cell_length_c 4",
        "_cell_length_b 5\n_cell_length_c 5",
    )
    with pytest.raises(ValueError, match="duplicate cell"):
        read_cif(conflicting_cell)
    assert read_cif(conflicting_cell, strict=False).structure.cell.a_angstrom == 5.0


def test_hall_number_disorder_and_duplicate_labels_have_defined_behavior() -> None:
    hall = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'", "_space_group.name_Hall 'P 1'"
    )
    assert read_cif(hall).structure.space_group == phasesmith.SpaceGroup.p1()
    numbered = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'", "_space_group.IT_number 229"
    )
    assert read_cif(numbered).structure.space_group.crystal_system == "cubic"

    duplicate = """
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
_atom_site_disorder_group
C1 C 0 0 0 A
C1 D 0.5 0.5 0.5 B
"""
    recovered = read_cif(duplicate)
    assert [s.source_label for s in recovered.structure.sites] == ["C1", "C1"]
    assert [s.site_id for s in recovered.structure.sites] == ["C1", "C1#2"]
    assert any(d.code == "duplicate_atom_label_renamed" for d in recovered.diagnostics)
    reserved = read_cif(duplicate + "C1#2 O 0.25 0 0 .\n").structure
    assert [s.site_id for s in reserved.sites] == ["C1", "C1#3", "C1#2"]
    for source in (
        duplicate + "_geom_bond_atom_site_label_1 C1\n",
        duplicate + "loop_\n_vendor_atom_label\n'C1/M2'\n",
        duplicate.replace("D 0.5 0.5 0.5", "C 1 0 0"),
        duplicate.replace("0.5 0.5 0.5", "0 0 0"),
    ):
        with pytest.raises(ValueError, match="duplicate atom label"):
            read_cif(source)
    inverse = duplicate.replace("loop_", "_space_group_name_H-M_alt 'P -1'\nloop_")
    inverse = inverse.replace("C 0 0 0", "C 0.25 0 0").replace("D 0.5 0.5 0.5", "C -0.25 0 0")
    with pytest.raises(ValueError, match="symmetry-equivalent"):
        read_cif(inverse)
    mixed = duplicate.replace(
        "_atom_site_disorder_group", "_atom_site_disorder_group\n_atom_site_occupancy"
    )
    mixed = mixed.replace("0 0 0 A", "0 0 0 A .65").replace("0.5 0.5 0.5 B", "0 0 0 B .35")
    assert [s.occupancy for s in read_cif(mixed).structure.sites] == [0.65, 0.35]
    permissive = read_cif(duplicate, strict=False).structure
    assert tuple(site.site_id for site in permissive.sites) == ("C1", "C1#2")
    assert tuple(site.disorder_group for site in permissive.sites) == ("A", "B")
    assert permissive.sites[1].element_symbol == "H"
    assert permissive.sites[1].isotope == 2


def test_general_hall_expression_is_parsed_without_a_legacy_backend() -> None:
    text = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'",
        "_symmetry_space_group_name_Hall 'A 2 -2ac'",
    )
    result = read_cif(text)
    assert len(result.structure.space_group.operations) == 8
    assert result.structure.metadata["symmetry_source"] == "hall"
    assert result.structure.metadata["space_group_hall"] == "A 2 -2ac"


def test_permissive_symmetry_precedence_does_not_repair_invalid_hm() -> None:
    text = CELL_ONLY_CIF.replace(
        "_space_group_name_H-M_alt 'I m -3 m'",
        """_space_group_name_H-M_alt 'F m 5 m'
_space_group_symop_operation_xyz 'x,y,z'""",
    )
    with pytest.raises(ValueError, match="F m 5 m"):
        read_cif(text)

    result = read_cif(text, strict=False)
    assert result.structure.space_group == phasesmith.SpaceGroup.p1()
    assert result.structure.metadata["symmetry_source"] == "explicit_operations"
    diagnostic = next(
        item for item in result.diagnostics if item.code == "invalid_space_group_definition_ignored"
    )
    assert diagnostic.tag == "_space_group_name_h-m_alt"
    assert "F m 5 m" in diagnostic.message


def test_missing_required_values_and_conflicting_anisotropic_definitions_fail() -> None:
    with pytest.raises(ValueError, match="required CIF value"):
        read_cif(CELL_ONLY_CIF.replace("_cell_length_a 4", "_cell_length_a ?"))
    conflicting = P21_CIF.replace(
        "_atom_site_aniso_B_12\nC1",
        """_atom_site_aniso_B_12
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
C1""",
    ).replace(
        "C1 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0",
        "C1 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0 1 1 1 0 0 0",
    )
    with pytest.raises(ValueError, match="anisotropic U and B"):
        read_cif(conflicting)

    orphan = P21_CIF.replace(
        "C1 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0",
        "Z9 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0",
    )
    with pytest.raises(ValueError, match="no matching atom"):
        read_cif(orphan)
    assert any(
        item.code == "orphan_anisotropic_rows"
        for item in read_cif(orphan, strict=False).diagnostics
    )

    duplicate = P21_CIF.replace(
        "C1 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0",
        "C1 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0\n"
        "C1 7.895683521 1.5791367042 2.3687050563 0 0 0",
    )
    with pytest.raises(ValueError, match="duplicate anisotropic"):
        read_cif(duplicate)
    permissive = read_cif(duplicate, strict=False)
    assert permissive.structure.sites[0].anisotropic_displacement is not None
    assert permissive.structure.sites[0].anisotropic_displacement.u_cif_angstrom2[
        0
    ] == pytest.approx(0.01)
    assert any(
        item.code == "duplicate_anisotropic_label_ignored" for item in permissive.diagnostics
    )


def test_cell_and_symmetry_only_cif_constructs_a_fixed_cell_lebail_phase() -> None:
    imported = read_cif(CELL_ONLY_CIF)
    assert imported.structure.sites == ()
    phase = LeBailPhase.from_structure(
        imported.structure,
        phase_id="alpha",
        wavelength_angstrom=1.5406,
        two_theta_min_deg=10.0,
        two_theta_max_deg=120.0,
    )
    assert isinstance(phase, phasesmith.Phase)
    assert phase.structure is imported.structure
    assert phase.reflections.reflection_count > 0
    assert np.all(phase.reflections.integrated_intensity == 1.0)
    assert np.all(
        (phase.reflections.two_theta_deg >= 10) & (phase.reflections.two_theta_deg <= 120)
    )
    assert replace(phase, scale=2.0).structure is imported.structure

    direct = LeBailPhase.from_cif(
        CELL_ONLY_CIF,
        phase_id="direct",
        wavelength_angstrom=1.5406,
        two_theta_min_deg=10.0,
        two_theta_max_deg=120.0,
    )
    np.testing.assert_array_equal(direct.reflections.hkl, phase.reflections.hkl)


def test_structure_record_round_trip_requires_no_parser_objects() -> None:
    original = read_cif(P21_CIF).structure
    encoded = json.dumps(phasesmith.structure_to_record(original))
    restored = phasesmith.structure_from_record(json.loads(encoded))
    assert restored.cell == original.cell
    assert restored.space_group == original.space_group
    assert restored.sites == original.sites
    assert restored.metadata == original.metadata
    with pytest.raises(NotImplementedError, match="anisotropic"):
        restored.to_isotropic_site_batch()


def test_resource_limits_and_unsupported_features_fail_visibly(tmp_path: Path) -> None:
    path = tmp_path / "large.cif"
    path.write_text(CELL_ONLY_CIF, encoding="utf-8")
    with pytest.raises(ValueError, match="max_bytes"):
        read_cif(path, limits=CifReadLimits(max_bytes=10))
    with pytest.raises(ValueError, match="max_atom_sites"):
        read_cif(P21_CIF, limits=CifReadLimits(max_atom_sites=1))

    magnetic = CELL_ONLY_CIF + "\n_atom_site_moment.label M1\n"
    with pytest.raises(NotImplementedError, match="magnetic"):
        read_cif(magnetic)
    permissive = read_cif(magnetic, strict=False)
    assert permissive.diagnostics[0].code == "unsupported_magnetic_features"


def test_backend_protocol_is_injectable_and_base_import_does_not_load_gemmi() -> None:
    structure = phasesmith.CrystalStructure(
        "fake",
        "Fake",
        phasesmith.UnitCell(1, 1, 1, 90, 90, 90),
        phasesmith.SpaceGroup.p1(),
    )

    class FakeBackend:
        name = "fake"
        version = "1"

        def parse_text(self, text, *, source_name, block, strict, limits):
            del text, source_name, block, strict, limits
            return CifReadResult(structure, (), "fake", ("fake",))

    assert read_cif("data_fake", backend=FakeBackend()).structure is structure

    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockGemmi(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == 'gemmi' or fullname.startswith('gemmi.'):
                    raise RuntimeError('gemmi import attempted')
                return None
        sys.meta_path.insert(0, BlockGemmi())
        import phasesmith
        from phasesmith.io.cif import read_cif
        assert 'gemmi' not in sys.modules
        assert phasesmith.UnitCell(1, 1, 1, 90, 90, 90).a_angstrom == 1
        result = read_cif('''data_test
        _cell_length_a 1
        _cell_length_b 1
        _cell_length_c 1
        _cell_angle_alpha 90
        _cell_angle_beta 90
        _cell_angle_gamma 90
        ''')
        assert result.structure.source.backend == 'phasesmith-native'
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "relative_path",
    (
        "iucr-qarr-1g/Al2O3.cif",
        "iucr-qarr-1g/CaF2.cif",
        "iucr-qarr-1g/ZnO.cif",
        "gsasii-pbso4-cw/PbSO4-Wyckoff.cif",
    ),
)
@pytest.mark.real_data
def test_native_cif_matches_optional_gemmi_oracle(relative_path: str) -> None:
    pytest.importorskip("gemmi")
    from phasesmith.io._gemmi import GemmiCifBackend

    path = Path(__file__).parents[1] / "validation" / "data" / relative_path
    native = read_cif(path).structure
    oracle = read_cif(path, backend=GemmiCifBackend()).structure
    assert native.cell.as_tuple() == pytest.approx(oracle.cell.as_tuple(), abs=1e-12)
    assert native.space_group == oracle.space_group
    assert len(native.sites) == len(oracle.sites)
    for actual, expected in zip(native.sites, oracle.sites, strict=True):
        assert actual.site_id == expected.site_id
        assert actual.element_symbol == expected.element_symbol
        assert actual.fractional_xyz == pytest.approx(expected.fractional_xyz, abs=1e-12)
        assert actual.occupancy == pytest.approx(expected.occupancy, abs=1e-12)


def test_empty_site_table_is_cell_only_when_every_field_is_missing() -> None:
    table = """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_description
? . ? . ?
"""
    result = read_cif(CELL_ONLY_CIF + table)
    assert not result.structure.sites
    assert any(d.code == "empty_atom_site_table" for d in result.diagnostics)
    for row in ["? 0 ? . ?", "? . ? . Fe", "? . ? . '.'", "? . ? . ?\n? . ? . ?"]:
        with pytest.raises(ValueError, match="label is missing"):
            read_cif(CELL_ONLY_CIF + table.replace("? . ? . ?", row))


def test_hash_labels_and_displacement_table_placeholders() -> None:
    source = P21_CIF.replace("C1 ", "C#1 ").replace("Fe1 Fe3+", "# genuine comment\nFe1 Fe3+")
    assert read_cif(source).structure.sites[0].source_label == "C#1"
    row = "C#1 0.7895683521(8) 1.5791367042 2.3687050563 0 0 0"
    result = read_cif(source.replace(row, "? . ? . ? . ?"))
    assert result.structure.sites[0].anisotropic_displacement is None
    assert any(d.code == "empty_anisotropic_table" for d in result.diagnostics)
    with pytest.raises(ValueError, match="anisotropic label"):
        read_cif(source.replace(row, "? . ? . ? . 0"))


def test_numbered_tensor_labels_require_a_unique_unclaimed_site() -> None:
    raw = P21_CIF.replace("C1 0.7895683521(8)", "C(1) 0.7895683521(8)")
    result = read_cif(raw)
    assert (
        result.structure.sites[0].anisotropic_displacement
        == read_cif(P21_CIF).structure.sites[0].anisotropic_displacement
    )
    assert any(d.code == "matched_numbered_anisotropic_label" for d in result.diagnostics)
    with pytest.raises(ValueError, match="no matching atom sites"):
        read_cif(raw + "C1 0.1 0.2 0.3 0 0 0\n")


def test_empty_uncertainty_preserves_nominal_and_truncation_is_rejected() -> None:
    result = read_cif(P21_CIF.replace("0.100(2)", "0.100()"))
    site = result.structure.sites[0]
    assert site.fractional_xyz[0] == pytest.approx(0.1)
    assert site.fractional_xyz_standard_uncertainty[0] is None
    assert any(d.code == "empty_standard_uncertainty" and d.row == 0 for d in result.diagnostics)
    with pytest.raises(ValueError):
        read_cif(P21_CIF.replace("0.100(2)", "0.100(2"))


def test_fractional_oxidation_state_retains_exact_scattering_identity() -> None:
    result = read_cif(P21_CIF.replace("Fe3+", "Fe3.5+"))
    site = result.structure.sites[1]
    assert (site.type_symbol, site.element_symbol, site.charge) == ("Fe3.5+", "Fe", None)
    assert any(d.code == "fractional_oxidation_state" for d in result.diagnostics)
    species = phasesmith.ScatteringSpecies.from_atom_site(site)
    assert (species.xray_key, species.neutron_key) == ("Fe3.5+", "Fe")
    with pytest.raises(ValueError, match="unknown X-ray"):
        phasesmith.XrayNonResonant().prepare([species])
    integral = read_cif(P21_CIF.replace("Fe3+", "Fe+3.00")).structure.sites[1]
    assert integral.charge == 3
    assert phasesmith.ScatteringSpecies.from_atom_site(integral).xray_key == "Fe3+"


def test_deposited_origin_is_verified_with_one_shift_for_every_operation() -> None:
    base = CELL_ONLY_CIF.replace("I m -3 m", "P 1 21 1")
    shifted = (
        base
        + """_space_group_IT_number 4
loop_
_space_group_symop_operation_xyz
x,y,z
-x,y+.500,-z+0.50
"""
    )
    result = read_cif(shifted)
    shifted_group = space_group_from_hall_symbol("P 2ybc")
    assert result.structure.space_group == shifted_group
    assert result.structure.space_group != read_cif(base).structure.space_group
    for other in [shifted.replace("y+.500", "y"), shifted + "_space_group_name_Hall 'P 2yb'"]:
        with pytest.raises(ValueError, match="disagree"):
            read_cif(other)
    with pytest.raises(ValueError):
        read_cif(shifted.replace(".500", ".4999"))
    cubic = CELL_ONLY_CIF.replace("I m -3 m", "Fm3m")
    assert read_cif(cubic).structure.space_group == space_group_by_number(225).space_group


def test_group_four_centered_c_axis_supercell_preserves_deposited_frame() -> None:
    cell = CELL_ONLY_CIF.replace("_space_group_name_H-M_alt 'I m -3 m'", "").replace(
        "_cell_angle_gamma 90", "_cell_angle_gamma 89.58"
    )
    hm = "_space_group_name_H-M_alt 'P 1 21 1 (c,2*a+c,b)'\n"
    hall = "_space_group_name_Hall 'P 2yb (-1/2*x+z,1/2*x,y)'\n"
    number = "_space_group_IT_number 4\n"
    operators = """loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,-y,z+1/2'
'x+1/2,y+1/2,z'
'-x+1/2,-y+1/2,z+1/2'
"""
    site = """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ru1 .57 .98 .30
"""
    expected = read_cif(cell + operators).structure.space_group
    for declaration in (
        hm,
        hall,
        hm.replace("P 1 21 1 (c,2*a+c,b)", "C 1 1 21"),
        hall + hm + number + operators,
    ):
        structure = read_cif(cell + declaration + site).structure
        assert structure.space_group == expected
        assert structure.cell.gamma_deg == 89.58
        assert tuple(structure.sites[0].fractional_xyz) == (0.57, 0.98, 0.30)
    for conflict in (number.replace("4", "3") + hm + hall, hm + operators.replace("z+1/2", "z")):
        with pytest.raises(ValueError, match="disagree"):
            read_cif(cell + conflict)


def test_centered_triclinic_setting_checks_each_declaration() -> None:
    text = (
        CELL_ONLY_CIF.replace("I m -3 m", "C -1")
        + """_space_group_IT_number 2
_space_group_name_Hall '-P 1 (1/2*x+1/2*y,1/2*x-1/2*y,-z)'
"""
    )
    group = read_cif(text).structure.space_group
    assert group == space_group_from_hall_symbol("-C 1")
    assert len(group.operations) == 4
    for conflict in [text.replace("C -1", "I -1"), text.replace("number 2", "number 1")]:
        with pytest.raises(ValueError, match="disagree"):
            read_cif(conflict)


def test_rounded_symmetry_equivalent_cell_keeps_the_reported_uncertainty() -> None:
    text = (
        CELL_ONLY_CIF.replace("length_a 4", "length_a 3.540(2)")
        .replace("length_b 4", "length_b 3.5398")
        .replace("length_c 4", "length_c 3.5398")
    )
    result = read_cif(text)
    assert result.structure.cell.as_tuple()[:3] == (3.5398,) * 3
    assert result.structure.cell_standard_uncertainties[:3] == (0.002, None, None)
    assert any("3.540(2)" in d.message and "3.5398" in d.message for d in result.diagnostics)
    for conflict in [
        text.replace("3.540(2)", "3.541(9)"),
        text.replace("length_c 3.5398", "length_c 3.5399"),
        text.replace("3.540(2)", "3.540"),
    ]:
        with pytest.raises(ValueError, match="metric"):
            read_cif(conflict)


@pytest.mark.parametrize(
    ("spelling", "hall"), [("P42 mnm", "-P 4n 2n"), ("P 42/n m c S", "P 4n 2n -1n")]
)
def test_incomplete_archive_spelling_requires_exact_corroboration(spelling: str, hall: str) -> None:
    base = CELL_ONLY_CIF.replace("I m -3 m", spelling)
    with pytest.raises(ValueError, match="unknown"):
        read_cif(base)
    with pytest.raises(ValueError):
        read_cif(base + "_space_group_name_Hall 'P 1'\n")
    result = read_cif(base + f"_space_group_name_Hall '{hall}'\n")
    assert result.structure.space_group == space_group_from_hall_symbol(hall)
    assert any(d.code == "corroborated_space_group_spelling" for d in result.diagnostics)


@pytest.mark.parametrize(
    ("labels", "tensor_label"), [(("Ti1", "Mg1"), "Ti1/Mg1"), (("M1(Ti)", "M1(Mg)"), "M1")]
)
def test_mixed_site_tensor_needs_explicit_membership_and_identical_coordinates(
    labels, tensor_label
) -> None:
    table = f"""loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_adp_type
_atom_site_occupancy
{labels[0]} Ti 0.2 0.1 0.4 Uani .5
{labels[1]} Mg 0.2 0.1 0.4 Uani .5
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
{tensor_label} .01 .02 .03 0 0 0
"""
    source = CELL_ONLY_CIF + table
    result = read_cif(source)
    first, second = result.structure.sites
    assert first.anisotropic_displacement == second.anisotropic_displacement
    assert first.anisotropic_displacement.u_cif_angstrom2[:3] == (0.01, 0.02, 0.03)
    assert sum(d.code == "matched_mixed_site_tensor" for d in result.diagnostics) == 2
    for conflict in [
        source.replace("Mg 0.2", "Mg 0.3"),
        source.replace("Uani", "Uiso", 1),
        source + f"{labels[0]} .01 .02 .03 0 0 0\n",
    ]:
        with pytest.raises(ValueError, match="no matching atom sites"):
            read_cif(conflict)
    if tensor_label == "M1":
        with pytest.raises(ValueError, match="no matching atom sites"):
            read_cif(source.replace("M1(Mg) Mg", "M1(Mg) Fe"))


_LABEL_CELL = CELL_ONLY_CIF.replace("'I m -3 m'", "'P 1'")
_SITE_XYZ = """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
"""
_U_TENSORS = """loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
"""


def test_strict_repeated_geometry_requires_a_unique_measured_bond_and_angle() -> None:
    source = (
        _LABEL_CELL
        + _SITE_XYZ
        + """C1 0 0 0
H1 .25 0 0
H1 0 .5 0
N1 0 .25 0
loop_
_geom_bond_atom_site_label_1
_geom_bond_atom_site_label_2
_geom_bond_distance
_geom_bond_site_symmetry_2
C1 H1 1.00(2) .
C1 N1 1.00(2) .
loop_
_geom_angle_atom_site_label_1
_geom_angle_atom_site_label_2
_geom_angle_atom_site_label_3
_geom_angle
H1 C1 N1 90.0(2)
"""
    )
    result = read_cif(source)
    assert [s.site_id for s in result.structure.sites] == ["C1", "H1", "H1#2", "N1"]
    assert any(d.code == "resolved_geometry_labels" for d in result.diagnostics)
    for inconsistent in (
        source.replace("0 .5 0", "0 -.25 0"),  # equal distances
        source.replace("0 .5 0", "0 -.26 0"),  # alternative within three uncertainties
        source.replace("90.0(2)", "60.0(2)"),
        source.replace("C1 H1 1.00(2) .", "C1 H1 1.00(2) ?"),
        source.replace("C1 H1 1.00(2) .", "C1 H1 1.00(2) 2_555"),
        source.replace("C1 H1 1.00(2)", "C1 H1 1.00"),
        source.replace("C1 N1 1.00(2) .", ""),
        source.replace("C1 N1 1.00(2)", "C1 N1 1.50(2)"),
        source + "_restr_special 'H1/N1'\n",
        source + "loop_\n_atom_site.fract_x\n0 .25 0 0\n",
        source + "_geom_bond.distance 2.0\n",  # alias outside the verified loop
    ):
        with pytest.raises(ValueError, match="duplicate atom label"):
            read_cif(inconsistent)


def test_strict_tensor_assignment_is_independent_of_row_order() -> None:
    sites = (
        _LABEL_CELL
        + _SITE_XYZ
        + """_atom_site_U_iso_or_equiv
_atom_site_adp_type
O1 0 0 0 .020(2) Uani
O1 .3 .3 .3 .040(2) Uani
"""
    )
    first = "O1 .010(1) .020 .030 0 0 0\n"
    second = "O1 .030(2) .040 .050 0 0 0\n"
    for rows in [second + first, first + second]:
        result = read_cif(sites + _U_TENSORS + rows)
        tensors = [s.anisotropic_displacement for s in result.structure.sites]
        assert [t.u_cif_angstrom2[0] for t in tensors] == [0.010, 0.030]
        assert [t.standard_uncertainty[0] for t in tensors] == [0.001, 0.002]
        assert sum(d.code == "matched_tensor_by_u_equiv" for d in result.diagnostics) == 2
    # At beta=120 degrees the metric gives (4/3 * (U11+U33-U13) + U22) / 3.
    oblique = sites.replace("_cell_angle_beta 90", "_cell_angle_beta 120")
    oblique = oblique.replace(".020(2) Uani", ".023(2) Uani").replace(
        ".040(2) Uani", ".045(2) Uani"
    )
    oblique_rows = second.replace(" 0 0 0\n", " 0 .009 0\n") + first.replace(
        " 0 0 0\n", " 0 .003 0\n"
    )
    matched = read_cif(oblique + _U_TENSORS + oblique_rows).structure.sites
    assert [s.anisotropic_displacement.u_cif_angstrom2[4] for s in matched] == [0.003, 0.009]
    assert read_cif(sites.replace(".040(2) Uani", ".040(2) Uiso") + _U_TENSORS + first)
    for source in (
        sites + _U_TENSORS + first,
        sites + _U_TENSORS + first + first,
        sites.replace(".040(2)", ".020(2)") + _U_TENSORS + first + second,
        sites.replace(".020(2)", ".021(2)") + _U_TENSORS + first + second,
    ):
        with pytest.raises(ValueError, match="ambiguous displacement tensor"):
            read_cif(source)


def test_strict_redundant_orphan_requires_exact_rows_and_disorder_evidence() -> None:
    sites = (
        _LABEL_CELL
        + _SITE_XYZ
        + """_atom_site_adp_type
_atom_site_occupancy
_atom_site_disorder_assembly
_atom_site_disorder_group
C1 0 0 0 Uani 1 A 1
C2 .25 .25 .25 Uiso .40 A 2
"""
    )
    tensor = ".010(1) .020 .030 0 0 0"
    source = sites + _U_TENSORS + f"C1 {tensor}\nC1A {tensor}\n"
    result = read_cif(source)
    assert [s.source_label for s in result.structure.sites] == ["C1", "C2"]
    assert result.structure.sites[0].anisotropic_displacement.standard_uncertainty[0] == 0.001
    assert any(d.code == "redundant_disorder_tensor" for d in result.diagnostics)
    for inconsistent in (
        source.replace("C1A .010(1)", "C1A .010(2)"),
        source.replace("Uani 1", "Uani .40"),
        source.replace("A 2", "A 1"),
        source.replace("C1A ", "C3A "),
        source + "_geom_bond_atom_site_label_1 C1A\n",
        source.replace("C1 .010", "_vendor_note\nC1 .010")
        .replace("0 0 0\nC1A", "0 0 0 yes\nC1A")
        .rstrip()
        + " no\n",
    ):
        with pytest.raises(ValueError, match="no matching atom sites"):
            read_cif(inconsistent)
