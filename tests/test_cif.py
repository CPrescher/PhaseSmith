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
    with pytest.raises(ValueError, match="duplicate atom"):
        read_cif(duplicate)
    permissive = read_cif(duplicate, strict=False).structure
    assert tuple(site.site_id for site in permissive.sites) == ("C1", "C1#2")
    assert tuple(site.disorder_group for site in permissive.sites) == ("A", "B")
    assert permissive.sites[1].element_symbol == "H"
    assert permissive.sites[1].isotope == 2


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
        env={"PYTHONPATH": str(Path(__file__).parents[1] / "python")},
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
