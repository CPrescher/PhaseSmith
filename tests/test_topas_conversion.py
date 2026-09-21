import json
from pathlib import Path

import numpy as np
import pytest
from phasesmith.io import read_cif, read_powder_data
from phasesmith.io.topas import convert_rowles_topas_bundle

TOPAS_TEMPLATE = """#include "row119.inc"
macro sample {{ {sample} }}
LP_Factor( 0)
Rp 250
Rs 250
start_X  21
finish_X  HAL
lpsd_th2_angular_range_degrees 2.994
lpsd_equitorial_divergence_degrees 0.3
Tube_Tails(, 0.04, , -0.555189814 min =-4; max =4;
           , , 0.772701111 min =-4; max =4;
           , , 0.00152295)
axial_conv
 filament_length  12
 sample_length  15
 receiving_slit_length  12
 primary_soller_angle    2.5
 secondary_soller_angle  2.5
lam
 la 0.0159 lo 1.534753 lh 3.6854
 la 0.5691 lo 1.540596 lh 0.437
 la 0.0762 lo 1.541058 lh 0.6
 la 0.2517 lo 1.54441 lh 0.52
 la 0.0871 lo 1.544721 lh 0.62
 la 0.4846078521 lo 1.392216 lh 0.5502872
 la 0.05 lo 1.4769 lh 0.5
prm !a_erf 492.78664
prm !edge_extra = 6.72369 / 1000;
prm !a_white = 279.471 / 1000000;
prm !b_white 61.20577
Absorption_Edge_Correction( 2,, 1.486
#prm a_cor = randomValue(4.758935, 0.01);
#prm c_cor = randomValue(12.991078, 0.05);
#prm a_flu = randomValue(5.463712, 0.02);
#prm a_zin = randomValue(3.249633, 0.01);
#prm c_zin = randomValue(5.206291, 0.02);
str
 phase_name "Corundum"
 site Al num_posns 12 x =0; y =0; z !z1_cor 0.3522 occ AL+3 1;
 site O num_posns 18 x !x2_cor 0.6937 y =0; z =1/4; occ O-2 1;
str
 phase_name "Fluorite"
str
 phase_name "Zincite"
 site O num_posns 2 x =1/3; y =2/3; z !z2_zincite 0.375 occ O-2 1;
"""


def source_bundle(root: Path) -> None:
    (root / "row119.inc").write_text("macro pinned {}\n", encoding="utf-8")
    pattern = np.array([[15.0, 100.0], [21.0, 120.0], [150.0, 20.0]])
    for sample in ("1a", "1e"):
        np.savetxt(root / f"{sample}_1000000_0-010_n001.xy", pattern)
        (root / f"robustness2_{sample}_4.INP").write_text(
            TOPAS_TEMPLATE.format(sample=sample), encoding="utf-8"
        )


def test_rowles_converter_materializes_common_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "converted"
    source.mkdir()
    source_bundle(source)

    manifest_path = convert_rowles_topas_bundle(source, destination)
    manifest = json.loads(manifest_path.read_text())

    radiation = manifest["common_model"]["radiation"]
    assert radiation["lambda1_angstrom"] == pytest.approx(1.5406505550906553)
    assert radiation["lambda2_angstrom"] == pytest.approx(1.5444899530696576)
    assert radiation["lambda2_over_lambda1_intensity"] == pytest.approx(0.5250271191693785)
    assert manifest["patterns"]["1a"]["weighed_weight_fractions"]["CaF2"] == 0.9481
    source_model = manifest["topas_source_model"]
    assert source_model["reference_line_index"] == 1
    assert len(source_model["emission_lines"]) == 7
    assert source_model["emission_lines"][0] == {
        "area": 0.0159,
        "wavelength_angstrom": 1.534753,
        "lorentzian_hwhm_milliangstrom": 3.6854,
        "reference": False,
    }
    assert source_model["emission_lines"][1]["reference"] is True
    assert source_model["absorption_edge_filter"] == {
        "kind": "error_function_high_pass",
        "edge_angstrom": 1.486,
        "sharpness_per_angstrom": 492.78664,
        "floor": pytest.approx(0.00672369),
    }
    assert source_model["angle_dependent_white_continuum"] == {
        "amplitude": pytest.approx(0.000279471),
        "gaussian_decay_per_angstrom2": 61.20577,
        "bragg_angle_factor": "1/tan(theta)",
        "supported": False,
    }
    assert source_model["instrument_geometry"] == {
        "source_to_sample_radius_mm": 250.0,
        "sample_to_detector_radius_mm": 250.0,
        "axial": {
            "filament_full_length_mm": 12.0,
            "illuminated_sample_full_length_mm": 15.0,
            "receiving_slit_full_length_mm": 12.0,
            "incident_soller_full_width_deg": 2.5,
            "diffracted_soller_full_width_deg": 2.5,
        },
        "linear_position_sensitive_detector": {
            "two_theta_angular_range_deg": 2.994,
            "equatorial_divergence_deg": 0.3,
        },
        "tube_tails": {
            "source_width_mm": 0.04,
            "left_tail_mm": -0.555189814,
            "right_tail_mm": 0.772701111,
            "relative_intensity": 0.00152295,
        },
    }
    assert len(manifest["translation"]["omitted"]) == 6
    assert read_powder_data(destination / "1a.xy", format="columns").x.size == 3
    for name in ("Al2O3.cif", "ZnO.cif", "CaF2.cif"):
        assert read_cif(destination / name).structure is not None


def test_rowles_converter_rejects_unrecognized_source_model(tmp_path: Path) -> None:
    source_bundle(tmp_path)
    path = tmp_path / "robustness2_1e_4.INP"
    path.write_text(path.read_text().replace("LP_Factor( 0)", "LP_Factor( 26.6)"))

    with pytest.raises(ValueError, match="LP_Factor"):
        convert_rowles_topas_bundle(tmp_path, tmp_path / "converted")
