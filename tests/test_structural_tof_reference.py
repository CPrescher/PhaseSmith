from __future__ import annotations

import numpy as np
import phasesmith
from phasesmith import scattering_reference
from phasesmith.crystallography_reference import (
    reference_cell_geometry,
    reference_structural_tof_pattern,
    reference_time_of_flight_neutron_lorentz,
)


def test_structural_tof_reference_matches_native_component_composition() -> None:
    structure = phasesmith.CrystalStructure(
        "tof-reference",
        "TOF reference",
        phasesmith.UnitCell(4.7, 5.1, 6.2, 82.0, 87.0, 74.0),
        phasesmith.SpaceGroup([phasesmith.SymmetryOperation.identity()]),
        (
            phasesmith.AtomSite("si", "Si1", "Si", "Si", (0.17, 0.23, 0.31), 0.82, 0.012),
            phasesmith.AtomSite("o", "O1", "O", "O", (0.37, 0.11, 0.19), 0.55, 0.018),
        ),
    )
    hkl = np.array([[1, 0, 1], [2, 1, 1], [1, 2, 3]], dtype=np.int32)
    multiplicity = np.array([2, 4, 2], dtype=np.uint64)
    instrument = phasesmith.TofInstrument(
        1.2,
        5_000.0,
        0.2,
        0.0,
        0.2,
        0.03,
        0.001,
        0.0,
        25.0,
        4.0,
        0.1,
        0.0,
        1.0,
        0.1,
        0.5,
    )
    tof_us = np.linspace(1_000.0, 25_000.0, 2_401)
    geometry = reference_cell_geometry(structure.cell)
    h_float = hkl.astype(np.float64)
    q_squared = np.einsum(
        "ri,ij,rj->r", h_float, geometry.reciprocal_metric, h_float
    )
    scattering_real, _ = scattering_reference.neutron_nuclear(["Si", "O"], hkl.shape[0])
    correction, _ = reference_time_of_flight_neutron_lorentz(q_squared, 88.05)
    expected = reference_structural_tof_pattern(
        structure,
        hkl,
        multiplicity,
        scattering_real.astype(np.complex128),
        correction,
        tof_us,
        instrument,
        scale=1.3,
        quadrature_order=768,
    )

    actual_structural = phasesmith.calculate_structure_factors(
        structure,
        hkl,
        multiplicity,
        phasesmith.NeutronNuclear(),
        correction=phasesmith.TimeOfFlightNeutronLorentz(88.05),
        scale=1.3,
    )
    actual = phasesmith.accumulate_tof(
        tof_us,
        expected.d_spacing_angstrom,
        actual_structural.integrated_intensity,
        instrument,
    )
    np.testing.assert_allclose(
        actual_structural.integrated_intensity,
        expected.structure_factors.integrated_intensity,
        rtol=4.0e-15,
        atol=4.0e-12,
    )
    # The independent high-order Legendre integration and the production
    # fixed quadrature use different nodes; their largest relative difference
    # is in the very small asymmetric tails.
    np.testing.assert_allclose(actual.y, expected.y, rtol=2.0e-6, atol=3.0e-12)
