from __future__ import annotations

import numpy as np
import phasesmith
import pytest
from phasesmith import scattering_reference


def test_xray_native_batch_matches_independent_reference_and_finite_differences() -> None:
    species = (
        phasesmith.ScatteringSpecies("H"),
        phasesmith.ScatteringSpecies("C"),
        phasesmith.ScatteringSpecies("O"),
        phasesmith.ScatteringSpecies("Si"),
        phasesmith.ScatteringSpecies("Fe"),
        phasesmith.ScatteringSpecies("Fe", charge=3),
    )
    s = np.linspace(0.0, 5.9, 43)
    prepared = phasesmith.XrayNonResonant().prepare(species)
    actual = prepared.evaluate(s)
    expected, expected_derivative = scattering_reference.xray_non_resonant(
        [value.xray_key for value in species], s
    )
    np.testing.assert_allclose(actual.amplitudes.real, expected, rtol=2e-15, atol=2e-14)
    np.testing.assert_allclose(
        actual.d_amplitudes_d_s.real, expected_derivative, rtol=3e-15, atol=3e-14
    )
    np.testing.assert_array_equal(actual.amplitudes.imag, 0.0)
    np.testing.assert_array_equal(actual.d_amplitudes_d_s.imag, 0.0)

    center = np.array([0.7, 1.3, 2.1])
    step = 1e-6
    analytical = prepared.evaluate(center).d_amplitudes_d_s
    finite_difference = (
        prepared.evaluate(center + step).amplitudes - prepared.evaluate(center - step).amplitudes
    ) / (2.0 * step)
    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-8, atol=2e-9)


def test_xray_state_resolution_is_exact_and_range_limited() -> None:
    assert phasesmith.ScatteringSpecies("Fe", charge=3).xray_key == "Fe3+"
    assert phasesmith.ScatteringSpecies("C", isotope=13).xray_key == "C"
    assert phasesmith.ScatteringSpecies("C", xray_table_key="Cval").xray_key == "Cval"
    assert phasesmith.xray_species_metadata("Fe3+") == phasesmith.XraySpeciesMetadata("Fe3+", 26)
    assert phasesmith.xray_species_metadata("missing") is None
    with pytest.raises(ValueError, match="unknown X-ray"):
        phasesmith.XrayNonResonant().prepare([phasesmith.ScatteringSpecies("Fe", charge=9)])
    prepared = phasesmith.XrayNonResonant().prepare([phasesmith.ScatteringSpecies("Fe")])
    assert prepared.evaluate([6.0]).amplitudes.shape == (1, 1)
    with pytest.raises(ValueError, match="must not exceed 6"):
        prepared.evaluate([6.000001])


def test_neutron_natural_isotope_and_energy_dependent_behavior_is_explicit() -> None:
    species = (
        phasesmith.ScatteringSpecies("H"),
        phasesmith.ScatteringSpecies("H", isotope=1),
        phasesmith.ScatteringSpecies("H", isotope=2),
        phasesmith.ScatteringSpecies("C"),
        phasesmith.ScatteringSpecies("O"),
        phasesmith.ScatteringSpecies("Si"),
        phasesmith.ScatteringSpecies("Fe", charge=3),
    )
    prepared = phasesmith.NeutronNuclear().prepare(species)
    s = np.array([0.0, 0.8, 3.0])
    actual = prepared.evaluate(s)
    expected, derivative = scattering_reference.neutron_nuclear(
        [value.neutron_key for value in species], s.size
    )
    np.testing.assert_array_equal(actual.amplitudes.real, expected)
    np.testing.assert_array_equal(actual.d_amplitudes_d_s.real, derivative)
    assert phasesmith.neutron_species_metadata("H-2") == phasesmith.NeutronSpeciesMetadata(
        "H-2", 1, 2, 6.6681, 0.0027, False, None
    )
    fluorine = phasesmith.neutron_species_metadata("F")
    assert fluorine is not None
    assert fluorine.derived_alias_of == "F-19"
    with pytest.raises(ValueError, match="energy-dependent model"):
        phasesmith.NeutronNuclear().prepare([phasesmith.ScatteringSpecies("Cd")])
    with pytest.raises(ValueError, match="unknown neutron"):
        phasesmith.NeutronNuclear().prepare([phasesmith.ScatteringSpecies("H", isotope=999)])


def test_prepared_models_cache_unique_species_and_return_immutable_arrays() -> None:
    species = [
        phasesmith.ScatteringSpecies("Si"),
        phasesmith.ScatteringSpecies("O"),
        phasesmith.ScatteringSpecies("O"),
    ]
    prepared = phasesmith.XrayNonResonant().prepare(species)
    assert prepared.unique_species_count == 2
    assert prepared.species == tuple(species)
    with pytest.raises(AttributeError):
        prepared.species = ()
    result = prepared.evaluate(np.empty(0))
    assert result.amplitudes.shape == (0, 3)
    assert not result.amplitudes.flags.writeable
    assert not result.d_amplitudes_d_s.flags.writeable


def test_structure_species_mapping_preserves_probe_relevant_identity() -> None:
    structure = phasesmith.CrystalStructure(
        "species",
        "Species",
        phasesmith.UnitCell(4, 4, 4, 90, 90, 90),
        phasesmith.SpaceGroup.p1(),
        (
            phasesmith.AtomSite("d", "D1", "D", "H", (0, 0, 0), isotope=2),
            phasesmith.AtomSite("fe", "Fe1", "Fe3+", "Fe", (0.1, 0.2, 0.3), charge=3),
            phasesmith.AtomSite("cv", "C1", "Cval", "C", (0.2, 0.3, 0.4)),
        ),
    )
    deuterium, iron, carbon = phasesmith.species_from_structure(structure)
    assert (deuterium.xray_key, deuterium.neutron_key) == ("H", "H-2")
    assert (iron.xray_key, iron.neutron_key) == ("Fe3+", "Fe")
    assert carbon.xray_key == "Cval"


def test_versioned_vectorized_provider_contract_validates_one_complete_call() -> None:
    descriptor = phasesmith.ScatteringProviderDescriptor(
        "example.scattering", "1.0", "xray", "electrons"
    )

    class Provider:
        calls = 0

        @property
        def descriptor(self):
            return descriptor

        def evaluate(self, context):
            self.calls += 1
            shape = (context.reflection_count, len(context.species))
            return phasesmith.ScatteringFactorBatch(
                np.full(shape, 2.0 + 0.5j), np.full(shape, -0.25j), descriptor
            )

    context = phasesmith.ScatteringContext(
        [phasesmith.ScatteringSpecies("Si"), phasesmith.ScatteringSpecies("O")],
        [0.1, 0.2, 0.3],
    )
    provider = Provider()
    result = phasesmith.evaluate_scattering_provider(provider, context)
    assert provider.calls == 1
    assert result.amplitudes.shape == (3, 2)

    incompatible = phasesmith.ScatteringProviderDescriptor(
        "example.future", "1", "xray", "electrons", api_version=99
    )

    class FutureProvider(Provider):
        @property
        def descriptor(self):
            return incompatible

    with pytest.raises(ValueError, match="incompatible"):
        phasesmith.evaluate_scattering_provider(FutureProvider(), context)


def test_scattering_boundaries_validate_types_shapes_and_metadata() -> None:
    with pytest.raises(ValueError, match="canonical"):
        phasesmith.ScatteringSpecies("si")
    with pytest.raises(ValueError, match="positive integer"):
        phasesmith.ScatteringSpecies("Si", isotope=0)
    with pytest.raises(ValueError, match="non-zero"):
        phasesmith.ScatteringSpecies("Fe", charge=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        phasesmith.ScatteringContext([], [[0.1]])
    with pytest.raises(ValueError, match="finite and non-negative"):
        phasesmith.ScatteringContext([], [np.nan])
    with pytest.raises(ValueError, match="same 2-D shape"):
        phasesmith.ScatteringFactorBatch(
            np.zeros((2, 1)),
            np.zeros(2),
            phasesmith.XRAY_NON_RESONANT_DESCRIPTOR,
        )
    with pytest.raises(ValueError, match="amplitude_unit"):
        phasesmith.ScatteringProviderDescriptor("bad.unit", "1", "xray", "fm")

    assert phasesmith.XRAY_TABLE_PROVENANCE.row_count == 211
    assert phasesmith.NEUTRON_TABLE_PROVENANCE.row_count == 367
    assert phasesmith.XRAY_TABLE_PROVENANCE.upstream_commit.startswith("663d2171")
    assert phasesmith.NEUTRON_TABLE_PROVENANCE.upstream_commit.startswith("182ef63a")
