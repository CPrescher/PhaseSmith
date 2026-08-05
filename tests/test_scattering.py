from __future__ import annotations

import numpy as np
import pytest
import rietveld
from rietveld import scattering_reference


def test_xray_native_batch_matches_independent_reference_and_finite_differences() -> None:
    species = (
        rietveld.ScatteringSpecies("H"),
        rietveld.ScatteringSpecies("C"),
        rietveld.ScatteringSpecies("O"),
        rietveld.ScatteringSpecies("Si"),
        rietveld.ScatteringSpecies("Fe"),
        rietveld.ScatteringSpecies("Fe", charge=3),
    )
    s = np.linspace(0.0, 5.9, 43)
    prepared = rietveld.XrayNonResonant().prepare(species)
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
    assert rietveld.ScatteringSpecies("Fe", charge=3).xray_key == "Fe3+"
    assert rietveld.ScatteringSpecies("C", isotope=13).xray_key == "C"
    assert rietveld.ScatteringSpecies("C", xray_table_key="Cval").xray_key == "Cval"
    assert rietveld.xray_species_metadata("Fe3+") == rietveld.XraySpeciesMetadata("Fe3+", 26)
    assert rietveld.xray_species_metadata("missing") is None
    with pytest.raises(ValueError, match="unknown X-ray"):
        rietveld.XrayNonResonant().prepare([rietveld.ScatteringSpecies("Fe", charge=9)])
    prepared = rietveld.XrayNonResonant().prepare([rietveld.ScatteringSpecies("Fe")])
    assert prepared.evaluate([6.0]).amplitudes.shape == (1, 1)
    with pytest.raises(ValueError, match="must not exceed 6"):
        prepared.evaluate([6.000001])


def test_neutron_natural_isotope_and_energy_dependent_behavior_is_explicit() -> None:
    species = (
        rietveld.ScatteringSpecies("H"),
        rietveld.ScatteringSpecies("H", isotope=1),
        rietveld.ScatteringSpecies("H", isotope=2),
        rietveld.ScatteringSpecies("C"),
        rietveld.ScatteringSpecies("O"),
        rietveld.ScatteringSpecies("Si"),
        rietveld.ScatteringSpecies("Fe", charge=3),
    )
    prepared = rietveld.NeutronNuclear().prepare(species)
    s = np.array([0.0, 0.8, 3.0])
    actual = prepared.evaluate(s)
    expected, derivative = scattering_reference.neutron_nuclear(
        [value.neutron_key for value in species], s.size
    )
    np.testing.assert_array_equal(actual.amplitudes.real, expected)
    np.testing.assert_array_equal(actual.d_amplitudes_d_s.real, derivative)
    assert rietveld.neutron_species_metadata("H-2") == rietveld.NeutronSpeciesMetadata(
        "H-2", 1, 2, 6.6681, 0.0027, False, None
    )
    fluorine = rietveld.neutron_species_metadata("F")
    assert fluorine is not None
    assert fluorine.derived_alias_of == "F-19"
    with pytest.raises(ValueError, match="energy-dependent model"):
        rietveld.NeutronNuclear().prepare([rietveld.ScatteringSpecies("Cd")])
    with pytest.raises(ValueError, match="unknown neutron"):
        rietveld.NeutronNuclear().prepare([rietveld.ScatteringSpecies("H", isotope=999)])


def test_prepared_models_cache_unique_species_and_return_immutable_arrays() -> None:
    species = [
        rietveld.ScatteringSpecies("Si"),
        rietveld.ScatteringSpecies("O"),
        rietveld.ScatteringSpecies("O"),
    ]
    prepared = rietveld.XrayNonResonant().prepare(species)
    assert prepared.unique_species_count == 2
    assert prepared.species == tuple(species)
    with pytest.raises(AttributeError):
        prepared.species = ()
    result = prepared.evaluate(np.empty(0))
    assert result.amplitudes.shape == (0, 3)
    assert not result.amplitudes.flags.writeable
    assert not result.d_amplitudes_d_s.flags.writeable


def test_structure_species_mapping_preserves_probe_relevant_identity() -> None:
    structure = rietveld.CrystalStructure(
        "species",
        "Species",
        rietveld.UnitCell(4, 4, 4, 90, 90, 90),
        rietveld.SpaceGroup.p1(),
        (
            rietveld.AtomSite("d", "D1", "D", "H", (0, 0, 0), isotope=2),
            rietveld.AtomSite("fe", "Fe1", "Fe3+", "Fe", (0.1, 0.2, 0.3), charge=3),
            rietveld.AtomSite("cv", "C1", "Cval", "C", (0.2, 0.3, 0.4)),
        ),
    )
    deuterium, iron, carbon = rietveld.species_from_structure(structure)
    assert (deuterium.xray_key, deuterium.neutron_key) == ("H", "H-2")
    assert (iron.xray_key, iron.neutron_key) == ("Fe3+", "Fe")
    assert carbon.xray_key == "Cval"


def test_versioned_vectorized_provider_contract_validates_one_complete_call() -> None:
    descriptor = rietveld.ScatteringProviderDescriptor(
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
            return rietveld.ScatteringFactorBatch(
                np.full(shape, 2.0 + 0.5j), np.full(shape, -0.25j), descriptor
            )

    context = rietveld.ScatteringContext(
        [rietveld.ScatteringSpecies("Si"), rietveld.ScatteringSpecies("O")],
        [0.1, 0.2, 0.3],
    )
    provider = Provider()
    result = rietveld.evaluate_scattering_provider(provider, context)
    assert provider.calls == 1
    assert result.amplitudes.shape == (3, 2)

    incompatible = rietveld.ScatteringProviderDescriptor(
        "example.future", "1", "xray", "electrons", api_version=99
    )

    class FutureProvider(Provider):
        @property
        def descriptor(self):
            return incompatible

    with pytest.raises(ValueError, match="incompatible"):
        rietveld.evaluate_scattering_provider(FutureProvider(), context)


def test_scattering_boundaries_validate_types_shapes_and_metadata() -> None:
    with pytest.raises(ValueError, match="canonical"):
        rietveld.ScatteringSpecies("si")
    with pytest.raises(ValueError, match="positive integer"):
        rietveld.ScatteringSpecies("Si", isotope=0)
    with pytest.raises(ValueError, match="non-zero"):
        rietveld.ScatteringSpecies("Fe", charge=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        rietveld.ScatteringContext([], [[0.1]])
    with pytest.raises(ValueError, match="finite and non-negative"):
        rietveld.ScatteringContext([], [np.nan])
    with pytest.raises(ValueError, match="same 2-D shape"):
        rietveld.ScatteringFactorBatch(
            np.zeros((2, 1)),
            np.zeros(2),
            rietveld.XRAY_NON_RESONANT_DESCRIPTOR,
        )
    with pytest.raises(ValueError, match="amplitude_unit"):
        rietveld.ScatteringProviderDescriptor("bad.unit", "1", "xray", "fm")

    assert rietveld.XRAY_TABLE_PROVENANCE.row_count == 211
    assert rietveld.NEUTRON_TABLE_PROVENANCE.row_count == 367
    assert rietveld.XRAY_TABLE_PROVENANCE.upstream_commit.startswith("663d2171")
    assert rietveld.NEUTRON_TABLE_PROVENANCE.upstream_commit.startswith("182ef63a")
