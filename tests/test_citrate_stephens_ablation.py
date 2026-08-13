from __future__ import annotations

import phasesmith
import pytest
from phasesmith.oracle import orthorhombic_stephens_from_gsasii
from phasesmith.validation.citrate_stephens_ablation import (
    _SOURCE_PARAMETERS,
    _shape_constraints,
    run_citrate_stephens_ablation,
)


def test_source_parameters_translate_to_positive_orthorhombic_variances() -> None:
    for values in _SOURCE_PARAMETERS.values():
        provider = orthorhombic_stephens_from_gsasii(
            values["coefficients"], values["lorentzian_fraction"]
        )
        assert all(value > 0.0 for value in provider.coefficients_angstrom_minus4)
        assert provider.lorentzian_fraction == values["lorentzian_fraction"]


def test_source_shape_constraints_leave_only_one_coefficient_free() -> None:
    provider = orthorhombic_stephens_from_gsasii(
        _SOURCE_PARAMETERS["tripotassium"]["coefficients"], 0.9
    )
    constraints = _shape_constraints("phase", provider)
    assert len(constraints) == 6
    targets = {constraint.target.name for constraint in constraints}
    assert "stephens.S400" not in targets
    assert "stephens.lorentzian_fraction" in targets


def test_citrate_stephens_ablation_rejects_unknown_case(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"tripotassium.*trirubidium"):
        run_citrate_stephens_ablation(tmp_path, "unknown")  # type: ignore[arg-type]


def test_public_provider_remains_outside_the_gsasii_runtime_boundary() -> None:
    assert phasesmith.StephensOrthorhombicBroadening.__module__ == "phasesmith.sample"
