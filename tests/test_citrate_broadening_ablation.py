from pathlib import Path

import pytest
from phasesmith.validation import CitrateBroadeningModelResult
from phasesmith.validation.citrate_broadening_ablation import (
    run_citrate_isotropic_broadening_ablation,
)


def test_citrate_broadening_result_rejects_invalid_correlation() -> None:
    with pytest.raises(ValueError, match="correlation"):
        CitrateBroadeningModelResult(
            model="size",
            free_parameter_count=4,
            jacobian_rank=4,
            poisson_rwp=0.2,
            delta_poisson_rwp=-0.03,
            profile_correlation=0.8,
            weight_fractions={"main": 0.98, "silicon": 0.02},
            sample_parameters={"crystallite_size_nm": 100.0},
            repeat_rwp_spread=0.0,
            maximum_sample_linear_correlation=1.01,
            identifiable=False,
            qualifications=("invalid test record",),
        )


def test_citrate_broadening_ablation_rejects_unknown_case(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"tripotassium.*trirubidium"):
        run_citrate_isotropic_broadening_ablation(tmp_path, "unknown")  # type: ignore[arg-type]
