"""Versioned, expanded configurations for element-type-only structure search.

The named profile preserves the reviewed 2026-10-31 search combination. It is
an explicit choice; low-level solver defaults are unchanged. Atom/memory/time
limits remain caller choices and supplied symmetry remains a prior.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, replace

from .solution import SolutionOptions, solve_structure_elements

__all__ = ["ElementSearchConfig", "search_structure_elements"]

# Reviewed frozen combination, excluding corpus paths, external programs and
# scheduler controls. Tests compare this expansion to the retained protocol.
_VALIDATED = dict.fromkeys(
    [
        "wilson_normalization",
        "phase_quality_selection",
        "phase_origin_selection",
        "chemical_assignment",
        "iterative_refinement",
        "isotropic_assignment_refinement",
        "filtered_peak_search",
        "batched_fft",
        "real_fft",
        "fused_phase_workspace",
        "cached_normal_refinement",
        "multiplicity_weighted_refinement",
        "ambiguous_assignments",
        "cached_assignment_refinement",
        "specialized_isotropic_refinement",
        "streamed_peak_search",
        "reuse_normal_matrices",
        "short_phase_starts",
        "blocked_normal_matrices",
        "phase_plateau_stopping",
        "assignment_convergence_stopping",
        "late_phase_plateau_stopping",
        "model_consensus_stopping",
        "joint_assignment_refinement",
        "cached_repair_evaluation",
        "work_sliced_phasing",
        "phase_first_selection",
        "score_only_evaluation",
        "reusable_fourier_workspaces",
        "symmetry_recycling",
        "tangent_refinement",
    ],
    True,
)
_VALIDATED.update(
    missing_reflections=False,
    resolution_extension=1.0,
    adaptive_checkpoints=False,
    independent_origin_sampling=False,
    workers=1,
)
_INPUTS = {
    "observations",
    "cell",
    "elements",
    "space_group",
    "options",
    "maximum_atoms",
    "cancellation",
}


def _expanded(controls):
    result = {
        name: p.default
        for name, p in inspect.signature(solve_structure_elements).parameters.items()
        if name not in _INPUTS
    }
    result.update(controls)
    return tuple(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class ElementSearchConfig:
    """Immutable options plus a fully expanded, versioned algorithm selection."""

    name: str
    options: SolutionOptions
    controls: tuple[tuple[str, bool | int | float], ...]

    def __post_init__(self):
        if self.name != "validated-v1" or not isinstance(self.options, SolutionOptions):
            raise ValueError("expected validated-v1 and SolutionOptions")
        if not isinstance(self.controls, tuple) or self.controls != _expanded(_VALIDATED):
            raise ValueError("validated-v1 controls must match their complete frozen expansion")

    @classmethod
    def validated_v1(cls, *, options: SolutionOptions | None = None):
        """Select the reviewed combination without changing global solver defaults.

        Options explicitly control memory, phasing, cycles and runtime. The
        default budget is general, not fitted to the MgSiO3 example's atom count.
        """
        if options is None:
            options = SolutionOptions(phasing="dual_space", completion=False)
        return cls("validated-v1", options, _expanded(_VALIDATED))

    def to_record(self):
        """JSON-safe complete expansion; no corpus paths or source answers."""
        return dict(
            schema="phasesmith.element-search-config.v1",
            name=self.name,
            options=asdict(self.options),
            controls=dict(self.controls),
        )


def search_structure_elements(
    observations,
    *,
    cell,
    elements,
    configuration: ElementSearchConfig,
    space_group=None,
    maximum_atoms=None,
    cancellation=None,
):
    """Run a named configuration and retain its expansion in the result record.

    No additional algorithm overrides are accepted; use the explicit low-level
    solve_structure_elements API for other configurations. The maximum atom
    count is an input-only resource bound and never a reference-derived answer.
    """
    if not isinstance(configuration, ElementSearchConfig):
        raise TypeError("configuration must be ElementSearchConfig")
    result = solve_structure_elements(
        observations,
        cell=cell,
        elements=elements,
        space_group=space_group,
        options=configuration.options,
        maximum_atoms=maximum_atoms,
        cancellation=cancellation,
        **dict(configuration.controls),
    )
    return replace(
        result, _configuration_json=json.dumps(configuration.to_record(), sort_keys=True)
    )
