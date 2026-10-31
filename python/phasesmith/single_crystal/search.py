"""Versioned, expanded configurations for element-type-only structure search.

The named profile preserves the reviewed 2026-10-31 search combination. It is
an explicit choice; low-level solver defaults are unchanged. Atom/memory/time
limits remain caller choices and supplied symmetry remains a prior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

from .solution import SolutionOptions, solve_structure_elements

__all__ = ["ElementSearchConfig", "search_structure_elements"]

# Fully expanded v1 values: changing low-level defaults cannot change this profile.
_VALIDATED_CONTROLS = (
    ("adaptive_checkpoints", False),
    ("adaptive_counts", True),
    ("ambiguous_assignments", True),
    ("assignment_convergence_stopping", True),
    ("batched_fft", True),
    ("beam_width", 4),
    ("blocked_normal_matrices", True),
    ("cached_assignment_refinement", True),
    ("cached_normal_refinement", True),
    ("cached_refinement", True),
    ("cached_repair_evaluation", True),
    ("calculated_shell_normalization", True),
    ("chemical_assignment", True),
    ("dual_checkpoints", True),
    ("dual_peak_cap", 128),
    ("dual_peak_sigma", 2.0),
    ("filtered_peak_search", True),
    ("fused_phase_workspace", True),
    ("independent_origin_sampling", False),
    ("integrated_peak_density", True),
    ("isotropic_assignment_refinement", True),
    ("iterative_refinement", True),
    ("joint_assignment_refinement", True),
    ("late_phase_plateau_stopping", True),
    ("missing_reflections", False),
    ("model_consensus_stopping", True),
    ("multiplicity_weighted_refinement", True),
    ("peak_density_radius_angstrom", 0.7),
    ("phase_first_selection", True),
    ("phase_origin_selection", True),
    ("phase_plateau_stopping", True),
    ("phase_quality_selection", True),
    ("real_fft", True),
    ("recycling_cycles", 10),
    ("refinement_shortlist", 3),
    ("resolution_extension", 1.0),
    ("reusable_fourier_workspaces", True),
    ("reuse_normal_matrices", True),
    ("score_only_evaluation", True),
    ("screening_iterations", 8),
    ("search_rounds", 8),
    ("short_phase_starts", True),
    ("specialized_isotropic_refinement", True),
    ("staged_refinement", True),
    ("streamed_peak_search", True),
    ("symmetry_recycling", True),
    ("tangent_in_recycling", True),
    ("tangent_refinement", True),
    ("wilson_normalization", True),
    ("work_sliced_phasing", True),
    ("workers", 1),
)


@dataclass(frozen=True, slots=True)
class ElementSearchConfig:
    """Immutable options plus a fully expanded, versioned algorithm selection."""

    name: str
    options: SolutionOptions
    controls: tuple[tuple[str, bool | int | float], ...]

    def __post_init__(self):
        if self.name != "validated-v1" or not isinstance(self.options, SolutionOptions):
            raise ValueError("expected validated-v1 and SolutionOptions")
        if (
            not isinstance(self.controls, tuple)
            or self.controls != _VALIDATED_CONTROLS
            or any(
                type(value) is not type(expected)
                for (_, value), (_, expected) in zip(
                    self.controls, _VALIDATED_CONTROLS, strict=True
                )
            )
        ):
            raise ValueError("validated-v1 controls must match their complete frozen expansion")

    @classmethod
    def validated_v1(cls, *, options: SolutionOptions | None = None):
        """Select the reviewed combination without changing global solver defaults.

        Options explicitly control memory, phasing, cycles and runtime. The
        default budget is general, not fitted to the MgSiO3 example's atom count.
        """
        if options is None:
            options = SolutionOptions(phasing="dual_space", completion=False)
        return cls("validated-v1", options, _VALIDATED_CONTROLS)

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
