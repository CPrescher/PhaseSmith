"""Experimental native structure solution from externally corrected HKLF 4 data.

Build with ``maturin develop --release --features experimental-solution``.
Run ``python -m phasesmith.experimental.solution --help`` for file-based use.
The supplied symmetry constrains optional site/ADP refinement after P1 phasing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..crystallography import UnitCell
from ..io.hkl import HklData, read_hkl
from ..io.space_groups import space_group_by_symbol
from ..structure import AtomSite, CrystalStructure
from ..symmetry import SpaceGroup
from .solution_symmetry import SymmetryRefinement, candidate_cif, symmetry_records


def _freeze(array: object, dtype: object = np.float64) -> np.ndarray:
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


class SolutionCancellation:
    """Thread-safe cooperative stop token; cancellation retains completed candidates."""

    __slots__ = ("_native",)

    def __init__(self) -> None:
        if not hasattr(_core, "_SolutionCancellation"):
            raise ImportError("cancellation requires the experimental-solution native feature")
        self._native = _core._SolutionCancellation()

    def cancel(self) -> bool:
        """Request a stop; return true only for the first request."""
        return self._native.cancel()

    @property
    def is_cancelled(self) -> bool:
        """Whether a stop was requested."""
        return self._native.is_cancelled()


@dataclass(frozen=True, slots=True)
class SolutionOptions:
    """Explicit reproducible phase starts, iteration limits, and numerical target."""

    seed: int = 0
    trials: int = 8
    cycles: int = 800
    phasing: str = "charge_flip"
    delta_sigma: float = 1.0
    weak_fraction: float = 0.2
    cleanup_cycles: int = 10
    polish_iterations: int = 50
    assignment_rounds: int = 4
    max_seconds: float = 60.0
    r1_target: float = 0.1
    grid_shape: tuple[int, int, int] | None = None
    refinement: str = "anisotropic"
    symmetry_tolerance_angstrom: float = 0.15
    symmetry_iterations: int = 80
    completion: bool = True

    def __post_init__(self) -> None:
        for name, low, high in (
            ("seed", 0, 2**64 - 1),
            ("trials", 1, 128),
            ("cycles", 1, 20_000),
            ("cleanup_cycles", 0, 100),
            ("polish_iterations", 0, 200),
            ("assignment_rounds", 0, 8),
            ("symmetry_iterations", 0, 200),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low},{high}]")
        for name, low, high in (
            ("delta_sigma", 0.1, 2.5),
            ("weak_fraction", 0.0, 0.5),
            ("max_seconds", 0.0, 3600.0),
            ("r1_target", 0.0, 1.0),
            ("symmetry_tolerance_angstrom", 1e-5, 0.5),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite in [{low},{high}]")
        if not isinstance(self.completion, bool):
            raise TypeError("completion must be boolean")
        if self.phasing not in ("charge_flip", "dual_space", "portfolio"):
            raise ValueError("phasing must be charge_flip, dual_space, or portfolio")
        if self.refinement not in ("none", "isotropic", "anisotropic"):
            raise ValueError("refinement must be none, isotropic, or anisotropic")
        if self.max_seconds == 0:
            raise ValueError("max_seconds must be strictly positive")
        if self.grid_shape is not None:
            shape = tuple(self.grid_shape)
            if len(shape) != 3 or any(
                not isinstance(n, int) or isinstance(n, bool) or not 8 <= n <= 128 or n % 2
                for n in shape
            ):
                raise ValueError("grid_shape must contain three even integers in [8,128]")
            object.__setattr__(self, "grid_shape", shape)


@dataclass(frozen=True, slots=True)
class CompletionAttempt:
    """Actual partial-framework work; a retained candidate can precede a stop."""

    applied: bool
    status: str
    diagnostic: str | None
    framework_species: tuple[str, ...]
    framework_atoms: int
    added_species: str | None
    added_atoms: int
    seed_count: int
    projection_trials: int
    orbit_trials: int
    refinements: int
    framework_hypotheses: int
    selected_framework: int | None
    framework_maximum_error: float | None
    initial_objective: float
    best_objective: float | None

    def __post_init__(self):
        object.__setattr__(self, "framework_species", tuple(self.framework_species))


@dataclass(frozen=True, slots=True)
class SolutionCandidate:
    """Full-cell atomic hypothesis; a good amplitude fit does not prove correctness."""

    species: tuple[str, ...]
    fractional_xyz: NDArray[np.float64]
    u_iso: float
    amplitude_scale: float
    r1: float
    objective: float
    trial: int
    polish_steps: int
    element_swaps: int
    refinement: SymmetryRefinement | None = None
    refinement_diagnostic: str | None = None
    completion: CompletionAttempt | None = None

    def as_structure(self, cell: UnitCell) -> CrystalStructure:
        """Create the refined asymmetric model, or the retained P1 hypothesis."""
        if self.refinement is not None:
            return self.refinement.as_structure(cell)
        sites = tuple(
            AtomSite(
                site_id=f"{element}{index + 1}",
                source_label=f"{element}{index + 1}",
                type_symbol=element,
                element_symbol=element,
                fractional_xyz=tuple(float(v) for v in point),
                u_iso_angstrom2=self.u_iso,
            )
            for index, (element, point) in enumerate(
                zip(self.species, self.fractional_xyz, strict=True)
            )
        )
        return CrystalStructure(
            structure_id="experimental-solution",
            name="Experimental full-cell P1 candidate",
            cell=cell,
            space_group=SpaceGroup.p1(),
            sites=sites,
            metadata={"status": "unvalidated structure-solution candidate"},
        )

    def to_record(self) -> dict[str, object]:
        """Finite JSON-compatible numerical record."""
        return {
            "species": list(self.species),
            "fractional_xyz": self.fractional_xyz.tolist(),
            "u_iso": self.u_iso,
            "amplitude_scale": self.amplitude_scale,
            "r1": self.r1,
            "objective": self.objective,
            "trial": self.trial,
            "polish_steps": self.polish_steps,
            "element_swaps": self.element_swaps,
            "refinement": None if self.refinement is None else self.refinement.to_record(),
            "refinement_diagnostic": self.refinement_diagnostic,
            "completion": None if self.completion is None else asdict(self.completion),
        }


@dataclass(frozen=True, slots=True)
class AdaptivePeakHistory:
    """Data-selected atomicity peaks, distinct from inferred structure counts."""

    sigma: float
    cap: int
    selected: tuple[int, ...]
    thresholds: tuple[float, ...]
    omitted: tuple[int, ...]

    def __post_init__(self):
        for name in ("selected", "thresholds", "omitted"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class DualSpaceCheckpoint:
    """One complete-model handoff, accepted only when the fit target is met."""

    completed_cycles: int
    best_cycle: int
    score: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class PhasingTrial:
    """Immutable record of an actual Fourier trial and its initialization."""

    method: str
    initialization: str
    patterson_vector_grid: tuple[int, int, int] | None
    seed: int
    completed_cycles: int
    best_cycle: int
    score_kind: str
    mask_sigma_angstrom: float | None
    hybrid_m: float | None = None
    sharpening_q: float | None = None
    omit_fraction: float | None = None
    omit_period: int | None = None

    checkpoints: tuple[DualSpaceCheckpoint, ...] = ()
    atomicity_mode: str | None = None
    adaptive_peaks: AdaptivePeakHistory | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "checkpoints",
            tuple(DualSpaceCheckpoint(**c) if isinstance(c, dict) else c for c in self.checkpoints),
        )
        if isinstance(self.adaptive_peaks, dict):
            object.__setattr__(self, "adaptive_peaks", AdaptivePeakHistory(**self.adaptive_peaks))
        if self.patterson_vector_grid is not None:
            object.__setattr__(self, "patterson_vector_grid", tuple(self.patterson_vector_grid))


@dataclass(frozen=True, slots=True)
class SharedMove:
    """One accepted orbit edit on the selected shared-search lineage."""

    kind: str
    element: str
    atoms: int
    r1: float
    score: float


@dataclass(frozen=True, slots=True)
class DensityCalibration:
    """Relative strong-atom mass scale for one parent in one setting."""

    trial: int
    lineage: int
    round: int
    scale: float | None
    anchors: int
    shortlist_limit: int


@dataclass(frozen=True, slots=True)
class RecyclingAudit:
    """One supplied-origin recycling attempt and whole-orbit batch proposal."""

    trial: int
    lineage: int
    hand: int
    origin_shift: tuple[float, float, float]
    cycles: int
    predicted_coefficients: int
    peaks: int
    batch_orbits: int
    label_scores: int
    label_changes: int
    batch_refined: bool
    tangent: TangentAudit | None = None

    def __post_init__(self):
        object.__setattr__(self, "origin_shift", tuple(self.origin_shift))
        if isinstance(self.tangent, dict):
            object.__setattr__(self, "tangent", TangentAudit(**self.tangent))


@dataclass(frozen=True, slots=True)
class TangentAudit:
    """Actual measured-only invariant preparation and reciprocal phase work."""

    trial: int
    variables: int
    strong: int
    triplets: int
    quartets: int
    truncated: bool
    steps: int
    annealed_steps: int
    proposals: int
    changed: int
    declined_steps: int
    quartet_before: tuple[float, ...]
    quartet_after: tuple[float, ...]
    symmetry_restorations: int = 0
    symmetry_declined_steps: int = 0

    def __post_init__(self):
        object.__setattr__(self, "quartet_before", tuple(self.quartet_before))
        object.__setattr__(self, "quartet_after", tuple(self.quartet_after))


@dataclass(frozen=True, slots=True)
class WilsonShell:
    s_squared: float
    families: int
    reduced_mean: float
    log_residual: float


@dataclass(frozen=True, slots=True)
class WilsonNormalizationAudit:
    families: int
    nonpositive_families: int
    log_scale: float
    slope: float
    mean_e_squared: float
    shells: tuple[WilsonShell, ...]

    def __post_init__(self):
        object.__setattr__(
            self,
            "shells",
            tuple(WilsonShell(**s) if isinstance(s, dict) else s for s in self.shells),
        )


@dataclass(frozen=True, slots=True)
class PhaseQualityAudit:
    trial: int
    allowed_families: int
    absent_families: int
    weak_families: int
    correlation: tuple[float, ...]
    r_weak: tuple[float, ...]
    cfom: tuple[float, ...]
    deferred_handoffs: int
    alternative_handoffs: int = 0
    calculated_shells: int = 0

    def __post_init__(self):
        for name in ("correlation", "r_weak", "cfom"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PhaseSettingAudit:
    """One phase-aligned supplied-symmetry setting and its completion effort."""

    hand: int
    shift: tuple[float, float, float]
    alpha: float
    chord: float
    orbits: int
    scales: tuple[float, ...]
    candidates: int
    evaluations: int
    changes: int
    screens: int
    promotions: int
    diagnostic: str | None
    chemical_scales: tuple[tuple[str, float, int], ...] = ()
    reassignment_evaluations: int = 0
    reassignment_changes: int = 0
    reassignment_screens: int = 0

    def __post_init__(self):
        object.__setattr__(self, "chemical_scales", tuple(tuple(s) for s in self.chemical_scales))
        object.__setattr__(self, "shift", tuple(self.shift))
        object.__setattr__(self, "scales", tuple(self.scales))


@dataclass(frozen=True, slots=True)
class PhaseSliceAudit:
    """One work/time-limited phase start, separate from downstream acceptance."""

    trial: int
    cycle_quota: int
    completed_cycles: int
    seconds: float
    stop_reason: str


@dataclass(frozen=True, slots=True)
class PhaseCheckpointAudit:
    """A phase-map selection decision; selected does not certify a structure."""

    trial: int
    cycles: int
    cfom: float
    score: float
    map_cosine: float | None
    selected: bool
    reason: str
    best_cfom: float | None = None


@dataclass(frozen=True, slots=True)
class PhaseOriginAudit:
    """Phase-only origin search and measured downstream work for one P1 map."""

    trial: int
    screening_families: int
    phase_families: int
    origin_candidates: int
    settings: tuple[PhaseSettingAudit, ...]
    origin_seconds: float
    recycling_seconds: float
    completion_seconds: float
    screening_seconds: float
    refinement_seconds: float
    diagnostic: str | None
    origin_sampling_shape: tuple[int, int, int] = (0, 0, 0)
    origin_harmonic_extent: tuple[int, int, int] = (0, 0, 0)
    assignment_parent_fits: int = 0
    assignment_neighbor_scores: int = 0
    assignment_neighbor_fits: int = 0
    assignment_retained: int = 0
    assignment_deferred_promotions: int = 0
    assignment_families: int = 0
    assignment_scores: tuple[float, ...] = ()
    ambiguity_scores: int = 0
    ambiguity_relaxations: int = 0
    ambiguity_rounds: int = 0
    ambiguity_retained: int = 0
    ambiguity_unresolved: bool = False
    ambiguity_metric_disagreements: int = 0

    def __post_init__(self):
        object.__setattr__(self, "assignment_scores", tuple(self.assignment_scores))
        object.__setattr__(self, "origin_sampling_shape", tuple(self.origin_sampling_shape))
        object.__setattr__(self, "origin_harmonic_extent", tuple(self.origin_harmonic_extent))
        object.__setattr__(
            self,
            "settings",
            tuple(PhaseSettingAudit(**a) if isinstance(a, dict) else a for a in self.settings),
        )


@dataclass(frozen=True, slots=True)
class ReflectionPredictionAudit:
    """Separate measured and predicted reciprocal support provenance."""

    shape: tuple[int, int, int]
    measured_rows: int
    missing_pairs: int
    extended_pairs: int
    observed_d_min: float
    requested_extension: float
    effective_extension: float
    effective_d_min: float

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True, slots=True)
class SharedSearchAudit:
    """Actual work and inferred topology, without source composition."""

    elements: tuple[str, ...]
    phasing_starts: int
    framework_hypotheses: int
    initial_prefix: int
    rounds: int
    orbit_proposals: int
    element_scores: int
    refinements: int
    screening_refinements: int
    final_refinements: int
    promoted_candidates: int
    screened_out: int
    integrated_peak_density: bool
    peak_density_radius_angstrom: float
    density_map_builds: int
    density_evaluations: int
    density_label_rejections: int
    density_label_rescues: int
    density_birth_skips: int
    density_relabel_skips: int
    density_calibrations: tuple[DensityCalibration, ...]
    staged_refinement: bool
    screening_iterations: int
    refinement_shortlist: int
    deletion_trials: int
    relabel_trials: int
    maximum_atoms: int
    beam_width: int
    required_elements_present: bool
    accepted_moves: tuple[SharedMove, ...]
    dual_checkpoints: bool = False
    cached_refinement: bool = False
    symmetry_recycling: bool = False
    recycling_cycles: int = 10
    recycling: tuple[RecyclingAudit, ...] = ()
    tangent_refinement: bool = False
    tangent_trials: tuple[TangentAudit, ...] = ()
    tangent_in_recycling: bool = True
    adaptive_counts: bool = False
    count_prefix_screenings: int = 0
    bounded_map_handoffs: int = 0
    wilson_normalization: bool = False
    phase_quality_selection: bool = False
    phase_first_selection: bool = False
    calculated_shell_normalization: bool = False
    phase_origin_selection: bool = False
    chemical_assignment: bool = False
    iterative_refinement: bool = False
    cached_normal_refinement: bool = False
    multiplicity_weighted_refinement: bool = False
    ambiguous_assignments: bool = False
    reuse_normal_matrices: bool = False
    normal_matrix_builds: int = 0
    normal_matrix_reuses: int = 0
    streamed_peak_search: bool = False
    specialized_isotropic_refinement: bool = False
    cached_assignment_refinement: bool = False
    specialized_isotropic_fits: int = 0
    assignment_cached_refinements: int = 0
    assignment_cached_normal_solves: int = 0
    assignment_normal_cache_fallbacks: int = 0
    ambiguity_continuations: int = 0
    weighted_refinement_fits: int = 0
    weighted_refinement_input_rows: int = 0
    weighted_refinement_family_rows: int = 0
    cached_normal_solves: int = 0
    normal_cache_fallbacks: int = 0
    missing_reflections: bool = False
    score_only_evaluation: bool = False
    reusable_fourier_workspaces: bool = False
    work_sliced_phasing: bool = False
    isotropic_assignment_refinement: bool = False
    filtered_peak_search: bool = False
    batched_fft: bool = False
    real_fft: bool = False
    fused_phase_workspace: bool = False
    cached_mask_cycles: int = 0
    fused_fourier_transforms: int = 0
    real_phase_transforms: int = 0
    batched_phase_transforms: int = 0
    streamed_peak_cycles: int = 0
    peak_rms_fallbacks: int = 0
    filtered_peak_cycles: int = 0
    phase_slices: tuple[PhaseSliceAudit, ...] = ()
    score_only_evaluations: int = 0
    fourier_workspace_cycles: int = 0
    adaptive_checkpoints: bool = False
    independent_origin_sampling: bool = False
    resolution_extension: float = 1.0
    prediction: ReflectionPredictionAudit | None = None
    prediction_steps: int = 0
    prediction_updates: int = 0
    prediction_clipped: int = 0
    prediction_seconds: float = 0.0
    phase_origins: tuple[PhaseOriginAudit, ...] = ()
    phase_checkpoints: tuple[PhaseCheckpointAudit, ...] = ()
    normalization: WilsonNormalizationAudit | None = None
    normalization_diagnostic: str | None = None
    phase_quality: tuple[PhaseQualityAudit, ...] = ()
    phase_model_trials: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self,
            "phase_checkpoints",
            tuple(
                PhaseCheckpointAudit(**c) if isinstance(c, dict) else c
                for c in self.phase_checkpoints
            ),
        )
        object.__setattr__(
            self,
            "phase_slices",
            tuple(PhaseSliceAudit(**a) if isinstance(a, dict) else a for a in self.phase_slices),
        )
        if isinstance(self.prediction, dict):
            object.__setattr__(self, "prediction", ReflectionPredictionAudit(**self.prediction))
        object.__setattr__(
            self,
            "phase_origins",
            tuple(PhaseOriginAudit(**a) if isinstance(a, dict) else a for a in self.phase_origins),
        )
        object.__setattr__(self, "phase_model_trials", tuple(self.phase_model_trials))
        if isinstance(self.normalization, dict):
            object.__setattr__(
                self, "normalization", WilsonNormalizationAudit(**self.normalization)
            )
        object.__setattr__(
            self,
            "phase_quality",
            tuple(PhaseQualityAudit(**q) if isinstance(q, dict) else q for q in self.phase_quality),
        )
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(
            self,
            "tangent_trials",
            tuple(
                t if isinstance(t, TangentAudit) else TangentAudit(**t) for t in self.tangent_trials
            ),
        )
        object.__setattr__(
            self,
            "recycling",
            tuple(
                r if isinstance(r, RecyclingAudit) else RecyclingAudit(**r) for r in self.recycling
            ),
        )
        object.__setattr__(
            self,
            "density_calibrations",
            tuple(
                c if isinstance(c, DensityCalibration) else DensityCalibration(**c)
                for c in self.density_calibrations
            ),
        )
        object.__setattr__(
            self,
            "accepted_moves",
            tuple(m if isinstance(m, SharedMove) else SharedMove(**m) for m in self.accepted_moves),
        )


@dataclass(frozen=True, slots=True)
class SolutionResult:
    """Candidates, diagnostic map, signed inputs, provenance, and truthful stop reason."""

    observations: HklData
    cell: UnitCell
    input_space_group: SpaceGroup
    composition: tuple[tuple[str, int], ...]
    options: SolutionOptions
    candidates: tuple[SolutionCandidate, ...]
    density: NDArray[np.float64]
    expanded_hkl: NDArray[np.int64]
    merged_intensity: NDArray[np.float64]
    merged_sigma: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    family_count: int
    absent_observations: int
    nonpositive_families: int
    d_min: float
    histories: tuple[NDArray[np.float64], ...]
    phasing_trials: tuple[PhasingTrial, ...]
    rejected_trials: tuple[tuple[int, str], ...]
    termination: str
    target_met: bool
    shared_search: SharedSearchAudit | None = None

    @property
    def best(self) -> SolutionCandidate | None:
        """Preferred method-ranked candidate, possibly still incorrect."""
        return self.candidates[0] if self.candidates else None

    def to_record(self) -> dict[str, object]:
        """Summarize a run without embedding large input or density arrays."""
        record = {
            "schema": "phasesmith.experimental-solution.v4",
            "experimental": True,
            "source_name": self.observations.source_name,
            "source_sha256": self.observations.source_sha256,
            "cell": self.cell.as_tuple(),
            "composition_full_cell": dict(self.composition),
            "input_symmetry_operation_count": len(self.input_space_group.operations),
            "input_symmetry_operations": symmetry_records(self.input_space_group),
            "output_space_group": (
                "supplied operations"
                if self.best is not None and self.best.refinement is not None
                else "P 1"
            ),
            "options": asdict(self.options),
            "grid_shape": self.density.shape,
            "density_units": "arbitrary",
            "source_observation_count": len(self.observations.hkl),
            "source_negative_count": int(np.count_nonzero(self.observations.intensity < 0)),
            "family_count": self.family_count,
            "absent_observations": self.absent_observations,
            "nonpositive_families": self.nonpositive_families,
            "d_min_angstrom": self.d_min,
            "termination": self.termination,
            "r1_target_met": self.target_met,
            "candidates": [c.to_record() for c in self.candidates],
            "phasing_histories": [h.tolist() for h in self.histories],
            "phasing_trials": [asdict(t) for t in self.phasing_trials],
            "rejected_trials": list(self.rejected_trials),
            "limitations": [
                "Numerical target is not proof of a correct or unique structure.",
                "Friedel merged; non-resonant X-ray model, full occupancy, no hydrogens.",
                "Element assignments are provisional; "
                "density-frame origin/hand transforms are explicit.",
                "Negative means clipped only for derived amplitudes; "
                "unweighted amplitude objective.",
            ],
        }
        if self.shared_search is not None:
            record["schema"] = "phasesmith.experimental-solution.v5"
            record["composition_source"] = "inferred_shared_orbit_search"
            record["input_elements"] = list(self.shared_search.elements)
            record["refinement_baseline_kind"] = (
                "initial_projected_isotropic_sites"
                if self.shared_search.staged_refinement
                else "initial_projected_common_u"
            )
            record["shared_search"] = asdict(self.shared_search)
            record["limitations"] = [
                item for item in record["limitations"] if "composition" not in item.lower()
            ] + ["Bounded shared orbit search with supplied exact symmetry; no uniqueness proof."]
        return record


def solve_structure(
    observations: HklData,
    *,
    cell: UnitCell,
    composition: Mapping[str, int],
    space_group: SpaceGroup | None = None,
    options: SolutionOptions | None = None,
    cancellation: SolutionCancellation | None = None,
) -> SolutionResult:
    """Phase corrected HKL intensities and return experimental full-cell candidates.

    ``composition`` gives non-H atom counts in the **whole conventional cell**,
    not the asymmetric unit or one formula unit. No coordinates are accepted.
    Absent ``space_group`` means P1, with Friedel merging; symmetry is not inferred.
    """
    if not hasattr(_core, "_solve_structure"):
        raise ImportError(
            "structure solution requires the experimental-solution native feature; "
            "build with maturin develop --release --features experimental-solution"
        )
    if not isinstance(observations, HklData) or not isinstance(cell, UnitCell):
        raise TypeError("observations and cell must be HklData and UnitCell")
    if not isinstance(composition, Mapping) or not composition:
        raise ValueError("composition must map non-hydrogen species to positive full-cell counts")
    if any(not isinstance(name, str) for name in composition):
        raise ValueError("composition species must be strings")
    counts = tuple(sorted(composition.items()))
    if (
        any(
            not isinstance(name, str)
            or not name
            or not isinstance(count, int)
            or isinstance(count, bool)
            or not 1 <= count <= 128
            for name, count in counts
        )
        or sum(count for _, count in counts) > 128
    ):
        raise ValueError(
            "composition must contain string species and positive counts totaling <=128"
        )
    group = SpaceGroup.p1() if space_group is None else space_group
    controls = SolutionOptions() if options is None else options
    if not isinstance(group, SpaceGroup) or not isinstance(controls, SolutionOptions):
        raise TypeError("space_group and options must be SpaceGroup and SolutionOptions")
    if cancellation is not None and not isinstance(cancellation, SolutionCancellation):
        raise TypeError("cancellation must be SolutionCancellation or None")
    return _run_solution(observations, cell, group, counts, controls, cancellation, None)


def solve_structure_elements(
    observations: HklData,
    *,
    cell: UnitCell,
    elements: Sequence[str],
    space_group: SpaceGroup | None = None,
    options: SolutionOptions | None = None,
    maximum_atoms: int | None = None,
    beam_width: int = 4,
    search_rounds: int = 8,
    staged_refinement: bool = True,
    screening_iterations: int = 8,
    refinement_shortlist: int = 3,
    integrated_peak_density: bool = True,
    peak_density_radius_angstrom: float = 0.7,
    dual_peak_sigma: float = 2.0,
    dual_peak_cap: int = 128,
    dual_checkpoints: bool = True,
    cached_refinement: bool = True,
    symmetry_recycling: bool = False,
    recycling_cycles: int = 10,
    tangent_refinement: bool = False,
    tangent_in_recycling: bool = True,
    adaptive_counts: bool = True,
    wilson_normalization: bool = False,
    phase_quality_selection: bool = False,
    phase_first_selection: bool = True,
    calculated_shell_normalization: bool = True,
    phase_origin_selection: bool = False,
    chemical_assignment: bool = False,
    iterative_refinement: bool = False,
    cached_normal_refinement: bool = False,
    multiplicity_weighted_refinement: bool = False,
    ambiguous_assignments: bool = False,
    cached_assignment_refinement: bool = False,
    specialized_isotropic_refinement: bool = False,
    streamed_peak_search: bool = False,
    reuse_normal_matrices: bool = False,
    missing_reflections: bool = False,
    resolution_extension: float = 1.0,
    score_only_evaluation: bool = False,
    reusable_fourier_workspaces: bool = False,
    work_sliced_phasing: bool = False,
    isotropic_assignment_refinement: bool = False,
    filtered_peak_search: bool = False,
    batched_fft: bool = False,
    real_fft: bool = False,
    fused_phase_workspace: bool = False,
    adaptive_checkpoints: bool = False,
    independent_origin_sampling: bool = False,
    cancellation: SolutionCancellation | None = None,
) -> SolutionResult:
    """Infer full-cell counts and labels by shared-map symmetry-orbit edits.

    Every supplied neutral non-H element is required for a target hit. The exact
    space group remains supplied. Charge-flipping and count-free dual-space or
    portfolio phasing use the same constrained refinement. Legacy completion and
    count-assignment switches are not used. Adaptive peak caps bound resources;
    their selected mask peaks are distinct from inferred atom counts.
    ``wilson_normalization=True`` fits signed, independent-family resolution
    shells with epsilon corrections and uses the resulting E values for phasing.
    ``phase_quality_selection=True`` ranks P1 maps before model fitting, with
    independently normalized calculated amplitudes for the weak statistic.
    ``phase_first_selection=False`` tests a checkpoint portfolio instead; this
    control is ignored when quality selection is off.
    ``calculated_shell_normalization=False`` retains the earlier statistic.
    ``cached_normal_refinement=True`` caches preconditioned normal matrices in final
    iterative fits, verifies steps against the original Jacobian and falls back
    to the matrix-free solver if needed. It requires iterative refinement.
    ``reuse_normal_matrices=True`` retains a checked CG normal matrix while only
    damping changes; it requires iterative refinement and at least one normal cache.
    ``streamed_peak_search=True`` guards vectorized RMS thresholds with scalar
    fallback and streams peak candidates; it requires ``filtered_peak_search``.
    ``specialized_isotropic_refinement=True`` uses a fused canonical isotropic
    cached evaluator with unchanged equations and generic fallback.
    ``cached_assignment_refinement=True`` uses guarded normal caches in short
    isotropic fits; it requires iterative and isotropic assignment refinement.
    ``ambiguous_assignments=True`` retains coupled near-element relabels through
    bounded isotropic competition and requires isotropic assignment refinement.
    ``multiplicity_weighted_refinement=True`` evaluates final constrained fits on
    measured symmetry/Friedel families with their original multiplicities.
    ``fused_phase_workspace=True`` reuses exact Gaussian axis geometry and, with
    packed real FFTs, transfers density arrays directly at Fourier boundaries.
    It requires reusable workspaces and preserves arithmetic and support.
    ``real_fft=True`` uses packed real-density Fourier cycles in P1 phasing,
    retains full spectra and observations at boundaries, and requires batched FFTs.
    It changes floating-point operation order and remains experimental.
    ``batched_fft=True`` batches contiguous Fourier lines after blocked layout
    transposes, retaining the original plans, axis order and normalization.
    It requires reusable Fourier workspaces.
    ``filtered_peak_search=True`` preserves adaptive peaks exactly while filtering
    below-threshold voxels before neighbor comparisons; it requires reusable workspaces.
    ``isotropic_assignment_refinement=True`` relaxes competing complete labelings
    on common independent-family observations before selective final-mode promotion.
    It requires phase-origin selection and chemical assignment.
    ``work_sliced_phasing=True`` uses half-cycle quotas for the first half of
    independent starts, retaining full-cycle fallback starts and existing wall limits.
    Existing model validation still decides acceptance; no input atom counts are used.
    It requires phase-origin/quality/phase-first selection without adaptive_checkpoints.
    ``score_only_evaluation=True`` computes unchanged scalar phase merits using
    cached fixed statistics, without allocating unused gradients.
    ``reusable_fourier_workspaces=True`` reuses full-grid P1 map/mask/complex
    buffers and input arrays; double precision, support and cycle rules are unchanged.
    ``adaptive_checkpoints=True`` selects stable quality-ready maps at geometric
    phase checkpoints before the cycle ceiling; full structure validation still
    follows. It requires phase-origin, phase-quality and phase-first selection.
    ``independent_origin_sampling=True`` derives origin FFT seed spacing from
    measured phase-equivalence frequencies instead of the density grid size.
    ``phase_origin_selection=True`` aligns P1 phases to the supplied symmetry
    before assigning peaks. Cached whole-orbit count/label proposals and short
    isotropic screens then precede full refinement. This replaces the earlier
    framework/edit and adaptive-count schedule; it remains opt-in. Per-map
    origin hypotheses, diagnostics, work counters and stage timings appear in
    ``shared_search.phase_origins``. This does not infer a space group.
    ``chemical_assignment=True`` adds motif scale proposals and coupled element
    reassignment after short coordinate/displacement fits; it requires phase-origin
    selection. ``iterative_refinement=True`` avoids dense normal-matrix construction
    for fits with at least 32 parameters. The dense Jacobian and original resource
    caps are retained. Both controls are experimental and off by default.
    Original observations still
    govern final refinement. ``missing_reflections=True`` regularizes unmeasured
    allowed coefficients using
    the observed Wilson envelope. ``resolution_extension`` extends reciprocal radius
    (1 means no extension, maximum 1.5); actual bounds and prediction work are audited.
    Both require Wilson normalization, phase-origin selection and dual-space phasing.
    Predicted coefficients never enter observation arrays, scoring or peak-mass integrals.
    ``adaptive_counts=True`` compares nested residual-orbit additions, repeats
    count proposals after framework refinement, and bounds poor early-map
    handoffs. ``False`` retains the previous single-batch/edit schedule.
    ``dual_checkpoints`` tests early maps with full requested refinement and
    resumes the same Fourier trajectory when a structure misses the target.
    ``symmetry_recycling=True`` adds ten supplied-symmetry phase/positivity
    cycles per retained origin before weak-peak assignment; ``recycling_cycles``
    controls the bounded cycle budget. ``tangent_refinement=True`` adds measured-only
    tangent phases, cooling perturbations and negative-quartet guards; it requires
    ``phasing="dual_space"`` in SolutionOptions. With symmetry recycling, tangent
    steps run after origin recovery and are checked after restoring symmetry.
    ``tangent_in_recycling=False`` retains the earlier unconstrained placement.
    Recycling remains opt-in because easy cases
    can become slower. Counts returned in ``composition`` are inferred, never inputs. The default
    atom cap is ceil(cell volume/8 A³), at most 128; it is a search bound.
    """
    if not hasattr(_core, "_SharedOrbitPool"):
        raise ImportError("shared element search requires experimental-solution native feature")
    if not isinstance(observations, HklData) or not isinstance(cell, UnitCell):
        raise TypeError("observations and cell must be HklData and UnitCell")
    if isinstance(elements, str):
        raise ValueError("elements must be a sequence of element symbols")
    elements = tuple(elements)
    if (
        not 1 <= len(elements) <= 8
        or any(not isinstance(e, str) for e in elements)
        or len(set(elements)) != len(elements)
    ):
        raise ValueError("elements must be 1..8 distinct neutral non-H symbols")
    for name, value in (
        ("chemical_assignment", chemical_assignment),
        ("iterative_refinement", iterative_refinement),
        ("cached_normal_refinement", cached_normal_refinement),
        ("multiplicity_weighted_refinement", multiplicity_weighted_refinement),
        ("ambiguous_assignments", ambiguous_assignments),
        ("cached_assignment_refinement", cached_assignment_refinement),
        ("specialized_isotropic_refinement", specialized_isotropic_refinement),
        ("streamed_peak_search", streamed_peak_search),
        ("reuse_normal_matrices", reuse_normal_matrices),
        ("missing_reflections", missing_reflections),
        ("adaptive_checkpoints", adaptive_checkpoints),
        ("score_only_evaluation", score_only_evaluation),
        ("reusable_fourier_workspaces", reusable_fourier_workspaces),
        ("work_sliced_phasing", work_sliced_phasing),
        ("isotropic_assignment_refinement", isotropic_assignment_refinement),
        ("filtered_peak_search", filtered_peak_search),
        ("batched_fft", batched_fft),
        ("real_fft", real_fft),
        ("fused_phase_workspace", fused_phase_workspace),
        ("independent_origin_sampling", independent_origin_sampling),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if (
        isinstance(resolution_extension, bool)
        or not np.isfinite(resolution_extension)
        or not 1 <= resolution_extension <= 1.5
    ):
        raise ValueError("resolution_extension must be finite in [1,1.5]")
    if (missing_reflections or resolution_extension > 1) and not (
        phase_origin_selection and wilson_normalization
    ):
        raise ValueError("prediction requires phase_origin_selection and wilson_normalization")
    if cached_assignment_refinement and not (
        iterative_refinement and isotropic_assignment_refinement
    ):
        raise ValueError(
            "cached_assignment_refinement requires iterative and isotropic assignment refinement"
        )
    if ambiguous_assignments and not isotropic_assignment_refinement:
        raise ValueError("ambiguous_assignments requires isotropic_assignment_refinement")
    if cached_normal_refinement and not iterative_refinement:
        raise ValueError("cached_normal_refinement requires iterative_refinement")
    if fused_phase_workspace and not reusable_fourier_workspaces:
        raise ValueError("fused_phase_workspace requires reusable_fourier_workspaces")
    if reuse_normal_matrices and not (
        iterative_refinement and (cached_normal_refinement or cached_assignment_refinement)
    ):
        raise ValueError("reuse_normal_matrices requires iterative refinement and a normal cache")
    if streamed_peak_search and not filtered_peak_search:
        raise ValueError("streamed_peak_search requires filtered_peak_search")
    if real_fft and not batched_fft:
        raise ValueError("real_fft requires batched_fft")
    if batched_fft and not reusable_fourier_workspaces:
        raise ValueError("batched_fft requires reusable_fourier_workspaces")
    if filtered_peak_search and not reusable_fourier_workspaces:
        raise ValueError("filtered_peak_search requires reusable_fourier_workspaces")
    if isotropic_assignment_refinement and not (phase_origin_selection and chemical_assignment):
        raise ValueError(
            "isotropic_assignment_refinement requires phase_origin_selection "
            "and chemical_assignment"
        )
    if work_sliced_phasing and not (
        phase_origin_selection
        and phase_quality_selection
        and phase_first_selection
        and not adaptive_checkpoints
    ):
        raise ValueError(
            "work_sliced_phasing requires phase-origin/quality/phase-first selection "
            "without adaptive_checkpoints"
        )
    if score_only_evaluation and not phase_quality_selection:
        raise ValueError("score_only_evaluation requires phase_quality_selection")
    if independent_origin_sampling and not phase_origin_selection:
        raise ValueError("independent_origin_sampling requires phase_origin_selection")
    if adaptive_checkpoints and not (
        phase_origin_selection
        and phase_quality_selection
        and dual_checkpoints
        and phase_first_selection
    ):
        raise ValueError(
            "adaptive_checkpoints requires phase-origin, quality and phase-first checkpoints"
        )
    if chemical_assignment and not phase_origin_selection:
        raise ValueError("chemical_assignment requires phase_origin_selection")
    if not isinstance(phase_origin_selection, bool):
        raise TypeError("phase_origin_selection must be boolean")
    if not isinstance(wilson_normalization, bool):
        raise TypeError("wilson_normalization must be boolean")
    if not isinstance(calculated_shell_normalization, bool):
        raise TypeError("calculated_shell_normalization must be boolean")
    if not isinstance(phase_first_selection, bool):
        raise TypeError("phase_first_selection must be boolean")
    if not isinstance(phase_quality_selection, bool):
        raise TypeError("phase_quality_selection must be boolean")
    if not isinstance(adaptive_counts, bool):
        raise TypeError("adaptive_counts must be boolean")
    if not isinstance(tangent_refinement, bool):
        raise TypeError("tangent_refinement must be boolean")
    if not isinstance(tangent_in_recycling, bool):
        raise TypeError("tangent_in_recycling must be boolean")
    if not isinstance(symmetry_recycling, bool):
        raise TypeError("symmetry_recycling must be boolean")
    if not isinstance(cached_refinement, bool):
        raise TypeError("cached_refinement must be boolean")
    if not isinstance(dual_checkpoints, bool):
        raise TypeError("dual_checkpoints must be boolean")
    if not isinstance(staged_refinement, bool):
        raise TypeError("staged_refinement must be boolean")
    if not isinstance(integrated_peak_density, bool):
        raise TypeError("integrated_peak_density must be boolean")
    if (
        isinstance(peak_density_radius_angstrom, bool)
        or not np.isfinite(peak_density_radius_angstrom)
        or not 1e-4 <= peak_density_radius_angstrom <= 2
    ):
        raise ValueError("peak_density_radius_angstrom must be finite in [1e-4,2]")
    if (
        isinstance(dual_peak_sigma, bool)
        or not np.isfinite(dual_peak_sigma)
        or not 0.1 <= dual_peak_sigma <= 10
    ):
        raise ValueError("dual_peak_sigma must be finite in [0.1,10]")
    if maximum_atoms is None:
        maximum_atoms = min(128, int(np.ceil(cell.geometry().volume_angstrom3 / 8)))
    for name, value, high in (
        ("maximum_atoms", maximum_atoms, 128),
        ("beam_width", beam_width, 8),
        ("search_rounds", search_rounds, 16),
        ("screening_iterations", screening_iterations, 40),
        ("refinement_shortlist", refinement_shortlist, 16),
        ("dual_peak_cap", dual_peak_cap, 128),
        ("recycling_cycles", recycling_cycles, 50),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= high:
            raise ValueError(f"{name} must be an integer in [1,{high}]")
    group = SpaceGroup.p1() if space_group is None else space_group
    controls = (
        options
        if options is not None
        else SolutionOptions(trials=1, cycles=800, max_seconds=30, r1_target=0.06, completion=False)
    )
    if not isinstance(group, SpaceGroup) or not isinstance(controls, SolutionOptions):
        raise TypeError("space_group and options must be SpaceGroup and SolutionOptions")
    if reusable_fourier_workspaces and controls.phasing != "dual_space":
        raise ValueError("reusable_fourier_workspaces requires dual_space phasing")
    if (missing_reflections or resolution_extension > 1) and controls.phasing != "dual_space":
        raise ValueError("prediction requires phasing=dual_space")
    if phase_quality_selection and controls.phasing != "dual_space":
        raise ValueError("phase_quality_selection requires phasing=dual_space")
    if tangent_refinement and controls.phasing != "dual_space":
        raise ValueError("tangent_refinement requires phasing=dual_space")
    if cancellation is not None and not isinstance(cancellation, SolutionCancellation):
        raise TypeError("cancellation must be SolutionCancellation or None")
    shared = (
        elements,
        maximum_atoms,
        beam_width,
        search_rounds,
        staged_refinement,
        screening_iterations,
        refinement_shortlist,
        integrated_peak_density,
        peak_density_radius_angstrom,
        dual_peak_sigma,
        dual_peak_cap,
        (
            dual_checkpoints,
            cached_refinement,
            symmetry_recycling,
            recycling_cycles,
            tangent_refinement,
            tangent_in_recycling,
            adaptive_counts,
            wilson_normalization,
            phase_quality_selection,
            phase_first_selection,
            calculated_shell_normalization,
            (
                phase_origin_selection,
                chemical_assignment,
                iterative_refinement,
                missing_reflections,
                resolution_extension,
                adaptive_checkpoints,
                independent_origin_sampling,
                score_only_evaluation,
                reusable_fourier_workspaces,
                work_sliced_phasing,
                isotropic_assignment_refinement,
                (
                    filtered_peak_search,
                    batched_fft,
                    cached_normal_refinement,
                    real_fft,
                    fused_phase_workspace,
                    multiplicity_weighted_refinement,
                    ambiguous_assignments,
                    cached_assignment_refinement,
                    specialized_isotropic_refinement,
                    streamed_peak_search,
                    reuse_normal_matrices,
                ),
            ),
        ),
    )
    return _run_solution(observations, cell, group, (), controls, cancellation, shared)


def _run_solution(observations, cell, group, counts, controls, cancellation, shared):
    raw = _core._solve_structure(
        observations.hkl.reshape(-1),
        observations.intensity,
        observations.sigma,
        list(cell.as_tuple()),
        group._native,
        list(counts),
        **asdict(controls),
        cancellation=None if cancellation is None else cancellation._native,
        shared_search=shared,
    )
    candidates = tuple(
        SolutionCandidate(
            species=tuple(c["species"]),
            fractional_xyz=_freeze(c["fractional_xyz"]),
            refinement=None
            if c["refinement"] is None
            else SymmetryRefinement.from_native(c["refinement"], group),
            refinement_diagnostic=c["refinement_diagnostic"],
            completion=None if c["completion"] is None else CompletionAttempt(**c["completion"]),
            **{
                key: c[key]
                for key in (
                    "u_iso",
                    "amplitude_scale",
                    "r1",
                    "objective",
                    "trial",
                    "polish_steps",
                    "element_swaps",
                )
            },
        )
        for c in raw["candidates"]
    )
    audit = None if raw["shared_search"] is None else SharedSearchAudit(**raw["shared_search"])
    if audit is not None:
        counts = tuple(sorted(Counter(candidates[0].species).items())) if candidates else ()
    return SolutionResult(
        observations,
        cell,
        group,
        counts,
        controls,
        candidates,
        _freeze(np.asarray(raw["density"]).reshape(raw["grid_shape"])),
        _freeze(raw["hkl"], np.int64),
        _freeze(raw["intensity"]),
        _freeze(raw["sigma"]),
        _freeze(raw["amplitudes"]),
        raw["family_count"],
        raw["absent_observations"],
        raw["nonpositive_families"],
        raw["d_min"],
        tuple(_freeze(h) for h in raw["histories"]),
        tuple(PhasingTrial(**t) for t in raw["phasing_trials"]),
        tuple(tuple(r) for r in raw["rejected_trials"]),
        raw["termination"],
        raw["target_met"],
        audit,
    )


def write_solution(result: SolutionResult, directory: str | Path) -> Path:
    """Write a fresh output folder containing a report, map/input arrays, and P1 CIF."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "result.json").write_text(
        json.dumps(result.to_record(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        destination / "density-and-observations.npz",
        density=result.density,
        input_hkl=result.observations.hkl,
        input_intensity=result.observations.intensity,
        input_sigma=result.observations.sigma,
        input_batch=result.observations.batch,
        expanded_hkl=result.expanded_hkl,
        merged_intensity=result.merged_intensity,
        merged_sigma=result.merged_sigma,
        amplitudes=result.amplitudes,
    )
    if result.best is not None:
        best = result.best
        refined = best.refinement
        full_u = np.full(len(best.species), best.u_iso) if refined is None else refined.full_u_equiv
        full_cif = None if refined is None else refined.full_u_cif
        (destination / "candidate-p1.cif").write_text(
            candidate_cif(
                result.cell, best.species, best.fractional_xyz, full_u, SpaceGroup.p1(), full_cif
            ),
            encoding="ascii",
        )
        if refined is not None:
            text = candidate_cif(
                result.cell,
                refined.species,
                refined.fractional_xyz,
                refined.u_equiv,
                result.input_space_group,
                refined.u_cif,
            )
            (destination / "candidate.cif").write_text(text, encoding="ascii")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Solve a pre-integrated HKL file; exit 3 when the numerical target is unmet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hkl", type=Path)
    parser.add_argument(
        "--cell",
        type=float,
        nargs=6,
        required=True,
        metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
    )
    parser.add_argument(
        "--composition", nargs="+", required=True, help="full-cell non-H counts, e.g. Si=3 O=6"
    )
    parser.add_argument(
        "--space-group",
        required=True,
        help="known HM symbol, e.g. 'P 31 2 1'; P1 for a complete list",
    )
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=800)
    parser.add_argument(
        "--phasing", choices=("charge_flip", "dual_space", "portfolio"), default="charge_flip"
    )
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--r1-target", type=float, default=0.1)
    parser.add_argument(
        "--refinement", choices=("none", "isotropic", "anisotropic"), default="anisotropic"
    )
    parser.add_argument(
        "--no-completion", action="store_true", help="disable weakest-species orbit completion"
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"output directory already exists: {args.output}")
        composition = {}
        for token in args.composition:
            name, count = token.split("=", 1)
            if name in composition:
                raise ValueError(f"duplicate composition species: {name}")
            composition[name] = int(count)
        result = solve_structure(
            read_hkl(args.hkl),
            cell=UnitCell(*args.cell),
            composition=composition,
            space_group=space_group_by_symbol(args.space_group).space_group,
            options=SolutionOptions(
                seed=args.seed,
                trials=args.trials,
                cycles=args.cycles,
                phasing=args.phasing,
                completion=not args.no_completion,
                max_seconds=args.max_seconds,
                r1_target=args.r1_target,
                refinement=args.refinement,
            ),
        )
        write_solution(result, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "termination": result.termination,
                    "r1_target_met": result.target_met,
                    "experimental": True,
                    "best_r1": None if result.best is None else result.best.r1,
                },
                allow_nan=False,
            )
        )
        return 0 if result.target_met else 3
    except (ValueError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"structure solution: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
