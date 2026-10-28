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

    atomicity_mode: str | None = None
    adaptive_peaks: AdaptivePeakHistory | None = None

    def __post_init__(self):
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

    def __post_init__(self):
        object.__setattr__(self, "elements", tuple(self.elements))
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
    cancellation: SolutionCancellation | None = None,
) -> SolutionResult:
    """Infer full-cell counts and labels by shared-map symmetry-orbit edits.

    Every supplied neutral non-H element is required for a target hit. The exact
    space group remains supplied. Charge-flipping and count-free dual-space or
    portfolio phasing use the same constrained refinement. Legacy completion and
    count-assignment switches are not used. Adaptive peak caps bound resources;
    their selected mask peaks are distinct from inferred atom counts.
    Counts returned in ``composition`` are inferred, never inputs. The default
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
