"""Ordered batches of monochromatic X-ray powder-stick patterns."""

from __future__ import annotations

import multiprocessing
import os
import threading
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .crystallography import calculate_structure_factor_values
from .execution import ExecutionPolicy
from .intensity_corrections import BraggBrentanoUnpolarizedLp
from .scattering import XrayNonResonant
from .structure import CrystalStructure, structure_from_record, structure_to_record
from .symmetry import CwTwoThetaRange, PreparedReflectionGenerator, SpaceGroup


@dataclass(frozen=True, slots=True, eq=False)
class PowderSticks:
    """Sorted 2θ positions (degrees) and unscaled, integrated X-ray intensities."""

    structure_id: str
    two_theta_deg: NDArray[np.float64]
    integrated_intensity: NDArray[np.float64]


def _merge_sticks(
    positions: NDArray[np.float64], intensities: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Merge within 1e-5 degrees of the first reflection in each group."""
    if positions.size == 0:
        return positions, intensities
    tolerance = 1e-5
    order = np.argsort(positions, kind="stable")
    x = positions[order]
    starts = np.r_[0, np.flatnonzero(np.diff(x) >= tolerance) + 1]
    ends = np.r_[starts[1:], len(x)]
    # Adjacent spacings can all be below tolerance while their span exceeds it.
    split = []
    needs_split = x[ends - 1] - x[starts] >= tolerance
    for start, end in zip(starts[needs_split], ends[needs_split], strict=True):
        anchor = start
        for i in range(start + 1, end):
            if x[i] - x[anchor] >= tolerance:
                split.append(i)
                anchor = i
    if split:
        starts = np.sort(np.r_[starts, split])
    return x[starts], np.add.reduceat(intensities[order], starts)


def _freeze_pair(
    positions: NDArray[np.float64], intensity: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    positions.flags.writeable = False
    intensity.flags.writeable = False
    return positions, intensity


class _GeneratorCache:
    """Thread-safe per-batch cache of prepared reflection generators."""

    __slots__ = ("_generators", "_lock")

    def __init__(self) -> None:
        self._generators: dict[SpaceGroup, PreparedReflectionGenerator] = {}
        self._lock = threading.Lock()

    def get(self, space_group: SpaceGroup) -> PreparedReflectionGenerator:
        with self._lock:
            generator = self._generators.get(space_group)
        if generator is not None:
            return generator
        created = PreparedReflectionGenerator(space_group)
        with self._lock:
            return self._generators.setdefault(space_group, created)


def _calculate(
    structure: CrystalStructure,
    scan: CwTwoThetaRange,
    generators: _GeneratorCache | None = None,
) -> PowderSticks:
    try:
        generator = (
            PreparedReflectionGenerator(structure.space_group)
            if generators is None
            else generators.get(structure.space_group)
        )
        reflections = generator.generate(structure.cell, scan)
        d = reflections.d_spacing_angstrom
        if not d.size:
            positions, intensity = _freeze_pair(d.copy(), np.empty(0, dtype=np.float64))
            return PowderSticks(structure.structure_id, positions, intensity)
        values = calculate_structure_factor_values(
            structure,
            reflections.hkl,
            reflections.multiplicity,
            XrayNonResonant(),
            correction=BraggBrentanoUnpolarizedLp(scan.wavelength_angstrom),
            execution=ExecutionPolicy(threads=1),
        )
        angles = 2 * np.rad2deg(np.arcsin(np.minimum(1.0, scan.wavelength_angstrom / (2 * d))))
        positions, intensity = _freeze_pair(*_merge_sticks(angles, values.integrated_intensity))
        return PowderSticks(structure.structure_id, positions, intensity)
    except (ValueError, TypeError, RuntimeError) as error:
        error.add_note(f"While calculating powder sticks for {structure.structure_id!r}")
        raise


def _compact_record(structure: CrystalStructure) -> dict[str, Any]:
    return structure_to_record(replace(structure, source=None, diagnostics=(), metadata={}))


def _calculate_record_chunk(
    records: tuple[dict[str, Any], ...], scan: CwTwoThetaRange
) -> tuple[PowderSticks, ...]:
    generators = _GeneratorCache()
    return tuple(_calculate(structure_from_record(rec), scan, generators) for rec in records)


def calculate_powder_sticks_batch(
    structures: Iterable[CrystalStructure],
    scan: CwTwoThetaRange,
    *,
    workers: int | None = None,
    backend: Literal["threads", "processes"] = "threads",
) -> tuple[PowderSticks, ...]:
    """Calculate one non-resonant X-ray Bragg-Brentano stick pattern per input.

    Uses the supplied sites, occupancies, displacement parameters and exact
    space group. Output order follows input order, including duplicates. Each
    independent calculation uses one native thread. A single worker runs inline.
    The default uses up to four threads, bounded by input count and CPU availability.
    Process workers use spawn and transfer parser-independent structure records.
    """
    if not isinstance(scan, CwTwoThetaRange):
        raise TypeError("scan must be a CwTwoThetaRange")
    if not (0 <= scan.min_deg <= scan.max_deg < 180):
        raise ValueError("scan must satisfy 0 <= min_deg <= max_deg < 180")
    if not np.isfinite(scan.wavelength_angstrom) or scan.wavelength_angstrom <= 0:
        raise ValueError("scan wavelength must be positive and finite")
    if workers is not None and (
        not isinstance(workers, int) or isinstance(workers, bool) or workers < 1
    ):
        raise ValueError("workers must be a positive integer or None")
    if backend not in ("threads", "processes"):
        raise ValueError("backend must be 'threads' or 'processes'")
    models = tuple(structures)
    for index, model in enumerate(models):
        if not isinstance(model, CrystalStructure):
            raise TypeError(f"structures[{index}] must be a CrystalStructure")
        if not model.sites:
            raise ValueError(f"structures[{index}] ({model.structure_id!r}) has no atom sites")
    if not models:
        return ()
    if workers is None:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        workers = min(4, cpu_count)
    workers = min(workers, len(models))
    generators = _GeneratorCache()
    calculate = partial(_calculate, scan=scan, generators=generators)
    if workers == 1:
        return tuple(map(calculate, models))
    if backend == "threads":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return tuple(pool.map(calculate, models))
    chunk_size = max(1, len(models) // (workers * 4))
    records = tuple(map(_compact_record, models))
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results: list[PowderSticks] = []
        for chunk_result in pool.map(partial(_calculate_record_chunk, scan=scan), chunks):
            for item in chunk_result:
                item.two_theta_deg.flags.writeable = False
                item.integrated_intensity.flags.writeable = False
            results.extend(chunk_result)
        return tuple(results)
