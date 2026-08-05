"""Readable independent reference for symmetry and bounded reflection sets."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystallography import UnitCell
from .crystallography_reference import reference_cell_geometry
from .symmetry import SpaceGroup


def _reciprocal_index(
    rotation: NDArray[np.int64], hkl: tuple[int, int, int]
) -> tuple[int, int, int]:
    transformed = rotation.T @ np.asarray(hkl, dtype=np.int64)
    return tuple(int(value) for value in transformed)


def _canonical_friedel(hkl: tuple[int, int, int]) -> tuple[int, int, int]:
    for value in hkl:
        if value > 0:
            return hkl
        if value < 0:
            return tuple(-index for index in hkl)
    return hkl


def reference_family(
    group: SpaceGroup,
    hkl: tuple[int, int, int],
    *,
    merge_friedel: bool,
) -> tuple[tuple[int, int, int], int]:
    """Build one reciprocal orbit directly from exact Python operations."""

    orbit: set[tuple[int, int, int]] = set()
    for operation in group.operations:
        member = _reciprocal_index(operation.rotation, hkl)
        orbit.add(member)
        if merge_friedel:
            orbit.add(tuple(-index for index in member))
    representatives = [_canonical_friedel(member) if merge_friedel else member for member in orbit]
    return min(representatives), len(orbit)


def reference_is_systematically_absent(group: SpaceGroup, hkl: tuple[int, int, int]) -> bool:
    """Evaluate grouped translation phases independently using complex roots."""

    if hkl == (0, 0, 0):
        return False
    groups: dict[tuple[int, int, int], list[Fraction]] = defaultdict(list)
    for operation in group.operations:
        transformed = _reciprocal_index(operation.rotation, hkl)
        phase = sum(
            (
                Fraction(index) * translation
                for index, translation in zip(hkl, operation.translation, strict=True)
            ),
            start=Fraction(0),
        )
        groups[transformed].append(phase % 1)
    return all(
        abs(sum(np.exp(2j * np.pi * float(phase)) for phase in phases)) < 1e-12
        for phases in groups.values()
    )


def reference_expand_sites(
    group: SpaceGroup,
    fractional_xyz: ArrayLike,
    *,
    tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Expand and periodically deduplicate sites with direct Python loops."""

    positions: list[NDArray[np.float64]] = []
    sources: list[int] = []
    for source, xyz in enumerate(np.asarray(fractional_xyz, dtype=np.float64)):
        site_positions: list[NDArray[np.float64]] = []
        for operation in group.operations:
            candidate = operation.apply_fractional(xyz)
            if not any(
                np.all(
                    np.minimum(np.abs(candidate - existing), 1.0 - np.abs(candidate - existing))
                    <= tolerance
                )
                for existing in site_positions
            ):
                site_positions.append(candidate)
        site_positions.sort(key=lambda value: tuple(float(item) for item in value))
        positions.extend(site_positions)
        sources.extend([source] * len(site_positions))
    return np.asarray(positions).reshape(-1, 3), np.asarray(sources, dtype=np.int64)


def reference_generate_d_spacing(
    cell: UnitCell,
    group: SpaceGroup,
    min_d_angstrom: float,
    max_d_angstrom: float,
    *,
    merge_friedel: bool,
) -> list[tuple[tuple[int, int, int], int, float]]:
    """Brute-force a d range using independent ellipsoid projection bounds."""

    geometry = reference_cell_geometry(cell)
    max_reciprocal = 1.0 / min_d_angstrom
    direct_lengths = np.sqrt(np.diag(geometry.direct_metric))
    bounds = np.ceil(max_reciprocal * direct_lengths).astype(np.int64)
    records: list[tuple[tuple[int, int, int], int, float]] = []
    for h in range(-int(bounds[0]), int(bounds[0]) + 1):
        for k in range(-int(bounds[1]), int(bounds[1]) + 1):
            for ell in range(-int(bounds[2]), int(bounds[2]) + 1):
                hkl = (h, k, ell)
                if hkl == (0, 0, 0):
                    continue
                vector = np.asarray(hkl, dtype=np.float64)
                reciprocal_squared = float(vector @ geometry.reciprocal_metric @ vector)
                d_spacing = reciprocal_squared**-0.5
                if not min_d_angstrom <= d_spacing <= max_d_angstrom:
                    continue
                canonical, multiplicity = reference_family(group, hkl, merge_friedel=merge_friedel)
                if canonical != hkl or reference_is_systematically_absent(group, hkl):
                    continue
                records.append((hkl, multiplicity, d_spacing))
    records.sort(key=lambda record: (1.0 / record[2], record[0]))
    return records
