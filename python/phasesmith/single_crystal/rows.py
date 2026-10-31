"""Row search (DAFi-based): exact independent Rust grouping, without indexing.

Based on Aslandukov et al. (2022), DOI 10.1107/S1600576722008081.
No original DAFi implementation source or executable is used.
"""

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from phasesmith import _core

from .reciprocal import triples

__all__ = ["RowSearchOptions", "RowSearchResult", "group_rows"]

ROW_SEARCH_NAME = "Row search (DAFi-based)"


@dataclass(frozen=True)
class RowSearchOptions:
    """The frozen sparse-grouping settings; cache budget is total binary MiB.

    None disables a work-count limit; seconds=0 returns an explicitly partial
    result. Cache zero disables retention; cache_unlimited overrides its budget.
    Optimized kernel switches retain the frozen Rust defaults.
    """

    direction_epsilon: float = 0.0033
    projection_epsilon: float = 0.002
    relative_epsilon: float = 0.02
    gap_epsilon: float = 0.02
    relative_gap: bool = True
    projection_grids: int = 4
    min_row: int = 4
    gap_min_row: int = 0
    phase_min_row: int = 3
    kmax: int = 5
    directions: int = 50000
    groups: int = 40
    batch: int = 6
    workers: int = 8
    smart: bool = True
    naive_max_points: int = 2048
    cache_megabytes: int = 512
    cache_unlimited: bool = False
    max_centers: int | None = None
    max_evaluations: int | None = None
    seconds: float = 300.0

    def __post_init__(self):
        size_max = int(np.iinfo(np.uintp).max)
        for name, value in asdict(self).items():
            if name in ("relative_gap", "smart", "cache_unlimited"):
                if type(value) is not bool:
                    raise ValueError(f"{name} must be boolean")
            elif name in (
                "direction_epsilon",
                "projection_epsilon",
                "relative_epsilon",
                "gap_epsilon",
                "seconds",
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} must be finite real")
            else:
                if value is None and name in ("max_centers", "max_evaluations"):
                    continue
                if type(value) is not int or not 0 <= value <= size_max:
                    raise ValueError(f"{name} must be a nonnegative platform-sized integer")
        if not (
            0.0005 <= self.direction_epsilon <= 0.1
            and 1e-8 <= self.projection_epsilon <= 0.1
            and 0 < self.relative_epsilon < 0.5
            and self.gap_epsilon > 0
            and self.seconds >= 0
        ):
            raise ValueError("invalid row-search tolerances or deadline")
        if (
            self.projection_grids not in (0, 2, 4)
            or self.min_row < 2
            or self.batch < 1
            or self.gap_min_row == 1
            or self.phase_min_row == 1
            or not 1 <= self.kmax <= 100
            or not 1 <= self.workers <= 8
        ):
            raise ValueError("invalid row-search sizes or worker count")
        if not self.cache_unlimited and self.cache_megabytes > size_max // (1024 * 1024):
            raise ValueError("cache_megabytes overflows platform byte count")

    def _native_options(self):
        result = asdict(self)
        for name in ("max_centers", "max_evaluations"):
            if result[name] is None:
                result[name] = int(np.iinfo(np.uintp).max)
        return result


@dataclass(frozen=True)
class RowSearchResult:
    """Grouping, not an indexed lattice or established physical grain count.

    Labels use one-based group IDs; zero is unassigned. Groups retain the exact
    ordered original-row indices, with coincident extra observations unassigned.
    diagnostics contains resolved options, budgets, work counters and timings.
    """

    labels: np.ndarray
    groups: tuple[np.ndarray, ...]
    mean: np.ndarray
    scale: np.ndarray
    diagnostics: dict
    options: RowSearchOptions
    method: str = "rows"

    @property
    def complete(self):
        """False for work/deadline exhaustion or cancellation, even with groups."""
        return not (self.diagnostics["budget_exhausted"] or self.diagnostics["cancelled"])


def group_rows(points, *, options=None):
    """Group unindexed reciprocal coordinates using the selected published method.

    All supplied coordinates participate. For held-out validation, call this on
    training points only, or use find_grains(..., method="rows"). Finite real
    shape (n,3), n<=100000, |q|<=1000; caller coordinate order is preserved.
    """
    options = RowSearchOptions() if options is None else options
    if not isinstance(options, RowSearchOptions):
        raise TypeError("options must be RowSearchOptions")
    q = triples(points, "points")
    if len(q) > 100000 or np.max(abs(q), initial=0) > 1000:
        raise ValueError("bounded search requires <=100000 positions and |q|<=1000")
    if not hasattr(_core, "experimental_row_search"):
        raise RuntimeError("Row search requires a PhaseSmith spots build")
    result = json.loads(
        _core.experimental_row_search(q, json.dumps(options._native_options(), allow_nan=False))
    )
    labels = np.asarray(result.pop("labels"), dtype=np.int64)
    groups = tuple(np.asarray(ids, dtype=np.int64) for ids in result.pop("groups"))
    mean, scale = (np.asarray(result.pop(name), dtype=np.float64) for name in ("mean", "scale"))
    for array in (labels, mean, scale, *groups):
        array.flags.writeable = False
    return RowSearchResult(labels, groups, mean, scale, result, options)
