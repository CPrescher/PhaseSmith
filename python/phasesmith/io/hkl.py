"""Strict, bounded SHELX HKLF 4 intensity input, preserving signed observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True, init=False)
class HklData:
    """Immutable corrected intensities; no amplitude conversion or merging."""

    hkl: NDArray[np.int64]
    intensity: NDArray[np.float64]
    sigma: NDArray[np.float64]
    batch: NDArray[np.int64]
    source_name: str | None
    source_sha256: str | None

    def __init__(
        self,
        hkl: ArrayLike,
        intensity: ArrayLike,
        sigma: ArrayLike,
        *,
        batch: ArrayLike | None = None,
        source_name: str | None = None,
        source_sha256: str | None = None,
    ) -> None:
        indices = np.asarray(hkl)
        if (
            indices.ndim != 2
            or indices.shape[1] != 3
            or not np.issubdtype(indices.dtype, np.integer)
            or not 1 <= len(indices) <= 100_000
            or np.any(indices < -1024)
            or np.any(indices > 1024)
            or np.any(np.all(indices == 0, axis=1))
        ):
            raise ValueError(
                "hkl must have integer shape (N,3), 1<=N<=100000, nonzero rows, |h|<=1024"
            )
        indices = np.array(indices, dtype=np.int64, order="C", copy=True)
        values = np.array(intensity, dtype=np.float64, order="C", copy=True)
        errors = np.array(sigma, dtype=np.float64, order="C", copy=True)
        if values.shape != (len(indices),) or errors.shape != values.shape:
            raise ValueError("intensity and sigma must be vectors matching hkl")
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1e100):
            raise ValueError("intensity must be finite with absolute values <=1e100")
        if not np.isfinite(errors).all() or np.any(errors <= 0):
            raise ValueError("sigma must be strictly positive and finite")
        batches = np.ones(len(indices), dtype=np.int64) if batch is None else np.asarray(batch)
        if (
            batches.shape != values.shape
            or not np.issubdtype(batches.dtype, np.integer)
            or np.any(batches < 1)
            or np.any(batches > np.iinfo(np.int32).max)
        ):
            raise ValueError(
                "batch must contain positive int32 labels; HKLF 5 overlap flags unsupported"
            )
        batches = np.array(batches, dtype=np.int64, order="C", copy=True)
        for name, value in (
            ("hkl", indices),
            ("intensity", values),
            ("sigma", errors),
            ("batch", batches),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "source_sha256", source_sha256)


def parse_hkl(text: str, *, source_name: str | None = None) -> HklData:
    """Read whitespace or fixed `3I4,2F8.2[,I4]` HKLF 4 records.

    An optional 000 terminator ends the data; subsequent non-comment records
    are rejected. Blank lines and lines starting with # or ! are ignored.
    No intensity threshold, sorting, rejection, merging, or scaling is applied.
    """
    if not isinstance(text, str) or len(text.encode("utf-8")) > 32 * 1024 * 1024:
        raise ValueError("HKL text must not exceed 32 MiB")
    indices, intensity, sigma, batches = [], [], [], []
    terminated = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", "!")):
            continue
        if terminated:
            raise ValueError(f"HKL line {line_number}: data after 000 terminator")
        tokens = line.split()
        try:
            if len(tokens) not in (5, 6):
                raise ValueError
            h = [int(v) for v in tokens[:3]]
            value, error = (float(v.replace("D", "E").replace("d", "e")) for v in tokens[3:5])
            batch = 1 if len(tokens) == 5 else int(tokens[5])
        except ValueError:
            try:
                if len(line) < 28 or (len(line) > 32 and line[32:].strip()):
                    raise ValueError
                h = [int(line[i : i + 4]) for i in (0, 4, 8)]
                value, error = float(line[12:20]), float(line[20:28])
                batch = int(line[28:32]) if line[28:32].strip() else 1
            except ValueError as exc:
                raise ValueError(
                    f"HKL line {line_number}: expected h k l I sigma(I) [batch]"
                ) from exc
        if h == [0, 0, 0]:
            terminated = True
            continue
        indices.append(h)
        intensity.append(value)
        sigma.append(error)
        batches.append(batch)
        if len(indices) > 100_000:
            raise ValueError("HKL input exceeds 100000 observations")
    return HklData(
        indices,
        intensity,
        sigma,
        batch=batches,
        source_name=source_name,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def read_hkl(path: str | Path) -> HklData:
    """Read corrected intensities with exact source-byte SHA-256 provenance."""
    source = Path(path)
    if source.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("HKL file exceeds 32 MiB")
    raw = source.read_bytes()
    data = parse_hkl(raw.decode("ascii"), source_name=str(source.resolve()))
    # ASCII decoding preserves exact bytes; newline translation is not used.
    assert data.source_sha256 == hashlib.sha256(raw).hexdigest()
    return data
