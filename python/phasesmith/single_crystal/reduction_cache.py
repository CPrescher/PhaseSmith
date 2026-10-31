"""Content-addressed numerical artifacts with atomic immutable cache entries."""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

__all__ = ["ReductionCache", "cache_key"]


def cache_key(stage, configuration, arrays, *, implementation):
    """Hash exact typed content, configuration and all implementation identity."""
    if (
        not isinstance(stage, str)
        or not stage
        or not isinstance(implementation, str)
        or not implementation
    ):
        raise ValueError("stage and implementation identity are required")
    config = asdict(configuration) if is_dataclass(configuration) else configuration
    h = hashlib.sha256(
        json.dumps([stage, config, implementation], sort_keys=True, allow_nan=False).encode()
    )
    for name, value in sorted(arrays.items()):
        a = np.asarray(value)
        if not isinstance(name, str) or a.dtype.hasobject:
            raise ValueError("cache requires named nonobject arrays")
        a = np.ascontiguousarray(a)
        h.update(json.dumps([name, a.dtype.str, a.shape]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class ReductionCache:
    """Disposable cache; corrupt/partial entries raise instead of reusing data."""

    schema = "phasesmith.experimental.reduction-cache.v1"

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, key):
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
        ):
            raise ValueError("invalid numerical cache key")
        return self.root / key

    def load(self, key):
        entry = self._path(key)
        if not entry.exists():
            return None
        path, meta = entry / "arrays.npz", entry / "metadata.json"
        if not path.is_file() or not meta.is_file():
            raise ValueError("incomplete numerical cache entry")
        record = json.loads(meta.read_text())
        if record.get("schema") != self.schema or record.get("key") != key:
            raise ValueError("numerical cache schema/key mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError("numerical cache checksum mismatch")
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
        schema = {k: [v.dtype.str, list(v.shape)] for k, v in arrays.items()}
        if schema != record.get("arrays"):
            raise ValueError("numerical cache array schema mismatch")
        return arrays

    def _identical(self, key, arrays):
        old = self.load(key)
        if old is None:
            return False
        if old.keys() != arrays.keys() or any(
            old[k].dtype != v.dtype or old[k].shape != v.shape or old[k].tobytes() != v.tobytes()
            for k, v in arrays.items()
        ):
            raise ValueError("immutable numerical cache key has conflicting content")
        return True

    def store(self, key, arrays):
        entry = self._path(key)
        arrays = {name: np.asarray(v) for name, v in arrays.items()}
        if any(not isinstance(k, str) or not k or a.dtype.hasobject for k, a in arrays.items()):
            raise ValueError("cache requires named numerical arrays")
        if self._identical(key, arrays):
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=self.root, prefix=".entry-"))
        try:
            path = temporary / "arrays.npz"
            np.savez_compressed(path, **arrays)
            record = dict(
                schema=self.schema,
                key=key,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                arrays={k: [a.dtype.str, list(a.shape)] for k, a in arrays.items()},
            )
            (temporary / "metadata.json").write_text(
                json.dumps(record, sort_keys=True, allow_nan=False)
            )
            try:
                os.rename(temporary, entry)
            except OSError:
                if not self._identical(key, arrays):
                    raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
