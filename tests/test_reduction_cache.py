from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from phasesmith.single_crystal import reduction_cache
from phasesmith.single_crystal.reduction_cache import ReductionCache, cache_key


def test_reduction_cache_streams_sha256_in_chunks_and_uses_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force multiple streaming chunks even on a compact test payload.
    monkeypatch.setattr(reduction_cache, "_HASH_CHUNK_BYTES", 64)
    cache = ReductionCache(tmp_path / "cache")
    payload = {
        "values": np.arange(256, dtype=np.float64),
        "counts": np.arange(64, dtype=np.uint32),
    }
    key = cache_key("stage-caf\u00e9", {"mode": "test-utf8"}, payload, implementation="v1")

    cache.store(key, payload)
    loaded = cache.load(key)
    assert loaded is not None
    for name, expected in payload.items():
        np.testing.assert_array_equal(loaded[name], expected)

    entry = tmp_path / "cache" / key
    meta = json.loads((entry / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == hashlib.sha256((entry / "arrays.npz").read_bytes()).hexdigest()

    # Corrupting one byte must fail checksum verification.
    archive = entry / "arrays.npz"
    raw = bytearray(archive.read_bytes())
    raw[-1] ^= 0xFF
    archive.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum mismatch"):
        cache.load(key)
