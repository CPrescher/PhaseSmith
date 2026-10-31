from pathlib import Path

import numpy as np
import pytest
from phasesmith.io.hdf5 import Hdf5Frames

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "crates/phasesmith-hdf5/tests/data/counts-bitshuffle-lz4.h5"
)


def test_native_decoded_cache_is_bounded_and_output_mutation_isolated():
    with Hdf5Frames(FIXTURE, "detector/data", cache_bytes=48) as reader:
        a = reader.read_frame(0)
        expected = a.copy()
        a[:] = 42
        assert reader.read_frame(0).tobytes() == expected.tobytes()
        reader.read_frame(1)
        assert reader.read_frame(0).tobytes() == expected.tobytes()
        stats = reader.cache_stats
        assert stats["requests"] == 4
        assert stats["decodes"] == 2
        assert stats["hits"] == 2
        assert stats["frames"] == 1
        assert stats["payload_bytes"] == stats["budget_bytes"] == 48
    with Hdf5Frames(FIXTURE, "detector/data", cache_bytes=1) as tiny:
        assert tiny.read_frame(0).tobytes() == tiny.read_frame(0).tobytes()
        assert tiny.cache_stats["hits"] == 0
        assert tiny.cache_stats["payload_bytes"] == 0
    for value in [-1, True, 4 * 1024**3 + 1]:
        with pytest.raises(ValueError, match="cache"):
            Hdf5Frames(FIXTURE, "detector/data", cache_bytes=value)


def test_two_source_handles_never_share_cached_data():
    with Hdf5Frames(FIXTURE, "detector/data", cache_bytes=1024) as first:
        first.read_frame(0)
        with Hdf5Frames(FIXTURE, "detector/data", cache_bytes=1024) as second:
            assert second.cache_stats["frames"] == 0
            np.testing.assert_array_equal(first.read_frame(0), second.read_frame(0))
            assert second.cache_stats["decodes"] == 1
