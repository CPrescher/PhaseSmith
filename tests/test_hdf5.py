"""Interoperability with a tiny independent h5py/Bitshuffle fixture."""

import builtins
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.io.hdf5 import Hdf5Frames

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "NativeHdf5Frames"), reason="bundled HDF5 feature build"
)
ROOT = Path(__file__).resolve().parents[1] / "crates/phasesmith-hdf5/tests/data"
PATH = ROOT / "counts-bitshuffle-lz4.h5"


def test_independent_compressed_counts_metadata_and_handle_ownership(monkeypatch):
    original_import = builtins.__import__

    def no_python_hdf5(name, *args, **kwargs):
        if name.split(".")[0] in ("h5py", "hdf5plugin"):
            raise AssertionError("production HDF5 must execute in Rust")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_python_hdf5)
    manifest = json.loads(PATH.with_suffix(".json").read_text())
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == manifest["sha256"]
    with Hdf5Frames(PATH, "detector/data") as reader:
        assert reader.shape == (2, 3, 4) and reader.dtype == "uint32"
        assert reader.filters[0][0] == 32008 and reader.filters[0][1]
        assert reader.scalar("detector/detector_information/pixel_size/xsize") == 75e-6
        with ThreadPoolExecutor(max_workers=4) as pool:
            frames = list(pool.map(reader.read_frame, [0, 1, 0, 1]))
        for index, frame in enumerate(frames):
            assert frame.dtype == np.uint32 and frame.flags.c_contiguous
            assert (
                hashlib.sha256(frame.tobytes()).hexdigest() == manifest["frame_sha256"][index % 2]
            )
        frames[0][:] = 7
        assert reader.read_frame(0)[0].tolist() == [0, 70000, 2**32 - 2, 2**32 - 1]
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.read_frame(0)
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == manifest["sha256"]


@pytest.mark.parametrize("bad", [-1, 2, True, 0.5])
def test_invalid_indices_are_rejected(bad):
    with Hdf5Frames(PATH, "detector/data") as reader, pytest.raises(ValueError):
        reader.read_frame(bad)


@pytest.mark.parametrize(
    "limits",
    [
        dict(max_frames=1),
        dict(max_frame_pixels=11),
        dict(max_chunk_bytes=47),
        dict(max_frame_pixels=0),
    ],
)
def test_admission_precedes_frame_reads(limits):
    with pytest.raises(ValueError):
        Hdf5Frames(PATH, "detector/data", **limits)
