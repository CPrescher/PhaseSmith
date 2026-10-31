"""Explicitly generate the small independent h5py/Bitshuffle interoperability fixture.

Requires test-only h5py and hdf5plugin. Existing fixture files are never replaced.
Run from any directory: python tools/generate_hdf5_test_fixture.py
"""

import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

out = Path(__file__).resolve().parents[1] / "crates/phasesmith-hdf5/tests/data"
out.mkdir(parents=True, exist_ok=True)
path = out / "counts-bitshuffle-lz4.h5"
manifest = out / "counts-bitshuffle-lz4.json"
if path.exists() or manifest.exists():
    raise FileExistsError("fixture exists; explicitly review/remove it before regeneration")
hdf5plugin.register(filters=("bshuf",), force=True)
counts = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
counts[0, 0, :4] = [0, 70000, 2**32 - 2, 2**32 - 1]
with h5py.File(path, "x") as file:
    dataset = file.create_dataset(
        "detector/data",
        data=counts,
        chunks=(1, 3, 4),
        track_times=False,
        **hdf5plugin.Bitshuffle(cname="lz4"),
    )
    for axis in ("x", "y"):
        file.create_dataset(
            f"detector/detector_information/pixel_size/{axis}size", data=75e-6, track_times=False
        )
    filters = [
        dataset.id.get_create_plist().get_filter(i)[:3]
        for i in range(dataset.id.get_create_plist().get_nfilters())
    ]
record = dict(
    schema="phasesmith.hdf5-test-fixture.v1",
    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    shape=list(counts.shape),
    source_dtype=str(counts.dtype),
    filters=filters,
    producer=dict(
        h5py=h5py.__version__, hdf5=h5py.version.hdf5_version, hdf5plugin=hdf5plugin.version
    ),
    generation_command="python tools/generate_hdf5_test_fixture.py",
    frame_sha256=[hashlib.sha256(frame.tobytes()).hexdigest() for frame in counts],
)
manifest.write_text(json.dumps(record, indent=2) + "\n")
