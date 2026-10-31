"""Measure real raw-frame detection/linking with frozen original-count masks.

Each trial rereads all 142 original HDF5 frames. Archived position arrays are
used only after the trial for exact comparison, never to generate detections.
Thread work times overlap and must not be added to serial wall categories.
"""

import argparse
import hashlib
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np
from enstatite_raw_workflow import RawFrames, poni_geometry
from phasesmith.experimental import rotation
from phasesmith.experimental.frame_validity import CombinedValidity, DacApertureMask, FrameValidity
from phasesmith.experimental.spots import SpotOptions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", nargs="+", type=int, default=[4, 6, 8, 10])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--reuse-verified-digest", action="store_true")
    args = p.parse_args()
    packet = json.loads((args.run / "worker-input.json").read_text())
    protocol = packet["protocol"]
    frames = RawFrames(packet["raw_path"], protocol["acquisition"], packet["raw_array"])
    geometry = poni_geometry(
        packet["poni_path"], frames.pixel1, frames.pixel2, protocol["acquisition"]["axis"]
    )
    estimated = FrameValidity.load(args.run / "frame-validity.npz")
    geometric = DacApertureMask(
        geometry,
        frames.bounds,
        frames.reader.shape[1:],
        frame_ids=estimated.frame_ids,
        **protocol["shadow_mask"]["geometric"],
    )
    validity = CombinedValidity((estimated, geometric))
    frames.close()
    original_detect = rotation._detect_frames
    original_measure = rotation._measure_frame
    original_link = rotation.link_spots
    records, hashes = [], []
    for workers in args.workers:
        repeats = []
        for _ in range(args.repeats):
            stats = dict(
                image_load_wall_seconds=0.0,
                mask_load_wall_seconds=0.0,
                frame_measurement_thread_seconds=0.0,
                linking_wall_seconds=0.0,
            )
            lock = threading.Lock()

            def timer(name, function, stats=stats, lock=lock):
                def call(*a, **kw):
                    start = time.perf_counter()
                    try:
                        return function(*a, **kw)
                    finally:
                        with lock:
                            stats[name] += time.perf_counter() - start

                return call

            def detect(*a, _stats=stats, **kw):
                start = time.perf_counter()
                try:
                    yield from original_detect(*a, **kw)
                finally:
                    _stats["ordered_detection_wall_seconds"] = time.perf_counter() - start

            rotation._detect_frames = detect
            rotation._measure_frame = timer("frame_measurement_thread_seconds", original_measure)
            rotation.link_spots = timer("linking_wall_seconds", original_link)
            frames = RawFrames(
                packet["raw_path"],
                protocol["acquisition"],
                packet["raw_array"],
                validity,
                execution={"reuse_verified_digest": args.reuse_verified_digest},
            )
            start = time.perf_counter()
            try:
                result = rotation.locate_series(
                    timer("image_load_wall_seconds", frames.load),
                    frames.bounds,
                    mask_loader=timer("mask_load_wall_seconds", frames.mask),
                    spot_options=SpotOptions(**protocol["detection"]),
                    rotation_options=rotation.RotationOptions(**protocol["linking"]),
                    gain=protocol["acquisition"]["gain"],
                    read_noise=protocol["acquisition"]["read_noise"],
                    workers=workers,
                )
                stats["total_wall_seconds"] = time.perf_counter() - start
                stats["track_assembly_wall_seconds"] = (
                    stats["total_wall_seconds"]
                    - stats["ordered_detection_wall_seconds"]
                    - stats["linking_wall_seconds"]
                )
            finally:
                frames.close()
                rotation._detect_frames = original_detect
                rotation._measure_frame = original_measure
                rotation.link_spots = original_link
            assert frames.loads == 142
            h = hashlib.sha256()
            with np.load(args.run / "positions.npz", allow_pickle=False) as baseline:
                for name in [
                    "boxes",
                    "frame_ranges",
                    "values",
                    "flags",
                    "observations",
                    "omega_bounds",
                ]:
                    value = getattr(result, name)
                    expected = baseline["original_flags" if name == "flags" else name]
                    assert value.dtype == expected.dtype and value.shape == expected.shape
                    assert value.tobytes() == expected.tobytes(), name
                    h.update(value.tobytes())
            expected_hashes = json.loads((args.run / "positions.json").read_text())["frame_hashes"]
            assert list(result.frame_hashes) == expected_hashes
            hashes.append(h.hexdigest())
            stats["sha256"] = hashes[-1]
            repeats.append(stats)
        record = dict(
            workers=workers,
            repeats=repeats,
            median_seconds=float(np.median([r["total_wall_seconds"] for r in repeats])),
            tracks=len(result),
            detections=len(result.observations),
            all_position_arrays_bitwise_identical=True,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    assert len(set(hashes)) == 1
    output = dict(
        schema="phasesmith.detection-execution.v1",
        records=records,
        maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interpretation=(
            "Three sequential repeats per worker count. All raw frames reread; frozen masks "
            "reused only in isolated benchmark. Serial load times overlap worker tasks; "
            "thread seconds overlap each other. Ordered detection, linking and track "
            "assembly partition total wall time."
        ),
    )
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
