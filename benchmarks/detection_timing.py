"""Observational detection timings; overlapping work is reported separately."""

import functools
import threading
import time
from contextlib import contextmanager


@contextmanager
def observe_detection(workflow):
    from phasesmith.experimental import frame_validity, rotation
    from phasesmith.io.hdf5 import Hdf5Frames

    stats = dict(
        schema="phasesmith.detection-wall-timings.v1",
        image_load_wall_seconds=0.0,
        mask_load_wall_seconds=0.0,
        hdf5_decode_wall_seconds=0.0,
        source_verification_wall_seconds=0.0,
        frame_measurement_thread_seconds=0.0,
        linking_wall_seconds=0.0,
    )
    active = False
    lock = threading.Lock()
    originals = []

    def patch(owner, name, replacement):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        setattr(owner, name, replacement(original))

    def timed(key):
        def decorate(function):
            @functools.wraps(function)
            def call(*args, **kwargs):
                if not active:
                    return function(*args, **kwargs)
                start = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    with lock:
                        stats[key] += elapsed

            return call

        return decorate

    def detected(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                yield from function(*args, **kwargs)
            finally:
                stats["ordered_detection_wall_seconds"] = time.perf_counter() - start

        return call

    def located(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            nonlocal active
            active = True
            stats["workers"] = kwargs.get("workers", 1)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stats["total_wall_seconds"] = time.perf_counter() - start
                stats["track_assembly_wall_seconds"] = (
                    stats["total_wall_seconds"]
                    - stats.get("ordered_detection_wall_seconds", 0.0)
                    - stats["linking_wall_seconds"]
                )
                active = False

        return call

    patch(workflow.RawFrames, "load", timed("image_load_wall_seconds"))
    patch(workflow.RawFrames, "mask", timed("mask_load_wall_seconds"))
    patch(Hdf5Frames, "read_frame", timed("hdf5_decode_wall_seconds"))
    patch(frame_validity._MaskAccess, "verify_frame", timed("source_verification_wall_seconds"))
    patch(rotation, "_measure_frame", timed("frame_measurement_thread_seconds"))
    patch(rotation, "link_spots", timed("linking_wall_seconds"))
    patch(rotation, "_detect_frames", detected)
    patch(rotation, "locate_series", located)
    stats["interpretation"] = (
        "Ordered detection, linking and track assembly partition total wall time. Image load "
        "includes HDF5 decode, orientation, source verification, reserved masks and raw hashing. "
        "Decode and verification are nested in image load. Image/mask loading overlaps native "
        "worker tasks; summed measurement thread seconds must not be added to wall categories. "
        "With native count preparation, hash/verification/composed-mask work occurs inside "
        "image load; the Python verification timer can be zero while all checks still run."
    )
    try:
        yield stats
    finally:
        for owner, name, function in reversed(originals):
            setattr(owner, name, function)
