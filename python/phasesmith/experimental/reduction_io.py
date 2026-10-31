"""Portable, pickle-free snapshots for experimental profile observations/banks."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .integration import IntegrationOptions, ReflectionTable
from .profiles import ProfileModel, ProfileObservations, ProfileOptions


def save_case(path, observations, model=None, *, compression_level=None):
    t = observations.summation
    metadata = dict(
        schema="phasesmith.experimental.profile-case.v1",
        observation_fingerprint=observations.fingerprint(),
        frame_hashes=list(t.frame_hashes),
        options=asdict(t.options),
        noise_model=t.noise_model,
        noise_parameters=t.noise_parameters,
        frame_execution=t.execution,
    )
    names = (
        "hkl",
        "predicted",
        "grain_ids",
        "values",
        "flags",
        "observation_ids",
        "frame_indices",
        "boxes",
        "frame_values",
        "frame_flags",
        "omega_bounds",
        "background_models",
        "angular_half_widths",
    )
    arrays = {"table_" + k: getattr(t, k) for k in names if getattr(t, k) is not None}
    arrays.update(
        {
            "obs_" + k: getattr(observations, k)
            for k in (
                "samples",
                "valid",
                "offsets",
                "covariance",
                "centres",
                "scales",
                "planar_transforms",
            )
        }
    )
    if model is not None:
        metadata["model"] = dict(
            options=asdict(model.options),
            detector_shape=model.detector_shape,
            angular_range=model.angular_range,
            training_fingerprint=model.training_fingerprint,
            execution=model.execution,
        )
        arrays.update(
            {
                "bank_" + k: getattr(model, k)
                for k in ("profiles", "counts", "shifts", "reference_ids", "reference_centres")
                if getattr(model, k) is not None
            }
        )
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8
    )
    # Exclusive create: archival cases are never silently overwritten.
    if compression_level is None:
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    else:
        from phasesmith.io.numeric_archive import save_npz

        save_npz(path, arrays, compression_level=compression_level)


def load_case(path):
    with np.load(path, allow_pickle=False) as z:
        a = dict(z)
    meta = json.loads(a.pop("metadata_json").tobytes())
    if meta.get("schema") != "phasesmith.experimental.profile-case.v1":
        raise ValueError("unsupported profile case schema")
    fields = {k.removeprefix("table_"): v for k, v in a.items() if k.startswith("table_")}
    table = ReflectionTable(
        **fields,
        frame_hashes=tuple(meta["frame_hashes"]),
        execution=meta.get("frame_execution"),
        options=IntegrationOptions(**meta["options"]),
        noise_model=meta["noise_model"],
        noise_parameters=None
        if meta["noise_parameters"] is None
        else tuple(meta["noise_parameters"]),
    )
    obs = ProfileObservations(
        table, **{k.removeprefix("obs_"): v for k, v in a.items() if k.startswith("obs_")}
    )
    model = None
    if "model" in meta:
        m = meta["model"]
        model = ProfileModel(
            **{k.removeprefix("bank_"): v for k, v in a.items() if k.startswith("bank_")},
            options=ProfileOptions(**m["options"]),
            detector_shape=tuple(m["detector_shape"]),
            angular_range=tuple(m["angular_range"]),
            training_fingerprint=m["training_fingerprint"],
            execution=m.get("execution"),
        )
    if meta.get("observation_fingerprint", obs.fingerprint()) != obs.fingerprint():
        raise ValueError("profile case observation fingerprint mismatch")
    return obs, model
