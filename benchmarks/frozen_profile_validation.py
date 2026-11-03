"""Reflection-family validation of shapes learned only from strong references.

Unlike the earlier withheld-image experiment, every scan region may supply
independent reference reflections. Entire abs-HKL families remain withheld.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from benchmark_3d_profiles import load_case
from phasesmith.experimental.profiles import (
    fit_learned_profiles,
    learn_profile_variations,
    learn_profiles,
)


def main(source, output):
    output.mkdir(parents=True, exist_ok=False)
    o, old, a = load_case(source)
    keys = np.abs(a["hkl"])
    heldout = np.array(
        [
            int.from_bytes(
                hashlib.sha256(
                    np.asarray(h, dtype="<i8").tobytes() + b"PhaseSmith-intensity-v1"
                ).digest()[:8],
                "little",
            )
            % 5
            == 0
            for h in keys
        ]
    )
    training = ~heldout
    options = replace(old.options, regions=(1, 1, 1), reference_weighting="equal", cutoff=0.0)
    start = time.perf_counter()
    model = learn_profiles(o, training, detector_shape=old.detector_shape, options=options)
    base_seconds = time.perf_counter() - start
    start = time.perf_counter()
    variations = learn_profile_variations(o, model)
    variation_seconds = time.perf_counter() - start
    assert not heldout[model.reference_ids].any()
    assert not heldout[variations.reference_ids].any()
    start = time.perf_counter()
    fitted = fit_learned_profiles(o, model, variations)
    fit_seconds = time.perf_counter() - start
    np.save(output / "excluded-hkl-families.npy", np.unique(keys[heldout], axis=0))
    extras = dict(
        coefficients=fitted.coefficients,
        coefficient_covariance=fitted.coefficient_covariance,
        variation_reference_ids=variations.reference_ids,
    )
    np.savez_compressed(
        output / "native.npz",
        **{
            k: v
            for k, v in a.items()
            if k
            not in [
                "profile",
                "profile_flags",
                "model_ids",
                "reference_membership",
                "training",
                "heldout",
                "coefficients",
                "coefficient_covariance",
                "variation_reference_ids",
            ]
        },
        profile=fitted.values,
        profile_flags=fitted.flags,
        model_ids=fitted.model_ids,
        reference_membership=fitted.reference_membership,
        training=training,
        heldout=heldout,
        **extras,
    )
    np.savez_compressed(
        output / "model.npz",
        profiles=model.profiles,
        counts=model.counts,
        shifts=model.shifts,
        reference_ids=model.reference_ids,
        reference_centres=model.reference_centres,
        variation_factors=variations.factors,
        variation_centres=variations.centres,
        variation_fractions=variations.fractions,
        variation_reference_ids=variations.reference_ids,
        variation_model_ids=variations.model_ids,
    )
    os.link(source / "observations.npz", output / "observations.npz")
    protocol = json.loads((source / "protocol.json").read_text())
    protocol.update(
        profile_options=asdict(options),
        frozen_variations=True,
        source_prepared_run=str(source.resolve()),
        validation=(
            "20 percent of abs-HKL families, SHA256 fixed salt; entire families "
            "excluded from both learning stages"
        ),
        learning=(
            "global scattering-coordinate profile; physical width fractions from "
            "strong references only; frozen nearest-reference medians for target "
            "measurements"
        ),
        heldout=(
            "reflection-family cross-validation throughout the scan; geometry fitted "
            "before this split"
        ),
        geometric_coverage=(
            "reported fitted nonnegative mixture mass in recorded valid pixels; "
            "fitting profile fractions are fixed independently of target counts"
        ),
        correction_scope=(
            "raw detector counts and local background; no intensity matching scale, "
            "absorption correction or structure prior"
        ),
        no_blind_validation_claim=True,
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    original_summary = json.loads((source / "native-summary.json").read_text())
    summary = dict(
        predictions=len(o.centres),
        groups=len(o.summation.observation_ids),
        samples=len(o.samples),
        references=model.counts.tolist(),
        variation_references=len(variations.reference_ids),
        training=int(training.sum()),
        heldout=int(heldout.sum()),
        fingerprint=o.fingerprint(),
        frame_hashes=original_summary["frame_hashes"],
        learning_seconds=base_seconds,
        variation_learning_seconds=variation_seconds,
        fitting_seconds=fit_seconds,
        profile_unflagged=int(np.count_nonzero(fitted.flags == 0)),
    )
    (output / "native-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    snapshot = output / "source-snapshot"
    for pattern in (
        "crates/phasesmith-spots/src/*.rs",
        "crates/phasesmith-py/src/*.rs",
        "python/phasesmith/experimental/*.py",
        "benchmarks/*profiles*.py",
        "benchmarks/frozen_profile_validation.py",
    ):
        for path in Path(".").glob(pattern):
            if not path.is_file():
                continue
            target = snapshot / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.source, a.output)
