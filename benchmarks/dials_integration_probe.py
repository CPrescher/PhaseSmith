"""External DIALS-only integration probe over the preserved TIFF pixel adapter.

Run with the isolated DIALS Python. Uses its own previously discovered crystal,
not a PhaseSmith cell or corrected-intensity reference. No runtime dependency.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from dials.array_family import flex
from dials.command_line.integrate import run_integration, working_phil
from dials.util.phil import parse
from dials.util.version import dials_version
from dials_position_probe import experiments
from dxtbx.model.experiment_list import ExperimentListFactory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--indexed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crystal-index", type=int, default=0)
    p.add_argument("--sensor", choices=("CCD", "PAD"), default="CCD")
    p.add_argument("--exclude-hkl", type=Path)
    p.add_argument(
        "--training-frames",
        type=Path,
        help="Optional frozen boolean mask limiting profile-reference exposures",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    workflow_start = time.perf_counter()
    exp, meta = experiments(args.case.resolve(), -1)
    exp[0].detector[0].set_type("SENSOR_" + args.sensor)
    exp[0].imageset.set_detector(exp[0].detector)
    indexed = ExperimentListFactory.from_json_file(
        str(args.indexed / "indexed.expt"), check_format=False
    )
    if not 0 <= args.crystal_index < len(indexed):
        raise ValueError("selected frozen crystal outside experiment list")
    exp[0].crystal = indexed[args.crystal_index].crystal
    exp[0].identifier = indexed[args.crystal_index].identifier
    exp[0].detector[0].set_trusted_range((0.5, 65534.5))
    reference = flex.reflection_table.from_file(str(args.indexed / "indexed.refl"))
    reference = reference.select(reference["id"] == args.crystal_index)
    reference["id"] = flex.int(len(reference), 0)
    for key in list(reference.experiment_identifiers().keys()):
        del reference.experiment_identifiers()[key]
    reference.experiment_identifiers()[0] = exp[0].identifier
    if args.exclude_hkl is not None:
        excluded = {tuple(h) for h in np.load(args.exclude_hkl)}
        selected = np.array(
            [tuple(abs(h)) not in excluded for h in np.asarray(reference["miller_index"])]
        )
        reference = reference.select(flex.bool(selected))
    boxes = np.asarray(list(reference["bbox"]))
    keep = np.ones(len(reference), bool)
    if args.training_frames is not None:
        training = np.load(args.training_frames)
        if training.dtype != bool or training.shape != (len(meta["files"]),):
            raise ValueError("training frame mask differs from the source scan")
        for i, box in enumerate(boxes):
            keep[i] = bool(training[max(0, box[4]) : min(len(training), box[5])].all())
    # Remove ceiling-hit reference shoeboxes before learning profiles.
    for frame in range(len(meta["files"])):
        data = np.load(args.case / "images" / f"frame_{frame + 1:04d}.npy")
        expected = (data != 0) & (data < 65535)
        assert np.array_equal(exp[0].imageset.get_mask(frame)[0].as_numpy_array(), expected)
        for i in np.flatnonzero((boxes[:, 4] <= frame) & (boxes[:, 5] > frame) & keep):
            x0, y0, x1, y1 = boxes[i, [0, 2, 1, 3]]
            if np.any(
                data[max(0, y0) : min(data.shape[0], y1), max(0, x0) : min(data.shape[1], x1)]
                >= 65535
            ):
                keep[i] = False
    removed = int(np.count_nonzero(~keep))
    reference = reference.select(flex.bool(keep))
    params = working_phil.fetch(
        source=parse("""
integration.mp.nproc=1
output.output_unintegrated_reflections=True
""")
    ).extract()
    (args.output / "parameters.phil").write_text(working_phil.format(python_object=params).as_str())
    protocol = dict(
        dials_version=dials_version(),
        selected_crystal=args.crystal_index,
        total_indexed_crystals=len(indexed),
        excluded_hkl=None if args.exclude_hkl is None else str(args.exclude_hkl.resolve()),
        excluded_hkl_sha256=None
        if args.exclude_hkl is None
        else hashlib.sha256(args.exclude_hkl.read_bytes()).hexdigest(),
        detector_sensor=exp[0].detector[0].get_type(),
        gain=exp[0].detector[0].get_gain(),
        case=str(args.case.resolve()),
        indexed_source=str(args.indexed.resolve()),
        indexed_experiment_sha256=hashlib.sha256(
            (args.indexed / "indexed.expt").read_bytes()
        ).hexdigest(),
        geometry=(
            "same TIFF pixels/affine calibration as prior position probe; integer ceiling excluded"
        ),
        reference_ceiling_boxes_excluded=removed,
        profile_reference_reflections=len(reference),
        training_frames=None
        if args.training_frames is None
        else str(args.training_frames.resolve()),
        training_frames_sha256=None
        if args.training_frames is None
        else hashlib.sha256(args.training_frames.read_bytes()).hexdigest(),
        options=(
            "default DIALS integration/profile/background algorithms; serial; "
            "retain unintegrated rows"
        ),
        interpretation=(
            "exploratory actual integration, not a controlled intensity-accuracy benchmark"
        ),
        limitations=(
            "DIALS owns its previous crystal/HKL basis and reference selection; "
            "counts/correction/partiality conventions must be reconciled "
            "before comparing intensities"
        ),
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start = time.perf_counter()
    preparation_seconds = start - workflow_start
    exp, result, _report = run_integration(params, exp, reference)
    integration_seconds = time.perf_counter() - start
    export_start = time.perf_counter()
    result.as_file(str(args.output / "integrated.refl"))
    exp.as_file(str(args.output / "integrated.expt"))
    keys = [
        k
        for k in result.keys()  # noqa: SIM118 -- C++ reflection table, not a Python dict.
        if any(
            s in k
            for s in (
                "intensity.",
                "background.",
                "partiality",
                "lp",
                "qe",
                "xyzcal",
                "miller_index",
                "flags",
            )
        )
    ]
    output_arrays = {k: np.asarray(result[k]) for k in keys}
    for name in (
        "integrated_sum",
        "integrated_prf",
        "overloaded",
        "dont_integrate",
        "overlapped_fg",
        "overlapped_bg",
        "failed_during_background_modelling",
        "failed_during_summation",
        "failed_during_profile_fitting",
    ):
        if hasattr(result.flags, name):
            output_arrays["quality." + name] = np.asarray(
                result.get_flags(getattr(result.flags, name))
            )
    np.savez_compressed(args.output / "arrays.npz", **output_arrays)
    record = dict(
        seconds=time.perf_counter() - start,
        preparation_seconds=preparation_seconds,
        integration_seconds=integration_seconds,
        export_seconds=time.perf_counter() - export_start,
        workflow_seconds=preparation_seconds + integration_seconds,
        timer_scope=(
            "post-indexing reference preparation, cached-pixel checks and serial integration; "
            "imports, indexing/refinement execution and exports excluded from workflow_seconds"
        ),
        reflections=len(result),
        integrated_sum=int(result.get_flags(result.flags.integrated_sum).count(True)),
        integrated_profile=int(result.get_flags(result.flags.integrated_prf).count(True)),
        overloaded=int(result.get_flags(result.flags.overloaded).count(True)),
        profile=exp[0].profile.to_dict(),
        columns=keys,
        protocol=protocol,
    )
    (args.output / "report.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
