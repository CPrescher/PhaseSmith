"""External DIALS-only position probe; never imported by the PhaseSmith package.

Run using the pinned isolated DIALS Python. The lossless NumPy image adapter
supplies explicit geometry; it implements no DIALS detection/indexing algorithms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from dials.array_family import flex
from dxtbx.format.Format import Format
from dxtbx.imageset import ImageSetFactory
from dxtbx.model import BeamFactory, DetectorFactory, GoniometerFactory, ScanFactory
from dxtbx.model.experiment_list import Experiment, ExperimentList


def models(meta, sign):
    vectors = np.asarray(meta["vectors"])
    center, fast, slow, incident, _ = vectors
    pixel_size = np.linalg.norm([fast, slow], axis=1)
    detector = DetectorFactory.make_detector(
        "PAD",
        tuple(fast / pixel_size[0]),
        tuple(slow / pixel_size[1]),
        tuple((center - (fast + slow) / 2) * 1000),
        tuple(pixel_size * 1000),
        tuple(meta["shape"][::-1]),
        trusted_range=(0.5, 65536),
        name="Explicit affine benchmark plane; no detector-specific corrections",
        gain=1,
    )
    beam = BeamFactory.make_beam(sample_to_source=tuple(-incident), wavelength=meta["wavelength"])
    goniometer = GoniometerFactory.known_axis((sign, 0, 0))
    start, end = meta["omega_bounds"][0]
    scan = ScanFactory.make_scan(
        (1, len(meta["files"])),
        [0.5] * len(meta["files"]),
        (start, end - start),
        {i + 1: i * 0.5 for i in range(len(meta["files"]))},
        deg=True,
    )
    return beam, detector, goniometer, scan


class FormatPhaseSmithBenchmark(Format):
    """Task-local lossless pixel loader with explicit calibrated models."""

    @staticmethod
    def understand(image_file):
        return Path(image_file).suffix == ".npy"

    def _start(self):
        self.meta = json.loads((Path(self._image_file).parent / "geometry.json").read_text())

    def _beam(self):
        return models(self.meta, -1)[0]

    def _detector(self):
        return models(self.meta, -1)[1]

    def _goniometer(self):
        return models(self.meta, -1)[2]

    def _scan(self):
        return models(self.meta, -1)[3]

    def get_raw_data(self):
        # uint16 -> int32 is exact; no resampling or count correction.
        return flex.int(np.load(self._image_file).astype(np.int32))

    def get_static_mask(self):
        return None  # Per-frame zeros are excluded by the trusted lower bound.


def experiments(case, sign):
    meta = json.loads((case / "images/geometry.json").read_text())
    beam, detector, goniometer, scan = models(meta, sign)
    images = ImageSetFactory.make_sequence(
        str(case / "images/frame_####.npy"),
        list(range(1, len(meta["files"]) + 1)),
        format_class=FormatPhaseSmithBenchmark,
        beam=beam,
        detector=detector,
        goniometer=goniometer,
        scan=scan,
    )
    return ExperimentList(
        [
            Experiment(
                imageset=images, beam=beam, detector=detector, goniometer=goniometer, scan=scan
            )
        ]
    ), meta


def check_geometry(case):
    errors = []
    for sign in (-1, 1):
        exp, meta = experiments(case, sign)
        xyz = np.array([[0.5, 0.5, 0.5], [1200.5, 1000.5, 38.5], [2047.5, 2047.5, 75.5]])
        t = flex.reflection_table()
        t["id"] = flex.int(3, 0)
        t["panel"] = flex.size_t(3, 0)
        t["xyzobs.px.value"] = flex.vec3_double(xyz)
        t["xyzobs.px.variance"] = flex.vec3_double(3, (0.25, 0.25, 1 / 12))
        t.centroid_px_to_mm(exp)
        t.map_centroids_to_reciprocal_space(exp)
        vectors = np.array(meta["vectors"])
        point = vectors[0] + (xyz[:, :1] - 0.5) * vectors[1] + (xyz[:, 1:2] - 0.5) * vectors[2]
        q = (point / np.linalg.norm(point, axis=1)[:, None] - vectors[3]) / meta["wavelength"]
        omega = np.radians(
            exp[0].scan.get_angle_from_array_index(
                flex.double(np.ascontiguousarray(xyz[:, 2])), deg=True
            )
        )
        c, s = np.cos(-sign * omega), np.sin(-sign * omega)
        expected = np.column_stack([q[:, 0], c * q[:, 1] - s * q[:, 2], s * q[:, 1] + c * q[:, 2]])
        error = float(np.max(np.abs(expected - t["rlp"].as_numpy_array())))
        assert error < 1e-12, (sign, error, expected, t["rlp"].as_numpy_array())
        # The per-frame trusted mask must exclude exactly zero counts.
        data = np.load(case / "images/frame_0001.npy")
        mask = exp[0].imageset.get_mask(0)[0].as_numpy_array()
        assert np.array_equal(mask, data != 0)
        errors.append(
            dict(sign=sign, max_reciprocal_error=error, mask_exact=True, pixels=int(data.size))
        )
    (case / "dials-geometry-check.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(json.dumps(errors), flush=True)


def find(case, output, sigma, workers=4, repeats=1, warmup=0):
    from dials.command_line.find_spots import phil_scope
    from dials.util.version import dials_version

    exp, meta = experiments(case, -1)
    from dials.util.phil import parse

    p = phil_scope.fetch(source=parse("")).extract()
    p.spotfinder.mp.nproc = workers
    p.spotfinder.mp.chunksize = 4
    p.spotfinder.filter.min_spot_size = 3
    if sigma is not None:
        p.spotfinder.threshold.dispersion.sigma_strong = sigma
    # Verify that no placeholder PHIL untrusted region masks the whole panel.
    from dials.algorithms.spot_finding.factory import SpotFinderFactory

    finder = SpotFinderFactory.from_parameters(experiments=exp, params=p)
    assert finder.mask_generator(exp[0].imageset)[0].count(False) == 0
    # Preserve the package defaults for dispersion/background/size filtering.
    output.mkdir(parents=True, exist_ok=False)
    (output / "parameters.phil").write_text(phil_scope.format(python_object=p).as_str())
    times, cpus = [], []
    previous = None
    for repeat in range(repeats + warmup):
        start = time.perf_counter()
        cpu = time.process_time()
        table = flex.reflection_table.from_observations(exp, p)
        seconds = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu
        xyz_check = table["xyzobs.px.value"].as_numpy_array()
        boxes_check = np.array(list(table["bbox"]), dtype=np.int64).reshape(-1, 6)
        order = np.lexsort(np.column_stack([xyz_check, boxes_check]).T[::-1])
        current = np.column_stack([xyz_check, boxes_check])[order]
        if previous is not None:
            assert previous.tobytes() == current.tobytes(), "DIALS repeat differs"
        previous = current.copy()
        if repeat >= warmup:
            times.append(seconds)
            cpus.append(cpu_seconds)
    seconds = float(np.median(times))
    table.as_file(str(output / "strong.refl"))
    xyz = table["xyzobs.px.value"].as_numpy_array()
    centroid = xyz.copy()
    centroid[:, :2] -= 0.5
    centroid[:, 2] = exp[0].scan.get_angle_from_array_index(
        flex.double(np.ascontiguousarray(xyz[:, 2])), deg=True
    )
    box = np.array(list(table["bbox"]), dtype=np.int64).reshape(-1, 6)
    # Check DIALS's reciprocal mapping against the same first-principles vectors.
    table.centroid_px_to_mm(exp)
    table.map_centroids_to_reciprocal_space(exp)
    np.savez(
        output / "positions.npz",
        xyz=centroid,
        bbox=box,
        ranges=box[:, 4:6],
        flags=table["flags"].as_numpy_array(),
        q=table["rlp"].as_numpy_array(),
        centroid_variance=table["xyzobs.px.variance"].as_numpy_array(),
    )
    table.as_file(str(output / "strong.refl"))
    exp.as_file(str(output / "imported.expt"))
    record = dict(
        version=dials_version(),
        seconds=seconds,
        positions=len(table),
        threshold_sigma=sigma,
        min_spot_size=3,
        workers=workers,
        times_s=times,
        cpu_parent_s=cpus,
        warmups=warmup,
        repeats_exact=True,
        algorithm=p.spotfinder.threshold.algorithm,
        geometry=meta,
        convention="pixel centres minus0.5; omega degrees",
    )
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "geometry"}), flush=True)


def index(case, positions, output, sign):
    from dials.algorithms.indexing.indexer import Indexer
    from dials.command_line.index import phil_scope

    exp, _ = experiments(case, sign)
    table = flex.reflection_table.from_file(str(positions / "strong.refl"))
    box = np.array(list(table["bbox"]), dtype=np.int64).reshape(-1, 6)
    first, last = box[:, 4], box[:, 5]
    block = (first // 5) % 5
    selection = (
        (last - first <= 8)
        & (first > 0)
        & (last < len(exp[0].imageset))
        & (((last - 1) // 5) % 5 == block)
        & (block != 0)
    )
    table = table.select(flex.bool(selection))
    table.centroid_px_to_mm(exp)
    table.map_centroids_to_reciprocal_space(exp)
    output.mkdir(parents=True, exist_ok=False)
    from dials.util.phil import parse

    p = phil_scope.fetch(
        source=parse("""
indexing.nproc=4
indexing.min_cell=1.5
indexing.max_cell=30
indexing.min_cell_volume=15
indexing.multiple_lattice_search.max_lattices=8
refinement.parameterisation.detector.fix=all
refinement.parameterisation.beam.fix=all
refinement.parameterisation.goniometer.fix=all
""")
    ).extract()
    (output / "parameters.phil").write_text(phil_scope.format(python_object=p).as_str())
    started = time.perf_counter()
    result = Indexer.from_parameters(table, exp, params=p)
    result.index()
    result.refined_experiments.as_file(str(output / "indexed.expt"))
    result.refined_reflections.as_file(str(output / "indexed.refl"))
    crystals = []
    for item in result.refined_experiments:
        if item.crystal is not None:
            a = np.asarray(item.crystal.get_A()).reshape(3, 3)
            crystals.append(
                dict(
                    cell=item.crystal.get_unit_cell().parameters(),
                    direct_basis=np.linalg.inv(a).tolist(),
                )
            )
    record = dict(
        seconds=time.perf_counter() - started,
        sign=sign,
        training_positions=len(table),
        crystals=crystals,
    )
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["find", "index", "geometry"])
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--sign", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    if args.action == "geometry":
        check_geometry(args.case)
    elif args.action == "find":
        find(args.case, args.output, args.sigma, args.workers, args.repeats, args.warmup)
    else:
        index(args.case, args.positions, args.output, args.sign)


if __name__ == "__main__":
    main()
