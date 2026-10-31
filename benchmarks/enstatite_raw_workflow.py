"""Explicit external-data raw-frame to PhaseSmith candidate-search example.

No Dioptas, vendor HKLs/peak tables, archived reductions, atomic models or external
structure solvers are used by the numerical worker. Supplied cell/symmetry and
scan/calibration assumptions are explicit. Large products must live outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_single_crystal_data import verify_dataset  # noqa: E402

PROTOCOL = ROOT / "validation/enstatite-raw-workflow-v4.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    # Run directories are new; existing stage records cannot silently change.
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def log(message):
    print(message, flush=True)


class RawFrames:
    """Read-only one-frame HDF5 loader with explicit orientation and mask policy."""

    def __init__(self, path, acquisition, expected):
        import h5py
        import hdf5plugin  # noqa: F401 -- register compression filters

        self.file = h5py.File(path, "r")
        self.data = self.file[expected["dataset"]]
        if list(self.data.shape) != expected["shape"] or self.data.dtype != np.dtype("uint32"):
            self.file.close()
            raise ValueError("raw HDF5 shape/dtype differs from registered source")
        self.acquisition = acquisition
        self.indices = np.arange(
            acquisition["first_frame"], acquisition["last_frame_inclusive"] + 1
        )
        if not len(self.indices) or self.indices[-1] >= len(self.data):
            raise ValueError("frame selection outside source array")
        self.bounds = exposure_bounds(self.indices, acquisition)
        detector = self.data.parent
        self.pixel1 = float(detector["detector_information/pixel_size/ysize"][()])
        self.pixel2 = float(detector["detector_information/pixel_size/xsize"][()])
        self.pending_masks = {}
        self.hashes = {}
        self.loads = 0

    def load(self, index):
        image = self.data[int(self.indices[index])]
        if self.acquisition["vertical_flip"]:
            image = image[::-1]
        image = np.ascontiguousarray(image)
        mask = image >= self.acquisition["mask_minimum_reserved_uint32"]
        self.pending_masks[index] = mask
        self.hashes[index] = hashlib.sha256(image.tobytes()).hexdigest()
        self.loads += 1
        return image

    def mask(self, index):
        return self.pending_masks.pop(index)

    def close(self):
        self.file.close()


def exposure_bounds(indices, acquisition):
    start = (
        acquisition["omega_at_original_frame_zero_degrees"]
        + np.asarray(indices) * acquisition["step_degrees"]
    )
    return np.column_stack((start, start + acquisition["exposure_width_degrees"]))


def poni_geometry(path, pixel1, pixel2, axis):
    """Use the existing validated undistorted PONI-plane conversion."""
    from phasesmith.experimental.reciprocal import PanelGeometry

    values = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            values[key.strip().lower()] = value.strip()
    # The registered legacy PONI describes the already vertically reversed array.
    if values.get("detector", "").lower() != "eiger2cdte_4m":
        raise ValueError("example requires its explicitly registered affine detector calibration")
    return PanelGeometry.from_poni(
        distance=float(values["distance"]),
        poni1=float(values["poni1"]),
        poni2=float(values["poni2"]),
        pixel1=pixel1,
        pixel2=pixel2,
        rot1=float(values["rot1"]),
        rot2=float(values["rot2"]),
        rot3=float(values["rot3"]),
        wavelength=float(values["wavelength"]) * 1e10,
        axis=axis,
    )


def family_partition(hkl, rotations):
    """Whole supplied-symmetry/Friedel families; invariant to row/observation order."""
    h = np.asarray(hkl, dtype=np.int64)
    images = np.concatenate(
        [h @ r for r in np.asarray(rotations, dtype=np.int64)]
        + [-h @ r for r in np.asarray(rotations, dtype=np.int64)]
    )
    canonical = np.array([min(map(tuple, images[i :: len(h)])) for i in range(len(h))], np.int64)
    folds = np.array(
        [
            int(hashlib.sha256(np.asarray(row, "<i8").tobytes()).hexdigest()[:8], 16) % 5
            for row in canonical
        ]
    )
    _, groups = np.unique(canonical, axis=0, return_inverse=True)
    return canonical, groups, folds


def interior_target_rows(centres, flags, shape):
    """Bridge a flagged detector footprint to the scaling mesh's center domain.

    The predictor supports half-pixel edges; the mesh is defined on pixel
    centers. Only pre-existing EDGE rows may be omitted by this adapter.
    """
    from phasesmith.experimental.integration import IntegrationFlag

    xy = np.asarray(centres)[:, :2]
    height, width = shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] <= width - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= height - 1)
    if np.any((~inside) & ((np.asarray(flags) & int(IntegrationFlag.EDGE)) == 0)):
        raise ValueError("unflagged reflection center outside scaling mesh domain")
    return inside


def aligned_setting(candidates, cell, selection, training):
    """Select from training-supported cells by an explicit supplied metric prior."""
    from phasesmith.experimental.indexing import cell_parameters
    from phasesmith.experimental.orthorhombic import find_orthorhombic_settings

    target = np.asarray(cell[:3])
    choices = []
    for index, candidate in enumerate(candidates):
        if np.count_nonzero(candidate.indexed & training) < selection["minimum_indexed_training"]:
            continue
        settings = find_orthorhombic_settings(
            candidate.direct_basis,
            max_angle=selection["max_cell_angle_deviation_degrees"],
            coefficient_limit=2,
        )
        for setting in settings:
            if setting.centring != "P":
                continue
            for permutation in itertools.permutations(range(3)):
                a = setting.direct_basis[list(permutation)].copy()
                if np.linalg.det(a) < 0:
                    a[2] *= -1
                relative = np.linalg.norm(a, axis=1) / target - 1
                if np.max(abs(relative)) <= selection["cell_metric_relative_tolerance"]:
                    choices.append(
                        (
                            float(relative @ relative),
                            -int(np.count_nonzero(candidate.indexed & training)),
                            index,
                            a,
                        )
                    )
    if not choices:
        raise ValueError("no training-supported primitive setting matches the declared cell prior")
    choices.sort(key=lambda row: row[:3])
    _, _, index, a = choices[0]
    return index, a, cell_parameters(a)


def stage_positions(packet, out, protocol):
    from phasesmith.experimental.rotation import RotationOptions, locate_series
    from phasesmith.experimental.spots import SpotOptions

    frames = RawFrames(packet["raw_path"], protocol["acquisition"], packet["raw_array"])
    try:
        geometry = poni_geometry(
            packet["poni_path"], frames.pixel1, frames.pixel2, protocol["acquisition"]["axis"]
        )
        positions = locate_series(
            frames.load,
            frames.bounds,
            mask_loader=frames.mask,
            spot_options=SpotOptions(**protocol["detection"]),
            rotation_options=RotationOptions(**protocol["linking"]),
            saturation=None,
            workers=protocol["acquisition"]["workers"],
            gain=protocol["acquisition"]["gain"],
            read_noise=protocol["acquisition"]["read_noise"],
            progress=lambda n, total: (
                log(f"detect {n}/{total}") if n % 20 == 0 or n == total else None
            ),
        )
        np.savez_compressed(
            out / "positions.npz",
            **{
                name: getattr(positions, name)
                for name in (
                    "boxes",
                    "frame_ranges",
                    "values",
                    "flags",
                    "observations",
                    "omega_bounds",
                )
            },
        )
        hashes = [frames.hashes[i] for i in range(len(frames.indices))]
        save_json(
            out / "positions.json",
            dict(
                frame_hashes=positions.frame_hashes,
                raw_frame_hashes=hashes,
                original_frame_indices=frames.indices.tolist(),
                selected_frames=len(frames.indices),
                linked_tracks=len(positions),
                frame_detections=len(positions.observations),
                unflagged=int((positions.flags == 0).sum()),
                geometry_vectors=geometry.vectors.tolist(),
                wavelength_angstrom=geometry.wavelength,
                pixel_sizes_m=[frames.pixel1, frames.pixel2],
                source_loads=frames.loads,
            ),
        )
        return positions, geometry, hashes
    finally:
        frames.close()


def stage_indexing(positions, geometry, packet, out, protocol):
    from phasesmith.experimental.geometry_refinement import GeometryOptions, refine_geometry
    from phasesmith.experimental.grains import GrainSearchOptions, find_grains
    from phasesmith.experimental.prediction import predict_spots
    from phasesmith.experimental.reciprocal import correct_positions

    selection = protocol["selection"]
    reciprocal = correct_positions(positions, geometry)
    first, last = positions.frame_ranges.T + protocol["acquisition"]["first_frame"]
    block = selection["frame_block_size"]
    fold = (first // block) % selection["folds"]
    use = (
        (positions.flags == 0)
        & (positions.values[:, 4] <= selection["max_track_observations"])
        & (reciprocal.rms <= selection["max_q_rms"])
        & (((last - 1) // block) % selection["folds"] == fold)
    )
    ids = np.flatnonzero(use)
    training = fold[ids] != selection["test_fold"]
    q = reciprocal.points[ids]
    log(f"index {len(q)} clean tracks; {training.sum()} training")
    result = find_grains(
        q,
        training=training,
        options=GrainSearchOptions(**protocol["indexing"]),
        progress=lambda *values: log(f"index progress {values}"),
    )
    candidates = []
    for c in result.candidates:
        candidates.append(
            dict(
                cell=c.cell.tolist(),
                volume=c.volume,
                direct_basis=c.direct_basis.tolist(),
                accepted=bool(c.accepted),
                training_matches=int((c.indexed & training).sum()),
                validation_matches=int((c.indexed & ~training).sum()),
            )
        )
    save_json(
        out / "index-candidates.json", dict(candidates=candidates, source="new raw-frame positions")
    )
    chosen, direct, cell = aligned_setting(
        result.candidates, packet["priors"]["cell_angstrom_degrees"], selection, training
    )
    indices = np.rint(q @ direct.T).astype(np.int64)
    basis = np.linalg.inv(direct).T
    residual = np.linalg.norm(q - indices @ basis, axis=1)
    eligible = (
        result.candidates[chosen].indexed
        & (residual < result.options.reciprocal_tolerance)
        & np.any(indices != 0, axis=1)
    )
    # Keep discovery held-out blocks out of the calibration too. Whole families
    # do not teach geometry through their other scan observations.
    _, _, family_fold = family_partition(indices, packet["rotations"])
    fit_training = training & (family_fold != 0)
    observed = positions.values[ids, :3]
    fitted = refine_geometry(
        observed,
        indices,
        basis,
        geometry,
        tuple(packet["raw_array"]["shape"][1:]),
        training=fit_training,
        eligible=eligible,
        reference_cell=packet["priors"]["cell_angstrom_degrees"],
        options=GeometryOptions(**protocol["geometry"]),
    )
    predictions = predict_spots(indices, fitted.reciprocal_basis, fitted.geometry, observed[:, 2])
    error = predictions.values - observed
    valid = eligible & (predictions.flags == 0)
    heldout = valid & ~fit_training
    xy_rms = float(np.sqrt(np.mean(np.sum(error[heldout, :2] ** 2, axis=1))))
    omega_rms = float(np.sqrt(np.mean(error[heldout, 2] ** 2)))
    gates = dict(
        minimum_training=int((valid & fit_training).sum()) >= selection["minimum_indexed_training"],
        minimum_withheld=int(heldout.sum()) >= selection["minimum_indexed_validation"],
        withheld_xy=xy_rms <= protocol["gates"]["maximum_withheld_xy_rms_pixels"],
        withheld_omega=omega_rms <= protocol["gates"]["maximum_withheld_omega_rms_degrees"],
        optimizer_converged=bool(fitted.converged),
        not_at_bounds=not fitted.at_bounds,
    )
    save_json(
        out / "geometry.json",
        dict(
            chosen_candidate=chosen,
            initial_cell=cell.tolist(),
            reciprocal_basis=fitted.reciprocal_basis.tolist(),
            geometry_vectors=fitted.geometry.vectors.tolist(),
            wavelength_angstrom=fitted.geometry.wavelength,
            priors=packet["priors"],
            training=int((valid & fit_training).sum()),
            withheld=int(heldout.sum()),
            withheld_xy_rms_pixels=xy_rms,
            withheld_omega_rms_degrees=omega_rms,
            parameters=fitted.parameters.tolist(),
            condition=float(fitted.scaled_condition),
            gates=gates,
            reference_coordinates_used=False,
        ),
    )
    np.savez_compressed(
        out / "indexed-positions.npz",
        track_ids=ids,
        hkl=indices,
        observed=observed,
        predicted=predictions.values,
        eligible=valid,
        training=fit_training,
    )
    if not all(gates.values()):
        raise ValueError(f"positional calibration gates failed: {gates}")
    updated_q = correct_positions(positions, fitted.geometry).points[ids]
    competitors = []
    for index, candidate in enumerate(result.candidates):
        own = candidate.indexed
        if index == chosen or (own & eligible & training).sum() > 0.2 * max(
            1, (own & training).sum()
        ):
            continue
        fit = own & training
        test = own & ~training
        if fit.sum() < 30 or test.sum() < 10 or np.linalg.matrix_rank(candidate.hkl[fit]) < 3:
            continue
        b = np.linalg.lstsq(candidate.hkl[fit], updated_q[fit], rcond=None)[0]
        rms = float(
            np.sqrt(np.mean(np.sum((candidate.hkl[test] @ b - updated_q[test]) ** 2, axis=1)))
        )
        if rms <= result.options.reciprocal_tolerance:
            observed_only = own & ~eligible
            competitors.append(
                dict(
                    basis=b.tolist(),
                    training=int(fit.sum()),
                    withheld=int(test.sum()),
                    withheld_q_rms=rms,
                    discovery_candidate=index,
                    track_ids=ids[observed_only].tolist(),
                    observed_hkl=candidate.hkl[observed_only].tolist(),
                    observed_crossings=observed[observed_only].tolist(),
                )
            )
    competitors.sort(key=lambda row: -row["training"])
    competitors = competitors[: protocol["prediction"]["max_competing_domains"]]
    save_json(
        out / "competing-domains.json",
        dict(
            domains=competitors,
            interpretation=(
                "position-supported alternative orientations; "
                "no phase/physical-grain identification"
            ),
        ),
    )
    return fitted, competitors


def stage_integrate(packet, out, protocol, fitted, competitors, first_hashes):
    from phasesmith.experimental.integration import IntegrationOptions
    from phasesmith.experimental.prediction import ScanPredictions, predict_scan
    from phasesmith.experimental.profile_refinement import ShapeFitOptions
    from phasesmith.experimental.profiles import (
        ProfileOptions,
        learn_profiles,
        prepare_profiles,
        select_observations,
    )
    from phasesmith.experimental.reduction import (
        ReductionConfig,
        profile_geometry,
        reduce_observations,
    )
    from phasesmith.experimental.reduction_io import save_case
    from phasesmith.io.space_groups import space_group_by_number

    frames = RawFrames(packet["raw_path"], protocol["acquisition"], packet["raw_array"])
    try:
        shape = tuple(packet["raw_array"]["shape"][1:])
        p = predict_scan(
            fitted.reciprocal_basis,
            fitted.geometry,
            frames.bounds,
            shape,
            qmax=protocol["prediction"]["qmax_inverse_angstrom"],
        )
        _, _, zeta = profile_geometry(p.values[:, :2], fitted.geometry)
        group = space_group_by_number(packet["priors"]["space_group_number"]).space_group
        keep = (~group.systematic_absences(p.hkl)) & (
            abs(zeta) >= protocol["prediction"]["minimum_abs_zeta"]
        )
        p = ScanPredictions(p.hkl[keep], p.values[keep], p.frame_indices[keep], p.qmax)
        target_count = len(p.hkl)
        parts = [p]
        grain_ids = [np.zeros(target_count, np.int64)]
        for index, competing in enumerate(competitors, start=1):
            # Provisional competing cells do not establish a complete reflection
            # lattice. Only their independently measured spot tracks exclude
            # background; no speculative dense supercell reflection list is made.
            crossings = np.asarray(competing["observed_crossings"], float).reshape(-1, 3)
            if not len(crossings):
                continue
            other = ScanPredictions(
                np.asarray(competing["observed_hkl"], np.int64),
                crossings,
                np.clip(
                    np.searchsorted(frames.bounds[:, 1], crossings[:, 2]), 0, len(frames.bounds) - 1
                ),
                p.qmax,
            )
            _, _, other_zeta = profile_geometry(other.values[:, :2], fitted.geometry)
            other_keep = abs(other_zeta) >= protocol["prediction"]["minimum_abs_zeta"]
            other = ScanPredictions(
                other.hkl[other_keep],
                other.values[other_keep],
                other.frame_indices[other_keep],
                other.qmax,
            )
            parts.append(other)
            grain_ids.append(np.full(len(other.hkl), index, np.int64))
        p = ScanPredictions(
            np.concatenate([item.hkl for item in parts]),
            np.concatenate([item.values for item in parts]),
            np.concatenate([item.frame_indices for item in parts]),
            p.qmax,
        )
        _, planar, zeta = profile_geometry(p.values[:, :2], fitted.geometry)
        half_widths = np.minimum(4.5, 0.9 / abs(zeta) + 0.5)
        observations = prepare_profiles(
            frames.load,
            frames.bounds,
            p,
            mask_loader=frames.mask,
            grain_ids=np.concatenate(grain_ids),
            options=IntegrationOptions(**protocol["integration"]),
            saturation=None,
            scales=np.column_stack((np.ones((len(zeta), 2)), zeta)),
            planar_transforms=planar,
            angular_half_widths=half_widths,
            progress=lambda n, total: (
                log(f"integrate {n}/{total}") if n % 20 == 0 or n == total else None
            ),
        )
        hashes = [frames.hashes[i] for i in range(len(frames.indices))]
        if hashes != first_hashes:
            raise ValueError("source frame bytes changed between detection and integration")
        # All predicted domains have already excluded each other's backgrounds
        # and flagged overlapping apertures. Reduce only the target domain.
        interior = interior_target_rows(
            p.values[:target_count], observations.summation.flags[:target_count], shape
        )
        outside_ids = np.flatnonzero(~interior)
        np.savez_compressed(
            out / "edge-exclusions.npz",
            hkl=p.hkl[outside_ids],
            predicted=p.values[outside_ids],
            aperture=observations.summation.values[outside_ids],
            flags=observations.summation.flags[outside_ids],
        )
        observations = select_observations(observations, np.flatnonzero(interior))
        target_hkl = observations.summation.hkl
        _, _, folds = family_partition(target_hkl, packet["rotations"])
        refs = folds != 0
        options = ProfileOptions(**protocol["profiles"])
        bank = learn_profiles(observations, refs, detector_shape=shape, options=options)
        if len(bank.reference_ids) < protocol["gates"]["minimum_profile_teachers"]:
            raise ValueError("insufficient independent training profile teachers")
        # Large shoebox arrays are written only to the explicitly external run directory.
        save_case(out / "raw-observations-and-bank.npz", observations, bank)
        r = protocol["reduction"]
        config = ReductionConfig(
            profile=options,
            profile_coordinates="scattering",
            strong_policy=r["strong_policy"],
            strong_snr=r["strong_snr"],
            shape_fit=ShapeFitOptions(
                workers=r["shape_fit_workers"], maximum_iterations=r["shape_fit_maximum_iterations"]
            ),
            polarization_axis=tuple(r["polarization_axis"]),
            polarization_fraction=r["polarization_fraction"],
            equivalent_error_floor=r["equivalent_error_floor"],
            empirical=dict(
                equivalence_operations=packet["rotations"],
                symmetry_provenance="Explicit supplied Pbca symmetry; no atomic model",
                detector_shape=list(shape),
                grid=r["empirical_grid"],
                options=dict(strength=r["empirical_strength"]),
            ),
        )
        log(
            f"reduce {target_count} target reflections; {len(bank.reference_ids)} training teachers"
        )
        reduced = reduce_observations(
            observations,
            shape,
            config,
            geometry=fitted.geometry,
            model=bank,
            reference_mask=refs,
            progress=lambda n, total: (
                log(f"reduce {n}/{total}") if n % 500 == 0 or n == total else None
            ),
        )
        values = reduced.intensity_sigma
        valid = (reduced.profile.flags == 0) & np.isfinite(values).all(axis=1) & (values[:, 1] > 0)
        if valid.sum() < protocol["gates"]["minimum_exported_observations"]:
            raise ValueError("insufficient quality-controlled signed observations")
        np.savez_compressed(
            out / "reduced-observations.npz",
            hkl=target_hkl,
            intensity=values[:, 0],
            sigma=values[:, 1],
            statistical_sigma=reduced.statistical_sigma,
            flags=reduced.profile.flags,
            use=valid,
            aperture=observations.summation.values,
            raw_profiles=reduced.profile.values,
            choices=reduced.choices,
            parameters=reduced.parameters,
        )
        save_json(
            out / "reduction.json",
            dict(
                predictions=target_count,
                reduction_predictions=len(target_hkl),
                flagged_edge_centres_excluded=len(outside_ids),
                all_domain_predictions=len(p.hkl),
                competing_domains=len(competitors),
                used=int(valid.sum()),
                negative_used=int((values[valid, 0] < 0).sum()),
                teachers=len(bank.reference_ids),
                frame_hashes_equal=True,
                source_loads=frames.loads,
                excluded_by_quality_flags=int((~valid).sum()),
                metadata=reduced.metadata,
            ),
        )
        return target_hkl[valid].astype(np.int64), values[valid]
    finally:
        frames.close()


def run_search(packet, out, protocol, hkl, values):
    from phasesmith.crystallography import UnitCell
    from phasesmith.experimental.solution import (
        SolutionOptions,
        estimate_solution_memory,
        solve_structure_elements,
        write_solution,
    )
    from phasesmith.io.hkl import HklData
    from phasesmith.io.space_groups import space_group_by_number

    data = HklData(hkl, values[:, 0], values[:, 1])
    cell = UnitCell(*packet["priors"]["cell_angstrom_degrees"])
    group = space_group_by_number(packet["priors"]["space_group_number"]).space_group
    search = protocol["search"]
    estimate = estimate_solution_memory(
        data,
        cell=cell,
        space_group=group,
        dense_memory_budget_bytes=search["dense_memory_budget_bytes"],
    )
    frozen = json.loads((ROOT / "validation/solution-holdout-freeze-v1.json").read_text())
    allowed = inspect.signature(solve_structure_elements).parameters
    controls = {k: v for k, v in frozen["controls"].items() if k in allowed}
    controls.update(symmetry_recycling=True, tangent_refinement=True)
    options = SolutionOptions(
        phasing="dual_space",
        seed=search["seed"],
        trials=search["trials"],
        cycles=search["cycles"],
        max_seconds=search["native_seconds"],
        r1_target=0.06,
        completion=False,
        dense_memory_budget_bytes=search["dense_memory_budget_bytes"],
    )
    save_json(
        out / "search-input.json",
        dict(
            priors=packet["priors"],
            resource=asdict(estimate),
            options=asdict(options),
            controls=controls,
            data_source="fresh raw-frame reduction",
        ),
    )
    started = time.perf_counter()
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=packet["priors"]["elements"],
        maximum_atoms=estimate.maximum_atoms,
        options=options,
        **controls,
    )
    elapsed = time.perf_counter() - started
    write_solution(result, out / "candidate")
    save_json(
        out / "search.json",
        dict(
            termination=result.termination,
            target_met=result.target_met,
            amplitude_r1=result.best.r1 if result.best else None,
            seconds=elapsed,
            model_saved_before_reference=True,
            resource=asdict(estimate),
            observation_count=len(data.hkl),
            rejected_trials=result.rejected_trials,
        ),
    )


def validate_worker_input(packet):
    if set(packet) != {
        "raw_path",
        "poni_path",
        "raw_array",
        "priors",
        "rotations",
        "protocol",
        "output",
        "through",
    }:
        raise ValueError("worker input has unexpected/reference fields")
    if set(packet["priors"]) != {
        "cell_angstrom_degrees",
        "space_group_number",
        "space_group_symbol",
        "wavelength_angstrom",
        "elements",
        "provenance",
    }:
        raise ValueError("worker priors have unexpected/reference fields")
    return packet


def numerical_worker(packet_path):
    packet = validate_worker_input(json.loads(Path(packet_path).read_text()))
    out = Path(packet["output"])
    protocol = packet["protocol"]
    started = time.perf_counter()
    positions, geometry, hashes = stage_positions(packet, out, protocol)
    fitted, competitors = stage_indexing(positions, geometry, packet, out, protocol)
    if packet["through"] != "geometry":
        stage_integrate(packet, out, protocol, fitted, competitors, hashes)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--search-only", str(packet_path)],
            timeout=protocol["search"]["hard_seconds"],
            check=True,
        )
    save_json(
        out / "worker-complete.json",
        dict(
            stage=packet["through"],
            seconds=time.perf_counter() - started,
            reference_model_read=False,
            source_hkls_read=False,
            archived_arrays_read=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--through", choices=("geometry", "search"), default="search")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--search-only", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.search_only:
        packet = validate_worker_input(json.loads(args.search_only.read_text()))
        out = Path(packet["output"])
        with np.load(out / "reduced-observations.npz", allow_pickle=False) as data:
            use = data["use"]
            run_search(
                packet,
                out,
                packet["protocol"],
                data["hkl"][use],
                np.column_stack((data["intensity"][use], data["sigma"][use])),
            )
        return 0
    if args.worker:
        numerical_worker(args.worker)
        return 0
    if args.data_root is None or args.output is None:
        parser.error("--data-root and --output are required")
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        parser.error("large workflow products must be outside the repository checkout")
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads((ROOT / protocol["dataset_manifest"]).read_text())
    verify_dataset(manifest, args.data_root)
    from phasesmith import _core
    from phasesmith.io.space_groups import space_group_by_number

    out.mkdir(parents=True, exist_ok=False)
    group = space_group_by_number(manifest["search_inputs"]["space_group_number"]).space_group
    packet = dict(
        raw_path=str((args.data_root / "eiger_0000.h5").resolve()),
        poni_path=str((args.data_root / "CeO2_20240620_1849_dioptas.poni").resolve()),
        raw_array=manifest["raw_array"],
        priors=manifest["search_inputs"],
        rotations=[np.array(op.rotation).tolist() for op in group.operations],
        protocol=protocol,
        output=str(out),
        through=args.through,
    )
    save_json(out / "worker-input.json", packet)
    save_json(
        out / "provenance.json",
        dict(
            protocol_sha256=digest(PROTOCOL),
            source_hashes={r["path"]: r["sha256"] for r in manifest["files"]},
            native_sha256=digest(_core.__file__),
            implementation_sha256=digest(__file__),
            git_commit=subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
        ),
    )
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"
    )
    with (out / "worker.log").open("x") as log_stream:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(out / "worker-input.json"),
                ],
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
    verify_dataset(manifest, args.data_root)
    report = dict(
        schema="phasesmith.enstatite-raw-workflow-result.v1",
        execution_returncode=returncode,
        source_files_unchanged=True,
        through=args.through,
        raw_to_structure_search_passed=False,
        reference_assessment="not_executed",
        standalone_structure_refinement="not_implemented",
    )
    if returncode == 0 and args.through == "search":
        import gemmi
        from solution_corpus.audit import match_geometry
        from solution_corpus.data import atom_block, expanded_sites

        saved = json.loads((out / "candidate/result.json").read_text())
        if saved["candidates"]:
            model = dict(saved["candidates"][0], cell=saved["cell"])
            labels, xyz = expanded_sites(
                atom_block(args.data_root / "dom1_v9.cif"),
                gemmi.find_spacegroup_by_name(manifest["search_inputs"]["space_group_symbol"]),
            )
            assessment = match_geometry(model, labels, xyz, saved["cell"])
            assessment["reference_read_after_worker_stopped"] = True
            report["reference_assessment"] = assessment
            report["raw_to_structure_search_passed"] = assessment["correct"]
        else:
            report["reference_assessment"] = dict(correct=False, reason="no candidate")
    save_json(out / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
