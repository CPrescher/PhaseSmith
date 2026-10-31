"""Two-lattice detector-space benchmark for opt-in FFT proposal stages.

No cell/HKL hints are passed to the inverse. Runs are serialized and assert
repeatable bases/labels plus recovery of both oriented synthetic lattices.
"""

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from phasesmith.experimental.grains import (
    GrainAliasOptions,
    GrainConditioningOptions,
    GrainOwnershipOptions,
    GrainProposalRefinementOptions,
    GrainRecoveryOptions,
    GrainSearchOptions,
    GrainStabilityOptions,
    find_grains_ensemble,
)
from phasesmith.experimental.prediction import predict_spots
from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("conditioned", "combined", "expanded-alias", "ownership", "recovery"),
        default=("conditioned", "combined"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(643)
    geometry = PanelGeometry.from_poni(
        distance=0.2,
        poni1=0.08,
        poni2=0.08,
        pixel1=8e-5,
        pixel2=8e-5,
        rot1=0.01,
        rot2=-0.03,
        rot3=0.05,
        wavelength=0.4,
        axis=[-0.97, 0.12, 0.2],
    )
    observations, truths = [], []
    for edges in ([4.3, 5.1, 7.7], [5.2, 6.2, 8.3]):
        direct = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = np.unique(rng.integers(-5, 6, (1400, 3)), axis=0)
        h = h[np.any(h != 0, axis=1)]
        predicted = predict_spots(
            h, np.linalg.inv(direct).T, geometry, rng.uniform(-200, -30, len(h))
        )
        observed = predicted.values[predicted.flags == 0].copy()
        observed += rng.normal(size=observed.shape) * [0.12, 0.12, 0.025]
        observations.append(observed)
        truths.append(direct)
    noise = rng.uniform([100, 100, -200], [2000, 2000, -30], (600, 3))
    observed = np.concatenate([*observations, noise])
    q = reciprocal_vectors(observed, geometry)
    options = GrainSearchOptions(max_length=15, grid_size=128, max_grains=4, workers=2)
    searches = tuple(replace(options, reciprocal_tolerance=t) for t in (0.003, 0.015))
    records = []
    for mode in args.modes:
        previous = None
        times = []
        cpu = []
        for repeat in range(args.repeats):
            start, process = time.perf_counter(), time.process_time()
            extra = (
                {}
                if mode == "conditioned"
                else dict(
                    stability=GrainStabilityOptions(),
                    proposal_refinement=GrainProposalRefinementOptions(),
                    observed=observed,
                    geometry=geometry,
                )
            )
            if mode in ("expanded-alias", "ownership", "recovery"):
                extra.update(
                    aliases=GrainAliasOptions(),
                    sparse_conditioning=GrainConditioningOptions(
                        candidates_per_seed=3,
                        subset_search=replace(
                            options,
                            max_length=options.max_length / 2,
                            grid_size=256,
                            fft_peaks=400,
                            basis_vectors=80,
                            max_pairs=128,
                            cell_refinements=80,
                            min_indexed=12,
                            max_grains=1,
                            reciprocal_tolerance=0.003,
                        ),
                    ),
                )
            if mode in ("ownership", "recovery"):
                extra["ownership"] = GrainOwnershipOptions()
            if mode == "recovery":
                extra["ownership"] = GrainOwnershipOptions(competitive_selection=True)
                extra["recovery"] = GrainRecoveryOptions()
                extra["proposal_refinement"] = GrainProposalRefinementOptions(distinct_budget=True)
            result = find_grains_ensemble(
                q,
                options=options,
                searches=searches,
                conditioning=GrainConditioningOptions(),
                **extra,
            )
            times.append(time.perf_counter() - start)
            cpu.append(time.process_time() - process)
            accepted = [c for c in result.candidates if c.accepted]
            if len(accepted) != 2:
                failure = dict(
                    mode=mode,
                    expected=2,
                    actual=len(accepted),
                    wall_seconds=times[-1],
                    truths=[a.tolist() for a in truths],
                    candidates=[
                        dict(
                            basis=c.direct_basis.tolist(),
                            volume=c.volume,
                            accepted=c.accepted,
                            training_fraction=c.training_fraction,
                            validation_fraction=c.validation_fraction,
                        )
                        for c in result.candidates
                    ],
                )
                Path(args.output).with_suffix(".failure.json").write_text(
                    json.dumps(failure, indent=2) + "\n"
                )
            assert len(accepted) == 2
            for truth in truths:
                assert any(
                    np.max(
                        abs(
                            c.direct_basis @ np.linalg.inv(truth)
                            - np.rint(c.direct_basis @ np.linalg.inv(truth))
                        )
                    )
                    < 0.01
                    and abs(c.volume / abs(np.linalg.det(truth)) - 1) < 0.01
                    for c in accepted
                )
            bases = np.array([c.direct_basis for c in result.candidates])
            if previous is not None:
                np.testing.assert_array_equal(bases, previous[0])
                np.testing.assert_array_equal(result.labels, previous[1])
            previous = bases, result.labels
            print(mode, repeat, times[-1], flush=True)
        records.append(
            dict(
                mode=mode,
                wall_seconds=times,
                cpu_seconds=cpu,
                exact_repeat=True,
                accepted=len(accepted),
            )
        )
    Path(args.output).write_text(
        json.dumps(
            dict(
                positions=len(q),
                signal_positions=list(map(len, observations)),
                options=asdict(options),
                searches=[asdict(s) for s in searches],
                platform=platform.platform(),
                results=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
