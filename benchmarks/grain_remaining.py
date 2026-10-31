"""Remaining discovery kernels and an opt-in strict two-component control.

Kernel stages are timed separately from complete experimental inverses.
The before path optionally loads an explicitly saved first-party source/binary.
"""

import argparse
import ast
import importlib.util
import json
import platform
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental import grains as G
from phasesmith.experimental.reciprocal import lattice_fourier


def legacy(root):
    path = root / "source-before/python/phasesmith/experimental/grains.py"
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_fourier_evaluator", "_fourier_peaks", "_triplets"}
    ]
    namespace = dict(G.__dict__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    name = "_remaining_before"
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(
        name + "._core", root / "core-before-remaining.abi3.so"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return namespace, module.ExperimentalLatticeEngine(4)


def control():
    from phasesmith.experimental.prediction import predict_spots
    from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors

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
        a = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = np.unique(rng.integers(-5, 6, (1400, 3)), axis=0)
        h = h[np.any(h != 0, axis=1)]
        predicted = predict_spots(h, np.linalg.inv(a).T, geometry, rng.uniform(-200, -30, len(h)))
        observed = predicted.values[predicted.flags == 0].copy()
        observed += rng.normal(size=observed.shape) * [0.12, 0.12, 0.025]
        observations.append(observed)
        truths.append(a)
    observed = np.concatenate(
        [*observations, rng.uniform([100, 100, -200], [2000, 2000, -30], (600, 3))]
    )
    q = reciprocal_vectors(observed, geometry)
    records = []
    for native in (False, True):
        options = G.GrainSearchOptions(
            max_length=15,
            grid_size=128,
            max_grains=4,
            workers=2,
            native_fourier_refinement=native,
        )
        sparse = G.GrainConditioningOptions(
            candidates_per_seed=3,
            subset_search=replace(
                options,
                max_length=7.5,
                grid_size=256,
                fft_peaks=400,
                basis_vectors=80,
                max_pairs=128,
                cell_refinements=80,
                min_indexed=12,
                max_grains=1,
                reciprocal_tolerance=0.003,
            ),
        )
        start = time.perf_counter()
        result = G.find_grains_ensemble(
            q,
            options=options,
            searches=tuple(replace(options, reciprocal_tolerance=t) for t in (0.003, 0.015)),
            conditioning=G.GrainConditioningOptions(),
            stability=G.GrainStabilityOptions(),
            proposal_refinement=G.GrainProposalRefinementOptions(distinct_budget=True),
            aliases=G.GrainAliasOptions(),
            sparse_conditioning=sparse,
            ownership=G.GrainOwnershipOptions(competitive_selection=True),
            recovery=G.GrainRecoveryOptions(),
            observed=observed,
            geometry=geometry,
        )
        elapsed = time.perf_counter() - start
        accepted = [c for c in result.candidates if c.accepted]
        record = dict(
            native=native,
            seconds=elapsed,
            accepted=len(accepted),
            candidates=[
                dict(volume=c.volume, accepted=c.accepted, basis=c.direct_basis.tolist())
                for c in result.candidates
            ],
        )
        records.append(record)
        assert len(accepted) == 2, record
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
            ), record
    return dict(points=len(q), signal_points=list(map(len, observations)), runs=records)


def kernels(args):
    rng = np.random.default_rng(842028)
    q = rng.uniform(-1.6, 1.6, (12000, 3))
    a = np.diag([4.7, 6.3, 8.1]) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    h = rng.integers(-5, 6, (400, 3))
    q[:400] = h @ np.linalg.inv(a).T + rng.normal(0, 0.0008, (400, 3))
    phase = q @ a[0]
    subset = q[abs(phase - np.rint(phase)) < 0.08]
    namespace, before_engine = legacy(Path(args.legacy_dir))
    engine = _core.ExperimentalLatticeEngine(4)
    options = G.GrainSearchOptions(
        grid_size=256,
        max_length=15,
        fft_peaks=400,
        basis_vectors=80,
        max_pairs=128,
        cell_refinements=80,
    )
    vectors = rng.uniform(-8, 8, (80, 3))
    records = []
    for repeat in range(args.repeats):
        start = time.perf_counter()
        old = namespace["_triplets"](subset, vectors, options, before_engine, limit=80)
        before = time.perf_counter() - start
        start = time.perf_counter()
        new = G._triplets(subset, vectors, options, engine, limit=80)
        after = time.perf_counter() - start
        assert old == new
        records.append(
            dict(
                repeat=repeat,
                stage="triplets",
                before=before,
                after=after,
                exact_records=True,
                returned=len(new),
            )
        )
        bases = np.array([vectors[list(row[2])] for row in new])
        reciprocal = np.array([np.linalg.inv(b).T for b in bases])
        start = time.perf_counter()
        expected = []
        for b, r in zip(bases, reciprocal, strict=True):
            h, error = before_engine.score(subset, b, r)
            expected.append(int(((error < 0.006) & np.any(h != 0, axis=1)).sum()))
        before = time.perf_counter() - start
        start = time.perf_counter()
        counts = engine.initial_support(subset, bases, reciprocal, 0.006)
        after = time.perf_counter() - start
        np.testing.assert_array_equal(counts, expected)
        records.append(
            dict(
                repeat=repeat,
                stage="first-cycle support",
                before=before,
                after=after,
                models=len(bases),
                exact_counts=True,
                initial_rejects=int((counts < 6).sum()),
            )
        )
        start = time.perf_counter()
        old_v, old_s, _ = namespace["_fourier_peaks"](before_engine, subset, options)
        before = time.perf_counter() - start
        start = time.perf_counter()
        new_v, new_s, _ = G._fourier_peaks(
            engine, subset, replace(options, native_fourier_refinement=True)
        )
        after = time.perf_counter() - start
        np.testing.assert_array_equal(new_s, lattice_fourier(subset, new_v)[:, 0])
        records.append(
            dict(
                repeat=repeat,
                stage="Fourier maxima + refinement",
                before=before,
                after=after,
                max_vector_difference=float(np.max(abs(new_v - old_v))),
                max_score_difference=float(np.max(abs(new_s - old_s))),
                returned=len(new_v),
            )
        )
    return dict(
        points=len(q),
        subset=len(subset),
        workers=4,
        runs=records,
        scope="Kernel stages only, including snapshots; excludes inverse/fits/validation.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = dict(platform=platform.platform(), result=control() if args.control else kernels(args))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
