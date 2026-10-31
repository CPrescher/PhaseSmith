"""Cold FFT direction/triplet kernels and shared three-cutoff proposal scoring.

Run with the experimental-spots release extension and single-threaded BLAS.
This is a kernel benchmark, NOT an end-to-end validated grain-search timing.
"""

import argparse
import json
import platform
from time import perf_counter

import numpy as np
from phasesmith import _core
from phasesmith.single_crystal.grains import GrainSearchOptions, _fourier_peaks, _triplets
from phasesmith.single_crystal.reciprocal import lattice_fourier
from phasesmith.single_crystal.reference import grains_reference as reference


def uncached_serial(q, options):
    """Original fixed-20-iteration ascent, with all rejected trials repeated."""
    engine = _core.ExperimentalLatticeEngine(options.workers)
    n = options.grid_size
    step = min(2 * np.max(abs(q)) / (n - 4), 1 / (2.1 * options.max_length))
    field = engine.build_field(q, n, step)
    initial, _, _ = field.maxima(options.min_length, options.max_length, options.fft_peaks)
    vectors = initial.copy()
    radius = 1.5 / (n * step)
    for _ in range(20):
        current = lattice_fourier(q, vectors)
        h = -current[:, 4:].reshape(-1, 3, 3)
        h += np.maximum(0, 0.01 - np.linalg.eigvalsh(h)[:, 0])[:, None, None] * np.eye(3)
        delta = np.linalg.solve(h, current[:, 1:4, None])[:, :, 0]
        delta *= np.minimum(1, 0.3 / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
        improved = np.zeros(len(vectors), bool)
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = np.clip(vectors + scale * delta, initial - radius, initial + radius)
            better = (lattice_fourier(q, trial)[:, 0] > current[:, 0]) & ~improved
            vectors[better] = trial[better]
            improved |= better
        if not improved.any():
            break
    return vectors, lattice_fourier(q, vectors)[:, 0]


def run(repeats):
    rng = np.random.default_rng(842024)
    # 12000 tracks with two minority lattice signals and diffuse contamination.
    q = rng.uniform(-1.6, 1.6, (12000, 3))
    for start, a in [(0, np.diag([4.7, 6.3, 8.1])), (400, np.diag([5.2, 7.1, 9.3]))]:
        a = a @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = rng.integers(-5, 6, (400, 3))
        q[start : start + 400] = h @ np.linalg.inv(a).T + rng.normal(0, 0.0008, (400, 3))
    options = GrainSearchOptions(grid_size=256, fft_peaks=400, basis_vectors=80, max_length=15)
    output = dict(
        hardware=platform.machine(),
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        points=len(q),
        fft_peaks=400,
        workers=4,
        runs=[],
        timing_scope=(
            "Cold kernel calls after import, including native input snapshots; "
            "not final search/validation"
        ),
    )
    for repeat in range(repeats):
        start = perf_counter()
        old, old_scores = uncached_serial(q, options)
        old_seconds = perf_counter() - start
        engine = _core.ExperimentalLatticeEngine(4)
        start = perf_counter()
        new, scores, _ = _fourier_peaks(engine, q, options)
        new_seconds = perf_counter() - start
        np.testing.assert_array_equal(new, old)
        np.testing.assert_array_equal(scores, old_scores)
        order = np.argsort(-scores, kind="stable")[:80]
        v = new[order]
        start = perf_counter()
        old_records = reference.phase_triplets(q, v, options)
        triplets_old = perf_counter() - start
        start = perf_counter()
        records = _triplets(q, v, options, engine)
        triplets_new = perf_counter() - start
        assert records == old_records
        models = np.array([v[list(row[2])] for row in records[:512]])
        charts = np.broadcast_to(np.eye(3) * 1000, (len(q), 3, 3)).copy()
        start = perf_counter()
        counts = engine.candidate_counts(
            q,
            models,
            np.linalg.inv(models).transpose(0, 2, 1),
            charts,
            np.ones(len(q), bool),
            np.array([0.003, 0.008, 0.015]),
            3.0,
            0.5,
        )
        shared_seconds = perf_counter() - start
        output["runs"].append(
            dict(
                repeat=repeat,
                serial_uncached_seconds=old_seconds,
                cached_parallel_seconds=new_seconds,
                direction_speedup=old_seconds / new_seconds,
                original_triplets_seconds=triplets_old,
                packed_triplets_seconds=triplets_new,
                shared_three_cutoffs_seconds=shared_seconds,
                scored_models=len(counts),
                exact_vectors_scores_and_triplet_order=True,
            )
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.repeats)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
