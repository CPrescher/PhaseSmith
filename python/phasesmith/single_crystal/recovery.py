"""Bounded FFT-seeded recovery; every input observation is global training.

Two measured Fourier directions condition a third-direction search. Local
fixed-HKL hypotheses fit coherent corrections without supplied cells or groups.
These are proposals; count/rank/null/held-out acceptance remains elsewhere.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np

__all__ = ["GrainRecoveryOptions"]


@dataclass(frozen=True)
class GrainRecoveryOptions:
    max_pairs: int = 64
    split_trials: int = 4096
    max_split_parents: int = 8
    paired_search: bool = True

    def __post_init__(self):
        if type(self.paired_search) is not bool:
            raise ValueError("paired_search must be boolean")
        for name, maximum in [("max_pairs", 128), ("split_trials", 8192), ("max_split_parents", 8)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1,{maximum}]")


def _paired_choices(engine, q, vectors, options, subset, j, i, ids):
    from .grains import _refine, _score, _valid_basis, _vectors

    block = q[ids]
    actual = subset
    step = min(2 * np.max(abs(block)) / (actual.grid_size - 4), 1 / (2.1 * actual.max_length))
    if np.max(abs(block)) / step >= actual.grid_size / 2 - 1:
        actual = replace(actual, grid_size=256)
    third_vectors, _, _ = _vectors(engine, block, actual, diverse=True)
    choices = []
    for third in third_vectors[: actual.cell_refinements]:
        rows = np.array([vectors[j], vectors[i], third])
        if not _valid_basis(rows, actual):
            continue
        basis = _refine(engine, block, rows, actual, integer_cores=(8, 12, 16, 20, 24, 48, 96))
        if basis is None or not _valid_basis(basis, options):
            continue
        _, error, keep = _score(engine, q, basis, actual.reciprocal_tolerance)
        count = int(keep.sum())
        if count < 12:
            continue
        volume = abs(float(np.linalg.det(basis)))
        excess = count - len(q) * volume * 4 * np.pi / 3 * actual.reciprocal_tolerance**3
        rms = float(np.sqrt(np.mean(error[keep] ** 2)))
        choices.append((-excess, rms, volume, basis))
    return choices


def paired_proposals(engine, q, vectors, options, recovery, progress=None):
    from .grains import _same_oriented_basis

    phase = q @ vectors.T
    masks = abs(phase - np.rint(phase)) < 0.08
    populations = masks.sum(axis=0)
    ranking = []
    for i in range(len(vectors)):
        for j in range(i):
            sine = (
                np.linalg.norm(np.cross(vectors[i], vectors[j]))
                / np.linalg.norm(vectors[i])
                / np.linalg.norm(vectors[j])
            )
            if sine < 0.2:
                continue
            ids = np.flatnonzero(masks[:, i] & masks[:, j])
            expected = populations[i] * populations[j] / len(q)
            excess = (len(ids) - expected) / np.sqrt(max(expected, 1))
            if len(ids) >= 20:
                ranking.append((-float(excess), j, i, ids))
    ranking.sort(key=lambda x: x[:3])
    subset = replace(
        options,
        max_length=options.max_length / 2,
        grid_size=128,
        fft_peaks=400,
        basis_vectors=80,
        max_pairs=128,
        cell_refinements=80,
        min_indexed=12,
        reciprocal_tolerance=options.reciprocal_tolerance / 5,
        max_grains=1,
    )
    from . import search_execution

    jobs = [
        (q, vectors, options, subset, j, i, ids) for _, j, i, ids in ranking[: recovery.max_pairs]
    ]
    session = search_execution.active()
    results = (
        session.map("paired_seed", jobs)
        if session is not None
        else (_paired_choices(engine, *job) for job in jobs)
    )
    additions, audit = [], []
    for rank, (negative_score, j, i, ids) in enumerate(ranking[: recovery.max_pairs]):
        if progress:
            progress()
        if len(additions) >= 80:
            break
        choices = next(results)
        local = []
        for choice in sorted(choices, key=lambda x: x[:3]):
            basis = choice[-1]
            if any(_same_oriented_basis(basis, old) for old in additions):
                continue
            if len(local) >= 3 or len(additions) >= 80:
                break
            additions.append(basis)
            local.append(
                dict(basis=basis.tolist(), core_excess=-choice[0], rms=choice[1], volume=choice[2])
            )
        audit.append(
            dict(
                rank=rank,
                seeds=[j, i],
                score=-negative_score,
                training_rows=ids.tolist(),
                candidates=local,
            )
        )
    return additions, dict(
        options=asdict(recovery),
        ranking_total=len(ranking),
        pairs=audit,
        proposed=len(additions),
        seed_vectors=vectors.tolist(),
    )


def _coherent_parent(
    engine,
    q,
    observed,
    geometry,
    basis,
    h,
    ids,
    charts,
    valid,
    options,
    ownership,
    tolerance,
    samples,
):
    from .grains import _refine, _same_oriented_basis, _score, _valid_basis
    from .ownership import _physical_fields, _physical_measure

    choices = []
    measure = _physical_measure(charts, valid, ownership)
    for trial, sample in enumerate(samples):
        hh = h[sample]
        if abs(np.linalg.det(hh)) < 0.5:
            continue
        try:
            reciprocal = np.linalg.solve(hh, q[sample])
            proposal = np.linalg.inv(reciprocal).T
        except np.linalg.LinAlgError:
            continue
        if not _valid_basis(proposal, options):
            continue
        change = np.linalg.solve(np.linalg.inv(basis).T, reciprocal)
        u, singular, vt = np.linalg.svd(change)
        rotation = u @ vt
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        if np.linalg.det(rotation) <= 0 or angle > 5 or np.max(abs(singular - 1)) > 0.05:
            continue
        if _score(engine, q[ids], proposal, tolerance)[2].sum() < 12:
            continue
        proposed = _refine(
            engine,
            q[ids],
            proposal,
            replace(options, reciprocal_tolerance=tolerance, min_indexed=12, max_grains=1),
            integer_cores=(8, 12, 16, 20, 24, 48, 96),
        )
        if proposed is None or not _valid_basis(proposed, options):
            continue
        if _score(engine, q, proposed, tolerance)[2].sum() < 12:
            continue
        ee, _, physical = _physical_fields(
            engine, q, [proposed], charts, valid, options, ownership, _measure=measure
        )
        if int(physical.sum()) < 17 or any(
            _same_oriented_basis(proposed, old[-1]) for old in choices
        ):
            continue
        choices.append((-float(ee.sum()), abs(float(np.linalg.det(proposed))), trial, proposed))
    return choices


def _parallel_coherent(engine, q, observed, geometry, parents, options, recovery, progress):
    from . import fitting_tasks
    from .grains import _same_oriented_basis, _score
    from .ownership import GrainOwnershipOptions, _physical_fields, observation_charts
    from .prediction import predict_spots

    ownership = GrainOwnershipOptions()
    charts, valid = observation_charts(observed, geometry, ownership)
    evidence, _, _ = _physical_fields(engine, q, parents, charts, valid, options, ownership)
    order = sorted(range(len(parents)), key=lambda i: (-float(evidence[i].sum()), i))[
        : recovery.max_split_parents
    ]
    rng = np.random.default_rng(options.seed + 6)
    tolerance = options.reciprocal_tolerance / 4
    entries, jobs = [], []
    for parent_id in order:
        if progress:
            progress()
        basis = parents[parent_id]
        h, _, near = _score(engine, q, basis, 4 * options.reciprocal_tolerance)
        predicted = predict_spots(
            h.astype(np.int64), np.linalg.inv(basis).T, geometry, observed[:, 2]
        )
        residual = predicted.values - observed
        wide = near & (predicted.flags == 0) & np.isfinite(residual).all(axis=1)
        wide &= (np.linalg.norm(residual[:, :2], axis=1) < 50) & (abs(residual[:, 2]) < 5)
        ids = np.flatnonzero(wide)
        entry = dict(
            parent=parent_id,
            training_rows=ids.tolist(),
            attempted=recovery.split_trials,
            candidates=[],
        )
        entries.append(entry)
        if len(ids) < 17:
            continue
        samples = []
        for trial in range(recovery.split_trials):
            if progress and trial % 256 == 0:
                progress()
            samples.append(rng.choice(ids, 3, replace=False))
        jobs.append(
            (
                q,
                observed,
                geometry,
                basis.copy(),
                h,
                ids,
                charts,
                valid,
                options,
                ownership,
                tolerance,
                np.asarray(samples),
            )
        )
    results = fitting_tasks.active().map("coherent_parent", jobs)
    found, audit = [], []
    for entry in entries:
        if len(entry["training_rows"]) < 17:
            audit.append(entry)
            continue
        choices = next(results)
        entry["all_trials"] = [
            dict(trial=trial, basis=a.tolist(), volume=volume, evidence=-score)
            for score, volume, trial, a in sorted(choices, key=lambda x: x[:3])
        ]
        for score, volume, trial, proposed in sorted(choices, key=lambda x: x[:3])[:8]:
            if any(_same_oriented_basis(proposed, old) for old in found):
                continue
            found.append(proposed)
            entry["candidates"].append(
                dict(trial=trial, basis=proposed.tolist(), volume=volume, evidence=-score)
            )
        audit.append(entry)
    return found, dict(
        parents=order,
        trials=recovery.split_trials,
        seed=options.seed + 6,
        records=audit,
        proposed=len(found),
        scope="global training only",
    )


def coherent_proposals(engine, q, observed, geometry, parents, options, recovery, progress=None):
    from . import fitting_tasks
    from .grains import _refine, _same_oriented_basis, _score, _valid_basis
    from .ownership import (
        GrainOwnershipOptions,
        _physical_fields,
        _physical_measure,
        observation_charts,
    )
    from .prediction import predict_spots

    if fitting_tasks.active() is not None:
        return _parallel_coherent(
            engine, q, observed, geometry, parents, options, recovery, progress
        )
    ownership = GrainOwnershipOptions()
    charts, valid = observation_charts(observed, geometry, ownership)
    measure = _physical_measure(charts, valid, ownership)
    evidence, _, _ = _physical_fields(
        engine, q, parents, charts, valid, options, ownership, _measure=measure
    )
    order = sorted(range(len(parents)), key=lambda i: (-float(evidence[i].sum()), i))[
        : recovery.max_split_parents
    ]
    rng = np.random.default_rng(options.seed + 6)
    found, audit = [], []
    tolerance = options.reciprocal_tolerance / 4
    for parent_id in order:
        if progress:
            progress()
        basis = parents[parent_id]
        h, _, near = _score(engine, q, basis, 4 * options.reciprocal_tolerance)
        predicted = predict_spots(
            h.astype(np.int64), np.linalg.inv(basis).T, geometry, observed[:, 2]
        )
        residual = predicted.values - observed
        wide = near & (predicted.flags == 0) & np.isfinite(residual).all(axis=1)
        wide &= (np.linalg.norm(residual[:, :2], axis=1) < 50) & (abs(residual[:, 2]) < 5)
        ids = np.flatnonzero(wide)
        choices = []
        entry = dict(
            parent=parent_id,
            training_rows=ids.tolist(),
            attempted=recovery.split_trials,
            candidates=[],
        )
        if len(ids) < 17:
            audit.append(entry)
            continue
        for trial in range(recovery.split_trials):
            if progress and trial % 256 == 0:
                progress()
            sample = rng.choice(ids, 3, replace=False)
            hh = h[sample]
            if abs(np.linalg.det(hh)) < 0.5:
                continue
            try:
                reciprocal = np.linalg.solve(hh, q[sample])
                proposal = np.linalg.inv(reciprocal).T
            except np.linalg.LinAlgError:
                continue
            if not _valid_basis(proposal, options):
                continue
            change = np.linalg.solve(np.linalg.inv(basis).T, reciprocal)
            u, singular, vt = np.linalg.svd(change)
            rotation = u @ vt
            angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
            if np.linalg.det(rotation) <= 0 or angle > 5 or np.max(abs(singular - 1)) > 0.05:
                continue
            if _score(engine, q[ids], proposal, tolerance)[2].sum() < 12:
                continue
            proposed = _refine(
                engine,
                q[ids],
                proposal,
                replace(options, reciprocal_tolerance=tolerance, min_indexed=12, max_grains=1),
                integer_cores=(8, 12, 16, 20, 24, 48, 96),
            )
            if proposed is None or not _valid_basis(proposed, options):
                continue
            if _score(engine, q, proposed, tolerance)[2].sum() < 12:
                continue
            ee, _, physical = _physical_fields(
                engine, q, [proposed], charts, valid, options, ownership, _measure=measure
            )
            if int(physical.sum()) < 17 or any(
                _same_oriented_basis(proposed, old[-1]) for old in choices
            ):
                continue
            choices.append((-float(ee.sum()), abs(float(np.linalg.det(proposed))), trial, proposed))
        entry["all_trials"] = [
            dict(trial=trial, basis=a.tolist(), volume=volume, evidence=-score)
            for score, volume, trial, a in sorted(choices, key=lambda x: x[:3])
        ]
        for score, volume, trial, proposed in sorted(choices, key=lambda x: x[:3])[:8]:
            if any(_same_oriented_basis(proposed, old) for old in found):
                continue
            found.append(proposed)
            entry["candidates"].append(
                dict(trial=trial, basis=proposed.tolist(), volume=volume, evidence=-score)
            )
        audit.append(entry)
    return found, dict(
        parents=order,
        trials=recovery.split_trials,
        seed=options.seed + 6,
        records=audit,
        proposed=len(found),
        scope="global training only",
    )


def _forward_training_fit(engine, q, observed, geometry, options, basis):
    from .grains import _score, _valid_basis
    from .prediction import fit_crystal_proposal

    h, _, near = _score(engine, q, basis, options.reciprocal_tolerance)
    record = dict(initial_basis=basis.tolist(), status="failed")
    try:
        fit = fit_crystal_proposal(
            observed, h.astype(np.int64), np.linalg.inv(basis).T, geometry, eligible=near
        )
        trial = np.linalg.inv(fit.reciprocal_basis).T
        if _valid_basis(trial, options):
            basis = trial
            record.update(status="completed", history=fit.history.tolist(), converged=fit.converged)
    except (ValueError, np.linalg.LinAlgError) as ex:
        record["reason"] = str(ex)
    return basis, record


def fit_training_proposals(engine, q, observed, geometry, models, options, progress=None):
    """Fit frozen global-training proposals once, retaining every input order."""
    from . import fitting_tasks

    pending = fitting_tasks.batches(
        "forward",
        (q, observed, geometry, options),
        [(i, (a.copy(order="K"),)) for i, a in enumerate(models)],
    )
    fitted, records = [], []
    for i, a in enumerate(models):
        if progress:
            progress()
        basis, record = (
            _forward_training_fit(engine, q, observed, geometry, options, a)
            if pending is None
            else pending.get(i)
        )
        fitted.append(basis)
        records.append(record)
    return fitted, records


def finish_training_pool(
    engine, q, observed, geometry, models, charts, valid, options, ownership, progress=None
):
    """Fit once and remove training-ineligible competitors before validation.

    No selection/test data or acceptance outcomes enter this stage. The final
    forward fit is reused by downstream validation; fitting it again would change
    membership and be a different protocol.
    """
    from .ownership import _physical_fields, assign_observed

    fitted, fits = fit_training_proposals(engine, q, observed, geometry, models, options, progress)
    original = list(range(len(fitted)))
    removed = []
    while fitted:
        labels, _, _ = assign_observed(engine, q, fitted, charts, valid, options, ownership)
        evidence, _, _ = _physical_fields(engine, q, fitted, charts, valid, options, ownership)
        failures = []
        for j, basis in enumerate(fitted):
            own = labels == j + 1
            h = np.rint(q[own] @ basis.T)
            count = int(own.sum())
            unique = len(np.unique(h, axis=0))
            rank = int(np.linalg.matrix_rank(h)) if count else 0
            if count < options.min_indexed or unique < options.min_indexed or rank < 3:
                failures.append((count, unique, float(evidence[j].sum()), j, rank))
        if not failures:
            break
        count, unique, e, j, rank = min(failures)
        removed.append(
            dict(
                original_id=original[j],
                owned=count,
                unique_hkl=unique,
                rank=rank,
                evidence=e,
                basis=fitted[j].tolist(),
            )
        )
        fitted.pop(j)
        original.pop(j)
    alternatives = _subcell_alternatives(engine, q, fitted, charts, valid, options, ownership)
    for record in alternatives:
        record["child_original_id"] = original[record["child"]]
        record["parent_original_id"] = original[record["parent"]]
    deferred = {record["child"] for record in alternatives}
    fitted = [basis for j, basis in enumerate(fitted) if j not in deferred]
    original = [index for j, index in enumerate(original) if j not in deferred]
    labels, _, _ = assign_observed(engine, q, fitted, charts, valid, options, ownership)
    return (
        fitted,
        labels,
        dict(
            all_forward_fits=fits,
            retained_original_ids=original,
            forward_fits=[fits[j] for j in original],
            removed=removed,
            subcell_alternatives=alternatives,
            scope="global training only; final forward fit already completed",
        ),
    )


def _subcell_alternatives(engine, q, models, charts, valid, options, ownership):
    """Recognise a supported reflection subset as a cell alternative, not a grain.

    Applied only after every competing model passes the training support gates.
    A parent needs twice the child's physical support, while the child has fewer
    than twenty exclusive observations. Exact integer HKL maps and the existing
    geometric equivalence bounds protect independent orientations. Alternatives
    remain exported; this does not prove crystallographic primitivity.
    """
    from .ownership import _physical_fields

    _, _, physical = _physical_fields(engine, q, models, charts, valid, options, ownership)
    counts = physical.sum(axis=1)
    h = [np.rint(q @ basis.T).astype(np.int64) for basis in models]
    alternatives = []
    for child, b in enumerate(models):
        if counts[child] < options.min_indexed:
            continue
        for parent, a in enumerate(models):
            if abs(np.linalg.det(a)) <= abs(np.linalg.det(b)) or counts[parent] < 2 * counts[child]:
                continue
            shared = physical[child] & physical[parent]
            exclusive = int((physical[child] & ~physical[parent]).sum())
            if shared.sum() < options.min_indexed or exclusive >= options.min_indexed:
                continue
            hc, hp = h[child][shared], h[parent][shared]
            if np.linalg.matrix_rank(hc) < 3:
                continue
            transform = np.rint(np.linalg.lstsq(hc, hp, rcond=None)[0]).astype(np.int64)
            index = abs(round(np.linalg.det(transform)))
            if not 2 <= index <= 12 or not np.array_equal(hc @ transform, hp):
                continue
            predicted = transform.T @ b
            coefficient = float(np.max(abs(a @ np.linalg.inv(b) - transform.T)))
            relative = float(np.linalg.norm(predicted - a) / np.linalg.norm(a))
            u, singular, vt = np.linalg.svd(np.linalg.solve(predicted, a))
            rotation = u @ vt
            angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
            stretch = float(np.max(abs(singular - 1)))
            if (
                coefficient >= 0.025
                or relative >= 0.02
                or np.linalg.det(rotation) <= 0
                or angle >= 2
                or stretch >= 0.02
            ):
                continue
            alternatives.append(
                dict(
                    child=child,
                    parent=parent,
                    index=index,
                    child_basis=b.tolist(),
                    parent_basis=a.tolist(),
                    hkl_transform=transform.tolist(),
                    child_physical=int(counts[child]),
                    parent_physical=int(counts[parent]),
                    shared=int(shared.sum()),
                    child_exclusive=exclusive,
                    coefficient_error=coefficient,
                    relative_cartesian_error=relative,
                    rotation_degrees=angle,
                    max_strain=stretch,
                )
            )
    return alternatives
