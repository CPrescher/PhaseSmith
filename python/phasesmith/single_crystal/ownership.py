"""Observation-aware ownership; chemistry-free and frozen before validation.

Reciprocal Jacobians and fused ownership run natively. This module owns fixed
measurement conventions and bounded training-only fitting/model selection.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from phasesmith import _core

from .reciprocal import PanelGeometry, mapped_xy, triples

__all__ = ["GrainOwnershipOptions"]


@dataclass(frozen=True)
class GrainOwnershipOptions:
    xy_scale: float = 2.0
    omega_scale: float = 0.3
    xy_gate: float = 3.0
    omega_gate: float = 0.5
    rank_rtol: float = 1e-12
    max_fits: int = 64
    refinement_cycles: int = 2
    alias_rounds: int = 1
    alias_fit_cycles: int = 8
    competitive_selection: bool = False
    preserve_input_models: bool = False
    native_fields: bool = False

    def __post_init__(self):
        if type(self.native_fields) is not bool:
            raise ValueError("native_fields must be boolean")
        if type(self.preserve_input_models) is not bool:
            raise ValueError("preserve_input_models must be boolean")
        if type(self.competitive_selection) is not bool:
            raise ValueError("competitive_selection must be boolean")
        for name in ("xy_scale", "omega_scale", "xy_gate", "omega_gate", "rank_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not self.rank_rtol < 1e-3:
            raise ValueError("rank_rtol must lie in (0,1e-3)")
        for name, lo, hi in [
            ("max_fits", 1, 128),
            ("refinement_cycles", 1, 8),
            ("alias_rounds", 0, 1),
            ("alias_fit_cycles", 1, 8),
        ]:
            v = getattr(self, name)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"{name} must be an integer in [{lo},{hi}]")


def reciprocal_with_jacobian(observed, geometry):
    """Fused q and dq/d(xpixel,ypixel,omegadeg), including an optional pixel map."""
    samples = triples(observed, "observed")
    if len(samples) > 100000 or not isinstance(geometry, PanelGeometry):
        raise ValueError("requires <=100000 observations and PanelGeometry")
    chain = None
    if geometry.pixel_map is not None:
        samples[:, :2] = mapped_xy(geometry.pixel_map, samples[:, :2])
        inverse = np.asarray(geometry.pixel_map.inverse_jacobian(samples[:, :2]))
        if inverse.shape != (len(samples), 2, 2) or not np.isfinite(inverse).all():
            raise ValueError("pixel inverse Jacobian must be finite (N,2,2)")
        chain = np.linalg.inv(inverse)
    if not hasattr(_core, "experimental_reciprocal_jacobian"):
        raise RuntimeError("ownership needs a rebuilt single-crystal extension")
    q, j = _core.experimental_reciprocal_jacobian(samples, geometry.vectors, geometry.wavelength)
    j = j.reshape(-1, 3, 3)
    if chain is not None:
        j[:, :, :2] = j[:, :, :2] @ chain
    return q, j


def observation_charts(observed, geometry, options=None):
    """Inverse observation Jacobians and a rank-validity mask, never model-fitted.

    Singular charts leave observations unresolved rather than fabricating a
    coordinate residual. Input populations are preserved. Under q'=qR the
    coherent chart transformation is K'=KR, also used for every null control.
    """
    options = GrainOwnershipOptions() if options is None else options
    if not isinstance(options, GrainOwnershipOptions):
        raise TypeError("options must be GrainOwnershipOptions")
    _, j = reciprocal_with_jacobian(observed, geometry)
    u, s, vt = np.linalg.svd(j)
    valid = s[:, -1] > options.rank_rtol * s[:, 0]
    inverse = np.zeros_like(s)
    np.divide(1, s, out=inverse, where=valid[:, None])
    chart = (vt.transpose(0, 2, 1) * inverse[:, None, :]) @ u.transpose(0, 2, 1)
    chart = np.ascontiguousarray(chart)
    chart.flags.writeable = False
    valid.flags.writeable = False
    return chart, valid


def assign_observed(engine, points, models, charts, valid, search, options):
    """Same q/HKL gates, with physical eligibility before stable ownership."""
    if not len(models):
        return np.zeros(len(points), np.int32), 0, 0
    rows = np.asarray(models, dtype=np.float64)
    parameters = np.array(
        [
            search.reciprocal_tolerance,
            search.ambiguity_ratio,
            options.xy_scale,
            options.omega_scale,
            options.xy_gate,
            options.omega_gate,
        ]
    )
    return engine.assign_observed(
        points,
        rows,
        np.linalg.inv(rows).transpose(0, 2, 1),
        np.asarray(charts),
        np.asarray(valid),
        parameters,
    )


def _physical_fields(engine, q, models, charts, valid, search, options):
    from .grains import _score

    if len(q) * len(models) > 8_000_000:
        raise ValueError("ownership pool exceeds 8 million position/model pairs")
    evidence = []
    core = []
    physical = []
    det = np.abs(np.linalg.det(charts))
    measure = np.zeros(len(q))
    np.divide(
        (2 * np.pi) ** 1.5 * options.xy_scale**2 * options.omega_scale,
        det,
        out=measure,
        where=valid,
    )
    if options.native_fields and len(models):
        rows = np.asarray(models, dtype=np.float64).reshape(-1, 3, 3)
        return engine.physical_fields(
            q,
            rows,
            np.linalg.inv(rows).transpose(0, 2, 1),
            abs(np.linalg.det(rows)),
            np.asarray(charts),
            np.asarray(valid),
            measure,
            np.array(
                [
                    search.reciprocal_tolerance,
                    options.xy_scale,
                    options.omega_scale,
                    options.xy_gate,
                    options.omega_gate,
                ]
            ),
        )
    for a in models:
        h, error, near = _score(engine, q, a, search.reciprocal_tolerance)
        delta = q - h @ np.linalg.inv(a).T
        r = np.einsum("nij,nj->ni", charts, delta)
        good = (
            valid
            & near
            & (np.linalg.norm(r[:, :2], axis=1) <= options.xy_gate)
            & (abs(r[:, 2]) <= options.omega_gate)
        )
        d2 = (
            np.sum((r[:, :2] / options.xy_scale) ** 2, axis=1)
            + (r[:, 2] / options.omega_scale) ** 2
        )
        e = np.zeros(len(q))
        normalisation = abs(np.linalg.det(a)) * measure[good]
        e[good] = np.logaddexp(0, -0.5 * d2[good] - np.log(normalisation))
        evidence.append(e)
        core.append((error < search.reciprocal_tolerance / 4) & near)
        physical.append(good)
    shape = (len(models), len(q))
    return (
        np.asarray(evidence).reshape(shape),
        np.asarray(core).reshape(shape),
        np.asarray(physical).reshape(shape),
    )


def _fit_one(
    engine,
    q,
    observed,
    geometry,
    a,
    search,
    options,
    seed,
    allowed=None,
    all_training=False,
    charts=None,
    valid=None,
    background=None,
):
    if all_training:
        return _fit_gls(
            engine, q, observed, geometry, a, search, options, allowed, charts, valid, background
        )
    from .grains import _score, _valid_basis
    from .prediction import RefinementOptions, predict_spots, refine_crystal

    h, _, near = _score(engine, q, a, search.reciprocal_tolerance)
    pred = predict_spots(h.astype(np.int64), np.linalg.inv(a).T, geometry, observed[:, 2])
    delta = pred.values - observed
    keep = near & (pred.flags == 0) & np.isfinite(delta).all(axis=1)
    keep &= (np.linalg.norm(delta[:, :2], axis=1) <= options.xy_gate) & (
        abs(delta[:, 2]) <= options.omega_gate
    )
    if allowed is not None:
        keep &= allowed
    ids = np.flatnonzero(keep)
    record = dict(training_rows=ids.tolist(), status="insufficient_training")
    if len(ids) < 17 or np.linalg.matrix_rank(h[ids]) < 3:
        return a, record
    train = np.ones(len(ids), bool)
    if not all_training:
        train[np.random.default_rng(seed).permutation(len(ids))[:5]] = False
    record["internal_training"] = train.tolist()
    record["scope"] = (
        "all global training; external validation reserved"
        if all_training
        else "internal training diagnostic"
    )
    try:
        fit = refine_crystal(
            observed[ids],
            h[ids].astype(np.int64),
            np.linalg.inv(a).T,
            geometry,
            training=train,
            options=RefinementOptions(xy_scale=options.xy_scale, omega_scale=options.omega_scale),
        )
        trial = np.linalg.inv(fit.reciprocal_basis).T
        if _valid_basis(trial, search):
            record.update(status="completed", converged=fit.converged, history=fit.history.tolist())
            return trial, record
        record["status"] = "invalid_refined_basis"
    except (ValueError, np.linalg.LinAlgError) as ex:
        record.update(status="fit_failure", reason=str(ex))
    return a, record


def _fit_gls(
    engine,
    q,
    observed,
    geometry,
    a,
    search,
    options,
    allowed=None,
    charts=None,
    valid=None,
    background=None,
):
    """Unvalidated training-only GLS proposal fit in ownership coordinates.

    The native pass fuses robust cost, gradient and normal accumulation. This
    uses original mean reciprocal tracks rather than a separately averaged
    detector centroid. External selection/test observations never enter fitting.
    """
    from .grains import _score, _valid_basis

    if charts is None:
        charts, valid = observation_charts(observed, geometry, options)
    h, _, near = _score(engine, q, a, search.reciprocal_tolerance)
    b = np.linalg.inv(a).T
    residual = np.einsum("nij,nj->ni", charts, q - h @ b)
    keep = (
        near
        & valid
        & (np.linalg.norm(residual[:, :2], axis=1) <= options.xy_gate)
        & (abs(residual[:, 2]) <= options.omega_gate)
    )
    if allowed is not None:
        keep &= allowed
    ids = np.flatnonzero(keep)
    record = dict(
        training_rows=ids.tolist(),
        internal_training=[True] * len(ids),
        scope="all global training; external validation reserved",
        optimizer="observation_gls",
        status="insufficient_training",
    )
    if len(ids) < 12 or np.linalg.matrix_rank(h[ids]) < 3:
        return a, record
    scales = np.array([options.xy_scale, options.xy_scale, options.omega_scale])
    enabled = np.ones(len(ids), bool)

    def evaluate(basis):
        return engine.observation_normal(
            q[ids],
            h[ids],
            charts[ids],
            enabled,
            basis,
            scales,
            2.5,
            None if background is None else background[ids],
        )

    try:
        normal, gradient, cost, _ = evaluate(b)
        history = [cost]
        damping = 1e-3
        converged = False
        b0 = b.copy()
        for _ in range(40):
            diagonal = np.sqrt(np.maximum(np.diag(normal), 1e-12))
            scaled = normal / diagonal[:, None] / diagonal[None, :]
            if np.linalg.matrix_rank(scaled) < 9:
                record["status"] = "rank_deficient"
                return a, record
            step = np.linalg.solve(scaled + damping * np.eye(9), -gradient / diagonal) / diagonal
            trial = b + step.reshape(3, 3)
            candidate = np.linalg.inv(trial).T
            if (
                np.linalg.det(trial) * np.linalg.det(b0) <= 0
                or np.linalg.norm(trial - b0) > 0.2 * np.linalg.norm(b0)
                or not _valid_basis(candidate, search)
            ):
                damping *= 10
                continue
            nn, gg, cc, _ = evaluate(trial)
            if cc < cost:
                delta = cost - cc
                b, normal, gradient, cost = trial, nn, gg, cc
                history.append(cost)
                damping = max(damping / 3, 1e-12)
                if np.linalg.norm(step) < 1e-10 or delta < 1e-10 * max(cost, 1):
                    converged = True
                    break
            else:
                if np.linalg.norm(step) < 1e-10:
                    converged = True
                    break
                damping *= 10
            if damping > 1e14:
                break
        record.update(status="completed", converged=converged, history=history)
        return np.linalg.inv(b).T, record
    except (ValueError, np.linalg.LinAlgError) as ex:
        record.update(status="fit_failure", reason=str(ex))
        return a, record


def _select(engine, q, models, charts, valid, search, options):
    evidence, core, physical = _physical_fields(engine, q, models, charts, valid, search, options)
    # The scalar determinant convention is unchanged; each invariant volume
    # is evaluated once instead of repeatedly inside model-pair competition.
    volumes = [abs(float(np.linalg.det(a))) for a in models]
    counts = core.sum(axis=1)
    phys = physical.sum(axis=1)
    # Compare nested indexing hypotheses on their common *training* signal.
    # A denser reciprocal lattice cannot become an independent component just
    # by winning extra, geometrically inconsistent q matches. HKL maps below
    # are exact integer equalities on rank-three shared observations.
    hkl = [np.rint(q @ a.T).astype(np.int64) for a in models]
    preferred = {}
    family = []
    for parent, a in enumerate(models):
        if counts[parent] < search.min_indexed or phys[parent] < search.min_indexed:
            continue
        for child, b in enumerate(models):
            if (
                volumes[child] >= volumes[parent]
                or counts[child] < search.min_indexed
                or phys[child] < search.min_indexed
            ):
                continue
            shared = physical[parent] & physical[child]
            extra = int((physical[parent] & ~physical[child]).sum())
            if shared.sum() < search.min_indexed or extra >= search.min_indexed:
                continue
            hc = hkl[child][shared]
            hp = hkl[parent][shared]
            if np.linalg.matrix_rank(hc) < 3:
                continue
            transform = np.rint(np.linalg.lstsq(hc, hp, rcond=None)[0]).astype(np.int64)
            index = abs(round(np.linalg.det(transform)))
            if not 2 <= index <= 12 or not np.array_equal(hc @ transform, hp):
                continue
            predicted = transform.T @ b
            coefficient = float(np.max(abs(a @ np.linalg.inv(b) - transform.T)))
            relative = float(np.linalg.norm(predicted - a) / np.linalg.norm(a))
            u, s, vt = np.linalg.svd(np.linalg.solve(predicted, a))
            rotation = u @ vt
            angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
            strain = float(np.max(abs(s - 1)))
            if (
                coefficient >= 0.025
                or relative >= 0.02
                or np.linalg.det(rotation) <= 0
                or angle >= 2
                or strain >= 0.02
            ):
                continue
            child_evidence = float(evidence[child, shared].sum())
            parent_evidence = float(evidence[parent, shared].sum())
            if child_evidence <= parent_evidence:
                continue
            family.append(
                dict(
                    parent_pool_id=parent,
                    child_pool_id=child,
                    index=index,
                    hkl_transform=transform.tolist(),
                    shared_physical=int(shared.sum()),
                    parent_additional_physical=extra,
                    child_common_evidence=child_evidence,
                    parent_common_evidence=parent_evidence,
                    coefficient_error=coefficient,
                    relative_cartesian_error=relative,
                    rotation_degrees=angle,
                    max_strain=strain,
                )
            )
            preferred.setdefault(parent, []).append(child)
    selected = []
    audit = []
    covered = np.zeros(len(q), bool)
    observed = np.zeros(len(q), bool)
    previous = np.zeros(len(q))
    for _ in range(search.max_grains):
        choices = []
        for i in range(len(models)):
            if (
                i in selected
                or i in preferred
                or counts[i] < search.min_indexed
                or phys[i] < search.min_indexed
            ):
                continue
            novel = int((core[i] & ~covered).sum())
            if options.competitive_selection:
                new_physical = int((physical[i] & (evidence[i] > previous)).sum())
            else:
                new_physical = int((physical[i] & ~observed).sum())
            if (
                not options.competitive_selection and novel < search.min_indexed
            ) or new_physical < search.min_indexed:
                continue
            gain = float(np.maximum(evidence[i] - previous, 0).sum())
            if gain > 0:
                choices.append((-gain, volumes[i], i, novel, new_physical))
        if not choices:
            break
        gain, _, i, novel, new_physical = min(choices)
        selected.append(i)
        covered |= core[i]
        observed |= physical[i]
        previous = np.maximum(previous, evidence[i])
        audit.append(
            dict(
                pool_id=i,
                tight_training=int(counts[i]),
                physical_training=int(phys[i]),
                novel_core=novel,
                novel_physical=new_physical,
                evidence=-gain,
            )
        )
    candidates = []
    for i, a in enumerate(models):
        reason = "selected" if i in selected else "redundant_or_lower_evidence"
        if i in preferred:
            reason = "nested_alias_without_independent_physical_support"
        if counts[i] < search.min_indexed:
            reason = "insufficient_tight_core"
        elif phys[i] < search.min_indexed:
            reason = "insufficient_physical_support"
        candidates.append(
            dict(
                pool_id=i,
                basis=a.tolist(),
                volume=volumes[i],
                tight_training=int(counts[i]),
                physical_training=int(phys[i]),
                evidence=float(evidence[i].sum()),
                status=reason,
            )
        )
    return [models[i] for i in selected], dict(
        selected=audit, candidates=candidates, families=family
    )


def _reciprocal_branch(
    engine, q, observed, geometry, models, charts, valid, search, options, aliases
):
    from .grains import _consolidate

    roots, _, consolidation = _consolidate(engine, q, models, search)
    branch, _, audit = refine_ownership_pool(
        engine,
        q,
        observed,
        geometry,
        roots,
        charts,
        valid,
        search,
        options,
        aliases,
        _preserve_reciprocal_branch=False,
    )
    return branch, dict(
        consolidation=consolidation, refinement=audit, roots=[a.tolist() for a in roots]
    )


def refine_ownership_pool(
    engine,
    q,
    observed,
    geometry,
    models,
    charts,
    valid,
    search,
    options,
    aliases,
    progress=None,
    input_aliases=None,
    _preserve_reciprocal_branch=True,
):
    """Training-only geometry fitting, late aliases and nonredundant selection."""
    from .grains import GrainAliasOptions, _alias_proposals, _consolidate, _same_oriented_basis

    models = [a.copy() for a in models]
    if not models:
        return [], np.zeros(len(q), np.int32), dict(selected=[], fits=[], joint=[])
    # Preserve the independent reciprocal-space trajectory before observation
    # fitting mutates any parent. Its training-only winners supply alternatives,
    # never assignments or validation outcomes. Both branches compete together.
    from . import fitting_tasks

    session = fitting_tasks.active()
    branch_ticket = None
    branch = []
    branch_audit = None
    if _preserve_reciprocal_branch and session is not None:
        branch_ticket = session.submit(
            "ownership_branch",
            (
                q,
                observed,
                geometry,
                [a.copy() for a in models],
                charts,
                valid,
                search,
                options,
                aliases,
            ),
        )
    if _preserve_reciprocal_branch and branch_ticket is None:
        roots, _, consolidation = _consolidate(engine, q, models, search)
        branch, _, branch_audit = refine_ownership_pool(
            engine,
            q,
            observed,
            geometry,
            roots,
            charts,
            valid,
            search,
            options,
            aliases,
            progress,
            _preserve_reciprocal_branch=False,
        )
        branch_audit = dict(
            consolidation=consolidation, refinement=branch_audit, roots=[a.tolist() for a in roots]
        )
    evidence, core, physical = _physical_fields(engine, q, models, charts, valid, search, options)
    # Preserve already eligible input proposals before geometry fitting changes
    # them. A fit variant is an additional hypothesis, not a replacement. Inputs
    # which cannot enter _select or its family comparison need not be appended.
    original_input_ids = (
        [
            i
            for i in range(len(models))
            if core[i].sum() >= search.min_indexed and physical[i].sum() >= search.min_indexed
        ]
        if options.preserve_input_models
        else []
    )
    original_inputs = [models[i].copy() for i in original_input_ids]
    order = sorted(
        range(len(models)),
        key=lambda i: (-float(evidence[i].sum()), abs(np.linalg.det(models[i])), i),
    )
    candidates = [
        (i, (models[i].copy(), search.seed + 4 + i, {})) for i in order if physical[i].sum() >= 17
    ]
    pending_fits = fitting_tasks.batches(
        "ownership", (q, observed, geometry, search, options, charts, valid), candidates
    )
    fitted = []
    fits = []
    for i in order:
        if len(fitted) >= options.max_fits:
            break
        if physical[i].sum() < 17 or any(
            _same_oriented_basis(models[i], models[j]) for j in fitted
        ):
            continue
        if progress:
            progress()
        a, record = (
            _fit_one(engine, q, observed, geometry, models[i], search, options, search.seed + 4 + i)
            if pending_fits is None
            else pending_fits.get(i)
        )
        models[i] = a
        fitted.append(i)
        fits.append(dict(pool_id=i, **record))
    alias_audit = []
    for cycle in range(options.alias_rounds):
        found, audit = _alias_proposals(
            engine,
            q,
            observed,
            geometry,
            [models[i] for i in fitted],
            search,
            aliases or GrainAliasOptions(),
            progress,
        )
        alias_audit.append(
            dict(cycle=cycle, root_pool_ids=fitted.copy(), pool_start=len(models), audit=audit)
        )
        models.extend(found)
    post_alias = polish_ownership_aliases(
        engine,
        q,
        observed,
        geometry,
        models,
        (input_aliases or []) + alias_audit,
        search,
        options,
        progress,
    )
    input_start = len(models)
    models.extend(original_inputs)
    branch_start = len(models)
    if branch_ticket is not None:
        branch, branch_audit = branch_ticket.result()
    models.extend(branch)
    selected, labels, audit, joint = select_and_refine_ownership(
        engine, q, observed, geometry, models, charts, valid, search, options, progress
    )
    return (
        selected,
        labels,
        dict(
            options=asdict(options),
            fits=fits,
            late_aliases=alias_audit,
            post_alias_fits=post_alias,
            selected=audit["selected"],
            candidate_selection=audit["candidates"],
            families=audit["families"],
            joint=joint,
            pool_size=len(models),
            unresolved_training=int((~valid).sum()),
            preserved_input_proposals=dict(
                pool_start=input_start, proposed=len(original_inputs), input_ids=original_input_ids
            ),
            preserved_reciprocal_branch=dict(
                pool_start=branch_start, proposed=len(branch), audit=branch_audit
            ),
        ),
    )


def _fit_alias_cell(
    engine,
    q,
    observed,
    geometry,
    search,
    options,
    charts,
    valid,
    basis,
    pool_id,
    alias_cycle,
    offset,
    progress=None,
):
    audit = []
    previous_rows = None
    for cycle in range(options.alias_fit_cycles):
        if progress:
            progress()
        before = basis
        basis, record = _fit_one(
            engine,
            q,
            observed,
            geometry,
            basis,
            search,
            options,
            search.seed + 1004 + alias_cycle * 1009 + offset * 101 + cycle,
            all_training=True,
            charts=charts,
            valid=valid,
        )
        audit.append(dict(pool_id=pool_id, cycle=cycle, **record))
        rows = record["training_rows"]
        if rows == previous_rows and np.max(abs(basis - before)) < 1e-8:
            break
        previous_rows = rows
    return basis, audit


def _parallel_alias_fits(
    engine, q, observed, geometry, models, alias_audit, search, options, progress
):
    from . import fitting_tasks

    audit = []
    charts, valid = observation_charts(observed, geometry, options)
    common = q, observed, geometry, search, options, charts, valid
    for entry in alias_audit:
        start = entry["pool_start"]
        count = entry["audit"]["proposed"]
        cases = [
            (j, (models[start + j].copy(), start + j, entry["cycle"], j)) for j in range(count)
        ]
        pending = fitting_tasks.batches("alias", common, cases)
        for j, case in cases:
            if pending is None:
                basis, records = _fit_alias_cell(engine, *common, *case, progress=progress)
            else:
                basis, records = pending.get(j)
                if progress:
                    for _ in records:
                        progress()
            models[start + j] = basis
            audit.extend(records)
    return audit


def polish_ownership_aliases(
    engine, q, observed, geometry, models, alias_audit, search, options, progress=None
):
    """Refit transformed cells on their own training support before selection.

    Integer reduction changes membership. Reusing only the parent's conditional
    fit can strand the child at the tight-core boundary. Global training scope,
    original strict physical gates and the fixed refinement budget are retained.
    """
    from . import fitting_tasks

    if fitting_tasks.active() is not None:
        return _parallel_alias_fits(
            engine, q, observed, geometry, models, alias_audit, search, options, progress
        )
    audit = []
    charts, valid = observation_charts(observed, geometry, options)
    for entry in alias_audit:
        start = entry["pool_start"]
        count = entry["audit"]["proposed"]
        for j in range(count):
            previous_rows = None
            for cycle in range(options.alias_fit_cycles):
                if progress:
                    progress()
                i = start + j
                before = models[i]
                a, record = _fit_one(
                    engine,
                    q,
                    observed,
                    geometry,
                    models[i],
                    search,
                    options,
                    search.seed + 1004 + entry["cycle"] * 1009 + j * 101 + cycle,
                    all_training=True,
                    charts=charts,
                    valid=valid,
                )
                models[i] = a
                audit.append(dict(pool_id=i, cycle=cycle, **record))
                rows = record["training_rows"]
                if rows == previous_rows and np.max(abs(a - before)) < 1e-8:
                    break
                previous_rows = rows
    return audit


def select_and_refine_ownership(
    engine, q, observed, geometry, models, charts, valid, search, options, progress=None
):
    """Replayable training-only selection from an immutable fitted FFT pool."""
    selected, audit = _select(engine, q, models, charts, valid, search, options)
    labels, _, _ = assign_observed(engine, q, selected, charts, valid, search, options)
    joint = []
    for cycle in range(options.refinement_cycles):
        logs = []
        updated = []
        # Frozen competitors for this simultaneous training update. Uniform
        # noise is normalised over the existing physical eligibility cylinder.
        noise = (
            (2 * np.pi) ** 1.5
            * options.xy_scale**2
            * options.omega_scale
            / (2 * np.pi * options.xy_gate**2 * options.omega_gate)
        )
        likelihoods = []
        from .grains import _score

        for a in selected:
            h, _, near = _score(engine, q, a, search.reciprocal_tolerance)
            r = np.einsum("nij,nj->ni", charts, q - h @ np.linalg.inv(a).T)
            eligible = (
                valid
                & near
                & (np.linalg.norm(r[:, :2], axis=1) <= options.xy_gate)
                & (abs(r[:, 2]) <= options.omega_gate)
            )
            scaled = r / [options.xy_scale, options.xy_scale, options.omega_scale]
            likelihoods.append(np.where(eligible, np.exp(-0.5 * np.sum(scaled**2, axis=1)), 0))
        total = noise + np.sum(likelihoods, axis=0)
        from . import fitting_tasks

        pending = fitting_tasks.batches(
            "ownership",
            (q, observed, geometry, search, options, charts, valid),
            [
                (
                    i,
                    (
                        a.copy(),
                        search.seed + 104 + cycle * 101 + i,
                        dict(background=total - likelihoods[i], all_training=True),
                    ),
                )
                for i, a in enumerate(selected)
            ],
        )
        for i, a in enumerate(selected):
            if progress:
                progress()
            a, record = (
                _fit_one(
                    engine,
                    q,
                    observed,
                    geometry,
                    a,
                    search,
                    options,
                    search.seed + 104 + cycle * 101 + i,
                    background=total - likelihoods[i],
                    all_training=True,
                    charts=charts,
                    valid=valid,
                )
                if pending is None
                else pending.get(i)
            )
            updated.append(a)
            logs.append(dict(model=i + 1, **record))
        newlabels, amb, multiple = assign_observed(
            engine, q, updated, charts, valid, search, options
        )
        changed = int((labels != newlabels).sum())
        joint.append(
            dict(cycle=cycle, changed=changed, ambiguous=amb, multiple=multiple, fits=logs)
        )
        selected, labels = updated, newlabels
        if not changed:
            break
    return selected, labels, audit, joint
