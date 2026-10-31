"""Opt-in ordered process scheduling for independent positional searches.

Each spawned worker owns one native numerical worker. Inputs/outputs contain
training proposals only. Parent-side ordering, competition and validation stay
outside the executor. No runtime dependency beyond the standard library.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

__all__ = []

_LIMITS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_ACTIVE = ContextVar("phasesmith_search_processes", default=None)
_ENGINE = None


def _initialize():
    global _ENGINE
    if any(os.environ.get(name) != "1" for name in _LIMITS):
        raise RuntimeError("search worker requires single-thread numerical environment")
    from phasesmith import _core

    from . import discovery_cache

    _ENGINE = _core.ExperimentalLatticeEngine(1)
    discovery_cache._ACTIVE.set(discovery_cache._Cache())


def _execute(kind, payload):
    from . import discovery_cache, grains, recovery

    wall, cpu = time.perf_counter(), time.process_time()
    if kind == "discover":
        result = grains._discover(*payload)
    elif kind == "conditioned_directions":
        q, options, conditioning, diverse = payload
        seed, _ = grains._conditioning_options(options, conditioning)
        result = grains._vectors(_ENGINE, q, seed, diverse=diverse)
    elif kind == "conditioned_seed":
        result = grains._conditioned_seed(_ENGINE, *payload)
    elif kind == "paired_seed":
        result = recovery._paired_choices(_ENGINE, *payload)
    elif kind == "fitting_batch":
        from . import fitting_tasks

        result = fitting_tasks._batch(_ENGINE, *payload)
    elif kind == "ownership_branch":
        from . import ownership

        result = ownership._reciprocal_branch(_ENGINE, *payload)
    elif kind == "coherent_parent":
        result = recovery._coherent_parent(_ENGINE, *payload)
    else:
        raise ValueError("unknown positional search task")
    report = dict(
        kind=kind,
        pid=os.getpid(),
        native_workers=1,
        numerical_environment={name: os.environ[name] for name in _LIMITS},
        seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu,
        fourier_cache=discovery_cache.statistics(),
    )
    return result, report


class _Ticket:
    def __init__(self, session, future):
        self.session = session
        self.future = future
        self.collected = False

    def result(self):
        result, report = self.future.result()
        if not self.collected:
            self.session.jobs.append(report)
            self.collected = True
        return result


class _Session:
    def __init__(self, workers, fit_tasks):
        self.workers = workers
        self.fit_tasks = fit_tasks
        self.submitted = 0
        self.jobs = []
        self.tickets = []
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
        )

    def submit(self, kind, payload):
        self.submitted += 1
        ticket = _Ticket(self, self.executor.submit(_execute, kind, payload))
        self.tickets.append(ticket)
        return ticket

    def map(self, kind, payloads):
        # Submit before yielding; read in input order, never completion order.
        futures = [self.executor.submit(_execute, kind, payload) for payload in payloads]
        self.submitted += len(futures)

        def results():
            for future in futures:
                result, report = future.result()
                self.jobs.append(report)
                yield result

        return results()

    def statistics(self):
        # Include unused speculative fits: they still consume CPU and shutdown waits.
        for ticket in self.tickets:
            ticket.result()
        return dict(
            executor="spawned processes",
            max_processes=self.workers,
            native_workers_per_search=1,
            merge_order="original task order",
            pids=sorted({job["pid"] for job in self.jobs}),
            tasks=len(self.jobs),
            submitted_tasks=self.submitted,
            fitting_tasks_enabled=self.fit_tasks,
            jobs=list(self.jobs),
            cache_scope="worker-local, at most 64 MiB each; discarded at session shutdown",
        )


class _Views:
    def __init__(self, session, engine, fit, options, conditioning, stability, sparse):
        import math
        from dataclasses import asdict, replace

        import numpy as np

        from .grains import GrainConditioningOptions, GrainStabilityOptions

        self.groups = {}
        jobs = []
        if conditioning is not None:
            jobs.append((fit, options, conditioning, False))
            self.groups["conditioning"] = None
        for name, limits, policy in [
            ("stability", conditioning or GrainConditioningOptions(), stability),
            ("sparse", sparse, stability or GrainStabilityOptions()),
        ]:
            if policy is None or (name == "sparse" and sparse is None):
                continue
            proposal = replace(options, min_indexed=policy.proposal_min_indexed)
            size = max(proposal.min_indexed, math.ceil(policy.fraction * len(fit)))
            if size > len(fit):
                raise ValueError("insufficient training points for stability proposal minimum")
            rng = np.random.default_rng(options.seed + 2)
            ids = [np.arange(len(fit))]
            ids.extend(
                np.sort(rng.choice(len(fit), size, replace=False)) for _ in range(policy.resamples)
            )
            self.groups[name] = (ids, asdict(policy), options.seed + 2)
            jobs.extend((fit[rows], proposal, limits, True) for rows in ids)
        from . import grains

        prepared = session.map("conditioned_directions", jobs)
        finishers = []
        for (q, o, limits, diverse), directions in zip(jobs, prepared, strict=True):
            finishers.append(
                grains._conditioned_proposals(
                    engine, q, o, limits, diverse=diverse, prepared=directions, _defer=True
                )
            )
        self.results = iter(finishers)

    def take(self, name):
        group = self.groups[name]
        if group is None:
            return next(self.results)()
        ids, policy, seed = group
        models, views = [], []
        for i, rows in enumerate(ids):
            found, audit = next(self.results)()
            views.append(dict(view=i, training_rows=rows.tolist(), pool_start=len(models), **audit))
            models.extend(found)
        return models, dict(
            options=policy, seed=seed, diverse=True, views=views, proposed=len(models)
        )


def conditioned_views(engine, fit, options, conditioning, stability, sparse):
    """Queue independent full/subsample views, retaining their original order."""
    session = active()
    return (
        None
        if session is None
        else _Views(session, engine, fit, options, conditioning, stability, sparse)
    )


def active():
    """Return the current parent-side search session, or None."""
    return _ACTIVE.get()


@contextmanager
def parallel_searches(workers=8, *, fit_tasks=True):
    """Schedule one inverse's independent searches in single-worker processes.

    Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and VECLIB_MAXIMUM_THREADS to 1
    before starting Python. No parent environment is changed. Use the normal
    multiprocessing main guard. Startup and shutdown belong in complete timing.
    The context can cover discovery and subsequent recovery for that inverse.
    """
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("ensemble workers must be an integer in [1,16]")
    if type(fit_tasks) is not bool:
        raise ValueError("fit_tasks must be boolean")
    current = active()
    if current is not None:
        if workers not in (1, current.workers):
            raise ValueError("nested search sessions must share the worker budget")
        yield current
        return
    if workers == 1:
        yield None
        return
    if any(os.environ.get(name) != "1" for name in _LIMITS):
        raise ValueError(
            "set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and "
            "VECLIB_MAXIMUM_THREADS=1 before starting Python for parallel searches"
        )
    session = _Session(workers, fit_tasks)
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)
        session.executor.shutdown(wait=True, cancel_futures=True)


def ensemble(function):
    """Expose a per-call process budget while preserving serial defaults."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with parallel_searches(kwargs.get("ensemble_workers", 1)):
            return function(*args, **kwargs)

    return wrapped
