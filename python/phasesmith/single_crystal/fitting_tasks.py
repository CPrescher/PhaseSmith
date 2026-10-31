"""Ordered batches of independent training fits for the existing process pool.

Frozen inputs are copied before dispatch. Speculative fit results are consumed
only if the original coordinator selects that fit; unused exceptions do not
change decisions. All numerical fit functions remain outside this transport.
"""

import math

from . import search_execution

__all__ = []


def active():
    session = search_execution.active()
    return session if session is not None and session.fit_tasks else None


class _Cases:
    def __init__(self, tickets):
        self.tickets = tickets
        self.results = {}

    def get(self, index):
        ticket = self.tickets[index]
        identity = id(ticket)
        if identity not in self.results:
            self.results[identity] = dict(ticket.result())
        result, error = self.results[identity][index]
        if error is not None:
            raise error
        return result


def batches(kind, common, cases):
    """Batch shared training arrays once per worker; preserve each case's index."""
    session = active()
    if session is None or len(cases) < 2:
        return None
    count = min(session.workers, len(cases))
    width = math.ceil(len(cases) / count)
    tickets = {}
    for start in range(0, len(cases), width):
        group = cases[start : start + width]
        ticket = session.submit("fitting_batch", (kind, common, group))
        tickets.update((i, ticket) for i, _ in group)
    return _Cases(tickets)


def _batch(engine, kind, common, cases):
    from . import grains, ownership, recovery

    results = []
    for index, case in cases:
        try:
            if kind == "ownership":
                q, observed, geometry, search, options, charts, valid = common
                basis, seed, params = case
                result = ownership._fit_one(
                    engine,
                    q,
                    observed,
                    geometry,
                    basis,
                    search,
                    options,
                    seed,
                    charts=charts,
                    valid=valid,
                    **params,
                )
            elif kind == "forward":
                result = recovery._forward_training_fit(engine, *common, *case)
            elif kind == "alias":
                result = ownership._fit_alias_cell(engine, *common, *case)
            elif kind == "polish":
                result = grains._polish_one(engine, *common, *case)
            else:
                raise ValueError("unknown training fitting task")
        except Exception as error:
            results.append((index, (None, error)))
        else:
            results.append((index, (result, None)))
    return results
