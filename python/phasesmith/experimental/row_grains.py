"""Training-only row proposals into the existing PhaseSmith lattice workflow."""

from dataclasses import replace

import numpy as np

from phasesmith import _core

from .rows import ROW_SEARCH_NAME, group_rows


def discover_rows(fit, options, row_options, progress):
    """Never receives held-out coordinates; keeps every raw group and fit audit."""
    from .grains import _consolidate, _discover

    if progress:
        progress(0, options.max_grains)
    grouping = group_rows(fit, options=row_options)
    if not grouping.complete:
        raise RuntimeError(
            "Row search exhausted its work/deadline budget; "
            "use group_rows to inspect partial output"
        )
    proposals, audits, vector_parts, score_parts = [], [], [], []
    # A row group is not already an indexed grain. Infer a basis from its own
    # training observations, then consolidate candidates against all training.
    # Existing FFT cell inference is reused here, not the whole-cloud FFT search.
    per_group = replace(options, max_grains=1)
    for number, members in enumerate(grouping.groups, 1):
        if progress:
            progress(min(options.max_grains - 1, number - 1), options.max_grains)
        audit = dict(group=number, training_members=members.tolist(), count=len(members))
        audits.append(audit)
        block = fit[members]
        if len(block) < options.min_indexed or np.linalg.matrix_rank(block) < 3:
            audit["status"] = "too_small_or_rank_deficient"
            continue
        try:
            models, _, vectors, scores, records = _discover(block, per_group)
        except (ValueError, np.linalg.LinAlgError) as exc:
            audit.update(status="cell_inference_failed", error=str(exc))
            continue
        audit.update(
            status="proposed" if models else "no_cell",
            inference=records,
            proposal_ids=list(range(len(proposals), len(proposals) + len(models))),
        )
        proposals.extend(models)
        vector_parts.append(vectors)
        score_parts.append(scores)
    engine = _core.ExperimentalLatticeEngine(options.workers)
    models, labels, consolidation = _consolidate(engine, fit, proposals, options)
    diagnostics = [
        dict(
            method="rows",
            display_name=ROW_SEARCH_NAME,
            row_options=grouping.diagnostics["options"],
            row_work={k: v for k, v in grouping.diagnostics.items() if k != "options"},
            groups=audits,
            consolidation=consolidation,
        )
    ]
    vectors = np.concatenate(vector_parts) if vector_parts else np.empty((0, 3))
    scores = np.concatenate(score_parts) if score_parts else np.empty(0)
    return models, labels, vectors, scores, diagnostics, grouping
