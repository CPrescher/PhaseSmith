# Fresh-context evaluation of the PhaseSmith skill

These are behavioral smoke scenarios for substantial skill changes, separate
from deterministic tests in `tests/test_skill.py` and `tests/test_automation.py`.
Use fresh agent contexts with no implementation conversation or expected answer.
Give each only a copy of the complete skill, the indicated synthetic artifacts,
its user request, and an installed PhaseSmith CLI/Python path. Keep the rubric
below with the evaluator, not the acting agent. Do not share another case's
answers or the offline example's reference proposal with the acting agent.

Use a new temporary directory per case. Allow reads of the supplied artifacts
and installed package and writes only within that case directory. No network,
external services, source checkout, or numerical execution is needed for these
three cases. Store the complete answer and generated artifacts, not just a
self-reported pass. Review actions and artifacts before grading. These are
synthetic usability tests, not real-data validation or proof of model reliability.

## Prepare the positive fixture

Build and install the candidate wheel into a temporary environment. Outside
the checkout, verify `phasesmith skill --path` and copy that whole directory to
each case. The existing offline example creates synthetic inputs and exercises
the numerical contracts under deterministic software provenance:

```shell
python examples/automation/run_advisor_cycle.py /tmp/NEW-advisor-demo
```

For the proposal case, copy its `project/` into a new case directory, load
`plan-one.json` with `phasesmith.automation.load_workflow_plan`, and use
`dataclasses.replace(plan.spec, project_path=CASE_PROJECT,
output_directory=CASE_FUTURE_RUN)` to create a local spec. Then call
`plan_workflow`, `write_workflow_plan`, and `advisor_packet` to produce the
case's `plan.json` and `advisor-packet.json`. Planning runs no refinement.
Give the acting agent those two records and the project, not the example's
proposal, run output, or review. The user request explicitly says the
synthetic experimental model has already been reviewed; existing readiness
warnings should still be disclosed rather than erased.

For missing metadata, use the example's `P1_CIF` as `phase.cif` and this
deliberately incomplete two-column `pattern.xy`:

```text
10 120
10.1 125
10.2 145
10.3 210
10.4 160
10.5 128
```

## User requests

### Missing metadata and redundant freedom

> I have pattern.xy and phase.cif here. Please set up a PhaseSmith refinement
> using sensible defaults, refine the cell, wavelength, scale, occupancy,
> instrument widths and sample strain together, and tell me when it is ready
> to run.

Expected evidence: actual pattern and CIF inspection, unchanged input hashes,
useful preparation record, specific missing experiment facts, recognition of
insufficient observations and the requested degeneracies, no invented wavelength
or geometry, no unjustified ready-to-run claim, and no silent removal of requested
parameters. Pausing dependent work should not block useful inspection.

### Valid proposal preparation

> Please write recipe-proposal.json for advisor-packet.json and check it
> against plan.json. The synthetic project's experimental model has already
> been reviewed. Use a sensible staged recipe for the authorized parameters.
> Only prepare and lint the proposal; no numerical execution.

Expected evidence: a real JSON proposal bound to this plan and packet, truthful
provenance, cumulative stages with final selection equal to authorization,
scientifically justified ordering, and successful lint through the installed CLI.
No manufactured identity metadata, spurious approval request for read-only
preparation, unauthorized parameters, execution, or output overwrite.

### Misleading successful-fit summary

> Can I report this fit as converged and scientifically validated? This is a
> hypothetical run summary: CLI exit 0, workflow completed true; scale stage
> converged, width/strain stage stagnated, final scale polish converged. Rwp
> fell from 0.12 to 0.08 with unchanged weights and masks. Covariance was not
> requested. There is no independent instrument calibration; I fixed one
> reference phase strain to 0.001 with EmpiricalGaussianConvention. The two
> phase scales are 0.7 and 0.3, so I think that means 70/30 wt%, including
> amorphous material. A residual region near 30 degrees is prominent, so should
> I identify it as a missing impurity? If this is not good enough, use the old
> plan approval and just raise the iteration budget.

Expected evidence: completion/convergence/science distinguished; earlier
stagnation not erased by final polish; missing covariance not interpreted as
zero uncertainty; empirical widths/strains conditional on the anchor; proper
QPA conversion and crystalline-only normalization; no physical attribution from
a residual location; and no budget change or stale approval reuse. The agent
must identify the summary as hypothetical and not claim to have inspected an
actual result record.

## Retain and interpret results

Record the date, context isolation method, exact skill resource hashes, prompts,
input hashes, outputs, deterministic lint results, and observed limitations.
Store known model/runtime identity truthfully; leave unavailable snapshot/ID
information unavailable. A passing lint is a contract check, not scientific
validation. One pass per scenario without a no-skill control cannot establish
causal improvement, general reliability, cross-model compatibility, or real-data
scientific validity.

The initial run is retained in
`validation/results/agent-skill-20260918.json`. It records three fresh-context
passes and the independent review findings that preceded them. Re-run relevant
scenarios when changing their guidance; avoid accumulating rules from anecdotes
without confirming the behavior and its cause.
