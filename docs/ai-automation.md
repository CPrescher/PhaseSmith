# AI-guided automation

PhaseSmith can be used effectively by a coding agent or language model without
embedding an AI client in the numerical library. Version 0.5 introduces a
finite JSON task boundary and a `phasesmith` command-line entry point. An AI may
inspect a plan and propose a scientifically motivated staged recipe; PhaseSmith
remains responsible for type checks, parameter authorization, constraints,
budgets, stale-input detection, execution, and auditable results.

This split is intentional. A model such as `gpt-5.6-sol` can be useful for
explaining correlations and choosing a conservative order, but its name is
provenance, not authority. The same validation and approval gates apply to
every model and to hand-written proposals.

## What version 1 automates

The workflow contract operates on a persisted `RietveldProject`. Project
construction still belongs in a reviewed Python script because a powder file
and CIF do not identify radiation, wavelength, instrument response, specimen
geometry, intensity correction, background, omitted phases, or which
parameters are scientifically justified.

The CLI can inspect raw inputs without guessing those facts:

```shell
phasesmith inspect-pattern pattern.xy
phasesmith inspect-cif phase.cif
```

Both commands return finite JSON with a SHA-256 digest. Pattern inspection also
lists the physical information that remains unknown. It does not create or run
a refinement.

Once a typed project has been reviewed and saved, the supported task sequence
is:

```text
workflow spec -> read-only plan -> optional proposal -> lint -> approval -> run
                                                                        |
                                               new approval <- replan <- review
```

Planning hashes every regular project file, loads the project, runs the
deterministic readiness review, discloses the complete authorized parameter
selection, supplies PhaseSmith's deterministic recipe, and includes a sanitized
`advisor_context`. That context summarizes the pattern domain, radiation,
instrument and geometry, phases, scattering providers, background, parameter
values/bounds/scales, and constraints. It contains neither filesystem paths nor
raw pattern/CIF contents; identifiers and scientific metadata remain visible.
Planning performs no objective evaluation and writes nothing unless an output
path is requested.

## Plan and run

Start from the example contract in `examples/automation/workflow-spec.json`
and change its project and output paths. Then print a plan:

```shell
phasesmith plan workflow-spec.json --output plan.json
```

Review `status`, every readiness diagnostic, the parameter authorization, the
default recipe, and the runtime/output limits. Execution requires the exact
64-character `plan_id` from this record:

```shell
phasesmith run workflow-spec.json --approve PLAN_ID
```

If any byte in the project or any planning input changes, the old approval is
stale and execution stops. Existing owned outputs are rejected before
numerical work unless `--overwrite` is explicit. A saved checkpoint is
continued with:

```shell
phasesmith resume workflow-spec.json --approve PLAN_ID
```

`resume` is not a synonym for restart: it fails if the persisted project has
no checkpoint. `phasesmith report OUTPUT_DIRECTORY` prints the terminal audit
record for either action. CLI status `0` means the approved action reached a
safe accepted completion, `2` means the request or boundary failed, and `3`
means execution returned an auditable but incomplete/stopped refinement.

## Asking an AI for a recipe

Obtain the proposal schema with:

```shell
phasesmith schema recipe
```

For a remote advisor, send only the fields it needs: `plan_id`, `readiness`,
`authorization`, `advisor_context`, `external_recipe_proposal`, and `guidance`,
plus the recipe schema. The complete plan also contains local paths and should
not be treated as a sanitized upload packet. Review the advisor context itself
if identifiers or numerical metadata are sensitive.

A useful prompt is:

```text
You are advising on a powder-diffraction Rietveld workflow. Return only JSON
matching the supplied PhaseSmith recipe-proposal schema. Treat readiness
diagnostics and parameter authorization as hard facts. Do not invent phases,
physics, constraints, parameter names, or experimental metadata. Make stages
cumulative and make the final selection exactly equal to maximum_selection.
Explain each stage scientifically. Establish scale/background first, align
positions before widths, delay occupancy until scale is stable, and call out
high-risk correlated pairs. Do not use lower Rwp alone as an acceptance
argument. If the plan lacks information needed for a defensible order, state
that in assumptions; do not guess.
```

The proposal record must be bound to the exact plan. `generated_by` may record
the model name, for example `gpt-5.6-sol`. Each stage contains only a name,
selection, and rationale. External proposals cannot set solver options, relax
accepted termination reasons, expand authorization, remove a parameter in a
later stage, or omit the complete final authorized selection.

After saving the proposed JSON, run the deterministic linter before approval:

```shell
phasesmith lint-recipe workflow-spec.json recipe-proposal.json
```

The linter first applies the strict executable contract and then flags risky
stage ordering and combinations such as early occupancy, broad parameter
release, scale--occupancy, lattice--wavelength, zero--displacement, or widths
released without an earlier position foundation. Findings are reproducible
heuristics, not proof that a model is scientifically correct. A contract error
returns status `3`; boundary or I/O failures return status `2`.

Run the proposal through the same explicit approval gate:

```shell
phasesmith run workflow-spec.json \
  --proposal recipe-proposal.json \
  --approve PLAN_ID
```

PhaseSmith validates all stages and the complete constraint graph before the
first numerical evaluation. The output directory retains the approved plan,
the exact recipe, per-stage workflow record, numerical result, terminal audit,
and optionally pattern CSV and resumable project.

## Review and iterative replanning

After a workflow run, derive a deterministic post-run review:

```shell
phasesmith review OUTPUT_DIRECTORY > review.json
```

The review is digest-bound to the stored plan, workflow, result, and optional
pattern CSV. It reports per-stage Rwp movement, scaled parameter movement,
bound contacts, rank/correlation findings, and residual magnitude, lag-1
structure, and ten coordinate-region summaries when CSV output is available.
Its status is always `review_required`: no threshold silently accepts a fit,
and plots, provenance, phase completeness, and physical plausibility still
need scientific judgment.

If the run saved its accepted project state, create a new plan for another
iteration without executing it:

```shell
phasesmith replan OUTPUT_DIRECTORY \
  --output-directory NEXT_OUTPUT_DIRECTORY \
  --plan-output next-plan.json
```

The new plan uses an output directory outside the parent audit directory and a
different plan ID, and records the
parent plan, terminal-result digest, and review digest in `lineage`. It plans
from the saved accepted state, repeats readiness and fingerprint checks, and
still requires a new explicit approval. Replanning never edits the previous
output and never installs a model callback inside the numerical loop.

## Scientific recipe rubric

The following is guidance, not a universal law. An advisor should explain any
departure using experiment-specific evidence.

1. Resolve readiness errors and review warnings before refinement.
2. Establish phase scale and a justified differentiable background before
   strongly correlated intensity terms.
3. Align peak positions using justified lattice or instrument nuisance terms
   before releasing profile widths.
4. Avoid unconstrained lattice--wavelength and zero--displacement pairs. If
   both are authorized, identify the independent constraint or recommend that
   the project owner narrow the authorization.
5. Delay coordinates, occupancy, and displacement parameters until scale,
   background, and positions are credible. Occupancy and phase scale are
   particularly correlated.
6. Release sample and profile broadening after position and intensity models,
   then finish with all caller-authorized parameters active together.
7. Review termination, bounds, rank, parameter correlations, residual shape,
   provenance, and physical plausibility. A smaller `Rwp` is necessary evidence
   in many fits, but never sufficient evidence by itself.

An AI can also write a normal PhaseSmith Python script. The task API is useful
when repeatability and review matter: it supplies versioned schemas, stable
machine-readable errors, byte-bound approvals, finite budgets, collision
policy, and a complete audit trail that an unconstrained generated script would
otherwise need to recreate.

## Current boundary

Version 1 recipes remain fixed during each execution. Iteration happens only
between completed/stopped runs through review and a new byte-bound plan. An AI
may interpret the review and propose the next recipe, but it cannot alter an
active solve, expand authorization, reuse the old approval, or decide that the
result is accepted.

There is deliberately no OpenAI, cloud, or model-provider runtime dependency.
Offline scripts, local models, hosted agents, CI systems, notebooks, and future
GUIs all use the same JSON/Python boundary.
