---
name: phasesmith-ai-workflows
description: Guide safe AI-assisted PhaseSmith Rietveld workflows. Use when planning, proposing or linting staged refinement recipes, preparing model-facing advisor packets, running an explicitly approved PhaseSmith plan, reviewing refinement evidence, or creating a lineage-bound next iteration from persisted project or automation JSON files.
---

# PhaseSmith AI workflows

Use PhaseSmith's versioned JSON/CLI boundary. Treat an AI as an external
scientific advisor, never as numerical authority or an in-solver callback.

## Establish the boundary

1. Read the repository `PROJECT_BRIEF.md` before changing architecture or
   numerical behavior.
2. Operate on a reviewed, persisted `RietveldProject` for runnable workflows.
3. Use `phasesmith inspect-pattern` and `phasesmith inspect-cif` only to report
   bounded facts about raw inputs. Never infer radiation, wavelength,
   instrument response, geometry, background, phases, or authorization from
   those files.
4. Keep GSAS-II outside the runtime. Treat it only as the repository's pinned
   black-box validation oracle.

## Prepare advice

Create and retain the full byte-bound plan:

```shell
phasesmith plan workflow-spec.json --output plan.json
```

Create the sanitized model-facing packet:

```shell
phasesmith advisor-packet plan.json > advisor-packet.json
```

Share the advisor packet, not the full plan. Confirm its privacy flags and
review visible identifiers and scientific metadata before remote disclosure.
Use `phasesmith schema recipe-proposal` as the exact response contract.

## Propose a recipe

Return only recipe-proposal JSON. Apply these rules:

- Bind `plan_id` to the packet's plan and
  `provenance.advisor_packet_id` to its packet ID.
- Record `provenance.kind`, proposer/model name, provider for models, optional
  model snapshot/version, client, prompt SHA-256, and request ID truthfully.
- State uncertainty in `assumptions`; do not invent absent physics.
- Use only `authorization.maximum_selection` fields and parameter names.
- Keep stages cumulative and make the final stage exactly equal to the maximum
  selection.
- Establish scale and justified background before correlated intensity terms.
- Align justified position terms before profile widths.
- Delay occupancy and sample/structural terms until scale, background, and
  positions are credible.
- Call out scale--occupancy, lattice--wavelength, and
  zero--displacement correlation risks.
- Never change solver options, termination policy, limits, or constraints.

Lint before asking for approval:

```shell
phasesmith lint-recipe plan.json recipe-proposal.json
```

Treat lint warnings as prompts for scientific review, not automatic rejection
or acceptance. Correct contract errors before numerical work.

## Approval and execution

Present the plan ID, readiness findings, proposed stages, assumptions, lint
findings, budgets, and outputs to the user. Do not execute until the user
explicitly approves that exact plan ID.

```shell
phasesmith run plan.json --proposal recipe-proposal.json --approve PLAN_ID
```

Never substitute a different plan ID, reuse an earlier approval, silently
overwrite outputs, or broaden authorization.

## Review and iterate

Create deterministic post-run evidence:

```shell
phasesmith review OUTPUT_DIRECTORY > review.json
```

Review termination, stage improvement, parameter motion, bounds, rank,
correlations, regional/serial residual structure, provenance, plots, and
physical plausibility together. Never accept a fit from lower Rwp alone.

Prepare a new plan and path-free review packet for another advisory pass:

```shell
phasesmith review-packet OUTPUT_DIRECTORY \
  --output-directory NEXT_OUTPUT_DIRECTORY \
  --plan-output next-plan.json > review-packet.json
```

Share the complete path-free `review-packet.json` so the prior review remains
available as evidence. Bind the next proposal only to its nested
`next_advisor_packet`, lint it against `next-plan.json`, present the new
evidence, and require approval of the new plan ID. Do not edit the parent audit
directory or place an AI callback inside an active solve.

## Stop conditions

Stop and ask for scientific input when readiness has errors, essential
experimental metadata is absent, requested parameters exceed authorization,
the proposed parameterization is not identifiable, or a result would require
claiming phase completeness or physical validity not supported by the packet.
