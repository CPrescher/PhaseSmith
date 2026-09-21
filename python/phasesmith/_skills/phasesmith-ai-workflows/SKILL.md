---
name: phasesmith-ai-workflows
description: Use PhaseSmith to inspect diffraction inputs, advise on staged Rietveld recipes, execute approved automation plans, and interpret refinement evidence. Apply to PhaseSmith scientific workflows and their JSON/CLI boundary, including missing experimental context, correlated parameters, and review or replanning of a fit.
---

# PhaseSmith AI workflows

Help the user carry out and judge a PhaseSmith diffraction workflow. PhaseSmith
performs the numerical calculation; the agent supplies advice and explanation.
Use the installed version's commands, schemas, and result records as the API
contract. This skill is for using PhaseSmith, not modifying its numerical core.
It needs neither a source checkout nor GSAS-II nor a model-provider SDK.

## Read what the task needs

| Task | Reference |
| --- | --- |
| Raw pattern/CIF, missing experiment facts, choice of model or parameter groups | [Experimental context](references/experiment.md) |
| Plan, advisor packet, recipe proposal, lint, approved execution, resume, or another iteration | [Automation protocol](references/workflow.md) |
| Fit quality, termination, correlations, QPA, or instrument/sample broadening claims | [Interpretation](references/interpretation.md) |

Inspect the user's existing artifacts before requesting information they may
already contain. For a raw-input task, use `phasesmith inspect-pattern` and
`phasesmith inspect-cif`; record known facts and remaining unknowns. A CIF
supplies a candidate structure, not evidence of phase completeness. Do not
choose a wavelength or geometry from an extension, a plausible cell, or a good
fit. Readiness checks cannot establish the truth of supplied metadata.

## Choose the appropriate boundary

The automation v1 CLI starts from a reviewed, persisted `RietveldProject`.
Project construction requires a Python script with explicit experimental and
model choices. Other PhaseSmith Python workflows exist; do not force TOF,
calibration, or Le Bail inputs through a CW Rietveld JSON contract. Consult the
installed API and the experimental-context reference for those tasks.

For the automation path, prepare a plan and a path-free advisor packet, propose
only caller-authorized stages, lint the proposal, then execute the reviewed
plan. Follow the workflow reference for the exact commands and contracts.
Existing approval of that exact plan is sufficient; do not request it again.
Approval of an earlier plan does not authorize a changed plan. Do not use a
lower-level API to bypass the chosen workflow's approval or authorization.

No refinement or approval is needed just to inspect inputs, explain a result,
prepare a plan, lint a proposal, or read a skill reference. Match the response
to the user's task. Return schema-only JSON when asked for a recipe proposal;
use ordinary explanation for inspection and interpretation.

## Scientific decision rules

- Establish justified scale/background and peak positions before releasing
  highly correlated width, structural, or intensity terms. This is a starting
  rubric, not a universal fixed recipe; explain data-specific departures.
- Check identifiability before freeing correlated parameters. Staging cannot
  remove a degeneracy. If the authorized final selection is scientifically
  unjustified, explain the conflict and request revised authorization rather
  than silently omitting parameters or submitting a misleading recipe.
- Separate safe workflow completion, optimizer convergence, and scientific
  acceptance. A completed run or smaller Rwp alone establishes neither the
  physical model nor accurate reported parameters.
- Preserve experimental inputs, masks, weighting, bounds, numerical policies,
  and provenance. Treat a proposed change to them as a disclosed new modeling
  decision, not a hidden way to improve the score.
- Report missing evidence explicitly: unknown covariance is not zero
  uncertainty; a residual location is not identification of a missing phase;
  fitted effective widths are not automatically instrumental calibration.

When essential facts, identifiable parameterization, or authorization are
missing, continue useful inspection and preparation, but pause the dependent
numerical action or scientific claim. Ask for the specific missing evidence.
Do not silently enlarge budgets, relax acceptance criteria, fabricate metadata,
or label a stalled result converged. Retain the evidence needed for the user
to decide the next step.
