"""Explicit and advisory staged workflows above the general Rietveld solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..control import CancellationCallback
from . import rietveld
from .core import (
    AffineConstraint,
    Constraint,
    FixedConstraint,
    LinearConstraint,
    ParameterKey,
    ResidualOptions,
    TerminationReason,
    evaluate_residuals,
)
from .runtime import CheckpointCallback, RefinementLogger

RecipeMode = Literal["explicit", "intelligent"]


def _selection_record(selection: rietveld.RietveldParameterSelection) -> dict[str, object]:
    return {
        "phase_scale": selection.phase_scale,
        "lattice": selection.lattice,
        "coordinates": selection.coordinates,
        "occupancy": selection.occupancy,
        "u_iso": selection.u_iso,
        "sample_physics": selection.sample_physics,
        "instrument_parameters": list(selection.instrument_parameters),
        "background": selection.background,
    }


@dataclass(frozen=True, slots=True)
class RietveldStage:
    """One caller-selected active parameter set and its acceptance policy."""

    name: str
    selection: rietveld.RietveldParameterSelection
    rationale: tuple[str, ...]
    options: rietveld.RietveldOptions | None = None
    accepted_terminations: tuple[TerminationReason, ...] = (
        TerminationReason.CONVERGED,
        TerminationReason.STAGNATED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("stage name must be a non-empty trimmed string")
        if not isinstance(self.selection, rietveld.RietveldParameterSelection):
            raise TypeError("stage selection must be RietveldParameterSelection")
        reasons = tuple(self.rationale)
        if not reasons or any(not reason or reason != reason.strip() for reason in reasons):
            raise ValueError("stages require non-empty rationale strings")
        object.__setattr__(self, "rationale", reasons)
        if self.options is not None and not isinstance(self.options, rietveld.RietveldOptions):
            raise TypeError("stage options must be RietveldOptions or None")
        accepted = tuple(self.accepted_terminations)
        if not accepted or len(set(accepted)) != len(accepted):
            raise ValueError("accepted stage terminations must be non-empty and unique")
        if any(not isinstance(reason, TerminationReason) for reason in accepted):
            raise TypeError("accepted stage terminations must be TerminationReason values")
        object.__setattr__(self, "accepted_terminations", accepted)

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "selection": _selection_record(self.selection),
            "rationale": list(self.rationale),
            "accepted_terminations": [reason.value for reason in self.accepted_terminations],
            "custom_options": self.options is not None,
        }


@dataclass(frozen=True, slots=True)
class RietveldRecipe:
    """An explicit sequence outside the numerical solver."""

    name: str
    stages: tuple[RietveldStage, ...]
    mode: RecipeMode = "explicit"
    planner_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("recipe name must be a non-empty trimmed string")
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, RietveldStage) for stage in stages):
            raise TypeError("recipes require at least one RietveldStage")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("recipe stage names must be unique")
        object.__setattr__(self, "stages", stages)
        if self.mode not in {"explicit", "intelligent"}:
            raise ValueError("recipe mode must be explicit or intelligent")
        notes = tuple(self.planner_notes)
        if any(not note or note != note.strip() for note in notes):
            raise ValueError("planner notes must be non-empty trimmed strings")
        object.__setattr__(self, "planner_notes", notes)

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "planner_notes": list(self.planner_notes),
            "stages": [stage.to_record() for stage in self.stages],
        }


@dataclass(frozen=True, slots=True)
class RietveldStageResult:
    """Auditable outcome from one recipe stage."""

    stage: RietveldStage
    starting_rwp: float
    result: rietveld.RietveldResult
    accepted: bool

    def to_record(self) -> dict[str, object]:
        return {
            "stage": self.stage.to_record(),
            "starting_rwp": self.starting_rwp,
            "final_rwp": self.result.metrics.rwp,
            "accepted": self.accepted,
            "termination": self.result.termination_reason.value,
            "termination_message": self.result.termination_message,
            "iterations": len(self.result.history),
            "evaluations": self.result.evaluations,
            "active_parameters": [spec.key.label for spec in self.result.parameters.specs],
        }


@dataclass(frozen=True, slots=True)
class RietveldWorkflowResult:
    """Complete or safely stopped staged workflow."""

    recipe: RietveldRecipe
    stages: tuple[RietveldStageResult, ...]
    completed: bool

    def __post_init__(self) -> None:
        values = tuple(self.stages)
        if not values or any(not isinstance(stage, RietveldStageResult) for stage in values):
            raise TypeError("workflow results require completed stage records")
        if len(values) > len(self.recipe.stages):
            raise ValueError("workflow contains more results than recipe stages")
        expected_completed = len(values) == len(self.recipe.stages) and values[-1].accepted
        if self.completed != expected_completed:
            raise ValueError("workflow completed flag does not match its stage results")
        object.__setattr__(self, "stages", values)

    @property
    def final_result(self) -> rietveld.RietveldResult:
        """Return the final attempted stage, including a rejected stopping stage."""

        return self.stages[-1].result

    @property
    def last_accepted_stage(self) -> RietveldStageResult | None:
        """Return the last state that the recipe acceptance policy permits promoting."""

        return next((stage for stage in reversed(self.stages) if stage.accepted), None)

    def to_record(self) -> dict[str, object]:
        return {
            "recipe": self.recipe.to_record(),
            "completed": self.completed,
            "stages": [stage.to_record() for stage in self.stages],
        }


def _constraint_keys(constraint: Constraint) -> tuple[ParameterKey, ...]:
    if isinstance(constraint, FixedConstraint):
        return (constraint.target,)
    if isinstance(constraint, AffineConstraint):
        return (constraint.target, constraint.source)
    if isinstance(constraint, LinearConstraint):
        return (constraint.target, *(key for key, _ in constraint.terms))
    raise TypeError("unsupported Rietveld constraint")


def _stage_constraints(
    constraints: tuple[Constraint, ...], keys: tuple[ParameterKey, ...]
) -> tuple[Constraint, ...]:
    available = set(keys)
    selected = []
    for constraint in constraints:
        involved = _constraint_keys(constraint)
        if involved[0] not in available:
            continue
        missing = [key.label for key in involved if key not in available]
        if missing:
            raise ValueError(
                f"stage selects constraint target {involved[0].label} without "
                f"dependencies {', '.join(missing)}"
            )
        selected.append(constraint)
    return tuple(selected)


def _selection_subset(
    selected: rietveld.RietveldParameterSelection,
    maximum: rietveld.RietveldParameterSelection,
) -> bool:
    families = (
        "phase_scale",
        "lattice",
        "coordinates",
        "occupancy",
        "u_iso",
        "sample_physics",
        "background",
    )
    return all(not getattr(selected, name) or getattr(maximum, name) for name in families) and set(
        selected.instrument_parameters
    ).issubset(maximum.instrument_parameters)


def _stage_input(
    input_data: rietveld.RietveldInput,
    selection: rietveld.RietveldParameterSelection,
) -> rietveld.RietveldInput:
    parameters = rietveld.build_parameter_set(
        input_data.phases,
        input_data.lattice_domains,
        selection,
        experiment=input_data.experiment,
        background=input_data.background,
    )
    constraints = _stage_constraints(input_data.constraints, parameters.keys)
    return replace(
        input_data,
        parameters=parameters,
        constraints=constraints,
        selection=selection,
    )


def _accepted_state(
    input_data: rietveld.RietveldInput,
    result: rietveld.RietveldResult,
) -> rietveld.RietveldInput:
    parameters = rietveld.build_parameter_set(
        result.phases,
        result.checkpoint.lattice_domains,
        input_data.selection,
        experiment=result.experiment,
        background=result.background,
    )
    return replace(
        input_data,
        experiment=result.experiment,
        phases=result.phases,
        lattice_domains=result.checkpoint.lattice_domains,
        parameters=parameters,
        background=result.background,
    )


def run_rietveld_recipe(
    input_data: rietveld.RietveldInput,
    recipe: RietveldRecipe,
    *,
    options: rietveld.RietveldOptions | None = None,
    cancellation: CancellationCallback | None = None,
    logger: RefinementLogger | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> RietveldWorkflowResult:
    """Run explicit stages, preserving only accepted physical states between them."""

    validate_rietveld_recipe(input_data, recipe)
    selected_options = rietveld.RietveldOptions() if options is None else options
    if not isinstance(selected_options, rietveld.RietveldOptions):
        raise TypeError("options must be RietveldOptions or None")

    current = input_data
    initial_calculation = rietveld.calculate(
        current.pattern,
        current.experiment,
        current.phases,
        background=current.background,
        support_fwhm=selected_options.support_fwhm,
        execution=selected_options.execution,
    )
    current_rwp = evaluate_residuals(
        current.pattern,
        initial_calculation.y,
        ResidualOptions(selected_options.use_uncertainty, 0),
    ).rwp
    results = []
    for index, stage in enumerate(recipe.stages):
        staged_input = _stage_input(current, stage.selection)
        stage_options = selected_options if stage.options is None else stage.options
        if index + 1 < len(recipe.stages) and stage_options.estimate_covariance:
            stage_options = replace(stage_options, estimate_covariance=False)
        result = rietveld.refine(
            staged_input,
            stage_options,
            cancellation=cancellation,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
        )
        accepted = result.termination_reason in stage.accepted_terminations
        results.append(RietveldStageResult(stage, current_rwp, result, accepted))
        if not accepted:
            break
        current = _accepted_state(current, result)
        current_rwp = result.metrics.rwp
    return RietveldWorkflowResult(
        recipe,
        tuple(results),
        len(results) == len(recipe.stages) and results[-1].accepted,
    )


def validate_rietveld_recipe(
    input_data: rietveld.RietveldInput,
    recipe: RietveldRecipe,
) -> None:
    """Validate every recipe stage without evaluating the numerical objective."""

    if not isinstance(input_data, rietveld.RietveldInput):
        raise TypeError("input_data must be RietveldInput")
    if not isinstance(recipe, RietveldRecipe):
        raise TypeError("recipe must be RietveldRecipe")
    for stage in recipe.stages:
        if not _selection_subset(stage.selection, input_data.selection):
            raise ValueError(
                f"stage {stage.name!r} selects parameters outside the input's maximum selection"
            )
        # Build every stage before numerical work. This validates the complete
        # caller-owned parameter and constraint contract without allowing an
        # early stage's filtered view to erase constraints needed later.
        _stage_input(input_data, stage.selection)


def _selection(
    maximum: rietveld.RietveldParameterSelection,
    *,
    phase_scale: bool = False,
    lattice: bool = False,
    coordinates: bool = False,
    occupancy: bool = False,
    u_iso: bool = False,
    sample_physics: bool = False,
    instrument_parameters: tuple[str, ...] = (),
    background: bool = False,
) -> rietveld.RietveldParameterSelection:
    return rietveld.RietveldParameterSelection(
        phase_scale=phase_scale and maximum.phase_scale,
        lattice=lattice and maximum.lattice,
        coordinates=coordinates and maximum.coordinates,
        occupancy=occupancy and maximum.occupancy,
        u_iso=u_iso and maximum.u_iso,
        sample_physics=sample_physics and maximum.sample_physics,
        instrument_parameters=tuple(
            name for name in maximum.instrument_parameters if name in instrument_parameters
        ),
        background=background and maximum.background,
    )


def _selection_has_parameters(selection: rietveld.RietveldParameterSelection) -> bool:
    return any(
        (
            selection.phase_scale,
            selection.lattice,
            selection.coordinates,
            selection.occupancy,
            selection.u_iso,
            selection.sample_physics,
            bool(selection.instrument_parameters),
            selection.background,
        )
    )


def intelligent_rietveld_recipe(
    input_data: rietveld.RietveldInput,
    *,
    name: str = "intelligent-cumulative",
) -> RietveldRecipe:
    """Propose a transparent cumulative sequence from caller-authorized families."""

    if not isinstance(input_data, rietveld.RietveldInput):
        raise TypeError("input_data must be RietveldInput")
    maximum = input_data.selection
    stages = []
    current: rietveld.RietveldParameterSelection | None = None

    def append_stage(
        stage_name: str,
        selection: rietveld.RietveldParameterSelection,
        *rationale: str,
    ) -> None:
        nonlocal current
        if selection == current or not _selection_has_parameters(selection):
            return
        stages.append(RietveldStage(stage_name, selection, tuple(rationale)))
        current = selection

    scale_background = _selection(
        maximum,
        phase_scale=True,
        background=True,
    )
    append_stage(
        "scale_background",
        scale_background,
        "Establish intensity scale and any differentiable background before correlated terms.",
    )

    positions = _selection(
        maximum,
        phase_scale=True,
        background=True,
        lattice=True,
        instrument_parameters=(
            "wavelength_angstrom",
            "zero_shift_deg",
            "sample_displacement_mm",
            "displace_x_micrometre",
            "displace_y_micrometre",
        ),
    )
    append_stage(
        "positions",
        positions,
        "Align reflection positions before refining peak widths or structural intensities.",
    )

    structure = _selection(
        maximum,
        phase_scale=True,
        background=True,
        lattice=True,
        coordinates=True,
        occupancy=True,
        u_iso=True,
        instrument_parameters=(
            "wavelength_angstrom",
            "zero_shift_deg",
            "sample_displacement_mm",
            "displace_x_micrometre",
            "displace_y_micrometre",
        ),
    )
    append_stage(
        "structure",
        structure,
        "Stabilize relative structural intensities after peak centers are aligned.",
        "Delay profile widths so they cannot initially mask intensity-model errors.",
    )

    append_stage(
        "final_polish",
        maximum,
        "Release authorized profile and sample broadening after positions and intensities.",
        "Finish with every caller-authorized parameter active together.",
    )
    if not stages:
        stages.append(
            RietveldStage(
                "evaluate_only",
                maximum,
                ("No refinable parameter family was authorized; evaluate the supplied state.",),
            )
        )
    notes = [
        "This plan is advisory workflow orchestration; the Rietveld solver remains general.",
        (
            "Every stage is cumulative and limited to parameter families authorized "
            "by input.selection."
        ),
    ]
    if not maximum.background:
        notes.append(
            "No differentiable background was authorized; the supplied background stays fixed."
        )
    if not maximum.lattice:
        notes.append("Lattice refinement was unavailable or disabled and was not proposed.")
    if {
        "displace_x_micrometre",
        "displace_y_micrometre",
    }.intersection(maximum.instrument_parameters):
        notes.append(
            "Debye-Scherrer X/Y displacement was authorized and is proposed in the "
            "position-alignment stage before profile widths."
        )
    if maximum.occupancy:
        notes.append(
            "Occupancy is delayed to the structural stage because it is strongly scale-correlated."
        )
    return RietveldRecipe(name, tuple(stages), "intelligent", tuple(notes))
