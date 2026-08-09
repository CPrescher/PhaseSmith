"""Small stateful facade for script-first structural Rietveld workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from ._native_persistence import load_native_rietveld_project
from .control import CancellationToken
from .persistence import PersistenceBundle, PersistenceError, load_bundle, save_bundle
from .radiation import MonochromaticRadiation, RadiationProbe
from .refinement import rietveld as native_rietveld
from .refinement.rietveld import (
    RietveldCalculationResult,
    RietveldCheckpoint,
    RietveldInput,
    RietveldOptions,
    RietveldResult,
    calculate,
    refine,
)
from .refinement.runtime import CheckpointCallback, RefinementLogger
from .refinement.workflow import (
    RietveldRecipe,
    RietveldWorkflowResult,
    intelligent_rietveld_recipe,
    run_rietveld_recipe,
)
from .reporting import write_rietveld_csv, write_rietveld_json


@dataclass(slots=True)
class RietveldProject:
    """Convenient mutable owner of one typed, resumable Rietveld request."""

    input: RietveldInput
    options: RietveldOptions = field(default_factory=RietveldOptions)
    checkpoint: RietveldCheckpoint | None = None
    last_result: RietveldResult | None = field(default=None, init=False)
    last_workflow: RietveldWorkflowResult | None = field(default=None, init=False)
    _cancellation: CancellationToken = field(default_factory=CancellationToken, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.input, RietveldInput):
            raise TypeError("input must be RietveldInput")
        if not isinstance(self.options, RietveldOptions):
            raise TypeError("options must be RietveldOptions")
        if self.checkpoint is not None and not isinstance(self.checkpoint, RietveldCheckpoint):
            raise TypeError("checkpoint must be RietveldCheckpoint or None")

    def calculate(self) -> RietveldCalculationResult:
        """Calculate from the last accepted state without changing it."""

        if self.checkpoint is None:
            experiment = self.input.experiment
            phases = self.input.phases
            background = self.input.background
        else:
            experiment = self.checkpoint.experiment or self.input.experiment
            phases = self.checkpoint.phases
            background = self.checkpoint.background
        return calculate(
            self.input.pattern,
            experiment,
            phases,
            background=background,
            support_fwhm=self.options.support_fwhm,
            execution=self.options.execution,
        )

    def refine(
        self,
        *,
        logger: RefinementLogger | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> RietveldResult:
        """Run or resume refinement and retain the last accepted checkpoint."""

        self._cancellation = CancellationToken()
        result = refine(
            self.input,
            self.options,
            checkpoint=self.checkpoint,
            cancellation=self._cancellation,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
        )
        self.checkpoint = result.checkpoint
        self.last_result = result
        self.last_workflow = None
        return result

    def stop(self, reason: str = "user_requested") -> bool:
        """Request cooperative stop at the next safe refinement boundary."""

        return self._cancellation.request(reason)

    def propose_intelligent_recipe(self) -> RietveldRecipe:
        """Return transparent staged advice without starting a refinement."""

        return intelligent_rietveld_recipe(self.input)

    def refine_recipe(
        self,
        recipe: RietveldRecipe,
        *,
        logger: RefinementLogger | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> RietveldWorkflowResult:
        """Run a caller-owned recipe and retain its last accepted physical state."""

        self._cancellation = CancellationToken()
        workflow = run_rietveld_recipe(
            self.input,
            recipe,
            options=self.options,
            cancellation=self._cancellation,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
        )
        result = workflow.final_result
        accepted = workflow.last_accepted_stage
        if accepted is not None:
            accepted_result = accepted.result
            self.input = replace(
                self.input,
                experiment=accepted_result.experiment,
                phases=accepted_result.phases,
                lattice_domains=accepted_result.checkpoint.lattice_domains,
                parameters=accepted_result.parameters,
                selection=accepted.stage.selection,
                background=accepted_result.background,
            )
        self.checkpoint = None
        self.last_result = result
        self.last_workflow = workflow
        return workflow

    def refine_intelligently(
        self,
        *,
        logger: RefinementLogger | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> RietveldWorkflowResult:
        """Plan, disclose, and run a cumulative recipe from authorized parameters."""

        return self.refine_recipe(
            self.propose_intelligent_recipe(),
            logger=logger,
            checkpoint_callback=checkpoint_callback,
        )

    def accept_result(self) -> None:
        """Promote the last result to the project input and clear restart state."""

        if self.last_result is None:
            raise ValueError("the project has no refinement result to accept")
        result = self.last_result
        self.input = replace(
            self.input,
            experiment=result.experiment,
            phases=result.phases,
            lattice_domains=result.checkpoint.lattice_domains,
            parameters=result.parameters,
            background=result.background,
        )
        self.checkpoint = None

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Persist the complete runnable project without pickle."""

        if (
            isinstance(self.input.experiment.radiation, MonochromaticRadiation)
            and (self.checkpoint is None or self.checkpoint._native is not None)
            and self.options.max_linearization_elements
            == RietveldOptions().max_linearization_elements
        ):
            try:
                request = native_rietveld._native_request(self.input, self.options)
            except TypeError:
                pass
            else:
                destination = Path(path).resolve()
                if destination.exists() and not destination.is_dir():
                    raise FileExistsError(
                        f"persistence path exists and is not a directory: {destination}"
                    )
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"persistence directory already exists: {destination}")
                probe = (
                    "xray"
                    if self.input.experiment.radiation.probe is RadiationProbe.X_RAY
                    else "neutron"
                )
                try:
                    saved = request.save_project(
                        str(destination),
                        "python-project",
                        0,
                        "Rietveld project",
                        "histogram",
                        "Observed pattern",
                        probe,
                        None if self.checkpoint is None else self.checkpoint._native,
                        overwrite,
                    )
                except ValueError as error:
                    raise PersistenceError(
                        f"cannot save native Rietveld project: {error}"
                    ) from error
                return Path(saved)
        return save_bundle(
            path,
            PersistenceBundle(
                pattern=self.input.pattern,
                experiment=self.input.experiment,
                rietveld_phases=self.input.phases,
                rietveld_domains=self.input.lattice_domains,
                rietveld_selection=self.input.selection,
                rietveld_options=self.options,
                rietveld_checkpoint=self.checkpoint,
                rietveld_background=self.input.background,
                parameters=self.input.parameters,
                constraints=self.input.constraints,
            ),
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, path: str | Path) -> RietveldProject:
        """Load a runnable project and optional continuation checkpoint."""

        source = Path(path).resolve()
        try:
            manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict) and "project" in manifest:
            try:
                input_data, options, checkpoint = load_native_rietveld_project(source)
            except (KeyError, OSError, TypeError, ValueError) as error:
                raise PersistenceError(f"cannot load native Rietveld project: {error}") from error
            return cls(input_data, options, checkpoint)
        bundle = load_bundle(path)
        return cls(
            bundle.to_rietveld_input(),
            RietveldOptions() if bundle.rietveld_options is None else bundle.rietveld_options,
            bundle.rietveld_checkpoint,
        )

    def write_reports(
        self,
        *,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Write requested JSON/CSV reports for the last result."""

        if self.last_result is None:
            raise ValueError("the project has no refinement result to report")
        json_result = (
            None if json_path is None else write_rietveld_json(self.last_result, json_path)
        )
        csv_result = (
            None
            if csv_path is None
            else write_rietveld_csv(self.last_result, self.input.pattern, csv_path)
        )
        return json_result, csv_result
