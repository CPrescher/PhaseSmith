"""Small stateful facade for script-first structural Rietveld workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .control import CancellationToken
from .persistence import PersistenceBundle, load_bundle, save_bundle
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
from .reporting import write_rietveld_csv, write_rietveld_json


@dataclass(slots=True)
class RietveldProject:
    """Convenient mutable owner of one typed, resumable Rietveld request."""

    input: RietveldInput
    options: RietveldOptions = field(default_factory=RietveldOptions)
    checkpoint: RietveldCheckpoint | None = None
    last_result: RietveldResult | None = field(default=None, init=False)
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
        return result

    def stop(self, reason: str = "user_requested") -> bool:
        """Request cooperative stop at the next safe refinement boundary."""

        return self._cancellation.request(reason)

    def accept_result(self) -> None:
        """Promote the last result to the project input and clear restart state."""

        if self.last_result is None:
            raise ValueError("the project has no refinement result to accept")
        result = self.last_result
        self.input = replace(
            self.input,
            experiment=result.experiment,
            phases=result.phases,
            parameters=result.parameters,
            background=result.background,
        )
        self.checkpoint = None

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Persist the complete runnable project without pickle."""

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
