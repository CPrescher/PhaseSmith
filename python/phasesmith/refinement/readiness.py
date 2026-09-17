"""Deterministic pre-refinement review of structural Rietveld requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..intensity_corrections import (
    BraggBrentanoPolarizedLp,
    BraggBrentanoUnpolarizedLp,
    ConstantWavelengthNeutronLorentz,
    NeutralIntegratedIntensityCorrection,
    TimeOfFlightNeutronLorentz,
)
from ..radiation import (
    BraggBrentanoGeometry,
    ComponentRadiation,
    DebyeScherrerGeometry,
    RadiationProbe,
)
from .rietveld import RietveldInput

ReadinessSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class RietveldReadinessDiagnostic:
    """One machine-readable finding from a pre-refinement review."""

    severity: ReadinessSeverity
    code: str
    message: str
    phase_id: str | None = None
    source_tag: str | None = None
    source_row: int | None = None

    def __post_init__(self) -> None:
        if self.severity not in ("info", "warning", "error"):
            raise ValueError("readiness severity must be info, warning, or error")
        if not self.code or self.code != self.code.strip():
            raise ValueError("readiness code must be a non-empty trimmed string")
        if not self.message or self.message != self.message.strip():
            raise ValueError("readiness message must be a non-empty trimmed string")
        if self.phase_id is not None and (
            not self.phase_id or self.phase_id != self.phase_id.strip()
        ):
            raise ValueError("phase_id must be a non-empty trimmed string or None")
        if self.source_row is not None and self.source_row < 0:
            raise ValueError("source_row must be non-negative")

    def to_record(self) -> dict[str, object]:
        """Return deterministic JSON-compatible fields."""

        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "phase_id": self.phase_id,
            "source_tag": self.source_tag,
            "source_row": self.source_row,
        }


@dataclass(frozen=True, slots=True)
class RietveldReadinessReport:
    """Immutable diagnostics for one already validated Rietveld request."""

    diagnostics: tuple[RietveldReadinessDiagnostic, ...]

    def __post_init__(self) -> None:
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, RietveldReadinessDiagnostic) for item in diagnostics):
            raise TypeError("diagnostics must contain RietveldReadinessDiagnostic values")
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def has_warnings(self) -> bool:
        """Return whether any finding requires user review."""

        return any(item.severity == "warning" for item in self.diagnostics)

    @property
    def has_errors(self) -> bool:
        """Return whether imported structure diagnostics contain an error."""

        return any(item.severity == "error" for item in self.diagnostics)

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible report."""

        return {
            "has_warnings": self.has_warnings,
            "has_errors": self.has_errors,
            "diagnostics": [item.to_record() for item in self.diagnostics],
        }


def _diagnostic(
    severity: ReadinessSeverity,
    code: str,
    message: str,
    *,
    phase_id: str | None = None,
) -> RietveldReadinessDiagnostic:
    return RietveldReadinessDiagnostic(severity, code, message, phase_id)


def _review_structure(input_data: RietveldInput) -> list[RietveldReadinessDiagnostic]:
    diagnostics: list[RietveldReadinessDiagnostic] = []
    for phase in input_data.phases:
        structure = phase.structure
        for imported in structure.diagnostics:
            diagnostics.append(
                RietveldReadinessDiagnostic(
                    imported.severity,
                    f"structure.{imported.code}",
                    imported.message,
                    phase.phase_id,
                    imported.tag,
                    imported.row,
                )
            )
        source = structure.source
        if source is None:
            diagnostics.append(
                _diagnostic(
                    "warning",
                    "structure.source_missing",
                    "The structure has no retained source provenance; confirm its setting "
                    "and conversion history.",
                    phase_id=phase.phase_id,
                )
            )
        else:
            source_name = "" if source.source_name is None else f" from {source.source_name!r}"
            diagnostics.append(
                _diagnostic(
                    "info",
                    "structure.source",
                    f"Structure block {source.block_name!r}{source_name} was read as "
                    f"{source.format} by {source.backend} {source.backend_version}.",
                    phase_id=phase.phase_id,
                )
            )
        metadata = structure.metadata
        symmetry_source = metadata.get("symmetry_source")
        identifiers = ", ".join(
            f"{label}={metadata[key]!r}"
            for key, label in (
                ("space_group_hall", "Hall"),
                ("space_group_hm", "Hermann-Mauguin"),
                ("space_group_number", "number"),
            )
            if key in metadata
        )
        if symmetry_source is not None:
            suffix = "" if not identifiers else f" ({identifiers})"
            diagnostics.append(
                _diagnostic(
                    "warning" if symmetry_source == "assumed_p1" else "info",
                    "structure.symmetry_assumed_p1"
                    if symmetry_source == "assumed_p1"
                    else "structure.symmetry",
                    f"Symmetry was resolved from {symmetry_source!r}{suffix}.",
                    phase_id=phase.phase_id,
                )
            )
        diagnostics.append(
            _diagnostic(
                "info",
                "structure.contents",
                f"The phase contains {len(structure.sites)} independent sites and "
                f"{phase.reflections.reflection_count} reflection families.",
                phase_id=phase.phase_id,
            )
        )
    return diagnostics


def _review_models(input_data: RietveldInput) -> list[RietveldReadinessDiagnostic]:
    experiment = input_data.experiment
    radiation = experiment.radiation
    probe = radiation.probe
    if isinstance(radiation, ComponentRadiation):
        radiation_message = (
            f"The active {probe.value} radiation model has "
            f"{radiation.components.component_count} fixed wavelength components."
        )
    else:
        radiation_message = (
            f"The active radiation model is monochromatic {probe.value} at "
            f"{radiation.wavelength_angstrom:.12g} angstrom."
        )
    diagnostics = [_diagnostic("info", "experiment.radiation", radiation_message)]

    geometry = experiment.geometry
    if geometry is None:
        diagnostics.append(
            _diagnostic(
                "warning",
                "experiment.geometry_missing",
                "No specimen geometry is configured; specimen-displacement corrections "
                "are inactive.",
            )
        )
    elif isinstance(geometry, BraggBrentanoGeometry):
        diagnostics.append(
            _diagnostic(
                "info",
                "experiment.geometry",
                "The active specimen geometry is symmetric Bragg-Brentano.",
            )
        )
    elif isinstance(geometry, DebyeScherrerGeometry):
        diagnostics.append(
            _diagnostic(
                "info",
                "experiment.geometry",
                "The active specimen geometry is Debye-Scherrer capillary.",
            )
        )
    if experiment.axial_geometry is not None:
        diagnostics.append(
            _diagnostic(
                "info",
                "experiment.axial_geometry",
                "An FCJ axial-divergence geometry is active.",
            )
        )

    for phase in input_data.phases:
        descriptor = phase.scattering.descriptor
        diagnostics.append(
            _diagnostic(
                "info",
                "phase.scattering",
                f"The active scattering provider is {descriptor.provider_id} "
                f"{descriptor.provider_version} for {descriptor.probe} data.",
                phase_id=phase.phase_id,
            )
        )
        expected_probe = "xray" if probe is RadiationProbe.X_RAY else "neutron"
        if descriptor.probe != expected_probe:
            diagnostics.append(
                _diagnostic(
                    "warning",
                    "phase.scattering_probe_mismatch",
                    f"The {descriptor.probe} scattering provider conflicts with the "
                    f"experiment's {probe.value} radiation.",
                    phase_id=phase.phase_id,
                )
            )

        correction = phase.intensity_correction
        diagnostics.append(
            _diagnostic(
                "warning"
                if isinstance(correction, NeutralIntegratedIntensityCorrection)
                else "info",
                "phase.neutral_intensity_correction"
                if isinstance(correction, NeutralIntegratedIntensityCorrection)
                else "phase.intensity_correction",
                "The neutral integrated-intensity correction is active; confirm that the "
                "data reduction already includes the required geometry factors."
                if isinstance(correction, NeutralIntegratedIntensityCorrection)
                else f"The active integrated-intensity correction is {type(correction).__name__}.",
                phase_id=phase.phase_id,
            )
        )
        if isinstance(correction, (BraggBrentanoUnpolarizedLp, BraggBrentanoPolarizedLp)):
            if probe is not RadiationProbe.X_RAY:
                diagnostics.append(
                    _diagnostic(
                        "warning",
                        "phase.intensity_correction_probe_mismatch",
                        "A Bragg-Brentano X-ray LP correction is paired with neutron radiation.",
                        phase_id=phase.phase_id,
                    )
                )
            if geometry is None:
                diagnostics.append(
                    _diagnostic(
                        "warning",
                        "phase.intensity_correction_geometry_unconfirmed",
                        "A Bragg-Brentano LP correction is active, but no specimen geometry "
                        "is configured to confirm that convention.",
                        phase_id=phase.phase_id,
                    )
                )
            elif not isinstance(geometry, BraggBrentanoGeometry):
                diagnostics.append(
                    _diagnostic(
                        "warning",
                        "phase.intensity_correction_geometry_mismatch",
                        "A Bragg-Brentano LP correction conflicts with the configured "
                        "specimen geometry.",
                        phase_id=phase.phase_id,
                    )
                )
        elif isinstance(correction, (ConstantWavelengthNeutronLorentz, TimeOfFlightNeutronLorentz)):
            if probe is not RadiationProbe.NEUTRON:
                diagnostics.append(
                    _diagnostic(
                        "warning",
                        "phase.intensity_correction_probe_mismatch",
                        "A neutron Lorentz correction is paired with X-ray radiation.",
                        phase_id=phase.phase_id,
                    )
                )
        correction_wavelength = (
            correction.wavelength_angstrom
            if isinstance(
                correction,
                (
                    BraggBrentanoUnpolarizedLp,
                    BraggBrentanoPolarizedLp,
                    ConstantWavelengthNeutronLorentz,
                ),
            )
            else None
        )
        if (
            correction_wavelength is not None
            and correction_wavelength != radiation.wavelength_angstrom
        ):
            diagnostics.append(
                _diagnostic(
                    "warning",
                    "phase.intensity_correction_wavelength_mismatch",
                    "The intensity-correction wavelength does not match the experiment's "
                    "reference wavelength.",
                    phase_id=phase.phase_id,
                )
            )
        if isinstance(correction, TimeOfFlightNeutronLorentz):
            diagnostics.append(
                _diagnostic(
                    "warning",
                    "phase.intensity_correction_experiment_mismatch",
                    "A time-of-flight Lorentz correction is paired with a "
                    "constant-wavelength experiment.",
                    phase_id=phase.phase_id,
                )
            )
    return diagnostics


def _review_selection(input_data: RietveldInput) -> list[RietveldReadinessDiagnostic]:
    selection = input_data.selection
    diagnostics: list[RietveldReadinessDiagnostic] = []
    if selection.phase_scale and selection.occupancy:
        diagnostics.append(
            _diagnostic(
                "warning",
                "selection.scale_occupancy_correlation",
                "Phase scale and site occupancy are both selected; constrain composition "
                "or stage them separately to avoid intensity-scale ambiguity.",
            )
        )
    if selection.lattice and "wavelength_angstrom" in selection.instrument_parameters:
        diagnostics.append(
            _diagnostic(
                "warning",
                "selection.lattice_wavelength_correlation",
                "Lattice and wavelength parameters are both selected and can be strongly "
                "correlated.",
            )
        )
    position_terms = set(selection.instrument_parameters)
    if "zero_shift_deg" in position_terms and position_terms.intersection(
        {
            "sample_displacement_mm",
            "displace_x_micrometre",
            "displace_y_micrometre",
        }
    ):
        diagnostics.append(
            _diagnostic(
                "warning",
                "selection.zero_displacement_correlation",
                "Zero shift and specimen displacement are both selected and can be "
                "strongly correlated.",
            )
        )
    return diagnostics


def review_rietveld_input(input_data: RietveldInput) -> RietveldReadinessReport:
    """Review conversion provenance and active models without changing the request."""

    if not isinstance(input_data, RietveldInput):
        raise TypeError("input_data must be RietveldInput")
    diagnostics = [
        *_review_structure(input_data),
        *_review_models(input_data),
        *_review_selection(input_data),
    ]
    if input_data.empirical_gaussian is not None:
        convention = input_data.empirical_gaussian
        diagnostics.append(
            RietveldReadinessDiagnostic(
                "info",
                "profile.empirical_gaussian",
                f"RMS strain of {convention.reference_phase_id} is fixed by convention to "
                f"{convention.reference_rms_microstrain:g}. Instrument Gaussian widths and "
                "sample strains are convention-dependent, not independently measured.",
                phase_id=convention.reference_phase_id,
            )
        )
    return RietveldReadinessReport(tuple(diagnostics))
