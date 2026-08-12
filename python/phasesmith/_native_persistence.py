"""Python dataclass reconstruction for the Rust-owned native project format."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _core
from .crystallography import UnitCell
from .execution import ExecutionPolicy
from .extensions import CompositePhysicsProvider
from .instrument import ConstantWavelengthInstrument, FcjGeometry
from .intensity_corrections import (
    BraggBrentanoPolarizedLp,
    BraggBrentanoUnpolarizedLp,
    ConstantWavelengthNeutronLorentz,
    NeutralIntegratedIntensityCorrection,
    TimeOfFlightNeutronLorentz,
)
from .pattern import PowderPattern
from .phase import ReciprocalMetric, RietveldPhase, StructuralReflectionBatch
from .radiation import (
    BraggBrentanoGeometry,
    ConstantWavelengthExperiment,
    DebyeScherrerGeometry,
    MonochromaticRadiation,
    RadiationProbe,
)
from .refinement.background import (
    AmorphousBackground,
    AmorphousPeak,
    ChebyshevBackground,
    CompositeBackground,
    DifferentiableBackground,
    PointBackground,
    PolynomialBackground,
)
from .refinement.core import (
    AffineConstraint,
    Bounds,
    Constraint,
    FixedConstraint,
    LinearConstraint,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
)
from .refinement.lattice import (
    CwStructuralReflectionDomain,
    LatticeParameterBounds,
    LatticeParameterization,
)
from .refinement.rietveld import (
    RietveldCheckpoint,
    RietveldInput,
    RietveldIterationRecord,
    RietveldOptions,
    RietveldParameterChange,
    RietveldParameterSelection,
    _apply_parameter_values,
    _apply_profile_background_values,
    build_parameter_set,
)
from .refinement.runtime import RefinementLimits
from .sample import (
    IsotropicLorentzianMicrostrainBroadening,
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    MarchDollasePreferredOrientation,
)
from .scattering import (
    NeutronNuclear,
    XrayFixedDispersion,
    XrayNonResonant,
    neutron_species_metadata,
    xray_species_metadata,
)
from .structure import AnisotropicDisplacement, AtomSite, CrystalStructure
from .symmetry import SpaceGroup, SymmetryOperation

MANIFEST_NAME = "manifest.json"
ARCHIVE_NAME = "arrays.npz"
_CHARGED_XRAY_KEY = re.compile(r"^(?P<element>[A-Z][a-z]?)(?P<charge>[1-9][0-9]*)(?P<sign>[+-])$")
_SPECIAL_XRAY_ELEMENTS = {"Cval": "C", "Siva": "Si"}


def load_native_rietveld_project(
    path: str | Path,
) -> tuple[RietveldInput, RietveldOptions, RietveldCheckpoint | None]:
    """Load one scripting facade through the validated Rust native codec."""

    source = Path(path).resolve()
    stored = _core._StoredRietveldProject.load(str(source))
    manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format_version") not in (2, 3, 4):
        raise ValueError("RietveldProject requires native project format 2, 3, or 4")
    project = manifest["project"]
    analyses = manifest["rietveld_analyses"]
    if len(project["histograms"]) != 1 or len(analyses) != 1:
        raise ValueError("the Python RietveldProject facade requires exactly one analysis")
    histogram = project["histograms"][0]
    analysis = analyses[0]
    if analysis["histogram_id"] != histogram["histogram_id"]:
        raise ValueError("native analysis and histogram IDs differ")
    with np.load(source / ARCHIVE_NAME, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    pattern = _pattern(histogram["pattern"], arrays)
    experiment = _experiment(histogram["experiment"])
    phase_records = {item["phase_id"]: item for item in project["phases"]}
    phases = tuple(
        _phase(phase_records[item["phase_id"]], item, arrays) for item in analysis["phases"]
    )
    domains = tuple(
        _domain(item["domain"], phase)
        for item, phase in zip(analysis["phases"], phases, strict=True)
    )
    background = _background(analysis["background"])
    selection = _selection(analysis["selection"])
    constraints = tuple(_constraint(item) for item in analysis["constraints"])
    parameters = build_parameter_set(
        phases,
        domains,
        selection,
        experiment=experiment,
        background=background,
    )
    input_data = RietveldInput(
        pattern,
        experiment,
        phases,
        domains,
        parameters,
        constraints,
        selection,
        background,
    )
    options = _options(analysis["options"], analysis["covariance"])
    checkpoint = _checkpoint(
        analysis["checkpoint"],
        input_data,
        stored.checkpoint(histogram["histogram_id"]),
    )
    return input_data, options, checkpoint


def _array(
    reference: dict[str, str] | None,
    arrays: dict[str, NDArray[np.generic]],
) -> NDArray[np.generic] | None:
    return None if reference is None else arrays[reference["array"]]


def _pattern(record: dict[str, Any], arrays: dict[str, NDArray[np.generic]]) -> PowderPattern:
    return PowderPattern(
        _array(record["x_deg"], arrays),
        observed_y=_array(record["observed_y"], arrays),
        uncertainty=_array(record["uncertainty"], arrays),
        mask=_array(record["mask"], arrays),
        background=_array(record["background_y"], arrays),
    )


def _experiment(record: dict[str, Any]) -> ConstantWavelengthExperiment:
    instrument = ConstantWavelengthInstrument(**record["instrument"])
    radiation_record = record["radiation"]
    probe = RadiationProbe.X_RAY if radiation_record["probe"] == "xray" else RadiationProbe.NEUTRON
    radiation = MonochromaticRadiation(probe, float(radiation_record["wavelength_angstrom"]))
    correction = record["position_correction"]
    bragg = correction["bragg_brentano_mm"]
    debye = correction["debye_scherrer_micrometre"]
    geometry = None
    if bragg is not None:
        geometry = BraggBrentanoGeometry(float(bragg[1]), float(bragg[0]))
    elif debye is not None:
        geometry = DebyeScherrerGeometry(float(debye[2]), float(debye[0]), float(debye[1]))
    axial_record = record["axial_geometry"]
    axial = None if axial_record is None else FcjGeometry(**axial_record)
    return ConstantWavelengthExperiment(
        radiation,
        instrument,
        float(correction["zero_shift_deg"]),
        geometry,
        axial,
    )


def _phase(
    record: dict[str, Any],
    state: dict[str, Any],
    arrays: dict[str, NDArray[np.generic]],
) -> RietveldPhase:
    definition = record["definition"]
    operations = tuple(
        SymmetryOperation(
            item["rotation"],
            tuple(Fraction(int(value[0]), int(value[1])) for value in item["translation"]),
        )
        for item in definition["operations"]
    )
    space_group = SpaceGroup(operations)
    cell = UnitCell(*map(float, definition["cell"]))
    fractional_xyz = _array(definition["fractional_xyz"], arrays)
    occupancy = _array(definition["occupancy"], arrays)
    u_iso = _array(definition["u_iso_angstrom2"], arrays)
    anisotropic_mask = _array(definition["anisotropic_mask"], arrays)
    u_aniso = _array(definition["u_aniso_cif_angstrom2"], arrays)
    sites = []
    scattering_model = definition["scattering_model"]
    for index, site_id in enumerate(state["site_ids"]):
        species = str(definition["scattering_species"][index])
        element, charge, isotope = _species_identity(species, scattering_model)
        anisotropic = (
            AnisotropicDisplacement(tuple(map(float, u_aniso[index])), "U_cif")
            if bool(anisotropic_mask[index])
            else None
        )
        sites.append(
            AtomSite(
                site_id=site_id,
                source_label=site_id,
                type_symbol=species,
                element_symbol=element,
                fractional_xyz=tuple(map(float, fractional_xyz[index])),
                occupancy=float(occupancy[index]),
                u_iso_angstrom2=float(u_iso[index]),
                anisotropic_displacement=anisotropic,
                charge=charge,
                isotope=isotope,
            )
        )
    structure = CrystalStructure(record["phase_id"], record["name"], cell, space_group, sites)
    hkl = _array(definition["hkl"], arrays)
    multiplicity = _array(definition["multiplicity"], arrays)
    reflection_ids = tuple(f"hkl:{int(row[0])},{int(row[1])},{int(row[2])}" for row in hkl)
    reflections = StructuralReflectionBatch(reflection_ids, hkl, multiplicity)
    scattering = _scattering(definition, arrays)
    correction = _correction(definition["correction_model"])
    physics = _sample_physics(state["sample_physics"], cell)
    return RietveldPhase(
        record["phase_id"],
        record["name"],
        structure,
        reflections,
        scattering,
        correction,
        float(definition["scale"]),
        physics,
        float(definition["coordinate_tolerance"]),
    )


def _species_identity(species: str, scattering_model: str) -> tuple[str, int | None, int | None]:
    if scattering_model == "neutron_nuclear":
        metadata = neutron_species_metadata(species)
        if metadata is None:
            raise ValueError(f"unknown native neutron species {species!r}")
        element = species.split("-", 1)[0]
        return element, None, metadata.isotope
    metadata = xray_species_metadata(species)
    if metadata is None:
        raise ValueError(f"unknown native X-ray species {species!r}")
    charged = _CHARGED_XRAY_KEY.fullmatch(species)
    if charged is not None:
        magnitude = int(charged.group("charge"))
        charge = magnitude if charged.group("sign") == "+" else -magnitude
        return charged.group("element"), charge, None
    return _SPECIAL_XRAY_ELEMENTS.get(species, species), None, None


def _scattering(
    definition: dict[str, Any], arrays: dict[str, NDArray[np.generic]]
) -> XrayNonResonant | XrayFixedDispersion | NeutronNuclear:
    if definition["scattering_model"] == "neutron_nuclear":
        return NeutronNuclear()
    real = _array(definition["scattering_real_offset"], arrays)
    imag = _array(definition["scattering_imag_offset"], arrays)
    if real is None and imag is None:
        return XrayNonResonant()
    if real is None or imag is None:
        raise ValueError("native X-ray dispersion offsets are incomplete")
    corrections: dict[str, complex] = {}
    for species, real_value, imag_value in zip(
        definition["scattering_species"], real, imag, strict=True
    ):
        element, _, _ = _species_identity(str(species), "xray_non_resonant")
        correction = complex(float(real_value), float(imag_value))
        previous = corrections.setdefault(element, correction)
        if previous != correction:
            raise ValueError(f"native X-ray dispersion differs between {element} sites")
    return XrayFixedDispersion(corrections)


def _correction(record: dict[str, Any]) -> object:
    kind = record["type"]
    if kind == "neutral":
        return NeutralIntegratedIntensityCorrection()
    if kind == "bragg_brentano_unpolarized_lp":
        return BraggBrentanoUnpolarizedLp(float(record["wavelength_angstrom"]))
    if kind == "bragg_brentano_polarized_lp":
        return BraggBrentanoPolarizedLp(
            float(record["wavelength_angstrom"]), float(record["polarization"])
        )
    if kind == "constant_wavelength_neutron_lorentz":
        return ConstantWavelengthNeutronLorentz(float(record["wavelength_angstrom"]))
    if kind == "time_of_flight_neutron_lorentz":
        return TimeOfFlightNeutronLorentz(float(record["two_theta_deg"]))
    raise ValueError(f"unsupported native correction model {kind!r}")


def _sample_physics(record: dict[str, Any] | None, cell: UnitCell) -> object | None:
    if record is None:
        return None
    kind = record["type"]
    if kind == "isotropic_size":
        return IsotropicSizeBroadening(
            float(record["crystallite_size_nm"]), float(record["shape_factor"])
        )
    if kind == "isotropic_microstrain":
        return IsotropicMicrostrainBroadening(float(record["rms_microstrain"]))
    if kind == "isotropic_lorentzian_microstrain":
        return IsotropicLorentzianMicrostrainBroadening(float(record["microstrain"]))
    if kind == "march_dollase":
        return MarchDollasePreferredOrientation(
            float(record["ratio"]),
            tuple(map(float, record["preferred_axis_hkl"])),
            ReciprocalMetric(cell.geometry().reciprocal_metric),
        )
    if kind == "composite":
        return CompositePhysicsProvider(
            tuple(_sample_physics(item, cell) for item in record["components"])
        )
    raise ValueError(f"unsupported native sample-physics model {kind!r}")


def _domain(
    record: dict[str, Any] | None, phase: RietveldPhase
) -> CwStructuralReflectionDomain | None:
    if record is None:
        return None
    parameterization = LatticeParameterization(phase.structure.space_group, phase.structure.cell)
    bounds = LatticeParameterBounds(parameterization, record["lower"], record["upper"])
    return CwStructuralReflectionDomain(
        phase.structure.space_group,
        parameterization,
        bounds,
        float(record["wavelength_angstrom"]),
        float(record["visible_two_theta_deg"][0]),
        float(record["visible_two_theta_deg"][1]),
        bool(record["merge_friedel"]),
        int(record["max_candidates"]),
        float(record["guard_scale"]),
    )


def _background(record: dict[str, Any] | None) -> DifferentiableBackground | None:
    if record is None:
        return None
    kind = record["type"]
    if kind == "polynomial":
        return PolynomialBackground(record["background_id"], tuple(record["coefficients"]))
    if kind == "chebyshev":
        return ChebyshevBackground(
            record["background_id"], tuple(record["coefficients"]), tuple(record["domain_deg"])
        )
    if kind == "point":
        return PointBackground(
            record["background_id"], tuple(record["knot_x"]), tuple(record["values"])
        )
    if kind == "amorphous":
        return AmorphousBackground(
            record["background_id"],
            tuple(AmorphousPeak(*map(float, values)) for values in record["peaks"]),
        )
    if kind == "composite":
        return CompositeBackground(
            record["background_id"], tuple(_background(item) for item in record["components"])
        )
    raise ValueError(f"unsupported native background model {kind!r}")


def _selection(record: dict[str, Any]) -> RietveldParameterSelection:
    return RietveldParameterSelection(
        phase_scale=bool(record["phase_scale"]),
        lattice=bool(record["lattice"]),
        coordinates=bool(record["coordinates"]),
        occupancy=bool(record["occupancy"]),
        u_iso=bool(record["u_iso"]),
        sample_physics=bool(record["sample_physics"]),
        instrument_parameters=tuple(record["instrument"]),
        background=bool(record["background"]),
    )


def _key(record: dict[str, Any]) -> ParameterKey:
    return ParameterKey(record["module"], record["owner_id"], record["name"])


def _constraint(record: dict[str, Any]) -> Constraint:
    kind = record["type"]
    if kind == "fixed":
        return FixedConstraint(_key(record["target"]), float(record["value"]))
    if kind == "affine":
        return AffineConstraint(
            _key(record["target"]),
            _key(record["source"]),
            float(record["multiplier"]),
            float(record["offset"]),
        )
    if kind == "linear":
        return LinearConstraint(
            _key(record["target"]),
            tuple((_key(term["source"]), float(term["coefficient"])) for term in record["terms"]),
            float(record["offset"]),
        )
    raise ValueError(f"unsupported native constraint {kind!r}")


def _options(record: dict[str, Any], covariance: dict[str, Any]) -> RietveldOptions:
    return RietveldOptions(
        limits=RefinementLimits(
            max_iterations=int(record["max_iterations"]),
            max_evaluations=int(record["max_evaluations"]),
            max_runtime_seconds=record["max_runtime_seconds"],
            max_consecutive_rejections=int(record["max_consecutive_rejections"]),
        ),
        min_iterations=int(record["min_iterations"]),
        objective_tolerance=float(record["objective_tolerance"]),
        parameter_tolerance=float(record["parameter_tolerance"]),
        initial_damping=float(record["initial_damping"]),
        damping_increase=float(record["damping_increase"]),
        damping_decrease=float(record["damping_decrease"]),
        cg_tolerance=float(record["cg_tolerance"]),
        max_cg_iterations=int(record["max_cg_iterations"]),
        max_scaled_parameter_step=float(record["max_scaled_parameter_step"]),
        max_backtracks=int(record["max_backtracks"]),
        use_uncertainty=bool(record["use_uncertainty"]),
        support_fwhm=float(record["support_fwhm"]),
        estimate_covariance=bool(covariance["enabled"]),
        max_covariance_parameters=int(covariance["max_parameters"]),
        unresolved_correlation=float(covariance["unresolved_correlation"]),
        execution=ExecutionPolicy(
            threads=record["requested_threads"],
            minimum_parallel_tasks=int(record["minimum_parallel_tasks"]),
        ),
    )


def _checkpoint(
    record: dict[str, Any] | None,
    input_data: RietveldInput,
    native: object | None,
) -> RietveldCheckpoint | None:
    if record is None:
        return None
    if native is None:
        raise ValueError("native checkpoint handle is missing")
    parameters = ParameterSet(
        [
            ParameterSpec(
                ParameterKey(module, owner, name),
                value,
                unit,
                Bounds(lower, upper),
                scale,
                refine,
            )
            for module, owner, name, value, unit, lower, upper, scale, refine in (
                native.parameter_records()
            )
        ]
    )
    values = parameters.values()
    experiment, background = _apply_profile_background_values(
        input_data.experiment, input_data.background, values
    )
    wavelength = experiment.radiation.wavelength_angstrom
    domains = tuple(
        None
        if domain is None
        else domain.__class__(
            domain.space_group,
            domain.parameterization,
            domain.bounds,
            wavelength,
            domain.visible_two_theta_min_deg,
            domain.visible_two_theta_max_deg,
            domain.merge_friedel,
            domain.max_candidates,
            domain.guard_scale,
        )
        for domain in input_data.lattice_domains
    )
    phases, _ = _apply_parameter_values(
        input_data.phases,
        domains,
        input_data.parameters,
        values,
        wavelength_angstrom=wavelength,
    )
    history = tuple(_iteration(item) for item in record["history"])
    return RietveldCheckpoint(
        int(record["completed_iterations"]),
        phases,
        domains,
        parameters,
        float(record["objective"]),
        float(record["damping"]),
        history,
        experiment,
        background,
        native,
    )


def _iteration(record: dict[str, Any]) -> RietveldIterationRecord:
    topology = tuple(
        f"phase {item['phase_id']}: +{len(item['added_reflection_ids'])} "
        f"-{len(item['removed_reflection_ids'])} guarded families"
        for item in record["topology_changes"]
    )
    return RietveldIterationRecord(
        int(record["iteration"]),
        float(record["rp"]),
        float(record["rwp"]),
        float(record["chi_square"]),
        float(record["reduced_chi_square"]),
        float(record["objective"]),
        float(record["objective_change"]),
        float(record["scaled_step_norm"]),
        float(record["damping"]),
        int(record["cg_iterations"]),
        int(record["backtracks"]),
        tuple(
            RietveldParameterChange(
                _key(item["key"]),
                float(item["before"]),
                float(item["after"]),
                float(item["scaled_change"]),
            )
            for item in record["parameter_changes"]
        ),
        topology,
    )
