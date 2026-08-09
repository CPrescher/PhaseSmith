"""Parser-independent immutable crystallographic structure models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .crystallography import AtomSiteBatch, UnitCell
from .symmetry import SpaceGroup, SymmetryOperation


@dataclass(frozen=True, slots=True)
class StructureDiagnostic:
    """One stable import or structure-conversion diagnostic."""

    severity: Literal["warning", "error"]
    code: str
    message: str
    tag: str | None = None
    row: int | None = None

    def __post_init__(self) -> None:
        """Validate stable machine and human-readable fields."""

        if self.severity not in ("warning", "error"):
            raise ValueError("diagnostic severity must be warning or error")
        if not self.code or not self.message:
            raise ValueError("diagnostic code and message must be non-empty")
        if self.row is not None and self.row < 0:
            raise ValueError("diagnostic row must be non-negative")


@dataclass(frozen=True, slots=True)
class StructureSource:
    """Plain source provenance retained after an optional parser is gone."""

    format: str
    block_name: str
    backend: str
    backend_version: str
    source_name: str | None = None


@dataclass(frozen=True, slots=True)
class AnisotropicDisplacement:
    """CIF U tensor in component order ``11,22,33,23,13,12``."""

    u_cif_angstrom2: tuple[float, float, float, float, float, float]
    source_convention: Literal["U_cif", "B_cif"]
    standard_uncertainty: tuple[
        float | None, float | None, float | None, float | None, float | None, float | None
    ] = (None, None, None, None, None, None)

    def __post_init__(self) -> None:
        """Require a finite symmetric-tensor component record."""

        if len(self.u_cif_angstrom2) != 6 or not np.isfinite(self.u_cif_angstrom2).all():
            raise ValueError("anisotropic displacement must contain six finite components")
        if self.source_convention not in ("U_cif", "B_cif"):
            raise ValueError("anisotropic source convention must be U_cif or B_cif")
        if len(self.standard_uncertainty) != 6 or any(
            value is not None and (not np.isfinite(value) or value < 0.0)
            for value in self.standard_uncertainty
        ):
            raise ValueError(
                "anisotropic standard uncertainties must contain six non-negative values"
            )


@dataclass(frozen=True, slots=True)
class AtomSite:
    """One independent crystallographic atom site with preserved identity."""

    site_id: str
    source_label: str
    type_symbol: str
    element_symbol: str
    fractional_xyz: tuple[float, float, float]
    occupancy: float = 1.0
    u_iso_angstrom2: float | None = None
    anisotropic_displacement: AnisotropicDisplacement | None = None
    charge: int | None = None
    isotope: int | None = None
    disorder_group: str | None = None
    fractional_xyz_standard_uncertainty: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    occupancy_standard_uncertainty: float | None = None
    u_iso_standard_uncertainty: float | None = None

    def __post_init__(self) -> None:
        """Validate physical scalar domains while preserving fractional wrapping."""

        if not self.site_id or self.site_id != self.site_id.strip():
            raise ValueError("site_id must be a non-empty trimmed string")
        if not self.source_label or not self.type_symbol or not self.element_symbol:
            raise ValueError("site label and chemical symbols must be non-empty")
        if len(self.fractional_xyz) != 3 or not np.isfinite(self.fractional_xyz).all():
            raise ValueError("fractional_xyz must contain three finite values")
        if not np.isfinite(self.occupancy) or self.occupancy < 0.0:
            raise ValueError("occupancy must be non-negative and finite")
        if self.u_iso_angstrom2 is not None and (
            not np.isfinite(self.u_iso_angstrom2) or self.u_iso_angstrom2 < 0.0
        ):
            raise ValueError("u_iso_angstrom2 must be non-negative and finite when present")
        if len(self.fractional_xyz_standard_uncertainty) != 3:
            raise ValueError("fractional coordinate uncertainties must contain three entries")
        if self.isotope is not None and self.isotope <= 0:
            raise ValueError("isotope mass number must be positive when present")
        uncertainties = (
            *self.fractional_xyz_standard_uncertainty,
            self.occupancy_standard_uncertainty,
            self.u_iso_standard_uncertainty,
        )
        if any(
            value is not None and (not np.isfinite(value) or value < 0.0) for value in uncertainties
        ):
            raise ValueError("standard uncertainties must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class CrystalStructure:
    """Typed cell, exact symmetry, optional atom sites, and plain provenance."""

    structure_id: str
    name: str
    cell: UnitCell
    space_group: SpaceGroup
    sites: tuple[AtomSite, ...] = ()
    source: StructureSource | None = None
    cell_standard_uncertainties: tuple[
        float | None, float | None, float | None, float | None, float | None, float | None
    ] = (None, None, None, None, None, None)
    diagnostics: tuple[StructureDiagnostic, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Freeze sequences/mappings and validate stable identities."""

        if not self.structure_id or self.structure_id != self.structure_id.strip():
            raise ValueError("structure_id must be a non-empty trimmed string")
        if not self.name or not self.name.strip():
            raise ValueError("structure name must be non-empty")
        if not isinstance(self.cell, UnitCell) or not isinstance(self.space_group, SpaceGroup):
            raise TypeError("cell and space_group must be typed crystallographic models")
        metric = self.cell.geometry().direct_metric
        components = np.array(
            [metric[0, 0], metric[1, 1], metric[2, 2], metric[1, 2], metric[0, 2], metric[0, 1]]
        )
        equations = self.space_group.metric_constraints.equations
        if equations.size:
            residual = np.abs(equations @ components)
            scale = max(float(np.max(np.abs(components))), 1.0)
            coefficient_scale = np.maximum(np.sum(np.abs(equations), axis=1), 1)
            if np.any(residual > 1e-10 * scale * coefficient_scale):
                raise ValueError("unit-cell metric is incompatible with the structure space group")
        sites = tuple(self.sites)
        if any(not isinstance(site, AtomSite) for site in sites):
            raise TypeError("sites must contain AtomSite values")
        if len({site.site_id for site in sites}) != len(sites):
            raise ValueError("site IDs must be unique within a structure")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, StructureDiagnostic) for item in diagnostics):
            raise TypeError("diagnostics must contain StructureDiagnostic values")
        if len(self.cell_standard_uncertainties) != 6:
            raise ValueError("cell_standard_uncertainties must contain six entries")
        if any(
            value is not None and (not np.isfinite(value) or value < 0.0)
            for value in self.cell_standard_uncertainties
        ):
            raise ValueError("cell standard uncertainties must be non-negative and finite")
        metadata = MappingProxyType({str(key): str(value) for key, value in self.metadata.items()})
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "metadata", metadata)

    def to_isotropic_site_batch(self) -> AtomSiteBatch:
        """Convert to the P1 batch only when no anisotropic model would be lost."""

        anisotropic = [site.site_id for site in self.sites if site.anisotropic_displacement]
        if anisotropic:
            joined = ", ".join(anisotropic)
            raise NotImplementedError(
                f"anisotropic displacement is not yet supported by structure factors: {joined}"
            )
        return AtomSiteBatch(
            [site.site_id for site in self.sites],
            [site.type_symbol for site in self.sites],
            np.asarray([site.fractional_xyz for site in self.sites], dtype=np.float64).reshape(
                -1, 3
            ),
            [site.occupancy for site in self.sites],
            [site.u_iso_angstrom2 or 0.0 for site in self.sites],
        )

    def to_site_batch(self) -> AtomSiteBatch:
        """Convert sites without discarding fixed CIF anisotropic tensors."""

        anisotropic = [site.anisotropic_displacement for site in self.sites]
        return AtomSiteBatch(
            [site.site_id for site in self.sites],
            [site.type_symbol for site in self.sites],
            np.asarray([site.fractional_xyz for site in self.sites], dtype=np.float64).reshape(
                -1, 3
            ),
            [site.occupancy for site in self.sites],
            [site.u_iso_angstrom2 or 0.0 for site in self.sites],
            anisotropic_mask=[value is not None for value in anisotropic],
            u_aniso_cif_angstrom2=[
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if value is None else value.u_cif_angstrom2
                for value in anisotropic
            ],
        )


def structure_to_record(structure: CrystalStructure) -> dict[str, Any]:
    """Serialize a structure to parser-independent JSON-compatible values."""

    if not isinstance(structure, CrystalStructure):
        raise TypeError("structure must be a CrystalStructure")
    operations = [
        {
            "rotation": operation.rotation.tolist(),
            "translation": [
                [value.numerator, value.denominator] for value in operation.translation
            ],
        }
        for operation in structure.space_group.operations
    ]
    sites = []
    for site in structure.sites:
        anisotropic = site.anisotropic_displacement
        sites.append(
            {
                "site_id": site.site_id,
                "source_label": site.source_label,
                "type_symbol": site.type_symbol,
                "element_symbol": site.element_symbol,
                "fractional_xyz": list(site.fractional_xyz),
                "occupancy": site.occupancy,
                "u_iso_angstrom2": site.u_iso_angstrom2,
                "anisotropic_displacement": None
                if anisotropic is None
                else {
                    "u_cif_angstrom2": list(anisotropic.u_cif_angstrom2),
                    "source_convention": anisotropic.source_convention,
                    "standard_uncertainty": list(anisotropic.standard_uncertainty),
                },
                "charge": site.charge,
                "isotope": site.isotope,
                "disorder_group": site.disorder_group,
                "fractional_xyz_standard_uncertainty": list(
                    site.fractional_xyz_standard_uncertainty
                ),
                "occupancy_standard_uncertainty": site.occupancy_standard_uncertainty,
                "u_iso_standard_uncertainty": site.u_iso_standard_uncertainty,
            }
        )
    return {
        "format_version": 1,
        "structure_id": structure.structure_id,
        "name": structure.name,
        "cell": list(structure.cell.as_tuple()),
        "cell_standard_uncertainties": list(structure.cell_standard_uncertainties),
        "operations": operations,
        "sites": sites,
        "source": None if structure.source is None else _dataclass_record(structure.source),
        "diagnostics": [_dataclass_record(item) for item in structure.diagnostics],
        "metadata": dict(structure.metadata),
    }


def structure_from_record(record: Mapping[str, Any]) -> CrystalStructure:
    """Restore a structure without importing its original parser backend."""

    if record.get("format_version") != 1:
        raise ValueError("unsupported structure record format version")
    operations = [
        SymmetryOperation(
            item["rotation"],
            tuple(Fraction(int(value[0]), int(value[1])) for value in item["translation"]),
        )
        for item in record["operations"]
    ]
    sites = []
    for item in record["sites"]:
        anisotropic_record = item["anisotropic_displacement"]
        anisotropic = (
            None
            if anisotropic_record is None
            else AnisotropicDisplacement(
                tuple(anisotropic_record["u_cif_angstrom2"]),
                anisotropic_record["source_convention"],
                tuple(anisotropic_record["standard_uncertainty"]),
            )
        )
        sites.append(
            AtomSite(
                site_id=item["site_id"],
                source_label=item["source_label"],
                type_symbol=item["type_symbol"],
                element_symbol=item["element_symbol"],
                fractional_xyz=tuple(item["fractional_xyz"]),
                occupancy=item["occupancy"],
                u_iso_angstrom2=item["u_iso_angstrom2"],
                anisotropic_displacement=anisotropic,
                charge=item["charge"],
                isotope=item["isotope"],
                disorder_group=item["disorder_group"],
                fractional_xyz_standard_uncertainty=tuple(
                    item["fractional_xyz_standard_uncertainty"]
                ),
                occupancy_standard_uncertainty=item["occupancy_standard_uncertainty"],
                u_iso_standard_uncertainty=item["u_iso_standard_uncertainty"],
            )
        )
    source_record = record["source"]
    source = None if source_record is None else StructureSource(**source_record)
    diagnostics = tuple(StructureDiagnostic(**item) for item in record["diagnostics"])
    return CrystalStructure(
        structure_id=record["structure_id"],
        name=record["name"],
        cell=UnitCell(*record["cell"]),
        space_group=SpaceGroup(operations),
        sites=tuple(sites),
        source=source,
        cell_standard_uncertainties=tuple(record["cell_standard_uncertainties"]),
        diagnostics=diagnostics,
        metadata=record["metadata"],
    )


def _dataclass_record(value: Any) -> dict[str, Any]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}
