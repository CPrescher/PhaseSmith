"""Gemmi-backed CIF syntax and setting adapter.

Gemmi objects are consumed entirely inside this module. Public results contain
only Rietveld Engine models, exact fractions, strings, and NumPy-compatible
scalars.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import gemmi
import numpy as np

from ..crystallography import UnitCell
from ..structure import (
    AnisotropicDisplacement,
    AtomSite,
    CrystalStructure,
    StructureDiagnostic,
    StructureSource,
)
from ..symmetry import SpaceGroup, SymmetryOperation
from .cif import CifReadLimits, CifReadResult

CELL_TAGS = (
    ("_cell.length_a", "_cell_length_a"),
    ("_cell.length_b", "_cell_length_b"),
    ("_cell.length_c", "_cell_length_c"),
    ("_cell.angle_alpha", "_cell_angle_alpha"),
    ("_cell.angle_beta", "_cell_angle_beta"),
    ("_cell.angle_gamma", "_cell_angle_gamma"),
)
EXPLICIT_OPERATION_TAGS = (
    "_space_group_symop.operation_xyz",
    "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",
)
HALL_TAGS = (
    "_space_group.name_Hall",
    "_space_group_name_Hall",
    "_symmetry_space_group_name_Hall",
)
HM_TAGS = (
    "_space_group.name_H-M_alt",
    "_space_group_name_H-M_alt",
    "_symmetry_space_group_name_H-M",
)
NUMBER_TAGS = (
    "_space_group.IT_number",
    "_space_group_IT_number",
    "_symmetry_Int_Tables_number",
)
ATOM_TAGS = {
    "label": ("_atom_site.label", "_atom_site_label"),
    "type_symbol": ("_atom_site.type_symbol", "_atom_site_type_symbol"),
    "fract_x": ("_atom_site.fract_x", "_atom_site_fract_x"),
    "fract_y": ("_atom_site.fract_y", "_atom_site_fract_y"),
    "fract_z": ("_atom_site.fract_z", "_atom_site_fract_z"),
    "cart_x": ("_atom_site.Cartn_x", "_atom_site_Cartn_x"),
    "cart_y": ("_atom_site.Cartn_y", "_atom_site_Cartn_y"),
    "cart_z": ("_atom_site.Cartn_z", "_atom_site_Cartn_z"),
    "occupancy": ("_atom_site.occupancy", "_atom_site_occupancy"),
    "u_iso": ("_atom_site.U_iso_or_equiv", "_atom_site_U_iso_or_equiv"),
    "b_iso": ("_atom_site.B_iso_or_equiv", "_atom_site_B_iso_or_equiv"),
    "disorder_group": ("_atom_site.disorder_group", "_atom_site_disorder_group"),
}
ANISO_LABEL_TAGS = ("_atom_site_aniso.label", "_atom_site_aniso_label")
ANISO_COMPONENTS = ("11", "22", "33", "23", "13", "12")
NUMBER_PATTERN = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\((\d+)\))?([Ee][+-]?\d+)?$")
TYPE_SYMBOL_PATTERN = re.compile(r"^(?:(\d+))?([A-Z][a-z]?|D|T)")
CHARGE_PATTERN = re.compile(r"(?:(\d+)([+-])|([+-])(\d+))$")
UNSUPPORTED_PREFIXES = {
    "magnetic": ("_atom_site_moment", "_space_group_symop_magn", "_space_group_magn"),
    "modulated": ("_cell_wave_vector", "_atom_site_fourier", "_space_group_symop_ssg"),
    "macromolecular": ("_entity_poly", "_pdbx_", "_atom_site.label_asym_id"),
}


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    value: float
    standard_uncertainty: float | None


class GemmiCifBackend:
    """CIF parser/space-group adapter tested with Gemmi 0.7.x."""

    name = "gemmi"
    version = gemmi.__version__

    def parse_text(
        self,
        text: str,
        *,
        source_name: str | None,
        block: str | None,
        strict: bool,
        limits: CifReadLimits,
    ) -> CifReadResult:
        """Parse one selected small-structure block into public models."""

        try:
            document = gemmi.cif.read_string(text)
        except RuntimeError as error:
            raise ValueError(f"invalid CIF syntax: {error}") from error
        if len(document) == 0:
            raise ValueError("CIF document contains no data blocks")
        if len(document) > limits.max_blocks:
            raise ValueError("CIF document exceeds max_blocks")
        blocks = list(document)
        available = _display_block_names(blocks)
        selected_index, initial_diagnostics = _select_block(available, block, strict)
        selected = blocks[selected_index]
        diagnostics = list(initial_diagnostics)
        _check_loop_limits(selected, limits)
        _check_unsupported_features(selected, diagnostics, strict)

        cell, cell_uncertainties = _parse_cell(selected, diagnostics, strict)
        space_group, symmetry_metadata = _parse_space_group(selected, diagnostics, strict)
        sites = _parse_sites(selected, cell, diagnostics, strict, limits)
        name = (
            _first_text(
                selected,
                (
                    "_chemical_name_common",
                    "_chemical.name_common",
                    "_chemical_formula_structural",
                    "_chemical_formula_sum",
                ),
            )
            or available[selected_index]
        )
        metadata = dict(symmetry_metadata)
        for key, tags in {
            "chemical_formula_sum": ("_chemical_formula_sum", "_chemical.formula_sum"),
            "chemical_formula_structural": (
                "_chemical_formula_structural",
                "_chemical.formula_structural",
            ),
            "radiation_wavelength": (
                "_diffrn_radiation_wavelength",
                "_diffrn_radiation.wavelength",
            ),
        }.items():
            value = _first_text(selected, tags)
            if value is not None:
                metadata[key] = value
        source = StructureSource(
            format="CIF",
            block_name=available[selected_index],
            backend=self.name,
            backend_version=self.version,
            source_name=source_name,
        )
        structure = CrystalStructure(
            structure_id=_stable_structure_id(available[selected_index]),
            name=name,
            cell=cell,
            space_group=space_group,
            sites=sites,
            source=source,
            cell_standard_uncertainties=cell_uncertainties,
            diagnostics=tuple(diagnostics),
            metadata=metadata,
        )
        return CifReadResult(
            structure=structure,
            diagnostics=tuple(diagnostics),
            selected_block=available[selected_index],
            available_blocks=available,
        )


def _display_block_names(blocks: list[Any]) -> tuple[str, ...]:
    names = tuple(
        block.name.strip() or f"unnamed_{index + 1}" for index, block in enumerate(blocks)
    )
    if len(set(names)) != len(names):
        raise ValueError("CIF data block names must be unique")
    return names


def _select_block(
    available: tuple[str, ...], requested: str | None, strict: bool
) -> tuple[int, tuple[StructureDiagnostic, ...]]:
    if requested is not None:
        try:
            return available.index(requested), ()
        except ValueError as error:
            raise ValueError(f"CIF block {requested!r} was not found") from error
    if len(available) == 1:
        return 0, ()
    if strict:
        raise ValueError("multi-block CIF requires an explicit block in strict mode")
    diagnostic = StructureDiagnostic(
        "warning",
        "multiple_blocks_first_selected",
        f"selected first of {len(available)} CIF blocks: {available[0]}",
    )
    return 0, (diagnostic,)


def _check_loop_limits(block: Any, limits: CifReadLimits) -> None:
    for item in block:
        if item.loop is not None and item.loop.length() > limits.max_loop_rows:
            raise ValueError("CIF loop exceeds max_loop_rows")


def _all_tags(block: Any) -> set[str]:
    tags: set[str] = set()
    for item in block:
        if item.pair is not None:
            tags.add(item.pair[0].lower())
        elif item.loop is not None:
            tags.update(tag.lower() for tag in item.loop.tags)
    return tags


def _check_unsupported_features(
    block: Any, diagnostics: list[StructureDiagnostic], strict: bool
) -> None:
    tags = _all_tags(block)
    for feature, prefixes in UNSUPPORTED_PREFIXES.items():
        matches = sorted(tag for tag in tags if any(tag.startswith(prefix) for prefix in prefixes))
        if not matches:
            continue
        message = f"{feature} CIF features are not supported: {', '.join(matches[:3])}"
        if strict:
            raise NotImplementedError(message)
        diagnostics.append(
            StructureDiagnostic("warning", f"unsupported_{feature}_features", message)
        )


def _parse_cell(
    block: Any, diagnostics: list[StructureDiagnostic], strict: bool
) -> tuple[UnitCell, tuple[float | None, ...]]:
    values = []
    uncertainties = []
    for aliases in CELL_TAGS:
        definitions = [
            (tag, value) for tag in aliases if (value := block.find_value(tag)) is not None
        ]
        if not definitions:
            raise ValueError(f"CIF cell parameter is missing: {aliases[0]}")
        tag, raw = definitions[0]
        parsed = _parse_number(raw, tag or aliases[0], diagnostics, required=True)
        if parsed is None:
            raise ValueError(f"CIF cell parameter is not numeric: {tag or aliases[0]}")
        for duplicate_tag, duplicate_raw in definitions[1:]:
            duplicate = _parse_number(
                duplicate_raw,
                duplicate_tag,
                diagnostics,
                required=True,
            )
            if duplicate is not None and not np.isclose(
                parsed.value, duplicate.value, rtol=1e-10, atol=1e-12
            ):
                message = f"duplicate cell definitions disagree: {tag} and {duplicate_tag}"
                if strict:
                    raise ValueError(message)
                diagnostics.append(
                    StructureDiagnostic(
                        "warning", "conflicting_cell_definition", message, duplicate_tag
                    )
                )
        values.append(parsed.value)
        uncertainties.append(parsed.standard_uncertainty)
    return UnitCell(*values), tuple(uncertainties)


def _parse_space_group(
    block: Any,
    diagnostics: list[StructureDiagnostic],
    strict: bool,
) -> tuple[SpaceGroup, dict[str, str]]:
    candidates: list[tuple[str, str, SpaceGroup]] = []
    operation_tag, operation_values = _first_column(block, EXPLICIT_OPERATION_TAGS)
    if operation_values:
        operations = []
        for row, raw in enumerate(operation_values):
            text = _as_text(raw)
            if text is None:
                raise ValueError(f"missing explicit symmetry operation at row {row}")
            try:
                operations.append(_operation_from_gemmi(gemmi.Op(text)))
            except (RuntimeError, ValueError) as error:
                raise ValueError(f"invalid symmetry operation {text!r} at row {row}") from error
        candidates.append(("explicit_operations", operation_tag or "", SpaceGroup(operations)))

    hall_tag, hall_raw = _first_raw(block, HALL_TAGS)
    hall = _as_text(hall_raw)
    if hall:
        match = next((group for group in gemmi.spacegroup_table() if group.hall == hall), None)
        if match is None:
            raise ValueError(f"unknown Hall symbol {hall!r}")
        candidates.append(("hall", hall_tag or "", _space_group_from_gemmi(match)))

    hm_tag, hm_raw = _first_raw(block, HM_TAGS)
    hm = _as_text(hm_raw)
    if hm:
        match = gemmi.find_spacegroup_by_name(hm)
        if match is None:
            raise ValueError(f"unknown Hermann-Mauguin symbol {hm!r}")
        candidates.append(("hermann_mauguin", hm_tag or "", _space_group_from_gemmi(match)))

    number_tag, number_raw = _first_raw(block, NUMBER_TAGS)
    number_text = _as_text(number_raw)
    if number_text:
        try:
            number = int(float(number_text))
            match = gemmi.find_spacegroup_by_number(number)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"invalid space-group number {number_text!r}") from error
        if match is None:
            raise ValueError(f"unknown space-group number {number}")
        candidates.append(
            ("international_number", number_tag or "", _space_group_from_gemmi(match))
        )

    if not candidates:
        diagnostics.append(
            StructureDiagnostic(
                "warning",
                "missing_space_group_assumed_p1",
                "no symmetry identifier was supplied; assumed P1",
            )
        )
        return SpaceGroup.p1(), {"symmetry_source": "assumed_p1"}
    source, tag, selected = candidates[0]
    for other_source, other_tag, candidate in candidates[1:]:
        if candidate == selected:
            continue
        message = (
            f"space-group definitions disagree: {source} ({tag}) takes precedence over "
            f"{other_source} ({other_tag})"
        )
        if strict:
            raise ValueError(message)
        diagnostics.append(
            StructureDiagnostic("warning", "conflicting_space_group_definition", message, other_tag)
        )
    metadata = {"symmetry_source": source}
    if hall:
        metadata["space_group_hall"] = hall
    if hm:
        metadata["space_group_hm"] = hm
    if number_text:
        metadata["space_group_number"] = number_text
    return selected, metadata


def _operation_from_gemmi(operation: Any) -> SymmetryOperation:
    denominator = int(operation.DEN)
    rotation = []
    for row in operation.rot:
        if any(int(value) % denominator != 0 for value in row):
            raise ValueError("Gemmi returned a non-integral crystallographic rotation")
        rotation.append([int(value) // denominator for value in row])
    translation = tuple(Fraction(int(value), denominator) for value in operation.tran)
    return SymmetryOperation(rotation, translation)


def _space_group_from_gemmi(group: Any) -> SpaceGroup:
    return SpaceGroup([_operation_from_gemmi(operation) for operation in group.operations()])


def _parse_sites(
    block: Any,
    cell: UnitCell,
    diagnostics: list[StructureDiagnostic],
    strict: bool,
    limits: CifReadLimits,
) -> tuple[AtomSite, ...]:
    columns = {name: _first_column(block, tags) for name, tags in ATOM_TAGS.items()}
    label_tag, labels = columns["label"]
    coordinate_presence = any(
        columns[name][1] for name in ("fract_x", "fract_y", "fract_z", "cart_x", "cart_y", "cart_z")
    )
    if not labels:
        if coordinate_presence:
            raise ValueError("atom-site coordinates require an atom-site label column")
        if _first_column(block, ANISO_LABEL_TAGS)[1]:
            raise ValueError("anisotropic displacement rows require atom-site rows")
        return ()
    if len(labels) > limits.max_atom_sites:
        raise ValueError("CIF atom-site loop exceeds max_atom_sites")
    for name, (tag, values) in columns.items():
        if values and len(values) != len(labels):
            raise ValueError(f"atom-site column length mismatch for {tag or name}")
    fractional_complete = all(columns[name][1] for name in ("fract_x", "fract_y", "fract_z"))
    cartesian_complete = all(columns[name][1] for name in ("cart_x", "cart_y", "cart_z"))
    if not fractional_complete and not cartesian_complete:
        raise ValueError("atom sites require complete fractional or Cartesian coordinates")
    anisotropic = _parse_anisotropic(block, diagnostics, strict, limits)
    direct_basis = cell.geometry().direct_basis
    sites = []
    attached_anisotropic_labels: set[str] = set()
    used_ids: dict[str, int] = {}
    for row, raw_label in enumerate(labels):
        label = _as_text(raw_label)
        if not label:
            if strict:
                raise ValueError(f"atom-site label is missing at row {row}")
            diagnostics.append(
                StructureDiagnostic(
                    "warning",
                    "skipped_atom_missing_label",
                    "skipped atom with missing label",
                    label_tag,
                    row,
                )
            )
            continue
        try:
            fractional, coordinate_su = _site_coordinates(
                columns, row, direct_basis, diagnostics, strict
            )
            type_symbol = _site_text(columns["type_symbol"][1], row) or _symbol_from_label(label)
            element_symbol, isotope, charge = _chemical_identity(type_symbol)
            occupancy = _optional_site_number(columns["occupancy"], row, diagnostics)
            occupancy_value = 1.0 if occupancy is None else occupancy.value
            if occupancy_value < 0.0:
                raise ValueError("occupancy is negative")
            u_value, u_su = _site_u_iso(columns, row, diagnostics, strict)
        except ValueError as error:
            if strict:
                raise ValueError(f"invalid atom site {label!r} at row {row}: {error}") from error
            diagnostics.append(
                StructureDiagnostic(
                    "warning",
                    "skipped_invalid_atom_site",
                    f"skipped atom {label!r}: {error}",
                    row=row,
                )
            )
            continue
        occurrence = used_ids.get(label, 0) + 1
        used_ids[label] = occurrence
        site_id = label if occurrence == 1 else f"{label}#{occurrence}"
        if occurrence > 1:
            message = f"duplicate atom label {label!r} was renamed to {site_id!r}"
            if strict:
                raise ValueError(message)
            diagnostics.append(
                StructureDiagnostic("warning", "duplicate_atom_label_renamed", message, row=row)
            )
        disorder_group = _site_text(columns["disorder_group"][1], row)
        sites.append(
            AtomSite(
                site_id=site_id,
                source_label=label,
                type_symbol=type_symbol,
                element_symbol=element_symbol,
                fractional_xyz=tuple(float(value) for value in fractional),
                occupancy=occupancy_value,
                u_iso_angstrom2=u_value,
                anisotropic_displacement=anisotropic.get(label),
                charge=charge,
                isotope=isotope,
                disorder_group=disorder_group,
                fractional_xyz_standard_uncertainty=coordinate_su,
                occupancy_standard_uncertainty=None
                if occupancy is None
                else occupancy.standard_uncertainty,
                u_iso_standard_uncertainty=u_su,
            )
        )
        if label in anisotropic:
            attached_anisotropic_labels.add(label)
    orphan_labels = sorted(set(anisotropic) - attached_anisotropic_labels)
    if orphan_labels:
        message = f"anisotropic rows have no matching atom sites: {', '.join(orphan_labels)}"
        if strict:
            raise ValueError(message)
        diagnostics.append(StructureDiagnostic("warning", "orphan_anisotropic_rows", message))
    return tuple(sites)


def _site_coordinates(
    columns: dict[str, tuple[str | None, list[str]]],
    row: int,
    direct_basis: np.ndarray,
    diagnostics: list[StructureDiagnostic],
    strict: bool,
) -> tuple[np.ndarray, tuple[float | None, float | None, float | None]]:
    fractional_values = []
    fractional_su = []
    if all(columns[name][1] for name in ("fract_x", "fract_y", "fract_z")):
        for name in ("fract_x", "fract_y", "fract_z"):
            parsed = _required_site_number(columns[name], row, diagnostics)
            fractional_values.append(parsed.value)
            fractional_su.append(parsed.standard_uncertainty)
    cartesian_values = []
    cartesian_su = []
    if all(columns[name][1] for name in ("cart_x", "cart_y", "cart_z")):
        for name in ("cart_x", "cart_y", "cart_z"):
            parsed = _required_site_number(columns[name], row, diagnostics)
            cartesian_values.append(parsed.value)
            cartesian_su.append(parsed.standard_uncertainty)
        if any(value is not None for value in cartesian_su):
            diagnostics.append(
                StructureDiagnostic(
                    "warning",
                    "cartesian_coordinate_uncertainty_not_transformed",
                    "Cartesian coordinate standard uncertainties are not transformed because "
                    "their covariance is unavailable",
                    row=row,
                )
            )
    if fractional_values:
        fractional = np.asarray(fractional_values, dtype=np.float64)
        if cartesian_values:
            converted = np.linalg.solve(direct_basis, np.asarray(cartesian_values))
            periodic_delta = np.abs(fractional - converted)
            periodic_delta = np.minimum(periodic_delta % 1.0, 1.0 - periodic_delta % 1.0)
            if np.max(periodic_delta) > 1e-8:
                message = "fractional and Cartesian atom coordinates disagree"
                if strict:
                    raise ValueError(message)
                diagnostics.append(
                    StructureDiagnostic("warning", "conflicting_atom_coordinates", message, row=row)
                )
        return fractional, tuple(fractional_su)
    fractional = np.linalg.solve(direct_basis, np.asarray(cartesian_values, dtype=np.float64))
    return fractional, (None, None, None)


def _site_u_iso(
    columns: dict[str, tuple[str | None, list[str]]],
    row: int,
    diagnostics: list[StructureDiagnostic],
    strict: bool,
) -> tuple[float | None, float | None]:
    u_value = _optional_site_number(columns["u_iso"], row, diagnostics)
    b_value = _optional_site_number(columns["b_iso"], row, diagnostics)
    converted_b = None if b_value is None else b_value.value / (8.0 * np.pi**2)
    if (
        u_value is not None
        and converted_b is not None
        and not np.isclose(u_value.value, converted_b, rtol=1e-8, atol=1e-12)
    ):
        message = "U_iso and B_iso definitions disagree"
        if strict:
            raise ValueError(message)
        diagnostics.append(
            StructureDiagnostic("warning", "conflicting_isotropic_displacement", message, row=row)
        )
    if u_value is not None:
        if u_value.value < 0.0:
            raise ValueError("U_iso is negative")
        return u_value.value, u_value.standard_uncertainty
    if b_value is not None:
        if converted_b is not None and converted_b < 0.0:
            raise ValueError("B_iso is negative")
        uncertainty = (
            None
            if b_value.standard_uncertainty is None
            else b_value.standard_uncertainty / (8.0 * np.pi**2)
        )
        return converted_b, uncertainty
    return None, None


def _parse_anisotropic(
    block: Any,
    diagnostics: list[StructureDiagnostic],
    strict: bool,
    limits: CifReadLimits,
) -> dict[str, AnisotropicDisplacement]:
    label_tag, labels = _first_column(block, ANISO_LABEL_TAGS)
    if not labels:
        return {}
    if len(labels) > limits.max_atom_sites:
        raise ValueError("CIF anisotropic loop exceeds max_atom_sites")
    u_columns = [
        _first_column(
            block,
            (f"_atom_site_aniso.U_{component}", f"_atom_site_aniso_U_{component}"),
        )
        for component in ANISO_COMPONENTS
    ]
    b_columns = [
        _first_column(
            block,
            (f"_atom_site_aniso.B_{component}", f"_atom_site_aniso_B_{component}"),
        )
        for component in ANISO_COMPONENTS
    ]
    u_complete = all(column[1] for column in u_columns)
    b_complete = all(column[1] for column in b_columns)
    u_present = any(column[1] for column in u_columns)
    b_present = any(column[1] for column in b_columns)
    if not u_complete and not b_complete:
        message = "anisotropic loop requires all six U or B components"
        if strict:
            raise ValueError(message)
        diagnostics.append(
            StructureDiagnostic(
                "warning", "ignored_incomplete_anisotropic_loop", message, label_tag
            )
        )
        return {}
    if (u_present and not u_complete) or (b_present and not b_complete):
        message = "secondary anisotropic U or B definition is incomplete"
        if strict:
            raise ValueError(message)
        diagnostics.append(
            StructureDiagnostic(
                "warning", "ignored_incomplete_anisotropic_definition", message, label_tag
            )
        )
    selected = u_columns if u_complete else b_columns
    convention = "U_cif" if u_complete else "B_cif"
    result = {}
    for row, raw_label in enumerate(labels):
        label = _as_text(raw_label)
        if not label:
            raise ValueError(f"anisotropic label is missing at row {row}")
        if label in result:
            message = f"duplicate anisotropic label {label!r} at row {row}"
            if strict:
                raise ValueError(message)
            diagnostics.append(
                StructureDiagnostic(
                    "warning", "duplicate_anisotropic_label_ignored", message, row=row
                )
            )
            continue
        parsed_components = [_required_site_number(column, row, diagnostics) for column in selected]
        components = [value.value for value in parsed_components]
        standard_uncertainties = [value.standard_uncertainty for value in parsed_components]
        if convention == "B_cif":
            components = [value / (8.0 * np.pi**2) for value in components]
            standard_uncertainties = [
                None if value is None else value / (8.0 * np.pi**2)
                for value in standard_uncertainties
            ]
        if u_complete and b_complete:
            converted_b = [
                _required_site_number(column, row, diagnostics).value / (8.0 * np.pi**2)
                for column in b_columns
            ]
            if not np.allclose(components, converted_b, rtol=1e-8, atol=1e-12):
                message = f"anisotropic U and B definitions disagree for {label!r}"
                if strict:
                    raise ValueError(message)
                diagnostics.append(
                    StructureDiagnostic(
                        "warning", "conflicting_anisotropic_displacement", message, row=row
                    )
                )
        result[label] = AnisotropicDisplacement(
            tuple(components), convention, tuple(standard_uncertainties)
        )
    return result


def _required_site_number(
    column: tuple[str | None, list[str]],
    row: int,
    diagnostics: list[StructureDiagnostic],
) -> ParsedNumber:
    tag, values = column
    if not values or row >= len(values):
        raise ValueError(f"required atom-site column is absent: {tag}")
    parsed = _parse_number(values[row], tag or "atom_site", diagnostics, required=True, row=row)
    if parsed is None:
        raise ValueError(f"required atom-site value is missing: {tag}")
    return parsed


def _optional_site_number(
    column: tuple[str | None, list[str]],
    row: int,
    diagnostics: list[StructureDiagnostic],
) -> ParsedNumber | None:
    tag, values = column
    if not values or row >= len(values):
        return None
    return _parse_number(values[row], tag or "atom_site", diagnostics, required=False, row=row)


def _parse_number(
    raw: str,
    tag: str,
    diagnostics: list[StructureDiagnostic],
    *,
    required: bool,
    row: int | None = None,
) -> ParsedNumber | None:
    text = _as_text(raw)
    if text is None:
        state = "unknown" if raw.strip() == "?" else "missing"
        diagnostics.append(
            StructureDiagnostic(
                "warning",
                f"{state}_cif_value",
                f"{state} CIF value for {tag}",
                tag,
                row,
            )
        )
        if required:
            raise ValueError(f"required CIF value for {tag} is {state}")
        return None
    match = NUMBER_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"CIF value for {tag} is not a number: {text!r}")
    mantissa, uncertainty_digits, exponent_text = match.groups()
    exponent = 0 if exponent_text is None else int(exponent_text[1:])
    value = float(mantissa) * 10.0**exponent
    uncertainty = None
    if uncertainty_digits is not None:
        decimal_places = len(mantissa.partition(".")[2])
        uncertainty = int(uncertainty_digits) * 10.0 ** (exponent - decimal_places)
    if not np.isfinite(value) or (uncertainty is not None and not np.isfinite(uncertainty)):
        raise ValueError(f"CIF value for {tag} is not finite")
    return ParsedNumber(value, uncertainty)


def _first_raw(block: Any, aliases: Iterable[str]) -> tuple[str | None, str | None]:
    for tag in aliases:
        value = block.find_value(tag)
        if value is not None:
            return tag, value
    return None, None


def _first_column(block: Any, aliases: Iterable[str]) -> tuple[str | None, list[str]]:
    for tag in aliases:
        values = list(block.find_values(tag))
        if values:
            return tag, values
    return None, []


def _first_text(block: Any, aliases: Iterable[str]) -> str | None:
    _, raw = _first_raw(block, aliases)
    return _as_text(raw)


def _as_text(raw: str | None) -> str | None:
    if raw is None or raw.strip() in (".", "?"):
        return None
    value = gemmi.cif.as_string(raw).strip()
    return value or None


def _site_text(values: list[str], row: int) -> str | None:
    if not values or row >= len(values):
        return None
    return _as_text(values[row])


def _symbol_from_label(label: str) -> str:
    match = re.match(r"^([A-Z][a-z]?|[A-Z])", label)
    if match is None:
        raise ValueError(f"cannot infer type symbol from atom label {label!r}")
    return match.group(1)


def _chemical_identity(type_symbol: str) -> tuple[str, int | None, int | None]:
    match = TYPE_SYMBOL_PATTERN.match(type_symbol)
    if match is None:
        raise ValueError(f"invalid atom type symbol {type_symbol!r}")
    isotope_text, element = match.groups()
    isotope = None if isotope_text is None else int(isotope_text)
    if element == "D":
        isotope = 2
        element = "H"
    elif element == "T":
        isotope = 3
        element = "H"
    charge_match = CHARGE_PATTERN.search(type_symbol)
    charge = None
    if charge_match:
        magnitude_text = charge_match.group(1) or charge_match.group(4)
        sign = charge_match.group(2) or charge_match.group(3)
        charge = int(magnitude_text) * (1 if sign == "+" else -1)
    return element, isotope, charge


def _stable_structure_id(block_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", block_name.strip()).strip("_")
    return value or "structure"
