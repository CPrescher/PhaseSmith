"""Typed output of native supplied-symmetry recovery and ADP refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..crystallography import UnitCell
from ..structure import AnisotropicDisplacement, AtomSite, CrystalStructure
from ..symmetry import SpaceGroup


@dataclass(frozen=True, slots=True)
class SymmetryRefinement:
    """Asymmetric-unit model in the supplied setting, with physical Cartesian ADPs."""

    space_group: SpaceGroup
    species: tuple[str, ...]
    fractional_xyz: NDArray[np.float64]
    u_cartesian: NDArray[np.float64]
    u_cif: NDArray[np.float64]
    u_equiv: NDArray[np.float64]
    full_u_cartesian: NDArray[np.float64]
    full_u_cif: NDArray[np.float64]
    full_u_equiv: NDArray[np.float64]
    hand: int
    origin_shift: NDArray[np.float64]
    maximum_matching_error: float
    parameter_count: int
    coordinate_parameter_count: int
    origin_gauge_dimension: int
    mode: str
    termination: str
    accepted_steps: int
    p1_r1: float
    p1_objective: float

    @classmethod
    def from_native(cls, record: dict, group: SpaceGroup) -> SymmetryRefinement:
        """Own immutable arrays rather than borrowing mutable native result buffers."""
        values = dict(record)
        values["species"] = tuple(values["species"])
        for name in (
            "fractional_xyz",
            "u_cartesian",
            "u_cif",
            "u_equiv",
            "full_u_cartesian",
            "full_u_cif",
            "full_u_equiv",
            "origin_shift",
        ):
            array = np.array(values[name], dtype=np.float64, copy=True, order="C")
            array.flags.writeable = False
            values[name] = array
        return cls(space_group=group, **values)

    def as_structure(self, cell: UnitCell) -> CrystalStructure:
        """Build an asymmetric-unit model with constrained anisotropic tensors."""
        sites = tuple(
            AtomSite(
                site_id=f"{element}{index + 1}",
                source_label=f"{element}{index + 1}",
                type_symbol=element,
                element_symbol=element,
                fractional_xyz=tuple(float(v) for v in point),
                u_iso_angstrom2=float(self.u_equiv[index]),
                anisotropic_displacement=AnisotropicDisplacement(
                    tuple(float(v) for v in self.u_cif[index]),
                    "U_cif",
                ),
            )
            for index, (element, point) in enumerate(
                zip(self.species, self.fractional_xyz, strict=True)
            )
        )
        return CrystalStructure(
            structure_id="experimental-symmetric-solution",
            name="Experimental symmetry-constrained candidate",
            cell=cell,
            space_group=self.space_group,
            sites=sites,
            metadata={"status": "unvalidated structure-solution candidate", "adp_mode": self.mode},
        )

    def to_record(self) -> dict[str, object]:
        """Finite numerical output; the parent result owns exact symmetry operations."""
        names = (
            "species",
            "fractional_xyz",
            "u_cartesian",
            "u_cif",
            "u_equiv",
            "full_u_cartesian",
            "full_u_cif",
            "full_u_equiv",
            "hand",
            "origin_shift",
            "maximum_matching_error",
            "parameter_count",
            "coordinate_parameter_count",
            "origin_gauge_dimension",
            "mode",
            "termination",
            "accepted_steps",
            "p1_r1",
            "p1_objective",
        )
        return {
            name: value.tolist() if isinstance(value := getattr(self, name), np.ndarray) else value
            for name in names
        }


def symmetry_records(group: SpaceGroup) -> list[dict[str, object]]:
    """Serialize exact operations; operation count alone cannot identify a setting."""
    return [
        {
            "rotation": op.rotation.tolist(),
            "translation": [[v.numerator, v.denominator] for v in op.translation],
        }
        for op in group.operations
    ]


def candidate_cif(
    cell: UnitCell,
    species: tuple[str, ...],
    xyz: NDArray,
    u_equiv: NDArray,
    group: SpaceGroup,
    u_cif: NDArray | None = None,
) -> str:
    """CIF with authoritative explicit symmetry and optional full anisotropic rows."""
    lines = ["data_experimental_candidate", "# Unvalidated PhaseSmith candidate."]
    if group == SpaceGroup.p1():
        lines += ["_space_group_name_H-M_alt 'P 1'", "_space_group_IT_number 1"]
    for key, value in zip(
        ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"),
        cell.as_tuple(),
        strict=True,
    ):
        lines.append(f"_cell_{key} {value:.12g}")
    lines += ["loop_", "_space_group_symop_id", "_space_group_symop_operation_xyz"]
    for index, op in enumerate(group.operations):
        expressions = []
        for row, translation in zip(op.rotation, op.translation, strict=True):
            expression = ""
            for coefficient, symbol in zip(row, "xyz", strict=True):
                if not coefficient:
                    continue
                term = symbol if abs(coefficient) == 1 else f"{abs(coefficient)}{symbol}"
                expression += ("+" if coefficient > 0 else "-") + term
            if translation:
                expression += "+" + str(translation)
            expressions.append(expression.lstrip("+") or "0")
        lines.append(f"{index + 1} '{','.join(expressions)}'")
    lines += [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_U_iso_or_equiv",
        "_atom_site_occupancy",
    ]
    for index, (element, point, u) in enumerate(zip(species, xyz, u_equiv, strict=True)):
        lines.append(
            f"{element}{index + 1} {element} "
            + " ".join(f"{v:.12g}" for v in point)
            + f" {u:.12g} 1"
        )
    if u_cif is not None:
        lines += [
            "loop_",
            "_atom_site_aniso_label",
            "_atom_site_aniso_U_11",
            "_atom_site_aniso_U_22",
            "_atom_site_aniso_U_33",
            "_atom_site_aniso_U_23",
            "_atom_site_aniso_U_13",
            "_atom_site_aniso_U_12",
        ]
        for index, (element, tensor) in enumerate(zip(species, u_cif, strict=True)):
            lines.append(f"{element}{index + 1} " + " ".join(f"{v:.12g}" for v in tensor))
    return "\n".join(lines) + "\n"
