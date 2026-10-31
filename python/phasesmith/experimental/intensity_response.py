"""Explicit homogeneous-slab assumptions; no material lookup or detector guessing."""

from dataclasses import asdict, dataclass

import numpy as np

from phasesmith import _core


@dataclass(frozen=True)
class SlabResponse:
    """A fixed geometric/absorption model, not measured counting efficiency.

    Supply attenuation at the stated wavelength explicitly. Density, thickness,
    threshold and provenance must not be inferred from a camera-name suffix.
    """

    thickness_mm: float
    attenuation_per_mm: float
    wavelength_angstrom: float
    material: str
    provenance: str
    density_g_cm3: float | None = None

    def __post_init__(self):
        for name in ("thickness_mm", "attenuation_per_mm", "wavelength_angstrom"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, complex))
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.thickness_mm * self.attenuation_per_mm)
            or self.thickness_mm * self.attenuation_per_mm == 0
        ):
            raise ValueError("optical depth overflows or underflows")
        if self.density_g_cm3 is not None and (
            isinstance(self.density_g_cm3, (bool, complex))
            or not np.isscalar(self.density_g_cm3)
            or not np.isfinite(self.density_g_cm3)
            or self.density_g_cm3 <= 0
        ):
            raise ValueError("density must be finite and positive when specified")
        if any(not isinstance(v, str) or not v.strip() for v in (self.material, self.provenance)):
            raise ValueError("material and response provenance are required")

    def evaluate(self, incidence_cosine):
        """Return eta, relative factor, depth(mm), and their three d/dcos partials."""
        c = np.asarray(incidence_cosine)
        if (
            c.ndim != 1
            or c.dtype.kind not in "fiu"
            or not np.isfinite(c).all()
            or ((c <= 0) | (c > 1)).any()
        ):
            raise ValueError("incidence cosine must be a finite vector in (0,1]")
        if not hasattr(_core, "experimental_slab_response"):
            raise RuntimeError("rebuild PhaseSmith with experimental intensity response")
        result = _core.experimental_slab_response(
            np.ascontiguousarray(c, float), float(self.thickness_mm), float(self.attenuation_per_mm)
        )
        result.flags.writeable = False
        return result

    def to_dict(self):
        return asdict(self)
