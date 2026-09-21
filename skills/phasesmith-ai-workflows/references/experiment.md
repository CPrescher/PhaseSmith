# Experimental prerequisites and model scope

Use this reference before constructing a project or proposing a recipe. A
readable pattern and CIF establish data availability; they do not establish a
complete experiment or an identifiable refinement. Record what is supplied,
what is independently measured, and what remains an assumption.

## Establish the supported route

- The version-1 automation commands in this skill operate on a persisted
  `RietveldProject`: a constant-wavelength (CW) structural refinement, with
  monochromatic or fixed wavelength-component radiation and angular coordinates
  in degrees `2theta`. The exact installed schemas and parameter authorization
  govern what can be proposed.
- PhaseSmith also provides CW/TOF Le Bail and structural TOF workflows. Their
  existence does not make TOF inputs accepted by this CW automation contract.
  Do not relabel microseconds as degrees or invent a TOF `workflow-spec` schema.
- Structural TOF uses the separate `phasesmith.refinement.tof_structural` API.
  Its `StructuralTofMultiBankInput` accepts one built-in neutron phase, fixed
  reflection topology, and explicit bank-local models; it does not accept CW
  sample-physics providers. Use its own documented construction and validation.
- Le Bail intensities are extracted reflection intensities. They do not by
  themselves establish a structural model, site occupancy, or phase composition.

## Evidence to collect before a runnable CW project

1. **Observations:** coordinate identity and units, counts versus normalized
   intensity, uncertainties and their origin, selected range, masks, and any
   preprocessing. A generic column file can have unknown coordinate units.
   Negative observations do not justify silently clipping data or inventing
   counting uncertainties. Preserve the original arrays and preprocessing record.
2. **Radiation:** X-ray versus neutron, wavelength in angstroms, or explicit
   fixed component wavelengths and relative integrated intensities. A familiar
   filename or an apparent doublet is not evidence of a particular spectrum.
3. **Instrument and geometry:** the profile calibration and its applicable
   range; Bragg--Brentano versus capillary geometry where relevant; radius,
   displacement conventions, and axial divergence if modeled. Identify which
   settings are calibrated, fixed approximations, or proposed refinement terms.
4. **Intensity model:** the scattering provider and radiation compatibility,
   integrated-intensity correction, and any sample correction or orientation
   model. A neutral correction is an explicit modeling choice, not proof that
   experimental corrections are unnecessary.
5. **Structure and phases:** source and conversion provenance, cell and space
   group setting, sites, species, occupancies, displacement parameters, and all
   candidate phases supported by evidence. Review import warnings, especially
   assumed symmetry. The list of supplied CIFs cannot establish phase completeness.
6. **Background and objective:** distinguish a fixed supplied background from a
   refinable residual background; avoid accounting for the same contribution
   twice. State whether supplied uncertainties or unit weights define the fit.
   Record masks and the fitting range so later metrics remain comparable.
7. **Purpose and freedom:** distinguish profile reproduction, lattice estimation,
   composition, and microstructure goals. List justified parameter bounds,
   independent constraints, authorized parameter families, and finite budgets.
   More adjustable parameters do not create more experimental information.

`phasesmith inspect-pattern` and `phasesmith inspect-cif` provide bounded input
facts and digests. They do not infer the missing physical choices or construct
a runnable project. Use their unknowns and the readiness diagnostics to make a
specific list of missing decisions. Continue read-only inspection where useful;
do not guess essential metadata to make execution possible. Planning evaluates
no objective, so `ready_for_approval` is not evidence of an adequate fit.

For TOF, independently establish bank identity, microsecond observations,
calibration, detector angle, incident-spectrum normalization, integrated-intensity
correction, and the domain of any fixed background. An already-applied correction
must not be applied again. Facility names do not supply these choices. Named TOF
absorption, extinction, or texture models are outside the current structural
facade; explicit `none` or `already_applied` sample-correction declarations are
required by its file constructor.

## Profile conventions and calibration

For CW, `theta` is half the `2theta` coordinate, converted to radians inside
trigonometric functions. PhaseSmith uses

\[
\begin{aligned}
q &= U\tan^2\theta + V\tan\theta + W, \\
\mathrm{FWHM}_{G} &= \sqrt{8\ln 2}\sqrt{q}, \\
\mathrm{FWHM}_{L} &= X\sec\theta + Y\tan\theta.
\end{aligned}
\]

`q`, U, V, W are Gaussian variance in degrees squared; X, Y and component FWHMs
are degrees `2theta`. Require positive Gaussian variance and nonnegative
Lorentzian width over the modeled reflection domain. Do not copy coefficients
from another package without checking units, variance/FWHM convention, and X/Y
ordering. PhaseSmith's RMS microstrain is dimensionless RMS `delta-d/d`;
Lorentzian microstrain has a different definition. A Scherrer size is a coherent
domain size in nanometres with an assumed shape factor, not a particle size.

Use an independently characterized instrument/reference specimen when absolute
sample broadening matters. Refining instrument and sample broadening together
can create indistinguishable angular dependencies; see [interpretation](interpretation.md).
Calibration itself has uncertainty and a valid angular/experimental range.

`calibrate_fundamental_profile` is an offline compression of an explicit physical
target into the production CW profile. Inspect `accepted`, warnings, global and
per-peak residual diagnostics, and the requested range before using its output.
`accepted=False` is a failed representation check; do not relax limits merely to
label the output a calibration. The target requires actual spectrum/geometry
inputs and omits some instrument effects; it cannot reconstruct missing hardware
metadata from a pattern. It is not evaluated inside ordinary refinement trials.

Finite support and `ProfileAccuracy` are part of the numerical model. Preserve
their settings when comparing fits or resuming. A faster policy requires its own
accuracy evidence; changing support must not be presented as a solver-only speed
comparison or a new physical explanation.

## Further manual detail

These online pages track development; prefer the documentation for the installed
release when available. The bundled guidance above is usable without a checkout.

- [AI-guided automation](https://phasesmith.readthedocs.io/en/latest/ai-automation/)
- [CW profile conventions](https://phasesmith.readthedocs.io/en/latest/cw-profile/)
- [Sample physics](https://phasesmith.readthedocs.io/en/latest/sample-physics/)
- [Fundamental-profile calibration](https://phasesmith.readthedocs.io/en/latest/fundamental-profile-calibration/)
- [Structural TOF contract](https://phasesmith.readthedocs.io/en/latest/tof-structural-readiness/)
