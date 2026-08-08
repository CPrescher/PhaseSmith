# Third-party dependency ledger

PhaseSmith source is MIT licensed. Optional dependencies keep their own
licenses and are not vendored.

## xypattern Smooth Bruckner behavior

- Purpose: compatibility semantics for the optional native Smooth Bruckner
  background-estimation preprocessing path.
- Upstream: <https://github.com/CPrescher/xypattern>
- Pinned commit/release: `6e4574d75d2d6fcefc633f9fbecc27b8f1bcd817`
  / 1.2.3.
- Source: `xypattern/util/smooth_bruckner.pyx` and the composition in
  `xypattern/auto_background.py`.
- License: MIT; the complete upstream notice is retained in
  `THIRD_PARTY_NOTICES.md` and included in package license files.
- Runtime status: no dependency. PhaseSmith owns its native implementation
  and independent NumPy reference; an isolated pinned install is used only for
  compatibility verification.
- Boundary: `phasesmith.background` is plain-array preprocessing. It does not
  import xypattern or Dioptas and does not enter differentiable refinement
  JVP/VJP calculations.

## Moyo

- Purpose: compiled-in pure-Rust conventional space-group identifiers and
  exact symmetry operations for all 530 Hall settings.
- Pinned release: `moyo = 0.15.0`.
- Upstream: <https://github.com/spglib/moyo>
- Database lineage: the provider's Hall-symbol database follows spglib.
- License: MIT OR Apache-2.0; PhaseSmith distributes it under the MIT option.
- Runtime status: required native dependency of `phasesmith-io`; no Python or C
  runtime is involved.
- Boundary: imported operations are converted to PhaseSmith integer/rational
  types and pass independent group-closure validation before use.

## Gemmi

- Purpose: optional alternative CIF parser and differential validation oracle.
- Tested range: `gemmi>=0.7.5,<0.8`.
- Runtime status: optional `cif` extra; neither the base package nor default CIF
  import loads it.
- Upstream: <https://github.com/project-gemmi/gemmi>
- Documentation: <https://gemmi.readthedocs.io/en/stable/>
- License: Mozilla Public License 2.0, or LGPL v3 at the user's option, as
  declared by upstream and the Python package metadata.
- Vendoring/modification: none.
- Boundary: parser objects remain inside `phasesmith.io._gemmi`; public and
  persisted models contain only independently defined PhaseSmith types.

Gemmi is not the default parser and is not a source for diffraction equations,
reflection intensities, or scattering tables. Exact symmetry validation and
all numerical diffraction work remain native PhaseSmith implementations.

## XrayDB Waasmaier--Kirfel data

- Purpose: generated built-in coefficients for non-resonant X-ray form factors.
- Upstream: <https://github.com/xraypy/XrayDB>
- Pinned commit: `663d2171bd301dc51dbe048cae459934e60347c2`.
- Source: `data_sources/waasmaeir_kirfel.dat`, SHA-256
  `047208c2e0e48808cc8a01eaf1e79cd133ba448c583d922ef162000771339089`.
- License: upstream dedicates the source data and database to the public domain
  under CC0 1.0.
- Vendoring/modification: source checkout is not vendored. A deterministic
  generated Rust table retains numerical rows and source identifiers.

Scientific attribution and conventions are recorded in
`docs/scattering-models.md`.

## periodictable neutron data

- Purpose: generated built-in real bound coherent neutron scattering lengths.
- Upstream: <https://github.com/python-periodictable/periodictable>
- Pinned commit: `182ef63a9ec118ef725aae5bb81860f4ba0fb573`.
- Source: `periodictable/nsf.py`, SHA-256
  `47c5f841100c91fb1d063be9503d0a1bc35a80ae1e9d0791f387ad05e5589ed9`.
- License: `nsf.py` explicitly declares itself public domain; the repository
  license declares the package public domain except for separately identified
  files not used here.
- Vendoring/modification: upstream Python is not vendored. A deterministic
  generated Rust table retains the selected values, uncertainties, identity,
  and energy-dependence flags.

The upstream file's own literature ledger is preserved as the data provenance
record. The built-in constant model rejects energy-dependent rows.
