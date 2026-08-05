# Third-party dependency ledger

Rietveld Engine source is MIT licensed. Optional dependencies keep their own
licenses and are not vendored.

## Gemmi

- Purpose: optional CIF syntax parsing and space-group name/setting resolution.
- Tested range: `gemmi>=0.7.5,<0.8`.
- Runtime status: optional `cif` extra; not imported by the base package path.
- Upstream: <https://github.com/project-gemmi/gemmi>
- Documentation: <https://gemmi.readthedocs.io/en/stable/>
- License: Mozilla Public License 2.0, or LGPL v3 at the user's option, as
  declared by upstream and the Python package metadata.
- Vendoring/modification: none.
- Boundary: parser objects remain inside `rietveld.io._gemmi`; public and
  persisted models contain only independently defined Rietveld Engine types.

Gemmi is not a source for diffraction equations, reflection intensities, or
scattering tables. Exact symmetry validation and all numerical diffraction
work remain native Rietveld Engine implementations.

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
