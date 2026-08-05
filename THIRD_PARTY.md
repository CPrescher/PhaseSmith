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
