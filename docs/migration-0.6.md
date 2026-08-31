# Migrating to 0.6

PhaseSmith 0.6 makes the documented module ownership visible in the exported
Python namespaces before the 1.0 compatibility boundary. Numerical equations,
units, finite support, defaults, persistence formats, and result values are
unchanged.

## Refinement methods use their owning modules

`phasesmith.refinement` now exports shared parameter, constraint, residual,
lattice, runtime, background, and optimizer infrastructure. Import a method's
records and entry point from its named module:

```python
# 0.5 aggregate imports
from phasesmith.refinement import LeBailInput, LeBailOptions, refine

# 0.6 durable imports
from phasesmith.refinement.lebail import LeBailInput, LeBailOptions, refine
```

The same rule applies to:

- `phasesmith.refinement.rietveld` for Rietveld records;
- `phasesmith.refinement.workflow` for staged Rietveld recipes;
- `phasesmith.refinement.readiness` for readiness review;
- `phasesmith.refinement.tof_lebail` for fixed-instrument TOF Le Bail;
- `phasesmith.refinement.tof_multibank` for joint TOF geometry refinement; and
- `phasesmith.refinement.tof_structural` for structural TOF refinement.

The named modules themselves are exported from `phasesmith.refinement`, so
`from phasesmith.refinement import lebail` remains supported. This removes the
method-ambiguous aggregate name `phasesmith.refinement.refine` from the public
snapshot without renaming the method-owned function.

## General I/O is separate from dataset conversion

The `phasesmith.io` aggregate contains the general `cif`, `powder`,
`space_groups`, and `tof_instrument` adapters and their principal records and
functions. Validation-dataset converters now use their existing named modules:

```python
# 0.5 aggregate import
from phasesmith.io import convert_rowles_topas_bundle

# 0.6 durable import
from phasesmith.io.topas import convert_rowles_topas_bundle
```

The same migration applies to `bath_ltl`, `iucr_silicon_standard`,
`iucr_sodium_citrate_silicon`, `iucr_tripotassium_citrate_silicon`,
`iucr_trirubidium_citrate_silicon`, and `xred`. These converters remain
available; only their promotion into the general I/O namespace changed.

## Top-level aliases

The top-level namespace remains a convenience surface for domain models,
calculation functions, and mathematical primitives. APIs with a distinct
adapter or orchestration contract now use their owner:

| 0.5 import | 0.6 durable import |
| --- | --- |
| `from phasesmith import WorkflowSpec` | `from phasesmith.automation import WorkflowSpec` |
| `from phasesmith import read_cif` | `from phasesmith.io.cif import read_cif` |
| `from phasesmith import read_powder_data` | `from phasesmith.io.powder import read_powder_data` |
| `from phasesmith import read_gsas_tof_instrument` | `from phasesmith.io.tof_instrument import read_gsas_tof_instrument` |
| `from phasesmith import space_group_by_number` | `from phasesmith.io.space_groups import space_group_by_number` |
| `from phasesmith import review_rietveld_input` | `from phasesmith.refinement.readiness import review_rietveld_input` |
| `from phasesmith import write_rietveld_json` | `from phasesmith.reporting import write_rietveld_json` |

All corresponding automation records, schema constants, and workflow
functions follow the first row's rule. The top-level `automation` module itself
remains exported.

## Compatibility window

The moved 0.5 spellings are absent from the 0.6 `__all__` lists and public API
snapshot. Explicit access still resolves to the same object in 0.6 and emits a
`DeprecationWarning`. This compatibility layer performs no data conversion and
does not change numerical behavior. It is scheduled for removal at 1.0, so
applications should migrate to the module-qualified paths now.
