# Versioned plain-data persistence

`rietveld.persistence` stores public models in a directory containing exactly
the library-owned `manifest.json` and `arrays.npz` files. JSON contains the
format version, typed records, units implicit in public field names, array
descriptors, and SHA-256 hashes. NPZ contains only numeric or boolean contiguous
arrays and is always loaded with `allow_pickle=False`.

```python
from rietveld import persistence

bundle = persistence.PersistenceBundle(
    pattern=pattern,
    instrument=instrument,
    experiment=experiment,
    fcj_geometry=geometry,
    wavelength_components=radiation,
    phases=(alpha, beta),
    rietveld_phases=(structural_alpha,),
    calculation_options=options,
    calculation_result=calculation,
    parameters=parameters,
    lebail_options=lebail_options,
    lebail_checkpoint=result.checkpoint,
    lebail_result=result,
    constraints=constraints,
    metadata={"sample_id": "example-17"},
)
persistence.save_bundle("analysis.rvp", bundle)
restored = persistence.load_bundle("analysis.rvp")
```

Format version 2 round-trips CW and TOF instruments, explicit monochromatic
experiment type, FCJ geometry, wavelength components, phases/reflections,
built-in physics providers, patterns, calculation inputs and results, typed
refinement parameters, Le Bail options, constraints,
complete Le Bail checkpoints, and complete Le Bail results including support
Jacobians and phase curves. It adds `rietveld_phases` with parser-independent
structures, structural reflection families, built-in X-ray/neutron scattering,
integrated-intensity corrections, phase scale, and optional reflection physics.
Format-1 bundles load through an explicit migration with no structural phases.
Bounds with infinite endpoints are represented by JSON `null`; non-finite
array values are rejected.

`restored.to_lebail_input()` reconstructs the complete typed input from the
persisted pattern, CW instrument, phases, parameters, and constraints. This
keeps restart scripts independent of internal record layout.

Saving to an existing directory requires `overwrite=True`. Only the two
library-owned filenames are replaced; unrelated files are not removed. Loading
checks the format version, archive hash, exact member set, every member's dtype,
shape and content hash, and numeric finiteness before constructing models.

## Third-party physics

Live Python objects are never pickled and package import never discovers code
implicitly. A third-party provider supplies a `PhysicsProviderCodec` explicitly
to both save and load. The record retains provider ID, provider version, and a
finite JSON configuration. Loading without the matching codec fails clearly.
This gives plugin packages control of migrations without granting a project
file permission to import arbitrary modules.

The format is intentionally independent of GSAS-II project files and GUI state.
Future incompatible schema changes increment `FORMAT_VERSION` and require an
explicit migration rather than silently guessing old units or fields.
The machine-readable top-level contract is
[`schemas/persistence-v2.schema.json`](../schemas/persistence-v2.schema.json).
The previous format remains documented by
[`schemas/persistence-v1.schema.json`](../schemas/persistence-v1.schema.json).
