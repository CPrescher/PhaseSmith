# Pre-1.0 API decisions

This record fixes the intended 1.0 direction without claiming that the current
0.x package is already compatibility-stable. It separates mathematical
primitives, normal domain workflows, compatibility adapters, and validation
artifacts so that a convenience import does not blur their contracts.

## Public surface tiers

The 1.0 documentation will use three explicit tiers:

1. **Domain APIs** are the normal scripting surface: typed instruments,
   radiation, phases, patterns, calculations, refinement requests/results, and
   persistence. These use physical units and typed records.
2. **Mathematical primitives** are public for equation tests, research, and
   advanced composition. They expose lower-level parameterizations and require
   the caller to own their physical interpretation.
3. **Adapters and validation APIs** translate an external format or convention.
   They are named for that convention and do not alter the domain API.

The `phasesmith.oracle` and `phasesmith.validation` modules are not runtime
dependencies of normal calculation or refinement. Their fixture schemas and
dataset manifests are versioned independently of the numerical API.

## Resolved decisions

### Direct `H, eta` remains a public primitive

`profile(delta, fwhm, eta)` and the corresponding `accumulate` interface remain
public. They are small, useful equation-level contracts and are already covered
by independent reference, normalization, derivative, and finite-support tests.
They are explicitly documented as mathematical primitives, not as the
recommended way to describe instrumental broadening. Normal diffraction
workflows should use component widths or typed CW/FCJ/TOF instrument models.

This decision preserves a useful exact interface without implying that `eta`
is an independently meaningful instrument parameter.

### Support-block Jacobians remain the Python default

The current default is retained for 1.0. `SupportJacobian` stores only the
inclusive finite-support samples for each peak, so memory scales with evaluated
work rather than `peak_count * parameter_count * sample_count`. Dense local
Jacobians remain available only through `jacobian_layout="dense"` or
`SupportJacobian.to_dense(sample_count)`.

This is already consistent across low-level, CW, FCJ, TOF, prepared-pattern,
and structural calculation entry points. Changing the default back to dense
would turn a compatibility allocation into the normal memory contract.

### Physical units are canonical; GSAS compatibility is an adapter

The stable domain API uses degrees, degrees squared, ångströms, millimetres,
micrometres, nanometres, RMS strain, and explicit integrated-intensity
corrections as named by each typed field. It will not adopt centidegree,
variance, project-dictionary, or profile-function-code semantics merely because
an external program stores them.

GSAS/GSAS-II file readers and `phasesmith.oracle.conventions` may perform
explicit, tested conversion at their boundary. Compatibility names remain out
of normal constructors and numerical kernels. A new conversion must document
its source equation, units, sign, validity range, and round-trip or black-box
test before it becomes public.

### Oracle fixtures stay narrow and provenance-gated

The existing fixtures are retained because they contain plain numerical
outputs needed for black-box regression tests, not GSAS-II runtime objects,
project dictionaries, or implementation code. Each retained fixture must have:

- an exact upstream repository revision and generator hash;
- a manifest declaring public/private probe use and native units;
- finite plain arrays with member and archive hashes;
- an independently derived PhaseSmith equation or implementation under test;
- a documented reason the fixture remains useful; and
- the upstream notice/citation recorded in the oracle documentation and
  distribution notices.

No GSAS-II source, compiled binary, project file, or mechanically translated
implementation may be added as a fixture. A new golden is opt-in, reviewed as
a numerical diff, and never regenerated implicitly. This is a conservative
release policy, not legal advice; redistribution questions that fall outside
plain black-box outputs require maintainer or counsel review before commit.

### Scattering tables remain source-reviewed additions

The current X-ray and neutron tables remain because their pinned sources,
hashes, scientific references, licenses/public-domain status, generation path,
and numerical checks are recorded in [Scattering models](scattering-models.md).
No additional table is admitted solely because another refinement program
contains it. Every new table needs an independently reviewable upstream source,
redistribution status, exact version/hash, generated-file provenance, and
scientific validation.

## Compatibility after 1.0

Removing or renaming a stable field, changing units/order, changing finite
support, or changing a default representation requires a documented major
version or a versioned migration path. Additive diagnostics and optional fields
may be introduced compatibly. Persistence and provider formats keep their own
format/API versions; accepting an older format does not make internal Rust ABI
or serialized implementation details public.

The first machine-readable snapshot was checked in as
`api/python-public-api-v0.4.1.json`. The cleanup baseline is
`api/python-public-api-v0.6.0.json`, and the reconciled candidate is
`api/python-public-api-v0.7.0.json`. Each record contains every name in
`__all__` for the six explicitly exported namespaces, their
domain/adapter/validation tier,
implementation target, and callable signature when introspection supports one.
`scripts/public_api_snapshot.py --check` regenerates the record without
timestamps, platform paths, or evaluated annotations and reports a reviewable
unified diff on any change.

The exact gate intentionally rejects additions as well as removals, aliases,
or signature changes. An intentional 0.x change therefore requires a new
versioned snapshot and changelog review; older snapshots remain immutable
release records. This establishes the baseline needed for a 1.0 release
candidate without retroactively promising compatibility for earlier 0.x
versions.

The 0.6 cleanup makes module ownership explicit. The top level no longer
exports automation, I/O-adapter, readiness, or reporting serialization members individually;
`phasesmith.refinement` exports shared infrastructure and named method modules;
and `phasesmith.io` exports general adapters rather than dataset-specific
validation converters. Explicit 0.5 spellings resolve with
`DeprecationWarning` during 0.6 and 0.7 and are scheduled for removal in 1.0. This
pre-1.0 compatibility window is documented in [Migrating to
0.6](migration-0.6.md); it does not restore those aliases to `__all__` or the
new public snapshot.
