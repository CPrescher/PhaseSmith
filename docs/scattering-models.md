# Scattering models and data provenance

Status: implementation unit 14 design and source gate approved; kernels and
public providers are in progress.

This document is the source ledger for built-in atomic scattering data. GSAS-II
is not a source for equations, coefficients, species names, or table contents.
It may later produce external comparison fixtures only.

## Public model boundary

Scattering is a reflection-batch calculation with API version 1. A provider
receives:

- stable typed species identities, including element, optional isotope, and
  optional charge/state;
- a contiguous one-dimensional array of
  `s = sin(theta) / wavelength = 1 / (2 d) = Q / (4 pi)` in inverse ångströms;
- an explicit probe/model identity.

It returns reflection-major complex amplitudes with shape
`(reflection_count, site_count)`, the derivative `df/ds` with the same shape,
an amplitude-unit label, and immutable provider provenance. The derivative is
required even when it is identically zero so later cell-derivative chaining
does not branch by model.

Built-in providers resolve and cache each unique species in Rust. A third-party
Python provider may implement the same versioned batch contract, but it is
called once for the complete species/reflection batch and never once per atom,
reflection, or profile sample. Providers are passed explicitly; there is no
global registry. Complex output reserves the same interface for later anomalous
or magnetic models without claiming that those models exist today.

## Non-resonant X-ray form factors

The first X-ray model uses the Waasmaier--Kirfel five-Gaussian fit:

```text
f0(s) = c + sum(i=1..5) a_i exp(-b_i s^2)
df0/ds = -2 s sum(i=1..5) a_i b_i exp(-b_i s^2)
```

`f0` is real and measured in electrons. The source declares the fit valid for
`0 <= s <= 6 inverse ångströms`; evaluation outside that interval is an error,
not an undocumented extrapolation. At `s = 0`, neutral-atom values are checked
against the atomic number within the serialization accuracy of the source fit.

The committed table is generated from:

- upstream: XrayDB, <https://github.com/xraypy/XrayDB>;
- pinned commit: `663d2171bd301dc51dbe048cae459934e60347c2`;
- source file: `data_sources/waasmaeir_kirfel.dat`;
- source SHA-256:
  `047208c2e0e48808cc8a01eaf1e79cd133ba448c583d922ef162000771339089`;
- upstream status: the source file and database are dedicated to the public
  domain under CC0 1.0 by the upstream `LICENSE`;
- scientific source: D. Waasmaier and A. Kirfel, *Acta Crystallographica A*
  **51**, 416--431 (1995), DOI `10.1107/S0108767394013292`.

All neutral and ionic rows in that pinned source are retained with their exact
source labels. Public typed species resolve an ionic row only when a matching
state exists; there is no silent ionic-to-neutral fallback. Callers may select
the neutral model explicitly when that approximation is intended. Anomalous
`f' + i f''` is not part of this model.

## Coherent neutron nuclear scattering

The first neutron model uses the real bound coherent nuclear scattering length
`b_c` in femtometres. For this baseline model,

```text
f(s) = b_c
df/ds = 0
```

Natural-abundance element rows and isotope rows are separate identities.
Deuterium and tritium are canonicalized as hydrogen isotopes 2 and 3. Missing
isotopes are errors; the resolver never substitutes a natural-abundance value.
Rows marked energy-dependent by the source are retained with a flag but are
rejected by the constant model. A future wavelength-dependent provider must
model them explicitly. Absorption, incoherent scattering, magnetic scattering,
and resonant interpolation are outside this slice.

The committed table is generated from:

- upstream: `periodictable`,
  <https://github.com/python-periodictable/periodictable>;
- pinned commit: `182ef63a9ec118ef725aae5bb81860f4ba0fb573`;
- source file: `periodictable/nsf.py`, embedded `nsftable`;
- source SHA-256:
  `47c5f841100c91fb1d063be9503d0a1bc35a80ae1e9d0791f387ad05e5589ed9`;
- upstream status: `nsf.py` explicitly declares itself public domain and the
  repository license declares `periodictable` public domain;
- source ledger: the upstream file records the selected Rauch/Sears tables and
  per-entry literature updates, including uncertainty and energy-dependence
  notes. Those notes remain upstream provenance; the generator does not infer
  replacement values.

The generated native record retains the source value, standard uncertainty
when present, natural/isotope identity, and energy-dependent flag. Only the
real `b_c` value participates in this baseline amplitude.

## Deterministic generation

`tools/generate_scattering_tables.py` accepts two local upstream checkouts. It
verifies both commit IDs and source checksums, parses only the named source
records, normalizes stable keys, rejects duplicates or malformed numbers, and
writes deterministic Rust tables. Generated files carry their origin and must
not be edited by hand. Regeneration is an explicit maintainer action; normal
builds and tests need neither upstream project nor network access.

Tests independently parse compact authored coefficient cases and compare the
Rust equation and derivative against NumPy. Table-integrity tests check row
counts, unique keys, source metadata, known species, exact serialized values,
zero-angle limits, isotope separation, and rejection of missing or
energy-dependent species. The full-table checksum is local to the generated
file so accidental edits cannot be hidden by loosening numerical tolerances.

## Unit 14 execution sequence

1. Commit this source/license and interface decision before table generation.
2. Add the checksum-verifying deterministic generator and generated native
   tables.
3. Implement dependency-free Rust lookup, prepared unique-species caches,
   X-ray/neutron batch values, and `df/ds` together.
4. Add thin PyO3 prepared-model bindings and the typed
   `rietveld.scattering` provider protocol.
5. Add an independent NumPy reference, boundary/invalid-input tests, table
   integrity tests, and finite-difference derivative tests.
6. Benchmark prepared multi-species batches and review that lookup is outside
   the reflection/site hot loop.
7. Run the full Rust/Python quality gate, update project status, review the
   complete diff, and commit the numerical slice.
