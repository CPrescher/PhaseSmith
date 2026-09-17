# Pawley oracle coverage

The external oracle remains pinned by `PINNED_GSASII.json` to
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23` (#5838). No GSAS-II implementation
code was copied, translated, or added as a dependency.

`tests/test_pawley.py::test_extraction_from_existing_pinned_gsasii_profile_fixture`
uses the existing hash-validated `cw_instrument_profile_v1` overlapping-reflection
fixture as observed data, starts all three Pawley areas at zero, and checks
recovered areas and calculated signal. The profile-error bound is the existing
6e-6 fixture convention; the corresponding area tolerance is 8e-6 relative.
The fixture has not been regenerated. This checks intensity units and the native
objective against external numerical output. It does **not** compare two Pawley
optimizers or unresolved intensity assignments.

A read-only audit of the exact pinned `GSASIIscriptable.py` found no dedicated
Pawley constructor/reflection-initialization method. The public scripting
manual exposes `General/doPawley` through generic dictionary inspection, while
the data-organization manual documents `Pawley ref` as phase-owned state:

- [GSASIIscriptable public manual](https://gsas-ii-scripting.readthedocs.io/en/latest/GSASIIscriptable.html)
- [GSAS-II data organization](https://gsas-ii-scripting.readthedocs.io/en/latest/objvarorg.html)

Those online pages track newer revisions and are documentation context, not a
replacement pin. A full optimizer comparison needs a reviewed revision-gated
probe for initializing the private Pawley reflection records, and an explicit
F-squared/multiplicity/LP-to-area translation. That comparison is deferred;
this implementation does not claim live GSAS-II Pawley optimizer parity.
The independent NumPy active-face enumeration, analytical derivative tests,
rank/bound cases and checksum-pinned real-data runs validate the solver instead.

The independently designed least-squares objective follows Pawley's published
method: G. S. Pawley (1981), J. Appl. Cryst. 14, 357–361,
[doi:10.1107/S0021889881009618](https://doi.org/10.1107/S0021889881009618).
