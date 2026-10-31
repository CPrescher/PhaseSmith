# Proposed shared single-crystal frame validity

Status: implemented as an experimental array/loader API; Dioptas consumer
migration and public namespace promotion remain follow-ups. See
[shared frame validity](single-crystal-frame-validity.md). The text below retains
the original proposal and ownership rationale. The MgSiO3 example currently
uses an explicit centre-ray DAC cone; PhaseSmith already consumes per-frame pixel
masks. A common frame-validity layer should now own shadow meaning across callers.

## Existing behavior inspected

Dioptas checkout `Dioptas-single-crystal`, commit
`651c27f98108fb19f60db9a8cb227755b2f9d6a2`, implements `FrameValidity`,
`SeatShadowOptions` and `build_seat_validity` in `dioptas/model/frame_validity.py`.
It estimates broad moving shadows from coarse original-frame illumination ratios
against illuminated reference views, retaining sufficiently large side-connected
components and a declared guard. Low global signal is diagnostic only; it does
not establish that a whole frame is blocked. Masks never change measured counts.
There is an explicit unavailable result when illuminated reference information
is insufficient. Its compressed/coarse snapshot preserves frame order, detector
shape, source hashes and provenance, without a full-resolution mask cube.

The numerical source is MIT-licensed. Its current entry point still depends on
Dioptas file readers, resource settings and display orientation; those must not
become PhaseSmith runtime dependencies. Dioptas separately owns mask preview,
reference-frame selection and persistent application caches. These files were
inspected, not modified or transplanted by this diagnosis.

PhaseSmith's `locate_series`, `integrate_series` and `prepare_profiles` already
accept per-frame `mask_loader`. Integration applies masks to foreground and
background pixels and retains source/mask fingerprints. However, the implementation now supplies
shared validity snapshots and a portable seat-shadow estimator.

## Proposed ownership and interface

A candidate normal namespace is `phasesmith.single_crystal.masking`, with the
usual experimental staging until public API/build promotion is reviewed.

- `FrameValidity`: immutable ordered-frame identity, detector shape, fixed mask,
  coarse moving exclusions, orientation convention and provenance; bounded
  `mask(frame)` access with True meaning excluded.
- `SeatShadowOptions` and `estimate_seat_shadows(frame_loader, ...)`: portable
  original-count estimator returning validity plus diagnostics or an explicit
  unavailable outcome. Callers supply arrays/readers and worker/resource limits.
  The existing Dioptas algorithm and tests provide compatibility targets.
- A separate geometric aperture provider: known DAC axis/opening and scan angles
  generate validity without intensity thresholds. It is an alternative or
  additional source of exclusions, not a universal 35-degree default.

Combine fixed/manual detector masks, reserved-value masks and moving shadows
once. The same immutable result must serve detection, positional calibration,
profile teachers, foreground/background integration, scaling eligibility and
search export. A predicted centre's visibility alone is insufficient: finite
pixel/omega supports must flag or explicitly model partially blocked rocking
profiles. Missing/blocked data must not turn into zero-valued intensity constraints.
Weak intensities, including negative values, remain measurements when their
coverage is valid.

Dioptas should retain UI controls, previews, user overrides, file loading and
application cache policy, and call this shared implementation. No Qt, HDF5 file
reader or application-owned global belongs in the numerical mask API. Coarse
mask estimation is optional preprocessing; profile kernels and fused analytical
derivatives remain native and unchanged.

## Validation before adoption

Preserve current Dioptas synthetic and Fe7C3 regression behavior, including
uniform dimming, partial shadows, guard/boundary semantics, native/display
orientation, fixed-mask composition, source order and unavailable references.
Verify masked pixels never teach a background/profile and that clipped rocking
support cannot pass as a fully observed reflection. Check serialization, source
identity, worker determinism and memory bounds. Compare its estimated mask with
the MgSiO3 aperture diagnostic using original counts and held-out equivalents;
retain any disagreement instead of choosing masks using structural R1.

The shared backend is now implemented experimentally. It preserves the existing
Dioptas numerical behavior on the documented compatibility fixture, adds full
exposure/pixel support for geometric masks, and leaves all Dioptas files unchanged.
Public API promotion and switching Dioptas to consume it remain follow-ups.
