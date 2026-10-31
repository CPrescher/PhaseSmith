# Measured original-frame input optimization

Every stage preserves original integer counts, masks, source identities and
all scientific controls. Native detector/profile/geometry equations and finite
support remain unchanged; independent numerical references stay authoritative.
Each full protocol is committed before frames, and each full run must reproduce
all retained v11 arrays/candidates/search histories bit for bit. Large input and
output products remain outside Git.

## 1. Reuse a freshly verified digest

Validity providers now return their freshly computed digest after checking the
current original frame. Combined providers still execute every check. Geometric
providers return None; ordinary hashing remains the fallback. A digest is reused
only for the same unchanged owned frame, never supplied unchecked by the caller.

At eight detection workers, three-repeat median time falls from 5.003 s to
4.138 s (1.21x), with all positions and frame fingerprints byte-identical.
Fresh full v21 runs take 32.34/23.48 s CLI and both pass all identity gates.
First-use indexing/setup and archive-write variation mean this is not evidence
of an overall improvement over the v20 23.46/23.76 s pair. These full measurements
are retained, not replaced by the isolated gain. Seventy-five focused tests pass.
The complete record is `benchmarks/enstatite-frame-input-stage1-macos.json`.

## 2. Native count preparation

`phasesmith_spots::frame_input` accepts owned original u32 snapshots and an
immutable scan plan. It applies the declared orientation in place, computes a
native-endian SHA256 once, verifies every inferred-provider digest, and composes
the reserved-count/block/fixed/DAC masks. The DAC kernel, whole-block expansion,
closed exposure and finite support are unchanged. SHA256 and complete masks are
checked independently against hashlib and the ordinary array frontend, including
partial edge blocks, fixed masks, reversed rows, zeros and high/reserved counts.
Expected hashes come from mask provenance; no unchecked current digest is accepted.

A private Rust pool prepares independent frames; results/errors remain ordered.
The byte cap admits input snapshots, native owned counts, masks, temporary DAC
mask and digest scratch before native copies. Plan storage, downstream detector
workspace and thread stacks are separate. Native row errors are delivered when
that row is requested, preserving earlier valid results/callbacks; no partial
scientific workflow result is returned on failure. The optional frontend batches
ordered HDF5 reads; existing numerical workers can overlap these native batches.

The initial native trial regresses to 6.042 s and is retained. Its software-only
ARM SHA backend and per-pixel mask divisions were corrected with upstream sha2's
AArch64 hardware feature (runtime CPU fallback) and row-run mask expansion. No
handwritten cryptography or numerical equation change is introduced. Cargo pins
sha2/sha2-asm; other architectures retain the ordinary upstream path. Corrected
four-frame batches take 4.325 s; one/two-frame batches 4.038/3.674 s. Every trial
keeps exact original positions/fingerprints. Two preparation workers plus eight
detector workers stay within the ten-worker execution choice. The full v22
example explicitly selects two-frame batches and a 256 MiB payload cap.

## 3. Source-bound decoded counts

`phasesmith_hdf5::cache::CachedReader` adds an optional 0–4 GiB decoded-payload
cap. It admits the first unique original frames that fit, without eviction or
scan-thrashing, and returns independent owned copies. Cache payloads never
contain masks, normalized counts or refined information. One fresh open source
and dataset own each cache, with canonical path, size, timestamps and platform
file identity checked before reuse. Changed/missing sources clear the cache
and fail until reopening; indirect/external storage is rejected for cached reads
until separate source validation exists. The uncached reader remains available.

Fallible cache reservation can decline admission and retain streaming. Payload
capacity excludes metadata, caller copies and ordinary workspace; process RSS
is measured separately. Caller mutation cannot alter cached counts. Tests cover
source changes, isolation between handles, tiny-budget streaming, bounds and
external links. Full runs create an empty cache and never read archived arrays.
It is shared across raw masking/detection/collection and closed before search.

The v23 example requests 1 GiB, which admits about 60 of 142 original frames.
The choice is a memory/payload control, never an atomic-reference or R1 choice.
Exact cache hit/decode/admission counters and observed performance are recorded
separately from native preparation. All physics, support, geometry and independent
reference/oracle fixtures remain unchanged.
