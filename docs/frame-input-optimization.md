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
