# Rust workflow coordinator: MgSiO3 validation

The supported Rust coordinator now runs from original HDF5 counts through
shared masks, detection/linking, grain discovery, geometry, learned profiles,
scaling and structure search without Python. Both repeated full replays recover
Mg16 Si16 O48. Reduction parity passes the existing numerical tolerances;
**strict final-candidate parity fails**. The final R1 and coordinate differences
are retained below, rather than hidden by changing tolerances or search controls.

Use the [native workflow guide](single-crystal-native-workflow.md) for build/run
commands and typed host contracts. This is an exposed development/regression
example with supplied cell, symmetry and element types, provisional spindle
angles and explicit aperture assumptions. It is not an untouched validation set
or a standalone structure-refinement workflow.

## Implementation and reproducibility

The coordinator lives in `phasesmith-workflows::single_crystal`, re-exported by
the default Rust facade. HDF5/filter input stays in `phasesmith-hdf5`, numeric
archives in `phasesmith-persistence`, and JSON/CIF translation in the example
host. The host does not start or embed an interpreter. Minimal facade builds
retain their existing feature boundary. No new external dependency version was
introduced; the optional coordinator reuses existing numerical crates and SHA-2.

The first recipe deliberately admits a supplied primitive-orthorhombic metric,
exact symmetry and elements, 5–225 exposures and the existing versioned scientific
controls. It rejects unsupported recipe settings instead of ignoring them.
Source/observer failures, cancellation, source changes, ordered callbacks,
independent payload/cache/worker budgets and partial mask blocks have native
test coverage. Every value/derivative kernel retains its original owner and
finite support; the independent Python reference equations remain available.
The optional observer verifies source hashes and reads the reference CIF only
after the native process stops.

Frozen protocols v26–v29 were committed before their corresponding frame reads.
The complete search-control expansion, source priors and 64 null rotations are
literal assets. Original cross-backend tolerances are unchanged. Performance
changes additionally require bitwise equality to the corrected native trial.
The missing angular-truncation flag found in trial 01 was fixed before timing
the accepted path; the failed output is preserved.

All eight source files (470,052,794 bytes), intermediate arrays, trial results,
diagnostic searches, executable and distributions remain outside Git at:

```text
/Users/clemens/ChatGPT/PhaseSmith/native-coordinator-0.8-20261031
```

The machine-readable receipt is
`validation/results/native-coordinator-0.8-macos.json`. It records exact source,
protocol, executable and output hashes, timing observations and gate outcomes.
The standalone executable and its three small assets were also rebuilt from a
clean Git source archive outside the checkout. The Python sdist intentionally
omits the Rust facade; Cargo package inspection includes its example/assets.

## Full replay comparison

The Python control uses frozen v25 and the ordinary installed 0.8 wheel. It
already delegates its heavy numerical kernels to Rust; this comparison measures
the remaining coordinator/array construction and execution changes. Native runs
use v29 and executable source `9b30fab`. Both observed paths verify the same
eight source files and grade the saved structure after the worker exits.

| Check | Observed difference / result | Existing gate |
| --- | --- | --- |
| Original masks, positions, HKL assignments, family partitions and edge flags | Identical | Pass |
| Exported HKLs, use mask, quality flags and negative count | Identical: 4,125 exports, 220 negative | Pass |
| Profile teachers / retained target groups / pixel samples | 246 / 7,277 / 13,267,065 | Same membership/order |
| Maximum prediction xy / omega difference | 1.42e-12 px / 3.41e-13 degrees | Pass: 1e-4 px / 1e-6 degrees |
| Intensity / sigma relative L2 difference | 4.45e-6 / 1.17e-5 | Pass: 1e-4 each |
| Maximum / 95th-percentile intensity difference in baseline sigma | 0.001024 / 1.08e-5 | Pass: 0.1 / 0.01 |
| Recovered composition | Mg16 Si16 O48 | Pass |
| Native candidate vs external reference, RMS / maximum | 0.00466 / 0.00683 Å | Pass: labelled structure, including strict 0.02 Å assessment |
| Python / native final R1 | 0.04085182070733226 / 0.041128354912261904 | Both meet search target |
| Absolute R1 difference | 0.0002765342 | **Fail: 0.0001** |
| Candidate coordinate difference, RMS / maximum after alignment | 0.000838 / 0.001868 Å | **Fail: maximum 0.001 Å** |

The strict comparator reports 15 passing gates and two failing gates. The
inherited cross-backend bitwise contract is also not met: small least-squares,
Jacobian/design and SVD constructions now use native scalar math/nalgebra rather
than NumPy. No global tolerance or recipe was relaxed. The ordinary Python
control still reproduces the preserved v25 arrays, complete candidates and
phasing histories bitwise.

Performance changes after the angular-flag correction preserve **all saved
numerical arrays, full candidate records and phasing histories bitwise** between
corrected trial 02, cached/pipelined trials and both final native runs. The direct
standalone run also matches. Added aperture/inactive-DAC diagnostic arrays are
recorded separately; no prior array was removed. Independent Gemmi parsing of
the final small-molecule CIF confirms 80 sites, composition and finite
coordinates/anisotropic ADPs.

### Why final R1 changes

Two diagnostic searches run the same native executable with saved original and
native intensities, explicitly separated from every raw-frame replay. They
reproduce the corresponding baseline and native R1 exactly. Thus this is not a
different Rust structure-search implementation or a changed search expansion.

Common-objective evaluations further isolate the effect:

| Retained model | R1 against original reduction | R1 against native reduction |
| --- | ---: | ---: |
| Python-coordinated candidate | 0.04085182070733236 | 0.04085177827505600 |
| Rust-coordinated candidate | 0.04112840013249510 | 0.04112835491226143 |

Swapping data for a fixed model changes R1 by about 4e-8; most of the 2.77e-4
gap belongs to the retained optimizer state. Both internal candidate polishes
terminate at `max_iterations`, with 54 versus 53 accepted steps and a minimum
ADP eigenvalue near the existing 1e-5 Å² lower bound. The evidence points to the
documented bound-sensitive candidate-polishing trajectory responding to small
input roundoff. It does not establish convergence or excuse the failed gates.
The native fixed model has a slightly higher objective on either dataset.
Changing this optimizer needs a separate numerical change and broader validation;
no solver, seed, damping, iteration count or ADP constraint was tuned here.

## Timing and measured optimization steps

Machine: Apple M4 Pro, 14 logical CPUs, 48 GiB RAM, macOS 26.6.2, release Rust
1.99.1. Replays ran sequentially after compilation/tests, in order Python 1,
native 1, native 2, Python 2. These are two warm-cache observations per path,
not a confidence interval or a cold-storage benchmark. Worker budgets remain
explicit: two preparation workers, four coarse/integration/learning workers,
eight detection workers and ten profile-fitting workers; search uses one worker.

| Measurement | Previous Python coordinator | Rust coordinator |
| --- | ---: | ---: |
| Full observed process, including verification/grading | 19.09–19.27 s | 16.24–16.57 s |
| Mean observed process | 19.18 s | 16.405 s |
| Numerical worker / coordinator interior | 18.11–18.26 s worker | 15.30–15.33 s coordinator |

The observed full-process reduction is **14.47%**, or **1.17× throughput**.
The standalone executable, launched with only `/usr/bin:/bin` in PATH, no HDF5
plugin path and no inherited environment, takes **15.81 s** including its own
raw/PONI verification. That standalone measurement verifies two numerical input
files; the fair observer comparison above verifies all eight registered files
and independently grades the reference.

Mean stage wall times, including their synchronous archives:

| Area | Python coordinator | Rust coordinator | Native share |
| --- | ---: | ---: | ---: |
| Shared shadow masks | 2.811 s | 2.911 s | 19.0% |
| Detection, linking and support checks | 3.715 s | 3.189 s | 20.8% |
| Grain discovery, indexing and geometry | 0.262 s | 0.131 s | 0.9% |
| Pixel collection, background and profile learning | 6.601 s | 5.500 s | 35.9% |
| Profile fitting, corrections, empirical scaling and export | 3.921 s | 3.001 s | 19.6% |
| Structure search and saved candidate | 0.746 s | 0.579 s | 3.8% |

Native rows partition coordinator wall time. Python integration/learning ends at
the measured `reduce_observations` entry; the remaining enclosing integration
stage belongs to fitting/scaling/export. Search includes Python child startup
on that path. Parent verification, imports, grading and uninstrumented worker
overhead explain the difference from full-process time; nested inclusive times
are never added together. Shadow masking is slightly slower in the native path;
the speed claim concerns measured totals, not every individual stage.

The native implementation was measured between each execution optimization:

| Corrected native trial | Coordinator wall | Reduction from preceding trial |
| --- | ---: | ---: |
| Sequential original port, trial 02 | 23.428 s | — |
| Cached DAC blocks and bounded coarse work, trial 03 | 20.733 s | 11.5% |
| Ordered overlap of preparation and frame tasks, trial 04 | 16.479 s | 20.5% |
| Exact whole/partial row-block fills, final two-run mean | 15.313 s | 7.1% |

These cumulative changes reduce native coordinator time by 34.6% from its first
corrected port. Earlier step observations are single runs; their saved-array
identity checks establish numerical preservation independently of timing noise.

The largest remaining area is pixel collection/profile learning (36%), followed
by detection (21%), profile fitting/scaling (20%) and masks (19%). Further work
should first profile repeated frame/mask materialization and background/support
traversal within those areas, retaining exact output order and bounded payloads.
Increasing every worker count is not supported by these measurements. Indexing
and search together use under 5%; they are lower priorities for this example.

## Validation

- Rust 1.89.0, complete workspace/all features: **526 passed, zero failed,
  34 ignored**; the two example hard-deadline tests also pass.
- Ordinary installed wheel: **3,284 passed, 11 skipped, 34 deselected**, including
  frozen-asset equality, independent numerical references and public API checks.
- Strict Clippy/all targets/all features, Rustdoc and formatting pass. Minimal
  facade/Python-extension builds and NumPy-only wheel smoke pass, including
  bundled Bitshuffle/LZ4 HDF5 without optional numerical Python packages.
- The Cargo example rebuilds from the external Git source archive. Its direct
  full replay requires no interpreter or installed HDF5/filter distribution.
- Ruff, public API/version/doc synchronization, strict MkDocs and generated-site
  local-link checks are recorded in the machine receipt. External URLs are not
  exhaustively live-checked.

GSAS-II does not cover these single-crystal orchestration operations. Existing
kernel finite-difference/reference tests, new analytical Jacobian/penalty tests,
frozen real-frame replay and separate source/reference assessment provide the
relevant evidence. No golden data, historical receipts or failed trial output
was rewritten. Structure recovery is supported here; strict final-candidate
equivalence and converged refinement are not claimed.

After preserving the executable, distributions, hashes and validation logs,
cleanup removed 6.20 GiB of allocated rebuildable space: this checkout's Cargo
target, the generated documentation site and temporary source reconstruction /
Git archive. The source commit and successful reconstruction log remain recorded;
raw data, all trial arrays, other checkouts and shared package/compiler caches
were preserved.
