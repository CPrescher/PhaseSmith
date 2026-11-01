# Dioptas original-frame workflow validation

The [application integration](dioptas-native-workflow.md) was tested using a local
ordinary PhaseSmith 0.8 wheel before publication. A fresh standalone e5b366c
worker and two real Dioptas Qt-button runs read the original MgSiO3 EIGER counts
with the frozen v40 scientific recipe. All large data and outputs stay external.
No standalone structure-refinement implementation was added.

The native numerical kernels/coordinator and acceptance implementation are
unchanged. The command-line file adapter moved into the optional facade I/O
module so the Python binding calls the same coordinator directly, releases the
GIL and reports stages without launching an external program. The CLI retains
its process guard; in-process calls use cooperative cancellation.

## Inputs and grading

The run uses original frames 2–143 inclusive, cell 8.8101 / 5.1824 / 18.2288 Å,
90° angles, element types Mg/Si/O, PONI geometry and the predeclared provisional
0.5° exposure convention, rotation axis (-1,0,0), row reversal and 35° DAC cone.
These are supplied assumptions, not blind acquisition/cell inference. Exact
space group, Laue group, atomic coordinates and counts are withheld. All eight
registered input files retain their original hashes.

Both workers finish and every prediction is hashed before reference coordinates
are parsed. The first retained candidate is graded; no reference-driven choice
among candidates is allowed. Both GUI runs recover **Pbca, Mg16 Si16 O48**, accept
the candidate under unchanged checks, and retain R1 **0.0514043045582282**.
Maximum/RMS reference-position errors are **0.006179 / 0.004263 Å**. This exposed
example is a regression control, not independent generalization evidence.

Compared with the fresh standalone run, both GUI runs preserve:

- All **68 arrays in 10 NPZ files** bitwise, including masks, detection/linking,
  geometry/indexing, profile bank, signed reduction and phasing density.
- The complete candidate records and phasing histories.
- Both the full-cell P1 CIF and selected-setting CIF, byte for byte.

There are **4,844 included observations, including 594 negative intensities**.
This is the unknown-symmetry v40 recipe, which retains primitive predictions;
it is not the older supplied-group recipe's 4,125-row reduction.

## Time and GUI behavior

| Measurement | Seconds |
| --- | ---: |
| Fresh standalone coordinator | 15.109 |
| First Dioptas coordinator | 15.144 |
| First complete GUI worker | 15.733 |
| Final Dioptas coordinator | 15.138 |
| Final complete GUI worker | 15.743 |
| Final through raw-frame preview | 15.854 |

The final coordinator breaks down as follows. Stage wall times include their
native output work; nested substeps are not added again.

| Stage | Seconds |
| --- | ---: |
| Shadow masks | 2.888 |
| Detection and linking | 3.079 |
| Indexing and geometry | 0.134 |
| Integration and profile learning | 5.602 |
| Profile fitting and scaling | 2.964 |
| Structure search | 0.468 |

These are sequential observations on one shared Apple M4 Pro, not randomized
performance trials. The small coordinator differences do not establish a speed
change. The GUI worker additionally includes boundary translation, metadata and
source verification.

A 25 ms GUI heartbeat fires 631 times in the final run; its maximum gap is
83.7 ms including the postprocessing image render (28.6 ms in the first run
without preview). Native HDF5 reads and processing still succeed with `h5py.File`
and `scipy.optimize.least_squares` blocked. This checks those specific fallback
paths; Python/Qt still owns input validation, image presentation and exports.

The final screen shows original frame 2, its -35 → -34.5° exposure, 20 saved
native spot observations and the 2068 × 2162 detector image. Preview payloads
must match the completed run's oriented frame hashes. Browsing uses bounded
native reads and does not replace the older lattice workflow or powder image.
Masks are not overlaid in this first preview. The initial empty legacy-view
screenshot is retained as the UI issue corrected during integration.

Both CIF and observation export buttons produce files identical to the native
products. The app displays **accepted search candidate**, and explicitly states
that full structure refinement has not been performed. Unit/Qt checks also cover
unaccepted-candidate reasons, cancellation, changed input/stale result handling,
existing-output rejection, worker exclusivity, source changes and orientation.
No normal user settings are saved by the validation harness.

## Package and test checks

Validation passes **567 workspace Rust tests**, strict all-target/all-feature
Clippy and the final file-host tests; **3,335 Python tests** pass with native CLI
cross-comparisons enabled. Eleven external-data tests remain skipped and 34 slow
cases remain deselected under the unchanged default policy. **178 Dioptas
single-crystal/settings tests** pass. Rust 1.89 minimal and all-feature facade
checks pass, along with API/version/format/reference-doc/skill checks.

The ordinary source distribution preserves the facade, file adapter and bundled
protocol, rebuilds into a wheel and passes native-capability/HDF5-filter smoke
checks in a fresh NumPy-only environment with optional numerical imports blocked.
The public API snapshot changes only the unreleased 0.8 surface. No runtime Python
dependency is added. Dioptas's registry lock is not given fictional 0.8 entries.
Cross-platform wheels/CI and wider user datasets remain release follow-up work.

The [compact receipt](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/dioptas-native-workflow-v1-macos.json)
pins source files, the application wheel, prediction barriers, grading, timings
and reconstruction checks. Full source data, outputs, test logs, screenshots,
exports and a ready-to-run Dioptas recipe are external under
`/Users/clemens/ChatGPT/PhaseSmith/dioptas-native-integration-20261101/`.
