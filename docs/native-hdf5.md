# Bundled native HDF5 input

`phasesmith-hdf5` is a read-only detector-frame adapter, separate from numerical
kernels. The Rust facade exposes `phasesmith::hdf5` with feature `hdf5`; the Python
extension uses the same feature and offers `phasesmith.io.hdf5.Hdf5Frames`.
Prebuilt release wheels enable this feature. The default Rust facade remains
free of HDF5 dependencies.

## Building and distribution

The feature builds pinned native sources through `hdf5-metno` 0.15.1,
`hdf5-metno-src` 0.10.4 and `hdf5-bitshuffle` 1.0.0. HDF5, zlib, LZF and
Bitshuffle/LZ4 are linked into the artifact; a separately installed HDF5,
`h5py`, `hdf5plugin`, Python interpreter or plugin directory is not needed by
the Rust reader. Building needs Rust >=1.85.1, a C compiler, CMake >=3.26 and Git
(the upstream Bitshuffle package applies its published patch during the build).
Initial Cargo dependency downloads need network/cache as usual. Licenses and
copyrights are included in `licenses/hdf5/` and wheel license metadata.

```sh
# Standalone executable, independent of Python:
cargo run --release -p phasesmith-hdf5 --example inspect_frames -- \
  /external/data/eiger_0000.h5 /entry_0000/ESRF-ID11/eiger/data 2 72 143
# Optional Python frontend:
maturin develop --release --features experimental-spots,experimental-solution,hdf5
```

Bitshuffle is registered through its safe upstream Rust API under the upstream
HDF5 mutex. No PhaseSmith unsafe code or process-global environment changes are
needed. The built-in pipeline covers Bitshuffle+LZ4 (ID 32008), gzip/deflate,
LZF and HDF5's standard filters. This does not promise every third-party filter:
additional decoders must be compatible and discoverable by HDF5. An unavailable
required decoder yields an error naming its filter ID before frame reads.

## Count and resource contracts

The caller supplies exactly one dataset path. It must be a nonempty 3D unsigned
uint16/uint32 array in `(frame, y, x)` order. Counts are promoted exactly to u32;
zeros, high counts and reserved uint32 sentinels remain unchanged. Frames are
owned contiguous arrays, without detector flipping, masking, normalization or
interpolation. Acquisition orientation and sentinel masks belong to callers.
Finite numeric scalar metadata are read only at explicitly supplied paths.

`Limits` admits frame count, pixels per frame and decoded chunk bytes before
pixels are read (defaults: 1,000,000 frames, 100,000,000 pixels, 256 MiB per
chunk). Products use checked arithmetic. Each read materializes one frame,
with temporary native array/copy storage; these limits are not a total-RSS cap.
Virtual datasets are rejected because their mapped source/chunk limits and
missing-source fill semantics are not yet validated. Explicit linked source
datasets are admitted using the actual opened dataset metadata. No directory
scan, angle inference or silent decompression fallback occurs. Handles are
read-only and close on drop; the Python adapter also has a context manager.

```python
from phasesmith.io.hdf5 import Hdf5Frames

with Hdf5Frames("/external/data/eiger_0000.h5", "/entry_0000/ESRF-ID11/eiger/data") as reader:
    print(reader.shape, reader.dtype, reader.filters, reader.library_version)
    original_counts = reader.read_frame(2)
```

## Validation

Native fixtures cover plain, gzip, LZF and Bitshuffle/LZ4 count round trips,
uint16 promotion, reserved values, nonfinite/non-scalar metadata and admission
limits. The independent 10-KiB fixture was written with h5py/hdf5plugin; its
producer versions, file/frame hashes and explicit regeneration command are in
`crates/phasesmith-hdf5/tests/data/counts-bitshuffle-lz4.json`. Tests read it in Rust and through
the NumPy boundary without importing Python HDF5 packages. Threaded reads,
handle closure and unchanged source bytes are checked.

The frozen MgSiO3 v8 protocol uses this reader and native illumination/geometric
masks in every raw-data pass, with v7's scientific settings unchanged. Its full
run is made in an environment without h5py/hdf5plugin. The raw example still
uses Python orchestration and existing Python geometry/profile/scaling
optimizers: success establishes native HDF5 integration, not an entirely
Rust-only raw-to-structure executable. Standalone structure refinement remains
outside the implemented search workflow. Raw inputs and large products stay
outside Git.

The v8 complete run passes: 142 original frames, 246 profile teachers, 4,125
signed search observations (220 negative), inferred Mg16 Si16 O48, amplitude
R1 0.04111588883. Maximum/RMS labelled-coordinate errors are
0.0065804/0.0045658 A. Full worker time is 95.85 s on this machine. Original
frame hashes, linked/indexed positions, edge exclusions and every reduced array
are bitwise identical to the v7 run. Input source files remain unchanged.
Reference coordinates are assessed only after the worker saves its candidate
and exits; source HKLs and archived arrays are not read by the worker.

The installed native wheel and a standalone Rust executable both decode the
original data with dynamic plugin loading disabled and a nonexistent plugin
path. The wheel environment has no h5py/hdf5plugin. The macOS binary inspection
shows only system-library dynamic dependencies. Three standalone Rust frame
hashes match the independent h5py control exactly. The small checked-in report
is `benchmarks/enstatite-bundled-hdf5-macos.json`; all raw data and complete
products remain in the external output directory recorded there. Workspace
verification passes 501 Rust tests (34 ignored), 3,217 Python tests (11 skipped,
34 deselected), formatting and strict all-feature/all-target Clippy. Linux and
Windows builds are configured in CI but have not been executed locally.
