# PhaseSmith

PhaseSmith is an early-stage powder-diffraction computation library with a
Rust numerical core and a typed Python/NumPy API. The current implementation
includes symmetric TCH, CW U/V/W/X/Y broadening, FCJ asymmetry, wavelength
components, extensible sample physics, multi-phase CW X-ray/neutron and neutron
TOF calculation, plus scripted Le Bail, [CW Pawley](docs/pawley.md) and [TOF Pawley](docs/tof-pawley.md) workflows. Analytical
derivatives are computed during fused peak accumulation. The crystallography
foundation includes native general-cell mathematics, P1 complex structure
factors, exact symmetry, bounded reflection generation, and prepared X-ray and
neutron scattering factors with analytical derivatives. General-symmetry
structural intensities and monochromatic structural patterns now have a fused
native values/JVP/VJP path, direct fixed CIF anisotropic displacement, and a
scriptable `RietveldPhase` API.
CIF-backed Le Bail can refine setting-aware lattice parameters with analytical
derivatives and guarded, stable-ID reflection-domain regeneration.
The first full CIF-backed Rietveld workflow now refines CW profile/background,
phase scale, lattice, symmetry-allowed coordinates, occupancy, and isotropic
displacement through matrix-free Rust JVP/VJP products with safe checkpoints
and structured logs.
Model-independent preprocessing now includes a native Smooth Bruckner
background implementation compatible with pinned xypattern/Dioptas behavior,
plus optional Chebyshev compression and plain NumPy subtraction results.

Install the Python interface from PyPI or the native Rust facade from
crates.io:

```shell
python -m pip install phasesmith
cargo add phasesmith
```

Use the [versioned documentation](https://phasesmith.readthedocs.io/) to follow
the shortest path from powder data and a CIF to background subtraction, Le
Bail extraction, Rietveld refinement, reports, and persistence. Documentation
sources live in [docs](docs/index.md), and the architecture and non-negotiable
numerical rules are in [PROJECT_BRIEF.md](PROJECT_BRIEF.md). The tag-driven
GitHub/PyPI/crates.io process is documented in
[docs/releasing.md](docs/releasing.md).

## Development

Requires Rust 1.85 or newer, Python 3.11 or newer, `uv`, and `maturin`.

```shell
uv venv
uv pip install -e '.[dev]'
maturin develop --uv
cargo test --workspace --all-features
uv run pytest
```

Editable environments retain the absolute checkout location. After moving or
renaming the repository, recreate `.venv`, repeat the three setup commands
above, and confirm `uv run python -c "import phasesmith; print(phasesmith.__file__)"`
points into the current checkout before running the gate.

### Native GUI applications

GUI applications are separate consumers of the published Rust library. A
future Tauri or other native application can depend on `phasesmith`, own its
presentation state and background jobs, and call the native workflows directly
without bundling Python.

Run the Rust benchmarks with `cargo bench -p phasesmith-core`. For a comparable
optimized Python-to-Rust measurement, build the release extension and require
release mode explicitly:

```shell
maturin develop --release --uv
uv run python benchmarks/profile.py --require-release
uv run python benchmarks/lebail.py --require-release
uv run python benchmarks/scattering.py --require-release
uv run python benchmarks/structural_pattern.py --require-release
uv run python benchmarks/lattice_refinement.py --require-release
uv run python benchmarks/background.py --require-release
uv run python benchmarks/real_data.py --require-release
```

The optional XRD-Rust comparison exercises both libraries through their public
stick-pattern APIs and validates positions and normalized intensities before
reporting speed. Install its benchmark-only dependencies, then retain the JSON
result:

```shell
uv pip install 'xrd-rust==0.3.5' 'pymatgen==2026.5.4'
uv run python benchmarks/compare_xrd_rust.py --require-release \
  --json-output validation/results/local-xrd-rust-performance.json
```

The reviewed environment, scope, results, and interpretation are in
[docs/xrd-rust-performance.md](docs/xrd-rust-performance.md). XRD-Rust and
pymatgen are not PhaseSmith runtime dependencies.

The joint multi-histogram PbSO4 workload is also executable entirely in Rust,
without building or launching Python:

```shell
cargo run --release -p phasesmith-workflows --example joint_pbso4 -- \
  validation/data/gsasii-pbso4-cw
```

Checksum-pinned validation also has a Python-free CLI. Successful runs write
the same stable JSON report used by the scripting adapter:

```shell
cargo run -p phasesmith-validation --bin phasesmith-validation -- datasets
cargo run --release -p phasesmith-validation --bin phasesmith-validation -- \
  run iucr-qarr-1g validation/data/iucr-qarr-1g
```

Ordinary `phasesmith.validation` calls delegate to these Rust workflows.
Python callbacks, custom execution policies, and explicit reference tests keep
the independent scripting path; the native CLI does not launch Python.

The real-data benchmark verifies pinned QARR 1g, APS sucrose, and official
PbSO4 X-ray/neutron tutorial inputs, records cold and warmed complete-workflow
timings, and hashes the timing-free scientific report. By default it compares
one- and two-thread QARR execution and runs both PbSO4 probes. Use `--dataset`,
`--threads`, and `--json-output` to select cases and retain a machine-readable
result. This is the PhaseSmith-only performance harness; the paired QARR and
PbSO4 GSAS-II commands below are cross-implementation scientific and
performance gates. The corresponding slow regression tests are opt-in:

```shell
uv run pytest -m real_data
```

The opt-in [opXRD robustness campaign](docs/opxrd-robustness.md) verifies a
checksum-pinned 14-pattern stratified selection, compares five residual and
background diagnostics, preserves invalid-grid and negative-count boundaries,
and can run three assumption-explicit common models against pinned GSAS-II:

```shell
uv run python benchmarks/opxrd_robustness.py --structural \
  --json-output validation/results/local-opxrd.json
```

The pinned external-oracle environment can compare the same support-limited CW
profile-and-derivative workload against GSAS-II. A second benchmark starts from
the crystal structure and compares structure factors, integrated intensities,
and the composed structural CW pattern. Numerical agreement is checked before
timings are reported:

```shell
uv run python benchmarks/compare_gsasii.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

uv run python benchmarks/compare_gsasii_structural.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

uv run python benchmarks/compare_gsasii_qarr.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/iucr-qarr-1g \
  --phasesmith-threads 2

uv run python benchmarks/compare_gsasii_pbso4.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/gsasii-pbso4-cw
```

The exact scope and interpretation are documented in
[docs/gsasii-performance.md](docs/gsasii-performance.md). This is a kernel-level
comparison. The QARR and PbSO4 drivers separately compare complete native
workflows and gate their reported scientific results before reporting speed.
The established Python PbSO4 report includes the supplied reference cell
alongside the per-probe staged fits and GSAS-II's joint refinement. The Rust
example separately exercises PhaseSmith's first-class summed joint objective.

Background estimation is an explicit preprocessing step:

```python
import phasesmith

subtracted = phasesmith.SmoothBrucknerBackground(
    smooth_width=0.1,
    iterations=50,
    chebyshev_order=50,
).subtract(two_theta, measured)

pattern = phasesmith.PowderPattern(
    two_theta,
    observed_y=measured,
    background=subtracted.background,
)
```

Set `chebyshev_order=None` to use the raw Bruckner envelope. Neither xypattern
nor Dioptas is required at runtime.

Refinement treats that array as a fixed broad baseline. Attach a low-order
analytical correction rather than replacing it:

```python
from phasesmith.refinement import ChebyshevBackground, lebail

residual = ChebyshevBackground("residual", (0.0, 0.0, 0.0), (two_theta[0], two_theta[-1]))
request = lebail.LeBailInput(pattern, instrument, phases).with_refinable_background(residual)
result = lebail.refine(request)
```

Structural Rietveld uses the same additive convention through
`RietveldInput.background`; linear CW Le Bail backgrounds are eliminated by a
weighted least-squares update each cycle, while nonlinear background terms
remain in the joint analytical solve.

```python
import numpy as np
from phasesmith import accumulate

x = np.linspace(20.0, 30.0, 10_001)
result = accumulate(
    x,
    positions=[24.0, 26.0],
    intensities=[100.0, 80.0],
    fwhms=[0.08, 0.1],
    etas=[0.3, 0.5],
)
print(result.y.shape, result.derivatives.local.values.shape)

# Dense materialization is explicit and intended for small compatibility uses.
dense = result.derivatives.local.to_dense(result.y.size)
print(dense.shape)  # (peak, parameter, sample)
```

Use `profile_tch` or `accumulate_tch` when Gaussian and Lorentzian component
FWHMs are the direct inputs. Explicit `tch_shape_from_gaussian_sigma` and
`profile_tch_from_gaussian_sigma` helpers are provided when Gaussian width is a
standard deviation; width conventions are never inferred.

An entire constant-wavelength reflection list, including all local and shared
instrument derivatives, is also one array-oriented call:

```python
from phasesmith import ConstantWavelengthInstrument, accumulate_cw

instrument = ConstantWavelengthInstrument(
    wavelength_angstrom=1.54056,
    u_deg2=2e-4,
    v_deg2=-1e-4,
    w_deg2=1.2e-4,
    x_deg=1.5e-3,
    y_deg=3e-3,
)
cw = accumulate_cw(x, [24.0, 26.0], [100.0, 80.0], instrument)
print(cw.derivatives.local_parameter_names)  # intensity, position
print(cw.derivatives.global_parameter_names)  # U, V, W, X, Y
```

Atomic scattering models are also script-first and can be prepared once for a
structure. X-ray species select exact neutral/ionic table states, while neutron
species retain natural/isotope identity:

```python
from phasesmith import ScatteringSpecies, XrayNonResonant

species = (
    ScatteringSpecies("Si"),
    ScatteringSpecies("O"),
    ScatteringSpecies("Fe", charge=3),
)
scattering = XrayNonResonant().prepare(species).evaluate([0.0, 0.5, 1.0])
print(scattering.amplitudes.shape)  # (reflection, site)
print(scattering.d_amplitudes_d_s.shape)  # analytical df/ds, same shape
```

A typed structure can be evaluated without assembling scattering arrays or
looping over atoms/reflections in Python. Correction geometry is explicit; the
default neutral model returns raw multiplicity-weighted structural intensity:

```python
from phasesmith import ExecutionPolicy, XrayNonResonant, calculate_structure_factor_values
from phasesmith.io.cif import read_cif

structure = read_cif("phase.cif").structure
structural = calculate_structure_factor_values(
    structure,
    hkl=[[1, 0, 0], [1, 1, 0]],
    multiplicity=[6, 12],
    scattering=XrayNonResonant(),
    scale=1.0,
    execution=ExecutionPolicy(threads=2),
)
print(structural.f, structural.integrated_intensity)
```

Reuse the immutable execution policy across repeated calls so its bounded
native worker pool is retained. Select `threads=1` when an embedding
application already owns outer parallelism. `execution=None` uses PhaseSmith's
bounded two-thread default.

Use `calculate_structure_factors` when the bounded dense analytical structural
Jacobian is also required.

Axial divergence is a separate typed model and composes with CW broadening
without expanding reflections in Python:

```python
from phasesmith import FcjGeometry, accumulate_cw_fcj

geometry = FcjGeometry(sample_over_radius=0.012, detector_over_radius=0.012)
asymmetric = accumulate_cw_fcj(x, [24.0, 26.0], [100.0, 80.0], instrument, geometry)
print(asymmetric.derivatives.global_parameter_names)
# U, V, W, X, Y, sample_over_radius, detector_over_radius
```

Discrete radiation components are optional. The ordinary CW calls above are
monochromatic; a K-alpha doublet is an explicit model:

```python
from phasesmith import WavelengthComponents, accumulate_cw_fcj_components

radiation = WavelengthComponents.doublet(
    reference_wavelength_angstrom=1.54056,
    secondary_wavelength_angstrom=1.54439,
    secondary_to_reference_intensity=0.5,
)
doublet = accumulate_cw_fcj_components(
    x, [24.0, 26.0], [100.0, 80.0], instrument, radiation, geometry
)
```

Sample physics is explicit and provider-based. Built-in and third-party models
return the same vectorized contribution schema, and Python is never called from
the native peak/sample loop:

```python
from phasesmith import (
    CompositePhysicsProvider,
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    ReflectionGeometryBatch,
    calculate_cw_pattern,
)

reflections = ReflectionGeometryBatch(
    hkl=[[1, 0, 0], [1, 1, 0]],
    d_spacing_angstrom=[3.72, 2.64],
    two_theta_deg=[24.0, 34.0],
    base_integrated_intensity=[100.0, 80.0],
)
sample = CompositePhysicsProvider(
    (
        IsotropicSizeBroadening(crystallite_size_nm=50.0),
        IsotropicMicrostrainBroadening(rms_microstrain=5e-4),
    )
)
calculated = calculate_cw_pattern(x, reflections, instrument, physics=sample)
print(calculated.derivatives.global_parameter_names)
```

The high-level script interface assigns durable IDs and calculates all phases
through one flattened native call:

```python
from phasesmith import Phase, PowderPattern, ReflectionBatch, calculate_pattern

alpha_reflections = ReflectionBatch(
    reflection_ids=["alpha-100", "alpha-110"],
    hkl=[[1, 0, 0], [1, 1, 0]],
    d_spacing_angstrom=[3.72, 2.64],
    two_theta_deg=[24.0, 34.0],
    integrated_intensity=[100.0, 80.0],
)
alpha = Phase(
    phase_id="alpha",
    name="Alpha phase",
    reflections=alpha_reflections,
    scale=1.0,
    physics=sample,
)
pattern_result = calculate_pattern(PowderPattern(x), instrument, [alpha])
print(pattern_result.reflection_keys)
print(pattern_result.derivatives.global_parameter_names)
```

Neutron CW is an explicit monochromatic probe configuration and cannot receive
an X-ray K-alpha doublet:

```python
from phasesmith import ConstantWavelengthExperiment, calculate_neutron_pattern

neutron = ConstantWavelengthExperiment.neutron(instrument)
neutron_result = calculate_neutron_pattern(PowderPattern(x), neutron, [alpha])
```

TOF reflections use d-spacing as their durable local coordinate. Values and
all local/shared derivatives are accumulated in one native call. Rust and
Python consumers can additionally use `TofLeBailInput` and
`refine_tof_lebail` for a typed microsecond-domain, fixed-instrument
nonnegative extraction workflow.
Rust callers may attach `TofChebyshevBackground` to refine an explicit-domain
Chebyshev residual on top of the fixed pattern background; omitting it preserves
the fixed background alone. Python can build a complete fixed-instrument request
from one bank, calibration, and CIF:

```python
from phasesmith.refinement.tof_lebail import (
    TofLeBailCancellation,
    TofLeBailInput,
    TofLeBailOptions,
    refine_tof_lebail,
)

request = TofLeBailInput.from_files(
    "PG3_17541.gsa",
    "PGHR_60-2015A.prm",
    "LaB6.cif",
    bank=2,
)
cancellation = TofLeBailCancellation()
events = []
result = refine_tof_lebail(
    request,
    TofLeBailOptions(cycles=20),
    cancellation=cancellation,
    progress=events.append,
)

# Continue a last-accepted state with a larger total cycle budget.
continued = refine_tof_lebail(
    request,
    TofLeBailOptions(cycles=50),
    checkpoint=result.checkpoint,
)
```

The file path is not tied to POWGEN: reduced center/density columns, GSAS SLOG
FXYE, and packed constant-step GSAS STD are supported, and legacy GSAS profile
functions 1 and 3 translate into the same typed coefficients. Other beamlines
can pass `TofPowderPattern` and `TofInstrument` directly. The pinned LANL nickel
example is the non-POWGEN acceptance gate for this supported profile family.

The lower-level profile API remains available:

```python
from phasesmith import TofInstrument, accumulate_tof

tof_instrument = TofInstrument(
    zero_us=-0.773,
    difc_us_per_angstrom=5084.83,
    difa_us_per_angstrom2=-2.63,
    difb_us_angstrom=0.0,
    alpha_coefficient=5.0,
    beta0_per_us=0.0333,
    beta1_angstrom4_per_us=0.000964,
    betaq_angstrom2_per_us=0.0,
    sigma0_us2=0.0,
    sigma1_us2_per_angstrom2=15.14,
    sigma2_us2_per_angstrom4=0.0,
    sigmaq_us2_per_angstrom=0.0,
    x_us_per_angstrom=0.0,
    y_us_per_angstrom2=0.0,
    z_us=0.0,
)
tof_x = np.linspace(2_000.0, 20_000.0, 6_001)  # bin centers, microseconds
tof_result = accumulate_tof(tof_x, [0.8, 1.5], [100.0, 80.0], tof_instrument)
print(tof_result.derivatives.local_parameter_names)  # intensity, d_spacing
print(tof_result.derivatives.global_parameter_names)  # 15 instrument rows
```

A complete Le Bail extraction uses the same typed pattern, instrument, and
phase models and requires no project file or hand-written optimizer callback:

```python
from phasesmith.refinement import lebail

observed = PowderPattern(
    x,
    observed_y=measured_y,
    background=background_y,
    uncertainty=sigma_y,
)
result = lebail.refine(lebail.LeBailInput(observed, instrument, (alpha,)))
print(result.termination_reason, result.metrics.rwp)
print([(item.reflection_id, item.integrated_intensity) for item in result.intensities])
```

For a CIF-backed single phase, the convenience constructor creates the
generated reflection domain and the symmetry-allowed bounded lattice parameter
set directly from the observed grid:

```python
request = lebail.LeBailInput.from_cif(
    observed,
    instrument,
    "phase.cif",
    phase_id="alpha",
)
result = lebail.refine(request)
print(result.phases[0].structure.cell)
```

When detector geometry and wavelength are already known from Dioptas/pyFAI but
no separate resolution standard was measured, a predominantly single-phase
pattern can provide a conservative **effective starting profile**:

```python
start = starting_profile_from_fwhm(wavelength_angstrom, fwhm_deg=0.04)
request = lebail.LeBailInput.from_cif(
    observed,
    start,
    "dominant-phase.cif",
    phase_id="dominant",
    refine_lattice=True,
)
profile_start = estimate_effective_profile(
    request,
    ProfileEstimationOptions(align_lattice=True),
)
print(profile_start.instrument, profile_start.active_parameters)
```

The wavelength is fixed. The returned widths may include sample broadening and
are intended as refinement starting values, not as an instrument-only
resolution calibration. See
[`docs/effective-profile-estimation.md`](docs/effective-profile-estimation.md).

A monochromatic structural refinement is likewise constructed directly from a
CIF. Parameter families are explicit and no GSAS-II installation is involved:

```python
from phasesmith.refinement import rietveld

request = rietveld.RietveldInput.from_cif(
    observed,
    experiment,
    "phase.cif",
    phase_id="alpha",
    selection=rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=True,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    ),
)
result = rietveld.refine(request)
print(result.termination_reason, result.metrics.rwp)
print(result.phases[0].structure.cell)
```

Multiphase calculations and Rietveld refinement can use a bounded worker pool.
The bounded default is two threads. Embedding applications that already
schedule independent work can select one thread; choose another fixed budget
for predictable GUI/script behavior, or `None` to use the available logical
CPUs:

```python
import phasesmith
from phasesmith.refinement import rietveld

options = rietveld.RietveldOptions(
    execution=phasesmith.ExecutionPolicy(threads=2),
)
result = rietveld.refine(request, options)
```

Independent phases execute concurrently, while their values and analytical
derivatives are combined in the original phase order.

For long-running scripts and application integration, the small project facade
keeps restart state, cooperative stop control, persistence, and plain reports
together while leaving `RietveldInput` fully accessible:

```python
from dataclasses import replace

from phasesmith import BraggBrentanoGeometry, RietveldParameterSelection, RietveldProject

experiment = replace(
    experiment,
    zero_shift_deg=0.0,
    geometry=BraggBrentanoGeometry(240.0, sample_displacement_mm=0.0),
)
selection = RietveldParameterSelection(
    phase_scale=True,
    lattice=True,
    sample_physics=True,
    instrument_parameters=("zero_shift_deg", "sample_displacement_mm"),
    background=True,
)
request = replace(request, experiment=experiment, selection=selection)
project = RietveldProject(request)
proposal = project.propose_intelligent_recipe()
print(proposal.to_record())  # advice and rationale; no refinement has run yet
workflow = project.refine_intelligently(logger=my_event_logger)
result = workflow.final_result
project.write_reports(json_path="result.json", csv_path="pattern.csv")
project.save("run-state")

# Another thread or a callback can stop safely; the accepted state is resumable.
project.stop("user_requested")
```

The intelligent path is optional workflow orchestration. `project.refine()`
continues to call the general solver once with exactly the caller-selected
parameters, and `project.refine_recipe(recipe)` runs an explicit user-defined
sequence.

Persisted projects also have a provider-neutral command boundary for coding
agents and language models. `phasesmith plan workflow.json` prints a read-only,
byte-bound plan, sanitized scientific advisor context, and deterministic
recipe; `phasesmith advisor-packet plan.json` creates a path-free model handoff
and `phasesmith lint-recipe plan.json recipe.json` checks a packet-bound,
structured-provenance proposal before `phasesmith run plan.json --approve
PLAN_ID [--proposal recipe.json]` executes it. Post-run `review` and
`review-packet` commands support lineage-bound iterations with a new approval
each time. All JSON contracts are available through `phasesmith schema` and
under `schemas/automation/`. The [agent skill](docs/agent-skill/index.md) ships
with the Python package: `phasesmith skill --path` locates it and
`phasesmith skill --print all` prints the complete instructions. Its canonical
source and the offline end-to-end example live under
`skills/phasesmith-ai-workflows/` and `examples/automation/`. Raw pattern/CIF
inspection, checkpoint resume, strict schemas, structured errors, and the
scientific recipe rubric are documented in [AI-guided
automation](docs/ai-automation.md). PhaseSmith does not depend on an AI SDK or
call a model from the solver.

Refinable backgrounds share one analytical interface. Built-ins include power
and Chebyshev series, fixed-knot linear interpolation, broad normalized
Gaussian amorphous components, and ordered composites. Smooth Bruckner remains
an explicit preprocessing operation and is never inserted automatically; when
supplied, its values remain the fixed baseline beneath the refinable model.

GSAS-II is used only as the optional pinned validation oracle described in
[`oracle/README.md`](oracle/README.md).

## License

PhaseSmith is licensed under the [MIT License](LICENSE). GSAS-II is
separately licensed, is used only as an optional external validation oracle,
and is not redistributed here.
