# External real-data validation

PhaseSmith does not bundle tutorial patterns and does not import or execute
GSAS-II. The command below explicitly downloads files from commit-pinned HTTPS
locations, verifies their byte sizes and SHA-256 hashes, and runs only
PhaseSmith code:

```bash
python tools/validate_real_data.py --fetch \
  --data-directory validation/data \
  --output validation/results/local.json
```

The QARR stages emit structured progress to standard error. In an interactive
terminal, press `q` or Ctrl+C once to request a graceful stop at the next safe
batch boundary; a second Ctrl+C forces interruption. A cooperative stop returns
a machine-readable `blocked` report with the last accepted Rwp and a nonzero
CLI exit status.

`validation/data/` and local result files are ignored. The two current cases
have deliberately different meanings:

- `aps-sucrose-11bmb` exercises the supported monochromatic FXYE → background
  → symmetry/reflection generation → Le Bail → analytical profile-refinement
  path on the official APS 11-BM sucrose tutorial pattern. Its `Rwp <= 0.22`
  threshold is a regression smoke gate for the present model, not a claim of
  numerical equivalence with GSAS-II. The tutorial's lower final residual uses
  additional staged background-peak, crystallite-size, microstrain, lattice,
  and repeated extraction refinements.
- `iucr-qarr-1g` verifies the 7,251-point 5–150° input and its explicit Cu Kα1/
  Kα2 instrument metadata, runs a native three-phase fixed-spectrum structural
  refinement, and converts the final polished scales with the Hill--Howard
  relation. The reviewed baseline returns Al2O3 33.250%, ZnO 32.936%, and CaF2
  33.814% against weighed targets of 31.37%, 34.21%, and 34.42%. Its largest
  absolute error is 1.881 weight-percentage points. The profile gates distinguish
  Poisson-weighted Rwp (0.19679, limit 0.20) from unit-weight Rwp (0.13282,
  limit 0.15); profile correlation is 0.99069.

The QARR checkpoint explicitly approximates anisotropic displacement with
trace-mean isotropic values, uses fixed Cu Kα1 dispersion offsets for both
doublet components, and does not yet apply the supplied SH/L=0.002 FCJ
asymmetry or absorption. Those limitations are emitted in the report rather
than hidden.

The QARR case comes from the [IUCr quantitative phase analysis round
robin](https://www.iucr.org/__data/iucr/powder/QARR/data-kit.htm), with files
retrieved from a commit-pinned public mirror because the legacy IUCr asset URL
does not permit automated retrieval. The sucrose files come from the official
[GSAS-II tutorial repository](https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/tree/e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/data).

The committed `validation/results/2026-08-07-baseline.json` records the first
reviewed run. Elapsed time is diagnostic host timing, not a cross-machine
performance acceptance threshold.
