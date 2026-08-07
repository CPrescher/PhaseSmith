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
  Kα2 instrument metadata. It then reports `blocked`, because the full
  structure-factor Rietveld request currently accepts a single monochromatic
  wavelength even though the lower-level profile kernel supports discrete
  components. No quantitative accuracy result is fabricated. The published
  weighed targets are Al2O3 31.37%, ZnO 34.21%, and CaF2 34.42%.

The QARR case comes from the [IUCr quantitative phase analysis round
robin](https://www.iucr.org/__data/iucr/powder/QARR/data-kit.htm), with files
retrieved from a commit-pinned public mirror because the legacy IUCr asset URL
does not permit automated retrieval. The sucrose files come from the official
[GSAS-II tutorial repository](https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/tree/e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/data).

The committed `validation/results/2026-08-07-baseline.json` records the first
reviewed run. Elapsed time is diagnostic host timing, not a cross-machine
performance acceptance threshold.
