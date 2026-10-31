# Native single-crystal profile and empirical-scale fitting

`phasesmith_spots::profile_refinement` and `empirical_fit` own these production
numerical fits. Python adapters validate NumPy arrays, select declared models and
return the existing result records. Neither native entry point requires Python,
SciPy, atomic coordinates, source intensities or an external crystallographic
program. Python still orchestrates the example and constructs its empirical
design; this is not yet a standalone Rust raw-to-structure application.

## Profile equation and parameter conventions

For each source reflection, integrate the existing finite, piecewise-constant
profile voxels over the actual pixel/exposure footprints. Half-open footprints
have zero contribution at contact without overlap. Masked samples have zero
weight; no missing mass is synthesized or renormalized. Existing affine detector
maps and signed rocking scales are unchanged. Parameters are detector shifts
`(dx,dy,domega)` in pixels/pixels/degrees and log spatial/rocking multipliers
`(s,t)`; grid steps multiply by `(exp(s),exp(s),exp(t))`.

Freeze counting weights within each inner solve. With background plane basis
`B`, plane covariance `C`, independent variance `V` and signed signal `y`, use
`W = (V + B C B^T)^-1`, applied groupwise by the existing native Woodbury GLS
kernel. Excluded pixels have zero inverse independent variance. Eliminate the
amplitude analytically: `I = (p^T W y)/(p^T W p)`. The scalar objective is
`Q = y^T W y - (p^T W y)^2/(p^T W p) + sum(r_j theta_j^2)`.
Differentiation gives
`dQ_j = -2 I (p_j^T W y - I p_j^T W p) + 2 r_j theta_j`.
Mass and all five analytical partials share the existing voxel/sample pass.
The native implementation also accumulates the profiled residual Gauss–Newton
normal from these same Gram products; the production shape optimizer uses its
analytical gradient with ten limited-memory positive-curvature pairs and a
feasible Armijo backtracking search, at most 30 trials per accepted iteration.
Only strictly improving feasible points are retained. Closed bounds, regularizer,
fixed noise iterations, relative-cost tolerance `1e-9`, projected-gradient
tolerance `1e-4`, and iteration cap remain explicit controls. Distinct optimizer
paths are evaluated with tolerances, not claimed bitwise equivalent to SciPy.

For expected counting noise, recompute `max(1,max(0,bg+max(I,0)*p)/gain+read^2)`
between the declared outer solves; supplied independent variances stay fixed.
The final fixed-shape intensity uses the existing native integrator. Conditional
joint uncertainty uses columns `(p,I*p_dx,I*p_dy,...)`, scales the GLS normal by
its diagonal, and pseudoinverts its native symmetric eigensystem with the existing
`1e-9` eigenvalue cutoff. Sigma is at least the fixed-shape counting sigma.
A three-column joint fit is used for x/y-only shape choices.

Weak and null reflections use caller-frozen centroid offsets, width and design
counting noise. They never fit their own shape or counting weights. Their actual
supplied variance and background covariance propagate through the fixed linear
filter; negative amplitudes remain signed. The template's finite first moment
and bank shift are now calculated natively. Every row remains present. Failed
conditional fits retain the baseline row with FAILED and false convergence.
Private native worker pools (1–4 threads) bound concurrent working buffers;
source-order consumption makes worker counts deterministic. Progress callbacks
run outside numerical kernels and support cancellation.

## Empirical equation and optimization

Preserve the existing explicit design and gauges:
`log scale = X_f b_f + log(max(1+X_d b_d,1e-8)) + log(max(1+X_a b_a,1e-8))`.
Derivatives below the component floor are zero. Components themselves are retained
unclipped for physical-domain checking. Corrections are `exp(-log scale)` for
all rows, including excluded teachers and signed weak observations.

Only caller-selected positive references in families with at least two selected
members teach the fit. At each Huber IRLS iteration, eliminate the weighted mean
of each family from both log intensities and every design-Jacobian column. Stack
`sqrt(w)*(center(model)-center(log(I)))`, `sqrt(strength)*P*b`, and the existing
`1000*min(min(D,A)-minimum_component,0)` barrier into one residual. Its analytical
Jacobian includes the raw design row attaining the lowest component. Accumulate
residual, gradient and normal deterministically in row order. Bound-constrained
damped normal steps use monotone column scales, a projected-gradient test and
at most 80 fused evaluations per inner solve. Strictly improving feasible
states are retained. Inner tolerances are `ftol=1e-10`, `gtol=1e-7`; a vanishing
damped step alone is not reported as convergence.

Update `w = (1/variance)*min(1,huber/abs(center(log(I)-model)/sqrt(variance)))`.
The original outer stopping tests remain coefficient-change norm `<1e-6` and
maximum relative weight change `<1e-5`. Final D/A components must exceed the
configured minimum within `1e-6`. No diagonal scale-parameter uncertainty is
invented. Designs are limited to 256 parameters and 20 million entries.

## Validation and scope

Independent `profile_fit_reference.py` directly integrates finite voxels and
uses dense NumPy covariance plus a SciPy control fit for small axis-aligned
fixtures. `intensity_reference.py` independently evaluates the model, centering
and robust empirical fit. Neither calls Rust. Existing independent affine
intersection controls and centered finite differences still cover the native
mass/translation/log-width partials. New tests compare noisy, masked, supplied-
and expected-variance profile fits/joint sigmas, randomized robust scaling with
outliers and family holdouts, signed/null rows, bounds, budgets, frozen fits,
worker determinism and cancellation. Standalone Rust tests exercise both fits
without an interpreter. GSAS-II has no oracle coverage for these empirical
single-crystal models; synthetic equations and the frozen external MgSiO3
workflow provide the controls instead.

`benchmarks/profile_scaling_fit.py` measures 64 real multi-exposure profiles on
the 31x31x23 bank and a 4,096-row, 173-parameter empirical design. Full raw-run
v10 is frozen before execution and compared with retained v9 through the same
strict gates. All large inputs and measured products remain outside Git.
