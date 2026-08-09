# Refinement, residuals, and derived quantities

These equations are implemented by [`crate::workflows`]. Optimizer state never
enters [`crate::core`].

## Residuals and agreement factors

For included sample `i`, observed `y_obs,i`, calculated `y_calc,i`, and optional
one-sigma uncertainty `σ_i`, the unweighted difference and optimization
residual are

```text
e_i = y_calc,i - y_obs,i
r_i = e_i/σ_i            when uncertainties are enabled
r_i = e_i                otherwise
w_i = 1/σ_i²             or 1 for unit weighting.
```

[`crate::workflows::evaluate_residuals`] reports fractions, not percentages:

```text
Rp  = Σ_i |e_i| / Σ_i |y_obs,i|
Rwp = sqrt[Σ_i w_i e_i² / Σ_i w_i y_obs,i²]
χ²  = Σ_i w_i e_i²
χ²_reduced = χ²/(n_included - n_free).
```

A false mask value removes a sample from the objective and all derivative
products. The minimized least-squares objective is

```text
Φ = (1/2) r^T r.
```

## Parameter scaling and exact constraints

[`crate::workflows::ParameterSpec`] maps a free numerical coordinate `z_j` to
a physical parameter using a fixed scale `s_j`. Constraint targets may be

```text
fixed:   p_t = constant
affine:  p_t = a p_s + b
linear:  p_t = b + Σ_j a_j p_j.
```

[`crate::workflows::ConstraintTransform`] topologically orders this acyclic
graph and constructs the exact physical-to-scaled-free derivative matrix `D`:

```text
δp = D δz
J_free v = J_physical (Dv)
J_free^T u = D^T (J_physical^T u).
```

Bounds apply in scaled free coordinates, while reports and checkpoints retain
physical values.

## Le Bail redistribution

For reflection `k`, current integrated intensity `I_k`, sampled normalized
profile `p_ki`, observed `y_i`, background `b_i`, and integration/statistical
weight `w_i`, one [`crate::workflows::iterate_lebail_once`] redistribution is

```text
P_i = Σ_j I_j p_ji
R_i = max(y_i-b_i,0)/P_i
I'_k = I_k [Σ_i w_i p_ki R_i]/[Σ_i w_i p_ki].
```

Only included samples inside the exact support block participate. Ratios are
not formed where `P_i` is below the configured threshold. Optional damping
uses a convex interpolation between `I_k` and `I'_k`; intensities remain
non-negative. Coincident profiles receive identical multiplicative updates and
therefore preserve their starting ratio.

Profile/lattice refinement alternates redistribution with an analytical
bounded step. Dynamic lattice domains keep topology fixed during one Jacobian
and line-search evaluation, then regenerate accepted reflection families and
transfer intensities by stable Miller-family ID.

## Damped Gauss–Newton

Rietveld and optional Le Bail parameter motion linearize the residual at the
current accepted state. For residual Jacobian `J`, damping `λ >= 0`, and step
`δ`, the normal equation is

```text
(J^T J + λI) δ = -J^T r.
```

The native Rietveld solvers use deterministic conjugate gradients with
analytical JVP/VJP products; they need not materialize the sample-by-parameter
Jacobian. Bounds cap the scaled step, and deterministic backtracking accepts a
trial only when the complete objective strictly decreases. Rejected trials do
not mutate the accepted state or guarded reflection topology.

For joint multi-histogram refinement,

```text
Φ_joint = Σ_h Φ_h
∇Φ_joint = Σ_h scatter_h^T ∇Φ_h.
```

Structural parameters shared by stable phase/site identity occur once in the
joint vector; scales, backgrounds, instruments, masks, and uncertainties remain
histogram-local. See [`crate::workflows::PreparedJointRietveldObjective`].

## Covariance and correlation

For a full-rank final weighted Jacobian, the scaled-free covariance starts from

```text
C_z = (J_free^T J_free)^-1.
```

When supplied uncertainties define absolute weights, `C_z` is not multiplied
by reduced chi-square. For unit-weight fits, the residual noise scale is
estimated and

```text
C_z <- χ²_reduced C_z.
```

Physical covariance propagates constraints and scaling exactly:

```text
C_p = D C_z D^T
corr(p_i,p_j) = C_p,ij / sqrt(C_p,ii C_p,jj).
```

Singular or rank-deficient parameterizations report diagnostics instead of a
misleading inverse.

## Quantitative phase analysis

For compatible refined phase scale `S_p`, formula units per cell `Z_p`, formula
mass `M_p`, and cell volume `V_p`, the Hill–Howard crystalline weight fraction
implemented by [`crate::workflows::quantitative_phase_analysis`] is

```text
W_p = S_p Z_p M_p V_p / Σ_i (S_i Z_i M_i V_i).
```

All scales must come from the same calculation and share structure-factor,
multiplicity, correction, and scale conventions. Fractions normalize only over
the supplied crystalline phases; amorphous or unidentified material requires a
separate experimental/internal-standard model.
