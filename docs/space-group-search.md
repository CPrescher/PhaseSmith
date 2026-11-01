# Native space-group assessment and search

The unknown-symmetry path retains all predicted reflections, including signed
weak measurements. It accepts cell and element priors without an exact space
group. Conventional Hall settings are enumerated from the pinned native database
and filtered by metric compatibility, which only permits a hypothesis; the metric
never establishes intensity symmetry. Alternative settings remain explicit.

Intensity equivalences use non-resonant Friedel symmetry. Inverse-variance means
are constructed only from supplied measured rows. For an intensity-symmetry
family with at least two Friedel-independent measurements, form weighted chi
square about its mean with variance `sigma² + (0.25 |I|)²`. Its degrees of freedom
are `members - 1`. Report the summed chi square/degrees of freedom and Rint.
At least 20 constraints are required to support a nontrivial Laue hypothesis;
reduced chi square above 4 rejects it. The largest supported rotational set is
a provisional processing hypothesis, not an assertion about the true symmetry.
Lower-symmetry alternatives and insufficient evidence remain visible. Assess
physical aperture intensities before any symmetry-dependent empirical scaling.

Absence evidence compares zero intensity `N(0,sigma²)` against a deliberately
approximate allowed-intensity model `N(mu,sigma²+mu²)` in eight resolution shells.
Estimate `mu` from all signed Friedel means in the shell, with a counting-noise
floor, without a space-group-specific exclusion. For a measured family predicted
absent, add

```text
log BF = 0.5 [log((sigma²+mu²)/sigma²)
              + (I-mu)²/(sigma²+mu²) - I²/sigma²]
```

Clip each family contribution to [-25,5] to limit the effect of isolated
contaminating crossings and of extreme Gaussian tails. Report raw counts of
measured forbidden families and positive >5-sigma violations alongside the score.
No synthetic zero is added for a missing, masked or excluded reflection.
This approximate evidence is a prescreen, not a calibrated posterior.

Search at most 12 ranked candidates, with independent 15-second per-candidate
and 180-second total budgets, using the existing native element-only search and
isotropic site polishing. Assess candidate models on the same measured
Friedel-expanded amplitude set, including reflections their hypothesis predicts
absent. Report R1 and `n log(SSE/sum(Fo²)) + p log(n)`, with n independent Friedel
families and p fitted site parameters plus scale. This BIC-like ranking assumes
homogeneous amplitude residuals and is not a calibrated probability. Retain
alternatives within delta 10 and explicitly report failed/unattempted settings.
A candidate's fit on its own absence-filtered data cannot establish its rank.

These are discrete hypothesis tests and orchestration changes, not new profile
terms. Existing fused coordinate/ADP value/derivative kernels remain their owner.
Independent Python equations, analytical tests and conventional-group synthetic
cases validate assessment. GSAS-II's powder oracle does not cover this search.
The conceptual comparison is the published P1-phasing/symmetry-assessment workflow
in [Sheldrick (2015)](https://doi.org/10.1107/S2053273314026370); no SHELX code is
used and full intrinsic phasing/space-group completeness is not claimed.

MgSiO3 is exposed development data. Protocol v30 removes Pbca and supplies no
Laue group to the numerical worker. It freezes these controls before the replay;
source files and large products remain external. Reference grading follows saved
models and process exit. Earlier known-group results remain unchanged evidence.
