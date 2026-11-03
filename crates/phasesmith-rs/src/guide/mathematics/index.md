# Mathematical reference

These pages define the mathematical models currently implemented by the
published Rust crates. Equations are rendered as plain, copyable text so the
reference works on docs.rs and in offline `cargo doc` output without external
JavaScript or a math-rendering service.

## Coverage

| Area | Implemented mathematics |
| --- | --- |
| [`peak_profiles`] | normalized pseudo-Voigt, TCH mixing, CW broadening, wavelength components, FCJ asymmetry, neutron TOF |
| [`crystallography`] | cell/reciprocal metrics, exact symmetry, reflection conditions, scattering factors, displacement factors, structure factors, LP corrections, Bragg positions |
| [`pattern_composition`] | multi-phase and multi-wavelength sums, size, microstrain, preferred orientation, derivative products |
| [`backgrounds`] | Smooth Bruckner preprocessing, polynomial, Chebyshev, point, amorphous, and composite backgrounds |
| [`refinement`] | residuals, R factors, constraints, Le Bail redistribution, damped Gauss–Newton, covariance, and quantitative phase analysis |
| [`single_crystal_geometry`] | detector supports/noise, rotation observations, reciprocal mapping, Ewald prediction and positional derivatives |
| [`single_crystal_intensities`] | finite voxel/affine profile masses, signed GLS amplitudes, conditional covariance, corrections and relative scaling |
| [`structure_search`] | signed HKLs, exact symmetry/absence evidence, phasing, atomic/ADP models, common-data scores and bounded acceptance |

Every section names the Rust module or type that owns the equation. Models not
implemented by the current API, such as magnetic scattering or extinction, are
not described as available mathematics. Aperture masks do not supply a physical
absorption correction. Single-crystal candidate ADP polishing is documented
separately from the supported powder structural models and does not constitute
a standalone structure-refinement workflow.

## Shared notation

- `x` is a sampled pattern coordinate.
- `φ = 2θ` is a diffraction angle; angular Rust APIs use degrees unless a
  formula explicitly says radians.
- `λ` and direct-cell lengths are in ångströms.
- `h = (h,k,l)^T` is a Miller-index column vector.
- `G` and `G* = G^-1` are direct and reciprocal metric tensors.
- `q² = h^T G* h = 1/d²`; `s = sqrt(q²)/2 = sin(θ)/λ`.
- `J v` and `J^T u` denote Jacobian-vector and transpose-Jacobian-vector
  products. They are analytical products, not finite differences.

[`peak_profiles`]: crate::guide::mathematics::peak_profiles
[`crystallography`]: crate::guide::mathematics::crystallography
[`pattern_composition`]: crate::guide::mathematics::pattern_composition
[`backgrounds`]: crate::guide::mathematics::backgrounds
[`refinement`]: crate::guide::mathematics::refinement
[`single_crystal_geometry`]: crate::guide::mathematics::single_crystal_geometry
[`single_crystal_intensities`]: crate::guide::mathematics::single_crystal_intensities
[`structure_search`]: crate::guide::mathematics::structure_search
