# Rowles common-subset model and TOPAS exclusions

## Scope

The Rowles archive supplies useful in-house Cu K-alpha patterns, structures,
and weighed phase fractions. PhaseSmith converts the deposited inputs to a
neutral XY/CIF/instrument/JSON bundle, but deliberately does not implement or
approximate the deposited TOPAS fundamental-parameters model. The original
source records are retained only as provenance metadata.

The executable comparison is instead a matched common subset shared with the
exact pinned GSAS-II revision `c0bc79b259cdf0065480b5fbd57674ddf12c4a23`:

- three fixed-cell structures with refinable isotropic displacement values;
- Cu K-alpha1/K-alpha2 wavelengths and intensity ratio;
- U/V/W Gaussian instrument broadening, zero shift, and SH/L axial asymmetry;
- Lorentzian coherent-domain size with shape factor one;
- Lorentzian isotropic microstrain;
- seven-term Chebyshev background and three non-negative phase scales.

PhaseSmith solves the linear phase-scale/background block exactly at each
alternation and refines smaller instrument and specimen blocks with analytical
derivatives. The Lorentzian microstrain convention is

```text
H_L(deg 2theta) = (180/pi) epsilon tan(theta),
epsilon = GSAS-II Mustrain * 1e-6.
```

This mapping was derived from the angular breadth convention and checked with
plain reflection arrays from the pinned oracle; no GSAS-II implementation code
was copied.

## Reviewed parity result

| Sample | Program | Al2O3 | ZnO | CaF2 | Poisson Rwp | Profile correlation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1a | PhaseSmith | 1.533% | 3.303% | 95.164% | 8.782% | 0.99694 |
| 1a | GSAS-II | 1.399% | 3.304% | 95.297% | 9.085% | 0.99689 |
| 1e | PhaseSmith | 57.278% | 14.013% | 28.709% | 8.264% | 0.99522 |
| 1e | GSAS-II | 57.038% | 14.078% | 28.884% | 8.195% | 0.99504 |

The automated gate requires absolute Rwp, unit-weight Rwp, and maximum phase
fraction differences no larger than 0.005, plus profile-correlation difference
no larger than 0.002. Both patterns pass.

## Explicitly excluded TOPAS terms

The common workflow does not use the deposited seven-line Voigt emission
spectrum, wavelength-edge filter, angle-dependent continuum, Soller-limited
axial geometry, LPSD equatorial divergence, tube tails, flat-plate absorption,
or TOPAS-specific Gaussian/Lorentzian size/strain macros. A separate offline
PhaseSmith calibration target now supports independently derived finite axial
source/sample/receiver geometry and triangular incident/diffracted Soller
transmissions, but it is not silently substituted into this matched parity
workflow and does not claim TOPAS equivalence. The deposited fields remain in
the neutral manifest so the conversion is auditable; they are not PhaseSmith
runtime refinement parameters.

The archive is CC BY 4.0 and is described by Curtin University at
[DOI 10.25917/5f44ad65411cc](https://doi.org/10.25917/5f44ad65411cc). GSAS-II
is separately licensed and is used only through the external oracle workflow.
