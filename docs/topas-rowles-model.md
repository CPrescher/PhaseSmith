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

## GSAS-II fundamental-parameters diagnostic

The deposited radii, axial lengths, 2.5 degree incident and diffracted Soller
angles, LPSD angular range, equatorial divergence, tube tails, five transmitted
Cu K-alpha lines, and edge transmission at each line center were also passed to
the exact pinned GSAS-II fundamental-parameters calibration. TOPAS documents
its `lh` emission widths as half widths; the adapter multiplies them by two for
the NIST FPA full-width convention used by GSAS-II. K-beta, the angle-dependent
continuum, and specimen absorption are outside this isolated K-alpha
calibration.

GSAS-II compressed 13 physical peaks from 21 to 147 degrees into
`U=1.628695`, `V=-2.814203`, `W=1.802305`, `X=0.349515`, `Y=6.109196`, and
`SH/L=0.027260`. The compressed profile has 6.952% Rwp and 0.99748 correlation
against the synthetic FPA target. That compression does not transfer as well to
the measured patterns:

| Sample | GSAS-II profile source | Poisson Rwp | Unit-weight Rwp | Correlation |
| --- | --- | ---: | ---: | ---: |
| 1a | empirical refinement | 9.085% | 7.686% | 0.99689 |
| 1a | fixed FPA compression | 13.928% | 15.084% | 0.98819 |
| 1e | empirical refinement | 8.195% | 9.111% | 0.99504 |
| 1e | fixed FPA compression | 11.710% | 13.941% | 0.98852 |

Thus GSAS-II itself does not recover the Rowles result by replacing its
empirical production profile with this compressed geometry. This is a negative
transferability result, not a claim that GSAS-II's FPA target is equivalent to
TOPAS. It argues for retaining PhaseSmith's compact production profile and
testing additional datasets before adding LPSD, tube-tail, or continuum terms.
The machine-readable result is
`validation/results/2026-08-11-rowles-gsasii-fpa.json`.

## Explicitly excluded TOPAS terms

The common workflow does not use the deposited seven-line Voigt emission
spectrum, wavelength-edge filter, angle-dependent continuum, Soller-limited
axial geometry, LPSD equatorial divergence, tube tails, flat-plate absorption,
or TOPAS-specific Gaussian/Lorentzian size/strain macros. A separate offline
PhaseSmith calibration target now supports independently derived finite axial
source/sample/receiver geometry and triangular incident/diffracted Soller
transmissions, but it is not silently substituted into this matched parity
workflow and does not claim TOPAS equivalence. That offline target also accepts
an explicit Gaussian wavelength passband and propagates transmitted line areas
into its compressed fixed spectrum. It does not reproduce the deposited TOPAS
wavelength-edge filter or angle-dependent continuum. The deposited fields
remain in the neutral manifest so the conversion is auditable; they are not
PhaseSmith runtime refinement parameters.

The archive is CC BY 4.0 and is described by Curtin University at
[DOI 10.25917/5f44ad65411cc](https://doi.org/10.25917/5f44ad65411cc). GSAS-II
is separately licensed and is used only through the external oracle workflow.
