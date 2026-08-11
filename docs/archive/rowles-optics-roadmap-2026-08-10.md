# Rowles laboratory model and TOPAS-compatible optics roadmap

> **Historical working note (2026-08-10).** This predates the completed
> GSAS-II-parity campaign and records an exploratory optics plan plus older
> PhaseSmith residuals. It is preserved for provenance, not as the current
> implementation plan or validation result. See `docs/topas-rowles-model.md`
> and `docs/gsasii-benchmark-report.md` for the reviewed conclusions.

## Scope and conclusion

The deposited Rowles inputs can be used to add the relevant physical model to
PhaseSmith, but “the TOPAS model” is not one profile function. It is a sequence
of wavelength-, instrument-, specimen-, and microstructure-space convolutions.
PhaseSmith already implements a useful common subset. The remaining optics
must be implemented independently from published equations and validated
against the deposited patterns; TOPAS source code is not an implementation
source.

The current slice does three things:

1. converts the checksum-pinned TOPAS v6 inputs into neutral XY, CIF,
   GSAS-II instrument, and JSON records;
2. preserves all seven deposited emission lines and the wavelength-edge
   parameters as typed, unit-explicit metadata;
3. compares native PhaseSmith and pinned GSAS-II refinements using the same
   converted common inputs.

It does **not** claim that either common-subset refinement reproduces the
deposited TOPAS fundamental-parameters calculation.

## Deposited model map

| Deposited term | PhaseSmith status | Conversion/workflow treatment |
| --- | --- | --- |
| Three structures, scales, cells, coordinates, displacement values | Supported | Translated to independent CIFs; the present workflow refines scales and isotropic displacement values but holds cells and coordinates fixed |
| Seven-term background | Supported | Native workflows use their own background parameterizations; these are documented as non-matched |
| `la`, `lo` discrete wavelengths | Supported | Common workflow reduces the Kα groups to area-weighted Kα1/Kα2 centroids; the complete seven-line table is retained |
| Lorentzian `lh` and Gaussian `lg` wavelength widths | Represented, production convolution missing | `EmissionLine` stores both HWHM values in mÅ; deterministic discretization is available only for convergence studies |
| Smooth Ni absorption-edge multiplier | Represented, production convolution missing | `ErrorFunctionWavelengthFilter` stores and evaluates the deposited wavelength-space edge |
| Angle-dependent white continuum | Missing | Explicitly omitted; it cannot be represented by fixed wavelength weights because the deposited expression contains a Bragg-angle factor |
| Full axial geometry and 2.5° primary/secondary Soller slits | Missing | Existing FCJ supports finite source/sample-receiver heights but not Soller-limited incident and diffracted beams |
| LPSD equatorial divergence and `Tube_Tails` | Missing | No equivalent production convolution yet |
| Flat-plate specimen absorption | Missing | The current Bragg–Brentano correction covers Lorentz-polarization and displacement, not finite-thickness absorption |
| `CS_L`, `CS_G`, `Strain_L`, `Strain_G` | Partial | PhaseSmith has Lorentzian isotropic size and Gaussian RMS microstrain; independent Gaussian-size and Lorentzian-strain terms still need explicit conventions |

TOPAS documents `la` as integrated line area, `lo` as wavelength in Å, and
`lh`/`lg` as Lorentzian/Gaussian HWHM in mÅ. Its source-emission model is a sum
of Voigt lines. See the official
[emission-line reference](https://topas.awh.durham.ac.uk/doku.php?id=l) and
[kernel description](https://topas.awh.durham.ac.uk/doku.php?id=manual_part_1).

## Common-subset real-pattern result

The comparison uses Rowles mixtures `1a` and `1e`, the 21–150° range, the same
converted structures, a common Cu Kα doublet, the same neutral U/V/W and SH/L
initializers, and the exact pinned GSAS-II revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`. Backgrounds and staged refinement
recipes remain native and are not parameter-for-parameter matched.

| Sample | Program | Al2O3 | ZnO | CaF2 | Poisson Rwp | Profile correlation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1a | PhaseSmith | 1.972% | 3.444% | 94.585% | 13.546% | 0.98366 |
| 1a | GSAS-II | 1.399% | 3.304% | 95.297% | 9.085% | 0.99689 |
| 1e | PhaseSmith | 57.837% | 14.168% | 27.994% | 22.454% | 0.95505 |
| 1e | GSAS-II | 57.038% | 14.078% | 28.884% | 8.195% | 0.99504 |

The largest PhaseSmith/GSAS-II phase-fraction difference is 0.713 percentage
points for `1a` and 0.889 percentage points for `1e`. Thus the current
structural/QPA path is already useful. The large residual gap—especially for
`1e`—is the direct motivation for the missing optics work.

## Rejected discrete-spectrum shortcut

The typed emission model can be discretized into existing
`WavelengthComponents`. This is valuable for unit tests and convergence
studies, but it is not an acceptable production implementation of continuous
Voigt emission lines. With nine Lorentzian nodes per deposited line (63 fixed
components), the full refinement became tens of times slower and worsened Rwp:

| Sample | Common Kα doublet | 63-component filtered quadrature |
| --- | ---: | ---: |
| 1a | 13.546% | 23.199% |
| 1e | 22.454% | 24.076% |

For a controlled single reflection, increasing from 9 to 33 nodes per line
still changed the profile by about 2.0% in relative L2 norm against a 65-node
calculation. Sparse delta components are therefore under-resolved relative to
the narrow angular peak. This diagnostic is not part of the accepted Rowles
workflow and is not recorded as a golden refinement result.

## Production implementation sequence

### 1. Continuous source-emission convolution

Implement a Rust-owned wavelength integral for a sum of normalized Voigt
lines. The line parameters remain typed as integrated area, central wavelength,
Lorentzian HWHM, and Gaussian HWHM. For every reflection and wavelength node,
Bragg's law maps wavelength to angle before the angular instrument/specimen
profile is evaluated. Adaptive order must be selected from a documented
resolution ratio, not a global fixed node count.

Values and analytical derivatives with respect to line area, centre, both
widths, and structural/instrument parameters must be accumulated in the same
line/reflection/sample pass. Finite wavelength and angular support endpoints
are inclusive and form part of the public numerical convention. The
independent Python reference should use a deliberately different, high-order
integration strategy.

### 2. Edge transmission and white continuum

The deposited smooth edge multiplier is

```text
T(lambda) = edge_extra
          + 0.5 [1 + erf(a_erf (lambda - edge))].
```

It must multiply the continuous emission profile before the angular
convolutions. The deposited white term is angle-dependent and must be a
separate source contribution, not hidden in fixed line areas. TOPAS describes
the ordering in its
[absorption-edge documentation](https://topas.awh.durham.ac.uk/flarum/public/d/489-absorption-edge-macro).

### 3. Soller-limited axial convolution

Extend the existing FCJ geometry with distinct source, specimen, and receiver
lengths plus primary and secondary Soller apertures. The implementation source
is the published axial-divergence treatment of Cheary and Coelho:
[part I](https://doi.org/10.1107/S0021889898006876) and
[part II](https://doi.org/10.1107/S0021889898006888). The current FCJ model
remains available as a smaller, independently useful geometry rather than
silently changing semantics.

### 4. Equatorial detector and tube-tail convolution

Add typed LPSD angular range, equatorial divergence, and tube-tail parameters.
These terms need an equation/source ledger, normalized area tests, support
tests, analytical derivatives, and a realistic multi-peak benchmark before
entering a refinement recipe.

### 5. Specimen absorption and remaining microstructure terms

Add the flat-plate attenuation convolution and explicit Gaussian/Lorentzian
size/strain conventions. The general physical decomposition follows the
fundamental-parameters approach of Cheary and Coelho,
[J. Appl. Cryst. 25 (1992) 109–121](https://doi.org/10.1107/S0021889891010804),
and the laboratory-diffractometer review by Cheary, Coelho, and Cline,
[J. Res. NIST 109 (2004) 1–25](https://doi.org/10.6028/jres.109.002).

## Acceptance gates

Each implementation slice must pass all normal numerical-change requirements
plus these real-pattern gates:

- deterministic values and derivatives across worker counts;
- analytical/centered-finite-difference agreement away from support changes;
- unit integrated area for every individual convolution and the composed
  profile;
- explicit low-, middle-, and high-angle moment comparisons against an
  independent Python reference;
- no regression above 1 percentage point in either Rowles QPA result;
- a decreasing Rwp trend on both patterns, with `1e` as the discriminating
  optics case;
- black-box comparison against the deposited TOPAS result when a separately
  licensed TOPAS v6 installation is available, and against pinned GSAS-II only
  for the common terms that GSAS-II actually represents.

The Rowles archive is provenance- and checksum-pinned in the validation
registry. The dataset is CC BY 4.0 and is described by Curtin University at
[DOI 10.25917/5f44ad65411cc](https://doi.org/10.25917/5f44ad65411cc). No TOPAS
or GSAS-II implementation code is copied or made a PhaseSmith dependency.
