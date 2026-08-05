# Pinned GSAS-II symmetry/reflection behavior study

Status: schema and documentation survey recorded; live fixture generation is
pending an available external pinned checkout.

This study is validation-only. GSAS-II is not imported by the package, no
GSAS-II objects cross into Rust or public Python models, and no implementation
source is copied or translated.

## Pinned target and public-first protocol

Use the exact revision already recorded for structural behavior in
`STRUCTURAL_PARAMETER_STUDY.md`. Prefer the public scripting API to create or
load one phase and obtain its powder reflection list. A small version-gated
probe may extract only intermediate fields that the public layer does not
expose.

For each case record plain JSON/NPZ values:

- exact input cell and symmetry operations after setting resolution;
- requested d, Q, monochromatic `2theta`, or TOF limits and boundary policy;
- emitted `hkl`, d-spacing, multiplicity, and absence status;
- asymmetric sites and expanded equivalent positions for special-position
  cases;
- the GSAS-II revision, adapter version, input checksum, and platform.

Required cases are P1, P-1, P2_1, a primitive glide group, I centring, F
centring, one three-fold screw, and one non-standard setting. Perturb limits
around a reflection boundary and perturb a special-position coordinate on both
sides of its tolerance. Run both ordinary non-resonant and an anomalous case to
identify whether Friedel mates are merged in each public workflow.

## Review questions

The fixture review must explicitly answer:

1. Which reciprocal representative and sort order are used?
2. Does reported multiplicity include Friedel mates, centring translations,
   or both?
3. At what stage are systematic absences removed?
4. Which cell/setting transformation is applied before reflection generation?
5. Are d/angle/TOF endpoints inclusive, and at what numerical tolerance?
6. How are special positions deduplicated and their multiplicities reported?

Differences are not automatically bugs. The native convention in
`docs/symmetry-reflections.md` remains authoritative unless a scientifically
necessary compatibility adapter is deliberately specified. No oracle
equivalence is claimed until the live fixture exists and its numerical diff is
reviewed.
