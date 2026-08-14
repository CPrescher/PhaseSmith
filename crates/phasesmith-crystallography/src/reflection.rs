//! Bounded, deterministic reciprocal-family generation.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::cell::{CELL_PARAMETER_COUNT, CellError, CellGeometry, UnitCell};
use crate::symmetry::{SpaceGroup, SymmetryError};

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const DEFAULT_METRIC_TOLERANCE: f64 = 1.0e-10;

/// Physical range used to select reciprocal families.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReflectionRange {
    /// Inclusive d-spacing interval in ångströms.
    DSpacing {
        /// Smallest included d-spacing.
        min_angstrom: f64,
        /// Largest included d-spacing.
        max_angstrom: f64,
    },
    /// Inclusive scattering-vector interval `Q = 2 pi / d`.
    ScatteringVector {
        /// Smallest included Q in inverse ångströms.
        min_inverse_angstrom: f64,
        /// Largest included Q in inverse ångströms.
        max_inverse_angstrom: f64,
    },
    /// Inclusive monochromatic constant-wavelength `2 theta` interval.
    CwTwoTheta {
        /// Smallest included `2 theta` in degrees.
        min_deg: f64,
        /// Largest included `2 theta` in degrees.
        max_deg: f64,
        /// Monochromatic wavelength in ångströms.
        wavelength_angstrom: f64,
    },
    /// Inclusive TOF interval with an explicit safe d-spacing search interval.
    Tof {
        /// Smallest included time-of-flight coordinate in microseconds.
        min_us: f64,
        /// Largest included time-of-flight coordinate in microseconds.
        max_us: f64,
        /// Smallest d-spacing searched, in ångströms.
        search_min_d_angstrom: f64,
        /// Largest d-spacing searched, in ångströms.
        search_max_d_angstrom: f64,
        /// TOF zero offset in microseconds.
        zero_us: f64,
        /// Linear calibration coefficient in microseconds per ångström.
        difc_us_per_angstrom: f64,
        /// Quadratic coefficient in microseconds per square ångström.
        difa_us_per_angstrom2: f64,
        /// Inverse-d coefficient in microsecond ångströms.
        difb_us_angstrom: f64,
    },
}

impl ReflectionRange {
    fn reciprocal_bounds(self) -> Result<(f64, f64), ReflectionGenerationError> {
        self.validate()?;
        Ok(match self {
            Self::DSpacing {
                min_angstrom,
                max_angstrom,
            } => (max_angstrom.recip(), min_angstrom.recip()),
            Self::ScatteringVector {
                min_inverse_angstrom,
                max_inverse_angstrom,
            } => (min_inverse_angstrom / TWO_PI, max_inverse_angstrom / TWO_PI),
            Self::CwTwoTheta {
                min_deg,
                max_deg,
                wavelength_angstrom,
            } => {
                let min_theta = 0.5 * min_deg.to_radians();
                let max_theta = 0.5 * max_deg.to_radians();
                (
                    2.0 * min_theta.sin() / wavelength_angstrom,
                    2.0 * max_theta.sin() / wavelength_angstrom,
                )
            }
            Self::Tof {
                search_min_d_angstrom,
                search_max_d_angstrom,
                ..
            } => (search_max_d_angstrom.recip(), search_min_d_angstrom.recip()),
        })
    }

    fn validate(self) -> Result<(), ReflectionGenerationError> {
        let finite = match self {
            Self::DSpacing {
                min_angstrom,
                max_angstrom,
            } => {
                min_angstrom.is_finite()
                    && max_angstrom.is_finite()
                    && min_angstrom > 0.0
                    && max_angstrom >= min_angstrom
            }
            Self::ScatteringVector {
                min_inverse_angstrom,
                max_inverse_angstrom,
            } => {
                min_inverse_angstrom.is_finite()
                    && max_inverse_angstrom.is_finite()
                    && min_inverse_angstrom >= 0.0
                    && max_inverse_angstrom > 0.0
                    && max_inverse_angstrom >= min_inverse_angstrom
            }
            Self::CwTwoTheta {
                min_deg,
                max_deg,
                wavelength_angstrom,
            } => {
                min_deg.is_finite()
                    && max_deg.is_finite()
                    && wavelength_angstrom.is_finite()
                    && min_deg >= 0.0
                    && max_deg < 180.0
                    && max_deg >= min_deg
                    && wavelength_angstrom > 0.0
            }
            Self::Tof {
                min_us,
                max_us,
                search_min_d_angstrom,
                search_max_d_angstrom,
                zero_us,
                difc_us_per_angstrom,
                difa_us_per_angstrom2,
                difb_us_angstrom,
            } => {
                [
                    min_us,
                    max_us,
                    search_min_d_angstrom,
                    search_max_d_angstrom,
                    zero_us,
                    difc_us_per_angstrom,
                    difa_us_per_angstrom2,
                    difb_us_angstrom,
                ]
                .into_iter()
                .all(f64::is_finite)
                    && max_us >= min_us
                    && search_min_d_angstrom > 0.0
                    && search_max_d_angstrom >= search_min_d_angstrom
            }
        };
        if finite {
            Ok(())
        } else {
            Err(ReflectionGenerationError::InvalidRange)
        }
    }

    fn contains(self, reciprocal_length: f64, d_spacing: f64) -> bool {
        match self {
            Self::DSpacing {
                min_angstrom,
                max_angstrom,
            } => inclusive_contains(d_spacing, min_angstrom, max_angstrom),
            Self::ScatteringVector {
                min_inverse_angstrom,
                max_inverse_angstrom,
            } => inclusive_contains(
                TWO_PI * reciprocal_length,
                min_inverse_angstrom,
                max_inverse_angstrom,
            ),
            Self::CwTwoTheta {
                min_deg,
                max_deg,
                wavelength_angstrom,
            } => {
                let argument = 0.5 * wavelength_angstrom * reciprocal_length;
                if argument > 1.0 {
                    return false;
                }
                let two_theta = 2.0 * argument.asin().to_degrees();
                inclusive_contains(two_theta, min_deg, max_deg)
            }
            Self::Tof {
                min_us,
                max_us,
                zero_us,
                difc_us_per_angstrom,
                difa_us_per_angstrom2,
                difb_us_angstrom,
                ..
            } => {
                let tof = zero_us
                    + difc_us_per_angstrom * d_spacing
                    + difa_us_per_angstrom2 * d_spacing * d_spacing
                    + difb_us_angstrom / d_spacing;
                inclusive_contains(tof, min_us, max_us)
            }
        }
    }
}

/// One generated reciprocal family with metric-dependent values.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedReflection {
    /// Stable canonical Miller-index ID.
    pub reflection_id: String,
    /// Canonical Miller representative used for calculation and stable identity.
    pub hkl: [i32; 3],
    /// Human-facing representative selected from the same exact reciprocal orbit.
    pub conventional_hkl: [i32; 3],
    /// Powder multiplicity under the configured Friedel policy.
    pub multiplicity: usize,
    /// D-spacing in ångströms.
    pub d_spacing_angstrom: f64,
    /// Reciprocal length `1 / d` in inverse ångströms, without `2 pi`.
    pub reciprocal_length_inverse_angstrom: f64,
    /// Analytical d-spacing derivatives in direct-cell parameter order.
    pub d_spacing_derivatives: [f64; CELL_PARAMETER_COUNT],
}

/// A prepared generator that caches group topology and recomputes cell metrics.
#[derive(Clone, Debug)]
pub struct PreparedReflectionGenerator {
    space_group: SpaceGroup,
    merge_friedel: bool,
    max_candidates: usize,
    metric_tolerance: f64,
}

impl PreparedReflectionGenerator {
    /// Create a generator with an explicit brute-force candidate safety limit.
    ///
    /// A candidate is an integer triplet in the safe reciprocal-metric box;
    /// the default metric compatibility tolerance is `1e-10` relative.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_candidates` is zero.
    pub fn new(
        space_group: SpaceGroup,
        merge_friedel: bool,
        max_candidates: usize,
    ) -> Result<Self, ReflectionGenerationError> {
        if max_candidates == 0 {
            return Err(ReflectionGenerationError::InvalidCandidateLimit);
        }
        Ok(Self {
            space_group,
            merge_friedel,
            max_candidates,
            metric_tolerance: DEFAULT_METRIC_TOLERANCE,
        })
    }

    /// Borrow the validated symmetry group.
    #[must_use]
    pub const fn space_group(&self) -> &SpaceGroup {
        &self.space_group
    }

    /// Whether Friedel mates are merged into one powder family.
    #[must_use]
    pub const fn merge_friedel(&self) -> bool {
        self.merge_friedel
    }

    /// Generate unique, allowed families sorted by increasing reciprocal length.
    ///
    /// The index box uses a reciprocal-eigenvalue lower bound plus exact
    /// ellipsoid projections, so skewed triclinic cells cannot omit valid
    /// indices. Endpoints are inclusive within 64 floating-point epsilons.
    /// Accidental equal-d families remain separate records.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid cell/range, a cell incompatible with the
    /// point group, exact symmetry arithmetic failure, or a candidate cube over
    /// the configured safety limit.
    pub fn generate(
        &self,
        cell: UnitCell,
        range: ReflectionRange,
    ) -> Result<Vec<GeneratedReflection>, ReflectionGenerationError> {
        let geometry = cell.geometry()?;
        validate_metric_compatibility(
            &geometry,
            self.space_group.metric_constraints().equations.as_slice(),
            self.metric_tolerance,
        )?;
        let (min_reciprocal, max_reciprocal) = range.reciprocal_bounds()?;
        let bounds = safe_index_bounds(&geometry, max_reciprocal)?;
        let mut sides = [0_usize; 3];
        for (index, side) in sides.iter_mut().enumerate() {
            *side = usize::try_from(2_i64 * i64::from(bounds[index]) + 1)
                .map_err(|_| ReflectionGenerationError::CandidateLimitExceeded)?;
        }
        let candidate_count = sides[0]
            .checked_mul(sides[1])
            .and_then(|value| value.checked_mul(sides[2]))
            .ok_or(ReflectionGenerationError::CandidateLimitExceeded)?;
        if candidate_count > self.max_candidates {
            return Err(ReflectionGenerationError::CandidateLimitExceeded);
        }

        let min_squared = min_reciprocal * min_reciprocal;
        let max_squared = max_reciprocal * max_reciprocal;
        let boundary_tolerance = 64.0 * f64::EPSILON * max_squared.max(1.0);
        let mut reflections = Vec::new();
        for h in -bounds[0]..=bounds[0] {
            for k in -bounds[1]..=bounds[1] {
                for l in -bounds[2]..=bounds[2] {
                    let hkl = [h, k, l];
                    if hkl == [0, 0, 0] {
                        continue;
                    }
                    let reciprocal_squared = geometry.q_squared(hkl);
                    if reciprocal_squared + boundary_tolerance < min_squared
                        || reciprocal_squared - boundary_tolerance > max_squared
                    {
                        continue;
                    }
                    let family = self
                        .space_group
                        .reflection_family(hkl, self.merge_friedel)?;
                    if family.canonical_hkl != hkl {
                        continue;
                    }
                    if self.space_group.is_systematically_absent(hkl)? {
                        continue;
                    }
                    let (d_spacing, derivatives) = geometry.d_spacing_and_derivatives(hkl)?;
                    let reciprocal_length = reciprocal_squared.sqrt();
                    if !range.contains(reciprocal_length, d_spacing) {
                        continue;
                    }
                    reflections.push(GeneratedReflection {
                        reflection_id: family.reflection_id,
                        hkl,
                        conventional_hkl: family.conventional_hkl,
                        multiplicity: family.multiplicity,
                        d_spacing_angstrom: d_spacing,
                        reciprocal_length_inverse_angstrom: reciprocal_length,
                        d_spacing_derivatives: derivatives,
                    });
                }
            }
        }
        reflections.sort_by(|left, right| {
            left.reciprocal_length_inverse_angstrom
                .total_cmp(&right.reciprocal_length_inverse_angstrom)
                .then_with(|| left.hkl.cmp(&right.hkl))
        });
        Ok(reflections)
    }
}

/// Reflection generation validation or numerical error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionGenerationError {
    /// Unit-cell geometry was invalid.
    Cell(CellError),
    /// Exact group arithmetic failed.
    Symmetry(SymmetryError),
    /// A physical range was non-finite, reversed, or outside its domain.
    InvalidRange,
    /// Candidate limit was zero.
    InvalidCandidateLimit,
    /// The safe index box exceeded the configured candidate limit.
    CandidateLimitExceeded,
    /// The cell metric violates exact rotational constraints.
    CellSymmetryMismatch,
    /// A positive reciprocal-metric eigenvalue could not be obtained.
    DegenerateReciprocalMetric,
}

impl Display for ReflectionGenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cell(error) => Display::fmt(error, formatter),
            Self::Symmetry(error) => Display::fmt(error, formatter),
            Self::InvalidRange => formatter.write_str("reflection range is invalid"),
            Self::InvalidCandidateLimit => {
                formatter.write_str("reflection candidate limit must be positive")
            }
            Self::CandidateLimitExceeded => {
                formatter.write_str("safe reflection candidate box exceeds the configured limit")
            }
            Self::CellSymmetryMismatch => {
                formatter.write_str("unit-cell metric is incompatible with the symmetry rotations")
            }
            Self::DegenerateReciprocalMetric => {
                formatter.write_str("reciprocal metric must be finite and positive definite")
            }
        }
    }
}

impl Error for ReflectionGenerationError {}

impl From<CellError> for ReflectionGenerationError {
    fn from(value: CellError) -> Self {
        Self::Cell(value)
    }
}

impl From<SymmetryError> for ReflectionGenerationError {
    fn from(value: SymmetryError) -> Self {
        Self::Symmetry(value)
    }
}

#[allow(clippy::cast_precision_loss)]
fn validate_metric_compatibility(
    geometry: &CellGeometry,
    equations: &[[i64; 6]],
    tolerance: f64,
) -> Result<(), ReflectionGenerationError> {
    let metric = geometry.direct_metric;
    let components = [
        metric[0][0],
        metric[1][1],
        metric[2][2],
        metric[1][2],
        metric[0][2],
        metric[0][1],
    ];
    let scale = components
        .iter()
        .copied()
        .map(f64::abs)
        .fold(1.0_f64, f64::max);
    for equation in equations {
        let residual = equation
            .iter()
            .zip(components)
            .map(|(coefficient, value)| *coefficient as f64 * value)
            .sum::<f64>();
        let coefficient_scale = equation.iter().copied().map(i64::unsigned_abs).sum::<u64>() as f64;
        if residual.abs() > tolerance * scale * coefficient_scale.max(1.0) {
            return Err(ReflectionGenerationError::CellSymmetryMismatch);
        }
    }
    Ok(())
}

fn inclusive_contains(value: f64, minimum: f64, maximum: f64) -> bool {
    let tolerance =
        64.0 * f64::EPSILON * value.abs().max(minimum.abs()).max(maximum.abs()).max(1.0);
    value + tolerance >= minimum && value - tolerance <= maximum
}

fn safe_index_bounds(
    geometry: &CellGeometry,
    max_reciprocal: f64,
) -> Result<[i32; 3], ReflectionGenerationError> {
    let direct_diagonal = [
        geometry.direct_metric[0][0],
        geometry.direct_metric[1][1],
        geometry.direct_metric[2][2],
    ];
    let reciprocal_eigenvalue_lower_bound = direct_diagonal.iter().sum::<f64>().recip();
    if !reciprocal_eigenvalue_lower_bound.is_finite() || reciprocal_eigenvalue_lower_bound <= 0.0 {
        return Err(ReflectionGenerationError::DegenerateReciprocalMetric);
    }
    let common_bound = max_reciprocal / reciprocal_eigenvalue_lower_bound.sqrt();
    let safety_factor = 1.0 + 64.0 * f64::EPSILON;
    let mut bounds = [0; 3];
    for (index, bound) in bounds.iter_mut().enumerate() {
        // Ellipsoid projection gives |h_i| <= q_max sqrt((G*)^-1_ii).
        // The reciprocal-eigenvalue bound above is a conservative cross-check.
        let projected = max_reciprocal * direct_diagonal[index].sqrt();
        let value = (projected.min(common_bound) * safety_factor).ceil() + 1.0;
        if !value.is_finite() || value > f64::from(i32::MAX - 1) {
            return Err(ReflectionGenerationError::CandidateLimitExceeded);
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            *bound = value as i32;
        }
    }
    Ok(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry::{Rational, SymmetryOperation};

    fn cubic_cell() -> UnitCell {
        UnitCell {
            a_angstrom: 1.0,
            b_angstrom: 1.0,
            c_angstrom: 1.0,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        }
    }

    fn identity_group() -> SpaceGroup {
        SpaceGroup::new(vec![SymmetryOperation::identity()]).expect("P1")
    }

    fn body_centred_group() -> SpaceGroup {
        let half = Rational::new(1, 2).expect("half");
        SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new(SymmetryOperation::identity().rotation(), [half, half, half])
                .expect("centring operation"),
        ])
        .expect("I centring group")
    }

    #[test]
    fn p1_cubic_generation_has_expected_families_and_multiplicity() {
        let generator =
            PreparedReflectionGenerator::new(identity_group(), true, 1_000_000).expect("generator");
        let reflections = generator
            .generate(
                cubic_cell(),
                ReflectionRange::DSpacing {
                    min_angstrom: 0.7,
                    max_angstrom: 1.0,
                },
            )
            .expect("reflections");
        assert_eq!(reflections.len(), 9);
        assert!(
            reflections
                .iter()
                .all(|reflection| reflection.multiplicity == 2)
        );
        assert!(reflections.windows(2).all(|pair| {
            pair[0].reciprocal_length_inverse_angstrom <= pair[1].reciprocal_length_inverse_angstrom
        }));
    }

    #[test]
    fn body_centring_removes_odd_index_sum() {
        let generator = PreparedReflectionGenerator::new(body_centred_group(), true, 1_000_000)
            .expect("generator");
        let reflections = generator
            .generate(
                cubic_cell(),
                ReflectionRange::DSpacing {
                    min_angstrom: 0.7,
                    max_angstrom: 1.0,
                },
            )
            .expect("reflections");
        assert_eq!(reflections.len(), 6);
        assert!(
            reflections
                .iter()
                .all(|reflection| reflection.hkl.into_iter().sum::<i32>() % 2 == 0)
        );
    }

    #[test]
    fn physical_range_forms_select_the_same_cubic_shell() {
        let generator =
            PreparedReflectionGenerator::new(identity_group(), true, 1_000_000).expect("generator");
        let d_range = generator
            .generate(
                cubic_cell(),
                ReflectionRange::DSpacing {
                    min_angstrom: 0.7,
                    max_angstrom: 1.0,
                },
            )
            .expect("d range");
        let q_range = generator
            .generate(
                cubic_cell(),
                ReflectionRange::ScatteringVector {
                    min_inverse_angstrom: TWO_PI,
                    max_inverse_angstrom: TWO_PI * 2.0_f64.sqrt(),
                },
            )
            .expect("Q range");
        let cw_range = generator
            .generate(
                cubic_cell(),
                ReflectionRange::CwTwoTheta {
                    min_deg: 2.0 * 0.5_f64.asin().to_degrees(),
                    max_deg: 2.0 * (0.5 * 2.0_f64.sqrt()).asin().to_degrees(),
                    wavelength_angstrom: 1.0,
                },
            )
            .expect("CW range");
        let expected = d_range.iter().map(|item| item.hkl).collect::<Vec<_>>();
        assert_eq!(
            q_range.iter().map(|item| item.hkl).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            cw_range.iter().map(|item| item.hkl).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn tof_filter_and_candidate_limit_are_explicit() {
        let generator =
            PreparedReflectionGenerator::new(identity_group(), true, 1_000_000).expect("generator");
        let reflections = generator
            .generate(
                cubic_cell(),
                ReflectionRange::Tof {
                    min_us: 900.0,
                    max_us: 1100.0,
                    search_min_d_angstrom: 0.5,
                    search_max_d_angstrom: 1.5,
                    zero_us: 0.0,
                    difc_us_per_angstrom: 1000.0,
                    difa_us_per_angstrom2: 0.0,
                    difb_us_angstrom: 0.0,
                },
            )
            .expect("TOF range");
        assert!(
            reflections
                .iter()
                .all(|reflection| (reflection.d_spacing_angstrom - 1.0).abs() < 1e-14)
        );

        let limited = PreparedReflectionGenerator::new(identity_group(), true, 10)
            .expect("limited generator");
        assert_eq!(
            limited.generate(
                cubic_cell(),
                ReflectionRange::DSpacing {
                    min_angstrom: 0.1,
                    max_angstrom: 1.0,
                }
            ),
            Err(ReflectionGenerationError::CandidateLimitExceeded)
        );
    }

    #[test]
    fn incompatible_cell_and_point_group_is_rejected() {
        let quarter_turn =
            SymmetryOperation::new([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [Rational::zero(); 3])
                .expect("quarter turn");
        let half_turn = quarter_turn.compose(quarter_turn).expect("half turn");
        let three_quarters = quarter_turn.compose(half_turn).expect("three-quarter turn");
        let tetragonal = SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            quarter_turn,
            half_turn,
            three_quarters,
        ])
        .expect("four-fold group");
        let generator =
            PreparedReflectionGenerator::new(tetragonal, true, 1_000_000).expect("generator");
        let incompatible = UnitCell {
            b_angstrom: 1.1,
            ..cubic_cell()
        };
        assert_eq!(
            generator.generate(
                incompatible,
                ReflectionRange::DSpacing {
                    min_angstrom: 0.5,
                    max_angstrom: 2.0,
                }
            ),
            Err(ReflectionGenerationError::CellSymmetryMismatch)
        );
    }
}
