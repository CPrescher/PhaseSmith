//! Exact crystallographic symmetry operations and reciprocal-space topology.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::OnceLock;

/// Maximum common phase denominator accepted by exact absence detection.
const MAX_PHASE_DENOMINATOR: i64 = 4096;
const MAX_PHASE_DENOMINATOR_USIZE: usize = 4096;
static CYCLOTOMIC_POLYNOMIALS: [OnceLock<Result<Vec<i64>, SymmetryError>>;
    MAX_PHASE_DENOMINATOR_USIZE + 1] = [const { OnceLock::new() }; MAX_PHASE_DENOMINATOR_USIZE + 1];

/// Reduced rational number used for exact fractional translations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    /// Construct a reduced rational number with a positive denominator.
    ///
    /// # Errors
    ///
    /// Returns [`SymmetryError::ZeroDenominator`] for a zero denominator.
    pub fn new(numerator: i64, denominator: i64) -> Result<Self, SymmetryError> {
        if denominator == 0 {
            return Err(SymmetryError::ZeroDenominator);
        }
        let sign = if denominator < 0 { -1 } else { 1 };
        let numerator = numerator
            .checked_mul(sign)
            .ok_or(SymmetryError::ArithmeticOverflow)?;
        let denominator = denominator
            .checked_mul(sign)
            .ok_or(SymmetryError::ArithmeticOverflow)?;
        let divisor = gcd_i64(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    /// Integer zero.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }

    /// Reduced numerator.
    #[must_use]
    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    /// Positive reduced denominator.
    #[must_use]
    pub const fn denominator(self) -> i64 {
        self.denominator
    }

    /// Canonical representative in the half-open interval `[0, 1)`.
    #[must_use]
    pub fn modulo_one(self) -> Self {
        Self {
            numerator: self.numerator.rem_euclid(self.denominator),
            denominator: self.denominator,
        }
        .reduced()
    }

    /// Floating-point value for applying an exact operation to coordinates.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    fn reduced(self) -> Self {
        let divisor = gcd_i64(self.numerator, self.denominator);
        Self {
            numerator: self.numerator / divisor,
            denominator: self.denominator / divisor,
        }
    }

    fn checked_add(self, other: Self) -> Result<Self, SymmetryError> {
        let common = lcm_i64(self.denominator, other.denominator)?;
        let left = self
            .numerator
            .checked_mul(common / self.denominator)
            .ok_or(SymmetryError::ArithmeticOverflow)?;
        let right = other
            .numerator
            .checked_mul(common / other.denominator)
            .ok_or(SymmetryError::ArithmeticOverflow)?;
        Self::new(
            left.checked_add(right)
                .ok_or(SymmetryError::ArithmeticOverflow)?,
            common,
        )
    }

    fn checked_mul_integer(self, factor: i64) -> Result<Self, SymmetryError> {
        Self::new(
            self.numerator
                .checked_mul(factor)
                .ok_or(SymmetryError::ArithmeticOverflow)?,
            self.denominator,
        )
    }
}

/// Exact affine symmetry operation `x' = R x + t` in fractional coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymmetryOperation {
    rotation: [[i32; 3]; 3],
    translation: [Rational; 3],
}

impl SymmetryOperation {
    /// Validate a unimodular integer rotation and normalize its translation.
    ///
    /// # Errors
    ///
    /// Returns [`SymmetryError::NonUnimodularRotation`] unless the determinant
    /// is exactly `+1` or `-1`.
    pub fn new(rotation: [[i32; 3]; 3], translation: [Rational; 3]) -> Result<Self, SymmetryError> {
        if determinant_i32(rotation).unsigned_abs() != 1 {
            return Err(SymmetryError::NonUnimodularRotation);
        }
        Ok(Self {
            rotation,
            translation: translation.map(Rational::modulo_one),
        })
    }

    /// Identity operation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [Rational::zero(); 3],
        }
    }

    /// Integer direct-space rotation matrix.
    #[must_use]
    pub const fn rotation(self) -> [[i32; 3]; 3] {
        self.rotation
    }

    /// Exact normalized fractional translation.
    #[must_use]
    pub const fn translation(self) -> [Rational; 3] {
        self.translation
    }

    /// Compose `self` after `right`, returning `self(right(x))`.
    ///
    /// # Errors
    ///
    /// Returns an arithmetic error if exact intermediate values overflow.
    pub fn compose(self, right: Self) -> Result<Self, SymmetryError> {
        let rotation = multiply_rotation(self.rotation, right.rotation)?;
        let mut translation = [Rational::zero(); 3];
        for (row, value) in translation.iter_mut().enumerate() {
            let mut translated = self.translation[row];
            for column in 0..3 {
                translated = translated.checked_add(
                    right.translation[column]
                        .checked_mul_integer(i64::from(self.rotation[row][column]))?,
                )?;
            }
            *value = translated.modulo_one();
        }
        Self::new(rotation, translation)
    }

    /// Apply the operation and wrap coordinates into `[0, 1)`.
    #[must_use]
    pub fn apply_fractional(self, xyz: [f64; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        for (row, value) in result.iter_mut().enumerate() {
            *value = (self.translation[row].as_f64()
                + (0..3)
                    .map(|column| f64::from(self.rotation[row][column]) * xyz[column])
                    .sum::<f64>())
            .rem_euclid(1.0);
        }
        result
    }

    /// Apply the reciprocal orbit action `h' = R^T h`.
    ///
    /// The full group contains inverse rotations, so this convention produces
    /// the same orbit as `R^-T h` while matching the structure-factor identity
    /// `F(h) = exp(2 pi i h.t) F(R^T h)`.
    ///
    /// # Errors
    ///
    /// Returns an arithmetic error if an index does not fit signed 32-bit.
    pub fn reciprocal_index(self, hkl: [i32; 3]) -> Result<[i32; 3], SymmetryError> {
        let mut result = [0; 3];
        for (column, value) in result.iter_mut().enumerate() {
            let sum = (0..3).try_fold(0_i64, |accumulator, row| {
                accumulator
                    .checked_add(i64::from(self.rotation[row][column]) * i64::from(hkl[row]))
                    .ok_or(SymmetryError::ArithmeticOverflow)
            })?;
            *value = i32::try_from(sum).map_err(|_| SymmetryError::ArithmeticOverflow)?;
        }
        Ok(result)
    }

    fn phase(self, hkl: [i32; 3]) -> Result<Rational, SymmetryError> {
        let mut phase = Rational::zero();
        for (index, translation) in hkl.into_iter().zip(self.translation) {
            phase = phase.checked_add(translation.checked_mul_integer(i64::from(index))?)?;
        }
        Ok(phase.modulo_one())
    }
}

/// Crystallographic system inferred from the exact rotational group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrystalSystem {
    /// No rotational metric constraints.
    Triclinic,
    /// One two-fold direction or mirror-normal family.
    Monoclinic,
    /// Three perpendicular two-fold direction families.
    Orthorhombic,
    /// A four-fold direction.
    Tetragonal,
    /// A three-fold direction without cubic topology.
    Trigonal,
    /// A six-fold direction.
    Hexagonal,
    /// Cubic rotational topology.
    Cubic,
}

/// Exact homogeneous constraints on direct-metric components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricConstraints {
    /// Integer equations in order `(g11, g22, g33, g23, g13, g12)`.
    pub equations: Vec<[i64; 6]>,
    /// Dimension of the symmetry-allowed metric subspace.
    pub independent_parameter_count: usize,
    /// Integer nullspace basis in the same metric-component order.
    ///
    /// Every compatible metric vector is a linear combination of these rows,
    /// and every such combination satisfies all [`Self::equations`].
    pub parameterization_basis: Vec<[i64; 6]>,
}

/// A validated, closed set of exact symmetry operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpaceGroup {
    operations: Vec<SymmetryOperation>,
    rotations: Vec<[[i32; 3]; 3]>,
    metric_constraints: MetricConstraints,
    crystal_system: CrystalSystem,
}

impl SpaceGroup {
    /// Validate identity, uniqueness, and exact group closure.
    ///
    /// Operation order is canonicalized so downstream results do not depend on
    /// input order.
    ///
    /// # Errors
    ///
    /// Returns a [`SymmetryError`] for an empty, duplicate, identity-free, or
    /// non-closed operation set.
    pub fn new(mut operations: Vec<SymmetryOperation>) -> Result<Self, SymmetryError> {
        if operations.is_empty() {
            return Err(SymmetryError::EmptyOperationSet);
        }
        operations.sort_unstable();
        if operations.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(SymmetryError::DuplicateOperation);
        }
        if operations
            .binary_search(&SymmetryOperation::identity())
            .is_err()
        {
            return Err(SymmetryError::MissingIdentity);
        }
        for left in &operations {
            for right in &operations {
                let product = left.compose(*right)?;
                if operations.binary_search(&product).is_err() {
                    return Err(SymmetryError::NotClosed);
                }
            }
        }
        let rotations = operations
            .iter()
            .map(|operation| operation.rotation)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let metric_constraints = derive_metric_constraints(&rotations)?;
        let crystal_system = classify_crystal_system(&rotations, &metric_constraints)?;
        Ok(Self {
            operations,
            rotations,
            metric_constraints,
            crystal_system,
        })
    }

    /// Canonically ordered exact operations.
    #[must_use]
    pub fn operations(&self) -> &[SymmetryOperation] {
        &self.operations
    }

    /// Canonically ordered unique point-group rotations.
    #[must_use]
    pub fn rotations(&self) -> &[[[i32; 3]; 3]] {
        &self.rotations
    }

    /// Inferred crystal system.
    #[must_use]
    pub const fn crystal_system(&self) -> CrystalSystem {
        self.crystal_system
    }

    /// Exact direct-metric constraints derived from every point-group rotation.
    #[must_use]
    pub const fn metric_constraints(&self) -> &MetricConstraints {
        &self.metric_constraints
    }

    /// Expand asymmetric fractional sites, deduplicating special positions.
    ///
    /// Results are sorted first by source site and then lexicographically.
    ///
    /// # Errors
    ///
    /// Returns a validation error for non-finite coordinates or an invalid
    /// periodic deduplication tolerance.
    pub fn expand_sites(
        &self,
        asymmetric_xyz: &[[f64; 3]],
        tolerance: f64,
    ) -> Result<ExpandedSites, SymmetryError> {
        if !tolerance.is_finite() || !(0.0..0.5).contains(&tolerance) || tolerance == 0.0 {
            return Err(SymmetryError::InvalidCoordinateTolerance);
        }
        if asymmetric_xyz
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(SymmetryError::NonFiniteCoordinate);
        }
        let mut positions = Vec::new();
        let mut source_site = Vec::new();
        let mut representative_rotation = Vec::new();
        for (source, xyz) in asymmetric_xyz.iter().copied().enumerate() {
            let mut site_positions = Vec::new();
            for operation in &self.operations {
                let candidate = operation.apply_fractional(xyz);
                if !site_positions
                    .iter()
                    .any(|(existing, _)| periodic_equal(*existing, candidate, tolerance))
                {
                    site_positions.push((candidate, operation.rotation()));
                }
            }
            site_positions.sort_by(|left, right| lexicographic_f64(&left.0, &right.0));
            source_site.extend(std::iter::repeat_n(source, site_positions.len()));
            positions.extend(site_positions.iter().map(|(position, _)| *position));
            representative_rotation.extend(site_positions.iter().map(|(_, rotation)| *rotation));
        }
        Ok(ExpandedSites {
            fractional_xyz: positions,
            source_site,
            representative_rotation,
        })
    }

    /// Test a general-position systematic absence by exact phase cancellation.
    ///
    /// Operations are grouped by their transformed reciprocal index. The sum
    /// of translation phases in every group is reduced exactly modulo the
    /// relevant cyclotomic polynomial; no floating tolerance is used.
    ///
    /// # Errors
    ///
    /// Returns an arithmetic error for unrepresentable indices or phase
    /// denominators beyond the documented exact bound.
    pub fn is_systematically_absent(&self, hkl: [i32; 3]) -> Result<bool, SymmetryError> {
        if hkl == [0, 0, 0] {
            return Ok(false);
        }
        let mut phase_groups: BTreeMap<[i32; 3], Vec<Rational>> = BTreeMap::new();
        for operation in &self.operations {
            phase_groups
                .entry(operation.reciprocal_index(hkl)?)
                .or_default()
                .push(operation.phase(hkl)?);
        }
        for phases in phase_groups.values() {
            if !exact_root_sum_is_zero(phases)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return the exact reciprocal orbit, optionally merging Friedel mates.
    ///
    /// # Errors
    ///
    /// Returns an arithmetic error for an unrepresentable transformed index.
    pub fn reflection_orbit(
        &self,
        hkl: [i32; 3],
        merge_friedel: bool,
    ) -> Result<Vec<[i32; 3]>, SymmetryError> {
        let mut orbit = BTreeSet::new();
        for operation in &self.operations {
            let transformed = operation.reciprocal_index(hkl)?;
            orbit.insert(transformed);
            if merge_friedel {
                orbit.insert(negate_hkl(transformed)?);
            }
        }
        Ok(orbit.into_iter().collect())
    }

    /// Canonical representative and multiplicity for one reciprocal family.
    ///
    /// # Errors
    ///
    /// Returns an arithmetic error for an unrepresentable orbit member.
    pub fn reflection_family(
        &self,
        hkl: [i32; 3],
        merge_friedel: bool,
    ) -> Result<ReflectionFamily, SymmetryError> {
        let orbit = self.reflection_orbit(hkl, merge_friedel)?;
        let canonical_hkl = orbit
            .iter()
            .copied()
            .map(|member| {
                if merge_friedel {
                    canonical_friedel_sign(member)
                } else {
                    member
                }
            })
            .min()
            .ok_or(SymmetryError::EmptyOperationSet)?;
        Ok(ReflectionFamily {
            reflection_id: reflection_id(canonical_hkl),
            canonical_hkl,
            multiplicity: orbit.len(),
            orbit,
        })
    }
}

/// Symmetry-expanded site positions and their asymmetric-site origins.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedSites {
    /// Unique fractional positions, source-site-major.
    pub fractional_xyz: Vec<[f64; 3]>,
    /// Source asymmetric-site index for each expanded position.
    pub source_site: Vec<usize>,
    /// Exact direct-space rotation used for each unique position.
    ///
    /// When multiple operations coincide at a special position, the first
    /// canonically ordered operation is retained. Coordinate derivatives are
    /// defined for fixed orbit topology and site-stabilizer-compatible
    /// directions.
    pub representative_rotation: Vec<[[i32; 3]; 3]>,
}

/// Reciprocal family topology independent of unit-cell dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectionFamily {
    /// Stable ID derived only from the canonical Miller index.
    pub reflection_id: String,
    /// Canonical Miller representative.
    pub canonical_hkl: [i32; 3],
    /// Number of distinct reciprocal indices in the orbit.
    pub multiplicity: usize,
    /// Sorted distinct orbit members.
    pub orbit: Vec<[i32; 3]>,
}

/// Exact symmetry validation or evaluation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetryError {
    /// A rational denominator was zero.
    ZeroDenominator,
    /// Integer/rational arithmetic overflowed.
    ArithmeticOverflow,
    /// A rotation determinant was not `+1` or `-1`.
    NonUnimodularRotation,
    /// No operations were supplied.
    EmptyOperationSet,
    /// The normalized operation set contained a duplicate.
    DuplicateOperation,
    /// The exact identity was missing.
    MissingIdentity,
    /// A product of supplied operations was absent.
    NotClosed,
    /// A coordinate was not finite.
    NonFiniteCoordinate,
    /// The periodic coordinate tolerance was outside `(0, 0.5)`.
    InvalidCoordinateTolerance,
    /// An exact phase common denominator exceeded the supported bound.
    PhaseDenominatorTooLarge,
    /// A finite rotation did not return to identity within 12 applications.
    UnsupportedRotationOrder,
}

impl Display for SymmetryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ZeroDenominator => "symmetry rational denominator must be non-zero",
            Self::ArithmeticOverflow => "symmetry integer arithmetic overflow",
            Self::NonUnimodularRotation => "symmetry rotation determinant must be exactly +1 or -1",
            Self::EmptyOperationSet => "space group must contain at least one operation",
            Self::DuplicateOperation => "space-group operations must be unique modulo translations",
            Self::MissingIdentity => "space-group operations must include the exact identity",
            Self::NotClosed => "space-group operation set is not closed under composition",
            Self::NonFiniteCoordinate => "fractional coordinates must be finite",
            Self::InvalidCoordinateTolerance => {
                "coordinate tolerance must be finite and lie strictly within (0, 0.5)"
            }
            Self::PhaseDenominatorTooLarge => "exact phase denominator exceeds the supported bound",
            Self::UnsupportedRotationOrder => {
                "symmetry rotation order must be finite and no greater than 12"
            }
        })
    }
}

impl Error for SymmetryError {}

fn reflection_id(hkl: [i32; 3]) -> String {
    format!("hkl:{},{},{}", hkl[0], hkl[1], hkl[2])
}

fn canonical_friedel_sign(hkl: [i32; 3]) -> [i32; 3] {
    for value in hkl {
        if value > 0 {
            return hkl;
        }
        if value < 0 {
            return hkl.map(i32::wrapping_neg);
        }
    }
    hkl
}

fn negate_hkl(hkl: [i32; 3]) -> Result<[i32; 3], SymmetryError> {
    Ok([
        hkl[0]
            .checked_neg()
            .ok_or(SymmetryError::ArithmeticOverflow)?,
        hkl[1]
            .checked_neg()
            .ok_or(SymmetryError::ArithmeticOverflow)?,
        hkl[2]
            .checked_neg()
            .ok_or(SymmetryError::ArithmeticOverflow)?,
    ])
}

fn periodic_equal(left: [f64; 3], right: [f64; 3], tolerance: f64) -> bool {
    left.into_iter().zip(right).all(|(a, b)| {
        let distance = (a - b).abs();
        distance.min(1.0 - distance) <= tolerance
    })
}

fn lexicographic_f64(left: &[f64; 3], right: &[f64; 3]) -> std::cmp::Ordering {
    left[0]
        .total_cmp(&right[0])
        .then_with(|| left[1].total_cmp(&right[1]))
        .then_with(|| left[2].total_cmp(&right[2]))
}

fn determinant_i32(matrix: [[i32; 3]; 3]) -> i64 {
    let matrix = matrix.map(|row| row.map(i64::from));
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn multiply_rotation(
    left: [[i32; 3]; 3],
    right: [[i32; 3]; 3],
) -> Result<[[i32; 3]; 3], SymmetryError> {
    let mut product = [[0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let value = (0..3).try_fold(0_i64, |accumulator, inner| {
                accumulator
                    .checked_add(i64::from(left[row][inner]) * i64::from(right[inner][column]))
                    .ok_or(SymmetryError::ArithmeticOverflow)
            })?;
            product[row][column] =
                i32::try_from(value).map_err(|_| SymmetryError::ArithmeticOverflow)?;
        }
    }
    Ok(product)
}

fn gcd_i64(left: i64, right: i64) -> i64 {
    let mut a = left.unsigned_abs();
    let mut b = right.unsigned_abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    i64::try_from(a.max(1)).unwrap_or(i64::MAX)
}

fn lcm_i64(left: i64, right: i64) -> Result<i64, SymmetryError> {
    (left / gcd_i64(left, right))
        .checked_mul(right)
        .ok_or(SymmetryError::ArithmeticOverflow)
}

fn exact_root_sum_is_zero(phases: &[Rational]) -> Result<bool, SymmetryError> {
    let denominator = phases
        .iter()
        .try_fold(1_i64, |common, phase| lcm_i64(common, phase.denominator))?;
    if denominator > MAX_PHASE_DENOMINATOR {
        return Err(SymmetryError::PhaseDenominatorTooLarge);
    }
    let size = usize::try_from(denominator).map_err(|_| SymmetryError::ArithmeticOverflow)?;
    let mut polynomial = vec![0_i64; size];
    for phase in phases {
        let exponent = phase
            .numerator
            .checked_mul(denominator / phase.denominator)
            .ok_or(SymmetryError::ArithmeticOverflow)?
            .rem_euclid(denominator);
        let index = usize::try_from(exponent).map_err(|_| SymmetryError::ArithmeticOverflow)?;
        polynomial[index] = polynomial[index]
            .checked_add(1)
            .ok_or(SymmetryError::ArithmeticOverflow)?;
    }
    let cyclotomic = cyclotomic_polynomial(size)?;
    Ok(polynomial_remainder(polynomial, &cyclotomic)?
        .into_iter()
        .all(|coefficient| coefficient == 0))
}

fn cyclotomic_polynomial(order: usize) -> Result<Vec<i64>, SymmetryError> {
    if order > MAX_PHASE_DENOMINATOR_USIZE {
        return Err(SymmetryError::PhaseDenominatorTooLarge);
    }
    CYCLOTOMIC_POLYNOMIALS[order]
        .get_or_init(|| calculate_cyclotomic_polynomial(order))
        .clone()
}

fn calculate_cyclotomic_polynomial(order: usize) -> Result<Vec<i64>, SymmetryError> {
    let mut polynomial = vec![0_i64; order + 1];
    polynomial[0] = -1;
    polynomial[order] = 1;
    for divisor in 1..order {
        if order % divisor == 0 {
            polynomial = polynomial_exact_quotient(polynomial, &cyclotomic_polynomial(divisor)?)?;
        }
    }
    trim_polynomial(&mut polynomial);
    Ok(polynomial)
}

fn polynomial_exact_quotient(
    mut dividend: Vec<i64>,
    divisor: &[i64],
) -> Result<Vec<i64>, SymmetryError> {
    let divisor_degree = divisor.len() - 1;
    if divisor[divisor_degree] != 1 || dividend.len() < divisor.len() {
        return Err(SymmetryError::ArithmeticOverflow);
    }
    let mut quotient = vec![0_i64; dividend.len() - divisor_degree];
    for degree in (divisor_degree..dividend.len()).rev() {
        let coefficient = dividend[degree];
        if coefficient == 0 {
            continue;
        }
        let offset = degree - divisor_degree;
        quotient[offset] = coefficient;
        for (index, divisor_coefficient) in divisor.iter().copied().enumerate() {
            let product = coefficient
                .checked_mul(divisor_coefficient)
                .ok_or(SymmetryError::ArithmeticOverflow)?;
            dividend[offset + index] = dividend[offset + index]
                .checked_sub(product)
                .ok_or(SymmetryError::ArithmeticOverflow)?;
        }
    }
    if dividend[..divisor_degree]
        .iter()
        .any(|coefficient| *coefficient != 0)
    {
        return Err(SymmetryError::ArithmeticOverflow);
    }
    trim_polynomial(&mut quotient);
    Ok(quotient)
}

fn polynomial_remainder(
    mut dividend: Vec<i64>,
    divisor: &[i64],
) -> Result<Vec<i64>, SymmetryError> {
    let divisor_degree = divisor.len() - 1;
    for degree in (divisor_degree..dividend.len()).rev() {
        let coefficient = dividend[degree];
        if coefficient == 0 {
            continue;
        }
        let offset = degree - divisor_degree;
        for (index, divisor_coefficient) in divisor.iter().copied().enumerate() {
            dividend[offset + index] = dividend[offset + index]
                .checked_sub(
                    coefficient
                        .checked_mul(divisor_coefficient)
                        .ok_or(SymmetryError::ArithmeticOverflow)?,
                )
                .ok_or(SymmetryError::ArithmeticOverflow)?;
        }
    }
    dividend.truncate(divisor_degree);
    trim_polynomial(&mut dividend);
    Ok(dividend)
}

fn trim_polynomial(polynomial: &mut Vec<i64>) {
    while polynomial.len() > 1 && polynomial.last() == Some(&0) {
        polynomial.pop();
    }
}

fn derive_metric_constraints(
    rotations: &[[[i32; 3]; 3]],
) -> Result<MetricConstraints, SymmetryError> {
    let components = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let mut equations = BTreeSet::new();
    for rotation in rotations {
        for &(output_row, output_column) in &components {
            let mut equation = [0_i64; 6];
            for (variable, &(metric_row, metric_column)) in components.iter().enumerate() {
                let transformed = if metric_row == metric_column {
                    i128::from(rotation[metric_row][output_row])
                        * i128::from(rotation[metric_row][output_column])
                } else {
                    i128::from(rotation[metric_row][output_row])
                        * i128::from(rotation[metric_column][output_column])
                        + i128::from(rotation[metric_column][output_row])
                            * i128::from(rotation[metric_row][output_column])
                };
                let original = i128::from(
                    (metric_row == output_row && metric_column == output_column)
                        || (metric_row == output_column && metric_column == output_row),
                );
                equation[variable] = i64::try_from(transformed - original)
                    .map_err(|_| SymmetryError::ArithmeticOverflow)?;
            }
            normalize_integer_equation(&mut equation);
            if equation != [0; 6] {
                equations.insert(equation);
            }
        }
    }
    let equations = equations.into_iter().collect::<Vec<_>>();
    let parameterization_basis = integer_nullspace(&equations)?;
    let independent_parameter_count = parameterization_basis.len();
    Ok(MetricConstraints {
        equations,
        independent_parameter_count,
        parameterization_basis,
    })
}

fn normalize_integer_equation(equation: &mut [i64; 6]) {
    let divisor = equation.iter().copied().fold(0_i64, gcd_i64_allow_zero);
    if divisor > 1 {
        for value in equation.iter_mut() {
            *value /= divisor;
        }
    }
    if let Some(first) = equation.iter().find(|value| **value != 0) {
        if *first < 0 {
            for value in equation.iter_mut() {
                *value = -*value;
            }
        }
    }
}

fn gcd_i64_allow_zero(left: i64, right: i64) -> i64 {
    if left == 0 {
        return right.abs();
    }
    if right == 0 {
        return left.abs();
    }
    gcd_i64(left, right)
}

#[derive(Clone, Copy)]
struct Fraction {
    numerator: i128,
    denominator: i128,
}

impl Fraction {
    fn new(numerator: i128, denominator: i128) -> Self {
        let sign = if denominator < 0 { -1 } else { 1 };
        let numerator = numerator * sign;
        let denominator = denominator * sign;
        let divisor = gcd_i128(numerator, denominator);
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }

    const fn is_zero(self) -> bool {
        self.numerator == 0
    }

    fn divide(self, right: Self) -> Self {
        Self::new(
            self.numerator * right.denominator,
            self.denominator * right.numerator,
        )
    }

    fn subtract(self, right: Self) -> Self {
        Self::new(
            self.numerator * right.denominator - right.numerator * self.denominator,
            self.denominator * right.denominator,
        )
    }

    fn multiply(self, right: Self) -> Self {
        Self::new(
            self.numerator * right.numerator,
            self.denominator * right.denominator,
        )
    }
}

fn gcd_i128(left: i128, right: i128) -> i128 {
    let mut a = left.abs();
    let mut b = right.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn rational_rref(equations: &[[i64; 6]]) -> (Vec<[Fraction; 6]>, Vec<usize>) {
    let mut matrix = equations
        .iter()
        .map(|row| row.map(|value| Fraction::new(i128::from(value), 1)))
        .collect::<Vec<_>>();
    let mut pivot_row = 0;
    let mut pivot_columns = Vec::new();
    for column in 0..6 {
        let Some(relative) = matrix[pivot_row..]
            .iter()
            .position(|row| !row[column].is_zero())
        else {
            continue;
        };
        matrix.swap(pivot_row, pivot_row + relative);
        let pivot = matrix[pivot_row][column];
        for value in &mut matrix[pivot_row][column..] {
            *value = value.divide(pivot);
        }
        for row in 0..matrix.len() {
            if row == pivot_row || matrix[row][column].is_zero() {
                continue;
            }
            let factor = matrix[row][column];
            let pivot_values = matrix[pivot_row];
            for (inner, value) in matrix[row].iter_mut().enumerate().skip(column) {
                *value = value.subtract(factor.multiply(pivot_values[inner]));
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
        if pivot_row == matrix.len() {
            break;
        }
    }
    (matrix, pivot_columns)
}

fn integer_nullspace(equations: &[[i64; 6]]) -> Result<Vec<[i64; 6]>, SymmetryError> {
    let (matrix, pivot_columns) = rational_rref(equations);
    let free_columns = (0..6)
        .filter(|column| !pivot_columns.contains(column))
        .collect::<Vec<_>>();
    let mut basis = Vec::with_capacity(free_columns.len());
    for free_column in free_columns {
        let mut vector = [Fraction::new(0, 1); 6];
        vector[free_column] = Fraction::new(1, 1);
        for (row, pivot_column) in pivot_columns.iter().copied().enumerate() {
            vector[pivot_column] = Fraction::new(
                -matrix[row][free_column].numerator,
                matrix[row][free_column].denominator,
            );
        }
        let common_denominator = vector.iter().try_fold(1_i128, |common, value| {
            lcm_i128(common, value.denominator).ok_or(SymmetryError::ArithmeticOverflow)
        })?;
        let mut integer_vector = [0_i64; 6];
        for (index, value) in vector.into_iter().enumerate() {
            let scaled = value
                .numerator
                .checked_mul(common_denominator / value.denominator)
                .ok_or(SymmetryError::ArithmeticOverflow)?;
            integer_vector[index] =
                i64::try_from(scaled).map_err(|_| SymmetryError::ArithmeticOverflow)?;
        }
        normalize_integer_equation(&mut integer_vector);
        if integer_vector[free_column] < 0 {
            for value in &mut integer_vector {
                *value = value
                    .checked_neg()
                    .ok_or(SymmetryError::ArithmeticOverflow)?;
            }
        }
        basis.push(integer_vector);
    }
    Ok(basis)
}

fn lcm_i128(left: i128, right: i128) -> Option<i128> {
    (left / gcd_i128(left, right)).checked_mul(right)
}

fn classify_crystal_system(
    rotations: &[[[i32; 3]; 3]],
    constraints: &MetricConstraints,
) -> Result<CrystalSystem, SymmetryError> {
    let mut orders = Vec::with_capacity(rotations.len());
    let mut proper_count = 0;
    for rotation in rotations {
        orders.push(rotation_order(*rotation)?);
        if determinant_i32(*rotation) == 1 {
            proper_count += 1;
        }
    }
    let has_order = |order| orders.contains(&order);
    Ok(if has_order(3) && proper_count >= 12 {
        CrystalSystem::Cubic
    } else if has_order(6) {
        CrystalSystem::Hexagonal
    } else if has_order(4) {
        CrystalSystem::Tetragonal
    } else if has_order(3) {
        CrystalSystem::Trigonal
    } else {
        match constraints.independent_parameter_count {
            0..=2 => CrystalSystem::Cubic,
            3 => CrystalSystem::Orthorhombic,
            4 | 5 => CrystalSystem::Monoclinic,
            _ => CrystalSystem::Triclinic,
        }
    })
}

fn rotation_order(rotation: [[i32; 3]; 3]) -> Result<usize, SymmetryError> {
    let mut product = SymmetryOperation::identity().rotation;
    for order in 1..=12 {
        product = multiply_rotation(product, rotation)?;
        if product == SymmetryOperation::identity().rotation {
            return Ok(order);
        }
    }
    Err(SymmetryError::UnsupportedRotationOrder)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half() -> Rational {
        Rational::new(1, 2).expect("one half")
    }

    fn operation(rotation: [[i32; 3]; 3], translation: [Rational; 3]) -> SymmetryOperation {
        SymmetryOperation::new(rotation, translation).expect("valid operation")
    }

    fn inversion_group() -> SpaceGroup {
        SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            operation([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3]),
        ])
        .expect("P -1")
    }

    #[test]
    fn rational_translation_and_composition_are_exact() {
        assert_eq!(Rational::new(-3, -6).expect("rational"), half());
        let screw = operation(
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [Rational::zero(), half(), Rational::zero()],
        );
        assert_eq!(
            screw.compose(screw).expect("screw squared"),
            SymmetryOperation::identity()
        );
        assert_eq!(
            screw.reciprocal_index([1, 2, 3]).expect("index"),
            [-1, 2, -3]
        );
    }

    #[test]
    fn operation_sets_require_identity_uniqueness_and_closure() {
        assert_eq!(
            SpaceGroup::new(vec![]),
            Err(SymmetryError::EmptyOperationSet)
        );
        assert_eq!(
            SpaceGroup::new(vec![SymmetryOperation::identity(); 2]),
            Err(SymmetryError::DuplicateOperation)
        );
        let inversion = operation([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3]);
        assert_eq!(
            SpaceGroup::new(vec![inversion]),
            Err(SymmetryError::MissingIdentity)
        );
        let quarter = Rational::new(1, 4).expect("quarter");
        let incomplete = operation(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [quarter, Rational::zero(), Rational::zero()],
        );
        assert_eq!(
            SpaceGroup::new(vec![SymmetryOperation::identity(), incomplete]),
            Err(SymmetryError::NotClosed)
        );
    }

    #[test]
    fn special_positions_are_deduplicated_periodically() {
        let group = inversion_group();
        let expanded = group
            .expand_sites(&[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], 1e-10)
            .expect("expanded sites");
        assert_eq!(expanded.source_site, vec![0, 1, 1]);
        assert_eq!(expanded.representative_rotation.len(), 3);
        assert_eq!(
            expanded.representative_rotation[0],
            group.operations()[0].rotation()
        );
        assert!(periodic_equal(
            expanded.fractional_xyz[0],
            [0.0, 0.0, 0.0],
            1e-15
        ));
        assert!(expanded.fractional_xyz.contains(&[0.1, 0.2, 0.3]));
        assert!(expanded.fractional_xyz.contains(&[0.9, 0.8, 0.7]));
        for (position, rotation) in expanded.fractional_xyz[1..]
            .iter()
            .zip(&expanded.representative_rotation[1..])
        {
            let operation = group
                .operations()
                .iter()
                .find(|operation| operation.rotation() == *rotation)
                .expect("representative operation");
            assert!(periodic_equal(
                *position,
                operation.apply_fractional([0.1, 0.2, 0.3]),
                1e-15
            ));
        }
    }

    #[test]
    fn centring_and_screw_absences_are_exact() {
        let body_centred = SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            operation(
                SymmetryOperation::identity().rotation,
                [half(), half(), half()],
            ),
        ])
        .expect("I lattice");
        assert!(
            body_centred
                .is_systematically_absent([1, 0, 0])
                .expect("absence")
        );
        assert!(
            !body_centred
                .is_systematically_absent([1, 1, 0])
                .expect("allowed")
        );

        let screw = operation(
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [Rational::zero(), half(), Rational::zero()],
        );
        let p21 = SpaceGroup::new(vec![SymmetryOperation::identity(), screw]).expect("P21");
        assert!(p21.is_systematically_absent([0, 1, 0]).expect("odd 0k0"));
        assert!(!p21.is_systematically_absent([0, 2, 0]).expect("even 0k0"));
        assert!(
            !p21.is_systematically_absent([1, 1, 0])
                .expect("general reflection")
        );
    }

    #[test]
    fn reflection_orbits_are_deterministic_and_metric_constraints_are_exact() {
        let group = inversion_group();
        let family = group.reflection_family([-1, 2, 3], true).expect("family");
        assert_eq!(family.canonical_hkl, [1, -2, -3]);
        assert_eq!(family.multiplicity, 2);
        assert_eq!(family.reflection_id, "hkl:1,-2,-3");
        assert_eq!(group.crystal_system(), CrystalSystem::Triclinic);
        assert_eq!(group.metric_constraints().independent_parameter_count, 6);

        let twofold = operation([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [Rational::zero(); 3]);
        let monoclinic =
            SpaceGroup::new(vec![SymmetryOperation::identity(), twofold]).expect("two-fold group");
        assert_eq!(monoclinic.crystal_system(), CrystalSystem::Monoclinic);
        assert_eq!(
            monoclinic.metric_constraints().independent_parameter_count,
            4
        );
        let constraints = monoclinic.metric_constraints();
        assert_eq!(constraints.parameterization_basis.len(), 4);
        for equation in &constraints.equations {
            for basis in &constraints.parameterization_basis {
                assert_eq!(
                    equation
                        .iter()
                        .zip(basis)
                        .map(|(left, right)| left * right)
                        .sum::<i64>(),
                    0
                );
            }
        }
    }
}
