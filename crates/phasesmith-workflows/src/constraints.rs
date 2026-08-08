//! Typed fixed, affine, and multi-source linear constraint transforms.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{ParameterKey, ParameterSet};

/// Set one target parameter to a constant during expansion.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedConstraint {
    /// Constrained parameter.
    target: ParameterKey,
    /// Constant physical value.
    value: f64,
}

impl FixedConstraint {
    /// Construct a finite fixed constraint.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::NonFiniteCoefficient`] for a non-finite value.
    pub fn new(target: ParameterKey, value: f64) -> Result<Self, ConstraintError> {
        if !value.is_finite() {
            return Err(ConstraintError::NonFiniteCoefficient);
        }
        Ok(Self { target, value })
    }

    /// Borrow the constrained parameter.
    #[must_use]
    pub const fn target(&self) -> &ParameterKey {
        &self.target
    }

    /// Return the constant physical value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }
}

/// Define `target = multiplier * source + offset`.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineConstraint {
    /// Constrained parameter.
    target: ParameterKey,
    /// Already-resolved source parameter.
    source: ParameterKey,
    /// Source multiplier.
    multiplier: f64,
    /// Physical offset.
    offset: f64,
}

impl AffineConstraint {
    /// Validate distinct keys and finite coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError`] for a self-reference or non-finite value.
    pub fn new(
        target: ParameterKey,
        source: ParameterKey,
        multiplier: f64,
        offset: f64,
    ) -> Result<Self, ConstraintError> {
        if target == source {
            return Err(ConstraintError::TargetIsSource { target });
        }
        if !multiplier.is_finite() || !offset.is_finite() {
            return Err(ConstraintError::NonFiniteCoefficient);
        }
        Ok(Self {
            target,
            source,
            multiplier,
            offset,
        })
    }

    /// Borrow the constrained parameter.
    #[must_use]
    pub const fn target(&self) -> &ParameterKey {
        &self.target
    }

    /// Borrow the already-resolved source.
    #[must_use]
    pub const fn source(&self) -> &ParameterKey {
        &self.source
    }

    /// Return the source multiplier.
    #[must_use]
    pub const fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Return the physical offset.
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }
}

/// One source/coefficient pair in a linear constraint.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearTerm {
    /// Already-resolved source parameter.
    source: ParameterKey,
    /// Source coefficient.
    coefficient: f64,
}

impl LinearTerm {
    /// Construct one finite source term.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::NonFiniteCoefficient`] for a non-finite coefficient.
    pub fn new(source: ParameterKey, coefficient: f64) -> Result<Self, ConstraintError> {
        if !coefficient.is_finite() {
            return Err(ConstraintError::NonFiniteCoefficient);
        }
        Ok(Self {
            source,
            coefficient,
        })
    }

    /// Borrow the source parameter.
    #[must_use]
    pub const fn source(&self) -> &ParameterKey {
        &self.source
    }

    /// Return the finite source coefficient.
    #[must_use]
    pub const fn coefficient(&self) -> f64 {
        self.coefficient
    }
}

/// Define `target = offset + sum(coefficient * source)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearConstraint {
    /// Constrained parameter.
    target: ParameterKey,
    /// Ordered, unique source terms.
    terms: Vec<LinearTerm>,
    /// Physical offset.
    offset: f64,
}

impl LinearConstraint {
    /// Validate a non-empty list of unique, non-target sources.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError`] for empty/duplicate/self sources or a
    /// non-finite offset.
    pub fn new(
        target: ParameterKey,
        terms: Vec<LinearTerm>,
        offset: f64,
    ) -> Result<Self, ConstraintError> {
        if terms.is_empty() {
            return Err(ConstraintError::EmptyLinearTerms);
        }
        if !offset.is_finite() {
            return Err(ConstraintError::NonFiniteCoefficient);
        }
        let mut sources = BTreeSet::new();
        for term in &terms {
            if term.source == target {
                return Err(ConstraintError::TargetIsSource {
                    target: target.clone(),
                });
            }
            if !sources.insert(term.source.clone()) {
                return Err(ConstraintError::DuplicateLinearSource {
                    source: term.source.clone(),
                });
            }
        }
        Ok(Self {
            target,
            terms,
            offset,
        })
    }

    /// Borrow the constrained parameter.
    #[must_use]
    pub const fn target(&self) -> &ParameterKey {
        &self.target
    }

    /// Borrow the ordered unique source terms.
    #[must_use]
    pub fn terms(&self) -> &[LinearTerm] {
        &self.terms
    }

    /// Return the physical offset.
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }
}

/// Supported native scalar constraint records.
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// Constant target.
    Fixed(FixedConstraint),
    /// One-source affine target.
    Affine(AffineConstraint),
    /// Multi-source linear target.
    Linear(LinearConstraint),
}

impl Constraint {
    /// Borrow the constrained target shared by every record variant.
    #[must_use]
    pub fn target(&self) -> &ParameterKey {
        match self {
            Self::Fixed(value) => &value.target,
            Self::Affine(value) => &value.target,
            Self::Linear(value) => &value.target,
        }
    }

    fn sources(&self) -> impl Iterator<Item = &ParameterKey> {
        let sources: Vec<&ParameterKey> = match self {
            Self::Fixed(_) => Vec::new(),
            Self::Affine(value) => vec![&value.source],
            Self::Linear(value) => value.terms.iter().map(|term| &term.source).collect(),
        };
        sources.into_iter()
    }
}

/// Row-major derivative `d physical values / d scaled free values`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintDerivativeMatrix {
    /// Parameter row count.
    pub rows: usize,
    /// Free-parameter column count.
    pub columns: usize,
    /// Row-major matrix elements.
    pub values: Vec<f64>,
}

impl ConstraintDerivativeMatrix {
    /// Borrow one parameter row.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.columns)?;
        self.values.get(start..start.checked_add(self.columns)?)
    }
}

/// Validated ordered mapping between free solver coordinates and all values.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintTransform {
    parameters: ParameterSet,
    constraints: Vec<Constraint>,
    free_keys: Vec<ParameterKey>,
}

impl ConstraintTransform {
    /// Validate target ownership, uniqueness, source ownership, and dependency order.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError`] for unknown keys, duplicate targets, or a
    /// source that is not already resolved (including cycles).
    pub fn new(
        parameters: ParameterSet,
        constraints: Vec<Constraint>,
    ) -> Result<Self, ConstraintError> {
        let known = parameters.key_set();
        let mut targets = BTreeSet::new();
        for constraint in &constraints {
            if !known.contains(constraint.target()) {
                return Err(ConstraintError::UnknownTarget {
                    target: constraint.target().clone(),
                });
            }
            if !targets.insert(constraint.target().clone()) {
                return Err(ConstraintError::DuplicateTarget {
                    target: constraint.target().clone(),
                });
            }
        }
        let mut resolved = known.difference(&targets).cloned().collect::<BTreeSet<_>>();
        for constraint in &constraints {
            for source in constraint.sources() {
                if !known.contains(source) {
                    return Err(ConstraintError::UnknownSource {
                        source: source.clone(),
                    });
                }
                if !resolved.contains(source) {
                    return Err(ConstraintError::UnresolvedDependency {
                        target: Box::new(constraint.target().clone()),
                        source: Box::new(source.clone()),
                    });
                }
            }
            resolved.insert(constraint.target().clone());
        }
        let free_keys = parameters
            .specs()
            .iter()
            .filter(|spec| spec.refine() && !targets.contains(spec.key()))
            .map(|spec| spec.key().clone())
            .collect();
        Ok(Self {
            parameters,
            constraints,
            free_keys,
        })
    }

    /// Borrow the validated parameter set.
    #[must_use]
    pub const fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }

    /// Borrow constraints in required dependency order.
    #[must_use]
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Borrow free identities in stable packing order.
    #[must_use]
    pub fn free_keys(&self) -> &[ParameterKey] {
        &self.free_keys
    }

    /// Pack current physical values into scaled free solver coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError`] if a stored value unexpectedly becomes
    /// non-finite.
    pub fn pack(&self) -> Result<Vec<f64>, ConstraintError> {
        self.pack_values(&self.parameters.values())
    }

    /// Pack caller-supplied physical values into scaled free coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError`] for a missing or non-finite free value.
    pub fn pack_values(
        &self,
        values: &BTreeMap<ParameterKey, f64>,
    ) -> Result<Vec<f64>, ConstraintError> {
        self.free_keys
            .iter()
            .map(|key| {
                let value = values
                    .get(key)
                    .copied()
                    .ok_or_else(|| ConstraintError::MissingValue { key: key.clone() })?;
                let spec =
                    self.parameters
                        .spec(key)
                        .ok_or_else(|| ConstraintError::UnknownSource {
                            source: key.clone(),
                        })?;
                let scaled = value / spec.scale();
                if !scaled.is_finite() {
                    return Err(ConstraintError::NonFiniteVector);
                }
                Ok(scaled)
            })
            .collect()
    }

    /// Expand scaled free coordinates into all bounded physical values.
    ///
    /// When `clip` is true, free physical values are projected to their bounds
    /// before constraints are evaluated. Constraint results are never clipped.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError`] for shape, finiteness, or final-bound failures.
    pub fn unpack(
        &self,
        vector: &[f64],
        clip: bool,
    ) -> Result<BTreeMap<ParameterKey, f64>, ConstraintError> {
        if vector.len() != self.free_keys.len() {
            return Err(ConstraintError::VectorLengthMismatch {
                expected: self.free_keys.len(),
                actual: vector.len(),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ConstraintError::NonFiniteVector);
        }
        let mut values = self.parameters.values();
        for (key, scaled) in self.free_keys.iter().zip(vector) {
            let spec = self
                .parameters
                .spec(key)
                .ok_or_else(|| ConstraintError::UnknownSource {
                    source: key.clone(),
                })?;
            let physical = scaled * spec.scale();
            values.insert(
                key.clone(),
                if clip {
                    spec.bounds().clip(physical)
                } else {
                    physical
                },
            );
        }
        for constraint in &self.constraints {
            let value = match constraint {
                Constraint::Fixed(value) => value.value,
                Constraint::Affine(value) => {
                    value.multiplier
                        * values.get(&value.source).copied().ok_or_else(|| {
                            ConstraintError::MissingValue {
                                key: value.source.clone(),
                            }
                        })?
                        + value.offset
                }
                Constraint::Linear(value) => {
                    let mut result = value.offset;
                    for term in &value.terms {
                        result += term.coefficient
                            * values.get(&term.source).copied().ok_or_else(|| {
                                ConstraintError::MissingValue {
                                    key: term.source.clone(),
                                }
                            })?;
                    }
                    result
                }
            };
            values.insert(constraint.target().clone(), value);
        }
        for spec in self.parameters.specs() {
            let value =
                values
                    .get(spec.key())
                    .copied()
                    .ok_or_else(|| ConstraintError::MissingValue {
                        key: spec.key().clone(),
                    })?;
            if !value.is_finite() || !spec.bounds().contains(value) {
                return Err(ConstraintError::ExpandedValueOutsideBounds {
                    key: spec.key().clone(),
                    value,
                });
            }
        }
        Ok(values)
    }

    /// Build the exact row-major physical-to-scaled-free derivative matrix.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::MatrixSizeOverflow`] if the matrix element
    /// count cannot be represented, or [`ConstraintError::InternalInvariant`]
    /// if validated transform state is unexpectedly inconsistent.
    pub fn derivative_matrix(&self) -> Result<ConstraintDerivativeMatrix, ConstraintError> {
        let rows = self.parameters.specs().len();
        let columns = self.free_keys.len();
        let element_count = rows
            .checked_mul(columns)
            .ok_or(ConstraintError::MatrixSizeOverflow)?;
        let mut values = vec![0.0; element_count];
        for (column, key) in self.free_keys.iter().enumerate() {
            let row = self
                .parameters
                .index_of(key)
                .ok_or(ConstraintError::InternalInvariant)?;
            let spec = self
                .parameters
                .spec(key)
                .ok_or(ConstraintError::InternalInvariant)?;
            *values
                .get_mut(row * columns + column)
                .ok_or(ConstraintError::InternalInvariant)? = spec.scale();
        }
        for constraint in &self.constraints {
            let target_row = self
                .parameters
                .index_of(constraint.target())
                .ok_or(ConstraintError::InternalInvariant)?;
            match constraint {
                Constraint::Fixed(_) => {}
                Constraint::Affine(constraint) => {
                    let source_row = self
                        .parameters
                        .index_of(&constraint.source)
                        .ok_or(ConstraintError::InternalInvariant)?;
                    for column in 0..columns {
                        let source_value = values
                            .get(source_row * columns + column)
                            .copied()
                            .ok_or(ConstraintError::InternalInvariant)?;
                        *values
                            .get_mut(target_row * columns + column)
                            .ok_or(ConstraintError::InternalInvariant)? =
                            constraint.multiplier * source_value;
                    }
                }
                Constraint::Linear(constraint) => {
                    for column in 0..columns {
                        let mut target_value = 0.0;
                        for term in &constraint.terms {
                            let source_row = self
                                .parameters
                                .index_of(&term.source)
                                .ok_or(ConstraintError::InternalInvariant)?;
                            target_value += term.coefficient
                                * values
                                    .get(source_row * columns + column)
                                    .copied()
                                    .ok_or(ConstraintError::InternalInvariant)?;
                        }
                        *values
                            .get_mut(target_row * columns + column)
                            .ok_or(ConstraintError::InternalInvariant)? = target_value;
                    }
                }
            }
        }
        Ok(ConstraintDerivativeMatrix {
            rows,
            columns,
            values,
        })
    }
}

/// Invalid native scalar constraint or transform input.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintError {
    /// A coefficient, offset, or fixed value is non-finite.
    NonFiniteCoefficient,
    /// A target is also used directly as its source.
    TargetIsSource {
        /// Invalid target.
        target: ParameterKey,
    },
    /// A linear constraint has no sources.
    EmptyLinearTerms,
    /// A linear constraint repeats one source.
    DuplicateLinearSource {
        /// Repeated source.
        source: ParameterKey,
    },
    /// A target does not belong to the parameter set.
    UnknownTarget {
        /// Missing target.
        target: ParameterKey,
    },
    /// More than one constraint owns a target.
    DuplicateTarget {
        /// Repeated target.
        target: ParameterKey,
    },
    /// A source does not belong to the parameter set.
    UnknownSource {
        /// Missing source.
        source: ParameterKey,
    },
    /// A source is constrained later or belongs to a dependency cycle.
    UnresolvedDependency {
        /// Target currently being resolved.
        target: Box<ParameterKey>,
        /// Unavailable source.
        source: Box<ParameterKey>,
    },
    /// A caller-supplied value map omits a free key.
    MissingValue {
        /// Missing key.
        key: ParameterKey,
    },
    /// A solver vector has the wrong length.
    VectorLengthMismatch {
        /// Required length.
        expected: usize,
        /// Received length.
        actual: usize,
    },
    /// A packed or unpacked solver value is non-finite.
    NonFiniteVector,
    /// Derivative matrix dimensions overflow the platform element count.
    MatrixSizeOverflow,
    /// Private validated transform state is unexpectedly inconsistent.
    InternalInvariant,
    /// One expanded physical value violates its declared bounds.
    ExpandedValueOutsideBounds {
        /// Invalid parameter.
        key: ParameterKey,
        /// Expanded value.
        value: f64,
    },
}

impl Display for ConstraintError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteCoefficient => {
                formatter.write_str("constraint values and coefficients must be finite")
            }
            Self::TargetIsSource { target } => {
                write!(
                    formatter,
                    "constraint target {target} cannot be its own source"
                )
            }
            Self::EmptyLinearTerms => {
                formatter.write_str("linear constraints require at least one source")
            }
            Self::DuplicateLinearSource { source } => {
                write!(formatter, "linear constraint repeats source {source}")
            }
            Self::UnknownTarget { target } => {
                write!(formatter, "constraint target {target} is not a parameter")
            }
            Self::DuplicateTarget { target } => {
                write!(
                    formatter,
                    "parameter {target} is constrained more than once"
                )
            }
            Self::UnknownSource { source } => {
                write!(formatter, "constraint source {source} is not a parameter")
            }
            Self::UnresolvedDependency { target, source } => write!(
                formatter,
                "constraint for {target} depends on unresolved source {source}"
            ),
            Self::MissingValue { key } => {
                write!(formatter, "missing value for free parameter {key}")
            }
            Self::VectorLengthMismatch { expected, actual } => write!(
                formatter,
                "free vector length {actual} does not match expected length {expected}"
            ),
            Self::NonFiniteVector => formatter.write_str("free parameter values must be finite"),
            Self::MatrixSizeOverflow => {
                formatter.write_str("constraint derivative matrix size overflow")
            }
            Self::InternalInvariant => {
                formatter.write_str("validated constraint transform state is inconsistent")
            }
            Self::ExpandedValueOutsideBounds { key, value } => write!(
                formatter,
                "expanded value {value} for {key} lies outside its bounds"
            ),
        }
    }
}

impl Error for ConstraintError {}
