//! Stable scalar parameter identities, bounds, and ordered sets.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Stable structured identity for one refinable scalar.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterKey {
    module: String,
    owner_id: String,
    name: String,
}

impl ParameterKey {
    /// Validate and own a three-segment parameter identity.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError::InvalidKeySegment`] for empty, untrimmed, or
    /// delimiter-ambiguous segments.
    pub fn new(
        module: impl Into<String>,
        owner_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Self, ParameterError> {
        let key = Self {
            module: module.into(),
            owner_id: owner_id.into(),
            name: name.into(),
        };
        validate_key_segment("module", &key.module)?;
        validate_key_segment("owner_id", &key.owner_id)?;
        validate_key_segment("name", &key.name)?;
        Ok(key)
    }

    /// Return the owning module segment.
    #[must_use]
    pub fn module(&self) -> &str {
        &self.module
    }

    /// Return the stable owner segment.
    #[must_use]
    pub fn owner_id(&self) -> &str {
        &self.owner_id
    }

    /// Return the local parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return `module[owner_id].name` for diagnostics and reports.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{}[{}].{}", self.module, self.owner_id, self.name)
    }
}

impl Display for ParameterKey {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label())
    }
}

/// Closed lower and upper scalar bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterBounds {
    /// Inclusive lower bound; negative infinity is allowed.
    lower: f64,
    /// Inclusive upper bound; positive infinity is allowed.
    upper: f64,
}

impl ParameterBounds {
    /// Validate ordered, non-NaN bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError::InvalidBounds`] for NaN or reversed bounds.
    pub fn new(lower: f64, upper: f64) -> Result<Self, ParameterError> {
        if lower.is_nan() || upper.is_nan() || lower > upper {
            return Err(ParameterError::InvalidBounds);
        }
        Ok(Self { lower, upper })
    }

    /// Return whether `value` is inside the closed interval.
    #[must_use]
    pub fn contains(self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// Project one scalar into the closed interval.
    #[must_use]
    pub fn clip(self, value: f64) -> f64 {
        value.clamp(self.lower, self.upper)
    }

    /// Return the inclusive lower bound.
    #[must_use]
    pub const fn lower(self) -> f64 {
        self.lower
    }

    /// Return the inclusive upper bound.
    #[must_use]
    pub const fn upper(self) -> f64 {
        self.upper
    }
}

impl Default for ParameterBounds {
    fn default() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }
}

/// Value, unit, scale, bounds, and selection for one scalar.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterSpec {
    /// Stable structured identity.
    key: ParameterKey,
    /// Current physical value.
    value: f64,
    /// Explicit physical unit label.
    unit: String,
    /// Closed physical bounds.
    bounds: ParameterBounds,
    /// Positive scale mapping physical values to solver coordinates.
    scale: f64,
    /// Whether this unconstrained parameter is selected for refinement.
    refine: bool,
}

impl ParameterSpec {
    /// Validate and construct one scalar specification.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] for non-finite values, empty units,
    /// out-of-bounds values, or non-positive/non-finite scales.
    pub fn new(
        key: ParameterKey,
        value: f64,
        unit: impl Into<String>,
        bounds: ParameterBounds,
        scale: f64,
        refine: bool,
    ) -> Result<Self, ParameterError> {
        let unit = unit.into();
        if !value.is_finite() {
            return Err(ParameterError::NonFiniteValue { key });
        }
        if unit.is_empty() {
            return Err(ParameterError::InvalidUnit { key });
        }
        if !bounds.contains(value) {
            return Err(ParameterError::ValueOutsideBounds { key, value });
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(ParameterError::InvalidScale { key });
        }
        Ok(Self {
            key,
            value,
            unit,
            bounds,
            scale,
            refine,
        })
    }

    /// Borrow the stable parameter identity.
    #[must_use]
    pub const fn key(&self) -> &ParameterKey {
        &self.key
    }

    /// Return the current physical value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Borrow the explicit unit label.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Return the physical bounds.
    #[must_use]
    pub const fn bounds(&self) -> ParameterBounds {
        self.bounds
    }

    /// Return the positive solver scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Return whether this unconstrained parameter is selected.
    #[must_use]
    pub const fn refine(&self) -> bool {
        self.refine
    }
}

/// Deterministically ordered immutable scalar specifications.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterSet {
    specs: Vec<ParameterSpec>,
    index_by_key: BTreeMap<ParameterKey, usize>,
}

impl ParameterSet {
    /// Preserve input order while validating unique stable keys.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError::DuplicateKey`] for repeated identities.
    pub fn new(specs: Vec<ParameterSpec>) -> Result<Self, ParameterError> {
        let mut index_by_key = BTreeMap::new();
        for (index, spec) in specs.iter().enumerate() {
            if index_by_key.insert(spec.key.clone(), index).is_some() {
                return Err(ParameterError::DuplicateKey {
                    key: spec.key.clone(),
                });
            }
        }
        Ok(Self {
            specs,
            index_by_key,
        })
    }

    /// Borrow specifications in stable packing order.
    #[must_use]
    pub fn specs(&self) -> &[ParameterSpec] {
        &self.specs
    }

    /// Return one specification by stable key.
    #[must_use]
    pub fn spec(&self, key: &ParameterKey) -> Option<&ParameterSpec> {
        self.index_by_key.get(key).map(|index| &self.specs[*index])
    }

    /// Return the stable row index of one parameter.
    #[must_use]
    pub fn index_of(&self, key: &ParameterKey) -> Option<usize> {
        self.index_by_key.get(key).copied()
    }

    /// Copy physical values into a key-addressed map.
    #[must_use]
    pub fn values(&self) -> BTreeMap<ParameterKey, f64> {
        self.specs
            .iter()
            .map(|spec| (spec.key.clone(), spec.value))
            .collect()
    }

    /// Return all stable keys as a set for dependency validation.
    pub(crate) fn key_set(&self) -> BTreeSet<ParameterKey> {
        self.index_by_key.keys().cloned().collect()
    }
}

/// Invalid native parameter identity or scalar specification.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterError {
    /// One key segment is empty, untrimmed, or contains a reserved delimiter.
    InvalidKeySegment {
        /// Stable segment name.
        segment: &'static str,
    },
    /// Bounds contain NaN or are reversed.
    InvalidBounds,
    /// A parameter value is non-finite.
    NonFiniteValue {
        /// Parameter identity.
        key: ParameterKey,
    },
    /// A unit label is empty.
    InvalidUnit {
        /// Parameter identity.
        key: ParameterKey,
    },
    /// A value is outside its declared bounds.
    ValueOutsideBounds {
        /// Parameter identity.
        key: ParameterKey,
        /// Rejected value.
        value: f64,
    },
    /// Solver scale is non-finite or non-positive.
    InvalidScale {
        /// Parameter identity.
        key: ParameterKey,
    },
    /// A set repeats one key.
    DuplicateKey {
        /// Repeated identity.
        key: ParameterKey,
    },
}

impl Display for ParameterError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKeySegment { segment } => write!(
                formatter,
                "parameter {segment} must be trimmed, non-empty, and delimiter-safe"
            ),
            Self::InvalidBounds => {
                formatter.write_str("parameter bounds must be ordered and must not contain NaN")
            }
            Self::NonFiniteValue { key } => {
                write!(formatter, "parameter {key} value must be finite")
            }
            Self::InvalidUnit { key } => {
                write!(formatter, "parameter {key} unit must be non-empty")
            }
            Self::ValueOutsideBounds { key, value } => {
                write!(
                    formatter,
                    "parameter {key} value {value} lies outside its bounds"
                )
            }
            Self::InvalidScale { key } => {
                write!(
                    formatter,
                    "parameter {key} scale must be positive and finite"
                )
            }
            Self::DuplicateKey { key } => write!(formatter, "duplicate parameter key {key}"),
        }
    }
}

impl Error for ParameterError {}

fn validate_key_segment(segment: &'static str, value: &str) -> Result<(), ParameterError> {
    if value.is_empty()
        || value.trim() != value
        || value
            .chars()
            .any(|character| matches!(character, '[' | ']' | '\n' | '\r' | '\t'))
    {
        return Err(ParameterError::InvalidKeySegment { segment });
    }
    Ok(())
}
