//! Prepared non-resonant X-ray and coherent neutron scattering models.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[path = "scattering_data.rs"]
mod scattering_data;

use scattering_data::{NEUTRON_ROWS, XRAY_ROWS};

/// Maximum `sin(theta) / wavelength` supported by the fitted X-ray table.
pub const XRAY_MAX_S_INVERSE_ANGSTROM: f64 = 6.0;

#[cfg(test)]
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
#[cfg(test)]
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug)]
struct XrayTableRow {
    key: &'static str,
    atomic_number: u8,
    a: [f64; 5],
    b: [f64; 5],
    c: f64,
}

#[derive(Clone, Copy, Debug)]
struct NeutronTableRow {
    key: &'static str,
    atomic_number: u8,
    isotope: Option<u16>,
    b_c_fm: f64,
    uncertainty_fm: Option<f64>,
    energy_dependent: bool,
    derived_alias_of: Option<&'static str>,
}

/// Immutable provenance attached to one generated native table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScatteringTableProvenance {
    /// Human-readable model/table name.
    pub name: &'static str,
    /// Pinned upstream Git commit.
    pub upstream_commit: &'static str,
    /// SHA-256 of the exact parsed upstream source file.
    pub source_sha256: &'static str,
    /// Stable FNV-1a fingerprint over every generated native record.
    pub table_fnv64: u64,
    /// Number of generated records.
    pub row_count: usize,
}

/// Provenance for the Waasmaier--Kirfel X-ray coefficient table.
pub const XRAY_TABLE_PROVENANCE: ScatteringTableProvenance = ScatteringTableProvenance {
    name: "XrayDB Waasmaier-Kirfel",
    upstream_commit: scattering_data::XRAY_SOURCE_COMMIT,
    source_sha256: scattering_data::XRAY_SOURCE_SHA256,
    table_fnv64: scattering_data::XRAY_TABLE_FNV64,
    row_count: 211,
};

/// Provenance for the bound coherent neutron scattering-length table.
pub const NEUTRON_TABLE_PROVENANCE: ScatteringTableProvenance = ScatteringTableProvenance {
    name: "periodictable bound coherent neutron lengths",
    upstream_commit: scattering_data::NEUTRON_SOURCE_COMMIT,
    source_sha256: scattering_data::NEUTRON_SOURCE_SHA256,
    table_fnv64: scattering_data::NEUTRON_TABLE_FNV64,
    row_count: 367,
};

/// Amplitude units used by a scattering batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatteringAmplitudeUnit {
    /// Electron units for non-resonant X-ray `f0`.
    Electrons,
    /// Femtometres for bound coherent neutron nuclear scattering.
    Femtometres,
}

/// Reflection-major amplitudes and `s` derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringBatch {
    /// Real amplitudes, shape `(reflection_count, site_count)` when flattened.
    pub real: Vec<f64>,
    /// Imaginary amplitudes, currently zero for both built-in baseline models.
    pub imag: Vec<f64>,
    /// Derivative of the real amplitude with respect to `s`.
    pub d_real_d_s: Vec<f64>,
    /// Derivative of the imaginary amplitude with respect to `s`.
    pub d_imag_d_s: Vec<f64>,
    /// Number of reflection rows.
    pub reflection_count: usize,
    /// Number of site/species columns.
    pub site_count: usize,
    /// Physical amplitude unit.
    pub unit: ScatteringAmplitudeUnit,
}

/// Public metadata for one neutron table identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeutronSpeciesMetadata {
    /// Stable canonical key such as `Fe` or `H-2`.
    pub key: &'static str,
    /// Atomic number.
    pub atomic_number: u8,
    /// Isotope mass number, or `None` for an elemental identity.
    pub isotope: Option<u16>,
    /// Real bound coherent length in femtometres.
    pub b_c_fm: f64,
    /// Source standard uncertainty in femtometres, when supplied.
    pub uncertainty_fm: Option<f64>,
    /// Whether the source marks the identity as energy dependent.
    pub energy_dependent: bool,
    /// Isotope row inherited by an elemental alias, if applicable.
    pub derived_alias_of: Option<&'static str>,
}

/// Public metadata for one exact X-ray table state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XraySpeciesMetadata {
    /// Exact Waasmaier--Kirfel source key such as `Fe` or `Fe3+`.
    pub key: &'static str,
    /// Atomic number associated with the electronic state.
    pub atomic_number: u8,
}

/// Invalid scattering-model preparation or batch input.
#[derive(Clone, Debug, PartialEq)]
pub enum ScatteringError {
    /// Exact X-ray table state is unavailable.
    UnknownXraySpecies(String),
    /// Exact neutron element/isotope identity is unavailable.
    UnknownNeutronSpecies(String),
    /// A constant neutron model was requested for an energy-dependent row.
    EnergyDependentNeutronSpecies(String),
    /// An `s` value is non-finite, negative, or beyond the X-ray fit range.
    InvalidScatteringVector,
    /// A reflection-major output allocation would overflow.
    AllocationOverflow,
}

impl Display for ScatteringError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownXraySpecies(key) => {
                write!(formatter, "unknown X-ray scattering species {key:?}")
            }
            Self::UnknownNeutronSpecies(key) => {
                write!(formatter, "unknown neutron scattering species {key:?}")
            }
            Self::EnergyDependentNeutronSpecies(key) => write!(
                formatter,
                "neutron species {key:?} requires an energy-dependent model"
            ),
            Self::InvalidScatteringVector => formatter.write_str(
                "scattering vector s must be finite and non-negative, and X-ray s must not exceed 6 inverse angstrom",
            ),
            Self::AllocationOverflow => formatter.write_str("scattering batch allocation overflow"),
        }
    }
}

impl Error for ScatteringError {}

/// Prepared non-resonant Waasmaier--Kirfel species cache.
#[derive(Clone, Debug)]
pub struct PreparedXrayScattering {
    unique_rows: Vec<&'static XrayTableRow>,
    site_to_unique: Vec<usize>,
}

impl PreparedXrayScattering {
    /// Resolve exact source-table labels once for one site batch.
    ///
    /// # Errors
    ///
    /// Returns [`ScatteringError::UnknownXraySpecies`] without a neutral-state
    /// fallback when any requested label is absent.
    pub fn new<I, S>(species: I) -> Result<Self, ScatteringError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let rows = species
            .into_iter()
            .map(|key| {
                let key = key.as_ref();
                xray_row(key).ok_or_else(|| ScatteringError::UnknownXraySpecies(key.to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (unique_rows, site_to_unique) = unique_rows(rows);
        Ok(Self {
            unique_rows,
            site_to_unique,
        })
    }

    /// Number of site columns in every evaluated batch.
    #[must_use]
    pub fn site_count(&self) -> usize {
        self.site_to_unique.len()
    }

    /// Number of distinct table rows evaluated for every reflection.
    #[must_use]
    pub fn unique_species_count(&self) -> usize {
        self.unique_rows.len()
    }

    /// Evaluate amplitudes and analytical `df0/ds` in one reflection/species pass.
    ///
    /// # Errors
    ///
    /// Returns [`ScatteringError`] for invalid `s` or allocation overflow.
    pub fn evaluate(&self, s_inverse_angstrom: &[f64]) -> Result<ScatteringBatch, ScatteringError> {
        if s_inverse_angstrom
            .iter()
            .any(|&s| !s.is_finite() || !(0.0..=XRAY_MAX_S_INVERSE_ANGSTROM).contains(&s))
        {
            return Err(ScatteringError::InvalidScatteringVector);
        }
        let output_count = output_count(s_inverse_angstrom.len(), self.site_count())?;
        let mut result = empty_batch(
            output_count,
            s_inverse_angstrom.len(),
            self.site_count(),
            ScatteringAmplitudeUnit::Electrons,
        );
        let mut unique_values = vec![(0.0, 0.0); self.unique_species_count()];
        for (reflection, &s) in s_inverse_angstrom.iter().enumerate() {
            for (index, row) in self.unique_rows.iter().enumerate() {
                unique_values[index] = xray_value_and_derivative(row, s);
            }
            let start = reflection * self.site_count();
            for (site, &unique) in self.site_to_unique.iter().enumerate() {
                (result.real[start + site], result.d_real_d_s[start + site]) =
                    unique_values[unique];
            }
        }
        Ok(result)
    }
}

/// Prepared constant coherent neutron species cache.
#[derive(Clone, Debug)]
pub struct PreparedNeutronScattering {
    unique_rows: Vec<&'static NeutronTableRow>,
    site_to_unique: Vec<usize>,
}

impl PreparedNeutronScattering {
    /// Resolve exact natural/isotope keys and reject energy-dependent rows.
    ///
    /// # Errors
    ///
    /// Returns [`ScatteringError`] for missing or energy-dependent identities.
    pub fn new<I, S>(species: I) -> Result<Self, ScatteringError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let rows = species
            .into_iter()
            .map(|key| {
                let key = key.as_ref();
                let row = neutron_row(key)
                    .ok_or_else(|| ScatteringError::UnknownNeutronSpecies(key.to_owned()))?;
                if row.energy_dependent {
                    return Err(ScatteringError::EnergyDependentNeutronSpecies(
                        key.to_owned(),
                    ));
                }
                Ok(row)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (unique_rows, site_to_unique) = unique_rows(rows);
        Ok(Self {
            unique_rows,
            site_to_unique,
        })
    }

    /// Number of site columns in every evaluated batch.
    #[must_use]
    pub fn site_count(&self) -> usize {
        self.site_to_unique.len()
    }

    /// Number of distinct table rows copied for every reflection.
    #[must_use]
    pub fn unique_species_count(&self) -> usize {
        self.unique_rows.len()
    }

    /// Evaluate constant amplitudes and identically zero `df/ds`.
    ///
    /// # Errors
    ///
    /// Returns [`ScatteringError`] for invalid `s` or allocation overflow.
    pub fn evaluate(&self, s_inverse_angstrom: &[f64]) -> Result<ScatteringBatch, ScatteringError> {
        if s_inverse_angstrom
            .iter()
            .any(|&s| !s.is_finite() || s < 0.0)
        {
            return Err(ScatteringError::InvalidScatteringVector);
        }
        let output_count = output_count(s_inverse_angstrom.len(), self.site_count())?;
        let mut result = empty_batch(
            output_count,
            s_inverse_angstrom.len(),
            self.site_count(),
            ScatteringAmplitudeUnit::Femtometres,
        );
        for reflection in 0..s_inverse_angstrom.len() {
            let start = reflection * self.site_count();
            for (site, &unique) in self.site_to_unique.iter().enumerate() {
                result.real[start + site] = self.unique_rows[unique].b_c_fm;
            }
        }
        Ok(result)
    }
}

/// Return metadata for one exact neutron key.
#[must_use]
pub fn neutron_species_metadata(key: &str) -> Option<NeutronSpeciesMetadata> {
    neutron_row(key).map(|row| NeutronSpeciesMetadata {
        key: row.key,
        atomic_number: row.atomic_number,
        isotope: row.isotope,
        b_c_fm: row.b_c_fm,
        uncertainty_fm: row.uncertainty_fm,
        energy_dependent: row.energy_dependent,
        derived_alias_of: row.derived_alias_of,
    })
}

/// Return metadata for one exact X-ray state key.
#[must_use]
pub fn xray_species_metadata(key: &str) -> Option<XraySpeciesMetadata> {
    xray_row(key).map(|row| XraySpeciesMetadata {
        key: row.key,
        atomic_number: row.atomic_number,
    })
}

fn xray_row(key: &str) -> Option<&'static XrayTableRow> {
    XRAY_ROWS
        .binary_search_by_key(&key, |row| row.key)
        .ok()
        .map(|index| &XRAY_ROWS[index])
}

fn neutron_row(key: &str) -> Option<&'static NeutronTableRow> {
    NEUTRON_ROWS
        .binary_search_by_key(&key, |row| row.key)
        .ok()
        .map(|index| &NEUTRON_ROWS[index])
}

fn unique_rows<T>(rows: Vec<&'static T>) -> (Vec<&'static T>, Vec<usize>) {
    let mut unique: Vec<&'static T> = Vec::new();
    let mut indices = Vec::with_capacity(rows.len());
    for row in rows {
        let index = unique
            .iter()
            .position(|&candidate| std::ptr::eq(candidate, row))
            .unwrap_or_else(|| {
                unique.push(row);
                unique.len() - 1
            });
        indices.push(index);
    }
    (unique, indices)
}

fn xray_value_and_derivative(row: &XrayTableRow, s: f64) -> (f64, f64) {
    let s_squared = s * s;
    let mut value = row.c;
    let mut weighted = 0.0;
    for index in 0..5 {
        let gaussian = row.a[index] * (-row.b[index] * s_squared).exp();
        value += gaussian;
        weighted += row.b[index] * gaussian;
    }
    (value, -2.0 * s * weighted)
}

fn output_count(reflection_count: usize, site_count: usize) -> Result<usize, ScatteringError> {
    reflection_count
        .checked_mul(site_count)
        .ok_or(ScatteringError::AllocationOverflow)
}

fn empty_batch(
    output_count: usize,
    reflection_count: usize,
    site_count: usize,
    unit: ScatteringAmplitudeUnit,
) -> ScatteringBatch {
    ScatteringBatch {
        real: vec![0.0; output_count],
        imag: vec![0.0; output_count],
        d_real_d_s: vec![0.0; output_count],
        d_imag_d_s: vec![0.0; output_count],
        reflection_count,
        site_count,
        unit,
    }
}

#[cfg(test)]
fn fnv_update(mut value: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(FNV_PRIME);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xray_table_fingerprint() -> u64 {
        let mut value = FNV_OFFSET;
        for row in XRAY_ROWS {
            value = fnv_update(value, row.key.as_bytes());
            value = fnv_update(value, &[0, row.atomic_number]);
            for number in row.a.into_iter().chain(row.b).chain([row.c]) {
                value = fnv_update(value, &number.to_le_bytes());
            }
        }
        value
    }

    fn neutron_table_fingerprint() -> u64 {
        let mut value = FNV_OFFSET;
        for row in NEUTRON_ROWS {
            value = fnv_update(value, row.key.as_bytes());
            value = fnv_update(value, &[0, row.atomic_number]);
            value = fnv_update(value, &row.isotope.unwrap_or(0).to_le_bytes());
            value = fnv_update(value, &row.b_c_fm.to_le_bytes());
            value = fnv_update(value, &[u8::from(row.uncertainty_fm.is_some())]);
            if let Some(uncertainty) = row.uncertainty_fm {
                value = fnv_update(value, &uncertainty.to_le_bytes());
            }
            value = fnv_update(value, &[u8::from(row.energy_dependent)]);
            value = fnv_update(value, row.derived_alias_of.unwrap_or("").as_bytes());
            value = fnv_update(value, &[0]);
        }
        value
    }

    #[test]
    fn generated_tables_have_stable_fingerprints_and_sorted_keys() {
        assert_eq!(XRAY_ROWS.len(), XRAY_TABLE_PROVENANCE.row_count);
        assert_eq!(NEUTRON_ROWS.len(), NEUTRON_TABLE_PROVENANCE.row_count);
        assert!(XRAY_ROWS.windows(2).all(|rows| rows[0].key < rows[1].key));
        assert!(
            NEUTRON_ROWS
                .windows(2)
                .all(|rows| rows[0].key < rows[1].key)
        );
        assert_eq!(xray_table_fingerprint(), XRAY_TABLE_PROVENANCE.table_fnv64);
        assert_eq!(
            neutron_table_fingerprint(),
            NEUTRON_TABLE_PROVENANCE.table_fnv64
        );
    }

    #[test]
    fn xray_values_and_derivatives_match_closed_forms() {
        let prepared = PreparedXrayScattering::new(["H", "Fe", "Fe"]).unwrap();
        assert_eq!(prepared.site_count(), 3);
        assert_eq!(prepared.unique_species_count(), 2);
        let result = prepared.evaluate(&[0.0, 0.7]).unwrap();
        assert_eq!(result.real[0].to_bits(), 0.999_978_f64.to_bits());
        assert_eq!(result.d_real_d_s[0].abs().to_bits(), 0.0_f64.to_bits());
        assert_eq!(result.real[1].to_bits(), result.real[2].to_bits());
        let step = 1.0e-6;
        let plus = prepared.evaluate(&[0.7 + step]).unwrap();
        let minus = prepared.evaluate(&[0.7 - step]).unwrap();
        let finite_difference = (plus.real[1] - minus.real[1]) / (2.0 * step);
        assert!((result.d_real_d_s[4] - finite_difference).abs() < 2.0e-9);
        assert!(result.imag.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn xray_species_and_range_errors_are_explicit() {
        assert_eq!(
            PreparedXrayScattering::new(["DefinitelyMissing"]).unwrap_err(),
            ScatteringError::UnknownXraySpecies("DefinitelyMissing".to_owned())
        );
        let ionic = PreparedXrayScattering::new(["Fe3+"]).unwrap();
        assert!(ionic.evaluate(&[6.0]).is_ok());
        for invalid in [-0.1, 6.000_001, f64::NAN, f64::INFINITY] {
            assert_eq!(
                ionic.evaluate(&[invalid]).unwrap_err(),
                ScatteringError::InvalidScatteringVector
            );
        }
    }

    #[test]
    fn neutron_isotopes_aliases_and_energy_dependence_are_explicit() {
        let prepared = PreparedNeutronScattering::new(["H", "H-2", "H-2"]).unwrap();
        assert_eq!(prepared.unique_species_count(), 2);
        let result = prepared.evaluate(&[0.0, 4.0]).unwrap();
        assert_eq!(&result.real[..3], &[-3.7409, 6.6681, 6.6681]);
        assert_eq!(&result.real[3..], &[-3.7409, 6.6681, 6.6681]);
        assert!(result.d_real_d_s.iter().all(|value| *value == 0.0));
        let fluorine = neutron_species_metadata("F").unwrap();
        assert_eq!(fluorine.derived_alias_of, Some("F-19"));
        assert_eq!(fluorine.isotope, None);
        assert_eq!(
            PreparedNeutronScattering::new(["Cd"]).unwrap_err(),
            ScatteringError::EnergyDependentNeutronSpecies("Cd".to_owned())
        );
        assert_eq!(
            PreparedNeutronScattering::new(["H-999"]).unwrap_err(),
            ScatteringError::UnknownNeutronSpecies("H-999".to_owned())
        );
    }
}
