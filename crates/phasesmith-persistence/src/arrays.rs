//! Deterministic `NumPy` NPY/NPZ array storage used by project bundles.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::{PersistenceError, ProjectReadLimits};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArrayDescriptor {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ArrayData {
    F64 {
        values: Vec<f64>,
        shape: Vec<usize>,
    },
    I32 {
        values: Vec<i32>,
        shape: Vec<usize>,
    },
    U64 {
        values: Vec<u64>,
        shape: Vec<usize>,
    },
    Bool {
        values: Vec<bool>,
        shape: Vec<usize>,
    },
}

impl ArrayData {
    fn dtype(&self) -> &'static str {
        match self {
            Self::F64 { .. } => "float64",
            Self::I32 { .. } => "int32",
            Self::U64 { .. } => "uint64",
            Self::Bool { .. } => "bool",
        }
    }

    fn npy_descr(&self) -> &'static str {
        match self {
            Self::F64 { .. } => "<f8",
            Self::I32 { .. } => "<i4",
            Self::U64 { .. } => "<u8",
            Self::Bool { .. } => "|b1",
        }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            Self::F64 { shape, .. }
            | Self::I32 { shape, .. }
            | Self::U64 { shape, .. }
            | Self::Bool { shape, .. } => shape,
        }
    }

    fn element_count(&self) -> usize {
        match self {
            Self::F64 { values, .. } => values.len(),
            Self::I32 { values, .. } => values.len(),
            Self::U64 { values, .. } => values.len(),
            Self::Bool { values, .. } => values.len(),
        }
    }

    fn raw_bytes(&self) -> Vec<u8> {
        match self {
            Self::F64 { values, .. } => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            Self::I32 { values, .. } => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            Self::U64 { values, .. } => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            Self::Bool { values, .. } => values.iter().map(|value| u8::from(*value)).collect(),
        }
    }

    pub(crate) fn descriptor(&self) -> ArrayDescriptor {
        let raw = self.raw_bytes();
        ArrayDescriptor {
            dtype: self.dtype().to_owned(),
            shape: self.shape().to_vec(),
            sha256: sha256_hex(&raw),
        }
    }

    pub(crate) fn f64(values: Vec<f64>, shape: Vec<usize>) -> Result<Self, PersistenceError> {
        validate_shape(values.len(), &shape)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PersistenceError::InvalidArray {
                message: "float array contains a non-finite value".to_owned(),
            });
        }
        Ok(Self::F64 { values, shape })
    }

    pub(crate) fn i32(values: Vec<i32>, shape: Vec<usize>) -> Result<Self, PersistenceError> {
        validate_shape(values.len(), &shape)?;
        Ok(Self::I32 { values, shape })
    }

    pub(crate) fn u64(values: Vec<u64>, shape: Vec<usize>) -> Result<Self, PersistenceError> {
        validate_shape(values.len(), &shape)?;
        Ok(Self::U64 { values, shape })
    }

    pub(crate) fn bool(values: Vec<bool>, shape: Vec<usize>) -> Result<Self, PersistenceError> {
        validate_shape(values.len(), &shape)?;
        Ok(Self::Bool { values, shape })
    }

    pub(crate) fn into_f64(self) -> Result<Vec<f64>, PersistenceError> {
        match self {
            Self::F64 { values, .. } => Ok(values),
            _ => Err(dtype_error("float64")),
        }
    }

    pub(crate) fn into_i32(self) -> Result<Vec<i32>, PersistenceError> {
        match self {
            Self::I32 { values, .. } => Ok(values),
            _ => Err(dtype_error("int32")),
        }
    }

    pub(crate) fn into_u64(self) -> Result<Vec<u64>, PersistenceError> {
        match self {
            Self::U64 { values, .. } => Ok(values),
            _ => Err(dtype_error("uint64")),
        }
    }

    pub(crate) fn into_bool(self) -> Result<Vec<bool>, PersistenceError> {
        match self {
            Self::Bool { values, .. } => Ok(values),
            _ => Err(dtype_error("bool")),
        }
    }
}

pub(crate) fn write_npz(arrays: &BTreeMap<String, ArrayData>) -> Result<Vec<u8>, PersistenceError> {
    let cursor = Cursor::new(Vec::new());
    let mut archive = ZipWriter::new(cursor);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .last_modified_time(zip::DateTime::default());
    for (name, array) in arrays {
        validate_array_name(name)?;
        archive
            .start_file(format!("{name}.npy"), options)
            .map_err(|error| zip_error(&error))?;
        archive.write_all(&write_npy(array)?)?;
    }
    archive
        .finish()
        .map(std::io::Cursor::into_inner)
        .map_err(|error| zip_error(&error))
}

pub(crate) fn read_npz(
    bytes: &[u8],
    descriptors: &BTreeMap<String, ArrayDescriptor>,
    limits: ProjectReadLimits,
) -> Result<BTreeMap<String, ArrayData>, PersistenceError> {
    for name in descriptors.keys() {
        validate_array_name(name)?;
    }
    let mut archive = ZipArchive::new(Cursor::new(bytes)).map_err(|error| zip_error(&error))?;
    if archive.len() != descriptors.len() || archive.len() > limits.max_arrays {
        return Err(PersistenceError::InvalidArchive {
            message: "archive members do not match the manifest".to_owned(),
        });
    }
    let expected = descriptors
        .keys()
        .map(|name| format!("{name}.npy"))
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    let mut encoded = BTreeMap::new();
    let mut total_uncompressed = 0_u64;
    for index in 0..archive.len() {
        let mut member = archive.by_index(index).map_err(|error| zip_error(&error))?;
        let name = member.name().to_owned();
        if !expected.contains(&name) || !observed.insert(name.clone()) || !member.is_file() {
            return Err(PersistenceError::InvalidArchive {
                message: "archive contains an unexpected or duplicate member".to_owned(),
            });
        }
        total_uncompressed = total_uncompressed
            .checked_add(member.size())
            .ok_or_else(|| PersistenceError::LimitExceeded {
                message: "archive uncompressed size overflow".to_owned(),
            })?;
        if total_uncompressed > limits.max_uncompressed_array_bytes {
            return Err(PersistenceError::LimitExceeded {
                message: "archive exceeds max_uncompressed_array_bytes".to_owned(),
            });
        }
        let capacity =
            usize::try_from(member.size()).map_err(|_| PersistenceError::LimitExceeded {
                message: "archive member does not fit memory limits".to_owned(),
            })?;
        let mut value = Vec::with_capacity(capacity);
        let read_limit = member.size().saturating_add(1);
        member.by_ref().take(read_limit).read_to_end(&mut value)?;
        if value.len() != capacity {
            return Err(PersistenceError::InvalidArchive {
                message: "archive member size does not match its ZIP record".to_owned(),
            });
        }
        encoded.insert(name, value);
    }
    let mut result = BTreeMap::new();
    for (name, descriptor) in descriptors {
        let member_name = format!("{name}.npy");
        let bytes =
            encoded
                .remove(&member_name)
                .ok_or_else(|| PersistenceError::InvalidArchive {
                    message: format!("archive member {member_name:?} is missing"),
                })?;
        let array = read_npy(&bytes, descriptor, limits)?;
        result.insert(name.clone(), array);
    }
    Ok(result)
}

fn write_npy(array: &ArrayData) -> Result<Vec<u8>, PersistenceError> {
    validate_shape(array.element_count(), array.shape())?;
    let shape = shape_literal(array.shape());
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}",
        array.npy_descr()
    );
    let prefix_len = 10;
    let padding = (64 - ((prefix_len + header.len() + 1) % 64)) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_length =
        u16::try_from(header.len()).map_err(|_| PersistenceError::InvalidArray {
            message: "NPY header is too long".to_owned(),
        })?;
    let raw = array.raw_bytes();
    let mut result = Vec::with_capacity(prefix_len + header.len() + raw.len());
    result.extend_from_slice(b"\x93NUMPY");
    result.extend_from_slice(&[1, 0]);
    result.extend_from_slice(&header_length.to_le_bytes());
    result.extend_from_slice(header.as_bytes());
    result.extend_from_slice(&raw);
    Ok(result)
}

fn read_npy(
    bytes: &[u8],
    descriptor: &ArrayDescriptor,
    limits: ProjectReadLimits,
) -> Result<ArrayData, PersistenceError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(invalid_npy("missing NPY magic"));
    }
    let (header_start, header_length): (usize, usize) = match (bytes[6], bytes[7]) {
        (1, 0) => (10, usize::from(u16::from_le_bytes([bytes[8], bytes[9]]))),
        (2 | 3, 0) if bytes.len() >= 12 => (
            12,
            usize::try_from(u32::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
            ]))
            .map_err(|_| invalid_npy("header length exceeds this platform"))?,
        ),
        _ => return Err(invalid_npy("unsupported NPY version")),
    };
    let data_start = header_start
        .checked_add(header_length)
        .filter(|value| *value <= bytes.len())
        .ok_or_else(|| invalid_npy("truncated NPY header"))?;
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|_| invalid_npy("NPY header is not UTF-8/ASCII"))?;
    if parse_header_bool(header, "fortran_order")? {
        return Err(invalid_npy("Fortran-order arrays are not supported"));
    }
    let shape = parse_shape(header)?;
    if shape != descriptor.shape {
        return Err(invalid_npy("NPY shape does not match the manifest"));
    }
    let element_count = checked_element_count(&shape)?;
    if element_count > limits.max_array_elements {
        return Err(PersistenceError::LimitExceeded {
            message: "array exceeds max_array_elements".to_owned(),
        });
    }
    let (descr, element_size): (&str, usize) = match descriptor.dtype.as_str() {
        "float64" => ("<f8", 8),
        "int32" => ("<i4", 4),
        "uint64" => ("<u8", 8),
        "bool" => ("|b1", 1),
        _ => return Err(invalid_npy("manifest dtype is unsupported")),
    };
    if parse_header_string(header, "descr")? != descr {
        return Err(invalid_npy("NPY dtype does not match the manifest"));
    }
    let expected_bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(|| invalid_npy("array byte count overflow"))?;
    let raw = &bytes[data_start..];
    if raw.len() != expected_bytes || sha256_hex(raw) != descriptor.sha256 {
        return Err(invalid_npy(
            "array length or SHA-256 does not match the manifest",
        ));
    }
    match descriptor.dtype.as_str() {
        "float64" => ArrayData::f64(
            raw.chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("exact f64 chunk length")))
                .collect(),
            shape,
        ),
        "int32" => ArrayData::i32(
            raw.chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("exact i32 chunk length")))
                .collect(),
            shape,
        ),
        "uint64" => ArrayData::u64(
            raw.chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("exact u64 chunk length")))
                .collect(),
            shape,
        ),
        "bool" => ArrayData::bool(
            raw.iter()
                .map(|value| match value {
                    0 => Ok(false),
                    1 => Ok(true),
                    _ => Err(invalid_npy("boolean array contains a non-boolean byte")),
                })
                .collect::<Result<Vec<_>, _>>()?,
            shape,
        ),
        _ => unreachable!(),
    }
}

fn parse_shape(header: &str) -> Result<Vec<usize>, PersistenceError> {
    let remainder = header_field_value(header, "shape")?;
    let open = remainder
        .find('(')
        .ok_or_else(|| invalid_npy("NPY shape tuple is missing"))?;
    let close = remainder[open + 1..]
        .find(')')
        .map(|index| open + 1 + index)
        .ok_or_else(|| invalid_npy("NPY shape tuple is unterminated"))?;
    let body = &remainder[open + 1..close];
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .map_err(|_| invalid_npy("NPY shape contains an invalid dimension"))
        })
        .collect()
}

fn parse_header_string<'a>(header: &'a str, key: &str) -> Result<&'a str, PersistenceError> {
    let value = header_field_value(header, key)?;
    let quote = value
        .chars()
        .next()
        .filter(|value| matches!(value, '\'' | '"'))
        .ok_or_else(|| invalid_npy("NPY header string is invalid"))?;
    let body = &value[quote.len_utf8()..];
    let end = body
        .find(quote)
        .ok_or_else(|| invalid_npy("NPY header string is unterminated"))?;
    Ok(&body[..end])
}

fn parse_header_bool(header: &str, key: &str) -> Result<bool, PersistenceError> {
    let value = header_field_value(header, key)?;
    if value.starts_with("False") {
        Ok(false)
    } else if value.starts_with("True") {
        Ok(true)
    } else {
        Err(invalid_npy("NPY header boolean is invalid"))
    }
}

fn header_field_value<'a>(header: &'a str, key: &str) -> Result<&'a str, PersistenceError> {
    let single = format!("'{key}'");
    let double = format!("\"{key}\"");
    let matches = header
        .match_indices(&single)
        .chain(header.match_indices(&double));
    let positions = matches
        .map(|(position, token)| (position, token.len()))
        .collect::<Vec<_>>();
    if positions.len() != 1 {
        return Err(invalid_npy("NPY header field is missing or duplicated"));
    }
    let (position, key_length) = positions[0];
    let remainder = header[position + key_length..].trim_start();
    let value = remainder
        .strip_prefix(':')
        .ok_or_else(|| invalid_npy("NPY header field separator is invalid"))?;
    Ok(value.trim_start())
}

fn shape_literal(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_owned(),
        [only] => format!("({only},)"),
        values => format!(
            "({})",
            values
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn validate_shape(element_count: usize, shape: &[usize]) -> Result<(), PersistenceError> {
    if checked_element_count(shape)? != element_count {
        return Err(PersistenceError::InvalidArray {
            message: "array shape does not match its element count".to_owned(),
        });
    }
    Ok(())
}

fn checked_element_count(shape: &[usize]) -> Result<usize, PersistenceError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| PersistenceError::InvalidArray {
                message: "array shape element count overflow".to_owned(),
            })
    })
}

fn validate_array_name(name: &str) -> Result<(), PersistenceError> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(PersistenceError::InvalidArray {
            message: format!("invalid array name {name:?}"),
        });
    }
    Ok(())
}

fn dtype_error(expected: &str) -> PersistenceError {
    PersistenceError::InvalidArray {
        message: format!("array dtype is not {expected}"),
    }
}

fn invalid_npy(message: &str) -> PersistenceError {
    PersistenceError::InvalidArchive {
        message: message.to_owned(),
    }
}

fn zip_error(error: &zip::result::ZipError) -> PersistenceError {
    PersistenceError::InvalidArchive {
        message: error.to_string(),
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::{parse_header_bool, parse_header_string, parse_shape};

    #[test]
    fn header_fields_are_parsed_by_key_instead_of_substring() {
        let valid = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }";
        assert_eq!(parse_header_string(valid, "descr").unwrap(), "<f8");
        assert!(!parse_header_bool(valid, "fortran_order").unwrap());
        assert_eq!(parse_shape(valid).unwrap(), [2, 3]);

        let spoofed =
            "{'descr': '>f8', 'note': \"'descr': '<f8'\", 'fortran_order': False, 'shape': (1,), }";
        assert!(parse_header_string(spoofed, "descr").is_err());
    }
}
