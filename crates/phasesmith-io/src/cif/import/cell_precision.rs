//! Reconcile redundant metric components printed at differing decimal precision.

use super::{
    CELL_TAGS, CifBlock, CifDiagnostic, CifIoError, SpaceGroup, UnitCell, validate_cell_metric,
};

fn mapped_axis(rotation: &[[i32; 3]; 3], axis: usize) -> Option<(usize, i32)> {
    let mut nonzero = (0..3).filter(|&i| rotation[i][axis] != 0);
    let i = nonzero.next()?;
    (nonzero.next().is_none() && matches!(rotation[i][axis], -1 | 1))
        .then_some((i, rotation[i][axis]))
}

/// A rotation that maps two direct vectors to two axes with the same sign
/// preserves their included angle. Using the same vector twice tests its length.
fn equivalent(group: &SpaceGroup, first: usize, second: usize) -> bool {
    const AXES: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let (left, right) = AXES[first];
    let target = AXES[second];
    group.rotations().iter().any(|rotation| {
        matches!((mapped_axis(rotation, left), mapped_axis(rotation, right)),
            (Some((axis1, sign1)), Some((axis2, sign2))) if sign1 == sign2
                && ((axis1, axis2) == target || (axis2, axis1) == target))
    })
}

fn decimal_places(literal: &str) -> Option<usize> {
    let nominal = literal.split('(').next()?;
    let (whole, decimals) = nominal.split_once('.').unwrap_or((nominal, ""));
    (whole
        .bytes()
        .chain(decimals.bytes())
        .all(|b| b.is_ascii_digit())
        && decimals.len() <= 15)
        .then_some(decimals.len())
}

// After rounding, the printed decimal values must be identical.
#[allow(clippy::float_cmp)]
pub(super) fn checked_cell(
    block: &CifBlock,
    cell: UnitCell,
    uncertainties: &[Option<f64>; 6],
    group: &SpaceGroup,
    diagnostics: &mut Vec<CifDiagnostic>,
) -> Result<UnitCell, CifIoError> {
    let Err(original_error) = validate_cell_metric(cell, group) else {
        return Ok(cell);
    };
    let raw = CELL_TAGS.map(|aliases| {
        block
            .first_raw(&aliases)
            .and_then(|(_, v)| v.text())
            .unwrap_or("")
    });
    let places = raw.map(decimal_places);
    let values = [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ];
    let mut reconciled = values;
    for i in 0..6 {
        let (Some(precision), Some(sigma)) = (places[i], uncertainties[i]) else {
            continue;
        };
        let mut candidates = (0..6).filter(|&j| {
            uncertainties[j].is_none()
                && places[j].is_some_and(|n| n > precision)
                && equivalent(group, i, j)
        });
        let Some(j) = candidates.next() else { continue };
        let finer = values[j];
        // The two nominal values must agree when printed to the precision of
        // the coarser one. Uncertainty by itself never authorizes a correction.
        if values[i] == finer
            || candidates.any(|k| values[k] != finer)
            || (values[i] - finer).abs() > sigma
            || format!("{finer:.precision$}").parse::<f64>().ok() != Some(values[i])
        {
            continue;
        }
        reconciled[i] = finer;
        let message = format!(
            "{} = {} and symmetry-equivalent {} = {}. Used {} for both and retained the reported uncertainty",
            CELL_TAGS[i][1], raw[i], CELL_TAGS[j][1], raw[j], raw[j],
        );
        diagnostics.push(
            CifDiagnostic::warning("rounded_equivalent_cell_parameter", message)
                .with_tag(CELL_TAGS[i][1]),
        );
    }
    if reconciled == values {
        return Err(original_error);
    }
    let [
        a_angstrom,
        b_angstrom,
        c_angstrom,
        alpha_deg,
        beta_deg,
        gamma_deg,
    ] = reconciled;
    let result = UnitCell {
        a_angstrom,
        b_angstrom,
        c_angstrom,
        alpha_deg,
        beta_deg,
        gamma_deg,
    };
    validate_cell_metric(result, group)?;
    Ok(result)
}
