//! Additional deposited cell settings and exact origin checks for CIF declarations.

use std::collections::BTreeSet;

use super::{
    SpaceGroup, SpaceGroupInfo, SpaceGroupLookupError, general_hall_group, hall_key, symbol_key,
};

pub(super) struct CellSetting {
    pub(super) number: i32,
    hm: &'static str,
    hall: &'static str,
    transformed_hall: &'static str,
}

// Re-express the transformed primitive generators and lattice translations in
// the deposited cell. Equivalent Hall spellings generate precisely the same
// operations in that cell, including its centering cosets.
const fn setting(
    number: i32,
    hm: &'static str,
    hall: &'static str,
    transformed_hall: &'static str,
) -> CellSetting {
    CellSetting {
        number,
        hm,
        hall,
        transformed_hall,
    }
}

pub(super) const SETTINGS: &[CellSetting] = &[
    setting(2, "A -1", "-A 1", "-P 1 (-x,-1/2*y+1/2*z,1/2*y+1/2*z)"),
    setting(2, "B -1", "-B 1", "-P 1 (-1/2*x+1/2*z,-y,1/2*x+1/2*z)"),
    setting(2, "C -1", "-C 1", "-P 1 (1/2*x+1/2*y,1/2*x-1/2*y,-z)"),
    setting(
        2,
        "I -1",
        "-I 1",
        "-P 1 (-1/2*x+1/2*y+1/2*z,1/2*x-1/2*y+1/2*z,1/2*x+1/2*y-1/2*z)",
    ),
    setting(
        2,
        "F -1",
        "-F 1",
        "-P 1 (1/2*y+1/2*z,1/2*x+1/2*z,1/2*x+1/2*y)",
    ),
    setting(4, "P 1 21 1 (a,b,c-1/4)", "P 2ybc", "P 2yb (x,y,z+1/4)"),
    // In (c, 2a+c, b), a primitive a translation becomes (-1/2,1/2,0)
    // and the original b-axis screw acts along the deposited c axis.
    setting(4, "C 1 1 21", "C 2c", "P 2yb (-1/2*x+z,1/2*x,y)"),
    setting(9, "F 1 d 1", "F -2yuw", "C -2yc (x+1/2*z,y,1/2*z)"),
    setting(
        12,
        "C 1 2/m 1 (a+c-1/4,b+1/4,c)",
        "-I 2yb",
        "-C 2y (x+1/4,y-1/4,-x+z-1/4)",
    ),
    setting(
        139,
        "F 4/m m m",
        "-F 4 2",
        "-I 4 2 (1/2*x+1/2*y,-1/2*x+1/2*y,z)",
    ),
];

impl CellSetting {
    pub(super) fn info(&self) -> Result<SpaceGroupInfo, SpaceGroupLookupError> {
        Ok(SpaceGroupInfo {
            number: self.number,
            hm_symbol: self.hm.to_owned(),
            hall_symbol: self.hall.to_owned(),
            setting: self.hm.to_owned(),
            space_group: general_hall_group(self.hall)?,
        })
    }
}

pub(super) fn by_hm(key: &str) -> Option<&'static CellSetting> {
    let key = match key {
        "p1211(c,2*a+c,b)" => "c1121",
        "i4/mmm(a+b,-a+b,c)" => "f4/mmm",
        _ => key,
    };
    SETTINGS.iter().find(|entry| symbol_key(entry.hm) == key)
}

pub(super) fn by_transformed_hall(value: &str) -> Option<&'static CellSetting> {
    let (base, transform) = value.split_once('(')?;
    let base = hall_key(base);
    let transform = symbol_key(transform);
    SETTINGS.iter().find(|entry| {
        let (prefix, suffix) = entry
            .transformed_hall
            .split_once('(')
            .expect("tabulated transformation");
        hall_key(prefix) == base && symbol_key(suffix) == transform
    })
}

type GridOperation = ([[i32; 3]; 3], [i64; 3]);

fn on_grid(group: &SpaceGroup) -> Option<Vec<GridOperation>> {
    group
        .operations()
        .iter()
        .map(|op| {
            let mut translation = [0; 3];
            for (component, t) in translation.iter_mut().zip(op.translation()) {
                if 24 % t.denominator() != 0 {
                    return None;
                }
                *component = t.numerator() * (24 / t.denominator());
            }
            Some((op.rotation(), translation))
        })
        .collect()
}

/// An unqualified H-M name specifies the basis but leaves the origin free.
/// Prove equivalence by one shared shift t' = t + (I-R)s for the entire group.
/// Search conventional shifts in 24ths, exactly, without moving any sites.
pub(super) fn same_up_to_origin(reference: &SpaceGroup, declared: &SpaceGroup) -> bool {
    if reference == declared {
        return true;
    }
    if reference.operations().len() != declared.operations().len()
        || reference.rotations() != declared.rotations()
    {
        return false;
    }
    let (Some(reference), Some(declared)) = (on_grid(reference), on_grid(declared)) else {
        return false;
    };
    let target: BTreeSet<_> = declared.into_iter().collect();
    for x in 0..24 {
        for y in 0..24 {
            for z in 0..24 {
                let shift = [x, y, z];
                if reference.iter().all(|&(rotation, t)| {
                    let translated = std::array::from_fn(|i| {
                        let rotated: i64 = rotation[i]
                            .iter()
                            .zip(shift)
                            .map(|(&r, s)| i64::from(r) * s)
                            .sum();
                        (t[i] + shift[i] - rotated).rem_euclid(24)
                    });
                    target.contains(&(rotation, translated))
                }) {
                    return true;
                }
            }
        }
    }
    false
}
