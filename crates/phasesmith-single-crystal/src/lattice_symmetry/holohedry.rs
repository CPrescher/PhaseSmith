//! Point-group operations of the holohedry (lattice point group) of each
//! lattice system in its conventional setting.
use super::{Bravais, LatticeSystem};

/// Integer rotation acting on fractional coordinates of the conventional
/// cell, `x' = W x`. Miller indices, as row vectors, transform as
/// `h' = h W`, so the reflections equivalent to `h` are `{h W}`.
pub type Rotation = [[i32; 3]; 3];

const IDENTITY: Rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const INVERSION: Rotation = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]];

fn multiply(a: Rotation, b: Rotation) -> Rotation {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// Closure of the generators under multiplication, sorted for determinism.
fn closure(generators: &[Rotation]) -> Vec<Rotation> {
    let mut group = vec![IDENTITY];
    let mut index = 0;
    while index < group.len() {
        let a = group[index];
        for &g in generators {
            let p = multiply(a, g);
            if !group.contains(&p) {
                group.push(p);
            }
        }
        index += 1;
    }
    group.sort_unstable();
    group
}

impl LatticeSystem {
    /// Rotations of the holohedry in the conventional setting of
    /// [`super::search`] (unique axis b for monoclinic, c for tetragonal and
    /// hexagonal, hexagonal axes with the obverse centring for rhombohedral),
    /// inversion included: `-1`, `2/m`, `mmm`, `4/mmm`, `-3m` (two-fold axes
    /// along a, b and a + b), `6/mmm` or `m-3m`. The number of operations is
    /// [`Bravais::holohedry_order`]. They act on Miller indices as row
    /// vectors (`h' = h W`); their orbits are the reflections equivalent in
    /// the Laue class of the lattice, with Friedel pairs merged.
    #[must_use]
    pub fn holohedry(self) -> Vec<Rotation> {
        let four_z = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
        let two_x = [[1, 0, 0], [0, -1, 0], [0, 0, -1]];
        let generators: Vec<Rotation> = match self {
            Self::Triclinic => vec![INVERSION],
            Self::Monoclinic => vec![INVERSION, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]],
            Self::Orthorhombic => vec![INVERSION, two_x, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]],
            Self::Tetragonal => vec![INVERSION, four_z, two_x],
            // Threefold (-y, x-y, z) and twofold (y, x, -z), as in R-3m.
            Self::Rhombohedral => vec![
                INVERSION,
                [[0, -1, 0], [1, -1, 0], [0, 0, 1]],
                [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
            ],
            // Sixfold (x-y, x, z) and twofold (y, x, -z), as in P6/mmm.
            Self::Hexagonal => vec![
                INVERSION,
                [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
                [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
            ],
            Self::Cubic => vec![INVERSION, four_z, [[0, 0, 1], [1, 0, 0], [0, 1, 0]]],
        };
        closure(&generators)
    }

    /// Hermann–Mauguin symbol of the holohedry (the Laue class of the
    /// lattice): `-1`, `2/m`, `mmm`, `4/mmm`, `-3m`, `6/mmm` or `m-3m`.
    #[must_use]
    pub fn laue_symbol(self) -> &'static str {
        match self {
            Self::Triclinic => "-1",
            Self::Monoclinic => "2/m",
            Self::Orthorhombic => "mmm",
            Self::Tetragonal => "4/mmm",
            Self::Rhombohedral => "-3m",
            Self::Hexagonal => "6/mmm",
            Self::Cubic => "m-3m",
        }
    }
}

impl Bravais {
    /// Rotations of the holohedry of the type's lattice system; see
    /// [`LatticeSystem::holohedry`].
    #[must_use]
    pub fn holohedry(self) -> Vec<Rotation> {
        self.system().holohedry()
    }
}

/// Indices `h W` of `hkl` under one rotation (row-vector convention).
#[must_use]
pub fn transform_index(hkl: [i64; 3], w: &Rotation) -> [i64; 3] {
    std::array::from_fn(|j| (0..3).map(|k| hkl[k] * i64::from(w[k][j])).sum())
}

/// Representative of the reflections equivalent to `hkl` under the
/// operations: the lexicographically largest member of the orbit `{h W}`.
/// With an operation set that contains the inversion, Friedel mates share a
/// representative.
#[must_use]
pub fn equivalent_representative(hkl: [i64; 3], operations: &[Rotation]) -> [i64; 3] {
    operations
        .iter()
        .map(|w| transform_index(hkl, w))
        .max()
        .unwrap_or(hkl)
}
