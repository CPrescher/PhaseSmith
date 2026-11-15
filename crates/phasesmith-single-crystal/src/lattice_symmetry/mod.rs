//! Lattice symmetry from the metric: Bravais-lattice search and
//! symmetry-constrained cell refinement in reciprocal space.
//!
//! Units and conventions follow [`crate::grains`]: direct bases are
//! row matrices in ångström (one lattice vector per row), reciprocal
//! positions are in inverse ångström without 2π, and a reciprocal position
//! is `q = h B` with `B` the inverse transpose of the direct basis. Integer
//! transforms act on rows: `conventional = transform · input`, so indices
//! transform as `h_conventional = h_input · transformᵀ`.
//!
//! # Bravais-lattice search
//!
//! [`search`](crate::lattice_symmetry::search) reports every one of the 14
//! Bravais types whose metric the lattice fits within
//! [`SearchOptions`](crate::lattice_symmetry::SearchOptions). It is an
//! exhaustive search over small integer combinations of a reduced basis, in
//! the spirit of Le Page (1982, *J. Appl. Cryst.* 15, 255) and the
//! `lattice_symmetry` module of cctbx (Grosse-Kunstleve et al.), but it
//! builds conventional cells directly instead of collecting twofold axes:
//!
//! 1. Selling reduction ([`reduce`](crate::spots::reduction::reduce)), then
//!    Niggli reduction by the algorithm of Křivý & Gruber (1976, *Acta
//!    Cryst.* A32, 297) with the floating-point tolerances of
//!    Grosse-Kunstleve, Sauter & Adams (2004, *Acta Cryst.* A60, 1). The
//!    Niggli cell is unique for a lattice, so what follows does not depend on
//!    the supplied basis. It is the reported aP cell.
//! 2. Candidate axes are the primitive lattice vectors `n · N` with integer
//!    `n` in `[-3, 3]³` of the Niggli basis `N`. Coefficients of 3 are
//!    needed, for example, for the hexagonal c axis of an acute rhombohedron
//!    (α < 60°).
//! 3. Triples of candidate axes in which one axis is perpendicular (within
//!    the angle tolerance) to the other two are the monoclinic, orthogonal and
//!    hexagonal axis systems. The integer matrix of a triple must have a
//!    determinant of 1 to 4. Its centring is read from the lattice itself:
//!    the rows of `transform⁻¹` are the fractional coordinates, in the
//!    conventional cell, of the input lattice's basis vectors, and the group
//!    they generate modulo the conventional lattice must be exactly one of
//!    the standard sets P, C, I, F or R (obverse). A cell that merely has a
//!    higher-symmetry metric, such as a doubled supercell, has a different
//!    coset set and is never reported.
//! 4. A triple is a candidate of a type when its lengths and angles meet the
//!    type's constraints within tolerance and, in addition, every axis that
//!    must be normal to a lattice plane (b for monoclinic, c for hexagonal
//!    axes, all three for orthogonal systems) has an obliquity in Le Page's
//!    sense, the angle between the axis and that plane's normal, within the
//!    angle tolerance. The obliquity does not depend on how the plane's own
//!    basis is chosen; without it, very oblique cells built from long, nearly
//!    parallel vectors can meet the 90° angles of a triclinic lattice by
//!    accident. Of the candidates of one type, the one with the smallest
//!    `a² + b² + c²` wins; equal sums (symmetry-equivalent settings) are
//!    separated by the smaller deviation, then by the cell parameters, then
//!    by the transform.
//!
//! Standard settings, chosen deterministically:
//!
//! | type | setting |
//! |---|---|
//! | aP | Niggli-reduced cell |
//! | mP, mC | unique axis b, β ≥ 90°, α ≥ 90° (sign choice), shortest a² + c²; mP also a ≤ c; mC C-centred |
//! | oP, oI, oF | a ≤ b ≤ c |
//! | oC | C-centred (c normal to the centred face), a ≤ b |
//! | tP, tI | unique axis c, a ≤ b (equal within tolerance) |
//! | hP | unique axis c, γ = 120°, β ≥ 90° (sign choice) |
//! | hR | hexagonal axes, obverse centring (⅔ ⅓ ⅓), γ = 120° |
//! | cP, cI, cF | a ≤ b ≤ c |
//!
//! Orthogonal settings also take α, β ≥ 90° as the sign choice. These are
//! lattice settings; a space group may call for a permutation (orthoenstatite
//! in *Pbca* is reported as 5.18 × 8.82 × 18.23 Å).
//!
//! The metric cannot prove the symmetry. A lattice may fit a higher-symmetry
//! metric by accident (pseudo-symmetry) or because the tolerance is wide;
//! only the intensities can confirm it. All types that fit are therefore
//! reported, highest symmetry first, with their deviations, and the
//! symmetry-constrained fit
//! [`refine_cell`](crate::lattice_symmetry::refine_cell) can compare their
//! residuals.
//!
//! # Constrained refinement
//!
//! [`refine_cell`](crate::lattice_symmetry::refine_cell) fits the orientation
//! and only the free cell parameters of a lattice system to indexed
//! reciprocal positions by least squares on the position residuals, with
//! standard uncertainties from the fit covariance;
//! [`index`](crate::lattice_symmetry::index) supplies the indices from a
//! conventional basis.
//!
//! See `docs/lattice-symmetry.md` for the method, defaults and limits.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::needless_range_loop
)]

use crate::grains::{M3, V3};
use crate::spots::SpotError;
use nalgebra::Matrix3;

mod refinement;
pub use refinement::{CellRefinement, Convergence, RefineOptions, index, refine_cell};

/// Exact integer change of basis acting on rows.
pub type Transform = [[i64; 3]; 3];
type Result<T> = std::result::Result<T, SpotError>;

/// The seven lattice systems, with rhombohedral lattices in hexagonal axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LatticeSystem {
    /// No constraint; six free cell parameters.
    Triclinic,
    /// Unique axis b: α = γ = 90°; a, b, c, β free.
    Monoclinic,
    /// α = β = γ = 90°; a, b, c free.
    Orthorhombic,
    /// a = b, α = β = γ = 90°; a, c free.
    Tetragonal,
    /// Hexagonal axes: a = b, α = β = 90°, γ = 120°; a, c free.
    Rhombohedral,
    /// a = b, α = β = 90°, γ = 120°; a, c free.
    Hexagonal,
    /// a = b = c, α = β = γ = 90°; a free.
    Cubic,
}
impl LatticeSystem {
    /// Number of free cell parameters: 6, 4, 3, 2, 2, 2 or 1.
    #[must_use]
    pub fn free_parameters(self) -> usize {
        match self {
            Self::Triclinic => 6,
            Self::Monoclinic => 4,
            Self::Orthorhombic => 3,
            Self::Tetragonal | Self::Rhombohedral | Self::Hexagonal => 2,
            Self::Cubic => 1,
        }
    }
}

/// Lattice centring of a conventional cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Centring {
    /// Primitive.
    P,
    /// C-face centred, translation (½ ½ 0).
    C,
    /// Body centred, translation (½ ½ ½).
    I,
    /// All-face centred, translations (0 ½ ½), (½ 0 ½), (½ ½ 0).
    F,
    /// Rhombohedral in hexagonal axes, obverse: (⅔ ⅓ ⅓), (⅓ ⅔ ⅔).
    R,
}
impl Centring {
    /// Lattice points per conventional cell: 1, 2, 2, 4 or 3.
    #[must_use]
    pub fn multiplicity(self) -> i64 {
        match self {
            Self::P => 1,
            Self::C | Self::I => 2,
            Self::F => 4,
            Self::R => 3,
        }
    }
    /// Whether conventional indices satisfy the centring's reflection
    /// condition (`h + k` even for C, `h + k + l` even for I, unmixed parity
    /// for F, `-h + k + l` divisible by 3 for obverse R).
    #[must_use]
    pub fn allows(self, hkl: [i64; 3]) -> bool {
        let [h, k, l] = hkl;
        match self {
            Self::P => true,
            Self::C => (h + k).rem_euclid(2) == 0,
            Self::I => (h + k + l).rem_euclid(2) == 0,
            Self::F => (h + k).rem_euclid(2) == 0 && (k + l).rem_euclid(2) == 0,
            Self::R => (-h + k + l).rem_euclid(3) == 0,
        }
    }
    /// Centring translations times the multiplicity, as integer residues.
    fn residues(self) -> &'static [[i64; 3]] {
        match self {
            Self::P => &[[0, 0, 0]],
            Self::C => &[[0, 0, 0], [1, 1, 0]],
            Self::I => &[[0, 0, 0], [1, 1, 1]],
            Self::F => &[[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0]],
            Self::R => &[[0, 0, 0], [1, 2, 2], [2, 1, 1]],
        }
    }
}

/// The 14 Bravais types. Rhombohedral lattices use hexagonal axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bravais {
    /// aP, triclinic.
    TriclinicP,
    /// mP, monoclinic primitive (unique axis b).
    MonoclinicP,
    /// mC, monoclinic C-centred (unique axis b).
    MonoclinicC,
    /// oP, orthorhombic primitive.
    OrthorhombicP,
    /// oC, orthorhombic C-centred.
    OrthorhombicC,
    /// oI, orthorhombic body centred.
    OrthorhombicI,
    /// oF, orthorhombic all-face centred.
    OrthorhombicF,
    /// tP, tetragonal primitive.
    TetragonalP,
    /// tI, tetragonal body centred.
    TetragonalI,
    /// hP, hexagonal primitive.
    HexagonalP,
    /// hR, rhombohedral, obverse in hexagonal axes.
    RhombohedralR,
    /// cP, cubic primitive.
    CubicP,
    /// cI, cubic body centred.
    CubicI,
    /// cF, cubic all-face centred.
    CubicF,
}
impl Bravais {
    /// All 14 types in the order of International Tables (aP … cF).
    pub const ALL: [Self; 14] = [
        Self::TriclinicP,
        Self::MonoclinicP,
        Self::MonoclinicC,
        Self::OrthorhombicP,
        Self::OrthorhombicC,
        Self::OrthorhombicI,
        Self::OrthorhombicF,
        Self::TetragonalP,
        Self::TetragonalI,
        Self::HexagonalP,
        Self::RhombohedralR,
        Self::CubicP,
        Self::CubicI,
        Self::CubicF,
    ];
    /// Pearson-style symbol: `aP`, `mP`, `mC`, `oP`, `oC`, `oI`, `oF`, `tP`,
    /// `tI`, `hP`, `hR`, `cP`, `cI` or `cF`.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        [
            "aP", "mP", "mC", "oP", "oC", "oI", "oF", "tP", "tI", "hP", "hR", "cP", "cI", "cF",
        ][self as usize]
    }
    /// Inverse of [`Bravais::symbol`].
    #[must_use]
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.symbol() == symbol)
    }
    /// Lattice system of the type.
    #[must_use]
    pub fn system(self) -> LatticeSystem {
        match self {
            Self::TriclinicP => LatticeSystem::Triclinic,
            Self::MonoclinicP | Self::MonoclinicC => LatticeSystem::Monoclinic,
            Self::OrthorhombicP
            | Self::OrthorhombicC
            | Self::OrthorhombicI
            | Self::OrthorhombicF => LatticeSystem::Orthorhombic,
            Self::TetragonalP | Self::TetragonalI => LatticeSystem::Tetragonal,
            Self::HexagonalP => LatticeSystem::Hexagonal,
            Self::RhombohedralR => LatticeSystem::Rhombohedral,
            Self::CubicP | Self::CubicI | Self::CubicF => LatticeSystem::Cubic,
        }
    }
    /// Centring of the conventional cell.
    #[must_use]
    pub fn centring(self) -> Centring {
        match self {
            Self::TriclinicP
            | Self::MonoclinicP
            | Self::OrthorhombicP
            | Self::TetragonalP
            | Self::HexagonalP
            | Self::CubicP => Centring::P,
            Self::MonoclinicC | Self::OrthorhombicC => Centring::C,
            Self::OrthorhombicI | Self::TetragonalI | Self::CubicI => Centring::I,
            Self::OrthorhombicF | Self::CubicF => Centring::F,
            Self::RhombohedralR => Centring::R,
        }
    }
    /// Order of the lattice point group (holohedry), inversion included:
    /// 2, 4, 8, 16, 12, 24 or 48.
    #[must_use]
    pub fn holohedry_order(self) -> usize {
        match self.system() {
            LatticeSystem::Triclinic => 2,
            LatticeSystem::Monoclinic => 4,
            LatticeSystem::Orthorhombic => 8,
            LatticeSystem::Tetragonal => 16,
            LatticeSystem::Rhombohedral => 12,
            LatticeSystem::Hexagonal => 24,
            LatticeSystem::Cubic => 48,
        }
    }
    /// Sort key: higher holohedry first, then F, I, C before P within a system.
    fn rank(self) -> (std::cmp::Reverse<usize>, std::cmp::Reverse<usize>) {
        (
            std::cmp::Reverse(self.holohedry_order()),
            std::cmp::Reverse(self as usize),
        )
    }
}

/// Metric tolerances of [`search`]. Non-exhaustive: start from
/// [`SearchOptions::default`] and assign fields.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SearchOptions {
    /// Largest relative mismatch, `(max − min) / mean`, among lengths that
    /// must be equal; default 0.01 (1 %).
    pub length_tolerance: f64,
    /// Largest deviation, degrees, of an angle that must be 90° or 120°, and
    /// largest obliquity of an axis that must be normal to a lattice plane;
    /// default 1°.
    pub angle_tolerance: f64,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            length_tolerance: 0.01,
            angle_tolerance: 1.0,
        }
    }
}
impl SearchOptions {
    /// Rejects non-finite or out-of-range tolerances.
    ///
    /// # Errors
    /// `length_tolerance` must lie in (0, 0.1] and `angle_tolerance` in
    /// (0, 10] degrees.
    pub fn check(&self) -> Result<()> {
        if !(self.length_tolerance > 0. && self.length_tolerance <= 0.1) {
            return Err(SpotError("length_tolerance must lie in (0, 0.1]".into()));
        }
        if !(self.angle_tolerance > 0. && self.angle_tolerance <= 10.) {
            return Err(SpotError(
                "angle_tolerance must lie in (0, 10] degrees".into(),
            ));
        }
        Ok(())
    }
}

/// One Bravais type the lattice fits, in its standard setting.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BravaisMatch {
    /// The Bravais type.
    pub bravais: Bravais,
    /// Integer transform, `conventional = transform · input`. Its determinant
    /// is the centring multiplicity for a right-handed primitive input (the
    /// negative of it for a left-handed one).
    pub transform: Transform,
    /// Conventional direct row basis, ångström, right handed; exactly
    /// `transform · input`.
    pub conventional: M3,
    /// Measured a, b, c (Å) and α, β, γ (degrees) of `conventional`.
    pub cell: [f64; 6],
    /// The cell with the type's constraints imposed: lengths that must be
    /// equal averaged, constrained angles set to 90° or 120°.
    pub constrained_cell: [f64; 6],
    /// Constrained cell in the orientation closest (least squares over the
    /// three rows) to `conventional`; a starting basis for [`refine_cell`].
    pub idealized: M3,
    /// Largest relative mismatch, `(max − min) / mean`, among lengths that
    /// must be equal; 0 when none must be.
    pub length_deviation: f64,
    /// Largest deviation, degrees, of a constrained angle from 90° or 120°
    /// or of a required plane normal's obliquity; 0 for aP.
    pub angle_deviation: f64,
}

pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn det(a: M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}
fn det_i(t: Transform) -> i64 {
    t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1])
        - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0])
        + t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0])
}
fn mul_i(a: Transform, b: Transform) -> Result<Transform> {
    let mut out = [[0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0_i64;
            for k in 0..3 {
                s = a[i][k]
                    .checked_mul(b[k][j])
                    .and_then(|x| s.checked_add(x))
                    .ok_or_else(|| SpotError("integer transform overflow".into()))?;
            }
            out[i][j] = s;
        }
    }
    Ok(out)
}
/// `transform · basis`, rows.
pub(crate) fn apply(t: Transform, a: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| t[i][k] as f64 * a[k][j]).sum()))
}

/// Lengths (Å) and angles α, β, γ (degrees) of a direct row basis.
#[must_use]
pub fn cell_parameters(a: M3) -> [f64; 6] {
    let l = a.map(norm);
    let angle = |i: usize, j: usize| {
        (dot(a[i], a[j]) / (l[i] * l[j]))
            .clamp(-1., 1.)
            .acos()
            .to_degrees()
    };
    [l[0], l[1], l[2], angle(1, 2), angle(0, 2), angle(0, 1)]
}

/// Direct row basis of a cell (Å, degrees) in the standard Cartesian frame:
/// a along x, b in the xy plane, right handed. None for an impossible
/// metric.
#[must_use]
pub fn cell_basis(cell: [f64; 6]) -> Option<M3> {
    let [a, b, c, al, be, ga] = cell;
    let (ca, cb, cg) = (
        al.to_radians().cos(),
        be.to_radians().cos(),
        ga.to_radians().cos(),
    );
    let sg = ga.to_radians().sin();
    let cy = (ca - cb * cg) / sg;
    let cz2 = 1. - cb * cb - cy * cy;
    if a <= 0. || b <= 0. || c <= 0. || sg <= 1e-12 || cz2 <= 1e-12 || !cz2.is_finite() {
        return None;
    }
    Some([
        [a, 0., 0.],
        [b * cg, b * sg, 0.],
        [c * cb, c * cy, c * cz2.sqrt()],
    ])
}

/// Rotation R minimising `Σ |R a0_i − a_i|²` over the rows (Kabsch).
pub(crate) fn best_rotation(a0: M3, a: M3) -> Matrix3<f64> {
    // H = Σ a0_i a_iᵀ; R = V diag(1, 1, d) Uᵀ.
    let h = Matrix3::<f64>::from_fn(|i, j| (0..3).map(|k| a0[k][i] * a[k][j]).sum());
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let v = vt.transpose();
    let d = (v * u.transpose()).determinant().signum();
    v * Matrix3::from_diagonal(&nalgebra::Vector3::new(1., 1., d)) * u.transpose()
}
/// Rows of `a0` rotated by `r`.
pub(crate) fn rotate_rows(a0: M3, r: &Matrix3<f64>) -> M3 {
    a0.map(|v| std::array::from_fn(|i| (0..3).map(|k| r[(i, k)] * v[k]).sum()))
}

/// The measured cell with a lattice system's constraints imposed.
#[must_use]
pub fn constrain_cell(system: LatticeSystem, cell: [f64; 6]) -> [f64; 6] {
    let [a, b, c, _, be, _] = cell;
    match system {
        LatticeSystem::Triclinic => cell,
        LatticeSystem::Monoclinic => [a, b, c, 90., be, 90.],
        LatticeSystem::Orthorhombic => [a, b, c, 90., 90., 90.],
        LatticeSystem::Tetragonal => {
            let m = f64::midpoint(a, b);
            [m, m, c, 90., 90., 90.]
        }
        LatticeSystem::Rhombohedral | LatticeSystem::Hexagonal => {
            let m = f64::midpoint(a, b);
            [m, m, c, 90., 90., 120.]
        }
        LatticeSystem::Cubic => {
            let m = (a + b + c) / 3.;
            [m, m, m, 90., 90., 90.]
        }
    }
}

/// Obliquities (Le Page 1982), degrees: the angle between each direct axis
/// and the normal of the lattice plane spanned by the other two, that is the
/// matching reciprocal axis. Unlike the cell angles it does not depend on how
/// that plane's own basis is chosen.
fn obliquities(v: M3) -> [f64; 3] {
    std::array::from_fn(|i| {
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let n = [
            v[j][1] * v[k][2] - v[j][2] * v[k][1],
            v[j][2] * v[k][0] - v[j][0] * v[k][2],
            v[j][0] * v[k][1] - v[j][1] * v[k][0],
        ];
        let along = dot(v[i], n).abs();
        let x = [
            v[i][1] * n[2] - v[i][2] * n[1],
            v[i][2] * n[0] - v[i][0] * n[2],
            v[i][0] * n[1] - v[i][1] * n[0],
        ];
        norm(x).atan2(along).to_degrees()
    })
}

/// Largest relative length mismatch, `(max − min) / mean` over lengths that
/// must be equal, and the largest angle deviation (degrees) of a measured
/// cell from a lattice system's constraints: of the constrained angles and of
/// the obliquities of the axes that must be normal to a lattice plane (b for
/// monoclinic, c for hexagonal axes, all three for orthogonal systems).
fn deviations(system: LatticeSystem, cell: [f64; 6], oblique: [f64; 3]) -> (f64, f64) {
    let spread = |l: &[f64]| {
        let m = l.iter().sum::<f64>() / l.len() as f64;
        (l.iter().copied().fold(f64::MIN, f64::max) - l.iter().copied().fold(f64::MAX, f64::min))
            / m
    };
    let off = |i: usize, v: f64| (cell[i] - v).abs();
    let orthogonal = off(3, 90.)
        .max(off(4, 90.))
        .max(off(5, 90.))
        .max(oblique[0])
        .max(oblique[1])
        .max(oblique[2]);
    match system {
        LatticeSystem::Triclinic => (0., 0.),
        LatticeSystem::Monoclinic => (0., off(3, 90.).max(off(5, 90.)).max(oblique[1])),
        LatticeSystem::Orthorhombic => (0., orthogonal),
        LatticeSystem::Tetragonal => (spread(&cell[..2]), orthogonal),
        LatticeSystem::Rhombohedral | LatticeSystem::Hexagonal => (
            spread(&cell[..2]),
            off(3, 90.)
                .max(off(4, 90.))
                .max(off(5, 120.))
                .max(oblique[2]),
        ),
        LatticeSystem::Cubic => (spread(&cell[..3]), orthogonal),
    }
}

/// Centring of the conventional cell `transform · primitive`, read from the
/// cosets of the primitive lattice; None for a non-standard coset set.
fn centring(t: Transform) -> Option<Centring> {
    let n = det_i(t);
    if !(1..=4).contains(&n) {
        return None;
    }
    // Adjugate rows: n times the fractional coordinates of the primitive
    // basis vectors in the conventional cell.
    let adj: Transform = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (r1, r2) = ((j + 1) % 3, (j + 2) % 3);
            let (c1, c2) = ((i + 1) % 3, (i + 2) % 3);
            t[r1][c1] * t[r2][c2] - t[r1][c2] * t[r2][c1]
        })
    });
    let mut residues = Vec::with_capacity(64);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let r: [i64; 3] = std::array::from_fn(|j| {
                    (x * adj[0][j] + y * adj[1][j] + z * adj[2][j]).rem_euclid(n)
                });
                if !residues.contains(&r) {
                    residues.push(r);
                }
            }
        }
    }
    residues.sort_unstable();
    [
        Centring::P,
        Centring::C,
        Centring::I,
        Centring::F,
        Centring::R,
    ]
    .into_iter()
    .find(|c| c.multiplicity() == n && residues == c.residues())
}

/// Niggli reduction (Křivý & Gruber 1976; tolerances after Grosse-Kunstleve,
/// Sauter & Adams 2004). Returns the reduced basis and the unimodular
/// transform with `reduced = transform · basis`; the input should already be
/// short (Selling reduced) so the integer coefficients stay small.
///
/// # Errors
/// Nonconvergence within 1000 steps or integer overflow.
pub fn niggli(basis: M3) -> Result<(M3, Transform)> {
    let volume = det(basis).abs();
    if !volume.is_finite() || volume <= 0. {
        return Err(SpotError("singular lattice basis".into()));
    }
    let e = 1e-7 * volume.powf(2. / 3.);
    let mut t: Transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let mut a = basis;
    let step = |s: Transform, t: &mut Transform, a: &mut M3| -> Result<()> {
        *t = mul_i(s, *t)?;
        *a = apply(s, *a);
        if t.iter().flatten().any(|x| x.abs() > 1_000_000) {
            return Err(SpotError(
                "Niggli transform coefficient budget exceeded".into(),
            ));
        }
        Ok(())
    };
    let sign = |x: f64| if x > 0. { 1 } else { -1 };
    for _ in 0..1000 {
        let g = |a: &M3| {
            (
                dot(a[0], a[0]),
                dot(a[1], a[1]),
                dot(a[2], a[2]),
                2. * dot(a[1], a[2]),
                2. * dot(a[0], a[2]),
                2. * dot(a[0], a[1]),
            )
        };
        // A1
        let (aa, bb, _, xi, eta, _) = g(&a);
        if aa > bb + e || ((aa - bb).abs() <= e && xi.abs() > eta.abs() + e) {
            step([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], &mut t, &mut a)?;
        }
        // A2
        let (_, bb, cc, _, eta, zeta) = g(&a);
        if bb > cc + e || ((bb - cc).abs() <= e && eta.abs() > zeta.abs() + e) {
            step([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], &mut t, &mut a)?;
            continue;
        }
        // A3 / A4: all products positive, or all non-positive.
        let (_, _, _, xi, eta, zeta) = g(&a);
        let s = |x: f64| i64::from(x > e) - i64::from(x < -e);
        let (l, m, n) = (s(xi), s(eta), s(zeta));
        let flips = if l * m * n == 1 {
            [l, m, n].map(|v| if v < 0 { -1 } else { 1 })
        } else {
            let mut f = [1, 1, 1];
            let mut zero = None;
            for (k, v) in [l, m, n].into_iter().enumerate() {
                if v == 1 {
                    f[k] = -1;
                } else if v == 0 {
                    zero = Some(k);
                }
            }
            if let Some(k) = zero.filter(|_| f[0] * f[1] * f[2] < 0) {
                f[k] = -1;
            }
            f
        };
        if flips != [1, 1, 1] {
            step(
                [[flips[0], 0, 0], [0, flips[1], 0], [0, 0, flips[2]]],
                &mut t,
                &mut a,
            )?;
        }
        let (aa, bb, _, xi, eta, zeta) = g(&a);
        // A5
        if xi.abs() > bb + e
            || ((xi - bb).abs() <= e && 2. * eta < zeta - e)
            || ((xi + bb).abs() <= e && zeta < -e)
        {
            step([[1, 0, 0], [0, 1, 0], [0, -sign(xi), 1]], &mut t, &mut a)?;
            continue;
        }
        // A6
        if eta.abs() > aa + e
            || ((eta - aa).abs() <= e && 2. * xi < zeta - e)
            || ((eta + aa).abs() <= e && zeta < -e)
        {
            step([[1, 0, 0], [0, 1, 0], [-sign(eta), 0, 1]], &mut t, &mut a)?;
            continue;
        }
        // A7
        if zeta.abs() > aa + e
            || ((zeta - aa).abs() <= e && 2. * xi < eta - e)
            || ((zeta + aa).abs() <= e && eta < -e)
        {
            step([[1, 0, 0], [-sign(zeta), 1, 0], [0, 0, 1]], &mut t, &mut a)?;
            continue;
        }
        // A8
        let sum = xi + eta + zeta + aa + bb;
        if sum < -e || (sum.abs() <= e && 2. * (aa + eta) + zeta > e) {
            step([[1, 0, 0], [0, 1, 0], [1, 1, 1]], &mut t, &mut a)?;
            continue;
        }
        return Ok((a, t));
    }
    Err(SpotError("Niggli reduction did not converge".into()))
}

/// A candidate setting of one type and its selection key.
struct Candidate {
    transform: Transform,
    conventional: M3,
    cell: [f64; 6],
    length_deviation: f64,
    angle_deviation: f64,
    sum: f64,
    score: f64,
}
impl Candidate {
    /// Shorter cell first; symmetry-equivalent (equal-sum) settings by the
    /// smaller deviation, the cell, then the transform.
    fn better_than(&self, other: &Self) -> bool {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let close = |x: f64, y: f64, rel: f64| (x - y).abs() <= rel * x.abs().max(y.abs());
        let by = |x: f64, y: f64, rel: f64| {
            if close(x, y, rel) {
                Equal
            } else if x < y {
                Less
            } else {
                Greater
            }
        };
        let mut order = by(self.sum, other.sum, 1e-9).then(by(self.score, other.score, 1e-9));
        for k in 0..6 {
            order = order.then(by(self.cell[k], other.cell[k], 1e-12));
        }
        order.then(self.transform.cmp(&other.transform)) == Less
    }
}

/// Every Bravais type whose metric the lattice of `basis` fits within
/// `options`, highest symmetry first, aP always last; one standard setting
/// per type (see the [module documentation](self)).
///
/// # Errors
/// Invalid options; a non-finite, singular or oversized (|x| > 10⁶ Å) basis;
/// reduction failure.
pub fn search(basis: M3, options: &SearchOptions) -> Result<Vec<BravaisMatch>> {
    options.check()?;
    if basis
        .iter()
        .flatten()
        .any(|x| !x.is_finite() || x.abs() > 1e6)
        || det(basis).abs() < 1e-9
    {
        return Err(SpotError("invalid lattice basis".into()));
    }
    let selling = crate::spots::reduction::reduce(basis, 1e-12, 10_000)?;
    let (_, niggli_t) = niggli(selling.basis)?;
    let to_niggli = mul_i(niggli_t, selling.transform)?;
    let reduced = apply(to_niggli, basis);

    // Primitive candidate axes, one sign each.
    let mut coefficients = Vec::new();
    for x in -3_i64..=3 {
        for y in -3_i64..=3 {
            for z in -3_i64..=3 {
                let n = [x, y, z];
                let first = n.iter().copied().find(|&v| v != 0);
                let g = gcd(gcd(x, y), z);
                if first.is_some_and(|f| f > 0) && g == 1 {
                    coefficients.push(n);
                }
            }
        }
    }
    let vectors: Vec<V3> = coefficients
        .iter()
        .map(|n| std::array::from_fn(|j| (0..3).map(|k| n[k] as f64 * reduced[k][j]).sum()))
        .collect();
    let lengths: Vec<f64> = vectors.iter().map(|&v| norm(v)).collect();
    let m = vectors.len();
    let sin_tol = options.angle_tolerance.to_radians().sin();
    let perpendicular: Vec<Vec<usize>> = (0..m)
        .map(|u| {
            (0..m)
                .filter(|&v| {
                    v != u && dot(vectors[u], vectors[v]).abs() <= sin_tol * lengths[u] * lengths[v]
                })
                .collect()
        })
        .collect();

    let mut best: Vec<Option<Candidate>> = (0..14).map(|_| None).collect();
    let mut consider = |rows: [usize; 3], hexagonal: bool| {
        let t0: Transform = rows.map(|r| coefficients[r]);
        let d0 = det_i(t0);
        if !(1..=4).contains(&d0.abs()) {
            return;
        }
        for pattern in [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]] {
            let s = pattern.map(|p| p * d0.signum());
            let t: Transform = std::array::from_fn(|i| t0[i].map(|x| x * s[i]));
            let Some(centre) = centring(t) else {
                continue;
            };
            let v: M3 = std::array::from_fn(|i| vectors[rows[i]].map(|x| x * s[i] as f64));
            let (ca, cb, cg) = (dot(v[1], v[2]), dot(v[0], v[2]), dot(v[0], v[1]));
            let cell = cell_parameters(v);
            let (la, lb, lc) = (cell[0], cell[1], cell[2]);
            let mut types: Vec<Bravais> = Vec::new();
            if hexagonal {
                if cg < 0. && la <= lb {
                    match centre {
                        Centring::P if cb <= 0. => types.push(Bravais::HexagonalP),
                        Centring::R => types.push(Bravais::RhombohedralR),
                        _ => {}
                    }
                }
            } else if ca <= 0. && cb <= 0. {
                match centre {
                    Centring::P if la <= lc => types.push(Bravais::MonoclinicP),
                    Centring::C => types.push(Bravais::MonoclinicC),
                    _ => {}
                }
                if cb.abs() <= sin_tol * la * lc {
                    let ordered = la <= lb && lb <= lc;
                    match centre {
                        Centring::P | Centring::I | Centring::F if ordered => {
                            types.push(match centre {
                                Centring::P => Bravais::OrthorhombicP,
                                Centring::I => Bravais::OrthorhombicI,
                                _ => Bravais::OrthorhombicF,
                            });
                            types.push(match centre {
                                Centring::P => Bravais::CubicP,
                                Centring::I => Bravais::CubicI,
                                _ => Bravais::CubicF,
                            });
                        }
                        Centring::C if la <= lb => types.push(Bravais::OrthorhombicC),
                        _ => {}
                    }
                    if la <= lb {
                        match centre {
                            Centring::P => types.push(Bravais::TetragonalP),
                            Centring::I => types.push(Bravais::TetragonalI),
                            _ => {}
                        }
                    }
                }
            }
            let oblique = if types.is_empty() {
                [0.; 3]
            } else {
                obliquities(v)
            };
            for bravais in types {
                let (length_deviation, angle_deviation) =
                    deviations(bravais.system(), cell, oblique);
                if length_deviation > options.length_tolerance
                    || angle_deviation > options.angle_tolerance
                {
                    continue;
                }
                let candidate = Candidate {
                    transform: t,
                    conventional: v,
                    cell,
                    length_deviation,
                    angle_deviation,
                    sum: la * la + lb * lb + lc * lc,
                    score: (length_deviation / options.length_tolerance)
                        .max(angle_deviation / options.angle_tolerance),
                };
                let slot = &mut best[bravais as usize];
                if slot.as_ref().is_none_or(|b| candidate.better_than(b)) {
                    *slot = Some(candidate);
                }
            }
        }
    };
    for u in 0..m {
        for &v in &perpendicular[u] {
            for &w in &perpendicular[u] {
                if v != w {
                    consider([v, u, w], false);
                    consider([v, w, u], true);
                }
            }
        }
    }

    let mut out = Vec::new();
    let mut push = |bravais: Bravais, to_reduced: Transform, c: Option<&Candidate>| -> Result<()> {
        let transform = mul_i(to_reduced, to_niggli)?;
        let conventional = apply(transform, basis);
        let cell = cell_parameters(conventional);
        let (length_deviation, angle_deviation) =
            c.map_or((0., 0.), |c| (c.length_deviation, c.angle_deviation));
        let constrained_cell = constrain_cell(bravais.system(), cell);
        let idealized = cell_basis(constrained_cell).map_or(conventional, |a0| {
            rotate_rows(a0, &best_rotation(a0, conventional))
        });
        out.push(BravaisMatch {
            bravais,
            transform,
            conventional,
            cell,
            constrained_cell,
            idealized,
            length_deviation,
            angle_deviation,
        });
        Ok(())
    };
    for bravais in Bravais::ALL {
        if let Some(c) = &best[bravais as usize] {
            debug_assert!(c.conventional.iter().flatten().all(|x| x.is_finite()));
            push(bravais, c.transform, Some(c))?;
        }
    }
    push(Bravais::TriclinicP, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], None)?;
    out.sort_by_key(|m| m.bravais.rank());
    Ok(out)
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
#[path = "../lattice_symmetry_tests.rs"]
mod tests;
