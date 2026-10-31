use super::{Error, Result};
use nalgebra::{DMatrix, Matrix3};
/// Three Cartesian components in the units declared by the owning record.
pub type V3 = [f64; 3];
/// Row-major 3-by-3 matrix with explicit basis convention.
pub type M3 = [V3; 3];
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn unit(a: V3) -> V3 {
    let n = norm(a);
    a.map(|x| x / n)
}
pub fn matrix(a: M3) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| a[i][j])
}
pub fn array(a: Matrix3<f64>) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)]))
}
pub fn inv_t(a: M3) -> Result<M3> {
    matrix(a)
        .try_inverse()
        .map(|x| array(x.transpose()))
        .ok_or_else(|| Error("singular basis".into()))
}
pub fn det(a: M3) -> f64 {
    matrix(a).determinant()
}
pub fn mul(a: M3, b: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
pub fn row(v: V3, a: M3) -> V3 {
    std::array::from_fn(|j| (0..3).map(|k| v[k] * a[k][j]).sum())
}
pub fn least(h: &[V3], q: &[V3]) -> Result<M3> {
    let a = DMatrix::from_fn(h.len(), 3, |i, j| h[i][j]);
    let b = DMatrix::from_fn(q.len(), 3, |i, j| q[i][j]);
    let svd = a.svd(true, true);
    let max = svd.singular_values.max();
    if svd
        .singular_values
        .iter()
        .filter(|&&x| x > max * (h.len() as f64) * f64::EPSILON)
        .count()
        < 3
    {
        return Err(Error("rank-deficient indexed positions".into()));
    }
    let x = svd
        .solve(&b, max * (h.len() as f64) * f64::EPSILON)
        .map_err(|e| Error(e.into()))?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| x[(i, j)])))
}
pub fn condition(a: M3) -> f64 {
    let s = matrix(a).svd(false, false).singular_values;
    s.max() / s.min()
}
pub fn median(a: &[f64]) -> f64 {
    let mut x = a.to_vec();
    x.sort_by(f64::total_cmp);
    let n = x.len();
    if n.is_multiple_of(2) {
        f64::midpoint(x[n / 2 - 1], x[n / 2])
    } else {
        x[n / 2]
    }
}
