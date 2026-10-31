//! BSD-3-Clause safeguarded cubic/quadratic line-search interpolation.
//! Vendored from lbfgsb-rs-pure 0.1.2, src/linesearch.rs::dcstep.
//! The upstream license and exact provenance are in licenses/optimizers/.
//! No optimizer driver or upstream model code is included.
#![allow(clippy::all, clippy::pedantic)]

#[allow(clippy::too_many_arguments)]
pub fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    // compute sign of derivative relative to dx
    let sgnd = dp * (*dx / dx.abs());

    // temporaries
    let stpc: f64;
    let mut stpf: f64;
    let stpq: f64;

    // Case analysis follows standard dcstep logic.
    if fp > *fx {
        // Case 1: higher function value. The minimum is between stx and stp.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max((*dx).abs()).max(dp.abs());
        let temp = (theta / s) * (theta / s) - (*dx) / s * (dp / s);
        let gamma = s * temp.max(0.0).sqrt();
        let gamma = if *stp < *stx { -gamma } else { gamma };
        let p = gamma - *dx + theta;
        let q = gamma - *dx + gamma + dp;
        let r = if q != 0.0 { p / q } else { 0.0 };
        stpc = *stx + r * (*stp - *stx);
        stpq = *stx + *dx / (((*fx - fp) / (*stp - *stx)) + *dx) / 2.0 * (*stp - *stx);
        stpf = if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpc
        } else {
            stpc + (stpq - stpc) / 2.0
        };
        *brackt = true;
    } else if sgnd < 0.0 {
        // Case 2: lower function value and derivatives of opposite sign -> bracketed
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max((*dx).abs()).max(dp.abs());
        let temp = (theta / s) * (theta / s) - (*dx) / s * (dp / s);
        let gamma = s * temp.max(0.0).sqrt();
        let gamma = if *stp > *stx { -gamma } else { gamma };
        let p = gamma - dp + theta;
        let q = gamma - dp + gamma + *dx;
        let r = if q != 0.0 { p / q } else { 0.0 };
        stpc = *stp + r * (*stx - *stp);
        stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        stpf = if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpc
        } else {
            stpq
        };
        *brackt = true;
    } else if dp.abs() < (*dx).abs() {
        // Case 3: lower function value, same sign, magnitude of derivative decreased.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max((*dx).abs()).max(dp.abs());
        let tmp = (theta / s) * (theta / s) - (*dx) / s * (dp / s);
        let gamma = s * tmp.max(0.0).sqrt();
        let gamma = if *stp > *stx { -gamma } else { gamma };
        let p = gamma - dp + theta;
        let q = gamma + (*dx - dp) + gamma;
        let r = if q != 0.0 { p / q } else { 0.0 };
        if r < 0.0 && gamma != 0.0 {
            stpc = *stp + r * (*stx - *stp);
        } else if *stp > *stx {
            stpc = stpmax;
        } else {
            stpc = stpmin;
        }
        stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if *brackt {
            stpf = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            if *stp > *stx {
                stpf = stpf.min(*stp + (*sty - *stp) * 0.66);
            } else {
                stpf = stpf.max(*stp + (*sty - *stp) * 0.66);
            }
        } else {
            stpf = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            stpf = stpf.min(stpmax).max(stpmin);
        }
    } else {
        // Case 4: lower function value, same sign, magnitude does not decrease.
        if *brackt {
            let theta = (fp - *fy) * 3.0 / (*sty - *stp) + *dy + dp;
            let s = theta.abs().max((*dy).abs()).max(dp.abs());
            let temp = (theta / s) * (theta / s) - *dy / s * (dp / s);
            let gamma = s * temp.max(0.0).sqrt();
            let gamma = if *stp > *sty { -gamma } else { gamma };
            let p = gamma - dp + theta;
            let q = gamma - dp + gamma + *dy;
            let r = if q != 0.0 { p / q } else { 0.0 };
            stpc = *stp + r * (*sty - *stp);
            stpf = stpc;
        } else if *stp > *stx {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }

    // Update the interval containing a minimizer.
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }

    // Compute the new trial step.
    *stp = stpf;
}
