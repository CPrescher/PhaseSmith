use super::map_frames;
use super::{
    Error, FrameSource, Indexed, Options, Result, bounds, grains, image,
    math::{M3, V3, cross, dot, inv_t, norm, row, unit},
};
use phasesmith_spots::{self as spots, Bounds, frame_input, prediction, profiles};
/// Owned contiguous profile samples and their exact finite group supports.
pub struct ProfileData {
    /// Signed signal, independent variance and fitted background per pixel.
    pub samples: Vec<[f64; 3]>,
    /// Availability of each retained original pixel sample.
    pub valid: Vec<bool>,
    /// Pixel-group offsets plus a final sentinel.
    pub offsets: Vec<usize>,
    /// Half-open detector-pixel rectangles.
    pub boxes: Vec<Bounds>,
    /// Reflection ID of each source-ordered exposure group.
    pub ids: Vec<usize>,
    /// Selected original-frame index of each exposure group.
    pub frames: Vec<usize>,
    /// Increasing full exposure bounds in degrees.
    pub angles: Vec<[f64; 2]>,
    /// Row-major 3-by-3 fitted background-plane covariance per group.
    pub covariance: Vec<[f64; 9]>,
    /// Frozen x/y/omega reflection centres, in pixels/degrees.
    pub centres: Vec<V3>,
    /// Signed local coordinate scales, with rocking zeta in the third component.
    pub scales: Vec<V3>,
    /// Row-major local angular pixel Jacobian for each reflection.
    pub planar: Vec<[f64; 4]>,
    /// Original quality bits; source rows are retained even when excluded.
    pub flags: Vec<u32>,
    /// Per-group flags including the parent reflection angular-truncation bit.
    pub native_flags: Vec<u32>,
    /// Unchanged eleven-column original aperture measurements per exposure.
    pub frame_values: Vec<[f64; 11]>,
    /// Twelve-column guarded-plane fit diagnostics per exposure group.
    pub background: Vec<[f64; 12]>,
}
impl ProfileData {
    fn empty(centres: Vec<V3>) -> Self {
        Self {
            samples: vec![],
            valid: vec![],
            offsets: vec![0],
            boxes: vec![],
            ids: vec![],
            frames: vec![],
            angles: vec![],
            covariance: vec![],
            centres,
            scales: vec![],
            planar: vec![],
            flags: vec![],
            native_flags: vec![],
            frame_values: vec![],
            background: vec![],
        }
    }
    fn inherit_angular_flags(&mut self, reflection_flags: &[u32]) -> Result<()> {
        if reflection_flags.len() != self.centres.len()
            || self.ids.iter().any(|&i| i >= reflection_flags.len())
        {
            return Err(Error("reflection/group flag dimensions differ".into()));
        }
        self.native_flags = self
            .flags
            .iter()
            .zip(&self.ids)
            .map(|(&f, &id)| f | (reflection_flags[id] & 256))
            .collect();
        Ok(())
    }
    /// Borrow the same ordered arrays without copying or changing group quality.
    #[must_use]
    pub fn borrowed(&self) -> profiles::Observations<'_> {
        profiles::Observations {
            samples: &self.samples,
            valid: &self.valid,
            offsets: &self.offsets,
            boxes: &self.boxes,
            ids: &self.ids,
            angles: &self.angles,
            covariance: &self.covariance,
            centres: &self.centres,
            scales: &self.scales,
            planar: &self.planar,
            flags: &self.native_flags,
        }
    }
}
/// Predictions, original shoeboxes, unchanged aperture sums, and learned profile bank.
pub struct Measurements {
    /// Detector height and width, in pixels.
    pub detector_shape: [usize; 2],
    /// Owned original pixel observations and finite supports.
    pub data: ProfileData,
    /// Integer Miller indices in the selected supplied setting.
    pub hkl: Vec<[i32; 3]>,
    /// Unchanged six-column summed aperture baseline per reflection.
    pub aperture: Vec<[f64; 6]>,
    /// Original quality bits; source rows are retained even when excluded.
    pub flags: Vec<u32>,
    /// Learned finite profile bank with independent teacher membership.
    pub bank: profiles::Learned,
    /// Zero-based local profile region before fallback.
    pub regions: Vec<usize>,
    /// Final profile bank index, including global fallback zero.
    pub assignments: Vec<usize>,
    /// Target predictions before pre-flagged edge-centre exclusion.
    pub target_count: usize,
    /// All target and measured competing crossings.
    pub all_count: usize,
    /// Retained indices of pre-flagged centres outside the scaling mesh.
    pub excluded_hkl: Vec<[i32; 3]>,
    /// Retained pre-flagged centres outside the scaling mesh.
    pub excluded_centres: Vec<V3>,
    /// Unchanged aperture rows for excluded edge centres.
    pub excluded_values: Vec<[f64; 6]>,
    /// Original quality flags for excluded edge centres.
    pub excluded_flags: Vec<u32>,
    /// Finite local profile voxel dimensions and widths.
    pub grid: profiles::Grid,
    /// Centre eligibility from the same shared original-image mask.
    pub visible: Vec<bool>,
}
/// Ray, local angular pixel transform and signed rocking zeta in one declared frame.
pub fn profile_geometry(xy: [f64; 2], g: spots::reciprocal::Geometry) -> (V3, [f64; 4], f64) {
    let beam = unit(g.beam);
    let axis = unit(g.axis);
    let p = std::array::from_fn(|i| g.origin[i] + xy[0] * g.fast[i] + xy[1] * g.slow[i]);
    let length = norm(p);
    let ray = p.map(|v| v / length);
    let e = cross(ray, beam);
    let n = norm(e);
    let zeta = dot(e, axis) / n;
    let e1 = e.map(|v| v / n);
    let e2 = cross(ray, e1);
    let scale = 180. / std::f64::consts::PI / length;
    (
        ray,
        [
            dot(e1, g.fast) * scale,
            dot(e1, g.slow) * scale,
            dot(e2, g.fast) * scale,
            dot(e2, g.slow) * scale,
        ],
        zeta,
    )
}
fn predictions(
    b: M3,
    g: spots::reciprocal::Geometry,
    angles: &[[f64; 2]],
    shape: [usize; 2],
) -> Result<(Vec<[i64; 3]>, Vec<V3>)> {
    let a = inv_t(b)?;
    let limits = a.map(|a| (2. * norm(a)).ceil() as i64);
    if limits.iter().map(|n| 2 * n + 1).product::<i64>() > 250_000 {
        return Err(Error("HKL enumeration budget exceeded".into()));
    }
    let mut h = Vec::new();
    for x in -limits[0]..=limits[0] {
        for y in -limits[1]..=limits[1] {
            for z in -limits[2]..=limits[2] {
                let hkl = [x, y, z];
                let n = norm(row(hkl.map(|x| x as f64), b));
                if n > 0. && n <= 2. {
                    h.push(hkl);
                }
            }
        }
    }
    let lo = angles[0][0];
    let hi = angles.last().unwrap()[1];
    let hints = vec![f64::midpoint(lo, hi); h.len()];
    let mut records = Vec::new();
    for other in [false, true] {
        let p = prediction::predict(g, b, &h, &hints, other)?;
        for (h, p) in h.iter().zip(p) {
            let v = p.values;
            if p.flags != 0
                || v[2] < lo
                || v[2] >= hi
                || v[0] < -0.5
                || v[0] >= shape[1] as f64 - 0.5
                || v[1] < -0.5
                || v[1] >= shape[0] as f64 - 0.5
            {
                continue;
            }
            for (frame, a) in angles.iter().enumerate() {
                if v[2] >= a[0] && v[2] < a[1] {
                    records.push((frame, v[2], *h, v));
                }
            }
        }
    }
    records.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
    Ok((
        records.iter().map(|r| r.2).collect(),
        records.iter().map(|r| r.3).collect(),
    ))
}
pub(super) fn measure(
    source: &mut impl FrameSource,
    plan: &frame_input::Plan,
    o: &Options,
    indexed: &Indexed,
) -> Result<Measurements> {
    let angles = bounds(o);
    let g = indexed.fit.state.geometry;
    let (hs, cs) = predictions(indexed.fit.state.basis, g, &angles, plan.shape)?;
    let mut hkl = Vec::new();
    let mut centres = Vec::new();
    for (h, c) in hs.into_iter().zip(cs) {
        if !o
            .priors
            .group
            .as_ref()
            .map(|group| group.is_systematically_absent(h.map(|x| x as i32)))
            .transpose()
            .map_err(|e| Error(e.to_string()))?
            .unwrap_or(false)
            && profile_geometry([c[0], c[1]], g).2.abs() >= 0.1
        {
            hkl.push(h.map(|x| x as i32));
            centres.push(c);
        }
    }
    let target_count = hkl.len();
    for competitor in &indexed.competitors {
        for (&h, &c) in competitor.hkl.iter().zip(&competitor.crossings) {
            if profile_geometry([c[0], c[1]], g).2.abs() >= 0.1 {
                hkl.push(h.map(|x| x as i32));
                centres.push(c);
            }
        }
    }
    let all_count = hkl.len();
    eprintln!("native integrate {target_count} target / {all_count} all crossings");
    let widths: Vec<_> = centres
        .iter()
        .map(|c| (0.9 / profile_geometry([c[0], c[1]], g).2.abs() + 0.5).min(4.5))
        .collect();
    let d = collect(source, plan, o, &angles, &centres, &widths)?;
    let (aperture, mut quality) =
        spots::integration::accumulate(centres.len(), &d.ids, &d.frame_values, &d.flags)?;
    for i in 0..quality.len() {
        if centres[i][2] - widths[i] < angles[0][0] - 1e-8
            || centres[i][2] + widths[i] > angles.last().unwrap()[1] + 1e-8
        {
            quality[i] |= 256;
        }
    }
    let keep: Vec<_> = (0..target_count)
        .filter(|&i| {
            centres[i][0] >= 0.
                && centres[i][0] <= plan.shape[1] as f64 - 1.
                && centres[i][1] >= 0.
                && centres[i][1] <= plan.shape[0] as f64 - 1.
        })
        .collect();
    let excluded: Vec<_> = (0..target_count).filter(|i| !keep.contains(i)).collect();
    if excluded.iter().any(|&i| quality[i] & spots::EDGE == 0) {
        return Err(Error("unflagged centre outside scaling mesh".into()));
    }
    let mut selected = select_groups(d, &keep);
    let visible = geometry_and_visibility(&mut selected, plan, g, &angles);
    let h: Vec<_> = keep.iter().map(|&i| hkl[i]).collect();
    let flags: Vec<_> = keep.iter().map(|&i| quality[i]).collect();
    let ProfileLearning {
        bank,
        regions,
        assignments,
        grid,
    } = learn(&mut selected, &h, &flags, plan, o, &angles)?;
    Ok(Measurements {
        detector_shape: plan.shape,
        data: selected,
        hkl: h,
        aperture: keep.iter().map(|&i| aperture[i]).collect(),
        flags,
        bank,
        regions,
        assignments,
        target_count,
        all_count,
        excluded_hkl: excluded.iter().map(|&i| hkl[i]).collect(),
        excluded_centres: excluded.iter().map(|&i| centres[i]).collect(),
        excluded_values: excluded.iter().map(|&i| aperture[i]).collect(),
        excluded_flags: excluded.iter().map(|&i| quality[i]).collect(),
        grid,
        visible,
    })
}

fn collect(
    source: &mut impl FrameSource,
    plan: &frame_input::Plan,
    o: &Options,
    angles: &[[f64; 2]],
    centres: &[V3],
    widths: &[f64],
) -> Result<ProfileData> {
    let mut d = ProfileData::empty(centres.to_vec());
    map_frames(
        source,
        plan,
        o,
        o.frame_workers,
        |p| {
            let frame = p.index;
            let data = image(&p, plan.shape);
            let ids: Vec<_> = (0..centres.len())
                .filter(|&i| {
                    centres[i][2] - widths[i] < angles[frame][1]
                        && centres[i][2] + widths[i] > angles[frame][0]
                })
                .collect();
            let boxes: Vec<_> = ids
                .iter()
                .map(|&i| {
                    let x = (centres[i][0] + 0.5).floor() as usize;
                    let y = (centres[i][1] + 0.5).floor() as usize;
                    Bounds {
                        x0: x.saturating_sub(7),
                        y0: y.saturating_sub(7),
                        x1: (x + 8).min(plan.shape[1]),
                        y1: (y + 8).min(plan.shape[0]),
                    }
                })
                .collect();
            let row = profiles::extract_frame(
                &spots::Image {
                    values: &data,
                    height: plan.shape[0],
                    width: plan.shape[1],
                    mask: &p.excluded,
                    variance: None,
                    gain: 1.,
                    read_noise: 0.,
                    saturation: None,
                },
                &boxes,
                3,
                6,
                20,
                4.,
            )?;
            Ok((ids, boxes, row))
        },
        |frame, (ids, boxes, (spots, bg, cov, samples, valid))| {
            if d.samples.len() + samples.len() > o.max_samples
                || d.ids.len() + ids.len() > 1_000_000
            {
                return Err(Error("profile sample/group budget exceeded".into()));
            }
            for (i, (&id, &b)) in ids.iter().zip(&boxes).enumerate() {
                d.ids.push(id);
                d.frames.push(frame);
                d.boxes.push(b);
                d.angles.push(angles[frame]);
                d.offsets
                    .push(d.offsets.last().unwrap() + (b.x1 - b.x0) * (b.y1 - b.y0));
                d.covariance.push(cov[i]);
                d.flags.push(spots[i].flags & !64);
                d.frame_values.push(spots[i].values);
                d.background.push(bg[i]);
            }
            d.samples.extend(samples);
            d.valid.extend(valid);
            Ok(())
        },
    )?;
    Ok(d)
}

fn select_groups(d: ProfileData, keep: &[usize]) -> ProfileData {
    let mut remap = vec![None; d.centres.len()];
    for (id, &old) in keep.iter().enumerate() {
        remap[old] = Some(id);
    }
    let mut selected = ProfileData::empty(keep.iter().map(|&i| d.centres[i]).collect());
    for k in 0..d.ids.len() {
        if let Some(id) = remap[d.ids[k]] {
            let range = d.offsets[k]..d.offsets[k + 1];
            selected
                .samples
                .extend_from_slice(&d.samples[range.clone()]);
            selected.valid.extend_from_slice(&d.valid[range]);
            selected.offsets.push(selected.samples.len());
            selected.boxes.push(d.boxes[k]);
            selected.ids.push(id);
            selected.frames.push(d.frames[k]);
            selected.angles.push(d.angles[k]);
            selected.covariance.push(d.covariance[k]);
            selected.flags.push(d.flags[k]);
            selected.frame_values.push(d.frame_values[k]);
            selected.background.push(d.background[k]);
        }
    }
    drop(d);
    selected
}

fn geometry_and_visibility(
    selected: &mut ProfileData,
    plan: &frame_input::Plan,
    g: spots::reciprocal::Geometry,
    angles: &[[f64; 2]],
) -> Vec<bool> {
    let mut visible = Vec::new();
    for c in &selected.centres {
        let (_, a, zeta) = profile_geometry([c[0], c[1]], g);
        selected.planar.push(a);
        selected.scales.push([1., 1., zeta]);
        let nearest = angles
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (c[2] - f64::midpoint(a.1[0], a.1[1]))
                    .abs()
                    .total_cmp(&(c[2] - f64::midpoint(b.1[0], b.1[1])).abs())
            })
            .unwrap()
            .0;
        let x = (c[0] + 0.5).floor() as usize;
        let y = (c[1] + 0.5).floor() as usize;
        let mut bad = false;
        for mask in &plan.blocks {
            let cw = plan.shape[1].div_ceil(mask.block);
            let ch = plan.shape[0].div_ceil(mask.block);
            bad |= mask.excluded[nearest * ch * cw + (y / mask.block) * cw + x / mask.block];
        }
        visible.push(!bad);
    }
    visible
}

struct ProfileLearning {
    bank: profiles::Learned,
    regions: Vec<usize>,
    assignments: Vec<usize>,
    grid: profiles::Grid,
}
fn learn(
    selected: &mut ProfileData,
    hkl: &[[i32; 3]],
    flags: &[u32],
    plan: &frame_input::Plan,
    o: &Options,
    angles: &[[f64; 2]],
) -> Result<ProfileLearning> {
    let partition = o.partition_group();
    let refs: Vec<_> = hkl
        .iter()
        .zip(flags.iter())
        .map(|(&h, &f)| grains::fold(grains::family(h.map(i64::from), &partition)) != 0 && f == 0)
        .collect();
    let lo = angles[0][0];
    let hi = angles.last().unwrap()[1];
    let regions: Vec<_> = selected
        .centres
        .iter()
        .map(|c| {
            let x = ((c[0] / plan.shape[1] as f64 * 2.).floor() as usize).min(1);
            let y = ((c[1] / plan.shape[0] as f64 * 2.).floor() as usize).min(1);
            let z = (((c[2] - lo) / (hi - lo) * 2.).floor() as usize).min(1);
            (z * 2 + y) * 2 + x
        })
        .collect();
    let grid = profiles::Grid {
        shape: [31, 31, 23],
        step: [0.0125, 0.0125, 0.05],
    };
    selected.inherit_angular_flags(flags)?;
    let bank = profiles::learn_with_execution(
        &selected.borrowed(),
        &regions,
        &refs,
        grid,
        9,
        8,
        20.,
        0.,
        100,
        false,
        profiles::Execution {
            workers: o.learning_workers,
            geometry_cache_bytes: o.geometry_cache_bytes,
        },
    )?;
    if bank.used.iter().filter(|&&v| v).count() < 40 {
        return Err(Error("insufficient independent profile teachers".into()));
    }
    let assignments = regions
        .iter()
        .map(|&r| {
            if bank.counts[r + 1] >= 8
                && bank.profiles[(r + 1) * grid.size()..(r + 2) * grid.size()]
                    .iter()
                    .sum::<f64>()
                    > 0.
            {
                r + 1
            } else {
                0
            }
        })
        .collect();
    Ok(ProfileLearning {
        bank,
        regions,
        assignments,
        grid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_angularly_truncated_parent_keeps_its_quality_through_the_native_profile_fit() {
        let mut d = ProfileData::empty(vec![[1., 1., 0.5]]);
        d.samples = vec![[10., 1., 1.]];
        d.valid = vec![true];
        d.offsets = vec![0, 1];
        d.boxes = vec![Bounds {
            x0: 1,
            y0: 1,
            x1: 2,
            y1: 2,
        }];
        d.ids = vec![0];
        d.frames = vec![0];
        // Last five percent of the learned angular support lies beyond the scan.
        d.angles = vec![[0., 0.95]];
        d.covariance = vec![[0.; 9]];
        d.scales = vec![[1.; 3]];
        d.planar = vec![[1., 0., 0., 1.]];
        d.flags = vec![0];
        d.inherit_angular_flags(&[256]).unwrap();
        let mut bank = vec![0.; 27];
        bank[13] = 1.;
        let grid = profiles::Grid {
            shape: [3; 3],
            step: [1.; 3],
        };
        let (values, flags) = profiles::fit(
            &d.borrowed(),
            &bank,
            grid,
            &[0],
            &[[0.; 3]],
            0.9,
            false,
            1.,
            0.,
        )
        .unwrap();
        assert!(values[0][0] > 0.);
        assert_eq!(flags[0] & 256, 256);
        assert_eq!(d.flags, [0]);
        d.inherit_angular_flags(&[0]).unwrap();
        let (_, flags) = profiles::fit(
            &d.borrowed(),
            &bank,
            grid,
            &[0],
            &[[0.; 3]],
            0.9,
            false,
            1.,
            0.,
        )
        .unwrap();
        assert_eq!(flags[0] & 256, 0);
    }
}

#[cfg(test)]
mod derivative_tests {
    use super::*;
    #[test]
    fn local_angular_pixel_jacobian_matches_centered_ray_angle_differences() {
        let g = spots::reciprocal::Geometry {
            origin: [-0.03, 0.04, 0.15],
            fast: [0., 0.000_075, 0.],
            slow: [0.000_075, 0., 0.],
            beam: [0., 0., 1.],
            axis: [-1., 0., 0.],
            wavelength: 0.2846,
        };
        let xy = [23., 13.];
        let (ray, jac, _) = profile_geometry(xy, g);
        let q = g.convert(&[[xy[0], xy[1], 0.]]).unwrap()[0];
        for k in 0..3 {
            assert!((q[k] * g.wavelength + g.beam[k] - ray[k]).abs() < 1e-14);
        }
        let e1 = unit(cross(ray, unit(g.beam)));
        let e2 = cross(ray, e1);
        let step = 1e-3;
        for axis in 0..2 {
            let mut plus = xy;
            let mut minus = xy;
            plus[axis] += step;
            minus[axis] -= step;
            let rp = profile_geometry(plus, g).0;
            let rm = profile_geometry(minus, g).0;
            for (i, e) in [e1, e2].iter().enumerate() {
                let positive = dot(rp, *e).atan2(dot(rp, ray)).to_degrees();
                let negative = dot(rm, *e).atan2(dot(rm, ray)).to_degrees();
                let actual = (positive - negative) / (2. * step);
                assert!((actual - jac[2 * i + axis]).abs() < 1e-10);
            }
        }
    }
}
