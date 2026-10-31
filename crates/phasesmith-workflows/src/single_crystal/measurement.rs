use super::{math::*, *};
use phasesmith_spots::{self as spots, Bounds, frame_input, prediction, profiles};
/// Owned contiguous profile samples and their exact finite group supports.
pub struct ProfileData {
    pub samples: Vec<[f64; 3]>,
    pub valid: Vec<bool>,
    pub offsets: Vec<usize>,
    pub boxes: Vec<Bounds>,
    pub ids: Vec<usize>,
    pub frames: Vec<usize>,
    pub angles: Vec<[f64; 2]>,
    pub covariance: Vec<[f64; 9]>,
    pub centres: Vec<V3>,
    pub scales: Vec<V3>,
    pub planar: Vec<[f64; 4]>,
    pub flags: Vec<u32>,
    pub native_flags: Vec<u32>,
    pub frame_values: Vec<[f64; 11]>,
    pub background: Vec<[f64; 12]>,
}
impl ProfileData {
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
    pub detector_shape: [usize; 2],
    pub data: ProfileData,
    pub hkl: Vec<[i32; 3]>,
    pub aperture: Vec<[f64; 6]>,
    pub flags: Vec<u32>,
    pub bank: profiles::Learned,
    pub regions: Vec<usize>,
    pub assignments: Vec<usize>,
    pub target_count: usize,
    pub all_count: usize,
    pub excluded_hkl: Vec<[i32; 3]>,
    pub excluded_centres: Vec<V3>,
    pub excluded_values: Vec<[f64; 6]>,
    pub excluded_flags: Vec<u32>,
    pub grid: profiles::Grid,
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
    let hints = vec![(lo + hi) / 2.; h.len()];
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
    let mut grains = Vec::new();
    for (h, c) in hs.into_iter().zip(cs) {
        if !o
            .priors
            .group
            .is_systematically_absent(h.map(|x| x as i32))
            .map_err(|e| Error(e.to_string()))?
            && profile_geometry([c[0], c[1]], g).2.abs() >= 0.1
        {
            hkl.push(h.map(|x| x as i32));
            centres.push(c);
            grains.push(0);
        }
    }
    let target_count = hkl.len();
    for (i, competitor) in indexed.competitors.iter().enumerate() {
        for (&h, &c) in competitor.hkl.iter().zip(&competitor.crossings) {
            if profile_geometry([c[0], c[1]], g).2.abs() >= 0.1 {
                hkl.push(h.map(|x| x as i32));
                centres.push(c);
                grains.push(i + 1);
            }
        }
    }
    let all_count = hkl.len();
    eprintln!("native integrate {target_count} target / {all_count} all crossings");
    let widths: Vec<_> = centres
        .iter()
        .map(|c| (0.9 / profile_geometry([c[0], c[1]], g).2.abs() + 0.5).min(4.5))
        .collect();
    let mut d = ProfileData {
        samples: vec![],
        valid: vec![],
        offsets: vec![0],
        boxes: vec![],
        ids: vec![],
        frames: vec![],
        angles: vec![],
        covariance: vec![],
        centres: centres.clone(),
        scales: vec![],
        planar: vec![],
        flags: vec![],
        native_flags: vec![],
        frame_values: vec![],
        background: vec![],
    };
    for first in (0..angles.len()).step_by(o.frame_workers) {
        o.checkpoint()?;
        let mut payloads = Vec::new();
        for frame in first..(first + o.frame_workers).min(angles.len()) {
            let p = prepared(source, plan, o.acquisition.first, frame)?;
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
            payloads.push((frame, p, data, ids, boxes));
        }
        let extracted = std::thread::scope(|scope| {
            let tasks: Vec<_> = payloads
                .iter()
                .map(|(_, p, data, _, boxes)| {
                    scope.spawn(move || {
                        profiles::extract_frame(
                            &spots::Image {
                                values: data,
                                height: plan.shape[0],
                                width: plan.shape[1],
                                mask: &p.excluded,
                                variance: None,
                                gain: 1.,
                                read_noise: 0.,
                                saturation: None,
                            },
                            boxes,
                            3,
                            6,
                            20,
                            4.,
                        )
                    })
                })
                .collect();
            tasks
                .into_iter()
                .map(|t| {
                    t.join()
                        .map_err(|_| Error("integration worker panicked".into()))?
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for ((frame, _, _, ids, boxes), (spots, bg, cov, samples, valid)) in
            payloads.into_iter().zip(extracted)
        {
            if d.samples.len() + samples.len() > o.max_samples {
                return Err(Error("profile sample budget exceeded".into()));
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
        }
    }
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
    let mut remap = vec![None; centres.len()];
    for (id, &old) in keep.iter().enumerate() {
        remap[old] = Some(id);
    }
    let mut selected = ProfileData {
        samples: vec![],
        valid: vec![],
        offsets: vec![0],
        boxes: vec![],
        ids: vec![],
        frames: vec![],
        angles: vec![],
        covariance: vec![],
        centres: keep.iter().map(|&i| centres[i]).collect(),
        scales: vec![],
        planar: vec![],
        flags: vec![],
        native_flags: vec![],
        frame_values: vec![],
        background: vec![],
    };
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
    let dac_masks = (0..angles.len())
        .map(|i| {
            spots::frame_validity::dac_mask(
                o.geometry,
                angles[i],
                spots::frame_validity::Shape {
                    height: plan.shape[0],
                    width: plan.shape[1],
                    block: 16,
                },
                plan.dacs[0].options,
            )
            .map(|m| m.excluded)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut visible = Vec::new();
    for c in &selected.centres {
        let (_, a, zeta) = profile_geometry([c[0], c[1]], g);
        selected.planar.push(a);
        selected.scales.push([1., 1., zeta]);
        let nearest = angles
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (c[2] - (a.1[0] + a.1[1]) / 2.)
                    .abs()
                    .total_cmp(&(c[2] - (b.1[0] + b.1[1]) / 2.).abs())
            })
            .unwrap()
            .0;
        let x = (c[0] + 0.5).floor() as usize;
        let y = (c[1] + 0.5).floor() as usize;
        let block = 16;
        let cw = plan.shape[1].div_ceil(block);
        let ch = plan.shape[0].div_ceil(block);
        let mut bad = plan.blocks[0].excluded[nearest * ch * cw + (y / block) * cw + x / block];
        bad |= dac_masks[nearest][(y / block) * cw + x / block];
        visible.push(!bad);
    }
    let h: Vec<_> = keep.iter().map(|&i| hkl[i]).collect();
    let flags: Vec<_> = keep.iter().map(|&i| quality[i]).collect();
    let refs: Vec<_> = h
        .iter()
        .zip(&flags)
        .map(|(&h, &f)| {
            grains::fold(grains::family(h.map(i64::from), &o.priors.group)) != 0 && f == 0
        })
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
    selected.native_flags = selected
        .flags
        .iter()
        .zip(&selected.ids)
        .map(|(&f, &id)| f | (flags[id] & 256))
        .collect();
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
