use super::{math::*, *};
use phasesmith_spots::{self as spots, Bounds, frame_input, frame_validity, rotation};
/// Owned linked tracks and source-ordered detected observations.
pub struct Positions {
    pub boxes: Vec<Bounds>,
    pub ranges: Vec<[usize; 2]>,
    pub values: Vec<[f64; 5]>,
    pub flags: Vec<u32>,
    pub original_flags: Vec<u32>,
    pub observations: Vec<[f64; 5]>,
    pub angles: Vec<[f64; 2]>,
    pub hashes: Vec<String>,
}
pub(super) fn validity_mask(plan: &frame_input::Plan, i: usize) -> Result<Vec<bool>> {
    let [h, w] = plan.shape;
    let mut output = vec![false; h * w];
    for m in &plan.blocks {
        let ch = h.div_ceil(m.block);
        let cw = w.div_ceil(m.block);
        for y in 0..h {
            for x in 0..w {
                output[y * w + x] |= m.excluded[i * ch * cw + (y / m.block) * cw + x / m.block];
            }
        }
    }
    for d in &plan.dacs {
        let mask = frame_validity::dac_mask(
            d.geometry,
            d.bounds[i],
            frame_validity::Shape {
                height: h,
                width: w,
                block: d.block,
            },
            d.options,
        )?
        .expand()?;
        for (a, b) in output.iter_mut().zip(mask) {
            *a |= b;
        }
    }
    Ok(output)
}
pub(super) fn detect(
    source: &mut impl FrameSource,
    plan: &frame_input::Plan,
    o: &Options,
) -> Result<Positions> {
    let angles = bounds(o);
    let mut tables = Vec::new();
    let mut hashes = Vec::new();
    for first in (0..plan.frames).step_by(o.detection_workers) {
        o.checkpoint()?;
        let batch: (Vec<_>, Vec<_>) = (first..(first + o.detection_workers).min(plan.frames))
            .map(|i| {
                let p = prepared(source, plan, o.acquisition.first, i)?;
                let data = image(&p, plan.shape);
                Ok((p, data))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        let rows = std::thread::scope(|scope| {
            let tasks: Vec<_> = batch
                .0
                .iter()
                .zip(&batch.1)
                .map(|(p, data)| {
                    scope.spawn(move || {
                        spots::detect(
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
                            spots::Options::default(),
                        )
                    })
                })
                .collect();
            tasks
                .into_iter()
                .map(|t| {
                    t.join()
                        .map_err(|_| Error("detection worker panicked".into()))?
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for (p, row) in batch.0.into_iter().zip(rows) {
            hashes.push(p.digest);
            tables.push(row.spots);
        }
        if tables.iter().map(Vec::len).sum::<usize>() > 1_000_000 {
            return Err(Error("rotation observation budget exceeded".into()));
        }
    }
    let mut frames = Vec::new();
    let mut xy = Vec::new();
    let mut detected = Vec::new();
    for (i, table) in tables.into_iter().enumerate() {
        for item in table {
            frames.push(i as u32);
            xy.push([item.values[0], item.values[1]]);
            detected.push(item);
        }
    }
    let mut segments = vec![0; angles.len()];
    for i in 1..angles.len() {
        segments[i] = segments[i - 1] + u32::from((angles[i - 1][1] - angles[i][0]).abs() > 1e-8);
    }
    let labels = rotation::link(&frames, &xy, &segments, 5., 10_000_000)?;
    let n = labels.iter().max().map_or(0, |x| *x as usize + 1);
    let mut groups = vec![vec![]; n];
    for (i, &label) in labels.iter().enumerate() {
        groups[label as usize].push(i);
    }
    let mut result = Positions {
        boxes: Vec::new(),
        ranges: Vec::new(),
        values: Vec::new(),
        flags: Vec::new(),
        original_flags: Vec::new(),
        observations: Vec::new(),
        angles,
        hashes,
    };
    for g in groups {
        let mut b = Bounds {
            x0: usize::MAX,
            y0: usize::MAX,
            x1: 0,
            y1: 0,
        };
        let mut fs = Vec::new();
        let mut flag = 0;
        let mut samples = Vec::new();
        for &i in &g {
            let s = &detected[i];
            let f = frames[i] as usize;
            b.x0 = b.x0.min(s.bounds.x0);
            b.y0 = b.y0.min(s.bounds.y0);
            b.x1 = b.x1.max(s.bounds.x1);
            b.y1 = b.y1.max(s.bounds.y1);
            flag |= s.flags;
            fs.push(f);
            samples.push([
                s.values[0],
                s.values[1],
                (result.angles[f][0] + result.angles[f][1]) / 2.,
                s.values[2],
            ]);
        }
        let first = *fs.first().expect("track");
        let last = *fs.last().expect("track");
        let summary = rotation::summarize(&samples)?;
        if summary[..3].iter().any(|x| !x.is_finite()) {
            flag |= 64;
        }
        if fs.windows(2).any(|x| x[0] == x[1]) {
            flag |= 128;
        }
        if first == 0
            || last + 1 == plan.frames
            || (first > 0 && segments[first - 1] != segments[first])
            || (last + 1 < plan.frames && segments[last + 1] != segments[last])
        {
            flag |= 256;
        }
        result.boxes.push(b);
        result.ranges.push([first, last + 1]);
        result.values.push(summary);
        result.flags.push(flag);
    }
    for (i, s) in detected.iter().enumerate() {
        let f = frames[i] as usize;
        result.observations.push([
            f64::from(labels[i]),
            f as f64,
            s.values[0],
            s.values[1],
            (result.angles[f][0] + result.angles[f][1]) / 2.,
        ]);
    }
    result.original_flags = result.flags.clone();
    for frame in 0..plan.frames {
        let mask = validity_mask(plan, frame)?;
        for (id, b) in result.boxes.iter().enumerate() {
            let [first, last] = result.ranges[id];
            let before = frame + 1 == first && segments[frame] == segments[first];
            let after = frame == last && segments[frame] == segments[last - 1];
            if ((frame >= first && frame < last) || before || after)
                && (b.y0..b.y1).any(|y| {
                    mask[y * plan.shape[1] + b.x0..y * plan.shape[1] + b.x1]
                        .iter()
                        .any(|&x| x)
                })
            {
                result.flags[id] |= 2;
            }
        }
    }
    Ok(result)
}
pub(super) fn reciprocal(
    p: &Positions,
    g: phasesmith_spots::reciprocal::Geometry,
) -> Result<(Vec<V3>, Vec<f64>)> {
    let samples: Vec<_> = p.observations.iter().map(|x| [x[2], x[3], x[4]]).collect();
    let qs = g.convert(&samples)?;
    let mut sums = vec![[0.; 3]; p.values.len()];
    let mut counts = vec![0usize; p.values.len()];
    for (r, &q) in p.observations.iter().zip(&qs) {
        let i = r[0] as usize;
        counts[i] += 1;
        for j in 0..3 {
            sums[i][j] += q[j];
        }
    }
    for (s, &n) in sums.iter_mut().zip(&counts) {
        for x in s {
            *x /= n as f64;
        }
    }
    let mut rms = vec![0.; counts.len()];
    for (r, &q) in p.observations.iter().zip(&qs) {
        let i = r[0] as usize;
        let d = std::array::from_fn(|j| q[j] - sums[i][j]);
        rms[i] += dot(d, d);
    }
    for (v, &n) in rms.iter_mut().zip(&counts) {
        *v = (*v / n as f64).sqrt();
    }
    Ok((sums, rms))
}
