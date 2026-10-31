//! Independent mathematics implementation: Aslandukov et al., JAC 55 (2022)
//! 1383–1391, doi:10.1107/S1600576722008081. See ALGORITHM.md for conventions.
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
pub type Point = [f64; 3];
const ROOT: f64 = std::f64::consts::FRAC_1_SQRT_2;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Options {
    pub direction_epsilon: f64,
    pub projection_epsilon: f64,
    pub projection_grids: usize,
    pub gap_min_row: usize,
    pub phase_min_row: usize,
    pub component_pruning: bool,
    pub phase_radius: bool,
    pub fast_small_rows: bool,
    pub profile_stage2: bool,
    pub compact_cache_ids: bool,
    pub prune_gapless_grids: bool,
    pub direct_fixed_pack: bool,
    pub stream_fixed_rows: bool,
    pub fused_projections: bool,
    pub smallest_cache_first: bool,
    pub filter_sparse_cells: bool,
    pub candidate_subset_cache: bool,
    pub reuse_phase_output: bool,
    pub extend_best: bool,
    pub relative_epsilon: f64,
    pub gap_epsilon: f64,
    pub relative_gap: bool,
    pub directions: usize,
    pub min_row: usize,
    pub kmax: usize,
    /// Total retained payload budget in binary MiB; zero disables caching.
    pub cache_megabytes: usize,
    /// Overrides the finite budget and retains complete row preparations.
    pub cache_unlimited: bool,
    pub groups: usize,
    pub batch: usize,
    pub workers: usize,
    pub smart: bool,
    /// Use the published naive first stage for at most this many unique points.
    /// Zero disables automatic selection; explicit smart=false always wins.
    pub naive_max_points: usize,
    pub max_centers: usize,
    pub max_evaluations: usize,
    pub seconds: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            direction_epsilon: 0.0033,
            projection_epsilon: 0.002,
            projection_grids: 2,
            gap_min_row: 0,
            phase_min_row: 0,
            component_pruning: true,
            phase_radius: false,
            fast_small_rows: true,
            profile_stage2: false,
            compact_cache_ids: true,
            prune_gapless_grids: true,
            direct_fixed_pack: true,
            stream_fixed_rows: true,
            fused_projections: true,
            smallest_cache_first: true,
            filter_sparse_cells: true,
            candidate_subset_cache: true,
            reuse_phase_output: true,
            extend_best: false,
            relative_epsilon: 0.02,
            gap_epsilon: 0.02,
            relative_gap: false,
            directions: 50000,
            min_row: 4,
            kmax: 5,
            cache_megabytes: 512,
            cache_unlimited: false,
            groups: 40,
            batch: 6,
            workers: 2,
            smart: true,
            naive_max_points: 0,
            max_centers: usize::MAX,
            max_evaluations: usize::MAX,
            seconds: 300.0,
        }
    }
}
impl Options {
    fn cache_budget_bytes(&self) -> Result<Option<usize>, String> {
        if self.cache_unlimited {
            Ok(None)
        } else {
            self.cache_megabytes
                .checked_mul(1024 * 1024)
                .map(Some)
                .ok_or_else(|| "cache_megabytes overflows the platform byte count".into())
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        self.cache_budget_bytes()?;
        if !(self.direction_epsilon >= 0.0005 && self.direction_epsilon <= 0.1)
            || !(self.projection_epsilon >= 1e-8 && self.projection_epsilon <= 0.1)
            || !(self.relative_epsilon > 0. && self.relative_epsilon < 0.5)
            || !(self.gap_epsilon > 0. && self.gap_epsilon.is_finite())
            || ![0, 2, 4].contains(&self.projection_grids)
            || (self.gap_min_row > 0 && self.gap_min_row < 2)
            || (self.phase_min_row > 0 && self.phase_min_row < 2)
            || self.min_row < 2
            || self.kmax == 0
            || self.kmax > 100
            || self.batch == 0
            || !(1..=8).contains(&self.workers)
            || !self.seconds.is_finite()
            || self.seconds < 0.
        {
            return Err("invalid options".into());
        }
        Ok(())
    }
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ResultData {
    pub labels: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
    pub mean: Point,
    pub scale: Point,
    pub stage1_seconds: f64,
    pub stage2_seconds: f64,
    pub compute_seconds: f64,
    pub centers_completed: usize,
    pub occupied_directions: usize,
    pub evaluations_completed: usize,
    pub budget_exhausted: bool,
    pub cancelled: bool,
    pub group_cap_reached: bool,
    pub duplicate_observations: usize,
    /// None when deserializing a result produced before this field existed.
    #[serde(default)]
    pub effective_smart: Option<bool>,
    pub cache_payload_bytes: usize,
    pub cached_directions: usize,
    pub cached_point_entries: usize,
    pub stage2_prepare_worker_seconds: f64,
    pub stage2_cached_worker_seconds: f64,
    pub stage2_uncached_worker_seconds: f64,
    pub stage2_cache_hits: usize,
    pub stage2_preparations: usize,
    pub stage2_cache_evictions: usize,
    pub cached_candidate_directions: usize,
    pub cached_candidate_point_entries: usize,
    pub cached_bitset_directions: usize,
    pub cached_bitset_bytes: usize,
    pub options: Options,
}
fn axis(v: Point) -> usize {
    let mut k = 0;
    for i in 1..3 {
        if v[i].abs() > v[k].abs() {
            k = i;
        }
    }
    k
}
fn others(k: usize) -> [usize; 2] {
    match k {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}
pub fn direction_key(v: Point, eps: f64) -> Option<usize> {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n <= 1e-15 {
        return None;
    }
    let k = axis(v);
    let ij = others(k);
    let sign = if v[k] < 0. { -1. } else { 1. };
    let side = (2. * ROOT / eps).floor() as usize + 1;
    let a = ((sign * v[ij[0]] / n + ROOT) / eps).floor().max(0.) as usize;
    let b = ((sign * v[ij[1]] / n + ROOT) / eps).floor().max(0.) as usize;
    Some((k * side + a) * side + b)
}
pub fn decode(key: usize, eps: f64) -> Point {
    let side = (2. * ROOT / eps).floor() as usize + 1;
    let k = key / (side * side);
    let a = (key / side) % side;
    let b = key % side;
    let ij = others(k);
    let mut v = [0.; 3];
    v[ij[0]] = (a as f64 + 0.5) * eps - ROOT;
    v[ij[1]] = (b as f64 + 0.5) * eps - ROOT;
    v[k] = (1. - v[ij[0]] * v[ij[0]] - v[ij[1]] * v[ij[1]])
        .max(0.)
        .sqrt();
    v
}
fn expired(start: Instant, o: &Options, cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Relaxed) || start.elapsed().as_secs_f64() >= o.seconds
}
/// Exact closed interval sweep; different integer multiples of one point count once.
pub fn interval_support(ds: &[f64], o: &Options) -> usize {
    let mut events = Vec::with_capacity(2 * o.kmax * ds.len());
    for (id, &d) in ds.iter().enumerate() {
        for k in 1..=o.kmax {
            events.push((d / (k as f64 + o.relative_epsilon), 0u8, id));
            events.push((d / (k as f64 - o.relative_epsilon), 1u8, id));
        }
    }
    events.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut counts = vec![0u16; ds.len()];
    let (mut active, mut best) = (0, 0);
    for (_, kind, id) in events {
        if kind == 0 {
            if counts[id] == 0 {
                active += 1;
            }
            counts[id] += 1;
            best = best.max(active);
        } else {
            counts[id] -= 1;
            if counts[id] == 0 {
                active -= 1;
            }
        }
    }
    best
}
fn histogram(p: &[Point], o: &Options, start: Instant, cancel: &AtomicBool) -> (Vec<u64>, usize) {
    let side = (2. * ROOT / o.direction_epsilon).floor() as usize + 1;
    let bins = 3 * side * side;
    let nc = p.len().min(o.max_centers);
    let partial = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..o.workers {
            handles.push(scope.spawn(move || {
                let mut hist = vec![0u64; bins];
                let mut head = vec![usize::MAX; bins];
                let mut sizes = vec![0usize; bins];
                let mut next = vec![usize::MAX; p.len()];
                let mut distances = vec![0.; p.len()];
                let mut touched = Vec::with_capacity(p.len());
                let mut done = 0;
                for c in (worker..nc).step_by(o.workers) {
                    if expired(start, o, cancel) {
                        break;
                    }
                    touched.clear();
                    for j in 0..p.len() {
                        if c == j {
                            continue;
                        }
                        let v = [p[j][0] - p[c][0], p[j][1] - p[c][1], p[j][2] - p[c][2]];
                        if let Some(key) = direction_key(v, o.direction_epsilon) {
                            if !o.smart {
                                if j > c {
                                    hist[key] += 1;
                                }
                                continue;
                            }
                            if sizes[key] == 0 {
                                touched.push(key);
                            }
                            next[j] = head[key];
                            head[key] = j;
                            sizes[key] += 1;
                            distances[j] = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                        }
                    }
                    if o.smart {
                        for &key in &touched {
                            if sizes[key] + 1 >= o.min_row {
                                let mut ds = Vec::with_capacity(sizes[key]);
                                let mut j = head[key];
                                while j != usize::MAX {
                                    ds.push(distances[j]);
                                    j = next[j];
                                }
                                let support = interval_support(&ds, o) + 1;
                                if support >= o.min_row {
                                    hist[key] += support as u64;
                                }
                            }
                            sizes[key] = 0;
                            head[key] = usize::MAX;
                        }
                    }
                    done += 1;
                }
                (hist, done)
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut hist = vec![0u64; bins];
    let mut done = 0;
    for (h, n) in partial {
        done += n;
        for (a, b) in hist.iter_mut().zip(h) {
            *a += b;
        }
    }
    (hist, done)
}
#[derive(Clone, Copy)]
struct Projected {
    key: u64,
    t: f64,
    id: usize,
}
fn radix_projected(values: &mut Vec<Projected>, max_key: u64) {
    radix_projected_scratch(values, max_key, &mut Vec::new());
}
fn radix_projected_scratch(
    values: &mut Vec<Projected>,
    max_key: u64,
    scratch: &mut Vec<Projected>,
) {
    if values.is_empty() {
        return;
    }
    scratch.resize(values.len(), values[0]);
    let mut shift = 0;
    while max_key >> shift > 0 {
        let mut counts = [0usize; 1024];
        for x in values.iter() {
            counts[((x.key >> shift) & 1023) as usize] += 1;
        }
        let mut sum = 0;
        for x in &mut counts {
            let n = *x;
            *x = sum;
            sum += n;
        }
        for &x in values.iter() {
            let k = ((x.key >> shift) & 1023) as usize;
            scratch[counts[k]] = x;
            counts[k] += 1;
        }
        std::mem::swap(values, scratch);
        shift += 10;
        if shift >= 64 {
            break;
        }
    }
}
fn modal_gap(gaps: &mut [f64], eps: f64, relative: bool) -> Option<f64> {
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable_by(f64::total_cmp);
    let (mut l, mut best, mut choice) = (0, 0, 0.);
    for r in 0..gaps.len() {
        while gaps[r] - gaps[l]
            > if relative {
                eps * (gaps[r] + gaps[l])
            } else {
                2. * eps
            }
        {
            l += 1;
        }
        if r - l + 1 > best {
            best = r - l + 1;
            choice = 0.5 * (gaps[l] + gaps[r]);
        }
    }
    if choice > 1e-12 {
        Some(choice)
    } else {
        None
    }
}
/// Circular phase window with a distinct integer-site constraint.
pub fn phase_subset_generic(row: &[(f64, usize)], d: f64, eps: f64) -> Vec<usize> {
    if row.is_empty() || d <= 1e-12 {
        return vec![];
    }
    let mut phase: Vec<_> = row
        .iter()
        .map(|&(t, id)| {
            let s = (t / d).floor() as i64;
            (t.rem_euclid(d), s, id)
        })
        .collect();
    phase.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    let n = phase.len();
    let mut twice = phase.clone();
    twice.extend(phase.iter().map(|&(r, s, id)| (r + d, s - 1, id)));
    let (mut l, mut best, mut best_l, mut best_r) = (0, 0, 0, 0);
    let mut sites = HashMap::<i64, usize>::new();
    for r in 0..2 * n {
        *sites.entry(twice[r].1).or_default() += 1;
        while twice[r].0 - twice[l].0 > d * eps || r - l + 1 > n {
            let site = twice[l].1;
            let count = sites.get_mut(&site).unwrap();
            *count -= 1;
            if *count == 0 {
                sites.remove(&site);
            }
            l += 1;
        }
        if l >= n {
            break;
        }
        if sites.len() > best {
            best = sites.len();
            best_l = l;
            best_r = r;
        }
    }
    let mut selected = HashMap::<i64, usize>::new();
    for &(_, s, id) in &twice[best_l..=best_r] {
        selected
            .entry(s)
            .and_modify(|a| *a = (*a).min(id))
            .or_insert(id);
    }
    let mut result: Vec<_> = selected.into_values().collect();
    result.sort_unstable();
    result
}
/// Stack-resident implementation of the same endpoint sweep for short rows.
/// Sorting, inclusive boundaries, distinct-site counting and tie order are exact.
pub fn phase_subset(row: &[(f64, usize)], d: f64, eps: f64) -> Vec<usize> {
    let mut out = Vec::new();
    phase_subset_into::<16, 32>(row, d, eps, &mut out);
    out
}
fn phase_subset_into<const HALF: usize, const FULL: usize>(
    row: &[(f64, usize)],
    d: f64,
    eps: f64,
    out: &mut Vec<usize>,
) {
    out.clear();
    let n = row.len();
    if n == 0 || d <= 1e-12 {
        return;
    }
    if n > 16 {
        out.extend(phase_subset_generic(row, d, eps));
        return;
    }
    let mut phase = [(0f64, 0i64, 0usize); FULL];
    for (i, &(t, id)) in row.iter().enumerate() {
        phase[i] = (t.rem_euclid(d), (t / d).floor() as i64, id);
    }
    phase[..n].sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    for i in 0..n {
        let (r, s, id) = phase[i];
        phase[n + i] = (r + d, s - 1, id);
    }
    let mut sites = [(0i64, 0usize); FULL];
    let mut ns = 0;
    let (mut distinct, mut l, mut best, mut bl, mut br) = (0, 0, 0, 0, 0);
    for r in 0..2 * n {
        let key = phase[r].1;
        let ix = if let Some(ix) = sites[..ns].iter().position(|&(s, _)| s == key) {
            ix
        } else {
            sites[ns] = (key, 0);
            ns += 1;
            ns - 1
        };
        if sites[ix].1 == 0 {
            distinct += 1;
        }
        sites[ix].1 += 1;
        while phase[r].0 - phase[l].0 > d * eps || r - l + 1 > n {
            let ix = sites[..ns]
                .iter()
                .position(|&(s, _)| s == phase[l].1)
                .unwrap();
            sites[ix].1 -= 1;
            if sites[ix].1 == 0 {
                distinct -= 1;
            }
            l += 1;
        }
        if l >= n {
            break;
        }
        if distinct > best {
            best = distinct;
            bl = l;
            br = r;
        }
    }
    let mut selected = [(0i64, 0usize); HALF];
    let mut count = 0;
    for &(_, site, id) in &phase[bl..=br] {
        if let Some(i) = selected[..count].iter().position(|&(s, _)| s == site) {
            selected[i].1 = selected[i].1.min(id);
        } else {
            selected[count] = (site, id);
            count += 1;
        }
    }
    out.extend(selected[..count].iter().map(|&(_, id)| id));
    out.sort_unstable();
}
fn phase_for_into(row: &[(f64, usize)], d: f64, o: &Options, out: &mut Vec<usize>) {
    let eps = o.relative_epsilon * if o.phase_radius { 2. } else { 1. };
    if !o.fast_small_rows {
        out.clear();
        out.extend(phase_subset_generic(row, d, eps));
        return;
    }
    match row.len() {
        0..=4 => phase_subset_into::<4, 8>(row, d, eps, out),
        5..=8 => phase_subset_into::<8, 16>(row, d, eps, out),
        _ => phase_subset_into::<16, 32>(row, d, eps, out),
    }
}

// A three-point row can meet a three-site threshold only by retaining all
// three points. Enumerate the same three circular windows; rejected subsets
// cannot contribute to the group. Remainders, sites and inclusive widths match
// the generic sweep exactly.
fn complete_phase_triplet(row: &[(f64, usize)], d: f64, eps: f64) -> bool {
    debug_assert_eq!(row.len(), 3);
    if d <= 1e-12 {
        return false;
    }
    let mut phase = [(0f64, 0i64, 0usize); 6];
    for (i, &(t, id)) in row.iter().enumerate() {
        phase[i] = (t.rem_euclid(d), (t / d).floor() as i64, id);
    }
    phase[..3].sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    for i in 0..3 {
        let (r, s, id) = phase[i];
        phase[i + 3] = (r + d, s - 1, id);
    }
    (0..3).any(|l| {
        phase[l + 2].0 - phase[l].0 <= d * eps
            && phase[l].1 != phase[l + 1].1
            && phase[l].1 != phase[l + 2].1
            && phase[l + 1].1 != phase[l + 2].1
    })
}
fn phase_for(row: &[(f64, usize)], d: f64, o: &Options) -> Vec<usize> {
    let eps = o.relative_epsilon * if o.phase_radius { 2. } else { 1. };
    if o.fast_small_rows {
        phase_subset(row, d, eps)
    } else {
        phase_subset_generic(row, d, eps)
    }
}
type Rows = Vec<Vec<(f64, usize)>>;
#[derive(Clone)]
enum CompactRows {
    Narrow { offsets: Vec<u16>, ids: Vec<u16> },
    Wide { offsets: Vec<u32>, ids: Vec<u32> },
}
#[derive(Clone, Copy)]
enum CachedRow<'a> {
    Narrow(&'a [u16]),
    Wide(&'a [u32]),
}
impl<'a> CachedRow<'a> {
    fn len(self) -> usize {
        match self {
            Self::Narrow(a) => a.len(),
            Self::Wide(a) => a.len(),
        }
    }
    fn iter(self) -> impl Iterator<Item = usize> + 'a {
        (0..self.len()).map(move |i| match self {
            Self::Narrow(a) => a[i] as usize,
            Self::Wide(a) => a[i] as usize,
        })
    }
}
impl CompactRows {
    fn from_u32(offsets: Vec<u32>, ids: Vec<u32>, compact: bool) -> Self {
        if compact && ids.len() <= u16::MAX as usize && ids.iter().all(|&i| i <= u16::MAX as u32) {
            Self::Narrow {
                offsets: offsets.into_iter().map(|x| x as u16).collect(),
                ids: ids.into_iter().map(|x| x as u16).collect(),
            }
        } else {
            Self::Wide { offsets, ids }
        }
    }
    fn pack(rows: Rows, compact: bool) -> Self {
        let count = rows.iter().map(Vec::len).sum();
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut ids = Vec::with_capacity(count);
        offsets.push(0);
        for row in rows {
            ids.extend(
                row.into_iter()
                    .map(|(_, id)| u32::try_from(id).expect("u32 observation budget")),
            );
            offsets.push(u32::try_from(ids.len()).expect("u32 observation budget"));
        }
        Self::from_u32(offsets, ids, compact)
    }
    fn nrows(&self) -> usize {
        match self {
            Self::Narrow { offsets, .. } => offsets.len() - 1,
            Self::Wide { offsets, .. } => offsets.len() - 1,
        }
    }
    fn entries(&self) -> usize {
        match self {
            Self::Narrow { ids, .. } => ids.len(),
            Self::Wide { ids, .. } => ids.len(),
        }
    }
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                Self::Narrow { offsets, ids } => 2 * (offsets.len() + ids.len()),
                Self::Wide { offsets, ids } => 4 * (offsets.len() + ids.len()),
            }
    }
    fn row(&self, i: usize) -> CachedRow<'_> {
        match self {
            Self::Narrow { offsets, ids } => {
                CachedRow::Narrow(&ids[offsets[i] as usize..offsets[i + 1] as usize])
            }
            Self::Wide { offsets, ids } => {
                CachedRow::Wide(&ids[offsets[i] as usize..offsets[i + 1] as usize])
            }
        }
    }
}

struct Prepared {
    candidate_only: bool,
    candidate_bits: Vec<u64>,
    grids: Vec<CompactRows>,
    stable: Vec<bool>,
    direction: Point,
}
struct Cell {
    x: i32,
    y: i32,
    start: usize,
    end: usize,
}
struct Fine {
    cells: Vec<Cell>,
    values: Vec<Projected>,
}
fn phase_size(o: &Options) -> usize {
    if o.phase_min_row > 0 {
        o.phase_min_row
    } else {
        o.min_row
    }
}
fn keep_size(o: &Options) -> usize {
    o.min_row
        .min(phase_size(o))
        .min(if o.extend_best { 3 } else { o.min_row })
        .min(if o.gap_min_row > 0 {
            o.gap_min_row
        } else {
            o.min_row
        })
}
fn fine_cells(p: &[Point], ids: &[usize], v: Point, o: &Options, swap: bool) -> Fine {
    let k = axis(v);
    let ij = others(k);
    let width = o.projection_epsilon;
    let bias = (2. / width).ceil() as i64 + 2;
    let side = 2 * bias + 1;
    let mut values = Vec::with_capacity(ids.len());
    for &id in ids {
        let t = p[id][k] / v[k];
        let mut a = ((p[id][ij[0]] - v[ij[0]] * t) / width).floor() as i32;
        let mut b = ((p[id][ij[1]] - v[ij[1]] * t) / width).floor() as i32;
        if swap {
            std::mem::swap(&mut a, &mut b);
        }
        values.push(Projected {
            key: ((a as i64 + bias) * side + b as i64 + bias) as u64,
            t,
            id,
        });
    }
    radix_projected(&mut values, (side * side) as u64);
    let mut cells = Vec::new();
    let mut l = 0;
    while l < values.len() {
        let mut r = l + 1;
        while r < values.len() && values[r].key == values[l].key {
            r += 1;
        }
        let key = values[l].key as i64;
        cells.push(Cell {
            x: (key / side - bias) as i32,
            y: (key % side - bias) as i32,
            start: l,
            end: r,
        });
        l = r;
    }
    Fine { cells, values }
}
fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}
fn join(parent: &mut [usize], i: usize, j: usize) {
    let a = root(parent, i);
    let b = root(parent, j);
    parent[a.max(b)] = a.min(b);
}
/// Connected fine-cell components are lossless cache envelopes, NOT rows.
/// Sorted-column sweeps avoid a hash lookup for every empty neighboring cell.
fn adaptive_components(p: &[Point], ids: &[usize], v: Point, o: &Options) -> (Rows, Vec<bool>) {
    let fine = fine_cells(p, ids, v, o, false);
    let cells = &fine.cells;
    let mut eligible = vec![true; cells.len()];
    if o.component_pruning {
        let mut neighbors = vec![[0usize; 9]; cells.len()];
        for (i, c) in cells.iter().enumerate() {
            neighbors[i][4] = c.end - c.start;
        }
        for j in 1..cells.len() {
            if cells[j].x == cells[j - 1].x && cells[j].y - cells[j - 1].y == 1 {
                neighbors[j][3] = cells[j - 1].end - cells[j - 1].start;
                neighbors[j - 1][5] = cells[j].end - cells[j].start;
            }
        }
        let mut cols = Vec::new();
        let mut l = 0;
        while l < cells.len() {
            let mut r = l + 1;
            while r < cells.len() && cells[r].x == cells[l].x {
                r += 1;
            }
            cols.push((l, r));
            l = r;
        }
        for pair in cols.windows(2) {
            let (a, b) = pair[0];
            let (c, d) = pair[1];
            if cells[c].x - cells[a].x != 1 {
                continue;
            }
            let mut lo = c;
            for i in a..b {
                while lo < d && cells[lo].y < cells[i].y - 1 {
                    lo += 1;
                }
                let mut j = lo;
                while j < d && cells[j].y <= cells[i].y + 1 {
                    let dy = cells[j].y - cells[i].y;
                    neighbors[i][(7 + dy) as usize] = cells[j].end - cells[j].start;
                    neighbors[j][(1 - dy) as usize] = cells[i].end - cells[i].start;
                    j += 1;
                }
            }
        }
        for (i, n) in neighbors.iter().enumerate() {
            eligible[i] = [-1i32, 1].into_iter().any(|dx| {
                [-1i32, 1].into_iter().any(|dy| {
                    n[4] + n[((dx + 1) * 3 + 1) as usize]
                        + n[(4 + dy) as usize]
                        + n[((dx + 1) * 3 + dy + 1) as usize]
                        >= keep_size(o)
                })
            });
        }
    }
    let mut parent: Vec<_> = (0..cells.len()).collect();
    let mut columns = Vec::new();
    let mut l = 0;
    while l < cells.len() {
        let mut r = l + 1;
        while r < cells.len() && cells[r].x == cells[l].x {
            r += 1;
        }
        for j in l + 1..r {
            if eligible[j] && eligible[j - 1] && cells[j].y - cells[j - 1].y <= 1 {
                join(&mut parent, j, j - 1);
            }
        }
        columns.push((l, r));
        l = r;
    }
    for pair in columns.windows(2) {
        let (a, b) = pair[0];
        let (c, d) = pair[1];
        if cells[c].x - cells[a].x != 1 {
            continue;
        }
        let mut lo = c;
        for i in a..b {
            while lo < d && cells[lo].y < cells[i].y - 1 {
                lo += 1;
            }
            let mut j = lo;
            while j < d && cells[j].y <= cells[i].y + 1 {
                if eligible[i] && eligible[j] {
                    join(&mut parent, i, j);
                }
                j += 1;
            }
        }
    }
    let roots: Vec<_> = (0..cells.len()).map(|i| root(&mut parent, i)).collect();
    let mut counts = vec![0usize; cells.len()];
    for (i, c) in cells.iter().enumerate() {
        if eligible[i] {
            counts[roots[i]] += c.end - c.start;
        }
    }
    let mut group = vec![usize::MAX; cells.len()];
    let mut rows: Rows = Vec::new();
    let mut bounds = Vec::<[i32; 4]>::new();
    for (i, c) in cells.iter().enumerate() {
        let r = roots[i];
        if !eligible[i] {
            continue;
        }
        if counts[r] < keep_size(o) {
            continue;
        }
        if group[r] == usize::MAX {
            group[r] = rows.len();
            rows.push(Vec::with_capacity(counts[r]));
            bounds.push([c.x, c.x, c.y, c.y]);
        }
        let g = group[r];
        rows[g].extend(fine.values[c.start..c.end].iter().map(|x| (x.t, x.id)));
        bounds[g][0] = bounds[g][0].min(c.x);
        bounds[g][1] = bounds[g][1].max(c.x);
        bounds[g][2] = bounds[g][2].min(c.y);
        bounds[g][3] = bounds[g][3].max(c.y);
    }
    let mut stable = Vec::new();
    for (row, b) in rows.iter_mut().zip(bounds) {
        row.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        stable.push(b[1] - b[0] <= 1 && b[3] - b[2] <= 1);
    }
    (rows, stable)
}
/// Canonical bounded 2x2-cell partitions; large components are never chained.
fn bounded_rows(p: &[Point], ids: &[usize], v: Point, o: &Options, swap: bool) -> Rows {
    let fine = fine_cells(p, ids, v, o, swap);
    let cells = &fine.cells;
    let index: HashMap<_, _> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| ((c.x, c.y), i))
        .collect();
    let mut used = vec![false; cells.len()];
    let mut rows = Vec::new();
    for (i, c) in cells.iter().enumerate() {
        if used[i] {
            continue;
        }
        let mut chosen = Vec::new();
        let mut best_n = 0;
        for y0 in [c.y - 1, c.y] {
            let mut candidate = Vec::new();
            let mut n = 0;
            for dx in [0, 1] {
                for dy in [0, 1] {
                    if let Some(&j) = index.get(&(c.x + dx, y0 + dy)) {
                        if !used[j] {
                            candidate.push(j);
                            n += cells[j].end - cells[j].start;
                        }
                    }
                }
            }
            if n > best_n {
                best_n = n;
                chosen = candidate;
            }
        }
        if best_n < keep_size(o) {
            used[i] = true;
            continue;
        }
        let mut row = Vec::with_capacity(best_n);
        for j in chosen {
            used[j] = true;
            let c = &cells[j];
            row.extend(fine.values[c.start..c.end].iter().map(|x| (x.t, x.id)));
        }
        row.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        rows.push(row);
    }
    rows
}
fn active_grids(p: &[Point], prepared: &Prepared, alive: &[bool], o: &Options) -> Vec<Rows> {
    let k = axis(prepared.direction);
    let vk = prepared.direction[k];
    if o.projection_grids != 0 {
        return prepared
            .grids
            .iter()
            .map(|grid| {
                (0..grid.nrows())
                    .map(|i| {
                        grid.row(i)
                            .iter()
                            .filter(|&id| alive[id])
                            .map(|id| (p[id][k] / vk, id))
                            .collect::<Vec<_>>()
                    })
                    .filter(|row| row.len() >= keep_size(o))
                    .collect()
            })
            .collect();
    }
    let mut grids = vec![vec![], vec![]];
    for i in 0..prepared.stable.len() {
        let ids: Vec<_> = prepared.grids[0]
            .row(i)
            .iter()
            .filter(|&id| alive[id])
            .collect();
        if ids.len() < keep_size(o) {
            continue;
        }
        if prepared.stable[i] {
            let row: Vec<_> = ids.into_iter().map(|id| (p[id][k] / vk, id)).collect();
            grids[0].push(row.clone());
            grids[1].push(row);
        } else {
            for (pass, grid) in grids.iter_mut().enumerate() {
                grid.extend(bounded_rows(p, &ids, prepared.direction, o, pass == 1));
            }
        }
    }
    grids
}
fn prepared_bytes(prepared: &Prepared) -> usize {
    prepared.grids.iter().map(CompactRows::bytes).sum::<usize>()
        + prepared.stable.len()
        + prepared.candidate_bits.len() * 8
        + std::mem::size_of::<Prepared>()
}
// A worker retains full preparations while they fit its actual payload share.
// On pressure, compact only geometric eligibility supersets; stale heap entries
// are ignored after compaction/replacement. Existing cache entries are immutable.
struct CacheAdmission {
    entries: Vec<Option<(usize, Prepared)>>,
    largest: BinaryHeap<(usize, usize)>,
    full: BinaryHeap<(usize, usize)>,
    bytes: usize,
    cap: Option<usize>,
    compressions: usize,
}
impl CacheAdmission {
    fn new(bytes: usize, cap: Option<usize>) -> Self {
        Self {
            entries: Vec::new(),
            largest: BinaryHeap::new(),
            full: BinaryHeap::new(),
            bytes,
            cap,
            compressions: 0,
        }
    }
    fn fits(&self, size: usize) -> bool {
        self.cap
            .is_none_or(|cap| size <= cap.saturating_sub(self.bytes))
    }
    fn push_heaps(&mut self, slot: usize, can_compact: bool) {
        let rows = &self.entries[slot].as_ref().unwrap().1;
        let size = prepared_bytes(rows);
        self.largest.push((size, slot));
        if can_compact && !rows.candidate_only {
            self.full.push((size, slot));
        }
    }
    fn compact_largest(&mut self, o: &Options, point_count: usize) -> bool {
        while let Some((size, slot)) = self.full.pop() {
            let Some((_, rows)) = &self.entries[slot] else {
                continue;
            };
            if rows.candidate_only || prepared_bytes(rows) != size {
                continue;
            }
            let subset = candidate_cache(rows, o, point_count);
            let smaller = prepared_bytes(&subset);
            if smaller < size {
                self.entries[slot].as_mut().unwrap().1 = subset;
                self.bytes -= size - smaller;
                self.largest.push((smaller, slot));
                self.compressions += 1;
                return true;
            }
        }
        false
    }
    fn largest_live(&mut self) -> Option<(usize, usize)> {
        while let Some(&(size, slot)) = self.largest.peek() {
            if self.entries[slot]
                .as_ref()
                .is_some_and(|(_, rows)| prepared_bytes(rows) == size)
            {
                return Some((size, slot));
            }
            self.largest.pop();
        }
        None
    }
    fn admit(&mut self, rank: usize, mut rows: Prepared, o: &Options, point_count: usize) -> usize {
        let can_compact = o.candidate_subset_cache
            && o.projection_grids != 0
            && self.cap.is_some_and(|cap| cap > 0);
        let mut size = prepared_bytes(&rows);
        if !self.fits(size) && can_compact {
            let subset = candidate_cache(&rows, o, point_count);
            if prepared_bytes(&subset) < size {
                rows = subset;
                size = prepared_bytes(&rows);
                self.compressions += 1;
            }
            while !self.fits(size) && self.compact_largest(o, point_count) {}
        }
        if self.fits(size) {
            self.bytes += size;
            let slot = self.entries.len();
            self.entries.push(Some((rank, rows)));
            self.push_heaps(slot, can_compact);
        } else if o.smallest_cache_first {
            if let Some((largest, slot)) = self.largest_live() {
                if size < largest {
                    self.largest.pop();
                    self.bytes = self.bytes - largest + size;
                    self.entries[slot] = Some((rank, rows));
                    self.push_heaps(slot, can_compact);
                    return 1;
                }
            }
        }
        0
    }
}

fn gap_size(o: &Options) -> usize {
    if o.gap_min_row > 0 {
        o.gap_min_row
    } else {
        o.min_row
    }
}

#[derive(Default)]
struct Stage2Scratch {
    cell_counts: Vec<u8>,
    touched: Vec<usize>,
    projected: [Vec<Projected>; 4],
    shared_keys: [Vec<u64>; 4],
    shared_t: Vec<f64>,
    radix: Vec<Projected>,
    flat: FlatScratch,
}
#[derive(Default)]
struct FlatScratch {
    values: Vec<(f64, usize)>,
    offsets: Vec<usize>,
    gaps: Vec<f64>,
    sub: Vec<usize>,
    group: Vec<usize>,
    best: Vec<usize>,
}
impl Stage2Scratch {
    fn emit_grid(
        &mut self,
        out: &mut Vec<Projected>,
        grid: usize,
        space: usize,
        keep: usize,
        filter: bool,
        ids: &[usize],
    ) {
        out.clear();
        let keys = &self.shared_keys[grid];
        debug_assert_eq!(keys.len(), ids.len());
        let counted = filter
            && space <= 2_000_000
            && keep <= u8::MAX as usize
            && keys.iter().all(|&key| key < space as u64);
        if counted {
            self.cell_counts.resize(space, 0);
            self.touched.clear();
            for &key in keys {
                let key = key as usize;
                let count = &mut self.cell_counts[key];
                if *count == 0 {
                    self.touched.push(key);
                }
                *count = count.saturating_add(1);
            }
        }
        for (i, &key) in keys.iter().enumerate() {
            if !counted || self.cell_counts[key as usize] as usize >= keep {
                out.push(Projected {
                    key,
                    t: self.shared_t[i],
                    id: ids[i],
                });
            }
        }
        if counted {
            for &key in &self.touched {
                self.cell_counts[key] = 0;
            }
        }
    }
    fn retain_eligible(&mut self, projected: &mut Vec<Projected>, cell_space: usize, keep: usize) {
        // Saturation is lossless: only the predicate count >= keep is used.
        // Rare huge grids/thresholds retain the original full-sort fallback.
        if cell_space > 2_000_000
            || keep > u8::MAX as usize
            || projected.iter().any(|x| x.key >= cell_space as u64)
        {
            return;
        }
        self.cell_counts.resize(cell_space, 0);
        self.touched.clear();
        for x in projected.iter() {
            let key = x.key as usize;
            let count = &mut self.cell_counts[key];
            if *count == 0 {
                self.touched.push(key);
            }
            *count = count.saturating_add(1);
        }
        projected.retain(|x| self.cell_counts[x.key as usize] as usize >= keep);
        for &key in &self.touched {
            self.cell_counts[key] = 0;
        }
    }
}
fn prepare_direction(p: &[Point], ids: &[usize], v: Point, o: &Options) -> Prepared {
    prepare_direction_scratch(p, ids, v, o, &mut Stage2Scratch::default())
}
fn prepare_direction_scratch(
    p: &[Point],
    ids: &[usize],
    v: Point,
    o: &Options,
    scratch: &mut Stage2Scratch,
) -> Prepared {
    let k = axis(v);
    if v[k].abs() < 1e-12 {
        return Prepared {
            grids: vec![CompactRows::pack(vec![], o.compact_cache_ids); o.projection_grids.max(1)],
            stable: vec![],
            direction: v,
            candidate_only: false,
            candidate_bits: Vec::new(),
        };
    }
    let ij = others(k);
    if o.projection_grids == 0 {
        let (components, stable) = adaptive_components(p, ids, v, o);
        return Prepared {
            grids: vec![CompactRows::pack(components, o.compact_cache_ids)],
            stable,
            direction: v,
            candidate_only: false,
            candidate_bits: Vec::new(),
        };
    }
    // Calculate the exact same four offset keys in a single point pass.
    // Share t and the unshifted projections; do not change arithmetic order
    // inside either floor expression.
    if o.fused_projections {
        scratch.shared_t.resize(ids.len(), 0.);
        for keys in &mut scratch.shared_keys {
            keys.resize(ids.len(), 0);
        }
        let width = 2. * o.projection_epsilon;
        let bias = (2. / width).ceil() as i64 + 2;
        let side = 2 * bias + 1;
        let [key0, key1, key2, key3] = &mut scratch.shared_keys;
        for (((((&id, longitudinal), k0), k1), k2), k3) in ids
            .iter()
            .zip(&mut scratch.shared_t)
            .zip(key0)
            .zip(key1)
            .zip(key2)
            .zip(key3)
        {
            let t = p[id][k] / v[k];
            let a = p[id][ij[0]] - v[ij[0]] * t;
            let b = p[id][ij[1]] - v[ij[1]] * t;
            let a0 = ((a + 0. * o.projection_epsilon) / width).floor() as i32;
            let a1 = ((a + 1. * o.projection_epsilon) / width).floor() as i32;
            let b0 = ((b + 0. * o.projection_epsilon) / width).floor() as i32;
            let b1 = ((b + 1. * o.projection_epsilon) / width).floor() as i32;
            let key = |a: i32, b: i32| ((a as i64 + bias) * side + b as i64 + bias) as u64;
            *longitudinal = t;
            *k0 = key(a0, b0);
            *k1 = key(a1, b1);
            *k2 = key(a1, b0);
            *k3 = key(a0, b1);
        }
    }
    let grids: Vec<CompactRows> = (0..o.projection_grids)
        .map(|grid| {
            let offsets = [(0., 0.), (1., 1.), (1., 0.), (0., 1.)];
            let (ox, oy) = offsets[grid];
            let width = 2. * o.projection_epsilon;
            let bias = (2. / width).ceil() as i64 + 2;
            let side = 2 * bias + 1;
            let mut projected = std::mem::take(&mut scratch.projected[grid]);
            if o.fused_projections {
                scratch.emit_grid(
                    &mut projected,
                    grid,
                    (side * side) as usize,
                    keep_size(o),
                    o.filter_sparse_cells,
                    ids,
                );
            } else {
                projected.clear();
                projected.reserve(ids.len());
                for &id in ids {
                    let t = p[id][k] / v[k];
                    let a = ((p[id][ij[0]] - v[ij[0]] * t + ox * o.projection_epsilon) / width)
                        .floor() as i32;
                    let b = ((p[id][ij[1]] - v[ij[1]] * t + oy * o.projection_epsilon) / width)
                        .floor() as i32;
                    projected.push(Projected {
                        key: ((a as i64 + bias) * side + b as i64 + bias) as u64,
                        t,
                        id,
                    });
                }
            }
            if o.filter_sparse_cells && !o.fused_projections {
                scratch.retain_eligible(&mut projected, (side * side) as usize, keep_size(o));
            }
            radix_projected_scratch(&mut projected, (side * side) as u64, &mut scratch.radix);
            let mut rows = Vec::new();
            let mut packed_ids = Vec::new();
            let mut packed_offsets = vec![0u32];
            let mut has_gap_row = false;
            let mut l = 0;
            while l < projected.len() {
                let mut r = l + 1;
                while r < projected.len() && projected[r].key == projected[l].key {
                    r += 1;
                }
                if r - l >= keep_size(o) {
                    projected[l..r]
                        .sort_unstable_by(|a, b| a.t.total_cmp(&b.t).then(a.id.cmp(&b.id)));
                    has_gap_row |= r - l >= gap_size(o);
                    if o.direct_fixed_pack {
                        packed_ids.extend(projected[l..r].iter().map(|x| x.id as u32));
                        packed_offsets.push(packed_ids.len() as u32);
                    } else {
                        rows.push(projected[l..r].iter().map(|x| (x.t, x.id)).collect());
                    }
                }
                l = r;
            }
            // Fixed grid membership cannot grow after deleting observations.
            // With no row large enough for spacing evidence, this grid can
            // never have a modal gap on any later iteration.
            scratch.projected[grid] = projected;
            if o.prune_gapless_grids && !has_gap_row {
                CompactRows::pack(vec![], o.compact_cache_ids)
            } else if o.direct_fixed_pack {
                CompactRows::from_u32(packed_offsets, packed_ids, o.compact_cache_ids)
            } else {
                CompactRows::pack(rows, o.compact_cache_ids)
            }
        })
        .collect();
    Prepared {
        grids,
        stable: vec![],
        direction: v,
        candidate_only: false,
        candidate_bits: Vec::new(),
    }
}
fn prepared_details_nested(
    p: &[Point],
    prepared: &Prepared,
    alive: &[bool],
    o: &Options,
) -> (Vec<usize>, Option<f64>, usize) {
    let mut best = Vec::new();
    let mut distance = None;
    let mut grid = 0;
    for (gi, rows) in active_grids(p, prepared, alive, o).into_iter().enumerate() {
        let mut gaps = Vec::new();
        for row in &rows {
            if row.len()
                < if o.gap_min_row > 0 {
                    o.gap_min_row
                } else {
                    o.min_row
                }
            {
                continue;
            }
            for pair in row.windows(2) {
                let d = pair[1].0 - pair[0].0;
                if d > 1e-12 {
                    gaps.push(d);
                }
            }
        }
        if let Some(d) = modal_gap(&mut gaps, o.gap_epsilon, o.relative_gap) {
            let mut group = Vec::new();
            for row in rows {
                if row.len() < phase_size(o) {
                    continue;
                }
                let sub = phase_for(&row, d, o);
                if sub.len() >= phase_size(o) {
                    group.extend(sub);
                }
            }
            group.sort_unstable();
            if group.len() > best.len() {
                best = group;
                distance = Some(d);
                grid = gi;
            }
        }
    }
    (best, distance, grid)
}
fn prepared_details_flat(
    p: &[Point],
    prepared: &Prepared,
    alive: &[bool],
    o: &Options,
    scratch: &mut FlatScratch,
) -> (Vec<usize>, Option<f64>, usize) {
    let k = axis(prepared.direction);
    let vk = prepared.direction[k];
    let FlatScratch {
        values,
        offsets,
        gaps,
        sub,
        group,
        best,
    } = scratch;
    best.clear();
    let mut distance = None;
    let mut best_grid = 0;
    for (gi, grid) in prepared.grids.iter().enumerate() {
        values.clear();
        offsets.clear();
        gaps.clear();
        offsets.push(0);
        for i in 0..grid.nrows() {
            let begin = values.len();
            for id in grid.row(i).iter().filter(|&id| alive[id]) {
                values.push((p[id][k] / vk, id));
            }
            if values.len() - begin >= keep_size(o) {
                offsets.push(values.len());
            } else {
                values.truncate(begin);
            }
        }
        for bounds in offsets.windows(2) {
            let row = &values[bounds[0]..bounds[1]];
            if row.len() >= gap_size(o) {
                for pair in row.windows(2) {
                    let d = pair[1].0 - pair[0].0;
                    if d > 1e-12 {
                        gaps.push(d);
                    }
                }
            }
        }
        if let Some(d) = modal_gap(gaps, o.gap_epsilon, o.relative_gap) {
            group.clear();
            for bounds in offsets.windows(2) {
                let row = &values[bounds[0]..bounds[1]];
                if row.len() < phase_size(o) {
                    continue;
                }
                if o.fast_small_rows && row.len() == 3 && phase_size(o) == 3 {
                    let eps = o.relative_epsilon * if o.phase_radius { 2. } else { 1. };
                    if complete_phase_triplet(row, d, eps) {
                        group.extend(row.iter().map(|&(_, id)| id));
                    }
                    continue;
                }
                if o.reuse_phase_output {
                    phase_for_into(row, d, o, sub);
                } else {
                    *sub = phase_for(row, d, o);
                }
                if sub.len() >= phase_size(o) {
                    group.extend(sub.iter().copied());
                }
            }
            group.sort_unstable();
            if group.len() > best.len() {
                std::mem::swap(best, group);
                distance = Some(d);
                best_grid = gi;
            }
        }
    }
    (best.clone(), distance, best_grid)
}
fn prepared_details(
    p: &[Point],
    prepared: &Prepared,
    alive: &[bool],
    o: &Options,
) -> (Vec<usize>, Option<f64>, usize) {
    prepared_details_scratch(p, prepared, alive, o, &mut FlatScratch::default())
}
fn prepared_details_scratch(
    p: &[Point],
    prepared: &Prepared,
    alive: &[bool],
    o: &Options,
    scratch: &mut FlatScratch,
) -> (Vec<usize>, Option<f64>, usize) {
    if o.stream_fixed_rows && o.projection_grids != 0 {
        prepared_details_flat(p, prepared, alive, o, scratch)
    } else {
        prepared_details_nested(p, prepared, alive, o)
    }
}

fn candidate_cache(prepared: &Prepared, o: &Options, point_count: usize) -> Prepared {
    let mut ids: Vec<u32> = prepared
        .grids
        .iter()
        .flat_map(|grid| {
            (0..grid.nrows()).flat_map(move |i| grid.row(i).iter().map(|id| id as u32))
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let narrow = o.compact_cache_ids
        && ids.len() <= u16::MAX as usize
        && ids.last().is_none_or(|&id| id <= u16::MAX as u32);
    let id_bytes =
        (ids.len() + 2) * (if narrow { 2 } else { 4 }) + std::mem::size_of::<CompactRows>();
    let bit_words = point_count.div_ceil(64);
    if bit_words > 0 && bit_words * 8 < id_bytes {
        let mut bits = vec![0u64; bit_words];
        for id in ids {
            bits[id as usize / 64] |= 1u64 << (id % 64);
        }
        return Prepared {
            grids: Vec::new(),
            stable: Vec::new(),
            direction: prepared.direction,
            candidate_only: true,
            candidate_bits: bits,
        };
    }
    let offsets = vec![0, ids.len() as u32];
    Prepared {
        grids: vec![CompactRows::from_u32(offsets, ids, o.compact_cache_ids)],
        stable: vec![],
        direction: prepared.direction,
        candidate_only: true,
        candidate_bits: Vec::new(),
    }
}
fn candidate_ids(prepared: &Prepared, alive: &[bool]) -> Vec<usize> {
    if prepared.candidate_bits.is_empty() {
        return prepared.grids[0]
            .row(0)
            .iter()
            .filter(|&id| alive[id])
            .collect();
    }
    let mut ids = Vec::new();
    for (base, &word) in prepared.candidate_bits.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let id = base * 64 + bit;
            if alive[id] {
                ids.push(id);
            }
            bits &= bits - 1;
        }
    }
    ids
}
fn prepared_entries(prepared: &Prepared) -> usize {
    prepared
        .grids
        .iter()
        .map(CompactRows::entries)
        .sum::<usize>()
        + prepared
            .candidate_bits
            .iter()
            .map(|x| x.count_ones() as usize)
            .sum::<usize>()
}
fn prepared_group_scratch(
    p: &[Point],
    prepared: &Prepared,
    alive: &[bool],
    o: &Options,
    scratch: &mut Stage2Scratch,
) -> Vec<usize> {
    if prepared.candidate_only {
        let ids = candidate_ids(prepared, alive);
        if ids.len() < keep_size(o) {
            return Vec::new();
        }
        let current = prepare_direction_scratch(p, &ids, prepared.direction, o, scratch);
        prepared_details_scratch(p, &current, alive, o, &mut scratch.flat).0
    } else {
        prepared_details_scratch(p, prepared, alive, o, &mut scratch.flat).0
    }
}
fn prepared_group(p: &[Point], prepared: &Prepared, alive: &[bool], o: &Options) -> Vec<usize> {
    prepared_group_scratch(p, prepared, alive, o, &mut Stage2Scratch::default())
}
fn extend_direction(p: &[Point], ids: &[usize], v: Point, o: &Options) -> Vec<usize> {
    let mut keep = o.clone();
    keep.extend_best = true;
    let prepared = prepare_direction(p, ids, v, &keep);
    let (base, d, grid) = prepared_details(p, &prepared, &vec![true; p.len()], o);
    let Some(d) = d else {
        return base;
    };
    let mut group = Vec::new();
    for row in &active_grids(p, &prepared, &vec![true; p.len()], o)[grid] {
        let sub = phase_for(row, d, o);
        if sub.len() >= 3 {
            group.extend(sub);
        }
    }
    group.sort_unstable();
    if group.len() > base.len() {
        group
    } else {
        base
    }
}
pub fn group_for_direction(p: &[Point], ids: &[usize], v: Point, o: &Options) -> Vec<usize> {
    prepared_group(p, &prepare_direction(p, ids, v, o), &vec![true; p.len()], o)
}
/// Conditional loss audit only: use an externally specified spacing.
/// Blind grouping never calls this function.
pub fn group_at_distance(p: &[Point], ids: &[usize], v: Point, d: f64, o: &Options) -> Vec<usize> {
    let mut known_spacing = o.clone();
    known_spacing.prune_gapless_grids = false;
    let prepared = prepare_direction(p, ids, v, &known_spacing);
    let mut best = Vec::new();
    for rows in active_grids(p, &prepared, &vec![true; p.len()], o) {
        let mut group = Vec::new();
        for row in rows {
            let sub = phase_for(&row, d, o);
            if sub.len() >= phase_size(o) {
                group.extend(sub);
            }
        }
        group.sort_unstable();
        if group.len() > best.len() {
            best = group;
        }
    }
    best
}
pub fn run(input: &[Point], o: &Options, cancel: &AtomicBool) -> Result<ResultData, String> {
    o.validate()?;
    let cache_budget_bytes = o.cache_budget_bytes()?;
    if input.len() > u32::MAX as usize {
        return Err("observation count exceeds u32 work budget".into());
    }
    if input.iter().flatten().any(|x| !x.is_finite()) {
        return Err("non-finite input".into());
    }
    let start = Instant::now();
    let mut order: Vec<_> = (0..input.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        input[a][0]
            .total_cmp(&input[b][0])
            .then(input[a][1].total_cmp(&input[b][1]))
            .then(input[a][2].total_cmp(&input[b][2]))
            .then(a.cmp(&b))
    });
    // Coincident observations are conserved but only the first can represent a site.
    let mut unique = Vec::new();
    for id in order {
        if unique.last().is_none_or(|&j| input[j] != input[id]) {
            unique.push(id);
        }
    }
    let duplicates = input.len() - unique.len();
    let mut mean = [0.; 3];
    let mut scale = [0.; 3];
    for &id in &unique {
        for k in 0..3 {
            mean[k] += input[id][k] / unique.len() as f64;
        }
    }
    for &id in &unique {
        for k in 0..3 {
            scale[k] = f64::max(scale[k], (input[id][k] - mean[k]).abs());
        }
    }
    for s in &mut scale {
        if *s == 0. {
            *s = 1.;
        }
    }
    let p: Vec<Point> = unique
        .iter()
        .map(|&id| std::array::from_fn(|k| (input[id][k] - mean[k]) / scale[k]))
        .collect();
    // Both methods are defined in Aslandukov et al. (2022), sections2.3.1–2.3.2.
    // The size crossover is execution policy, not a cell/chemistry/label input.
    // Normalization, ranking budget and all second-stage equations are shared.
    let mut direction_options = o.clone();
    if o.naive_max_points > 0 && p.len() <= o.naive_max_points {
        direction_options.smart = false;
    }
    let (hist, centers) = histogram(&p, &direction_options, start, cancel);
    let stage1 = start.elapsed().as_secs_f64();
    let mut ranked: Vec<_> = hist
        .into_iter()
        .enumerate()
        .filter(|&(_, v)| v > 0)
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let occupied = ranked.len();
    ranked.truncate(o.directions);
    let mut cache: Vec<Option<Prepared>> = (0..ranked.len()).map(|_| None).collect();
    let mut alive = vec![true; p.len()];
    let mut ids: Vec<_> = (0..p.len()).collect();
    let mut groups = Vec::new();
    let mut evaluations = 0;
    let mut profile_seconds = [0.; 3];
    let mut profile_counts = [0usize; 3];
    let mut work_exhausted = centers < p.len();
    while groups.len() < o.groups
        && ids.len() >= o.min_row
        && !expired(start, o, cancel)
        && !work_exhausted
    {
        let remaining = o.max_evaluations.saturating_sub(evaluations);
        let count = ranked.len().min(remaining);
        if count == 0 {
            work_exhausted = remaining == 0;
            break;
        }
        let partial = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for w in 0..o.workers {
                let p = &p;
                let ids = &ids;
                let ranked = &ranked;
                let cache = &cache;
                let alive = &alive;
                handles.push(scope.spawn(move || {
                    let mut candidates = Vec::new();
                    let mut scratch = Stage2Scratch::default();
                    let mut timings = [0.; 3];
                    let mut counts = [0usize; 3];
                    let bytes = cache
                        .iter()
                        .enumerate()
                        .filter(|(rank, _)| rank % o.workers == w)
                        .filter_map(|(_, x)| x.as_ref())
                        .map(prepared_bytes)
                        .sum::<usize>();
                    let cap = cache_budget_bytes.map(|b| b / o.workers);
                    let mut admission = CacheAdmission::new(bytes, cap);
                    let mut done = 0;
                    for rank in (w..count).step_by(o.workers) {
                        if expired(start, o, cancel) {
                            break;
                        }
                        let g = if let Some(rows) = &cache[rank] {
                            let tick = o.profile_stage2.then(Instant::now);
                            let g = prepared_group_scratch(p, rows, alive, o, &mut scratch);
                            if let Some(t) = tick {
                                timings[1] += t.elapsed().as_secs_f64();
                            }
                            counts[0] += 1;
                            g
                        } else {
                            let tick = o.profile_stage2.then(Instant::now);
                            let rows = prepare_direction_scratch(
                                p,
                                ids,
                                decode(ranked[rank].0, o.direction_epsilon),
                                o,
                                &mut scratch,
                            );
                            if let Some(t) = tick {
                                timings[0] += t.elapsed().as_secs_f64();
                            }
                            let tick = o.profile_stage2.then(Instant::now);
                            let g = prepared_group_scratch(p, &rows, alive, o, &mut scratch);
                            if let Some(t) = tick {
                                timings[2] += t.elapsed().as_secs_f64();
                            }
                            counts[1] += 1;
                            counts[2] += admission.admit(rank, rows, o, p.len());
                            g
                        };
                        if g.len() >= o.min_row {
                            candidates.push((rank, g));
                        }
                        done += 1;
                    }
                    (candidates, done, admission.entries, timings, counts)
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut candidates = Vec::new();
        let mut done = 0;
        for (c, n, additions, timings, counts) in partial {
            for (a, b) in profile_seconds.iter_mut().zip(timings) {
                *a += b;
            }
            for (a, b) in profile_counts.iter_mut().zip(counts) {
                *a += b;
            }
            candidates.extend(c);
            done += n;
            for (rank, rows) in additions.into_iter().flatten() {
                cache[rank] = Some(rows);
            }
        }
        evaluations += done;
        if done < count || count < ranked.len() {
            work_exhausted = true;
        }
        if candidates.is_empty() {
            break;
        }
        let mut removed = vec![false; p.len()];
        for batch_index in 0..o.batch.min(o.groups - groups.len()) {
            for (_, g) in &mut candidates {
                g.retain(|&id| !removed[id]);
            }
            candidates.sort_unstable_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
            if candidates[0].1.len() < o.min_row {
                break;
            }
            let mut g = std::mem::take(&mut candidates[0].1);
            if o.extend_best && batch_index == 0 {
                let ext = extend_direction(
                    &p,
                    &ids,
                    decode(ranked[candidates[0].0].0, o.direction_epsilon),
                    o,
                );
                if ext.len() > g.len() {
                    g = ext;
                }
            }
            for &id in &g {
                removed[id] = true;
            }
            groups.push(g);
        }
        for (id, &r) in removed.iter().enumerate() {
            if r {
                alive[id] = false;
            }
        }
        ids.retain(|&id| !removed[id]);
    }
    let mut labels = vec![0; input.len()];
    let mut output_groups = Vec::new();
    for (k, g) in groups.iter().enumerate() {
        let mut mapped: Vec<_> = g.iter().map(|&id| unique[id]).collect();
        mapped.sort_unstable();
        for &id in &mapped {
            labels[id] = k + 1;
        }
        output_groups.push(mapped);
    }
    Ok(ResultData {
        labels,
        groups: output_groups,
        mean,
        scale,
        stage1_seconds: stage1,
        stage2_seconds: start.elapsed().as_secs_f64() - stage1,
        compute_seconds: start.elapsed().as_secs_f64(),
        centers_completed: centers,
        occupied_directions: occupied,
        evaluations_completed: evaluations,
        budget_exhausted: work_exhausted || start.elapsed().as_secs_f64() >= o.seconds,
        cancelled: cancel.load(Ordering::Relaxed),
        group_cap_reached: groups.len() >= o.groups,
        duplicate_observations: duplicates,
        effective_smart: Some(direction_options.smart),
        cache_payload_bytes: cache
            .iter()
            .filter_map(Option::as_ref)
            .map(prepared_bytes)
            .sum(),
        cached_directions: cache.iter().filter(|x| x.is_some()).count(),
        cached_point_entries: cache
            .iter()
            .filter_map(Option::as_ref)
            .map(prepared_entries)
            .sum(),
        stage2_prepare_worker_seconds: profile_seconds[0],
        stage2_cached_worker_seconds: profile_seconds[1],
        stage2_uncached_worker_seconds: profile_seconds[2],
        stage2_cache_hits: profile_counts[0],
        stage2_preparations: profile_counts[1],
        stage2_cache_evictions: profile_counts[2],
        cached_candidate_directions: cache
            .iter()
            .filter_map(Option::as_ref)
            .filter(|x| x.candidate_only)
            .count(),
        cached_candidate_point_entries: cache
            .iter()
            .filter_map(Option::as_ref)
            .filter(|x| x.candidate_only)
            .map(prepared_entries)
            .sum(),
        cached_bitset_directions: cache
            .iter()
            .filter_map(Option::as_ref)
            .filter(|x| !x.candidate_bits.is_empty())
            .count(),
        cached_bitset_bytes: cache
            .iter()
            .filter_map(Option::as_ref)
            .map(|x| x.candidate_bits.len() * 8)
            .sum(),
        options: o.clone(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn antipodes() {
        assert_eq!(
            direction_key([-0.36, -0.8, 0.48], 0.01),
            direction_key([0.36, 0.8, -0.48], 0.01)
        );
    }
    #[test]
    fn interval_closed() {
        let o = Options::default();
        assert_eq!(interval_support(&[1., 2., 3.], &o), 3);
        assert_eq!(interval_support(&[1.], &o), 1);
    }
    #[test]
    fn modulo_wrap_and_duplicates() {
        let g = phase_subset(
            &[(-0.001, 0), (0.999, 1), (1.001, 2), (2.001, 3), (3.001, 4)],
            1.,
            0.02,
        );
        assert_eq!(g.len(), 4);
    }
    #[test]
    fn cancel() {
        let c = AtomicBool::new(true);
        let r = run(&[[0., 0., 0.], [1., 0., 0.]], &Options::default(), &c).unwrap();
        assert!(r.cancelled);
        assert_eq!(r.labels, vec![0, 0]);
    }
    #[test]
    fn nonfinite() {
        assert!(run(
            &[[f64::NAN, 0., 0.]],
            &Options::default(),
            &AtomicBool::new(false)
        )
        .is_err());
    }
    #[test]
    fn conservation_degenerate() {
        let q = vec![[0.; 3]; 10];
        let r = run(&q, &Options::default(), &AtomicBool::new(false)).unwrap();
        assert_eq!(r.labels.len(), 10);
        assert_eq!(r.duplicate_observations, 9);
        assert!(r.groups.is_empty());
    }
}

#[cfg(test)]
mod stage2_exact_tests {
    use super::*;
    fn legacy_layout(mut o: Options) -> Options {
        o.compact_cache_ids = false;
        o.prune_gapless_grids = false;
        o.direct_fixed_pack = false;
        o.stream_fixed_rows = false;
        o.fused_projections = false;
        o.smallest_cache_first = false;
        o.filter_sparse_cells = false;
        o.candidate_subset_cache = false;
        o.reuse_phase_output = false;
        o
    }
    #[test]
    fn narrow_cache_preserves_id_and_offset_limits() {
        let a = CompactRows::from_u32(vec![0, 2], vec![0, 65535], true);
        assert!(matches!(a, CompactRows::Narrow { .. }));
        assert_eq!(a.row(0).iter().collect::<Vec<_>>(), vec![0, 65535]);
        let b = CompactRows::from_u32(vec![0, 2], vec![1, 65536], true);
        assert!(matches!(b, CompactRows::Wide { .. }));
        assert_eq!(b.row(0).iter().collect::<Vec<_>>(), vec![1, 65536]);
        let c = CompactRows::from_u32(vec![0, 65535], vec![3; 65535], true);
        assert!(matches!(c, CompactRows::Narrow { .. }));
        assert_eq!(c.row(0).len(), 65535);
        let d = CompactRows::from_u32(vec![0, 65536], vec![3; 65536], true);
        assert!(matches!(d, CompactRows::Wide { .. }));
        assert_eq!(d.row(0).len(), 65536);
    }
    #[test]
    fn known_spacing_does_not_need_a_four_point_gap_row() {
        let p = [[0., 0.101, 0.101], [0.2, 0.101, 0.101], [0.4, 0.101, 0.101]];
        let ids = [0, 1, 2];
        let o = Options {
            projection_grids: 4,
            phase_min_row: 3,
            ..Options::default()
        };
        assert!(group_for_direction(&p, &ids, [1., 0., 0.], &o).is_empty());
        assert_eq!(
            group_at_distance(&p, &ids, [1., 0., 0.], 0.2, &o),
            vec![0, 1, 2]
        );
    }
    #[test]
    fn flat_fused_pruned_rows_match_nested_after_every_deletion_pattern() {
        let mut p = Vec::new();
        for row in 0..8 {
            for site in 0..6 {
                let jitter = if site % 2 == 0 {
                    0.00000001
                } else {
                    -0.00000001
                };
                p.push([
                    site as f64 * 0.08 + row as f64 * 0.0001,
                    row as f64 * 0.004 + 0.002 + jitter,
                    (row % 3) as f64 * 0.004 + 0.004 - jitter,
                ]);
            }
        }
        let ids: Vec<_> = (0..p.len()).collect();
        for grids in [2, 4] {
            for phase in [0, 3] {
                for gap in [0, 2, 3] {
                    for radius in [false, true] {
                        let o = Options {
                            projection_grids: grids,
                            phase_min_row: phase,
                            gap_min_row: gap,
                            phase_radius: radius,
                            relative_gap: true,
                            ..Options::default()
                        };
                        let slow = legacy_layout(o.clone());
                        let prepared = prepare_direction(&p, &ids, [1., 0., 0.], &o);
                        let reference = prepare_direction(&p, &ids, [1., 0., 0.], &slow);
                        for mask in 0..64 {
                            let alive: Vec<_> = (0..p.len())
                                .map(|i| mask == 0 || (i * 37 + mask * 17) % 11 > 3)
                                .collect();
                            assert_eq!(
                                prepared_details(&p, &prepared, &alive, &o),
                                prepared_details(&p, &reference, &alive, &slow)
                            );
                            let active: Vec<_> =
                                ids.iter().copied().filter(|&i| alive[i]).collect();
                            let fresh = prepare_direction(&p, &active, [1., 0., 0.], &o);
                            assert_eq!(
                                prepared_details(&p, &prepared, &alive, &o),
                                prepared_details(&p, &fresh, &alive, &o)
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn fused_projection_keys_match_scalar_at_cell_boundaries() {
        let e = 0.002;
        let v = [0.93, -0.3, 0.21];
        let mut p = Vec::new();
        for i in -20..=20 {
            for (y, z) in [
                (i as f64 * e, (i % 7) as f64 * e),
                ((i as f64 * e).next_up(), ((i % 7) as f64 * e).next_down()),
            ] {
                let t = (i % 5) as f64 * 0.1;
                p.push([v[0] * t, y + v[1] * t, z + v[2] * t]);
            }
        }
        let ids: Vec<_> = (0..p.len()).collect();
        let o = Options {
            projection_grids: 4,
            phase_min_row: 3,
            prune_gapless_grids: false,
            ..Options::default()
        };
        let mut slow = o.clone();
        slow.fused_projections = false;
        let a = prepare_direction(&p, &ids, v, &o);
        let b = prepare_direction(&p, &ids, v, &slow);
        for (ga, gb) in a.grids.iter().zip(&b.grids) {
            assert_eq!(ga.nrows(), gb.nrows());
            for i in 0..ga.nrows() {
                assert_eq!(
                    ga.row(i).iter().collect::<Vec<_>>(),
                    gb.row(i).iter().collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn execution_switches_preserve_complete_grouping() {
        let p: Vec<_> = (-2..=2)
            .flat_map(|i| {
                (-2..=2).flat_map(move |j| {
                    (-2..=2).map(move |k| [i as f64 * 0.19, j as f64 * 0.23, k as f64 * 0.31])
                })
            })
            .collect();
        let o = Options {
            projection_grids: 4,
            phase_min_row: 3,
            relative_gap: true,
            ..Options::default()
        };
        let reference = run(&p, &legacy_layout(o.clone()), &AtomicBool::new(false)).unwrap();
        for which in 0..11 {
            let mut next = o.clone();
            match which {
                0 => next.compact_cache_ids = false,
                1 => next.prune_gapless_grids = false,
                2 => next.direct_fixed_pack = false,
                3 => next.stream_fixed_rows = false,
                4 => next.fused_projections = false,
                5 => next.smallest_cache_first = false,
                6 => next.cache_megabytes = 0,
                7 => next.filter_sparse_cells = false,
                8 => next.candidate_subset_cache = false,
                9 => next.reuse_phase_output = false,
                _ => next.workers = 1,
            }
            let out = run(&p, &next, &AtomicBool::new(false)).unwrap();
            assert_eq!(out.labels, reference.labels);
            assert_eq!(out.groups, reference.groups);
        }
    }
}

#[cfg(test)]
mod eligibility_cache_tests {
    use super::*;
    #[test]
    fn byte_counts_saturate_without_changing_eligibility_and_reset() {
        let input: Vec<_> = (0..600)
            .map(|id| Projected {
                key: if id < 500 { 3 } else { (id % 11) as u64 },
                t: id as f64,
                id,
            })
            .collect();
        let mut scratch = Stage2Scratch::default();
        for keep in [2, 3, 4, 200, 255] {
            for _ in 0..3 {
                let mut actual = input.clone();
                scratch.retain_eligible(&mut actual, 20, keep);
                let expected: Vec<_> = input
                    .iter()
                    .filter(|p| input.iter().filter(|q| q.key == p.key).count() >= keep)
                    .map(|p| p.id)
                    .collect();
                assert_eq!(actual.iter().map(|p| p.id).collect::<Vec<_>>(), expected);
            }
        }
        for (space, keep) in [(20, 256), (2_000_001, 3), (2, 3)] {
            let mut actual = input.clone();
            scratch.retain_eligible(&mut actual, space, keep);
            assert_eq!(
                actual.iter().map(|p| p.id).collect::<Vec<_>>(),
                input.iter().map(|p| p.id).collect::<Vec<_>>()
            );
        }
        let mut actual = input.clone();
        scratch.retain_eligible(&mut actual, 20, 3);
        assert_eq!(actual.len(), input.len());
    }
    #[test]
    fn candidate_cache_keeps_geometric_support_for_a_later_spacing_mode() {
        let mut p = Vec::new();
        for t in [0., 0.1, 0.2, 0.3] {
            p.push([t, 0.101, 0.101]);
        }
        for t in [0., 0.17, 0.34, 0.51] {
            p.push([t, 0.201, 0.201]);
        }
        for t in [0., 0.17, 0.34] {
            p.push([t, 0.301, 0.301]);
        }
        let ids: Vec<_> = (0..p.len()).collect();
        let o = Options {
            projection_grids: 4,
            phase_min_row: 3,
            relative_gap: true,
            ..Options::default()
        };
        let rows = prepare_direction(&p, &ids, [1., 0., 0.], &o);
        let cached = candidate_cache(&rows, &o, p.len());
        let all = vec![true; p.len()];
        let later: Vec<_> = (0..p.len()).map(|i| i >= 4).collect();
        let before = prepared_group(&p, &rows, &all, &o);
        let after = prepared_group(&p, &rows, &later, &o);
        assert!(before.len() < after.len());
        assert_eq!(after, vec![4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(prepared_group(&p, &cached, &all, &o), before);
        assert_eq!(prepared_group(&p, &cached, &later, &o), after);
    }
    #[test]
    fn candidate_superset_preserves_all_fixed_grid_deletions() {
        let mut p = Vec::new();
        for i in 0..80 {
            p.push([
                (i % 8) as f64 * 0.05 + (i / 8) as f64 * 0.00001,
                (i / 8) as f64 * 0.004 + if i % 2 == 0 { 0.00199999 } else { 0.00200001 },
                ((i / 8) % 3) as f64 * 0.004 + if i % 3 == 0 { 0.00399999 } else { 0.00400001 },
            ]);
        }
        let ids: Vec<_> = (0..p.len()).collect();
        for grids in [2, 4] {
            for phase in [0, 3] {
                for gap in [0, 2, 3] {
                    let o = Options {
                        projection_grids: grids,
                        phase_min_row: phase,
                        gap_min_row: gap,
                        relative_gap: true,
                        ..Options::default()
                    };
                    let rows = prepare_direction(&p, &ids, [1., 0., 0.], &o);
                    let cached = candidate_cache(&rows, &o, p.len());
                    for mask in 0..128 {
                        let alive: Vec<_> = (0..p.len())
                            .map(|i| mask == 0 || (i * 37 + mask * 17) % 19 > 5)
                            .collect();
                        assert_eq!(
                            prepared_group(&p, &rows, &alive, &o),
                            prepared_group(&p, &cached, &alive, &o)
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod mask_and_phase_tests {
    use super::*;
    #[test]
    fn eligibility_bitset_handles_word_edges_and_sparse_fallback() {
        let o = Options::default();
        let rows = Prepared {
            candidate_only: false,
            candidate_bits: vec![],
            grids: vec![CompactRows::from_u32(
                vec![0, 6],
                vec![0, 63, 64, 65, 127, 129],
                true,
            )],
            stable: vec![],
            direction: [1., 0., 0.],
        };
        let compact = candidate_cache(&rows, &o, 130);
        assert_eq!(compact.candidate_bits.len(), 3);
        let mut alive = vec![true; 130];
        alive[63] = false;
        assert_eq!(candidate_ids(&compact, &alive), vec![0, 64, 65, 127, 129]);
        let sparse = candidate_cache(&rows, &o, 100_000);
        assert!(sparse.candidate_bits.is_empty());
        assert_eq!(
            candidate_ids(&sparse, &vec![true; 100_000]),
            vec![0, 63, 64, 65, 127, 129]
        );
        let empty = Prepared {
            candidate_only: false,
            candidate_bits: vec![],
            grids: vec![CompactRows::pack(vec![], true)],
            stable: vec![],
            direction: [1., 0., 0.],
        };
        let cached = candidate_cache(&empty, &o, 0);
        assert!(candidate_ids(&cached, &[]).is_empty());
    }
    #[test]
    fn reused_bucketed_phase_output_matches_original_all_row_sizes() {
        let mut state = 573u64;
        let mut out = vec![usize::MAX; 100];
        for n in 0..48 {
            for _ in 0..20 {
                let row: Vec<_> = (0..n)
                    .map(|id| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((state >> 32) as f64 / u32::MAX as f64 * 4. - 2., id)
                    })
                    .collect();
                for radius in [false, true] {
                    for fast in [false, true] {
                        let o = Options {
                            phase_radius: radius,
                            fast_small_rows: fast,
                            ..Options::default()
                        };
                        phase_for_into(&row, 0.17, &o, &mut out);
                        assert_eq!(out, phase_for(&row, 0.17, &o));
                    }
                }
            }
        }
        for row in [
            vec![(0., 0), (1.25, 1)],
            vec![(0., 0), (1.2500001, 1)],
            vec![(-0.001, 0), (0.999, 1), (1.001, 2), (2.001, 3)],
        ] {
            phase_subset_into::<4, 8>(&row, 1., 0.25, &mut out);
            assert_eq!(out, phase_subset_generic(&row, 1., 0.25));
        }
    }
}

#[cfg(test)]
mod remaining_stage2_tests {
    use super::*;
    #[test]
    fn complete_triplet_matches_generic_at_wrapped_and_closed_boundaries() {
        let values = [
            -2., -1.25, -1., -0.02, 0., 0.02, 0.25, 0.98, 1., 1.02, 1.25, 2.,
        ];
        for d in [0.17, 1.] {
            for a in values {
                for b in values {
                    for c in values {
                        let row = [(a * d, 9), (b * d, 1), (c * d, 5)];
                        for eps in [0., 0.02, 0.25, 0.49, 0.98] {
                            assert_eq!(
                                complete_phase_triplet(&row, d, eps),
                                phase_subset_generic(&row, d, eps).len() == 3,
                                "{row:?} d={d} eps={eps}"
                            );
                        }
                    }
                }
            }
        }
        for t in [2.02_f64.next_down(), 2.02, 2.02_f64.next_up()] {
            let row = [(0., 0), (1., 1), (t, 2)];
            assert_eq!(
                complete_phase_triplet(&row, 1., 0.02),
                phase_subset_generic(&row, 1., 0.02).len() == 3
            );
        }
        assert!(!complete_phase_triplet(
            &[(0., 0), (1., 1), (2., 2)],
            1e-12,
            0.02
        ));
    }
    #[test]
    fn shared_emission_matches_original_filter_and_resets_counts() {
        let mut scratch = Stage2Scratch::default();
        for n in [600, 0, 3, 100, 600] {
            let ids: Vec<_> = (0..n).map(|id| (id * 37) % n).collect();
            scratch.shared_t = (0..n).map(|id| id as f64 * 0.17).collect();
            for grid in 0..4 {
                scratch.shared_keys[grid] = (0..n)
                    .map(|id| match grid {
                        0 => {
                            if id < 500 {
                                3
                            } else {
                                (id % 11) as u64
                            }
                        }
                        1 => (id % 31) as u64,
                        2 => (id % 2) as u64,
                        _ => (id % 61) as u64,
                    })
                    .collect();
            }
            for grid in 0..4 {
                for space in [20, 64, 2_000_001] {
                    for keep in [2, 3, 200, 255, 256] {
                        for filter in [true, false] {
                            let mut expected: Vec<_> = scratch.shared_keys[grid]
                                .iter()
                                .enumerate()
                                .map(|(i, &key)| Projected {
                                    key,
                                    t: scratch.shared_t[i],
                                    id: ids[i],
                                })
                                .collect();
                            if filter {
                                Stage2Scratch::default().retain_eligible(
                                    &mut expected,
                                    space,
                                    keep,
                                );
                            }
                            let mut actual = Vec::new();
                            scratch.emit_grid(&mut actual, grid, space, keep, filter, &ids);
                            assert_eq!(
                                actual
                                    .iter()
                                    .map(|x| (x.key, x.t.to_bits(), x.id))
                                    .collect::<Vec<_>>(),
                                expected
                                    .iter()
                                    .map(|x| (x.key, x.t.to_bits(), x.id))
                                    .collect::<Vec<_>>()
                            );
                            assert!(scratch.cell_counts.iter().all(|&n| n == 0));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn reused_radix_storage_preserves_order_across_lengths_and_pass_counts() {
        let mut scratch = Vec::new();
        let mut state = 573u64;
        for max_key in [0, 1, 1023, 1024, 1 << 30, u64::MAX] {
            for n in [257, 0, 3, 128] {
                let mut actual: Vec<_> = (0..n)
                    .map(|id| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        Projected {
                            key: state & max_key,
                            t: id as f64,
                            id,
                        }
                    })
                    .collect();
                let mut expected = actual.clone();
                expected.sort_by_key(|x| x.key);
                radix_projected_scratch(&mut actual, max_key, &mut scratch);
                assert_eq!(
                    actual.iter().map(|x| (x.key, x.id)).collect::<Vec<_>>(),
                    expected.iter().map(|x| (x.key, x.id)).collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn reused_flat_rows_match_nested_reference_through_deletions() {
        let p: Vec<_> = (-2..=2)
            .flat_map(|i| {
                (-2..=2).flat_map(move |j| {
                    (-2..=2).map(move |k| {
                        [
                            i as f64 * 0.19 + j as f64 * 0.001,
                            j as f64 * 0.23,
                            k as f64 * 0.31,
                        ]
                    })
                })
            })
            .collect();
        let ids: Vec<_> = (0..p.len()).collect();
        let mut scratch = FlatScratch::default();
        for grids in [2, 4] {
            for phase in [3, 4] {
                let o = Options {
                    projection_grids: grids,
                    phase_min_row: phase,
                    relative_gap: true,
                    ..Options::default()
                };
                for v in [[1., 0., 0.], [0., 1., 0.], [0.93, -0.3, 0.21]] {
                    let rows = prepare_direction(&p, &ids, v, &o);
                    for mask in 0..32 {
                        let alive: Vec<_> = (0..p.len())
                            .map(|i| mask == 0 || (i * 37 + mask * 17) % 19 > 5)
                            .collect();
                        let a = prepared_details_flat(&p, &rows, &alive, &o, &mut scratch);
                        let b = prepared_details_nested(&p, &rows, &alive, &o);
                        assert_eq!(
                            (a.0, a.1.map(f64::to_bits), a.2),
                            (b.0, b.1.map(f64::to_bits), b.2)
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    fn rows(n: usize) -> Prepared {
        Prepared {
            candidate_only: false,
            candidate_bits: vec![],
            stable: vec![],
            direction: [1., 0., 0.],
            grids: (0..4)
                .map(|_| CompactRows::from_u32(vec![0, n as u32], (0..n as u32).collect(), true))
                .collect(),
        }
    }
    fn check_bytes(cache: &CacheAdmission, existing: usize) {
        let actual = existing
            + cache
                .entries
                .iter()
                .flatten()
                .map(|(_, p)| prepared_bytes(p))
                .sum::<usize>();
        assert_eq!(cache.bytes, actual);
        if let Some(cap) = cache.cap {
            assert!(actual <= cap);
        }
    }
    #[test]
    fn full_rows_use_available_capacity_without_per_direction_quota() {
        let o = Options::default();
        let sizes = [4, 10, 20, 500, 1000];
        let total = sizes
            .iter()
            .map(|&n| prepared_bytes(&rows(n)))
            .sum::<usize>();
        let mut cache = CacheAdmission::new(37, Some(37 + total));
        for (rank, &n) in sizes.iter().enumerate() {
            assert_eq!(cache.admit(rank, rows(n), &o, 2048), 0);
            check_bytes(&cache, 37);
        }
        assert_eq!(cache.entries.len(), sizes.len());
        assert_eq!(cache.compressions, 0);
        assert!(cache
            .entries
            .iter()
            .flatten()
            .all(|(_, p)| !p.candidate_only));
    }
    #[test]
    fn pressure_compacts_geometric_supersets_before_losing_directions() {
        let o = Options::default();
        let sizes: Vec<_> = (0..40).map(|i| 3 + (i * 137) % 1000).collect();
        let minimum = sizes
            .iter()
            .map(|&n| {
                let p = rows(n);
                prepared_bytes(&p).min(prepared_bytes(&candidate_cache(&p, &o, 2048)))
            })
            .sum::<usize>();
        for extra in [0, 300, 1000] {
            let mut cache = CacheAdmission::new(37, Some(37 + minimum + extra));
            for (rank, &n) in sizes.iter().enumerate() {
                assert_eq!(cache.admit(rank, rows(n), &o, 2048), 0);
                check_bytes(&cache, 37);
            }
            assert_eq!(cache.entries.len(), sizes.len());
            assert!(cache.compressions > 0);
            for (rank, p) in cache.entries.iter().flatten() {
                if p.candidate_only {
                    assert_eq!(
                        candidate_ids(p, &vec![true; 2048]),
                        (0..sizes[*rank]).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[test]
    fn stale_heap_entries_and_eviction_respect_payload_at_tight_budgets() {
        for cap in [0, 300, 1000, 5000] {
            for compact in [false, true] {
                let o = Options {
                    candidate_subset_cache: compact,
                    ..Options::default()
                };
                let mut cache = CacheAdmission::new(0, Some(cap));
                for rank in 0..200 {
                    cache.admit(rank, rows(3 + (rank * 137) % 1000), &o, 2048);
                    check_bytes(&cache, 0);
                    let maximum = cache
                        .entries
                        .iter()
                        .flatten()
                        .map(|(_, p)| prepared_bytes(p))
                        .max();
                    assert_eq!(cache.largest_live().map(|(size, _)| size), maximum);
                }
            }
        }
        let o = Options::default();
        let mut unlimited = CacheAdmission::new(0, None);
        for rank in 0..20 {
            unlimited.admit(rank, rows(1000), &o, 2048);
        }
        assert_eq!(unlimited.entries.len(), 20);
        assert_eq!(unlimited.compressions, 0);
        assert!(unlimited
            .entries
            .iter()
            .flatten()
            .all(|(_, p)| !p.candidate_only));
    }
    #[test]
    fn pressure_compaction_preserves_groups_after_deletion() {
        let p: Vec<_> = (-2..=2)
            .flat_map(|i| {
                (-2..=2).flat_map(move |j| {
                    (-2..=2).map(move |k| {
                        [
                            i as f64 * 0.19 + j as f64 * 0.001,
                            j as f64 * 0.23,
                            k as f64 * 0.31,
                        ]
                    })
                })
            })
            .collect();
        let ids: Vec<_> = (0..p.len()).collect();
        let o = Options {
            projection_grids: 4,
            phase_min_row: 3,
            relative_gap: true,
            ..Options::default()
        };
        let vv = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.93, -0.3, 0.21]];
        let minimum = vv
            .iter()
            .map(|&v| {
                let r = prepare_direction(&p, &ids, v, &o);
                prepared_bytes(&r).min(prepared_bytes(&candidate_cache(&r, &o, p.len())))
            })
            .sum();
        let mut cache = CacheAdmission::new(0, Some(minimum));
        for (rank, &v) in vv.iter().enumerate() {
            cache.admit(rank, prepare_direction(&p, &ids, v, &o), &o, p.len());
            check_bytes(&cache, 0);
        }
        assert_eq!(cache.entries.len(), vv.len());
        assert!(cache.compressions > 0);
        for (rank, r) in cache.entries.iter().flatten() {
            let original = prepare_direction(&p, &ids, vv[*rank], &o);
            for mask in 0..32 {
                let alive: Vec<_> = (0..p.len())
                    .map(|i| mask == 0 || (i * 37 + mask * 17) % 19 > 5)
                    .collect();
                assert_eq!(
                    prepared_group(&p, r, &alive, &o),
                    prepared_group(&p, &original, &alive, &o)
                );
            }
        }
    }
}
