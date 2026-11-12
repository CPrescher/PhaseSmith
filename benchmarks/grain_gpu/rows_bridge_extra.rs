// Analysis-only FFI and GPU-preparation hook; never compiled into production crates.
type SortCallback = unsafe extern "C" fn(
    *const u32,
    usize,
    usize,
    usize,
    *mut u32,
    usize,
    *const f64,
    *const f64,
    usize,
    f64,
) -> usize;
thread_local! {
    static GPU_PROJECTIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static BATCH_ELEMENTS: std::cell::Cell<usize> = const { std::cell::Cell::new(8_388_608) };
    static SORT_CALLBACK: std::cell::Cell<Option<SortCallback>> = const { std::cell::Cell::new(None) };
}

fn gpu_initial_preparations(
    p: &[Point],
    ranked: &[(usize, u64)],
    o: &Options,
    start: Instant,
    cancel: &AtomicBool,
) -> Result<Vec<std::sync::Mutex<Option<Prepared>>>, String> {
    let Some(callback) = SORT_CALLBACK.with(|c| c.get()) else {
        return Ok(Vec::new());
    };
    if ![2, 4].contains(&o.projection_grids) || !o.direct_fixed_pack || !o.fused_projections {
        return Err(
            "GPU prototype supports fixed 2/4 projection grids and fused/direct packing only"
                .into(),
        );
    }
    let width = 2. * o.projection_epsilon;
    let bias = (2. / width).ceil() as i64 + 2;
    let side = 2 * bias + 1;
    if side * side > u32::MAX as i64 {
        return Err("GPU cell keys exceed u32".into());
    }
    // Default at most 32 MiB input keys per GPU batch; output scratch at most 96 MiB.
    let batch_size =
        (BATCH_ELEMENTS.with(|x| x.get()) / (o.projection_grids * p.len().max(1))).clamp(1, 4096);
    let mut result = Vec::with_capacity(ranked.len());
    for batch in ranked.chunks(batch_size) {
        if expired(start, o, cancel) {
            break;
        }
        let directions: Vec<Point> = batch
            .iter()
            .map(|&(key, _)| decode(key, o.direction_epsilon))
            .collect();
        let n = p.len();
        let grids = o.projection_grids;
        let skip_cpu_projection = GPU_PROJECTIONS.with(|x| x.get());
        let mut keys = if skip_cpu_projection {
            Vec::new()
        } else {
            vec![0u32; directions.len() * grids * n]
        };
        if !skip_cpu_projection {
            std::thread::scope(|scope| {
                let chunk_directions = directions.len().div_ceil(o.workers);
                for (part, output) in keys.chunks_mut(chunk_directions * grids * n).enumerate() {
                    let vectors = &directions[part * chunk_directions
                        ..(part * chunk_directions + output.len() / (grids * n))];
                    scope.spawn(move || {
                        for (di, &v) in vectors.iter().enumerate() {
                            let k = axis(v);
                            let ij = others(k);
                            for (id, point) in p.iter().enumerate() {
                                let t = point[k] / v[k];
                                let a = point[ij[0]] - v[ij[0]] * t;
                                let b = point[ij[1]] - v[ij[1]] * t;
                                let a0 = ((a + 0. * o.projection_epsilon) / width).floor() as i32;
                                let a1 = ((a + 1. * o.projection_epsilon) / width).floor() as i32;
                                let b0 = ((b + 0. * o.projection_epsilon) / width).floor() as i32;
                                let b1 = ((b + 1. * o.projection_epsilon) / width).floor() as i32;
                                for (g, (aa, bb)) in [(a0, b0), (a1, b1), (a1, b0), (a0, b1)]
                                    .into_iter()
                                    .take(grids)
                                    .enumerate()
                                {
                                    output[(di * grids + g) * n + id] =
                                        ((i64::from(aa) + bias) * side + i64::from(bb) + bias)
                                            as u32;
                                }
                            }
                        }
                    });
                }
            });
        }
        let capacity = directions.len() * grids * n;
        let mut retained = vec![0u32; capacity * 3];
        let count = unsafe {
            callback(
                keys.as_ptr(),
                directions.len() * grids,
                n,
                keep_size(o),
                retained.as_mut_ptr(),
                capacity,
                p.as_ptr().cast::<f64>(),
                directions.as_ptr().cast::<f64>(),
                grids,
                o.projection_epsilon,
            )
        };
        if count > capacity {
            return Err("GPU sort callback failed or exceeded output capacity".into());
        }
        retained.truncate(count * 3);
        // Compact GPU output arrives in arbitrary atomic-counter order. Restore
        // exact key, longitudinal-f64, original-id ordering on retained records.
        let mut records: Vec<(usize, u32, f64, usize)> = retained
            .chunks_exact(3)
            .map(|r| {
                let row = r[0] as usize;
                let id = r[2] as usize;
                assert!(row < directions.len() * grids && id < n);
                let v = directions[row / grids];
                let k = axis(v);
                (row, r[1], p[id][k] / v[k], id)
            })
            .collect();
        records.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.total_cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        let mut cursor = 0;
        for (di, &direction) in directions.iter().enumerate() {
            let mut packed = Vec::with_capacity(grids);
            for g in 0..grids {
                let row = di * grids + g;
                let mut offsets = vec![0u32];
                let mut ids = Vec::new();
                let mut gap_row = false;
                while cursor < records.len() && records[cursor].0 == row {
                    let key = records[cursor].1;
                    let start = ids.len();
                    while cursor < records.len()
                        && records[cursor].0 == row
                        && records[cursor].1 == key
                    {
                        ids.push(records[cursor].3 as u32);
                        cursor += 1;
                    }
                    gap_row |= ids.len() - start >= gap_size(o);
                    offsets.push(ids.len() as u32);
                }
                packed.push(if o.prune_gapless_grids && !gap_row {
                    CompactRows::pack(vec![], o.compact_cache_ids)
                } else {
                    CompactRows::from_u32(offsets, ids, o.compact_cache_ids)
                });
            }
            result.push(std::sync::Mutex::new(Some(Prepared {
                candidate_only: false,
                candidate_bits: vec![],
                grids: packed,
                stable: vec![],
                direction,
            })));
        }
        assert_eq!(cursor, records.len());
    }
    Ok(result)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn row_gpu_run(
    points: *const f64,
    n: usize,
    options: *const std::ffi::c_char,
    callback: Option<SortCallback>,
) -> *mut std::ffi::c_char {
    let result = std::panic::catch_unwind(|| {
        let input = unsafe { std::slice::from_raw_parts(points.cast::<Point>(), n) };
        let text = unsafe { std::ffi::CStr::from_ptr(options) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let options: Options = serde_json::from_str(text).map_err(|e| e.to_string())?;
        SORT_CALLBACK.with(|c| c.set(callback));
        let result = run(input, &options, &AtomicBool::new(false));
        SORT_CALLBACK.with(|c| c.set(None));
        result.and_then(|r| serde_json::to_string(&r).map_err(|e| e.to_string()))
    });
    let text = match result {
        Ok(Ok(text)) => text,
        Ok(Err(error)) => serde_json::json!({"error":error}).to_string(),
        Err(_) => {
            SORT_CALLBACK.with(|c| c.set(None));
            "{\"error\":\"native prototype panicked\"}".into()
        }
    };
    std::ffi::CString::new(text).unwrap().into_raw()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn row_gpu_free(ptr: *mut std::ffi::c_char) {
    if !ptr.is_null() {
        drop(unsafe { std::ffi::CString::from_raw(ptr) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn row_gpu_set_batch(elements: usize) {
    BATCH_ELEMENTS.with(|x| x.set(elements));
}

#[unsafe(no_mangle)]
pub extern "C" fn row_gpu_set_projection(enabled: bool) {
    GPU_PROJECTIONS.with(|x| x.set(enabled));
}
