//! Original-count HDF5 to candidate search, with no interpreter or external solver.
// Source frame and array dimensions are checked by native admission below 1e8.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
use phasesmith::{
    crystallography::UnitCell,
    hdf5::{Limits, cache::CachedReader},
    io::space_group_by_number,
    persistence::numeric_archive::{NumericArray, write_numeric_npz},
    solution,
    spots::{frame_input, reciprocal::Geometry},
    workflows::single_crystal as sc,
};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
struct Source(CachedReader);
impl sc::FrameSource for Source {
    fn shape(&self) -> [usize; 3] {
        self.0.shape()
    }
    fn read(&mut self, i: usize) -> sc::Result<Vec<u32>> {
        self.0
            .read_frame(i)
            .map(|f| f.pixels)
            .map_err(|e| sc::Error(e.to_string()))
    }
}
fn json_file(path: &Path, v: &Value) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, v)?;
    f.write_all(b"\n")?;
    Ok(())
}
struct Array {
    key: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
fn f64s(key: &str, shape: Vec<usize>, values: impl IntoIterator<Item = f64>) -> Array {
    Array {
        key: key.into(),
        dtype: "<f8",
        shape,
        bytes: values.into_iter().flat_map(f64::to_le_bytes).collect(),
    }
}
fn i64s(key: &str, shape: Vec<usize>, values: impl IntoIterator<Item = i64>) -> Array {
    Array {
        key: key.into(),
        dtype: "<i8",
        shape,
        bytes: values.into_iter().flat_map(i64::to_le_bytes).collect(),
    }
}
fn flags(key: &str, values: &[u32]) -> Array {
    Array {
        key: key.into(),
        dtype: "<u4",
        shape: vec![values.len()],
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}
fn booleans(key: &str, shape: Vec<usize>, values: &[bool]) -> Array {
    Array {
        key: key.into(),
        dtype: "|b1",
        shape,
        bytes: values.iter().map(|&v| u8::from(v)).collect(),
    }
}
fn npz(path: &Path, a: &[Array]) -> Result<()> {
    let views: Vec<_> = a
        .iter()
        .map(|a| NumericArray {
            name: &a.key,
            dtype: a.dtype,
            shape: &a.shape,
            bytes: &a.bytes,
        })
        .collect();
    write_numeric_npz(path, &views, 1)?;
    Ok(())
}
fn boxes(key: &str, b: &[phasesmith::spots::Bounds]) -> Array {
    i64s(
        key,
        vec![b.len(), 4],
        b.iter()
            .flat_map(|b| [b.x0 as i64, b.y0 as i64, b.x1 as i64, b.y1 as i64]),
    )
}
macro_rules! record{($value:expr;$($field:ident),* $(,)?)=>{{let v=$value;let mut m=serde_json::Map::new();$(m.insert(stringify!($field).into(),json!(v.$field));)*Value::Object(m)}};}
fn candidate(c: &solution::Candidate) -> Value {
    let mut v = record!(c;species,fractional_xyz,u_iso,amplitude_scale,r1,objective,trial,polish_steps,element_swaps,refinement_diagnostic);
    v["completion"] = Value::Null;
    v["refinement"]=c.refinement.as_ref().map_or(Value::Null,|r|record!(r;species,fractional_xyz,u_cartesian,u_cif,u_equiv,full_u_cartesian,full_u_cif,full_u_equiv,hand,origin_shift,maximum_matching_error,parameter_count,coordinate_parameter_count,origin_gauge_dimension,mode,termination,accepted_steps,p1_r1,p1_objective));
    v
}
struct Save {
    root: PathBuf,
    timings: Vec<(String, f64)>,
    shape: [usize; 2],
    cell: [f64; 6],
    priors: Value,
    angles: Vec<[f64; 2]>,
}
impl Save {
    fn archive(&self, name: &str, a: &[Array]) -> sc::Result<()> {
        npz(&self.root.join(name), a).map_err(|e| sc::Error(e.to_string()))
    }
    #[allow(clippy::needless_pass_by_value)] // Consume the detached output record immediately after serialization.
    fn json(&self, name: &str, v: Value) -> sc::Result<()> {
        json_file(&self.root.join(name), &v).map_err(|e| sc::Error(e.to_string()))
    }
}
impl sc::Observer for Save {
    fn stage(&mut self, name: &str, seconds: f64) -> sc::Result<()> {
        eprintln!("{name}: {seconds:.6} s");
        self.timings.push((name.into(), seconds));
        self.json(
            &format!("timing-{name}.json"),
            json!({"stage":name,"seconds":seconds}),
        )
    }
    fn masks(&mut self, p: &frame_input::Plan, scales: &[f64]) -> sc::Result<()> {
        let a = &p.blocks[0];
        self.archive(
            "frame-validity.npz",
            &[
                booleans(
                    "excluded",
                    vec![
                        p.frames,
                        self.shape[0].div_ceil(a.block),
                        self.shape[1].div_ceil(a.block),
                    ],
                    &a.excluded,
                ),
                f64s(
                    "omega_bounds",
                    vec![p.frames, 2],
                    self.angles.iter().flatten().copied(),
                ),
                f64s(
                    "detection_scales",
                    vec![scales.len()],
                    scales.iter().copied(),
                ),
            ],
        )?;
        self.json("frame-validity.json",json!({"raw_frame_hashes":p.expected_hashes,"block":a.block,"orientation_vertical_flip":p.vertical_flip,"reserved_minimum":p.reserved_minimum,"providers":["original-count illumination","whole-exposure DAC cone"],"normalization_of_counts":false}))
    }
    fn positions(&mut self, p: &sc::Positions) -> sc::Result<()> {
        self.archive(
            "positions.npz",
            &[
                boxes("boxes", &p.boxes),
                i64s(
                    "frame_ranges",
                    vec![p.ranges.len(), 2],
                    p.ranges.iter().flatten().map(|&x| x as i64),
                ),
                f64s(
                    "values",
                    vec![p.values.len(), 5],
                    p.values.iter().flatten().copied(),
                ),
                flags("flags", &p.flags),
                flags("original_flags", &p.original_flags),
                f64s(
                    "observations",
                    vec![p.observations.len(), 5],
                    p.observations.iter().flatten().copied(),
                ),
                f64s(
                    "omega_bounds",
                    vec![p.angles.len(), 2],
                    p.angles.iter().flatten().copied(),
                ),
            ],
        )?;
        self.json("positions.json",json!({"raw_frame_hashes":p.hashes,"linked_tracks":p.values.len(),"frame_detections":p.observations.len()}))
    }
    fn indexed(&mut self, p: &sc::Indexed) -> sc::Result<()> {
        let f = &p.fit;
        self.archive(
            "indexed-positions.npz",
            &[
                i64s(
                    "track_ids",
                    vec![p.track_ids.len()],
                    p.track_ids.iter().map(|&i| i as i64),
                ),
                i64s("hkl", vec![p.hkl.len(), 3], p.hkl.iter().flatten().copied()),
                f64s(
                    "observed",
                    vec![p.observed.len(), 3],
                    p.observed.iter().flatten().copied(),
                ),
                f64s(
                    "predicted",
                    vec![f.predictions.len(), 3],
                    f.predictions.iter().flat_map(|p| p.values),
                ),
                booleans("eligible", vec![p.eligible.len()], &p.eligible),
                booleans("training", vec![p.training.len()], &p.training),
            ],
        )?;
        self.json("index-candidates.json",json!({"candidates":p.grains.iter().map(|g|json!({"direct_basis":g.direct,"training_matches":g.training,"validation_matches":g.withheld,"accepted":g.accepted,"null_counts":g.null_counts})).collect::<Vec<_>>()}))?;
        self.json("geometry.json",json!({"chosen_candidate":p.selected,"reciprocal_basis":f.state.basis,"geometry_vectors":[f.state.geometry.origin,f.state.geometry.fast,f.state.geometry.slow,f.state.geometry.beam,f.state.geometry.axis],"parameters":f.parameters,"converged":f.converged,"at_bounds":f.at_bounds,"evaluations":f.evaluations,"objective_history":f.history,"rms":f.rms,"counts":f.counts,"reference_coordinates_used":false,"priors":self.priors}))
    }
    #[allow(clippy::too_many_lines)] // One explicit lossless archive schema; no numerical algorithm.
    fn profiles(&mut self, m: &sc::Measurements) -> sc::Result<()> {
        let d = &m.data;
        let n = m.hkl.len();
        self.archive(
            "edge-exclusions.npz",
            &[
                i64s(
                    "hkl",
                    vec![m.excluded_hkl.len(), 3],
                    m.excluded_hkl.iter().flatten().map(|&x| i64::from(x)),
                ),
                f64s(
                    "predicted",
                    vec![m.excluded_centres.len(), 3],
                    m.excluded_centres.iter().flatten().copied(),
                ),
                f64s(
                    "aperture",
                    vec![m.excluded_values.len(), 6],
                    m.excluded_values.iter().flatten().copied(),
                ),
                flags("flags", &m.excluded_flags),
            ],
        )?;
        self.archive(
            "raw-observations-and-bank.npz",
            &[
                i64s(
                    "table_hkl",
                    vec![n, 3],
                    m.hkl.iter().flatten().map(|&h| i64::from(h)),
                ),
                f64s(
                    "table_predicted",
                    vec![n, 3],
                    d.centres.iter().flatten().copied(),
                ),
                i64s("table_grain_ids", vec![n], vec![0; n]),
                f64s(
                    "table_values",
                    vec![n, 6],
                    m.aperture.iter().flatten().copied(),
                ),
                flags("table_flags", &m.flags),
                i64s(
                    "table_observation_ids",
                    vec![d.ids.len()],
                    d.ids.iter().map(|&x| x as i64),
                ),
                i64s(
                    "table_frame_indices",
                    vec![d.frames.len()],
                    d.frames.iter().map(|&x| x as i64),
                ),
                boxes("table_boxes", &d.boxes),
                f64s(
                    "table_frame_values",
                    vec![d.frame_values.len(), 11],
                    d.frame_values.iter().flatten().copied(),
                ),
                flags("table_frame_flags", &d.flags),
                f64s(
                    "table_omega_bounds",
                    vec![self.angles.len(), 2],
                    self.angles.iter().flatten().copied(),
                ),
                f64s(
                    "table_background_models",
                    vec![d.background.len(), 12],
                    d.background.iter().flatten().copied(),
                ),
                f64s(
                    "table_angular_half_widths",
                    vec![n],
                    d.scales.iter().map(|s| (0.9 / s[2].abs() + 0.5).min(4.5)),
                ),
                f64s(
                    "obs_samples",
                    vec![d.samples.len(), 3],
                    d.samples.iter().flatten().copied(),
                ),
                booleans("obs_valid", vec![d.valid.len()], &d.valid),
                i64s(
                    "obs_offsets",
                    vec![d.offsets.len()],
                    d.offsets.iter().map(|&x| x as i64),
                ),
                f64s(
                    "obs_covariance",
                    vec![d.covariance.len(), 9],
                    d.covariance.iter().flatten().copied(),
                ),
                f64s(
                    "obs_centres",
                    vec![n, 3],
                    d.centres.iter().flatten().copied(),
                ),
                f64s("obs_scales", vec![n, 3], d.scales.iter().flatten().copied()),
                f64s(
                    "obs_planar_transforms",
                    vec![n, 2, 2],
                    d.planar.iter().flatten().copied(),
                ),
                f64s(
                    "bank_profiles",
                    vec![m.bank.counts.len(), m.grid.size()],
                    m.bank.profiles.iter().copied(),
                ),
                i64s(
                    "bank_counts",
                    vec![m.bank.counts.len()],
                    m.bank.counts.iter().map(|&x| x as i64),
                ),
                f64s(
                    "bank_shifts",
                    vec![m.bank.shifts.len(), 3],
                    m.bank.shifts.iter().flatten().copied(),
                ),
                i64s(
                    "bank_reference_ids",
                    vec![m.bank.used.iter().filter(|&&x| x).count()],
                    m.bank
                        .used
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &x)| x.then_some(i as i64)),
                ),
                f64s(
                    "bank_reference_centres",
                    vec![m.bank.used.iter().filter(|&&x| x).count(), 3],
                    m.bank
                        .reference_centres
                        .iter()
                        .zip(&m.bank.used)
                        .filter(|(_, u)| **u)
                        .flat_map(|(p, _)| p.iter().copied()),
                ),
            ],
        )?;
        self.json("measurement.json",json!({"target_predictions":m.target_count,"all_predictions":m.all_count,"retained":n,"samples":d.samples.len(),"teachers":m.bank.used.iter().filter(|&&x|x).count(),"learning_workers":m.bank.execution.workers,"learning_cache_bytes":m.bank.execution.cache_bytes}))
    }
    fn reduced(&mut self, r: &sc::Reduced) -> sc::Result<()> {
        let n = r.hkl.len();
        self.archive(
            "reduced-observations.npz",
            &[
                i64s(
                    "hkl",
                    vec![n, 3],
                    r.hkl.iter().flatten().map(|&x| i64::from(x)),
                ),
                f64s("intensity", vec![n], r.values.iter().map(|r| r[0])),
                f64s("sigma", vec![n], r.values.iter().map(|r| r[1])),
                f64s(
                    "statistical_sigma",
                    vec![n],
                    r.statistical_sigma.iter().copied(),
                ),
                flags("flags", &r.flags),
                booleans("use", vec![n], &r.use_mask),
                booleans("visible", vec![n], &r.visible),
                f64s(
                    "raw_profiles",
                    vec![n, 10],
                    r.profiles.iter().flatten().copied(),
                ),
                f64s(
                    "parameters",
                    vec![n, 5],
                    r.parameters.iter().flatten().copied(),
                ),
                i64s("choices", vec![n], r.choices.iter().map(|&x| i64::from(x))),
            ],
        )?;
        self.json("reduction.json",json!({"used":r.use_mask.iter().filter(|&&x|x).count(),"negative_used":r.values.iter().zip(&r.use_mask).filter(|(v,u)|**u&&v[0]<0.).count(),"error_floor":r.error_floor,"scaling_coefficients":r.scaling.coefficients,"converged":r.scaling.converged}))
    }
    fn solution(
        &mut self,
        s: &solution::Solution,
        search: &solution::SharedSearchSummary,
    ) -> sc::Result<()> {
        fs::create_dir(self.root.join("candidate")).map_err(|e| sc::Error(e.to_string()))?;
        self.json("candidate/result.json",json!({"schema":"phasesmith.single-crystal-native-search.v1","experimental":false,"cell":self.cell,"candidates":s.candidates.iter().map(candidate).collect::<Vec<_>>(),"phasing_histories":s.histories,"termination":s.termination,"r1_target_met":s.target_met,"input_elements":search.elements,"grid_shape":s.grid_shape,"family_count":s.data.family_count,"d_min_angstrom":s.data.d_min,"standalone_structure_refinement":false}))?;
        self.json("search.json",json!({"termination":s.termination,"target_met":s.target_met,"amplitude_r1":s.candidates.first().map(|c|c.r1),"model_saved_before_reference":true,"timing_record":"timing-structure_search.json"}))?;
        let mut audit = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join("candidate/native-audit.txt"))
            .map_err(|e| sc::Error(e.to_string()))?;
        write!(audit,"Native debug diagnostics; version-bound, not a stable exchange schema.\n{search:#?}\n{:#?}\n",s.phasing_trials).map_err(|e|sc::Error(e.to_string()))?;
        self.archive(
            "candidate/density.npz",
            &[
                f64s("density", s.grid_shape.to_vec(), s.density.iter().copied()),
                i64s(
                    "hkl",
                    vec![s.data.hkl.len(), 3],
                    s.data.hkl.iter().flatten().map(|&h| i64::from(h)),
                ),
                f64s(
                    "amplitudes",
                    vec![s.data.amplitudes.len()],
                    s.data.amplitudes.iter().copied(),
                ),
            ],
        )?;
        if let Some(c) = s.candidates.first() {
            let mut text = format!(
                "data_phasesmith_candidate\n# Unvalidated structure-search candidate; no uniqueness claim.\n_cell_length_a {}\n_cell_length_b {}\n_cell_length_c {}\n_cell_angle_alpha {}\n_cell_angle_beta {}\n_cell_angle_gamma {}\n_space_group_name_H-M_alt 'P 1'\nloop_\n_space_group_symop_operation_xyz\n'x,y,z'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_U_iso_or_equiv\n",
                self.cell[0], self.cell[1], self.cell[2], self.cell[3], self.cell[4], self.cell[5]
            );
            for (i, (symbol, xyz)) in c.species.iter().zip(&c.fractional_xyz).enumerate() {
                let u = c.refinement.as_ref().map_or(c.u_iso, |r| r.full_u_equiv[i]);
                writeln!(
                    &mut text,
                    "{symbol}{} {symbol} {:.15} {:.15} {:.15} {:.15}",
                    i + 1,
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    u
                )
                .map_err(|e| sc::Error(e.to_string()))?;
            }
            if let Some(refined) = &c.refinement {
                text.push_str("loop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\n_atom_site_aniso_U_22\n_atom_site_aniso_U_33\n_atom_site_aniso_U_23\n_atom_site_aniso_U_13\n_atom_site_aniso_U_12\n");
                for (i, (symbol, u)) in c.species.iter().zip(&refined.full_u_cif).enumerate() {
                    writeln!(
                        &mut text,
                        "{symbol}{} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15}",
                        i + 1,
                        u[0],
                        u[1],
                        u[2],
                        u[3],
                        u[4],
                        u[5]
                    )
                    .map_err(|e| sc::Error(e.to_string()))?;
                }
            }
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.root.join("candidate/best.cif"))
                .map_err(|e| sc::Error(e.to_string()))?;
            f.write_all(text.as_bytes())
                .map_err(|e| sc::Error(e.to_string()))?;
        }
        Ok(())
    }
}
fn matrix(a: sc::M3, b: sc::M3) -> sc::M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn poni(path: &Path, pixel1: f64, pixel2: f64, axis: sc::V3) -> Result<Geometry> {
    let text = fs::read_to_string(path)?;
    let mut map = std::collections::BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
        if let Some((k, v)) = line.split_once(':') {
            map.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
    }
    let value =
        |name: &str| -> Result<f64> { Ok(map.get(name).ok_or("missing PONI parameter")?.parse()?) };
    let (r1, r2, r3) = (value("rot1")?, value("rot2")?, value("rot3")?);
    let (s1, c1) = r1.sin_cos();
    let (s2, c2) = r2.sin_cos();
    let (s3, c3) = r3.sin_cos();
    let r = matrix(
        matrix(
            [[c3, -s3, 0.], [s3, c3, 0.], [0., 0., 1.]],
            [[c2, 0., -s2], [0., 1., 0.], [s2, 0., c2]],
        ),
        [[1., 0., 0.], [0., c1, s1], [0., -s1, c1]],
    );
    let mv = |a: sc::V3| std::array::from_fn(|i| (0..3).map(|j| r[i][j] * a[j]).sum());
    Ok(Geometry {
        origin: mv([
            pixel1 / 2. - value("poni1")?,
            pixel2 / 2. - value("poni2")?,
            value("distance")?,
        ]),
        fast: mv([0., pixel2, 0.]),
        slow: mv([pixel1, 0., 0.]),
        beam: [0., 0., 1.],
        axis,
        wavelength: value("wavelength")? * 1e10,
    })
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "--help") {
        println!(
            "Rust-only MgSiO3: single_crystal_workflow --mgsio3 DATA_ROOT PROTOCOL.json NEW_EXTERNAL_OUTPUT\nPortable input: single_crystal_workflow INPUT.json NULL_ROTATIONS.json NEW_EXTERNAL_OUTPUT [DIAGNOSTIC_OBSERVATIONS.json]"
        );
        return Ok(());
    }
    let Request {
        mut packet,
        rotations,
        out,
        diagnostic,
        registered,
    } = request(&args)?;
    verify_registered(&registered)?;
    if out.exists() {
        return Err("output directory must be new".into());
    }
    validate_adapter(&packet, &out)?;
    packet["output"] = json!(out);
    let raw = packet["raw_path"].as_str().ok_or("raw_path missing")?;
    let dataset = packet["raw_array"]["dataset"]
        .as_str()
        .ok_or("dataset missing")?;
    let protocol = &packet["protocol"];
    let acquisition = &protocol["acquisition"];
    let reader = CachedReader::open(
        raw,
        dataset,
        Limits::default(),
        int(&protocol["frame_input_execution"], "decoded_cache_bytes")?,
    )?;
    let shape = reader.shape();
    let expected: Vec<usize> = serde_json::from_value(packet["raw_array"]["shape"].clone())?;
    if shape.as_slice() != expected {
        return Err("raw shape differs from registered source".into());
    }
    let detector = dataset.rsplit_once('/').ok_or("invalid dataset")?.0;
    let pixel1 = reader.scalar(&format!("{detector}/detector_information/pixel_size/ysize"))?;
    let pixel2 = reader.scalar(&format!("{detector}/detector_information/pixel_size/xsize"))?;
    let axis = serde_json::from_value(acquisition["axis"].clone())?;
    let geometry = poni(
        Path::new(packet["poni_path"].as_str().ok_or("PONI path missing")?),
        pixel1,
        pixel2,
        axis,
    )?;
    let cell: [f64; 6] = serde_json::from_value(packet["priors"]["cell_angstrom_degrees"].clone())?;
    let options = decode_options(&packet, rotations, geometry)?;
    fs::create_dir(&out)?;
    json_file(&out.join("input.json"), &packet)?;
    let mut save = Save {
        root: out.clone(),
        timings: vec![],
        shape: [shape[1], shape[2]],
        cell,
        priors: packet["priors"].clone(),
        angles: (options.acquisition.first..=options.acquisition.last)
            .map(|i| {
                let a = &options.acquisition;
                let low = a.omega_zero + i as f64 * a.omega_step;
                [low, low + a.exposure]
            })
            .collect(),
    };
    let started = Instant::now();
    if let Some(path) = diagnostic {
        run_diagnostic(&path, &options, &mut save)?;
        return Ok(());
    }
    let mut source = Source(reader);
    match sc::run(&mut source, &options, &mut save) {
        Ok(result) => {
            verify_registered(&registered)?;
            json_file(
                &out.join("timings.json"),
                &json!({"seconds":started.elapsed().as_secs_f64(),"coordinator_seconds":result.seconds,"stages":save.timings,"native_only":true}),
            )?;
            json_file(
                &out.join("worker-complete.json"),
                &json!({"backend":"Rust executable","reference_model_read":false,"source_hkls_read":false,"archived_arrays_read":false,"loaded_blocked_python_packages":[],"standalone_structure_refinement":false}),
            )?;
            println!(
                "{}",
                json!({"seconds":result.seconds,"observations":result.reduced.use_mask.iter().filter(|&&v|v).count(),"r1":result.solution.candidates.first().map(|c|c.r1),"termination":result.solution.termination})
            );
            Ok(())
        }
        Err(e) => {
            json_file(
                &out.join("failure.json"),
                &json!({"error":e.to_string(),"seconds":started.elapsed().as_secs_f64(),"completed_stages":save.timings}),
            )?;
            Err(e.into())
        }
    }
}

fn int(v: &Value, k: &str) -> Result<usize> {
    Ok(usize::try_from(
        v[k].as_u64()
            .ok_or_else(|| format!("missing integer {k}"))?,
    )?)
}
fn number(v: &Value, k: &str) -> Result<f64> {
    v[k].as_f64()
        .ok_or_else(|| format!("missing number {k}").into())
}

fn decode_options(
    packet: &Value,
    rotations: Vec<sc::M3>,
    geometry: Geometry,
) -> Result<sc::Options> {
    let protocol = &packet["protocol"];
    let acquisition = &protocol["acquisition"];
    let cell: [f64; 6] = serde_json::from_value(packet["priors"]["cell_angstrom_degrees"].clone())?;
    let options = sc::Options {
        acquisition: sc::Acquisition {
            first: int(acquisition, "first_frame")?,
            last: int(acquisition, "last_frame_inclusive")?,
            omega_zero: number(acquisition, "omega_at_original_frame_zero_degrees")?,
            omega_step: number(acquisition, "step_degrees")?,
            exposure: number(acquisition, "exposure_width_degrees")?,
            vertical_flip: acquisition["vertical_flip"]
                .as_bool()
                .ok_or("orientation missing")?,
            reserved_minimum: u32::try_from(int(acquisition, "mask_minimum_reserved_uint32")?)?,
        },
        geometry,
        priors: sc::Priors {
            cell: UnitCell {
                a_angstrom: cell[0],
                b_angstrom: cell[1],
                c_angstrom: cell[2],
                alpha_deg: cell[3],
                beta_deg: cell[4],
                gamma_deg: cell[5],
            },
            group: space_group_by_number(i32::try_from(
                packet["priors"]["space_group_number"]
                    .as_i64()
                    .ok_or("space group missing")?,
            )?)?
            .space_group,
            elements: serde_json::from_value(packet["priors"]["elements"].clone())?,
            provenance: packet["priors"]["provenance"]
                .as_str()
                .ok_or("prior provenance missing")?
                .into(),
        },
        null_rotations: rotations,
        shadows: sc::ShadowOptions {
            illumination_block: int(&protocol["shadow_mask"]["illumination"], "block")?,
            illumination: phasesmith::spots::frame_validity::SeatOptions {
                threshold: number(&protocol["shadow_mask"]["illumination"], "threshold")?,
                guard_pixels: int(&protocol["shadow_mask"]["illumination"], "guard_pixels")?,
                reference_first: None,
                reference_last: None,
            },
            aperture_block: int(&protocol["shadow_mask"]["geometric"], "pixel_block")?,
            aperture: phasesmith::spots::frame_validity::DacOptions {
                axis_at_zero: serde_json::from_value(
                    protocol["shadow_mask"]["geometric"]["axis_at_omega_zero"].clone(),
                )?,
                half_opening_degrees: number(
                    &protocol["shadow_mask"]["geometric"],
                    "half_opening_degrees",
                )?,
            },
        },
        mask_workers: int(acquisition, "workers")?,
        preparation_workers: int(
            &protocol["frame_input_execution"],
            "native_preparation_workers",
        )?,
        preparation_batch: int(&protocol["frame_input_execution"], "batch_frames")?,
        max_in_flight_frame_bytes: int(
            &protocol["frame_preparation_execution"],
            "max_in_flight_bytes",
        )?,
        cancellation: solution::Cancellation::default(),
        detection_workers: int(protocol, "detection_workers")?,
        profile_workers: int(&protocol["reduction"], "shape_fit_workers")?,
        learning_workers: int(&protocol["profile_execution"], "workers")?,
        frame_workers: int(&protocol["frame_preparation_execution"], "workers")?,
        geometry_cache_bytes: int(&protocol["profile_execution"], "geometry_cache_bytes")?,
        frame_payload_bytes: int(&protocol["frame_input_execution"], "payload_bytes")?,
        max_samples: 20_000_000,
        search_seconds: number(&protocol["search"], "native_seconds")?,
        dense_memory_bytes: int(&protocol["search"], "dense_memory_budget_bytes")?,
        search_trials: int(&protocol["search"], "trials")?,
        search_cycles: int(&protocol["search"], "cycles")?,
        seed: int(&protocol["search"], "seed")? as u64,
    };
    Ok(options)
}

fn validate_adapter(packet: &Value, out: &Path) -> Result<()> {
    if !out.is_absolute() {
        return Err("output must be an absolute external path".into());
    }
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("missing repository root")?
        .canonicalize()?;
    let parent = out
        .parent()
        .ok_or("output parent missing")?
        .canonicalize()?;
    if parent.starts_with(repository) {
        return Err("large output must stay outside the repository".into());
    }
    let frozen: Value =
        serde_json::from_str(include_str!("data/native-workflow-controls-v1.json"))?;
    for name in [
        "detection",
        "linking",
        "indexing",
        "selection",
        "geometry",
        "prediction",
        "integration",
        "profiles",
        "reduction",
        "measurement_validity",
        "search_configuration",
    ] {
        if packet["protocol"][name] != frozen[name] {
            return Err(format!("native orthorhombic-rotation-v1 does not silently override unsupported {name} settings").into());
        }
    }
    if packet["protocol"]["shadow_mask"]["mode"] != "combined"
        || packet["protocol"]["acquisition"]["gain"] != json!(1.0)
        || packet["protocol"]["acquisition"]["read_noise"] != json!(0.0)
    {
        return Err("native v1 requires combined masks and the declared unit-gain zero-read-noise example model".into());
    }
    Ok(())
}

struct Request {
    packet: Value,
    rotations: Vec<sc::M3>,
    out: PathBuf,
    diagnostic: Option<PathBuf>,
    registered: Vec<(PathBuf, String)>,
}
fn request(args: &[String]) -> Result<Request> {
    if args.get(1).is_some_and(|a| a == "--mgsio3") {
        if args.len() != 5 {
            return Err("usage: --mgsio3 DATA_ROOT PROTOCOL.json NEW_EXTERNAL_OUTPUT".into());
        }
        let source: Value = serde_json::from_str(include_str!("data/mgsio3-source.json"))?;
        let protocol: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
        let root = Path::new(&args[2]).canonicalize()?;
        let rotations = serde_json::from_str(include_str!("data/grain-null-rotations-v1.json"))?;
        let registered = source["files"]
            .as_array()
            .ok_or("missing source fingerprints")?
            .iter()
            .map(|v| {
                Ok((
                    root.join(v["path"].as_str().ok_or("source filename missing")?),
                    v["sha256"].as_str().ok_or("source digest missing")?.into(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let packet = json!({"raw_path":root.join("eiger_0000.h5"),"poni_path":root.join("CeO2_20240620_1849_dioptas.poni"),"raw_array":source["raw_array"],"priors":source["priors"],"rotations":[],"protocol":protocol,"output":args[4],"through":"search"});
        return Ok(Request {
            packet,
            rotations,
            out: PathBuf::from(&args[4]),
            diagnostic: None,
            registered,
        });
    }
    if !(4..=5).contains(&args.len()) {
        return Err("use --help for the native input and output arguments".into());
    }
    Ok(Request {
        packet: serde_json::from_slice(&fs::read(&args[1])?)?,
        rotations: serde_json::from_slice(&fs::read(&args[2])?)?,
        out: PathBuf::from(&args[3]),
        diagnostic: args.get(4).map(PathBuf::from),
        registered: vec![],
    })
}
fn verify_registered(files: &[(PathBuf, String)]) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    for (path, expected) in files {
        let mut f = fs::File::open(path)?;
        let mut hash = Sha256::new();
        let mut bytes = vec![0u8; 65_536];
        loop {
            let n = f.read(&mut bytes)?;
            if n == 0 {
                break;
            }
            hash.update(&bytes[..n]);
        }
        if format!("{:x}", hash.finalize()) != *expected {
            return Err(format!("registered source digest differs: {}", path.display()).into());
        }
    }
    Ok(())
}

fn run_diagnostic(path: &Path, options: &sc::Options, save: &mut Save) -> Result<()> {
    let started = Instant::now();
    let rows: Vec<([i32; 3], [f64; 2])> = serde_json::from_slice(&fs::read(path)?)?;
    let observations: Vec<_> = rows
        .into_iter()
        .map(|(h, v)| solution::Observation {
            hkl: h,
            intensity: v[0],
            sigma: v[1],
        })
        .collect();
    let (result, audit) = sc::search_observations(&observations, options)?;
    sc::Observer::solution(save, &result, &audit)?;
    println!(
        "{}",
        json!({"diagnostic_search_only":true,"r1":result.candidates.first().map(|c|c.r1),"seconds":started.elapsed().as_secs_f64()})
    );
    Ok(())
}
