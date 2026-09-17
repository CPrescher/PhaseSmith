//! Native joint TOF Pawley: density observations, bank-local areas and shared cells.
#![allow(clippy::many_single_char_names)] // Conventional matrix and geometry indices.
use crate::pawley::err;
use crate::pawley_operator::{PawleyColumn, PawleyJacobian};
use crate::pawley_solver::{PawleyProblem, refine_problem_with_runtime};
use crate::{
    Constraint, ConstraintTransform, ParameterBounds, ParameterSet, ParameterSpec,
    PawleyCheckpoint, PawleyCoincidentGroup, PawleyError, PawleyEvaluation, PawleyOptions,
    PawleyResult, PawleySolver, RefinementLimits, RefinementRuntime, ResidualOptions,
    TofChebyshevBackground, TofSharedLatticePhase, pawley_key, tof_lattice_geometry,
};
use phasesmith_core::{GridView, TOF_GLOBAL_PARAMETER_NAMES, TofInstrument, accumulate_tof_batch};
use phasesmith_model::TofPatternRecord;
use std::collections::{BTreeMap, BTreeSet};

/// A bank-local reflection family list, with optional shared-cell geometry by phase ID.
#[derive(Clone, Debug, PartialEq)]
pub struct TofPawleyPhase {
    /// Stable identity, without a colon (reserved for bank/phase ownership).
    pub id: String,
    /// Stable unique family identities.
    pub reflection_ids: Vec<String>,
    /// HKLs, required when this phase has a shared cell.
    pub hkl: Vec<[i32; 3]>,
    /// Fixed d-spacings when no shared cell is supplied, in angstroms.
    pub d_spacing_angstrom: Vec<f64>,
    /// Bank-local initial integrated areas in observed-y times microseconds.
    pub intensities: Vec<f64>,
}
/// One independent TOF observation bank. Axes are bin centers, not bin widths.
#[derive(Clone, Debug, PartialEq)]
pub struct TofPawleyBank {
    /// Stable bank identity without a colon.
    pub id: String,
    /// Density observations on a strictly increasing microsecond grid.
    pub pattern: TofPatternRecord,
    /// Bank-local calibration and asymmetric profile coefficients.
    pub instrument: TofInstrument,
    /// Ordered family lists.
    pub phases: Vec<TofPawleyPhase>,
    /// Optional linear Chebyshev background on an explicit microsecond domain.
    pub background: Option<TofChebyshevBackground>,
    /// Explicit observation/incident-spectrum normalization provenance.
    /// Pawley absorbs fixed amplitude corrections into each bank-local area.
    pub normalization: String,
}
/// One genuinely joint objective; a single bank is its exact reduction.
#[derive(Clone, Debug, PartialEq)]
pub struct TofPawleyInput {
    /// Independent bank observations in deterministic output order.
    pub banks: Vec<TofPawleyBank>,
    /// Shared bounded cells, matched to bank phases by stable identity.
    pub shared_lattice: Vec<TofSharedLatticePhase>,
    /// Explicit signed-area policy; nonnegative by default in constructors.
    pub signed_intensities: bool,
    /// Truncated asymmetric exponential-tail control, fixed during fitting.
    pub tail_log: f64,
    /// Complete physical table. Areas/background are initially free.
    pub parameters: ParameterSet,
    /// Exact constraints; bank-local areas are never tied automatically.
    pub constraints: Vec<Constraint>,
}
/// Accepted joint state, including every bank and immutable normalization metadata.
pub type TofPawleyCheckpoint = PawleyCheckpoint<TofPawleyInput>;
/// Native fit with concatenated sample/family outputs in bank, phase, family order.
pub type TofPawleyResult = PawleyResult<TofPawleyInput>;

impl TofPawleyInput {
    /// Construct the default area/background refinement request.
    /// # Errors
    /// Rejects malformed scientific records, bounds or identities.
    pub fn new(
        banks: Vec<TofPawleyBank>,
        shared_lattice: Vec<TofSharedLatticePhase>,
        signed_intensities: bool,
        tail_log: f64,
    ) -> Result<Self, PawleyError> {
        let parameters = tof_pawley_parameters(&banks, &shared_lattice, signed_intensities)?;
        let result = Self {
            banks,
            shared_lattice,
            signed_intensities,
            tail_log,
            parameters,
            constraints: Vec::new(),
        };
        result.validate()?;
        Ok(result)
    }
    /// Cumulative sample boundaries for splitting joint result arrays.
    #[must_use]
    pub fn sample_offsets(&self) -> Vec<usize> {
        let mut offsets = vec![0];
        for b in &self.banks {
            offsets.push(offsets.last().copied().unwrap_or(0) + b.pattern.sample_count());
        }
        offsets
    }
    /// Validate topology, complete parameter contract and calibration anchoring.
    /// # Errors
    /// Rejects malformed data and an unanchored shared-length/DIFC selection.
    #[allow(clippy::too_many_lines)] // Complete bank, topology and gauge validation.
    pub fn validate(&self) -> Result<(), PawleyError> {
        if self.banks.is_empty() || !self.tail_log.is_finite() || self.tail_log <= 0.0 {
            return Err(err(
                "TOF Pawley requires banks and positive finite tail_log",
            ));
        }
        let mut cells = BTreeSet::new();
        for cell in &self.shared_lattice {
            if !cells.insert(cell.phase_id().as_str()) {
                return Err(err("duplicate shared Pawley cell"));
            }
            cell.validate_cell(cell.initial_cell()).map_err(err)?;
            if !self
                .banks
                .iter()
                .any(|b| b.phases.iter().any(|p| p.id == cell.phase_id().as_str()))
            {
                return Err(err("unused shared Pawley cell"));
            }
        }
        let mut banks = BTreeSet::new();
        for bank in &self.banks {
            identity(&bank.id)?;
            if !banks.insert(&bank.id)
                || bank.phases.is_empty()
                || bank.normalization.trim().is_empty()
            {
                return Err(err(
                    "duplicate/empty TOF bank or missing density normalization provenance",
                ));
            }
            bank.pattern.validate().map_err(err)?;
            bank.instrument.validate().map_err(err)?;
            if bank.pattern.sample_count() < 2 {
                return Err(err("TOF Pawley requires at least two samples per bank"));
            }
            if let Some(bg) = &bank.background {
                bg.basis(&bank.pattern.tof_us).map_err(err)?;
            }
            let mut phases = BTreeSet::new();
            for phase in &bank.phases {
                identity(&phase.id)?;
                if !phases.insert(&phase.id) {
                    return Err(err("duplicate TOF Pawley phase"));
                }
                let n = phase.reflection_ids.len();
                if n == 0
                    || phase.intensities.len() != n
                    || phase.d_spacing_angstrom.len() != n
                    || (!phase.hkl.is_empty() && phase.hkl.len() != n)
                    || (cells.contains(phase.id.as_str()) && phase.hkl.len() != n)
                {
                    return Err(err("TOF Pawley family shape mismatch"));
                }
                if phase
                    .d_spacing_angstrom
                    .iter()
                    .any(|v| !v.is_finite() || *v <= 0.0)
                {
                    return Err(err("TOF d-spacings must be positive finite"));
                }
                let mut refs = BTreeSet::new();
                for id in &phase.reflection_ids {
                    if !refs.insert(id) {
                        return Err(err("duplicate TOF family identity"));
                    }
                    pawley_key("intensity", &owner(&bank.id, &phase.id), id)?;
                }
            }
        }
        let expected =
            tof_pawley_parameters(&self.banks, &self.shared_lattice, self.signed_intensities)?;
        if expected.specs().len() != self.parameters.specs().len() {
            return Err(err("TOF Pawley parameter table shape mismatch"));
        }
        for (a, b) in expected.specs().iter().zip(self.parameters.specs()) {
            if a.key() != b.key()
                || a.unit() != b.unit()
                || b.bounds().lower() < a.bounds().lower()
                || b.bounds().upper() > a.bounds().upper()
            {
                return Err(err("TOF Pawley parameter identity/unit/bound mismatch"));
            }
        }
        let t = ConstraintTransform::new(self.parameters.clone(), self.constraints.clone())
            .map_err(err)?;
        t.unpack(&t.pack().map_err(err)?, false).map_err(err)?;
        let chain = t.derivative_matrix().map_err(err)?;
        let varying = |i: usize| chain.row(i).is_some_and(|r| r.iter().any(|v| *v != 0.0));
        for cell in &self.shared_lattice {
            let length_varies = self.parameters.specs().iter().enumerate().any(|(i, s)| {
                s.key().module() == "pawley_lattice"
                    && s.key().owner_id() == cell.phase_id().as_str()
                    && !s.key().name().ends_with("deg")
                    && varying(i)
            });
            let observing_banks = self
                .banks
                .iter()
                .filter(|b| b.phases.iter().any(|p| p.id == cell.phase_id().as_str()));
            if length_varies
                && observing_banks.clone().all(|b| {
                    pawley_key("profile", &b.id, "difc")
                        .ok()
                        .and_then(|key| self.parameters.index_of(&key))
                        .is_some_and(varying)
                })
            {
                return Err(err(
                    "shared cell lengths and their observing banks' DIFC values are unanchored; fix one observing bank DIFC",
                ));
            }
        }
        Ok(())
    }
}
fn identity(id: &str) -> Result<(), PawleyError> {
    if id.contains(':') || id.trim() != id || id.is_empty() {
        return Err(err(
            "TOF bank/phase identity must be nonempty and contain no colon",
        ));
    }
    pawley_key("profile", id, "identity")?;
    Ok(())
}
fn owner(bank: &str, phase: &str) -> String {
    format!("{bank}:{phase}")
}
/// Build stable bank-local area/profile/background and shared lattice parameters.
/// # Errors
/// Rejects invalid initial values or parameter identities.
#[allow(clippy::too_many_lines)] // Stable public table order and physical units.
pub fn tof_pawley_parameters(
    banks: &[TofPawleyBank],
    cells: &[TofSharedLatticePhase],
    signed: bool,
) -> Result<ParameterSet, PawleyError> {
    let mut specs = Vec::new();
    let unbounded = ParameterBounds::new(f64::NEG_INFINITY, f64::INFINITY).map_err(err)?;
    let mut add =
        |family, owner: &str, name: &str, value, unit, bounds, refine| -> Result<(), PawleyError> {
            specs.push(
                ParameterSpec::new(
                    pawley_key(family, owner, name)?,
                    value,
                    unit,
                    bounds,
                    f64::abs(value).max(1e-3),
                    refine,
                )
                .map_err(err)?,
            );
            Ok(())
        };
    for bank in banks {
        for phase in &bank.phases {
            if phase.reflection_ids.len() != phase.intensities.len() {
                return Err(err("TOF reflection/area shape mismatch"));
            }
            for (id, &area) in phase.reflection_ids.iter().zip(&phase.intensities) {
                add(
                    "intensity",
                    &owner(&bank.id, &phase.id),
                    id,
                    area,
                    "y*microsecond",
                    ParameterBounds::new(
                        if signed { f64::NEG_INFINITY } else { 0.0 },
                        f64::INFINITY,
                    )
                    .map_err(err)?,
                    true,
                )?;
            }
        }
        let units = [
            "microsecond",
            "microsecond/angstrom",
            "microsecond/angstrom^2",
            "microsecond*angstrom",
            "angstrom/microsecond",
            "1/microsecond",
            "angstrom^4/microsecond",
            "angstrom^2/microsecond",
            "microsecond^2",
            "microsecond^2/angstrom^2",
            "microsecond^2/angstrom^4",
            "microsecond^2/angstrom",
            "microsecond/angstrom",
            "microsecond/angstrom^2",
            "microsecond",
        ];
        for ((name, value), unit) in TOF_GLOBAL_PARAMETER_NAMES
            .iter()
            .zip(bank.instrument.values())
            .zip(units)
        {
            add(
                "profile",
                &bank.id,
                name,
                value,
                unit,
                if *name == "difc" {
                    ParameterBounds::new(f64::MIN_POSITIVE, f64::INFINITY).map_err(err)?
                } else {
                    unbounded
                },
                false,
            )?;
        }
        if let Some(bg) = &bank.background {
            for (j, &v) in bg.coefficients().iter().enumerate() {
                add(
                    "background",
                    &bank.id,
                    &format!("c{j}"),
                    v,
                    "y",
                    unbounded,
                    true,
                )?;
            }
        }
    }
    for cell in cells {
        let par = cell.parameterization();
        let values = par.values_from_cell(cell.initial_cell()).map_err(err)?;
        for (j, (name, value)) in par.parameter_names().iter().zip(values).enumerate() {
            add(
                "lattice",
                cell.phase_id().as_str(),
                name,
                value,
                if name.ends_with("deg") {
                    "degree"
                } else {
                    "angstrom"
                },
                ParameterBounds::new(cell.bounds().lower()[j], cell.bounds().upper()[j])
                    .map_err(err)?,
                false,
            )?;
        }
    }
    ParameterSet::new(specs).map_err(err)
}
impl PawleyProblem for TofPawleyInput {
    fn validate(&self) -> Result<(), PawleyError> {
        self.validate()
    }
    fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }
    fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }
    fn sigma(&self, mut sample: usize) -> f64 {
        for b in &self.banks {
            if sample < b.pattern.sample_count() {
                return b.pattern.uncertainty.as_ref().map_or(1.0, |s| s[sample]);
            }
            sample -= b.pattern.sample_count();
        }
        unreachable!("validated sample index")
    }
    fn known_uncertainties(&self) -> bool {
        self.banks.iter().all(|b| b.pattern.uncertainty.is_some())
    }
    fn evaluate(&self, o: &PawleyOptions, free: &[f64]) -> Result<PawleyEvaluation, PawleyError> {
        evaluate_tof_pawley(self, free, o)
    }
}
/// Evaluate the joint density objective and native analytical products.
/// # Errors
/// Rejects invalid profiles/calibration, out-of-bounds cells or allocation limits.
#[allow(clippy::too_many_lines)]
pub fn evaluate_tof_pawley(
    input: &TofPawleyInput,
    free: &[f64],
    options: &PawleyOptions,
) -> Result<PawleyEvaluation, PawleyError> {
    input.validate()?;
    options.validate()?;
    let t = ConstraintTransform::new(input.parameters.clone(), input.constraints.clone())
        .map_err(err)?;
    let values = t.unpack(free, false).map_err(err)?;
    if options.use_uncertainty
        && input.banks.iter().any(|b| b.pattern.uncertainty.is_some())
        && input.banks.iter().any(|b| b.pattern.uncertainty.is_none())
    {
        return Err(err(
            "TOF joint uncertainty weighting requires sigmas in every bank; supply them or select unit weights",
        ));
    }
    let offsets = input.sample_offsets();
    let n = *offsets.last().ok_or_else(|| err("missing bank"))?;
    let p = input.parameters.specs().len();
    let k = free.len();
    let peaks = input
        .banks
        .iter()
        .flat_map(|b| &b.phases)
        .map(|p| p.intensities.len())
        .sum::<usize>();
    let estimate = n
        .checked_mul(
            p.checked_mul(2)
                .and_then(|v| v.checked_add(k.checked_mul(2)?))
                .and_then(|v| v.checked_add(peaks.checked_mul(2)?))
                .and_then(|v| v.checked_add(40))
                .ok_or_else(|| err("allocation overflow"))?,
        )
        .and_then(|v| v.checked_add(p.checked_mul(k)?.checked_mul(8)?))
        .and_then(|v| v.checked_add(k.checked_mul(k)?.checked_mul(16)?))
        .ok_or_else(|| err("allocation overflow"))?;
    if estimate > options.max_elements {
        return Err(err("TOF Pawley workspace element limit exceeded"));
    }
    let mut physical = vec![PawleyColumn::empty(); p];
    let mut calculated = Vec::with_capacity(n);
    let mut background = Vec::with_capacity(n);
    let mut observed = Vec::with_capacity(n);
    let mut sigma = Vec::with_capacity(n);
    let mut included = Vec::with_capacity(n);
    let mut intensities = Vec::new();
    let mut positions = Vec::new();
    let mut unobserved = Vec::new();
    let mut coincident = Vec::new();
    let get = |family, owner: &str, name: &str| -> Result<f64, PawleyError> {
        values
            .get(&pawley_key(family, owner, name)?)
            .copied()
            .ok_or_else(|| err("missing TOF parameter"))
    };
    let index = |family, owner: &str, name: &str| -> Result<usize, PawleyError> {
        input
            .parameters
            .index_of(&pawley_key(family, owner, name)?)
            .ok_or_else(|| err("missing TOF parameter"))
    };
    for (bank_index, bank) in input.banks.iter().enumerate() {
        let start = offsets[bank_index];
        let count = bank.pattern.sample_count();
        let mut coefficients = [0.0; 15];
        for (i, name) in TOF_GLOBAL_PARAMETER_NAMES.iter().enumerate() {
            coefficients[i] = get("profile", &bank.id, name)?;
        }
        let instrument = TofInstrument::from_values(coefficients).map_err(err)?;
        let mut d = Vec::new();
        let mut areas = Vec::new();
        let mut geometries = Vec::new();
        for phase in &bank.phases {
            let geometry = input
                .shared_lattice
                .iter()
                .find(|c| c.phase_id().as_str() == phase.id)
                .map(|c| {
                    let par = c.parameterization();
                    let v = par
                        .parameter_names()
                        .iter()
                        .map(|name| get("lattice", &phase.id, name))
                        .collect::<Result<Vec<_>, _>>()?;
                    let cell = par.to_cell(&v).map_err(err)?;
                    c.validate_cell(cell).map_err(err)?;
                    tof_lattice_geometry(par, cell, &phase.hkl, instrument).map_err(err)
                })
                .transpose()?;
            d.extend_from_slice(
                geometry
                    .as_ref()
                    .map_or(&phase.d_spacing_angstrom, |g| &g.d_spacing_angstrom),
            );
            for id in &phase.reflection_ids {
                areas.push(get("intensity", &owner(&bank.id, &phase.id), id)?);
            }
            geometries.push(geometry);
        }
        // Preserve a one-to-one calibrated d-to-time map over the retained domain.
        let lo = d.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let critical = (-instrument.difb_us_angstrom / instrument.difa_us_per_angstrom2).cbrt();
        for spacing in [lo, hi, critical] {
            if spacing >= lo
                && spacing <= hi
                && instrument.difc_us_per_angstrom
                    + 2.0 * instrument.difa_us_per_angstrom2 * spacing
                    - instrument.difb_us_angstrom / spacing.powi(2)
                    <= 0.0
            {
                return Err(err(
                    "TOF calibration must be increasing over the retained d-spacing domain",
                ));
            }
        }
        let accumulation = accumulate_tof_batch(
            GridView::new(&bank.pattern.tof_us).map_err(err)?,
            &d,
            &areas,
            instrument,
            options.support_fwhm,
            input.tail_log,
        )
        .map_err(err)?;
        let local = &accumulation.derivatives.local;
        let global = accumulation
            .derivatives
            .global
            .as_ref()
            .ok_or_else(|| err("missing TOF derivatives"))?;
        let mut r = 0;
        let mut groups = BTreeMap::<u64, Vec<((String, String), f64)>>::new();
        for (phase, geometry) in bank.phases.iter().zip(&geometries) {
            let owner = owner(&bank.id, &phase.id);
            for (j, id) in phase.reflection_ids.iter().enumerate() {
                let col = index("intensity", &owner, id)?;
                physical[col] = PawleyColumn {
                    start: start + local.starts[r],
                    values: (local.offsets[r]..local.offsets[r + 1])
                        .map(|a| local.values[a * local.parameter_count])
                        .collect(),
                };
                let mut seen = false;
                for a in local.offsets[r]..local.offsets[r + 1] {
                    let sample = local.starts[r] + a - local.offsets[r];
                    seen |= bank.pattern.mask.as_ref().is_none_or(|m| m[sample])
                        && local.values[a * local.parameter_count] != 0.0;
                    if let Some(g) = geometry {
                        for (q, name) in g.parameter_names.iter().enumerate() {
                            physical[index("lattice", &phase.id, name)?].add_global(
                                start + sample,
                                local.values[a * local.parameter_count + 1]
                                    * g.d_d_spacing_d_parameters[j * g.parameter_names.len() + q],
                                n,
                            );
                        }
                    }
                }
                let key = (owner.clone(), id.clone());
                if seen {
                    groups
                        .entry(d[r].to_bits())
                        .or_default()
                        .push((key, areas[r]));
                } else {
                    unobserved.push(key);
                }
                let derived =
                    phasesmith_core::TofProfileParameters::from_instrument(d[r], instrument)
                        .map_err(err)?;
                positions.push(derived.position_us);
                r += 1;
            }
        }
        for members in groups.into_values().filter(|g| g.len() > 1) {
            coincident.push(PawleyCoincidentGroup {
                total_intensity: members.iter().map(|(_, a)| a).sum(),
                members: members.into_iter().map(|(k, _)| k).collect(),
            });
        }
        for (j, name) in TOF_GLOBAL_PARAMETER_NAMES.iter().enumerate() {
            physical[index("profile", &bank.id, name)?] = PawleyColumn {
                start,
                values: global.values[j * count..(j + 1) * count].to_vec(),
            };
        }
        let mut bg = bank.pattern.background_y.clone();
        if let Some(model) = &bank.background {
            let basis = model.basis(&bank.pattern.tof_us).map_err(err)?;
            for j in 0..model.coefficients().len() {
                let name = format!("c{j}");
                let value = get("background", &bank.id, &name)?;
                let column: Vec<_> = (0..count)
                    .map(|i| basis.values[i * basis.columns + j])
                    .collect();
                for (i, &v) in column.iter().enumerate() {
                    bg[i] += value * v;
                }
                physical[index("background", &bank.id, &name)?] = PawleyColumn {
                    start,
                    values: column,
                };
            }
        }
        calculated.extend(accumulation.y.iter().zip(&bg).map(|(a, b)| a + b));
        background.extend(bg);
        intensities.extend(areas);
        observed.extend_from_slice(
            bank.pattern
                .observed_y
                .as_ref()
                .ok_or_else(|| err("TOF Pawley requires observations"))?,
        );
        sigma.extend((0..count).map(|i| bank.pattern.uncertainty.as_ref().map_or(1.0, |s| s[i])));
        included.extend((0..count).map(|i| bank.pattern.mask.as_ref().is_none_or(|m| m[i])));
    }
    let chain = t.derivative_matrix().map_err(err)?;
    let operator = PawleyJacobian::new(n, k, physical, &chain.values)?;
    let weights: Vec<_> = included
        .iter()
        .map(|yes| if *yes { 1.0 } else { 0.0 })
        .collect();
    let inactive_columns = operator
        .column_norms(&weights)?
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v == 0.0).then_some(i))
        .collect::<Vec<_>>();
    let residuals = crate::residuals::evaluate_residual_arrays(
        n,
        Some(&observed),
        Some(&sigma),
        Some(&included),
        &calculated,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: k - inactive_columns.len(),
        },
    )
    .map_err(err)?;
    let jacobian = (options.solver == PawleySolver::Dense)
        .then(|| operator.materialize(options.max_elements))
        .transpose()?;
    Ok(PawleyEvaluation {
        calculated_y: calculated,
        background_y: background,
        intensities,
        positions,
        jacobian,
        jacobian_operator: operator,
        residuals,
        inactive_columns,
        unobserved_reflections: unobserved,
        coincident_groups: coincident,
    })
}
/// Refine all banks in one bounded least-squares objective.
/// # Errors
/// Rejects invalid inputs and numerical failures outside recoverable trial rejection.
pub fn refine_tof_pawley(
    input: &TofPawleyInput,
    options: &PawleyOptions,
) -> Result<TofPawleyResult, PawleyError> {
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).map_err(err)?;
    refine_tof_pawley_with_runtime(input, options, None, &mut runtime)
}
/// Refine with atomic joint acceptance, cancellation and complete-state continuation.
/// # Errors
/// Rejects stale/corrupt checkpoints and invalid scientific state.
pub fn refine_tof_pawley_with_runtime(
    input: &TofPawleyInput,
    options: &PawleyOptions,
    restart: Option<&TofPawleyCheckpoint>,
    runtime: &mut RefinementRuntime<TofPawleyCheckpoint>,
) -> Result<TofPawleyResult, PawleyError> {
    refine_problem_with_runtime(input, options, restart, runtime)
}

/// A joint TOF Pawley analysis whose bank IDs own shared project histograms.
#[derive(Clone, Debug, PartialEq)]
pub struct TofPawleyAnalysis {
    /// Stable identity of this joint analysis.
    pub analysis_id: phasesmith_model::RecordId,
    /// Every participating bank and shared cell.
    pub input: TofPawleyInput,
    /// Shared numerical controls.
    pub options: PawleyOptions,
    /// Optional last jointly accepted state.
    pub checkpoint: Option<TofPawleyCheckpoint>,
}
/// Lossless shared project ownership for one or more joint TOF Pawley analyses.
#[derive(Clone, Debug, PartialEq)]
pub struct TofPawleyProjectState {
    /// Shared histogram arrays, experiments and optional structural phases.
    pub project: phasesmith_model::ProjectRecord,
    /// Ordered joint analyses.
    pub analyses: Vec<TofPawleyAnalysis>,
}
impl TofPawleyProjectState {
    /// Validate ownership, scientific identity and complete checkpoint state.
    /// # Errors
    /// Rejects stale checkpoints and mismatched histogram/cell/phase identities.
    pub fn validate(&self) -> Result<(), PawleyError> {
        self.project.validate().map_err(err)?;
        let mut ids = BTreeSet::new();
        for a in &self.analyses {
            if !ids.insert(&a.analysis_id) {
                return Err(err("duplicate TOF Pawley analysis"));
            }
            a.input.validate()?;
            a.options.validate()?;
            if let Some(cp) = &a.checkpoint {
                if cp.input != a.input
                    || cp.options != a.options
                    || cp.support_local
                    || !cp.damping.is_finite()
                    || cp.damping <= 0.0
                    || cp.chi_square_history.is_empty()
                    || cp
                        .chi_square_history
                        .iter()
                        .any(|v| !v.is_finite() || *v < 0.0)
                    || cp.chi_square_history.windows(2).any(|v| v[1] >= v[0])
                {
                    return Err(err("stale TOF Pawley checkpoint"));
                }
                ConstraintTransform::new(a.input.parameters.clone(), a.input.constraints.clone())
                    .map_err(err)?
                    .unpack(&cp.free, false)
                    .map_err(err)?;
            }
            for bank in &a.input.banks {
                let h = self
                    .project
                    .tof_histograms
                    .iter()
                    .find(|h| h.histogram_id.as_str() == bank.id)
                    .ok_or_else(|| err("unknown TOF Pawley histogram owner"))?;
                if h.pattern != bank.pattern || h.experiment.instrument != bank.instrument {
                    return Err(err(
                        "TOF Pawley analysis differs from shared histogram state",
                    ));
                }
                if !h.phase_ids.is_empty()
                    && h.phase_ids
                        .iter()
                        .map(phasesmith_model::RecordId::as_str)
                        .ne(bank.phases.iter().map(|p| p.id.as_str()))
                {
                    return Err(err("TOF Pawley shared phase order mismatch"));
                }
            }
            for cell in &a.input.shared_lattice {
                if let Some(phase) = self
                    .project
                    .phases
                    .iter()
                    .find(|p| p.phase_id == *cell.phase_id())
                {
                    if phase.definition.cell != cell.initial_cell()
                        || &phase.definition.space_group != cell.parameterization().space_group()
                    {
                        return Err(err("TOF Pawley shared cell/symmetry mismatch"));
                    }
                }
            }
        }
        Ok(())
    }
}
