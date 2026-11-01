//! Reusable P1 phasing from a supplied rotational intensity hypothesis, without an atomic model.
use std::time::{Duration,Instant};
use phasesmith_crystallography::{SpaceGroup,UnitCell};
use crate::{Observation,SolveOptions,SolutionError,DualSpaceOptions,PeakSelection,PhasingExecution,prepare_data,normalize_wilson,dual_space_with_execution,error};
/// A shared P1 map and actual starts, independent of exact space-group translations.
#[derive(Clone,Debug)]
pub struct PhaseOnlyResult {
    /// P1 density in an arbitrary origin and hand.
    pub density: Vec<f64>,
    /// Actual FFT dimensions.
    pub shape:[usize;3],
    /// Score histories for every attempted start.
    pub histories:Vec<Vec<f64>>,
    /// Selected start, by common mixed-amplitude misfit; no atomic-reference ranking.
    pub selected_start:usize,
    /// Actual phase-only seconds, including unsuccessful starts.
    pub seconds:f64,
}
/// Create one reusable P1 phase map before testing exact space-group translations.
/// The rotational group must have zero translations; no systematic absence is imposed.
/// This reuses the existing Wilson, Patterson, atomicity and Fourier kernels unchanged.
/// # Errors
/// Invalid cell/data/control, nonzero symmetry translations, oversized grid or no completed map.
#[allow(clippy::cast_possible_truncation,clippy::cast_sign_loss,clippy::cast_precision_loss)]
pub fn phase_only(observations:&[Observation],cell:UnitCell,rotational_group:&SpaceGroup,elements:&[String],options:&SolveOptions)->Result<PhaseOnlyResult,SolutionError>{
    crate::solver::validate_options(options)?;
    if elements.is_empty() || elements.len()>8 || rotational_group.operations().iter().any(|op|op.translation().iter().any(|t|t.numerator()!=0)) {
        return Err(error("phase-only input needs element types and a translation-free intensity group"));
    }
    let started=Instant::now();let deadline=started+Duration::from_secs_f64(options.max_seconds);
    let data=prepare_data(observations,cell,rotational_group)?;
    let mut shape=[0;3];
    for (a,length) in [cell.a_angstrom,cell.b_angstrom,cell.c_angstrom].iter().enumerate(){
        let extent=length/data.d_min;
        if !extent.is_finite() || extent>=64. {return Err(error("phase-only resolution sphere exceeds 128-grid support"));}
        shape[a]=(2*extent.ceil() as usize+2).max(8).next_power_of_two();
    }
    let support=crate::solver::reciprocal_support(shape,cell,rotational_group,&data)?;
    let norm=normalize_wilson(cell,rotational_group,&data.hkl,&data.intensity,elements)?;
    let mixed:Vec<_>=norm.e_values.iter().zip(&data.amplitudes).map(|(&e,&f)|e.sqrt()*f.sqrt()).collect();
    let mut histories=Vec::new();let mut best=f64::INFINITY;let mut density=None;let mut selected_start=0;
    for start in 0..options.trials {
        if crate::stopped(Some(deadline),options.cancellation.as_ref()){break;}
        let seed=options.seed.wrapping_add(start as u64);
        let result=dual_space_with_execution(shape,cell,&data.hkl,&mixed,&crate::solver::initial_phases(&data.hkl,seed),&support,
            &DualSpaceOptions{cycles:options.cycles,peak_selection:PeakSelection::Adaptive{sigma:2.,cap:128},mask_sigma:0.35*data.d_min,
            hybrid_m:3.,omit_fraction:0.1,seed,patterson_rank:if start%4==3 {None}else{Some(start)},cancellation:options.cancellation.clone()},Some(deadline),None,
            PhasingExecution{reusable_fourier_workspaces:true,filtered_peak_search:true,streamed_peak_search:true,batched_fft:true,real_fft:true,fused_phase_workspace:true,..PhasingExecution::default()})?;
        let score=result.trial.history.get(result.trial.best_cycle).copied().unwrap_or(f64::INFINITY);
        if score<best {best=score;selected_start=start;density=Some(result.trial.density);}
        histories.push(result.trial.history);
    }
    Ok(PhaseOnlyResult{density:density.ok_or_else(||error("no phase-only start completed within budget"))?,shape,histories,selected_start,seconds:started.elapsed().as_secs_f64()})
}
