//! Python/native exchange: load a mixed bundle, resume Pawley, preserve other methods.
use phasesmith_persistence::{
    ProjectReadLimits, ProjectSaveOptions, load_project_bundle, save_project_bundle,
};
use phasesmith_workflows::{RefinementRuntime, refine_pawley_with_runtime};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: pawley_bundle_exchange INPUT_BUNDLE OUTPUT_BUNDLE".into());
    }
    let mut bundle = load_project_bundle(&args[1], ProjectReadLimits::default())?;
    for analysis in &mut bundle.pawley_analyses {
        let mut runtime =
            RefinementRuntime::new(phasesmith_workflows::RefinementLimits::default(), None)?;
        let result = refine_pawley_with_runtime(
            &analysis.input,
            &analysis.options,
            analysis.checkpoint.as_ref(),
            &mut runtime,
        )?;
        analysis.checkpoint = Some(result.checkpoint);
    }
    for analysis in &mut bundle.tof_pawley_analyses {
        let mut runtime =
            RefinementRuntime::new(phasesmith_workflows::RefinementLimits::default(), None)?;
        let result = phasesmith_workflows::refine_tof_pawley_with_runtime(
            &analysis.input,
            &analysis.options,
            analysis.checkpoint.as_ref(),
            &mut runtime,
        )?;
        analysis.checkpoint = Some(result.checkpoint);
    }
    save_project_bundle(&args[2], &bundle, ProjectSaveOptions::default())?;
    Ok(())
}
