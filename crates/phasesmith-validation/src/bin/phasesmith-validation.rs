//! Command-line entry point for Python-free checksum-pinned validation.

use std::error::Error;
use std::path::Path;

use phasesmith_validation::{
    run_pbso4_neutron_validation, run_pbso4_xray_validation, run_qarr_1g_validation,
    run_sucrose_lebail_validation, validation_datasets,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [command] if command == "datasets" => {
            for dataset in validation_datasets() {
                println!("{}", dataset.dataset_id);
            }
        }
        [command, runner, directory] if command == "run" => {
            let directory = Path::new(directory);
            let report = match runner.as_str() {
                "aps-sucrose-11bmb" => run_sucrose_lebail_validation(directory)?,
                "iucr-qarr-1g" => run_qarr_1g_validation(directory)?,
                "gsasii-pbso4-cw-neutron" => run_pbso4_neutron_validation(directory)?,
                "gsasii-pbso4-cw-x-ray" => run_pbso4_xray_validation(directory)?,
                _ => return Err(format!("unknown validation runner {runner:?}").into()),
            };
            println!("{}", report.to_json()?);
            if report.status != phasesmith_validation::ValidationStatus::Passed {
                std::process::exit(1);
            }
        }
        _ => {
            return Err(concat!(
                "usage: phasesmith-validation datasets | ",
                "phasesmith-validation run <runner> <dataset-directory>"
            )
            .into());
        }
    }
    Ok(())
}
