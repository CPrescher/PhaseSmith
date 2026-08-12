//! Command-line entry point for Python-free checksum-pinned validation.

use std::error::Error;
use std::path::Path;

use phasesmith_validation::{
    run_echidna_lab6_validation, run_nickel_tof_validation, run_nist_srm660c_validation,
    run_pbso4_neutron_validation, run_pbso4_xray_validation, run_powgen_tof_validation,
    run_qarr_1g_validation, run_qarr_1h_validation, run_sucrose_lebail_validation,
    validation_datasets,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [command] if command == "datasets" => {
            for dataset in validation_datasets() {
                println!("{}", dataset.dataset_id);
            }
        }
        [command] if command == "manifest" => {
            println!("{}", serde_json::to_string_pretty(&validation_datasets())?);
        }
        [command, runner, directory] if command == "run" => {
            let directory = Path::new(directory);
            let report = match runner.as_str() {
                "ansto-echidna-lab6-cw-neutron" => run_echidna_lab6_validation(directory)?,
                "aps-sucrose-11bmb" => run_sucrose_lebail_validation(directory)?,
                "iucr-qarr-1g" => run_qarr_1g_validation(directory)?,
                "iucr-qarr-1h" => run_qarr_1h_validation(directory)?,
                "nist-srm660c-lab6-xray" => run_nist_srm660c_validation(directory)?,
                "lanl-nickel-tof" => run_nickel_tof_validation(directory)?,
                "powgen-lab6-tof-calibration" => run_powgen_tof_validation(directory)?,
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
                "phasesmith-validation manifest | ",
                "phasesmith-validation run <runner> <dataset-directory>"
            )
            .into());
        }
    }
    Ok(())
}
