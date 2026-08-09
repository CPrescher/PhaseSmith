//! Python-free contracts for checksum-pinned real-data validation.

mod dataset;
mod pbso4;
mod qarr;
mod report;
mod sucrose;

pub use dataset::{
    DatasetVerificationError, ExternalValidationFile, ValidationDataset, validation_dataset,
    validation_datasets, verify_validation_dataset,
};
pub use pbso4::{Pbso4ValidationError, run_pbso4_neutron_validation, run_pbso4_xray_validation};
pub use qarr::{QarrValidationError, run_qarr_1g_validation};
pub use report::{
    RealDataValidationReport, ValidationCheck, ValidationContractError, ValidationStatus,
};
pub use sucrose::{SucroseValidationError, run_sucrose_lebail_validation};
