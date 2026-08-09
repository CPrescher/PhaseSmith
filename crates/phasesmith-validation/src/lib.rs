//! Python-free contracts for checksum-pinned real-data validation.

mod dataset;
mod report;
mod sucrose;

pub use dataset::{
    DatasetVerificationError, ExternalValidationFile, ValidationDataset, validation_dataset,
    validation_datasets, verify_validation_dataset,
};
pub use report::{
    RealDataValidationReport, ValidationCheck, ValidationContractError, ValidationStatus,
};
pub use sucrose::{SucroseValidationError, run_sucrose_lebail_validation};
