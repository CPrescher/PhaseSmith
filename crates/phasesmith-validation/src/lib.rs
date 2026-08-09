//! Python-free contracts for checksum-pinned real-data validation.

mod dataset;
mod report;

pub use dataset::{
    DatasetVerificationError, ExternalValidationFile, ValidationDataset, validation_dataset,
    validation_datasets, verify_validation_dataset,
};
pub use report::{
    RealDataValidationReport, ValidationCheck, ValidationContractError, ValidationStatus,
};
