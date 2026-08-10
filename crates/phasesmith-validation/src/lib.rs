//! Python-free contracts for checksum-pinned real-data validation.

mod background;
mod dataset;
mod echidna;
mod nist_srm660c;
mod pbso4;
mod powgen_tof;
mod qarr;
mod report;
mod sucrose;

pub use dataset::{
    DatasetVerificationError, ExternalValidationFile, ValidationDataset, ValidationPurpose,
    validation_dataset, validation_datasets, verify_validation_dataset,
};
pub use echidna::{EchidnaValidationError, run_echidna_lab6_validation};
pub use nist_srm660c::{NistSrm660cValidationError, run_nist_srm660c_validation};
pub use pbso4::{Pbso4ValidationError, run_pbso4_neutron_validation, run_pbso4_xray_validation};
pub use powgen_tof::{
    PowgenTofValidationError, run_powgen_tof_readiness, run_powgen_tof_validation,
};
pub use qarr::{QarrValidationError, run_qarr_1g_validation, run_qarr_1h_validation};
pub use report::{
    RealDataValidationReport, ValidationCheck, ValidationContractError, ValidationStatus,
};
pub use sucrose::{SucroseValidationError, run_sucrose_lebail_validation};
