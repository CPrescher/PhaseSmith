#![doc = include_str!("guide/index.md")]

#[doc = include_str!("guide/getting_started.md")]
pub mod getting_started {}

#[doc = include_str!("guide/architecture.md")]
pub mod architecture {}

#[doc = include_str!("guide/scientific_conventions.md")]
pub mod scientific_conventions {}

#[doc = include_str!("guide/workflows.md")]
pub mod workflows {}

#[doc = include_str!("guide/cif_inputs.md")]
pub mod cif_inputs {}

#[doc = include_str!("guide/real_data_rietveld.md")]
pub mod real_data_rietveld {}

#[doc = include_str!("guide/refinement_operations.md")]
pub mod refinement_operations {}

#[doc = include_str!("guide/application_hosts.md")]
pub mod application_hosts {}

/// Mathematical definitions for the implemented scientific models.
pub mod mathematics;
