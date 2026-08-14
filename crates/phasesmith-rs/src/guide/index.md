# Native Rust guide

These pages explain how the public crates fit together. They complement the
item reference generated from each type and function:

1. Start with [`getting_started`] for small compiled examples.
2. Read [`architecture`] before choosing where application-owned code belongs.
3. Keep [`scientific_conventions`] open when exchanging physical arrays.
4. Read [`cif_inputs`] before turning crystallographic files into model state.
5. Follow [`real_data_rietveld`] for a measured X-ray/neutron refinement.
6. Use [`refinement_operations`] for staging, limits, checkpoints, and result
   acceptance.
7. Use [`mathematics`] for the equations behind every implemented model.
8. Use [`workflows`] to select another calculation or refinement layer.
9. Follow [`application_hosts`] when integrating a desktop, service, or CLI.

[`getting_started`]: crate::guide::getting_started
[`architecture`]: crate::guide::architecture
[`scientific_conventions`]: crate::guide::scientific_conventions
[`cif_inputs`]: crate::guide::cif_inputs
[`real_data_rietveld`]: crate::guide::real_data_rietveld
[`refinement_operations`]: crate::guide::refinement_operations
[`mathematics`]: crate::guide::mathematics
[`workflows`]: crate::guide::workflows
[`application_hosts`]: crate::guide::application_hosts
