//! Standalone host for the same file workflow used by GUI applications.
fn main() -> phasesmith::single_crystal_files::Result<()> {
    phasesmith::single_crystal_files::run_cli()
}
