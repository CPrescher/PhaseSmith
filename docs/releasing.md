# Releasing PhaseSmith

PhaseSmith uses one SemVer for the Rust workspace and Python distribution.
A `v<version>` tag starts `.github/workflows/release.yml`; an untagged manual
`workflow_dispatch` builds and tests packages without uploading to registries or
creating a GitHub Release. Review the [0.8 dry-run record](release-dry-run-0.8.md)
for the exact candidate and results, rather than treating this procedure as a receipt.

## What the release workflow checks

The workflow validates versions, verifies all eleven public Rust package archives
in dependency order, and builds strict documentation. It builds ABI3 wheels for
Python 3.11+ on manylinux x86-64/AArch64, macOS Intel/Apple Silicon and Windows
x86-64. Each wheel is tested on a matching native runner, including Linux ARM.
The source distribution is built and installed outside the checkout, then tested.
Numerical thread limits are inherited before Python starts so exact serial/spawned
positional checks use the documented configuration, including with NumPy 1.26.

Manual runs retain wheel/sdist, Rust archive and rendered documentation artifacts.
Tag runs additionally attest Python distributions, publish PyPI and Rust packages,
and attach the distributions plus SHA-256 checksums to a GitHub Release. Registry
publishing waits for all package/documentation jobs. The crates.io job skips
crates that already carry the release version (`scripts/published_crates.py`
reads the registry index), so after a partial failure, such as a crate without
a trusted publisher, fix the cause and re-run the failed jobs. The GitHub
Release job follows once crates.io succeeds. General CI separately checks
Python 3.11–3.14, minimum NumPy, MSRV, Clippy, Rust tests and API contracts.
GUI applications are separate consumers, outside this library release.

## Registry configuration and new crate names

PyPI trusted publishing uses owner `CPrescher`, repository `PhaseSmith`, workflow
`release.yml` and environment `pypi`. The corresponding Rust environment is
`crates-io`; its workflow job is enabled by repository variable
`CRATES_IO_TRUSTED_PUBLISHING=true`.

Rust trusted publishing cannot create a new crate name. Its first version must
be uploaded by an authenticated owner before that crate's trusted publisher can
be configured ([Rust project documentation](https://blog.rust-lang.org/2025/07/11/crates-io-development-update-2025-07/)).
The 0.8 workspace has two new names: `phasesmith-single-crystal` and
`phasesmith-hdf5`. Three formerly separate, unpublished development crates have
been consolidated; see [migration](migration-0.8.md).

The existing registry variable does not prove publisher coverage for these new
names. Before tagging 0.8, use this reviewed bootstrap sequence:

1. Finish every check below on the exact release commit. Review all eleven
   `.crate` payloads and notices. Verify ownership and that 0.8.0 is unpublished.
2. Temporarily set `CRATES_IO_TRUSTED_PUBLISHING=false` to prevent a tag workflow
   from duplicating the owner-authenticated Rust publication.
3. With an owner credential, first run the nonpublishing command:

   ```sh
   cargo publish --workspace --locked --dry-run \
     --exclude phasesmith-py --exclude phasesmith-validation
   ```

4. Only after final publication approval, run the same command without
   `--dry-run`. This publishes the entire dependency-ordered 0.8 workspace,
   including dependencies needed by the new crate. Do not publish just the new
   crate against unavailable 0.8 dependencies. If interrupted, inspect registry
   state (`python scripts/published_crates.py --version <version>`) and
   explicitly exclude already-published packages before resuming.
5. Configure the same trusted publisher on all eleven public crates. Tag the
   reviewed commit with the Rust job disabled, verify PyPI/GitHub publication,
   then restore `CRATES_IO_TRUSTED_PUBLISHING=true` for later releases.

These are final publication operations, not part of a dry run. Never put registry
credentials in scripts, output logs or committed files. If more new crate names
are added later, repeat their ownership/publisher review before tagging.

## Release checklist

1. Review the [package audit](package-review-0.8.md) and scientific limitations.
   Update `CHANGELOG.md`, replace candidate-only installation/status text in the
   reviewed release commit, and synchronize Cargo/Python versions.
2. Review the versioned public API snapshot; run
   `python scripts/public_api_snapshot.py --check` and
   `python scripts/release_version.py --tag v<version>`.
3. Run `cargo fmt --all -- --check`, strict all-feature/all-target Clippy,
   `cargo test --workspace --all-features`, strict rustdoc and the complete
   default-marker Python suite. CI must pass minimum Rust and NumPy as well.
   Run `python scripts/sync_math_docs.py --check`,
   `python scripts/sync_skill.py --check`, Ruff and `mkdocs build --strict`.
4. Verify the installed wheel outside the checkout, including NumPy-only native
   capability smoke, the runnable single-crystal tutorial and
   `phasesmith skill --path` / `phasesmith skill --print all`. Run the offline
   advisor-cycle example in a new directory. Substantial skill changes also need
   the fresh agent scenarios in `tests/skill_evaluation/README.md`.
5. Run the Release workflow manually on the exact candidate branch. Wait for all
   build/test jobs and review artifact contents/checksums. Record successful run
   URLs and any intentionally excluded external-data/oracle tests. A queued run
   or `cargo package --no-verify` is not a completed release dry run.
6. Integrate the reviewed candidate into `develop` and `main` without losing other
   work. Confirm the release commit and green CI, and complete the new-name
   bootstrap above if applicable. An integration change needs revalidation.
7. After final publication approval, create and push one annotated tag:

   ```sh
   git tag -a v<version> -m "PhaseSmith <version>"
   git push origin v<version>
   ```

8. Watch every publishing job. Verify installed PyPI files, crates.io versions,
   GitHub checksums/attestations and Read the Docs `stable` for that version.
   A successful local documentation build does not update the public site. The
   Read the Docs project must build the new main/tag and activate its version.

Never move or reuse a published tag. Published crate versions can only be yanked;
corrected releases require a new version. PyPI files cannot be replaced either.
