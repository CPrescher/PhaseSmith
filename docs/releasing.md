# Releasing PhaseSmith

PhaseSmith uses one SemVer value for the Rust workspace, Python distribution,
and Tauri configuration. A `v<version>` tag starts
`.github/workflows/release.yml`; an untagged manual dispatch builds and tests
the artifacts without publishing anything.

The release workflow builds CPython-ABI3 wheels for Python 3.11 and newer on
manylinux x86-64/AArch64, macOS x86-64/Apple Silicon, and Windows x86-64. It
also builds an sdist, tests native wheels, creates build-provenance
attestations, publishes through PyPI trusted publishing, and attaches every
distribution plus `SHA256SUMS` to a GitHub Release. The Tauri runtime probe is
not a product GUI and is deliberately not bundled as a release artifact.

## One-time registry bootstrap

PyPI can create the first `phasesmith` project from a pending trusted
publisher. Configure it for:

- owner `CPrescher`;
- repository `PhaseSmith`;
- workflow `release.yml`;
- environment `pypi`.

Create matching `pypi` and `crates-io` GitHub environments. Protection rules
are optional but recommended for the publishing jobs.

crates.io requires each first crate version to be published by an authenticated
owner before trusted publishing can be configured. Log in locally, review the
dry run, and publish the initial dependency-ordered workspace exactly once:

```shell
cargo publish --workspace --locked --dry-run \
  --exclude phasesmith-desktop \
  --exclude phasesmith-py \
  --exclude phasesmith-validation \
  --exclude phasesmith-tauri

cargo publish --workspace --locked \
  --exclude phasesmith-desktop \
  --exclude phasesmith-py \
  --exclude phasesmith-validation \
  --exclude phasesmith-tauri
```

Publishing is permanent. Do not run the second command until the packaged
sources, ownership, version, and release commit have been reviewed.

After the first upload, add the same trusted publisher to all nine public
crates (`phasesmith` and its eight `phasesmith-*` dependencies), using workflow
`release.yml` and environment `crates-io`. Do not create the repository
variable yet: the initial tag must not try to upload version 0.1.0 twice. The
crates.io job remains safely skipped while the variable is absent.

## Release checklist

1. Update `CHANGELOG.md` and synchronize the versions in `Cargo.toml`,
   `pyproject.toml`, and `apps/phasesmith-desktop/src-tauri/tauri.conf.json`.
2. Run `python scripts/release_version.py --tag v<version>`.
3. Run the complete Rust and Python gates and the crates.io dry run.
4. Run the Release workflow manually. This builds and tests artifacts but does
   not publish them.
5. Commit and push the release state to `main`; confirm CI is green.
6. For version 0.1.0 only, perform the crates.io bootstrap above before tagging.
7. Create one annotated tag on the reviewed commit and push it:

   ```shell
   git tag -a v<version> -m "PhaseSmith <version>"
   git push origin v<version>
   ```

8. Watch every Release workflow job. Verify the GitHub Release, PyPI files,
   crates.io versions, checksums, and installation from clean environments.
9. After the successful 0.1.0 workflow, create the repository variable
   `CRATES_IO_TRUSTED_PUBLISHING=true`. Later tags will then publish crates.io
   through OIDC as part of the workflow.

Never move or reuse a published tag. A bad crates.io version can only be
yanked, and a corrected release must use a new version. PyPI files for an
existing version likewise cannot be replaced.
