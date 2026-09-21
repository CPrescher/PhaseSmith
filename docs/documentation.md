# Documentation development

The public site uses MkDocs Material and is hosted by Read the Docs. The build
does not install PhaseSmith itself, so documentation previews do not compile
the native extension and cannot accidentally import development code.

## Build locally

```shell
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python scripts/sync_math_docs.py --check
python scripts/sync_skill.py --check
mkdocs build --strict
```

For a live preview:

```shell
mkdocs serve
```

The generated `site/` directory is disposable and ignored by Git. CI uses the
same strict build as Read the Docs, so unresolved internal links and invalid
configuration fail before merge.

## Shared mathematical reference

The canonical equations live in
`crates/phasesmith-rs/src/guide/mathematics/` so they ship with the facade crate
and render on docs.rs. Committed pages under `docs/mathematics/` are generated
from those sources for the Python-facing Read the Docs site:

```shell
python scripts/sync_math_docs.py
python scripts/sync_math_docs.py --check
```

The generator changes rustdoc-only links into portable Rust owner names, adds
the matching Python modules, and links both API references. Edit the canonical
Rust Markdown and regenerate; do not edit the generated pages directly. CI
rejects stale generated copies before building MkDocs.

## Shared agent skill

The canonical user-facing skill lives in `skills/phasesmith-ai-workflows/`.
Edit its `SKILL.md` and focused `references/` files there, then regenerate:

```shell
python scripts/sync_skill.py
python scripts/sync_skill.py --check
```

The generator needs only the Python standard library and does not import
PhaseSmith. It copies the whole skill to
`python/phasesmith/_skills/phasesmith-ai-workflows/` for wheels/sdists and renders
the entrypoint and references under `docs/agent-skill/`. The entrypoint's YAML
frontmatter is omitted from the rendered page and its headings are adjusted;
the operating instructions are otherwise the same. Agent UI metadata ships
with the skill but is not part of the rendered protocol. Do not edit either
generated tree directly. Regeneration removes obsolete files in those two
managed trees. CI, Read the Docs, and release validation reject stale copies.

`tests/test_skill.py` exercises discovery from an unrelated working directory,
complete resource parity with the canonical skill, text output, CLI errors,
and drift detection. The release wheel and sdist test jobs run these tests
against the installed package. The offline advisor-cycle example exercises the
actual plan/proposal/lint/run/review contracts; it is synthetic software advice,
not a measurement of model quality. For substantial instruction changes, also
test fresh agent contexts on the scenarios in
`tests/skill_evaluation/README.md`, and retain their outcomes separately from
deterministic test results.

## Mathematical notation

MkDocs-authored pages render TeX with MathJax. Use `\(...\)` for inline
notation and `\[...\]` for display equations. Use an aligned environment for
related rows:

```text
\[
\begin{aligned}
y &= f(x), \\
\frac{\partial y}{\partial x} &= f'(x).
\end{aligned}
\]
```

In MkDocs-only source, do not put mathematical formulas in fenced `text`
blocks. Fences are reserved for literal file formats, array layouts, CLI
output, and pseudocode that users may need to copy. The shared mathematical
reference remains generated from rustdoc-compatible Markdown; migrate its
literal formulas only through a coordinated canonical-source and generator
change, never by editing `docs/mathematics/` directly. Keep physical units and
parameter conventions next to the equation or public field they disambiguate;
keep implementation-milestone history in planning documents rather than user
guides.

## Build the Rust API documentation

docs.rs renders the rustdoc content shipped inside each crate. Build the same
workspace documentation locally and reject broken intra-doc links or warnings:

```shell
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo test --doc --workspace
```

Open `target/doc/phasesmith/index.html` to review the facade landing page and
the `phasesmith::guide` hierarchy. Runnable rustdoc examples belong in
doctested code fences; longer cross-language explanations belong in this MkDocs
site.

## Read the Docs project setup

Repository configuration is committed in `.readthedocs.yaml`; the dashboard
only needs the one-time project import:

1. Sign in to Read the Docs Community with the GitHub account that administers
   `CPrescher/PhaseSmith`.
2. Install or authorize the Read the Docs GitHub App for the repository.
3. Choose **Add project**, select `CPrescher/PhaseSmith`, and continue.
4. Use project name and slug `phasesmith`, repository default branch `main`,
   and configuration file `.readthedocs.yaml`.
5. Confirm the first `latest` build succeeds.
6. Under **Admin → Automation Rules**, add an enabled rule matching **SemVer
   versions**, type **Tag**, with action **Activate version**. Future release
   tags will then build without a dashboard step.
7. In **Versions**, activate the existing `v0.1.0` tag manually; automation
   rules apply only to versions created after the rule. Read the Docs maps the
   greatest stable semantic release tag to `stable`.
8. Set the project’s default version to `stable` after that build succeeds.

The intended public URL is
[`https://phasesmith.readthedocs.io/`](https://phasesmith.readthedocs.io/).
GitHub integration then rebuilds `latest` on `main` updates. The automation
rule activates and builds each new semantic release tag.

## Version policy

- `latest` documents the current `main` branch and may describe unreleased
  behavior.
- `stable` follows the newest stable semantic release tag.
- Explicit tag URLs such as `/en/v0.1.0/` preserve the documentation shipped
  with that release.

Keep scientific conventions and public compatibility promises in versioned
pages. Avoid linking versioned documentation to mutable source files when the
content belongs in the docs themselves.
