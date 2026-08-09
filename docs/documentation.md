# Documentation development

The public site uses MkDocs Material and is hosted by Read the Docs. The build
does not install PhaseSmith itself, so documentation previews do not compile
the native extension and cannot accidentally import development code.

## Build locally

```shell
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
mkdocs build --strict
```

For a live preview:

```shell
mkdocs serve
```

The generated `site/` directory is disposable and ignored by Git. CI uses the
same strict build as Read the Docs, so unresolved internal links and invalid
configuration fail before merge.

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
