# Public API snapshots

`python-public-api-v<version>.json` freezes every name in `__all__` for the
top-level package and its explicit I/O, refinement, integration, oracle, and
validation namespaces. Each entry records its public kind, implementation
target, and `inspect.signature` result when Python exposes one.

The snapshot contains no timestamp, platform path, object address, or evaluated
annotation. It is therefore reviewable and deterministic across supported
Python versions. Domain, adapter, and separately versioned validation surfaces
remain labelled as distinct tiers; inclusion does not merge their compatibility
policies.

Check the current release surface with:

```shell
python scripts/public_api_snapshot.py --check
```

When an intentional pre-1.0 API change is approved, first update the project
version, changelog, and migration notes, then write the new versioned path
without overwriting an older baseline:

```shell
python scripts/public_api_snapshot.py --write
```

The exact comparison deliberately rejects additions as well as removals or
signature changes. That makes every public-surface change visible in review.
Older snapshots are retained; they are release records, not generated cache.

The combined 0.7 candidate adds `python-public-api-v0.7.0.json` and
`python-pawley-api-v0.7.0.json`. The Pawley snapshot contains CW/TOF workflows
and mixed project bundles. The earlier unreleased Pawley snapshot is retained.
`python-public-api-develop-2e5edb7.json` preserves the post-release develop
variant formerly stored under the 0.5.0 name; the canonical 0.5.0 file now
matches tag `v0.5.0` exactly. See the release candidate review for the audit.
