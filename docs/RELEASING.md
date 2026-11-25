# Releasing Serenade

`Cargo.toml`'s `package.version` is the single source of truth. Increase it and
refresh the derived `Cargo.lock` with Cargo, then merge the change into `main`.
No tag creation, workflow dispatch, environment approval or local publication is
required. The existing repository secret `CARGO_REGISTRY_TOKEN` authenticates
crates.io publication; it must be a valid token authorized to publish `serenade`.
GitHub publication uses the workflow's `contents: write` permission.

The [release workflow](../.github/workflows/release.yml) detects a version increase
between the previous and new `main` commits and resolves the new commit once.
Every job checks out that immutable commit. The workflow:

1. Tests, builds, packages and smoke-tests the three applications on Windows
   x86_64, Linux x86_64, macOS x86_64 and macOS ARM64.
2. Tests the minimal-feature library, checks its dependency graph, builds its
   documentation with warnings denied and validates the default-feature crate
   using `cargo publish --locked --dry-run`.
3. Creates the matching `v<package.version>` tag and a draft GitHub release,
   uploads all four archives and SHA-256 checksums, and verifies downloaded assets.
4. Publishes the crate using `cargo +stable publish --locked` and the existing
   token, confirms the registry's crate records that exact source commit, then
   makes the GitHub release public.

Both distributions therefore identify the same version and commit. A source JSON
asset records the version, commit and archive hashes. Default Cargo features build
exactly `tpe_hyperparameter_optm`, `evaluator` and `serving`; library consumers can
use `--no-default-features` to avoid application dependencies. The package allowlist
includes source, examples, tests, documentation, Apache-2.0 license and the upstream
WSKNN BSD notice, excluding local data and build artifacts.

## Retries and recovery

GitHub and crates.io cannot be published transactionally. A failed upload leaves
a recoverable draft rather than requiring deletion or a separate local publish.
Rerun the failed workflow: publication jobs for each version are serialized, matching draft assets
can be replaced, and a matching already published crate is skipped. Cargo upload
timeouts are checked against the actual registry archive before reporting failure.
An already public GitHub release is downloaded and verified, never overwritten;
its original archives can differ from a later rebuild. Missing crate publication
can still be completed for that matching release.

A conflicting tag, a crate from another commit, missing provenance or corrupted
assets fails closed. Never move a published tag or reuse a version for different
source. Investigate such a conflict rather than trying to overwrite it. Registry
or GitHub outages require a workflow retry after service recovery, not manual
publication. If the original run is unavailable, dispatch the workflow with
`publish=true` and its existing version tag to resume that exact commit.

Pushes to `main` without a version increase skip builds/publication. Pushes to
`release/**` and dispatches with `publish=false` validate without publishing.
An explicit `release/<version>` branch is optional. Pushing a matching version
tag or dispatching with `publish=true` also builds, validates and publishes that
tag's commit; the tag must match its Cargo version. No release path edits the
version or selects a second source commit.

Local checks (no publication):

```sh
cargo test --locked
cargo test --locked --no-default-features
cargo doc --locked --no-default-features --no-deps
cargo build --locked --release
cargo publish --locked --dry-run
python -m unittest discover -s scripts
```
