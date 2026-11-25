# Releasing Serenade

The release workflow validates one immutable source commit before distributing
GitHub binaries and the crates.io package. The release tag must equal
`v<package.version>` from that commit's `Cargo.toml`. Default Cargo features build
exactly `tpe_hyperparameter_optm`, `evaluator` and `serving`; package users can
disable default features to compile only the recommendation library.

Before enabling publication, a crate owner must configure the `serenade` crate's
crates.io trusted publisher for GitHub repository `bolcom/serenade`, workflow
`release.yml`, environment `crates-io`. Create that GitHub environment and restrict
its publishing branches/tags and reviewers according to maintainer policy. The
[official crates.io authentication action](https://github.com/rust-lang/crates-io-auth-action)
exchanges GitHub OIDC identity for a temporary token and revokes it after the job.
No long-lived Cargo API token is stored in the repository or GitHub secrets.

Increment `package.version` on `main` to trigger the existing automatic release,
push a matching version tag, or dispatch the workflow with `publish=true` and an
existing matching tag. Dispatch with `publish=false` to validate without publishing.
Never reuse a version for different source. Both distributions check out the same
resolved commit; crate validation, minimal-feature tests/docs and all four native
binary builds must pass before publication. Cargo's publish command validates the
package again; disabling features for validation does not change the published
manifest's default features or `cargo install serenade` behavior.

The package allowlist includes source, examples, tests, documentation, Apache-2.0
license and the upstream WSKNN BSD notice. It excludes datasets, build artifacts,
local indexes and development reports. Review with `cargo package --list`.

GitHub and crates.io cannot be published transactionally. The job publishes the
verified GitHub release first, then the corresponding crate. If crates.io fails,
inspect the published tag and source, fix publisher permissions if necessary, and
publish that exact commit with `cargo publish --locked --no-default-features`.
Do not move the tag or increment the version to disguise an incomplete release.
The workflow refuses to overwrite existing GitHub releases; a partial release
requires maintainer recovery, rather than an automatic replacement.

Local checks (no publication):

```sh
cargo test --locked
cargo test --locked --no-default-features
cargo doc --locked --no-default-features --no-deps
cargo build --locked --release
cargo publish --locked --no-default-features --dry-run
python -m unittest discover -s scripts
```
