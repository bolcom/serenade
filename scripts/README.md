# Repository scripts

## Mandatory for GitHub releases

The following files are required by [.github/workflows/release.yml](../.github/workflows/release.yml).
Keep them unless the workflow and its dependencies are updated together.

| File | Release responsibility |
|---|---|
| `release_version.py` | Bumps and validates Cargo manifest and lockfile versions. |
| `release_pr.py` | Prepares the release PR and merges its tested commit. |
| `test_release_pr.py` | Checks merge and publication guards. |
| `test_release_version.py` | Validates release version logic before tagging. |
| `package_release.py` | Smoke-tests and packages native release applications. |
| `test_package_release.py` | Validates packaging before building release artifacts. |

## Release a PR

1. Push a `release/*` branch and open a non-draft PR into `main`.
2. Update the branch with the latest `main`.
3. In Actions, select **Binary release**, **Run workflow**, branch **main**,
   and enter the PR number.

This is the only workflow and its only trigger. It:

- Bumps the patch version in `Cargo.toml` and `Cargo.lock` on the PR branch.
  An explicit version increase is preserved, including on retries.
- Tests, builds, packages and smoke-tests Windows x86_64, Linux x86_64,
  macOS x86_64 and macOS ARM64 binaries from that exact commit.
- Sets the `release-binaries` commit status and squash-merges the PR only
  after all four builds pass. Required GitHub reviews and checks still apply.
- Checks that the merged tree matches the tested tree, then tags the merged
  commit and publishes the built archives and checksums.

If the PR or `main` changes during the build, update the branch and restart.
Failed builds leave the PR open. An incomplete release draft must be inspected
and removed before retrying publication.

Repository settings must allow squash merges and workflow tokens to write
contents, pull requests and commit statuses. To prevent manual merges before
release builds pass, require the `release-binaries` status in the `main` branch
rules. The workflow does not bypass branch protection.

## Optional development tools

These files are not required by the GitHub release workflow:

| File | Purpose |
|---|---|
| `validate_workflows.py` | Locally smoke-tests built applications using the packaging helpers. |
| `scale_local_datasets.py` | Scales local train/valid/test datasets, including the item catalog. |
| `scale_benchmark_sample.py` | Scales the bundled sample and produces throughput query prefixes. |
| `test_scale_benchmark_sample.py` | Tests bundled-sample scaling. |

Benchmark usage is documented in [docs/knn-benchmark.md](../docs/knn-benchmark.md).

For commands to expand the bundled train, validation and test splits to millions
of sessions or billions of interactions, see
[synthetic scaling](../docs/knn-benchmark.md#synthetic-scaling). That section also
explains the different output formats of the two scaling scripts.
