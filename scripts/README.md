# Repository scripts

## Mandatory for GitHub releases

The following files are required by [.github/workflows/release.yml](../.github/workflows/release.yml).
Keep them unless the workflow and its dependencies are updated together.

| File | Release responsibility |
|---|---|
| `release_version.py` | Detects package version increases for publication. |
| `test_release_version.py` | Validates release version logic before tagging. |
| `package_release.py` | Smoke-tests and packages native release applications. |
| `test_package_release.py` | Validates packaging before building release artifacts. |

## Build and publish a release

1. Create a branch from `main` named `release/<version>`.
2. Increase the package version in both `Cargo.toml` and `Cargo.lock`.
3. Push the branch and open a PR. Pushes to `release/**` test and build binaries
   as workflow artifacts; they do not publish a GitHub Release.
4. Merge the PR normally. A push to `main` that increases the package version
   tests, builds, packages and smoke-tests Windows x86_64, Linux x86_64,
   macOS x86_64 and macOS ARM64 binaries from the merged commit, then creates
   the version tag and publishes the archives and checksums to GitHub Releases.

Pushes to `main` without a version increase skip
binary builds and publication. Publication requires every platform build to
succeed. A failed run can be rerun after investigating the failure; an incomplete
release draft must be inspected and removed explicitly before retrying publication.

### Manual runs and tag releases

The **Binary release** workflow also supports **Run workflow** for build-only
validation. To publish manually, first create a version tag on the intended
commit, then select **Publish a GitHub Release** and enter that existing tag
(for example `v0.3.6`). Pushing a version tag also builds and publishes it.
Do not start a tag release while an automatic release for the same version is
running.

Workflow tokens must be allowed to write repository contents for publication.
Normal PR reviews still apply.

## Optional development tools

These files are not required by the GitHub release workflow:

| File | Purpose |
|---|---|
| `validate_workflows.py` | Locally smoke-tests built applications using the packaging helpers. |
| `scale_local_datasets.py` | Scales local train/valid/test datasets, including the item catalog. |
| `scale_benchmark_sample.py` | Scales the bundled sample and produces throughput query prefixes. |
| `test_scale_benchmark_sample.py` | Tests bundled-sample scaling. |

Evaluation and benchmark commands are documented in the [README](../README.md#offline-evaluation-and-optional-benchmarking).

Scale extracted train/valid/test splits with `python scripts/scale_local_datasets.py --input-dir assets/example --output target/scaled --factor 100 --catalog-factor 10`.
This writes TSV event files with remapped sessions and catalogs. The sample scaler
instead writes a training TSV and query prefixes; use `--help` for its options.
