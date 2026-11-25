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
| `publish_release.py` | Publishes or resumes matching GitHub and crates.io releases. |
| `test_publish_release.py` | Tests publication recovery and source identity checks without publishing. |

## Build and publish a release

Increase `package.version` in `Cargo.toml`, refresh `Cargo.lock` with Cargo and
merge into `main`. The workflow automatically tests and publishes all four native
binary archives and the corresponding crate from the same commit, using the
existing `CARGO_REGISTRY_TOKEN` secret. No separate tag or publication step is
needed. Pushes to `release/**` validate without publication.

See [Releasing Serenade](../docs/RELEASING.md) for validation, retry behavior and
optional manual workflow runs. Failed publication can be resumed by rerunning the
workflow; do not delete drafts or move version tags.

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
