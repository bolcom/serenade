# Repository scripts

## Mandatory for GitHub releases

The following files are required by [.github/workflows/release.yml](../.github/workflows/release.yml).
Keep them unless the workflow and its dependencies are updated together.

| File | Release responsibility |
|---|---|
| `release_version.py` | Determines the release tag from version changes. |
| `test_release_version.py` | Validates release version logic before tagging. |
| `package_release.py` | Smoke-tests and packages native release applications. |
| `test_package_release.py` | Validates packaging before building release artifacts. |

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
