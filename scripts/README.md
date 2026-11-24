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

## Merge a PR

1. Open a non-draft PR into `main` from a branch in this repository.
2. Update the branch with the latest `main`.
3. Click **Enable auto-merge** on the PR.

The single workflow, **Automatically increment app version, build and deploy
binaries, merge to main**, then:

- Increments the patch version in `Cargo.toml` and `Cargo.lock` on the PR branch.
  An explicit version increase is preserved, including on retries.
- Tests, builds, packages and smoke-tests Windows x86_64, Linux x86_64,
  macOS x86_64 and macOS ARM64 binaries from that exact commit.
- Passes the `release-binaries` status and merges using the PR's selected merge
  method. Required GitHub reviews and checks still apply.
- Verifies that the merged tree matches the built tree, then tags the merged
  commit and deploys the archives and checksums to GitHub Releases.

### Quick merge without a release

Check this box in the PR description before enabling auto-merge:

```markdown
- [x] Skip version bump, binary builds and release; merge changes only
```

The PR template includes the checkbox, unchecked by default. Checking it skips
all version changes, binary builds and release publication. Required reviews
and other branch checks still apply.

Disabling auto-merge cancels a pending release. Editing the checkbox or pushing
changes while auto-merge is enabled starts a new run; stale runs cannot publish.
If `main` changes, update the PR branch. Failed builds leave the PR open; rerun
failed jobs after fixing the failure. An incomplete release draft must be
inspected and removed before retrying the publication job.

### One-time repository settings (administrator)

- Enable **Allow auto-merge** in Settings > General > Pull Requests.
- Require the `release-binaries` status on `main`. Otherwise GitHub can merge
  immediately, before the workflow bumps the version or builds binaries.
- Allow workflow tokens to write contents, pull requests and commit statuses.

Opening a PR posts a pending `release-binaries` status. **Enable auto-merge**
starts the release; **Merge pull request** cannot be intercepted by a workflow.
There are no manual workflow inputs or separate push/tag workflows. Fork PRs
are excluded; use a branch in this repository. The workflow reads automation
from `main`; PR code runs only in build jobs with read-only permissions and
without persisted checkout credentials.

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
