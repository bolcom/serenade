# Serenade binary packages

Packages contain serving, evaluator, and tpe_hyperparameter_optm (with .exe on
Windows), this guide, LICENSE.md, configs/hpo.toml, assets/example/workflows.toml and the bundled
train.txt, valid.txt and test.txt example data. No Rust installation is needed.
The example corpus is the existing repository example.zip, distributed with Serenade.

When an official release is published, its archives and `.sha256` files appear at
https://github.com/bkersbergen/serenade/releases. Choose your platform below.
Before extracting, verify the downloaded archive against its sidecar:
`sha256sum -c ARCHIVE.sha256` on Linux, `shasum -a 256 -c ARCHIVE.sha256`
on macOS, or compare `(Get-FileHash .\ARCHIVE.zip -Algorithm SHA256).Hash`
with `Get-Content .\ARCHIVE.zip.sha256` in PowerShell.

Extract the archive and open a terminal in its single top-level directory.
On Linux/macOS run:

```sh
./tpe_hyperparameter_optm configs/hpo.toml
./evaluator hpo-results/stan.toml
./serving hpo-results/stan.toml
```

On Windows PowerShell run:

```powershell
.\tpe_hyperparameter_optm.exe configs/hpo.toml
.\evaluator.exe hpo-results/stan.toml
.\serving.exe hpo-results/stan.toml
```

The default HPO file enables all five algorithms and performs 50 seeded trials per algorithm.
Edit `configs/hpo.toml` to change datasets, budget, seed or ranges. Set
`enabled = false` in any `[hyperparam.algorithms.NAME]` section to skip it.
See the range examples in `configs/hpo.toml` and the repository guide
`docs/HYPERPARAMETER_SEARCH.md` for range details.
Append `stan` to the TPE command to optimize only that enabled algorithm.
Run `./evaluator hpo-results` (Windows: `.\evaluator.exe hpo-results`) to evaluate
all exported models on test data with their own selected settings.
Keep the working directory at the package root: all dataset paths are relative
to it. TPE writes one complete `hpo-results/NAME.toml` per enabled algorithm, plus JSON reports.
The same TOML works unchanged for evaluation and serving. Serving creates
sessions.db and listens on port 8081. Request recommendations with:

```sh
curl 'http://localhost:8081/v1/recommend?item_id=13598&session_id=example&user_consent=true'
```

In Windows PowerShell use `curl.exe` for the same request.

Stop serving with Ctrl+C. Use a writable directory for results and sessions.db.
The package verifier runs these operations from a fresh extracted archive,
checks nonempty numeric recommendations, and verifies that reloading the selected
model reproduces its validation score.

## Platforms

- Windows x86_64: MSVC, ZIP; Windows Server 2022 runner, static C runtime.
- Linux x86_64: GNU, tar.gz; Ubuntu 22.04 runner. Requires glibc 2.35 or newer
  and libstdc++6 providing GLIBCXX_3.4.30; this is not a musl/static Linux build.
- macOS Intel x86_64: macos-15-intel, tar.gz.
- macOS Apple Silicon aarch64: macos-15, tar.gz.

macOS packages target macOS 15 or later and are unsigned and unnotarized.
RocksDB is compiled into the executables; users do not install RocksDB or LLVM.
LLVM/libclang and CMake are build dependencies only. Native runner smoke tests
must pass before any archive is uploaded. Linux ARM64 is not provided.

## Validation and release maintenance

In GitHub Actions select **Binary release**, **Run workflow**, and the feature
branch or **main** to validate the merged workflow. Pushes to release/** branches
also run validation. Both paths build all four platforms without
creating a GitHub release.
Download serenade-* from the run's Artifacts section (requires GitHub sign-in).
Unzip the Actions artifact wrapper to obtain the platform archive and its SHA-256
sidecar. Artifacts expire after 14 days.

After validating the intended commit, an authorized maintainer can create and
push a version tag such as v0.4.0. Tag pushes run the same tests, release builds,
package extraction and smoke tests. Only after every platform passes does a
separate job with contents:write create one draft release, upload all archives
and SHA-256 files, and publish it. The normal GITHUB_TOKEN is sufficient.
Crates.io publication is separate and is not performed here.

Existing releases are never overwritten. If upload fails, a draft can remain;
inspect it, explicitly delete the incomplete draft, then rerun the workflow.
A build failure cannot publish a partial release. Do not create release tags
as part of validation.

Manual validation exercises the downloaded-package workflow but skips the
publication job. The tag trigger, cross-job artifact download, release-token
permissions, draft creation, asset upload, and final publication still require
an explicitly approved tagged release for end-to-end verification. Publication
uses a draft to keep assets private until upload succeeds; it is not a
transaction, and an interrupted publication can leave a draft for inspection.

For local packaging after building the required target binaries:

```sh
python scripts/package_release.py --target x86_64-unknown-linux-gnu --platform linux-x86_64 --version validation-local
```

Archives have a single serenade-VERSION-PLATFORM directory. The exact file set
and hashes are checked after extraction; smoke-test outputs are never packaged.
Cargo.lock is enforced. Archive timestamps and compiler output are not promised
to be byte-for-byte reproducible across builds.
