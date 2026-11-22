# Serenade binary packages

Packages contain serving, evaluator, and tpe_hyperparameter_optm (with .exe on
Windows), this guide, LICENSE.md, assets/example/workflows.toml and the bundled
train.txt, valid.txt and test.txt example data. No Rust installation is needed.
The example corpus is the existing repository example.zip, distributed with Serenade.

Extract the archive and open a terminal in its single top-level directory.
On Linux/macOS run:

```sh
./evaluator assets/example/workflows.toml vmis evaluation.json
./tpe_hyperparameter_optm assets/example/workflows.toml vmis hpo
./serving hpo/vmis-serving.toml
```

On Windows PowerShell use .\evaluator.exe, .\tpe_hyperparameter_optm.exe and
.\serving.exe with the same arguments. The example performs three seeded trials.
Keep the working directory at the package root: all dataset paths are relative
to it. TPE writes its selected model to hpo/vmis-serving.toml. Serving creates
sessions.db and listens on port 8081. Request recommendations with:

```sh
curl 'http://localhost:8081/v1/recommend?item_id=13598&session_id=example&user_consent=true'
```

Stop serving with Ctrl+C. Use a writable directory for results and sessions.db.
The package verifier runs these operations from a fresh extracted archive,
checks nonempty numeric recommendations, and verifies that reloading the selected
model reproduces its validation score.

## Platforms

- Windows x86_64: MSVC, ZIP; Windows Server 2022 runner, static C runtime.
- Linux x86_64: GNU, tar.gz; Ubuntu 22.04 runner. Requires glibc 2.35 or newer
  and the system C++ runtime (libstdc++6); this is not a musl/static Linux build.
- macOS Intel x86_64: macos-15-intel, tar.gz.
- macOS Apple Silicon aarch64: macos-15, tar.gz.

macOS packages target macOS 15 or later and are unsigned and unnotarized.
RocksDB is compiled into the executables; users do not install RocksDB or LLVM.
LLVM/libclang and CMake are build dependencies only. Native runner smoke tests
must pass before any archive is uploaded. Linux ARM64 is not provided.

## Validation and release maintenance

In GitHub Actions select **Binary release**, **Run workflow**, and the feature
branch. Once this workflow is on the default branch, GitHub enables manual
dispatch. Pushes to release/** branches also run validation, allowing the workflow
to be checked before merging it. Both paths build all four platforms without
creating a GitHub release.
Download binaries-* from the run's Artifacts section (requires GitHub sign-in).
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

For local packaging after building the required target binaries:

```sh
python scripts/package_release.py --target x86_64-unknown-linux-gnu --platform linux-x86_64 --version validation-local
```

Archives have a single serenade-VERSION-PLATFORM directory. The exact file set
and hashes are checked after extraction; smoke-test outputs are never packaged.
Cargo.lock is enforced. Archive timestamps and compiler output are not promised
to be byte-for-byte reproducible across builds.
