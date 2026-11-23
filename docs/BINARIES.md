# Using Serenade binaries

Download your platform's archive from [GitHub Releases](https://github.com/bolcom/serenade/releases), extract it, and open a terminal in the extracted directory. Packages include `tpe_hyperparameter_optm`, `evaluator`, `serving`, example data, and configurations. No Rust installation is required.

## Platform requirements

| Platform | Package | Requirements |
| --- | --- | --- |
| Windows x86_64 | ZIP | Windows x86_64 |
| Linux x86_64 | tar.gz | glibc 2.35+ and libstdc++ with GLIBCXX_3.4.30+ |
| macOS Intel / Apple Silicon | tar.gz | macOS 15+; choose the matching architecture |

macOS executables are unsigned and unnotarized. RocksDB is included in the executables.

Optionally verify the archive against its accompanying `.sha256` file using `sha256sum -c ARCHIVE.sha256` on Linux, `shasum -a 256 -c ARCHIVE.sha256` on macOS, or `Get-FileHash .\ARCHIVE.zip -Algorithm SHA256` in PowerShell.

## Optimize, evaluate, serve

Run from the extracted package directory so relative dataset paths resolve correctly:

```sh
./tpe_hyperparameter_optm configs/hpo.toml
./evaluator hpo-results/stan.toml
./serving hpo-results/stan.toml
```

In Windows PowerShell:

```powershell
.\tpe_hyperparameter_optm.exe configs/hpo.toml
.\evaluator.exe hpo-results/stan.toml
.\serving.exe hpo-results/stan.toml
```

The default HPO configuration runs 50 trials for each of the five algorithms. Edit dataset paths, trial budget, objective, and algorithm ranges in `configs/hpo.toml`; see [Configuration](CONFIG.md). Training and validation select the best settings; test data is used only for evaluation.

HPO writes one complete `ALGORITHM.toml` per enabled algorithm to `hpo-results`, plus trial and comparison reports. The same TOML works for evaluation and serving. Choose the algorithm using validation results before evaluating its held-out test performance.

To optimize only STAN, append `stan` to the HPO command. To evaluate every generated configuration, run `./evaluator hpo-results` (PowerShell: `.\evaluator.exe hpo-results`). Directory evaluation writes `ALGORITHM.evaluation.json` beside each TOML. Use a fresh output directory when changing the enabled algorithms, because reruns leave older algorithms' files in place.

The standalone evaluator reports ranking metrics at 20. HPO uses the cutoff in its configured objective; the serving response length is controlled separately by `model.num_items_to_recommend`.

## Request recommendations

Both example configurations use port **8081**:

```sh
curl "http://localhost:8081/v1/recommend?item_id=13598&session_id=example&user_consent=true"
```

Use `curl.exe` in Windows PowerShell. The response is a JSON array of recommended item IDs. Reuse a session ID to build its click history.

Open [the internal page](http://localhost:8081/) to inspect the running service. [Prometheus metrics](http://localhost:8081/internal/prometheus) are available for monitoring. Stop serving with Ctrl+C.

Run from a writable directory: HPO and evaluation write reports, and serving creates `sessions.db`. If startup fails, check the dataset paths and whether port 8081 is already in use. See [Data preparation](Preparation.md) to use your own interactions.

## Example archive in the source repository

`assets/example` contains only `example.zip`, with `hpo.toml`, `train.txt`, `valid.txt`, and `test.txt`. Binary packages already include these extracted files. For a source checkout, extract the three datasets into `assets/example`, place the archived `hpo.toml` at `configs/hpo.toml`, and run commands from the repository root. This ZIP is the single source of the example configuration; packaging installs it at `configs/hpo.toml`. The archived configuration uses port **8081** and paths relative to that root.

## Publishing a new version

Increase the package version in `Cargo.toml` and the Serenade entry in `Cargo.lock`, then merge the pull request into `main`. The Binary release workflow compares the version before and after the push. A higher stable `major.minor.patch` version triggers tests and builds for all four platforms, followed by automatic creation of the version tag and GitHub Release with archives and checksums. No manual tag or release is needed. Unchanged or lower versions skip the builds on `main`.

Release branches (`release/**`) still run validation builds without publishing. Tag pushes and manual workflow runs remain available for retries. If a build fails, rerun the failed workflow jobs. If publication fails after creating a draft, inspect and remove the incomplete draft before rerunning; existing releases are never overwritten.
