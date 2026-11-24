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


## Workflow

Download and extract a [release](https://github.com/bolcom/serenade/releases).
Windows packages contain `.exe` files; Linux and macOS use the same names without
`.exe` (prefix commands with `./`). The spelling `tpe_hyperparameter_optm` is intentional.

**1. Optimize**

```powershell
.\tpe_hyperparameter_optm.exe --config workspace/hpo.toml
```

**2. Evaluate and optionally benchmark**

```powershell
.\evaluator.exe --model workspace/best-vmis.toml --model workspace/best-wsknn.toml --benchmark
```

**3. Serve**

```powershell
.\serving.exe --model workspace/best-vmis.toml --config workspace/serenade.toml
```

The release includes a ready-to-run `workspace/` with `train.txt`, `validation.txt`,
`test.txt`, `hpo.toml`, and `serenade.toml`. A workspace is just a directory convention.
HPO writes `best-<algorithm>.toml` for every enabled algorithm beside `hpo.toml`.
These portable files contain only model parameters. HPO uses training and validation
sessions; it never opens the test set. Trial diagnostics are optional (`save_records`).

`hpo.toml` owns search spaces, objective/cutoff, trials, seed, evaluation workers and
independent `[data] index_capacity`. `serenade.toml` owns host, port, worker count,
business filters and independent index capacity. It contains no optimized parameters.
The example server listens on port 8081.

Dataset defaults follow the configuration: HPO finds training and validation data
beside `hpo.toml`; evaluation finds training and test data beside the supplied models;
serving finds training data beside its model, even when its serving configuration
is in another directory. Config-file paths resolve relative to their containing
file; CLI paths resolve relative to the current working directory. Absolute paths
and quoted paths with spaces work, including on separate storage volumes.

Evaluation accepts repeated `--model` arguments, including multiple versions of
one algorithm. It writes `evaluation.csv`, or `benchmark.csv` with `--benchmark`,
beside the models. Models from different directories require `--dataset-dir DIR`
(which also sets the default output directory), or both `--train FILE --test FILE`
and `--output FILE`. It never chooses the first model's dataset in that case.

Advanced overrides: HPO accepts `--train`, `--validation`, `--output` (directory),
`--algorithm`, and `--index-capacity`. Evaluation accepts `--dataset-dir`, `--train`,
`--test`, `--output` (CSV), `--index-capacity`, `--workers`, and `--business-logic true|false`.
Serving accepts `--train` or `--index`; `[data] training_data_path` and `index_path`
in `serenade.toml` provide configuration-relative overrides. None requires copying
production data. All three use independent index capacity 2500 by default and reject
incompatible retrieval limits. Persisted indexes must already support that capacity.

Quality evaluation preserves all eight metrics at 20, reusing matching training
indexes and test data. Benchmarking preserves median aggregation, separate serial
latency and parallel throughput, with `--iterations 5`, `--warmup 1`, and optional
`--snapshot-dir DIR` for snapshot save/load measurements. HPO retains shared immutable
data, objective-only parallel evaluation, reusable worker buffers and sequential TPE trials.

For source builds, use `cargo run --release --bin NAME -- ARGS`. Extract
`assets/example/example.zip` into `workspace/` for the same example workflow.
See [configuration details](CONFIG.md) and [data preparation](Preparation.md).
