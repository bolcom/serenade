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

Quality evaluation reports all fifteen metrics at 20 in two console tables, reusing matching training
indexes and test data. Benchmarking preserves median aggregation, separate serial
latency and parallel throughput, with `--iterations 5`, `--warmup 1`, and optional
`--snapshot-dir DIR` for snapshot save/load measurements. HPO retains shared immutable
data, objective-only parallel evaluation, reusable worker buffers and sequential TPE trials.

For source builds, use `cargo run --release --bin NAME -- ARGS`. Extract
`assets/example/example.zip` into `workspace/` for the same example workflow.
See [configuration details](CONFIG.md) and [data preparation](Preparation.md).

The standard `cargo build --release` builds exactly `tpe_hyperparameter_optm`,
`evaluator`, and `serving`. Binary targets are declared explicitly in `Cargo.toml`;
automatic discovery in `src/bin/` is disabled.

To construct and persist an index without starting the server:

```sh
serving --model workspace/best-vmis.toml --config workspace/serenade.toml --build-index index.vmis --training-statistics training-stats.bin
```

`--training-statistics` is optional and exports raw corpus aggregates for legacy
Avro indexes. This mode uses the configured index capacity and exits before opening
the session database or HTTP listener. Load the snapshot with `serving --model
workspace/best-vmis.toml --config workspace/serenade.toml --index index.vmis`.
Evaluator benchmark snapshots remain available through `--snapshot-dir`.

To score predictions produced by another model:

```sh
evaluator --train workspace/train.txt --predictions predictions.txt
```

Each line contains comma-separated recommendation IDs, a semicolon, then
comma-separated next-item IDs: `10,20,30;20,40`. Empty recommendation lists and
trailing commas are accepted; next-item lists must contain at least one ID.
This mode prints the evaluation count and all fifteen metrics at 20 as CSV to stdout.
Malformed prediction rows fail with the filename and line number.
These commands replace the former `build_index` and `evaluate_file` executables.
