# Offline evaluation

Run commands from the repository root. The examples below were tested in PowerShell against the bundled corpus. Build release binaries as shown in the [quick start](../README.md) before using `--offline`.

The existing `assets/example/example.zip` contains `train.txt`, `valid.txt`, `test.txt` and a legacy `example.toml`. Extract it without replacing the checked-in `workflows.toml`:

```powershell
Expand-Archive -Path assets/example/example.zip -DestinationPath assets/example -Force
```

Use `assets/example/workflows.toml`, whose paths are relative to the repository root. The archive's legacy config uses paths relative to the extracted directory. Training has 113,601 events in 23,753 sessions; validation has 1,451 events in 330 sessions; test has 1,233 events in 302 sessions.

Training is a tab-separated file with a header and exactly three columns in this order. Validation and test accept whitespace-separated columns in the same order:

```text
SessionId	ItemId	Time
1	13598	1442011200
1	12068	1442011201
```

Session IDs are nonnegative integers fitting u32; item IDs fit u64; Time is finite nonnegative Unix seconds (fractional seconds accepted, historical timestamps rounded to u32). Events are sorted within sessions, preserving file order for equal times and repeated clicks. Training sessions are internally remapped to dense IDs. Keep entire sessions in separate, preferably chronological train/validation/test splits. Only the training file populates the historical index; holdout sessions are never added to it.

Dataset settings:

```toml
[data]
training_data_path = "assets/example/train.txt"

[hyperparam]
test_data_path = "assets/example/test.txt"
validation_data_path = "assets/example/valid.txt"
training_data_path = "assets/example/train.txt"
```

Evaluation reads `[data] training_data_path` and `[hyperparam] test_data_path`. HPO reads its own training and validation paths. Keep the two training paths equal for a consistent experiment.

Evaluate one algorithm or all five:

```powershell
cargo run --release --offline --bin evaluator -- assets/example/workflows.toml vstan evaluation-results-single.json
cargo run --release --offline --bin evaluator -- assets/example/workflows.toml all evaluation-results.json
```

The interface is `evaluator CONFIG [ALGORITHM|all] [RESULTS.json]`. Omitting the selection uses `[model] algorithm`, defaulting to VMIS; the default output is `evaluation-results.json`. `vstan` is accepted and reported as the existing canonical identifier `vsknn_stan`.

All-mode loads the training index and test data once, keeps the shared m/k/history/IDF and business settings, and uses the configured scoring options for the selected algorithm and existing algorithm defaults for the other four. To evaluate optimized settings directly, run:

```powershell
.\evaluator.exe hpo-results/stan.toml
.\evaluator.exe hpo-results
.\serving.exe hpo-results/stan.toml
```

The first command evaluates exactly the generated STAN model. The second loads every TOML in the directory and writes `ALGORITHM.evaluation.json` beside it, using each file's own model, training data and test data. Use a fresh HPO output directory to exclude older results. No edits or parameter copying are needed. The third serves the same file unchanged. Linux/macOS use `./evaluator` and `./serving` without `.exe`; local builds are in `target/release`. See [HPO](HYPERPARAMETER_SEARCH.md) for `configs/hpo.toml`.

Every session of length n contributes n-1 predictions. At each point only the observed prefix (limited by `max_items_in_session`) and its last timestamp are passed to the model. The cutoff is always 20, regardless of the serving recommendation count. Business filtering comes from `[logic] enable_business_logic`. Index statistics use training data only.

The existing metric definitions are preserved: MRR and HitRate assess the immediate next item; NDCG assesses remaining session items using Serenade's historical discount formula; Coverage is the fraction of distinct training items recommended across all points. These NDCG values should be compared within Serenade, rather than assumed interchangeable with other libraries' definitions.

A measured release run (931 prediction points, shared m=500, k=50, history=2, business filtering enabled):

| Algorithm | MRR@20 | NDCG@20 | HitRate@20 | Coverage@20 | Seconds | Mean us | P95 us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vmis | 0.327020 | 0.356097 | 0.641246 | 0.281248 | 0.012 | 5.3 | 13.3 |
| sknn | 0.344060 | 0.367913 | 0.643394 | 0.262591 | 0.016 | 9.7 | 25.5 |
| vsknn | 0.307648 | 0.340374 | 0.633727 | 0.266867 | 0.017 | 10.2 | 28.2 |
| stan | 0.329825 | 0.351682 | 0.630505 | 0.261314 | 0.018 | 10.4 | 30.2 |
| vsknn_stan | 0.319984 | 0.340876 | 0.629431 | 0.266311 | 0.019 | 11.3 | 32.3 |

Full metrics and scoring settings are in [evaluation.json](workflow-results/evaluation.json). Duration covers the evaluation loop and metric setup, excluding file loading and index preparation. Prediction latency covers scoring, filtering and recommendation sorting, excluding metrics, HTTP and session storage. Timings vary by machine and load; the tiny example is not a production performance benchmark.

Session traversal, score accumulation and ties are deterministic. Item-score ties prefer lower item IDs. Candidate retrieval and neighbor ties have stable ordering; fixed seeds control HPO. Accuracy metrics reproduce across repeated runs on the same build/data; durations do not. Stable ties can change the ordering of previously unspecified equal scores.

For a custom dataset, replace the three paths in a copy of the example configuration and retain the same file format. Do not use test results to select an algorithm or hyperparameters: tune on validation, freeze the choice, then report test performance.

A chronological `.vmis` snapshot can also be used as a training path. The tested example creates one with capacity 500:

```powershell
cargo run --release --offline --bin build_index -- assets/example/workflows.toml target/example.vmis
```

Use a snapshot built with capacity at least as large as the maximum m you will evaluate or search. Positional scoring and reference IDF require chronological raw-event metadata; incompatible older snapshots are rejected with a rebuild instruction.

The automated bundled-data check runs standalone and combined evaluation/HPO, checks seeded repeatability and fresh-index equivalence, reloads optimized configs, and starts each HTTP server using the unchanged generated files in a temporary package directory:

```powershell
python scripts/validate_workflows.py
```

It requires Python 3 and the release binaries, uses the actual archive, and cleans up its temporary servers and files.
