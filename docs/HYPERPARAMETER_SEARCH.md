# TPE hyperparameter optimization

Edit [configs/hpo.toml](../configs/hpo.toml), then run from the repository or extracted release root:

```powershell
.\tpe_hyperparameter_optm.exe configs/hpo.toml
.\evaluator.exe hpo-results/stan.toml
.\serving.exe hpo-results/stan.toml
```

Linux/macOS use `./tpe_hyperparameter_optm`, `./evaluator`, and `./serving` without `.exe`. Locally built executables are in `target/release/`; binary packages put them at the extracted root. Repository users first extract `assets/example/example.zip` into `assets/example` as described in [Evaluation](EVALUATION.md). Packages already contain the data.

The default file enables all five algorithms and gives each 50 trials with seed 42. Shared `[hyperparam]` fields specify `training_data_path`, `validation_data_path`, `test_data_path`, `num_trials`, `seed`, `output_dir`, and `enable_business_logic`. Change paths and budget in this one file. The former `num_iterations` key is rejected; use `num_trials`. TPE loads only training and validation data. It selects each algorithm's complete best trial using validation MRR@20; ties keep the earliest trial. Test data never influences optimization.

Disable an algorithm by setting its `enabled` field to `false`; leave its section and ranges intact:

```toml
[hyperparam.algorithms.stan]
enabled = false
lambda_spw_range = [0.25, 2.05]
lambda_snh_range = [1.0, 8.0]
lambda_inh_range = [0.5, 4.0]
```

Every supported algorithm defaults to enabled if its section or flag is omitted. A run with no enabled algorithms is rejected. The CLI remains `CONFIG [ALGORITHM|all] [OUTPUT_DIRECTORY]`. To optimize just one enabled algorithm or override the directory:

```powershell
.\tpe_hyperparameter_optm.exe configs/hpo.toml stan
.\tpe_hyperparameter_optm.exe configs/hpo.toml vstan another-results
```

`vstan` is an alias for `vsknn_stan`. An explicitly selected disabled algorithm is rejected. `all` means all enabled algorithms. Legacy files without algorithm sections or `output_dir` retain their default `[model] algorithm` selection and existing shared range fields.

Algorithm sections support these common integer ranges: `m_most_recent_sessions_range` (default `[100, 2500]`), `neighborhood_size_k_range` (`[50, 1500]`), and `max_items_in_session_range` (`[1, 20]`). For example:

```toml
[hyperparam.algorithms.vmis]
enabled = true
m_most_recent_sessions_range = [100, 400]
neighborhood_size_k_range = [20, 80]
max_items_in_session_range = [2, 8]
idf_weighting_range = [0.25, 2.75]
```

| Algorithm | Additional searched ranges | Defaults |
| --- | --- | --- |
| `vmis` | `idf_weighting_range` | `[0, 5]` |
| `sknn` | `pop_boost_range` | `[0, 5]` |
| `vsknn` | `idf_weighting_range` | `[0, 5]` |
| `stan` | `lambda_spw_range`, `lambda_snh_range`, `lambda_inh_range` | `[0.1, 10]` each |
| `vsknn_stan` | STAN ranges plus `lambda_ipw_range`, `lambda_idf_range` | `[0.1, 10]` each |

Only applicable axes are optimized. Algorithm ranges override legacy shared `[hyperparam]` ranges, which override defaults. Legacy common names are `n_most_recent_sessions_range` and `last_items_in_session_range`. Integer intervals include both endpoints; floating intervals are half-open and retain decimal precision. Endpoints must be finite and ordered; integer axes require positive integral endpoints. Decay lambdas must be strictly positive; IDF weights and popularity boost allow zero. The effective k is capped to sampled m and recorded. Similarity, weighting modes, normalization and reminder options remain fixed at the starting algorithm's settings/defaults. Searched lambda components are enabled by the numeric search.

Each algorithm uses an independent seeded RNG and TPE estimators. Running it alone reproduces its all-mode trials on the same build and data. Training and validation load once; all algorithms share one index with the maximum required candidate budget. Model-dependent IDF statistics are refreshed safely without reloading training or rebuilding postings. Snapshots must have sufficient capacity and chronological metadata.

The output directory contains exactly one application configuration per enabled algorithm: `vmis.toml`, `sknn.toml`, `vsknn.toml`, `stan.toml`, `vsknn_stan.toml`. Each contains all selected model values, training and test paths, business filtering, and server settings, and is verified through `AppConfig` before reporting success. Pass the same file unchanged to evaluator or serving. Relative paths resolve from the process working directory; keep running from the repository/package root. `TRAINING_DATA` and `NUM_WORKERS` environment overrides still apply; avoid overriding training when reproducing a result.

Reports remain separate: `ALGORITHM-trials.jsonl` records every trial, `ALGORITHM.json` records the best validation MRR@20, exact model, seed, budget, search ranges and datasets, and `comparison.json` compares enabled algorithms' best validation results. Existing `save_records` and `out_path` fields still parse; reports always go into the output directory. Reruns replace matching files but leave older files for disabled algorithms untouched. Use a fresh output directory when changing the enabled set so directory evaluation covers only the current run.

Evaluate all optimized models on their configured held-out test set with:

```powershell
.\evaluator.exe hpo-results
```

This writes one `ALGORITHM.evaluation.json` alongside each TOML. See [Evaluation](EVALUATION.md) and [binary packages](BINARIES.md).
