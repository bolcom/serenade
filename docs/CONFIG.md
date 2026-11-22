Application Configuration
===

All application configuration parameters can be set with a single config file in TOML format. Some parameters (currently only `training_data_path` and `num_workers`) can alternatively be set via environment variables.

Minimum Configuration
---

Serving needs a valid `data.training_data_path`; business filtering defaults to enabled. The user must have a training data file, with appropriate data formatting (see [dataset format](EVALUATION.md)). The training path can also be set through the `TRAINING_DATA` environment variable. HPO dataset paths are optional for serving.

Full Parameter List
---

| Config Section | Parameter | Type | Description | Required | Default | Sources |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | `training_data_path` | str | Path to training data file | :heavy_check_mark: | | Config file or environment variable |
| `server` | `num_workers` | int | Number of server worker threads | | Number of CPUs detected | Config file or environment variable |
| `server` | `host` | str | Host at which server should listen | | `"0.0.0.0"` | Config file |
| `server` | `port` | int | Port at which server should listen | | `8080` | Config file |
| `log` | `level` | str | Server logging level (not implemented) | | | Config file |
| `model` | `m_most_recent_sessions` | int | <!-- TODO --> | | `500` | Config file |
| `model` | `neighborhood_size_k` | int | Number of similar sessions to compare to current session | | `500` | Config file |
| `model` | `num_items_to_recommend` | int | Number of predictions the model should make | | `21` | Config file |
| `model` | `max_items_in_session` | int | Size of current session history to consider as model input | | `2` | Config file |

Example
---

Example configuration file specifying all of the above parameters:

```toml
[server]
num_workers = 4
host = "0.0.0.0"
port = 8080

[log]
level = "info"  # not implemented

[data]
training_data_path = "/path/to/training/data"

[model]
m_most_recent_sessions = 500
neighborhood_size_k = 500
num_items_to_recommend = 21
max_items_in_session = 2

[logic]
enable_business_logic = true
```

Algorithm selection
---

Existing configurations default to legacy VMIS. Add `algorithm` inside `[model]`: `vmis`, `sknn`, `vsknn`, `stan` or `vsknn_stan` (`vstan` alias). Retrieval limits, current-session limits, business logic and serving routes remain shared.

```toml
[model]
algorithm = "vsknn_stan"
m_most_recent_sessions = 500
neighborhood_size_k = 100
max_items_in_session = 10
num_items_to_recommend = 20

[model.knn]
lambda_spw = 1.02
lambda_snh = 5
lambda_inh = 2.05
lambda_ipw = 1.02
lambda_idf = 5
```

Scoring options under `[model.knn]`:

| Option | Applies to | Default / values |
| --- | --- | --- |
| similarity | SKNN / VS-KNN | SKNN jaccard (cosine, binary, tanimoto); VS-KNN vec (cosine) |
| weighting, weighting_score | VS-KNN | div; same, linear, log, quadratic also supported |
| weighting_time | VS-KNN | false; rounded-day decay when enabled |
| idf_weighting | VS-KNN | 0; floating score IDF multiplier |
| idf_weighting_session | VS-KNN | false; replaces session position weights with reference IDF |
| remind | SKNN / VS-KNN / STAN variants | SKNN false (true boosts last three clicks); others true (false excludes current items) |
| normalize | all reference models | true for SKNN/VS-KNN, false for STAN variants |
| pop_boost | SKNN | 0; positive neighbor popularity boost |
| lambda_spw | STAN variants | 1.02; current-position exponential decay |
| lambda_snh | STAN variants | 5 days; session-age exponential decay |
| lambda_inh | STAN variants | 2.05; historical-position distance decay |
| lambda_ipw | VSKNN_STAN | 1.02; latest matching current-position decay |
| lambda_idf | VSKNN_STAN | 5; additive score IDF multiplier |

Set a lambda to `"none"` (legacy unquoted `none` also parses) to disable its component. Decay lambdas must be positive finite numbers. Legacy `[model] idf_weighting` now accepts floating-point values, default 1; integer configurations remain compatible. This VMIS weight is separate from reference IDF. TPE searches only relevant axes, with floating-point scoring parameters and per-algorithm lambda ranges; see [HPO](HYPERPARAMETER_SEARCH.md).

Build a chronological snapshot using the tested [example command](EVALUATION.md). Use the same loader for serving, evaluation and HPO.

Set `data.training_data_path` to OUTPUT.vmis for serving/evaluation. Hyperparameter data paths can also reference snapshots; build with at least the maximum candidate budget searched. Serving does not require a hyperparameter section. Evaluation needs its test path; HPO needs training and validation paths. Avro indexes require raw training aggregate metadata (`training-stats.bin`; see [Preparation](Preparation.md)) and support set-based scoring, but positional scoring and reference event-frequency IDF require rebuilding from CSV. See [compatibility and migration](knn-compatibility.md) and [measured results](knn-benchmark.md), including explicitly unsupported optional Python extensions.


Unified HPO and application configurations
---

[configs/hpo.toml](../configs/hpo.toml) defines shared training, validation and test paths, TPE iterations/seed, `output_dir`, and all five enabled algorithms under `[hyperparam.algorithms.NAME]`. Set `enabled = false` to skip an algorithm; all-disabled runs fail. Algorithm-specific `PARAMETER_range = [low, high]` settings override shared legacy ranges; see [search ranges and defaults](HYPERPARAMETER_SEARCH.md).

HPO files may omit `[data]` and `[model]`; TPE uses `[hyperparam] training_data_path`. The exported `hpo-results/NAME.toml` supplies `[data]`, the exact selected `[model]` and `[model.knn]`, `[logic]`, `[server]`, and optional `[hyperparam]` dataset fields. Both evaluator and serving load this same file through `AppConfig`. Search ranges, algorithm sections, seed and trial budget are optional for these applications. Evaluation uses `[hyperparam] test_data_path`; serving ignores holdout paths. Legacy application configs and CLI overrides continue to work.

```powershell
.\tpe_hyperparameter_optm.exe configs/hpo.toml
.\evaluator.exe hpo-results/stan.toml
.\serving.exe hpo-results/stan.toml
```

Run from the repository/extracted package root so relative data paths resolve correctly. Linux/macOS use `./` and omit `.exe`.
