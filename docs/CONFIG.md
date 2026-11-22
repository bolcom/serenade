Application Configuration
===

All application configuration parameters can be set with a single config file in TOML format. Some parameters (currently only `training_data_path` and `num_workers`) can alternatively be set via environment variables.

Minimum Configuration
---

The only parameter value that is required and has no default value to fall back on is `training_data_path`. The user must have a training data file, with appropriate data formatting (see [DATA](DATA.md)). Since this parameter can be set with an environment variable, the config file can be left out entirely. In this case, the default values specified below are used for all remaining parameters.

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

Set a lambda to `none` to disable its component. Decay lambdas must be positive finite numbers. Legacy `[model] idf_weighting` remains integer/default 1; it is separate from reference IDF. TPE uses the selected algorithm and scoring config; the existing IDF search axis maps to VS-KNN idf_weighting or VSKNN_STAN lambda_idf. It has no effect for SKNN/STAN. Other scoring settings remain configurable and fixed during TPE searches. TPE is the only supported optimizer.

Build/load a chronological snapshot:

```powershell
cargo run --release --bin build_index -- CONFIG OUTPUT.vmis
```

Set `data.training_data_path` to OUTPUT.vmis for serving/evaluation. Hyperparameter data paths can also reference snapshots; build with at least the maximum candidate budget searched. Existing required hyperparameter config fields still apply. Avro indexes require raw training aggregate metadata (`training-stats.bin`; see [Preparation](Preparation.md)) and support set-based scoring, but positional scoring and reference event-frequency IDF require rebuilding from CSV. See [compatibility and migration](knn-compatibility.md) and [measured results](knn-benchmark.md), including explicitly unsupported optional Python extensions.
