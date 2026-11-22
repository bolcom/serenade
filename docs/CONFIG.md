# Configuration

Start with `configs/hpo.toml` in the extracted package. HPO generates a complete TOML for each enabled algorithm; pass that file unchanged to `evaluator` and `serving`.

## Hyperparameter optimization

Edit the existing `[hyperparam]` section:

```toml
[hyperparam]
training_data_path = "assets/example/train.txt"
validation_data_path = "assets/example/valid.txt"
test_data_path = "assets/example/test.txt"
objective = "ndcg@8"
num_trials = 50
seed = 42
output_dir = "hpo-results"
enable_business_logic = true
```

Paths resolve from the terminal's working directory. Supported objectives are `mrr@k`, `ndcg@k`, and `hitrate@k`, where `k` is a positive integer; the default is `mrr@20`. MRR and HitRate assess the next item; NDCG assesses remaining session items using Serenade's discount formula. Compare NDCG results within Serenade because definitions differ across libraries.

Algorithms are `vmis`, `sknn`, `vsknn`, `stan`, and `vsknn_stan`. Set `enabled = false` in an algorithm's existing section to skip it. At least one must remain enabled.

Each `[hyperparam.algorithms.NAME]` section accepts these common ranges:

| Range | Meaning | Default |
| --- | --- | --- |
| `m_most_recent_sessions_range` | Candidate session budget | `[100, 2500]` |
| `neighborhood_size_k_range` | Number of nearest neighbors | `[50, 1500]` |
| `max_items_in_session_range` | Recent clicks used from the current session | `[1, 20]` |

Additional ranges depend on the algorithm:

| Algorithm | Ranges | Defaults |
| --- | --- | --- |
| `vmis`, `vsknn` | `idf_weighting_range` | `[0.0, 5.0]` |
| `sknn` | `pop_boost_range` | `[0.0, 5.0]` |
| `stan` | `lambda_spw_range`, `lambda_snh_range`, `lambda_inh_range` | `[0.1, 10.0]` each |
| `vsknn_stan` | STAN ranges plus `lambda_ipw_range`, `lambda_idf_range` | `[0.1, 10.0]` each |

For example, edit the existing STAN section:

```toml
[hyperparam.algorithms.stan]
enabled = true
m_most_recent_sessions_range = [100, 500]
neighborhood_size_k_range = [20, 100]
max_items_in_session_range = [2, 10]
lambda_spw_range = [0.1, 10.0]
lambda_snh_range = [0.1, 10.0]
lambda_inh_range = [0.1, 10.0]
```

Integer ranges include both endpoints and require positive integers. Decay lambdas must be positive; IDF weights and popularity boost allow zero. Keep lower bounds at or below upper bounds. Use `num_trials` for the trial budget; `num_iterations` is no longer accepted.

## Evaluation and serving

Generated files contain the selected `[model]` and algorithm-specific `[model.knn]` values. Relevant settings are:

| Section / setting | Purpose |
| --- | --- |
| `[data] training_data_path` | Historical training interactions |
| `[hyperparam] test_data_path` | Held-out interactions for evaluation |
| `[model] algorithm` | Selected algorithm |
| `[model] m_most_recent_sessions` | Candidate session budget |
| `[model] neighborhood_size_k` | Nearest-neighbor count |
| `[model] max_items_in_session` | Current session history limit |
| `[model] num_items_to_recommend` | Serving response length (default 21) |
| `[logic] enable_business_logic` | Apply business-rule filtering |
| `[server] host` | Listening interface |
| `[server] port` | Listening port |
| `[server] num_workers` | HTTP worker count |

Both bundled example configurations set:

```toml
[server]
host = "0.0.0.0"
port = 8081
num_workers = 4
```

Serving needs the training path; evaluation also needs the test path. Keep HPO's and the generated model's training paths consistent. Environment variables `TRAINING_DATA` and `NUM_WORKERS` can override the training path and worker count; avoid overriding training when reproducing an optimized result.

See [Using the binaries](BINARIES.md) and [Data preparation](Preparation.md).
