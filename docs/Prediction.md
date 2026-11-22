# Serving recommendations

All five algorithms use the existing server, VMISIndex, RocksDB session store and `/v1/recommend` endpoint. The same configuration loader handles serving, evaluation and TPE output. VMIS remains the default for older configurations that omit `algorithm`.

Set `[model] algorithm` to `vmis`, `sknn`, `vsknn`, `stan` or `vsknn_stan`. The existing alias `vstan` is also accepted. Example scoring settings:

```toml
[model]
algorithm = "vstan"
m_most_recent_sessions = 500
neighborhood_size_k = 50
max_items_in_session = 2
num_items_to_recommend = 20

[model.knn]
lambda_spw = 1.02
lambda_snh = 5
lambda_inh = 2.05
lambda_ipw = 1.02
lambda_idf = 5
```

`lambda_snh` is in days; other positional lambdas are dimensionless. Floating-point values are preserved. Use `"none"` to disable a lambda. See [all scoring settings](CONFIG.md). VMIS uses `[model] idf_weighting`; VS-KNN uses `[model.knn] idf_weighting`; the hybrid uses `lambda_idf`. These settings have the same interpretation offline and online.

Prepare data, build and optimize using the [quick start](../README.md) or [TPE guide](HYPERPARAMETER_SEARCH.md). Start with a generated configuration from the repository root:

```powershell
cargo run --release --offline --bin serving -- hpo-results/stan-serving.toml
```

The generated file contains server settings, the HPO training path, business filtering and the complete best model. No HPO section is required for serving. Alternatively, replace both model sections in an existing config with `ALGORITHM-model.toml`, keep its `[data]` training corpus consistent with HPO, and copy the HPO business-filtering setting into `[logic]`.

The example configuration binds port 8081 with four workers. The application default remains port 8080. The process stores consenting users' recent clicks in `./sessions.db`, relative to the working directory. Run one server per database and stop it with Ctrl+C before starting another model.

From a second PowerShell terminal, submit a click:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8081/v1/recommend?item_id=13598&session_id=quickstart-1&user_consent=true'
```

The HTTP response is a JSON array of recommended integer item IDs; PowerShell displays each ID on its own line. A tested STAN example returned 20 items beginning with `[12068, 10, 7812, 2498, 17935]`. Reuse the session ID and change the item ID to submit subsequent clicks. With `user_consent=false`, prediction uses only the submitted click and does not persist session history. Unknown items can return an empty array.

STAN-family online predictions use the current wall-clock timestamp, while offline predictions use the observed event timestamp. Consequently, serving the old example corpus is a request/response smoke check, not a reproduction of historical offline scores; sufficiently old sessions can underflow time-decay scores. Use recent production training data for online time-aware ranking.

`python scripts/validate_workflows.py` (after the release build) has tested optimized serving for all five algorithms, each on an isolated port and session database, including save/reload and fresh-index validation-score agreement.
