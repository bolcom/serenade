# Serenade

Serenade is a Rust service for session-based recommendations: it predicts the next items from a visitor's recent clicks. One shared **VMISIndex** retrieves historical neighbors for all five scoring algorithms; evaluation, TPE optimization and HTTP serving use the same prediction and business-filtering code.

| Identifier | Algorithm |
| --- | --- |
| `vmis` (default) | Scalable VMIS-kNN, with recent-session retrieval and position-weighted scoring. |
| `sknn` | Session-based kNN using set similarity. |
| `vsknn` | Vector Multiplication Session-based kNN with position and optional IDF weights. |
| `stan` | Sequence and Time Aware Neighborhood with position and session-age decay. |
| `vsknn_stan` (`vstan` alias) | VSKNN_STAN hybrid combining vector, sequence, time and IDF weighting. |

Quick start from the repository root in PowerShell, with Rust 1.89 or newer and the native build dependencies installed:

```powershell
Expand-Archive -Path assets/example/example.zip -DestinationPath assets/example -Force
cargo build --release --bins
cargo run --release --offline --bin evaluator -- assets/example/workflows.toml all evaluation-results.json
cargo run --release --offline --bin tpe_hyperparameter_optm -- assets/example/workflows.toml all hpo-results
cargo run --release --offline --bin serving -- hpo-results/stan-serving.toml
```

The included configuration runs three seeded trials per algorithm. STAN has the highest validation MRR in this short example; use a larger budget before choosing a production model. In another PowerShell terminal:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8081/v1/recommend?item_id=13598&session_id=quickstart-1&user_consent=true'
```

The response is a JSON array of item IDs. Stop the service with Ctrl+C. The example uses port 8081; the application default remains 8080. `--offline` assumes dependencies were fetched by the build step.

Detailed guides: [offline evaluation and data format](docs/EVALUATION.md), [TPE optimization](docs/HYPERPARAMETER_SEARCH.md), [serving](docs/Prediction.md), [configuration](docs/CONFIG.md), [index preparation](docs/Preparation.md), and [algorithm compatibility](docs/knn-compatibility.md).

[Serenade paper](https://doi.org/10.1145/3514221.3517901) | [Apache 2.0 license](LICENSE.md)

Development uses normal feature branches in the canonical checkout (`D:\git\serenade`); do not create worktrees or shadow checkouts unless explicitly requested.
