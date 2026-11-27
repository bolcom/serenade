# Serenade — Scalable Session-Based Recommendations

[![Crates.io](https://img.shields.io/crates/v/serenade?style=flat-square)](https://crates.io/crates/serenade)
[![Crates.io](https://img.shields.io/crates/d/serenade?style=flat-square)](https://crates.io/crates/serenade)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE.md)
[![Build Status](https://github.com/bolcom/serenade/actions/workflows/release.yml/badge.svg)](https://github.com/bolcom/serenade/actions/workflows/release.yml)
[![API Docs](https://docs.rs/serenade/badge.svg)](https://docs.rs/serenade)
[![Contributors](https://img.shields.io/github/contributors/bolcom/serenade?style=flat-square)](https://github.com/bolcom/serenade/graphs/contributors)

[<img src="https://raw.githubusercontent.com/bolcom/serenade/main/docs/airlab.png" width="250" align="right">](https://icai.ai/airlab/)
[<img src="https://raw.githubusercontent.com/bolcom/serenade/main/docs/uva.png" width="250" align="right">](https://irlab.science.uva.nl/)

Serenade is an open-source, high-performance session-based recommendation system developed through academic research in collaboration between **[bol.com](https://www.bol.com/) and the [University of Amsterdam (AIRLab)](https://icai.ai/airlab/)**. The system was introduced in our [**SIGMOD 2022 paper, *Serenade: Low-Latency Session-Based Recommendation in e-Commerce at Scale***](https://doi.org/10.1145/3514221.3517901).

The research introduces two complementary contributions:

- **VMIS-kNN — Scalable recommendation algorithm.** An efficient adaptation of the state-of-the-art VS-KNN algorithm. VMIS-kNN introduces specialized indexing data structures and an optimized neighbor retrieval algorithm using bounded candidate selection and early stopping. With fixed retrieval parameters, the computational complexity of candidate retrieval depends on the length of the evolving session rather than the size of the historical training corpus.
- **Serenade — Production recommendation system.** A scalable, stateful recommendation service built around VMIS-kNN, combining low-latency session management, low-latency recommendation serving, and efficient resource utilization. Originally deployed at bol.com, Serenade demonstrates that high-quality session-based nearest-neighbor recommendations can operate at industrial scale with low latency and modest computational resources.

## Performance at scale

Our research demonstrates that session-based nearest-neighbor methods can achieve **millisecond-level prediction latency at industrial scale**, even in large e-commerce environments.

- **Under 1.7 ms p90 prediction latency** in microbenchmarks using datasets of up to 60 million interactions and 1.76 million distinct items.
- **1,000 recommendations per second** served using just two vCPUs.
- **Under 7 ms p90 end-to-end latency** in a Kubernetes production deployment, including network overhead, session retrieval from RocksDB, business-rule filtering, and response serialization.
- **2.3 billion historical interactions** processed, with 582 million retained after index pruning, covering approximately 6.5 million distinct products.
- **Approximately 11 GB of index memory** per serving instance at this production scale.

*These performance figures originate from the original SIGMOD 2022 research and production deployment of VMIS-kNN. They do not represent benchmark results for all algorithms supported by the extended Serenade implementation.*

## Supported recommendation algorithms

Building on the original VMIS-kNN research, **we extend Serenade with scalable adaptations of six established session-based kNN algorithms**, preserving their distinctive scoring methods while leveraging VMIS-kNN's efficient neighbor retrieval architecture.

| Algorithm | Paper | Publication |
|---|---|---|
| **VMIS-kNN** | [Serenade: Low-Latency Session-Based Recommendation in e-Commerce at Scale](https://doi.org/10.1145/3514221.3517901) | SIGMOD '22 |
| **SKNN** | [When Recurrent Neural Networks meet the Neighborhood for Session-Based Recommendation](https://doi.org/10.1145/3109859.3109872) | RecSys '17 |
| **S-SKNN (SeqContextKNN)** | [Evaluation of session-based recommendation algorithms](https://doi.org/10.1007/s11257-018-9209-6) | UMUAI '18 |
| **WSKNN** | [WSKNN: Weighted session-based KNN algorithm for next-item recommendation](https://doi.org/10.21105/joss.05639) | JOSS '23 |
| **VS-KNN** | [Evaluation of session-based recommendation algorithms](https://doi.org/10.1007/s11257-018-9209-6) | UMUAI '18 |
| **STAN** | [Sequence and Time Aware Neighborhood for Session-based Recommendations: STAN](https://doi.org/10.1145/3331184.3331322) | SIGIR '19 |
| **VSKNN-STAN (VSTAN)** | [Empirical analysis of session-based recommendation algorithms: A comparison of neural and non-neural approaches](https://doi.org/10.1007/s11257-020-09277-1) | UMUAI '21 |

As described in [Illoominate – Data Shapley Values for Debugging Neighborhood-based Recommender Systems)](https://doi.org/10.1145/3828670), session-based kNN recommendation can be decomposed into two functions: neighbor retrieval ($f_{ret}$) and item scoring ($f_{pred}$).

**Our contribution is to introduce scalable adaptations of established session-based kNN algorithms by leveraging the VMISIndex for neighbor retrieval, while preserving their distinctive similarity and item scoring functions.** Through bounded, recency-ordered indexing and early stopping, neighbor retrieval scales with the length of the active session rather than the size of the historical session corpus, given fixed retrieval limits.

## What Serenade offers

Serenade provides a complete workflow for developing, evaluating, and serving session-based nearest-neighbor recommendation models:

- **Scalable recommendation algorithms:** Seven kNN approaches using the shared VMIS indexing and candidate retrieval architecture.
- **Automated hyperparameter optimization:** [Tree-structured Parzen Estimator (TPE) optimization](https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) across selected algorithms, with configurable objectives such as `mrr@20`, `ndcg@8`, and `hitrate@10`.
- **Offline evaluation and benchmarking:** Compare optimized models on held-out data using 15 recommendation quality metrics, and optionally measure prediction latency, throughput, index construction time and memory usage.
- **Low-latency serving:** An HTTP recommendation service with session management backed by RocksDB.
- **Ready-to-run binaries:** Windows, Linux, and macOS executables, without requiring Rust or a local build environment.
- **Rust Cargo crates:** Direct, in-process recommendations using the same algorithms, index and model configurations as the standalone applications.

## Getting started

## Ready-to-run binaries: optimize, evaluate, serve

**TPE hyperparameter optimization → Offline evaluation (with optional benchmarking) → low-latency serving**

Each optimized algorithm produces a model configuration file that can be used directly by both the evaluator and the serving application.

#### Download and installation

Download and extract a [binary release](https://github.com/bolcom/serenade/releases). The release packages contain all three executables and an example workspace.

The following commands use Windows PowerShell. On Linux and macOS, use `./` instead of `.\` and omit the `.exe` extension.

Alternatively, install the executables from crates.io:

```sh
cargo install serenade
```

Installing from source requires a Rust toolchain and native RocksDB build dependencies. The standard `cargo build --release` builds the same three executable targets: `tpe_hyperparameter_optm`, `evaluator`, and `serving`.

See [binary packages](docs/BINARIES.md), [configuration](docs/CONFIG.md), and [data preparation](docs/Preparation.md) for details on preparing your own datasets and configuring the applications.

#### Step 1 — Optimize

Optimize the hyperparameters of the selected recommendation algorithms using training and validation data:

```powershell
.\tpe_hyperparameter_optm.exe --config workspace/hpo.toml
```

The optimizer writes a `best-*.toml` model configuration for each selected algorithm. The spelling `tpe_hyperparameter_optm` is intentional.

#### Step 2 — Evaluate and benchmark

Evaluate one or more optimized models on held-out test data. Add `--benchmark` to measure computational performance in addition to recommendation quality.

```powershell
.\evaluator.exe --model workspace/best-vmis.toml --model workspace/best-sknn.toml --model workspace/best-ssknn.toml --model workspace/best-wsknn.toml --model workspace/best-vsknn.toml --model workspace/best-stan.toml --model workspace/best-vsknn_stan.toml --benchmark
```

The evaluator reports results for each model configuration and writes them to a CSV file. See [supported evaluation metrics](#supported-evaluation-metrics) for the available recommendation quality metrics.

#### Step 3 — Serve

Start the recommendation service with the selected model configuration:

```powershell
.\serving.exe --model workspace/best-vmis.toml --config workspace/serenade.toml
```

The model configuration determines recommendation behavior; `serenade.toml` configures the serving infrastructure.

#### Step 4 — Request recommendations

Once the service is running, request recommendations over HTTP.

**Using cURL:**

```sh
curl "http://localhost:8081/v1/recommend?item_id=13598&session_id=example&user_consent=true"
```

Use `curl.exe` in Windows PowerShell.

**Using Python:**

Install the `requests` package with `python -m pip install requests`, then run:

```python
import requests

response = requests.get(
    "http://localhost:8081/v1/recommend",
    params={
        "session_id": "144",
        "user_consent": "true",
        "item_id": "13598",
    },
    timeout=10,
)
response.raise_for_status()
print(response.json())
```

Example response (shortened):

```json
[2835,10,12068,4313,3097,8028,3545,7812,17519,1164]
```

The returned JSON array is a ranked list of recommended item IDs. The actual items and list length depend on the dataset, algorithm, configuration, and session history.

The service also exposes an [internal page](http://localhost:8081/) and [Prometheus metrics](http://localhost:8081/internal/prometheus).


## Advanced usage

### Index export and persistence

Construct and persist an index without starting the HTTP server:

```sh
serving --model workspace/best-vmis.toml --config workspace/serenade.toml --build-index index.vmis --training-statistics training-stats.bin
```

`--training-statistics` is optional and exports raw corpus aggregates for legacy Avro indexes. This mode uses the configured index capacity and exits before opening the session database or HTTP listener.

Load the persisted index when starting the service:

```sh
serving --model workspace/best-vmis.toml --config workspace/serenade.toml --index index.vmis
```

Evaluator benchmark snapshots are available through `--snapshot-dir`.

### Evaluation of external predictions

Use Serenade's evaluation metrics to score predictions produced by another recommendation model:

```sh
evaluator --train workspace/train.txt --predictions predictions.txt
```

Each line contains comma-separated recommendation IDs, a semicolon, then comma-separated next-item IDs: `10,20,30;20,40`. Empty recommendation lists and trailing commas are accepted; next-item lists must contain at least one ID.

This mode prints the evaluation count and all fifteen metrics at 20 as CSV to stdout. Malformed prediction rows fail with the filename and line number.

These two advanced modes replace the former `build_index` and `evaluate_file` executables.


## Rust Cargo crates

Serenade can also be embedded directly in Rust applications without running an HTTP service. All seven algorithms are available through the library and use the same implementation and model configurations as the standalone applications.

Install the library with minimal dependencies:

```sh
cargo add serenade --no-default-features
```

Load an index and reuse an HPO-generated model configuration to generate recommendations:

```rust
use serenade::{Recommender, VMISIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index = VMISIndex::load("train.txt")?;
    let vmis = Recommender::from_config(&index, "best-vmis.toml")?;
    let session = &[13598, 2835, 12068];
    let recommendations = vmis.predict(session, 20)?;
    println!("{recommendations:?}");
    Ok(())
}
```

Multiple recommenders can share a single immutable index without copying sessions or postings. The library also supports programmatic model configuration, explicit index capacity, persisted indexes and concurrent predictions. Minimal-feature builds exclude Actix, RocksDB, Prometheus and TPE.

See the **[Rust API documentation on docs.rs](https://docs.rs/serenade)** for installation, examples, configuration, index sharing, concurrency and persistence. A complete runnable example is available with `cargo run --no-default-features --example library -- train.txt best-vmis.toml` from the repository.

## Supported evaluation metrics

The evaluator reports all metrics below at K=20; K is the maximum number of recommendations considered. Relevant items are the remaining items in the held-out session, except MRR and HitRate, which assess the immediate next item.

| Metric | What it measures |
| --- | --- |
| MRR@K | Measures how highly the immediate next item ranks, using its reciprocal rank and assigning zero when it is absent. |
| NDCG@K | Measures how well relevant items rank near the top, using Serenade's historical discount formula and normalizing against an ideal ranking. |
| HitRate@K | Measures the fraction of prediction cases where the immediate next item appears among the top K recommendations. |
| Precision@K | Measures how many of the K recommendation slots contain relevant items, rewarding accurate recommendations. |
| Recall@K | Measures the fraction of remaining session items retrieved among the top K recommendations. |
| F1@K | Balances precision and recall through the harmonic mean of their evaluation-wide averages. |
| MAP@K | Measures precision at each relevant rank, rewarding lists that retrieve multiple relevant items early. |
| Popularity@K | Measures the average training frequency of recommended items, normalized by the most frequent item's count, to reveal popularity bias. |
| Coverage@K | Measures the fraction of distinct training items that appear in recommendations across all prediction cases. |
| Gini@K | Measures how unequally recommendation exposure is distributed across the catalog; lower values indicate more even exposure. |
| ShannonEntropy@K | Measures the spread of recommendation exposure using RecBole's entropy convention: natural-log entropy divided by the number of distinct exposed items. |
| LongTailShare@K | Measures how often less-popular items are recommended as a share of K slots, complementing average popularity with a percentage based on the least-frequent 10% of training items. |
| IntraListDiversity@K | Measures how different items within each recommendation list are, using cosine distance between their training-session interaction vectors. |
| EmptyResultRate@K | Measures the fraction of prediction cases that return no recommendations, highlighting failures to produce results. |
| FillRate@K | Measures the fraction of K recommendation slots filled with distinct items, highlighting incomplete result lists. |

See [metric definitions](docs/METRICS.md) for formulas, edge cases and supported HPO objectives.

## Research and citation

If you use Serenade in academic work, please cite the original research:

**Barrie Kersbergen, Olivier Sprangers, and Sebastian Schelter.** *Serenade: Low-Latency Session-Based Recommendation in e-Commerce at Scale.* Proceedings of the 2022 International Conference on Management of Data (SIGMOD '22), pp. 150–159.

[Read the paper](https://doi.org/10.1145/3514221.3517901)

```bibtex
@inproceedings{Kersbergen2022SerenadeScaleVMIS,
  author    = {Kersbergen, Barrie and Sprangers, Olivier and Schelter, Sebastian},
  title     = {Serenade - Low-Latency Session-Based Recommendation in e-Commerce at Scale},
  booktitle = {Proceedings of the 2022 International Conference on Management of Data},
  year      = {2022},
  pages     = {150--159},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3514221.3517901}
}
```

## Contributing

Contributions to Serenade are welcome, including improvements to recommendation algorithms, scalability, evaluation, and serving capabilities.

## License

Serenade is licensed under the [Apache License 2.0](LICENSE.md).
