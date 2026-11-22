# Serenade — Scalable Session-Based Recommendations

[![Crates.io](https://img.shields.io/crates/v/serenade?style=flat-square)](https://crates.io/crates/serenade)
[![Crates.io](https://img.shields.io/crates/d/serenade?style=flat-square)](https://crates.io/crates/serenade)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/bolcom/serenade/LICENSE.md)
[![Build Status](https://github.com/bkersbergen/serenade/actions/workflows/release.yml/badge.svg)](https://github.com/bkersbergen/serenade/actions/workflows/release.yml)
[![API Docs](https://docs.rs/serenade/badge.svg)](https://docs.rs/serenade)
[![Contributors](https://img.shields.io/github/contributors/bolcom/serenade?style=flat-square)](https://github.com/bolcom/serenade/graphs/contributors)

[<img src="https://raw.githubusercontent.com/bolcom/serenade/main/docs/airlab.png" width="250" align="right">](https://icai.ai/airlab/)
[<img src="https://raw.githubusercontent.com/bolcom/serenade/main/docs/uva.png" width="250" align="right">](https://irlab.science.uva.nl/)

Serenade is an open-source, high-performance session-based recommendation system developed through academic research in collaboration between **[bol.com](https://www.bol.com/) and the [University of Amsterdam (AIRLab)](https://icai.ai/airlab/)**. The system was introduced in our [**SIGMOD 2022 paper, *Serenade: Low-Latency Session-Based Recommendation in e-Commerce at Scale***](https://doi.org/10.1145/3514221.3517901).

The research introduces two complementary contributions:

VMIS-kNN — Scalable recommendation algorithm. An efficient adaptation of the state-of-the-art VS-KNN algorithm. VMIS-kNN introduces specialized indexing data structures and an optimized neighbor retrieval algorithm using bounded candidate selection and early stopping. With fixed retrieval parameters, the computational complexity of candidate retrieval depends on the length of the evolving session rather than the size of the historical training corpus, enabling low-latency recommendations across billions of interactions.

Serenade — Production recommendation system. A scalable, stateful recommendation service built around VMIS-kNN, combining real-time session management, low-latency recommendation serving, and efficient resource utilization. Originally deployed at bol.com, Serenade demonstrates that high-quality session-based nearest-neighbor recommendations can operate at industrial scale with low latency and modest computational resources.

## Performance at scale

Our research demonstrates that session-based nearest-neighbor methods can achieve **millisecond-level prediction latency at industrial scale**, even in large e-commerce environments.

- **Under 1.7 ms p90 prediction latency** in microbenchmarks using datasets of up to 60 million interactions and 1.76 million distinct items.
- **1,000 recommendations per second** served using just two vCPUs.
- **Under 7 ms p90 end-to-end latency** in a Kubernetes production deployment, including network overhead, session retrieval from RocksDB, business-rule filtering, and response serialization.
- **2.3 billion historical interactions** processed, with 582 million retained after index pruning, covering approximately 6.5 million distinct products.
- **Approximately 11 GB of index memory** per serving instance at this production scale.

These results demonstrate how carefully designed algorithms and data structures can make session-based nearest-neighbor recommendation practical at massive scale.

*These performance figures originate from the original SIGMOD 2022 research and production deployment of VMIS-kNN. They do not represent benchmark results for all algorithms supported by the extended Serenade implementation.*

## Extending Serenade with additional scalable kNN algorithms

Building on the original VMIS-kNN research, **we have adapted four additional established session-based kNN algorithms to benefit from VMIS's scalable indexing and efficient neighbor retrieval approach**.

The extension includes SKNN, VS-KNN, STAN, and VSKNN-STAN:

**We adapt their neighbor retrieval to VMIS's bounded, recency-ordered index and early-stopping strategy**, while preserving their distinctive scoring methods. With fixed retrieval limits, candidate retrieval scales with the length of the evolving session rather than the size of the training corpus, enabling these kNN algorithms to benefit from VMIS's scalability.

## What Serenade offers

Serenade provides a complete workflow for developing, evaluating, and serving session-based nearest-neighbor recommendation models:

- **Scalable recommendation algorithms:** Five kNN approaches using the shared VMIS indexing and candidate retrieval approach.
- **Automated hyperparameter optimization:** Tree-structured Parzen Estimator (TPE) optimization across selected algorithms, with configurable objectives such as `mrr@20`, `ndcg@8`, and `hitrate@10`.
- **Offline evaluation:** Evaluate optimized models on held-out data using ranking and recommendation quality metrics.
- **Low-latency serving:** An HTTP recommendation service with session management backed by RocksDB.
- **Ready-to-run binaries:** Windows, Linux, and macOS executables, without requiring Rust or a local build environment.

The intended workflow is straightforward:

**TPE hyperparameter optimization → Offline evaluation → Real-time serving**

Each optimized algorithm produces a single configuration file that can be used directly by both the evaluator and serving applications.

## Quick start

### 1. Download Serenade

Download the appropriate package from [GitHub Releases](https://github.com/bkersbergen/serenade/releases).

| Platform | Package |
| --- | --- |
| Windows (x86_64) | ZIP |
| Linux (x86_64) | tar.gz |
| macOS (Intel) | tar.gz |
| macOS (Apple Silicon) | tar.gz |

Extract the archive and open a terminal in the extracted directory.

The package includes the three executables (`tpe_hyperparameter_optm`, `evaluator`, and `serving`), example configurations, and a small example dataset.

### 2. Optimize hyperparameters

Edit `configs/hpo.toml` to choose which algorithms to optimize, the search ranges, and the evaluation objective.

For example:

```toml
[hyperparam]
objective = "ndcg@8"
```

All five algorithms are enabled by default. Individual algorithms can be disabled in their configuration sections.

Run TPE optimization:

```bash
./tpe_hyperparameter_optm configs/hpo.toml
```

Serenade optimizes the enabled algorithms and generates a configuration containing the best hyperparameters for each algorithm, for example:

```text
hpo-results/
    vmis.toml
    sknn.toml
    vsknn.toml
    stan.toml
    vsknn_stan.toml
```

### 3. Evaluate an optimized algorithm

Evaluate STAN using its generated configuration:

```bash
./evaluator hpo-results/stan.toml
```

The evaluator reports recommendation quality on the held-out test dataset.

### 4. Serve recommendations

Start the HTTP recommendation service using **the same configuration file**:

```bash
./serving hpo-results/stan.toml
```

Request recommendations:

```bash
curl "http://localhost:8081/v1/recommend?item_id=13598&session_id=example&user_consent=true"
```

Serenade returns a JSON array of recommended item IDs.

On Windows, use the `.exe` applications with PowerShell, for example:

```powershell
.\tpe_hyperparameter_optm.exe configs/hpo.toml
.\evaluator.exe hpo-results/stan.toml
.\serving.exe hpo-results/stan.toml
```

Use `curl.exe` for the request in Windows PowerShell.

The example paths are relative to the extracted package directory. The bundled `configs/hpo.toml` configures port **8081**, which is used throughout these examples. The serving port is configured in the generated TOML file.

Open [http://localhost:8081/](http://localhost:8081/) in your browser to view Serenade's internal page while the service is running.

Serenade exposes [Prometheus metrics](http://localhost:8081/internal/prometheus) out-of-the-box for monitoring.

### 5. Retrieve recommendations using Python

Install the `requests` package with `python -m pip install requests`, then run:

```python
import requests
from requests.exceptions import HTTPError

try:
    myurl = 'http://localhost:8081/v1/recommend'
    params = dict(
        session_id='144',
        user_consent='true',
        item_id='13598',
    )
    response = requests.get(url=myurl, params=params, timeout=10)
    response.raise_for_status()
    # Access JSON content.
    json_response = response.json()
    print(json_response)
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')
```

Example response (the items and list length depend on the dataset, algorithm, configuration, and session history):

```json
[2835,10,12068,4313,3097,8028,3545,7812,17519,1164,17935,1277,13335,8655,14664,14556,6868,13509,9248,2498,11724]
```

The returned JSON array is a list of recommended item IDs.

For more details, see [Binary packages](docs/BINARIES.md), [Configuration](docs/CONFIG.md), and [Data preparation](docs/Preparation.md).

## Research and citation

If you use Serenade in academic work, please cite the original research:

**Barrie Kersbergen, Olivier Sprangers, and Sebastian Schelter.** *Serenade: Low-Latency Session-Based Recommendation in e-Commerce at Scale.* Proceedings of the 2022 International Conference on Management of Data (SIGMOD '22), pp. 150–159.

[Read the paper](https://doi.org/10.1145/3514221.3517901)

```bibtex
@inproceedings{Kersbergen2022SerenadeScale,
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
