# Historical example dataset evaluation

This report predates deterministic tie-breaking and the shared offline workflow. For current tested commands and reproducible measurements, see [offline evaluation](EVALUATION.md).

November 22, 2026; feature/start, release mode, Windows, Intel Core i9-10900KF, rustc 1.100.0.

Training: D:/git/serenade/assets/example/train.txt. Shared settings from example.toml: m=500, k=50, last 2 current events, legacy IDF weight=1, business filtering on. Evaluate at 20 recommendations (bundled serving config requests 21). Each algorithm uses its reference scoring defaults; VMIS uses legacy scoring. No hyperparameter tuning. CSV assigns non-adult/for-sale attributes, so business filtering introduces no catalog restrictions here. Session age uses the latest observed event timestamp.

Metrics follow Serenade: all remaining session items are targets, repository NDCG definition unchanged. Equal-score ordering was unspecified in this historical run; the current implementation uses deterministic ties.

## valid

| Model | MRR@20 | NDCG@20 | HitRate@20 | Coverage@20 | Popularity@20 | p50 us | p90 us | p99 us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vmis | 0.3131 | 0.3448 | 0.6111 | 0.2914 | 0.0549 | 2.700 | 6.300 | 14.200 |
| sknn | 0.3180 | 0.3522 | 0.6128 | 0.2749 | 0.0590 | 6.000 | 15.800 | 39.200 |
| vsknn | 0.3058 | 0.3330 | 0.6004 | 0.2844 | 0.0599 | 6.100 | 16.100 | 43.300 |
| stan | 0.3063 | 0.3371 | 0.6155 | 0.2724 | 0.0575 | 6.300 | 20.100 | 55.600 |
| vsknn_stan | 0.2992 | 0.3292 | 0.6120 | 0.2774 | 0.0557 | 6.200 | 17.500 | 47.600 |

## test

| Model | MRR@20 | NDCG@20 | HitRate@20 | Coverage@20 | Popularity@20 | p50 us | p90 us | p99 us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vmis | 0.3275 | 0.3539 | 0.6412 | 0.2761 | 0.0501 | 3.100 | 7.900 | 18.300 |
| sknn | 0.3487 | 0.3669 | 0.6423 | 0.2610 | 0.0545 | 6.800 | 22.200 | 45.900 |
| vsknn | 0.3082 | 0.3404 | 0.6391 | 0.2742 | 0.0556 | 6.500 | 19.100 | 42.200 |
| stan | 0.3302 | 0.3512 | 0.6273 | 0.2575 | 0.0552 | 6.800 | 23.200 | 51.800 |
| vsknn_stan | 0.3198 | 0.3409 | 0.6262 | 0.2619 | 0.0534 | 7.700 | 25.100 | 54.700 |

SKNN leads MRR and NDCG on both splits; test MRR is approximately 6.5% above VMIS. VMIS is fastest and has the highest coverage. Untuned results on these small splits do not establish significance or universal superiority.

Quality uses one pass over every prediction prefix: validation 1,121; test 931. Latency pools three passes (3,363 and 2,793 calls), including cold predictions, excluding heap-to-vector conversion and metric updates. Single-process latency is not serving throughput.

## Index costs

All models: estimated owned allocation 4,108,716 bytes (3.92 MiB). Includes vector/hash-table capacities and owned elements; excludes allocator overhead, hash control bytes and process/runtime memory. This is not RSS. Reference IDF replaces values in the same map, with no extra index.

Test-run diagnostics:

| Model | CSV build ms | Save ms | Snapshot load ms | Snapshot bytes |
| --- | ---: | ---: | ---: | ---: |
| vmis | 51.518 | 9.390 | 10.986 | 2257388 |
| sknn | 45.949 | 12.467 | 7.316 | 2257388 |
| vsknn | 43.572 | 7.824 | 6.423 | 2257388 |
| stan | 44.005 | 9.296 | 6.959 | 2257388 |
| vsknn_stan | 46.428 | 10.084 | 7.061 | 2258828 |

One build/save/load per run, subject to OS cache effects; no confidence intervals. [Full results CSV](example-knn-results.csv) also includes precision, recall and F1.

## Inputs

| File | Events | Sessions | Items | SHA256 |
| --- | ---: | ---: | ---: | --- |
| train.txt | 113601 | 23753 | 18009 | 3db80b44e517ae2ad74514b65cba32c1a05c41fe0889d1bd73a8a6889c47be99 |
| valid.txt | 1451 | 330 | 995 | 8c2a753d77a8e9d0352f9de7bd77c4bdeb715396a77e08b37f73d74ac993af84 |
| test.txt | 1233 | 302 | 862 | e495ce49c47a1af809c5264bed3c99460ee44aec2ce30468f3cf99aa65c0a181 |

## Reproduce

From the canonical repository, `D:\git\serenade`:

```powershell
cargo run --release --example knn_benchmark -- D:/git/serenade/assets/example/train.txt D:/git/serenade/assets/example/test.txt sknn target/sknn.vmis D:/git/serenade/assets/example/example.toml
```

Repeat for vmis, vsknn, stan, vsknn_stan; substitute valid.txt for test.txt. Optional final config argument supplies shared retrieval/history/business settings; algorithm selection supplies scoring defaults. Benchmark cutoff is always 20. Raw logs and snapshots are in ignored target/example-evaluation/.
