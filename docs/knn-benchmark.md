# kNN benchmark

Release smoke comparison: Windows, Intel Core i9-10900KF, rustc 1.100.0, November 21 2026. Baseline is main `a142ab81dd49e774075f98094409eb103747eebe`, with the console build repair, a read-only heap diagnostic and identical benchmark harness. Baseline training/scoring code is unchanged.

Inputs: local `illoominate/data/rsc15_100k/processed/{train,valid}.csv` (1,679,460 and 521,568 bytes). SHA256 train `8d0fb36473f58192d3ee411f2cd1be2facaf1910ee2e943257053fa56c8791cd`, valid `6dda944b5700994b0a7bd9d5d40194d6f91f1c5d18281d7b8185c50c80f914b5`. Data is not bundled.

All runs: m=500, k=100, last 10 events, 20 recommendations, business filters off, reference scoring defaults. Evaluate all 12,455 prefixes in session-ID order; quality uses first pass, latency pools three passes (37,365 predictions), including cold predictions. Prediction timing excludes metrics and heap-to-vector conversion. Age uses last observed validation timestamp. Training normalizers use identical raw data. Repository metrics are unchanged, including the existing NDCG formula; check definitions before comparing other evaluators.

| Model | MRR@20 | NDCG@20 | HitRate@20 | Coverage@20 | Popularity@20 | p50 us | p90 us | p99 us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main VMIS | .1777 | .2222 | .3920 | .9460 | .1372 | 9.7 | 18.5 | 31.8 |
| Updated VMIS | .1775 | .2221 | .3943 | .9460 | .1372 | 11.1 | 20.5 | 34.1 |
| SKNN | .1656 | .2120 | .3755 | .9346 | .1689 | 34.4 | 85.9 | 134.7 |
| VS-KNN | .1771 | .2185 | .3926 | .9335 | .1687 | 35.9 | 98.5 | 151.3 |
| STAN | .1793 | .2185 | .3921 | .9356 | .1589 | 37.7 | 100.0 | 152.3 |
| VSKNN_STAN | .1825 | .2163 | .3866 | .9467 | .1230 | 39.9 | 102.4 | 155.6 |

| Model | CSV build ms | Save ms | Snapshot load ms | Owned heap estimate bytes | Snapshot bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main VMIS | 20.649 | N/A | N/A | 1,073,636 | N/A |
| Updated VMIS | 23.449 | 4.503 | 3.901 | 1,387,376 | 861,340 |
| SKNN | 21.774 | 4.261 | 2.139 | 1,387,376 | 861,340 |
| VS-KNN | 20.706 | 4.322 | 2.114 | 1,387,376 | 861,340 |
| STAN | 21.531 | 4.444 | 2.059 | 1,387,376 | 861,340 |
| VSKNN_STAN | 22.350 | 4.397 | 2.391 | 1,387,376 | 861,516 |

Heap estimates sum vector/hash-table capacities and owned elements; exclude allocator overhead, hash control bytes and runtime memory. They are structural allocation estimates, not RSS measurements. Preserving events increases this estimate by 29.2%. All variants share one index shape; reference IDF replaces values rather than adding another map. Large legacy reader preallocations were removed.

Reader fixes deliberately retain final events/sessions and update repeat timestamps. These can change VMIS quality; equal-score output order is also unspecified. A preliminary updated VMIS run returned .1777 MRR/.2222 NDCG, showing the small tie-sensitive variation. Fixed-index regression tests verify scoring invariance with historical repetitions. Inference deduplication adds modest VMIS latency (p99 +2.3 us here); new scorers allocate per-request sets/maps for reference semantics. These single-process measurements are not production throughput or large-index scaling claims. Hyperparameters were not optimized.

Reproduce:

```powershell
cargo run --release --example knn_benchmark -- TRAIN VALID vmis target/vmis.vmis
# repeat with sknn, vsknn, stan, vsknn_stan
cargo test --all-targets
```

The example emits quality, prediction percentiles, creation/loading time and size diagnostics. Main snapshot persistence is N/A because it lacks this writer. Snapshot loading uses a warm OS cache and includes deserialization/model validation. Large-scale multi-worker benchmarking remains deployment-specific.

An end-to-end release smoke check built a VSKNN_STAN snapshot with build_index, then evaluated it through the existing evaluator CLI: 12,455 predictions, MRR .1825. The evaluator stopwatch previously passed percentage values directly to a 0..1 quantile API, reporting the maximum for every percentile. This is corrected with a deterministic percentile test; benchmark tables above use independently sorted raw nanosecond samples.
