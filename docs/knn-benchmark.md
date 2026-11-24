# kNN benchmark

## Run

From the repository root, using actual input paths:

```powershell
cargo run --release --example knn_benchmark -- TRAIN VALID all --iterations 3 --warmup 1 --output target/knn-benchmark/results.csv --raw target/knn-benchmark/iterations.csv --snapshot-dir target/knn-benchmark/snapshots
```

Inputs: raw interaction files with a `SessionId ItemId Time` header, separated by whitespace. Use a release build without competing CPU or disk workloads.

## Configuration

Select `all`, one algorithm, or a comma-separated list: `vmis`, `sknn`, `vsknn`, `stan`, `vsknn_stan`.

| Option | Default | Purpose |
|---|---|---|
| `--iterations N` | 3 | Measured iterations, each building a fresh index |
| `--warmup N` | 1 | Excluded warm-up iterations; zero allowed |
| `--passes N` | 3 | Prediction passes per iteration |
| `--cutoff N` | 20 | Recommendation and quality cutoff |
| `--output FILE` | `target/knn-benchmark/results.csv` | Summary CSV; overwritten |
| `--raw FILE` | disabled | Per-iteration CSV, including labeled warm-ups |
| `--snapshot-dir DIR` | disabled | Enable snapshot save/load measurements |
| `--config FILE` | none | Shared k, m, history, business logic and legacy VMIS IDF weight |
| `--dataset NAME TRAIN VALID` | none | Add a dataset; repeat for multiple datasets |
| `--quiet` | false | Suppress stderr progress and overview |

Scoring defaults come from `KnnConfig::for_algorithm`; effective parameters are recorded in CSV. The legacy positional `TRAIN VALID ALGORITHM SNAPSHOT_PREFIX CONFIG` form remains supported.

## Output

| Artifact | Contents |
|---|---|
| `results.csv` | One row per dataset, algorithm and configuration; alphabetically ordered columns |
| `results.metadata.json` | Schema version, command, git commit/dirty flag, build/runtime environment, input checksums, config and scaling manifests |
| Optional raw CSV | Per-iteration measurements with `phase` and `iteration`; request samples are not retained |
| stderr | Progress and per-algorithm quality/performance tables |

Empty CSV cells mean unavailable or unmeasured. Units appear in column names; quality values are fractions. Metadata status is `complete` only after a successful run. Completed rows survive a later failure.

Dataset columns include name, session/item scaling factors, seed when available, training sessions/interactions/items, evaluation sessions/events/prefixes, and predictions per iteration. Training counts use raw events before index pruning.

Distribution columns: `training_session_length_events_qX` and `training_item_popularity_events_qX`, with X = 0, 10, 50, 90, 95, 99, 100. Counts include repeat interactions. Quantiles use linear interpolation at `(n-1)*q`; q0/q100 are min/max.

## Quality metrics

Computed once per model outside timing. Column names include the effective cutoff, for example `MRR@20`.

| Metric | Definition |
|---|---|
| MRR | Reciprocal rank of the immediate next item |
| HitRate | Whether the immediate next item is recommended |
| NDCG | Binary relevance against the remaining suffix; existing gain weights: 1 at rank 1, then `1/log2(rank)`; ideal length `min(cutoff, suffix length)` |
| Precision | Distinct recommended suffix items divided by cutoff |
| Recall | Distinct recommended suffix items divided by suffix event count, including repeats |
| F1 | Harmonic mean of aggregated precision and recall |
| Coverage | Unique recommended items divided by training catalog size |
| Popularity | Mean recommended-item frequency normalized by the most frequent training item |

MRR, HitRate, NDCG, Precision, Recall and Popularity weight each evaluation prefix equally.

## Timing and memory

Summary measurements have `_median`, `_min`, `_max` suffixes across measured iterations; warm-ups are excluded. For even iteration counts, median averages the two middle values.

| Measurement stem | Includes |
|---|---|
| `index_build_ms` | Raw-file parsing, index construction, validation and model preparation |
| `snapshot_save_ms` | Serialization and buffered flush; no fsync |
| `snapshot_load_ms` | Deserialization, validation and model preparation; excludes destruction |
| `prediction_latency_p50_ms`, `p90_ms`, `p95_ms`, `p99_ms` | `predict_with_model`, including production allocations; excludes result destruction, ranking, quality evaluation, sample insertion and I/O |
| `index_owned_heap_estimate_bytes` | Capacity-based estimate of owned index vectors and hash-map entries |
| `snapshot_size_bytes` | Saved file size |

Latency percentiles use sorted samples at `round((n-1)*p)` within each iteration, converted to milliseconds, then aggregated across iterations. Samples are never pooled.

All algorithms use the same session-ID-sorted evaluation sessions, chronological items and ascending prefixes. History uses the last `max_history_items` items. Timing buffers are preallocated; preparation, statistics, checksums, sorting, logging and artifact writes occur outside measured sections.

Prediction is sequential and single-threaded. Iterations rebuild indexes in one process; OS caches are not flushed and CPUs are not pinned. Algorithm order is fixed.

Heap estimates exclude allocator overhead, stack metadata, input/evaluation buffers, prediction temporaries, RSS and peak build memory. Snapshot size is measured on disk.

## Synthetic scaling

### Train, validation and test splits

Extract the bundled example, then scale sessions and item catalogs:

```powershell
python -m zipfile -e assets/example/example.zip target/example-source
python scripts/scale_local_datasets.py --input-dir target/example-source --output target/scaled-example-100x-10x-items --factor 100 --catalog-factor 10
```

For local data, replace `--input-dir` with `D:\data`. Inputs must be named `train.txt`, `valid.txt`, `test.txt`. Outputs are `train.tsv`, `valid.tsv`, `test.tsv`, `manifest.json`. Existing output files are overwritten.

- `--factor`: multiplies sessions and interactions in every split.
- `--catalog-factor`: multiplies unique items; defaults to 1 and must divide `--factor`.
- Both factors must be positive integers. Keep the manifest beside the outputs for benchmark metadata.

| Factor | Catalog factor | Training sessions | Training interactions | Training items |
|---:|---:|---:|---:|---:|
| 100 | 10 | 2,375,300 | 11,360,100 | 180,090 |
| 1,000 | 100 | 23,753,000 | 113,601,000 | 1,800,900 |
| 10,000 | 1,000 | 237,530,000 | 1,136,010,000 | 18,009,000 |

Use `--factor 10000 --catalog-factor 1000` for the largest configuration. Validation starts at 330 sessions/1,451 events; test starts at 302 sessions/1,233 events. Both scale by the session factor.

The generator repeats whole sessions, preserves timestamps and session-length proportions, and consistently remaps disjoint catalogs across splits. Each source item's count per catalog copy scales by `factor / catalog-factor`. No new behavior is generated.

Start small to estimate disk and benchmark RAM needs. Generation streams expanded rows, then rereads outputs to verify distributions; source sessions and expanded item counts require memory. ID limits are checked.

Benchmark the generated files:

```powershell
cargo run --release --example knn_benchmark -- target/scaled-example-100x-10x-items/train.tsv target/scaled-example-100x-10x-items/valid.tsv all --iterations 3 --warmup 1
```

For test evaluation, replace `valid.tsv` with `test.tsv` and set a different `--output` path.

### Throughput query prefixes

```powershell
python scripts/scale_benchmark_sample.py --archive assets/example/example.zip --output target/scaled-sample-1000x --factor 1000 --seed 42
$env:BENCH_TRAINING_DATA = "target/scaled-sample-1000x/train.tsv"
$env:BENCH_QUERIES = "target/scaled-sample-1000x/queries.tsv"
cargo run --release --example prediction_benchmark
Remove-Item Env:BENCH_TRAINING_DATA, Env:BENCH_QUERIES
```

This script writes `train.tsv`, `queries.tsv`, `manifest.json`. It scales training sessions only, keeps the catalog fixed, shuffles sessions with the seed and rebases timestamps. Queries come from original validation/test prefixes; separate scaled evaluation splits are not produced. Very large factors can exceed its checked `u32` timestamp limit.

Both generators use Python's standard library and produce deterministic data for identical inputs/options.

## Validation

```powershell
cargo test --example knn_benchmark --offline
cargo fmt --all -- --check
```

Tests cover aggregation, missing measurements, latency percentiles, CSV consistency and all five algorithms, with and without snapshots.
