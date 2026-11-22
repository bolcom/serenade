# Prediction performance and startup behavior

Serving validates configuration and index references before opening HTTP listeners.
Explicit unreadable configuration, malformed supplied values, zero serving limits,
invalid ports, empty indexes, malformed training records and missing IDF entries
fail startup with context. Compatibility constructors remain available; the
`try_new` and `try_new_from_csv` constructors return errors. Public prediction
returns an empty heap for empty histories or zero k, m, or recommendation count.
IDF weighting zero remains supported.

Session read/decode failures recommend from the current item and skip the write.
Write failures also recommend from the current item. Records dated in the future
expire, so backward clock movement cannot underflow elapsed-time arithmetic.

## Optional blocking executor

Inline execution remains the default. Set `PREDICTION_EXECUTOR=blocking` to move
session storage and prediction to a dedicated pool. `PREDICTION_THREADS` defaults
to HTTP workers; `PREDICTION_QUEUE_CAPACITY` defaults to twice that count. Both
must be positive. Running work is bounded by the thread count, waiting work by
the queue capacity. Submission is nonblocking; saturation returns HTTP 503.
Accepted jobs finish when the pool shuts down. Prediction panics cancel their
response without killing a worker. The pool does not provide atomic updates
for concurrent requests to the same session.

## Reproduction

`cargo run --release --example prediction_benchmark` generates deterministic
training data with 20,000 sessions, 400,000 interactions and a 100,003-item ID
space. It prints index build time and CSV rows for throughput, p50, p95, p99 and
rejections. It compares inline execution against four blocking workers and eight
queued jobs, fresh buffers against reusable buffers, histories of 1 and 20 items,
and concurrency 1/2/4/8/16/32/64. Defaults are ten seconds of warmup, thirty seconds
of measurement and five repetitions per case. Runtime overrides:

```powershell
$env:BENCH_SECONDS = '30'
$env:BENCH_WARMUP = '10'
$env:BENCH_REPEATS = '5'
$env:BENCH_CONCURRENCY = '1,2,4,8,16,32,64'
cargo run --release --example prediction_benchmark
```

These are closed-loop prediction measurements, including dispatch and queue
waiting in blocking mode. They exclude HTTP, session storage, history construction
and business rules. Failed submissions are reported separately from successful
throughput. Results therefore cannot establish production HTTP latency or the
benefit of moving RocksDB operations off HTTP workers. Use representative data
and an external HTTP load generator with consent both enabled and disabled before
changing the default. The scratch API is additive; serving continues with fresh
buffers until representative measurements justify retaining worker-local memory.

The index builder now uses one 16-byte interaction record instead of parallel
interaction arrays, a permutation, sorted copies, unique-item copies and an unused
cloned session map. It processes contiguous item groups, preserving descending
timestamps and descending session IDs for equal timestamps. Prediction combines
unique counting and duplicate detection in one map; evaluation consumes slices
and iterators. Allocation and peak-memory profiling should be run separately
from latency measurements.

## Preliminary uniform synthetic measurements

Release build on Windows, Intel Core i9-10900KF (10 cores / 20 logical CPUs),
Rust 1.100.0. Each case used one second of warmup, two seconds of measurement
and three repetitions. The table reports the median throughput and the median
of each run's p95/p99, rather than pooled percentiles. These short synthetic
runs are directional; the first single-client cases overlapped a test run.
Raw per-run results are in [the results file](docs/prediction-benchmark-results.json).

| Clients | History | Mode | Scratch | Success/s | p95 us | p99 us | Reject/s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | inline | false | 285,380 | 3.9 | 5.0 | 0 |
| 1 | 1 | inline | true | 387,000 | 3.2 | 4.0 | 0 |
| 1 | 1 | blocking | false | 47,368 | 24.2 | 26.9 | 0 |
| 1 | 1 | blocking | true | 51,072 | 21.6 | 24.4 | 0 |
| 1 | 20 | inline | false | 117,665 | 10.9 | 13.0 | 0 |
| 1 | 20 | inline | true | 157,882 | 7.9 | 10.4 | 0 |
| 1 | 20 | blocking | false | 37,428 | 29.0 | 34.2 | 0 |
| 1 | 20 | blocking | true | 40,566 | 26.8 | 31.0 | 0 |
| 4 | 1 | inline | false | 1,111,614 | 4.8 | 5.5 | 0 |
| 4 | 1 | inline | true | 1,607,002 | 2.9 | 3.4 | 0 |
| 4 | 1 | blocking | false | 200,843 | 27.5 | 30.3 | 0 |
| 4 | 1 | blocking | true | 217,474 | 26.2 | 29.1 | 0 |
| 4 | 20 | inline | false | 527,904 | 10.1 | 12.0 | 0 |
| 4 | 20 | inline | true | 669,288 | 8.0 | 9.5 | 0 |
| 4 | 20 | blocking | false | 157,642 | 34.6 | 38.5 | 0 |
| 4 | 20 | blocking | true | 170,628 | 31.9 | 35.1 | 0 |
| 16 | 1 | inline | false | 3,428,501 | 5.3 | 5.7 | 0 |
| 16 | 1 | inline | true | 4,851,487 | 3.7 | 4.1 | 0 |
| 16 | 1 | blocking | false | 576,412 | 29.8 | 34.3 | 1,903,209 |
| 16 | 1 | blocking | true | 694,871 | 25.9 | 29.8 | 1,375,596 |
| 16 | 20 | inline | false | 1,631,582 | 11.5 | 13.2 | 0 |
| 16 | 20 | inline | true | 2,093,155 | 8.9 | 10.1 | 0 |
| 16 | 20 | blocking | false | 358,199 | 46.0 | 53.9 | 3,051,836 |
| 16 | 20 | blocking | true | 414,853 | 39.8 | 47.2 | 2,732,208 |

Dispatch overhead outweighed the benefit of offloading these short predictions.
At 16 clients, immediate retries also created heavy queue saturation; successful
latencies exclude rejected requests. Inline remains the default. Reusing scratch
buffers improved throughput in this dataset, but production adoption still needs
representative request and memory measurements. No before/after baseline or
allocation profile was measured, so these numbers do not quantify the index
builder or allocation changes against the previous implementation.
Avro session vectors now grow to the observed session IDs instead of reserving
150 million entries upfront.

## Original-code comparison

The earlier scratch-buffer comparisons do **not** measure improvement over the
original repository. Use the matched baseline results below to assess that claim.

## Sample-based scaling

The bundled `assets/example/example.zip` contains `train.txt`, `test.txt` and
`valid.txt`. The training sample has 113,601 interactions, 23,753 sessions and
18,009 distinct items. Session lengths have median 3, p90 9, p95 13, p99 22,
p99.5 28 and maximum 196. The most popular 1% of items account for 9.38% of
interactions. The empirical heavy tails replace the uniform assumptions above.

`scripts/scale_benchmark_sample.py` replicates complete sessions with new session
IDs, shuffling their order independently in each copy. It preserves exact item
popularity, session-length frequencies, repeated items, co-occurrence and relative
event times. The catalog stays fixed at the 18,009 observed items. This scales
volume and session count; it does not predict the full dataset's unseen catalog,
fit a theoretical power law, or invent new co-occurrence patterns. Repeated
sessions can make neighbors more homogeneous than in a genuinely larger sample.

The query file contains every nonterminal prefix from test and validation
sessions, shuffled deterministically. Its 2,052 histories have median length 2,
p99 81 and maximum 102. All current query items exist in the training catalog.
History limits select the most recent items; limit 0 keeps the complete prefix.
The output CSV's `history` field denotes that limit, not a fixed history length.

```powershell
python scripts/scale_benchmark_sample.py --factor 100
$env:BENCH_TRAINING_DATA = 'target/scaled-sample/train.tsv'
$env:BENCH_QUERIES = 'target/scaled-sample/queries.tsv'
$env:BENCH_HISTORY_LIMITS = '1,20,0'
$env:BENCH_SECONDS = '3'
$env:BENCH_WARMUP = '1'
$env:BENCH_REPEATS = '3'
$env:BENCH_CONCURRENCY = '1,4,16'
cargo run --release --example prediction_benchmark
```

The factor-100 dataset contains 11,360,100 interactions and 2,375,300 sessions
(272 MB TSV). Generation is seeded with 42. Data files remain under ignored
`target/`; [the manifest](docs/scaled-sample-manifest.json) records the source
archive hash and distribution statistics. Validate the scaling invariants with
`python -m unittest discover -s scripts -p 'test_scale*.py'`.

### Sample-based measurements

Release build on the same Windows/i9 machine as above. All 108 runs completed
with three seconds of measurement after one second of warmup, repeated three
times per case. The table shows medians across repetitions for **uncapped**
histories (history=0), including the sample's long tail. Each percentile is the
median of the per-run percentile, not a pooled percentile. All capped and uncapped
measurements are in [the raw results](docs/scaled-sample-benchmark-results.json).

| Clients | Mode | Scratch | Success/s | p95 us | p99 us | Reject/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | inline | false | 14,197 | 116.4 | 147.1 | 0 |
| 1 | inline | true | 16,443 | 102.2 | 127.7 | 0 |
| 1 | blocking | false | 10,476 | 145.7 | 178.3 | 0 |
| 1 | blocking | true | 11,643 | 134.3 | 166.8 | 0 |
| 4 | inline | false | 68,283 | 99.7 | 126.2 | 0 |
| 4 | inline | true | 79,221 | 89.8 | 114.8 | 0 |
| 4 | blocking | false | 49,675 | 124.3 | 151.7 | 0 |
| 4 | blocking | true | 56,683 | 111.6 | 137.0 | 0 |
| 16 | inline | false | 252,032 | 118.5 | 158.6 | 0 |
| 16 | inline | true | 295,283 | 105.4 | 143.3 | 0 |
| 16 | blocking | false | 57,739 | 281.3 | 331.4 | 4,776,809 |
| 16 | blocking | true | 66,120 | 256.6 | 321.4 | 4,687,794 |

The capped-run index built in 3.379 seconds. Observed process peak working set
was 993 MB (947 MiB), including startup; this is a sampled high-water reading,
not an allocation count or a before/after memory comparison.
Inline plus reusable scratch delivered higher throughput than fresh buffers in
this workload. Blocking dispatch still reduced throughput and increased latency;
at 16 clients immediate retries saturated the queue. Successful-request p95/p99
exclude rejections. Inline remains the serving default. These short prediction-only
runs exclude RocksDB and HTTP; they do not establish production request latency
or a speedup versus the repository before the code changes.

### Before restoring capacity hints (historical)

Baseline: Git commit `a142ab81dd49e774075f98094409eb103747eebe`. The original
`src` tree is compiled unchanged with the same current Cargo.toml/Cargo.lock,
compiler and release profile as the working tree. Both use the identical
[harness](examples/original_comparison.rs), default inline `predict` API, fresh
buffers, 11.36 million training interactions, 500-session sampling, k=100 and
21 recommendations. This measures the actual current default, without opt-in
scratch reuse or executor dispatch. Original source behavior, including bugs,
is retained; this is not a proof of recommendation-output equivalence.

Three alternating original/current process pairs; each case warms for one
second and measures for three seconds. Table entries are medians across runs.
History 0 uses complete held-out prefixes; history 20 caps them to 20 items.

| History cap | Clients | Original/s | Current/s | Throughput change | Original p95/p99 us | Current p95/p99 us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1 | 17,845 | 16,520 | -7.4% | 94.0 / 122.3 | 97.6 / 126.8 |
| 20 | 4 | 82,681 | 71,384 | -13.7% | 82.5 / 103.2 | 93.9 / 116.1 |
| 20 | 16 | 292,690 | 243,197 | -16.9% | 100.7 / 128.3 | 117.9 / 146.9 |
| 0 | 1 | 18,209 | 16,582 | -8.9% | 94.2 / 119.2 | 98.8 / 125.7 |
| 0 | 4 | 81,968 | 70,020 | -14.6% | 85.0 / 108.3 | 98.7 / 125.6 |
| 0 | 16 | 278,192 | 243,644 | -12.4% | 108.9 / 149.4 | 121.1 / 159.1 |

Median index startup: original 5.569 s; current 3.085 s.

The default prediction path regresses in this comparison: throughput is lower
and p95/p99 are higher. The earlier scratch-buffer benefit must not be described
as an overall speedup over the original. Prediction performance remains unresolved.
These short runs exclude HTTP and RocksDB, and the fixed observed item catalog
limits extrapolation to larger catalogs.

[Raw results](docs/original-comparison-results.json). Generate the dataset using `python scripts/scale_benchmark_sample.py --factor 100`,
then run `python scripts/benchmark_original.py` on Windows. This
builds both variants offline, alternates runs and saves their raw measurements.

## Capacity hints: latest matched measurements

The default prediction path now reserves the original capacities after clearing
its scratch buffers: 1,000 scored items, m candidate sessions and timestamps,
and the current history length for duplicate tracking. Reused buffers retain
their capacity. Scoring formulas and serving scratch ownership are unchanged.

The earlier regression tables above describe the version **before this fix**.
The same 36-run original-versus-current comparison now gives these median results:

| History cap | Clients | Original/s | Current/s | Throughput change | Original p95/p99 us | Current p95/p99 us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1 | 19,310 | 19,283 | -0.1% | 85.6 / 107.7 | 85.1 / 107.1 |
| 20 | 4 | 85,276 | 86,507 | +1.4% | 79.1 / 96.7 | 78.6 / 95.9 |
| 20 | 16 | 296,783 | 292,954 | -1.3% | 99.2 / 127.3 | 98.9 / 126.4 |
| 0 | 1 | 18,031 | 18,869 | +4.6% | 93.7 / 121.1 | 90.0 / 115.2 |
| 0 | 4 | 82,306 | 84,023 | +2.1% | 85.2 / 108.7 | 83.6 / 106.7 |
| 0 | 16 | 288,624 | 296,150 | +2.6% | 103.5 / 144.6 | 100.4 / 141.2 |

The measured 7-17% regression is recovered. The remaining small differences
should be treated as roughly equivalent prediction performance, given the short
runs. All cases pass the manual performance check (5% throughput and 10% tail
latency tolerances); these tolerances are explicit and configurable. This is not
a stable shared-runner CI test or evidence of a large speedup over the original.

Median index startup: original 5.107 s; current 3.057 s.

[Raw latest results](docs/capacity-comparison-results.json). Reproduce after
generating the scaled dataset:

```powershell
python scripts/benchmark_original.py --output docs/capacity-comparison-results.json
python scripts/check_prediction_performance.py docs/capacity-comparison-results.json
```

The build script fixes the baseline commit, cleans only Serenade artifacts
between builds, and verifies dependency paths to prevent shared Cargo artifacts
from selecting the wrong source tree. Dependencies use the same lockfile.

### Allocation probe and output stability

The single-threaded [allocation probe](examples/prediction_allocations.rs)
counts allocation/reallocation requests only during predictions over all 2,052
complete held-out histories; file parsing, index construction and output dumps
are excluded. Before the capacity fix: 48,169 calls; after: 12,312 calls
(74.4% fewer). The original code made 14,364 calls. Cumulative requested bytes
increase from 84.5 MB to 119.8 MB when restoring the original capacity hints
(original: 119.9 MB). These are allocation requests, not peak/live memory.

A full-score comparison did **not** establish output equivalence. The unmodified
original executable itself changes scores or candidate membership on 1,538 of
2,052 histories across two process runs. Neighbor selection at tied scores is
sensitive to hash-map iteration, and the scaled dataset contains many ties.
Comparing pre-fix and post-fix snapshots is therefore not a reliable equivalence
test. Existing unit tests pass, but deterministic tie handling remains unresolved;
worker-local scratch reuse has not been enabled in serving.

[Allocation counts and score diagnostics](docs/capacity-allocation-results.json).
To probe the current source independently of shared baseline artifacts:

```powershell
cargo clean --release -p serenade
$env:BENCH_TRAINING_DATA='target/scaled-sample/train.tsv'
$env:BENCH_QUERIES='target/scaled-sample/queries.tsv'
# Optional full-score dump (excluded from allocation counting):
$env:BENCH_SNAPSHOT='target/current-scores.tsv'
cargo run --release --offline --example prediction_allocations
```

### Training statistics cleanup (2026-11-22)

Local release build on Windows, Intel Core i9-10900KF; single measurements,
with warm filesystem caches. These measure the current implementation, not a
before/after speedup or production-scale guarantee.

| Workload | Runtime | Peak process working set |
| --- | ---: | ---: |
| Existing replicated sample: 11,360,100 raw events, 2,375,300 sessions, 18,009 items; build and save snapshot plus metadata | 4.73 s | 478.5 MiB |
| Collector alone: 10,000,000 completed session summaries representing 505,000,000 interactions | 0.439 s | 10.0 MiB |

The full build logged 1.484 s for CSV reading, chronological grouping and
statistics, plus 0.756 s for index preparation. Total time includes serialization
and process startup. Peak working set was sampled every 20 ms using Windows
`PeakWorkingSet64`; it includes the raw event array, index and runtime. The
collector test used 10 ms sampling and includes its test harness. Its input is
generated summaries, so it does **not** measure parsing 505 million events.
Percentile storage is bounded: two 4096-value buffers and two 1000-centroid
t-digests. Exact catalog counting separately needs a set proportional to catalog
size. The serving index gains aggregate metadata, without per-click timestamps.
No full production or billion-record corpus was benchmarked.

Reproduce with the existing `target/scaled-sample/train.tsv` in a build_index
configuration, and run the collector benchmark with:

```powershell
cargo test --release --lib benchmark_ten_million_session_summaries -- --ignored --nocapture
```

Validation: `cargo fmt --check`, `cargo clippy --all-targets`,
`cargo test --all-targets` (43 passed; one manual benchmark ignored), and
`cargo build --release --bins` passed. Clippy emits existing style warnings and
Cargo reports future incompatibilities in nalgebra/net2/snap. Each of VMIS-kNN,
SKNN, VS-KNN, STAN and V-STAN completed 931 predictions through `evaluator` on the
example test corpus and a two-trial TPE smoke test. Those randomized smoke tests
validate execution, not optimization quality. TPE now receives negative MRR
because its library minimizes the objective; displayed scores remain positive.
Unit tests separately cover raw p99.5 posting pruning, real Avro metadata,
fractional timestamp extrema, exact small quantiles and streaming tail accuracy.
