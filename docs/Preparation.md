Preparation
===

### Data format

Serenade needs training data for generating recommendations and test data to evaluate the quality of your recommendations.
It supports binary AVRO files and CSV text files as input. 
If your data has less than 100 million records it is advised to use CSV files to get started with Serenade. A machine with 16GB of RAM memory is advised.

A CSV file must be TAB (\t) separated and look something like this:
```
SessionId       ItemId  Time
10036   14957   1592337718.0
10036   14713   1592337765.0
10036   2625    1592338184.0
10037   7267    1591979344.0
10037   13892   1591979380.0
10037   7267    1591979504.0
```

* **SessionId** A unique session identifier (unsigned 32 bit integers supported)
* **ItemId** an identifier for a product or item that a visitor interacted with. (unsigned 64 bit integers supported)
* **Time** epoch in seconds. (32 bit float and unsigned 32 bit integers supported)


Configure Application
---

All application configuration parameters can be set with a single config file in TOML format. Some parameters (currently only `training_data_path` and `num_workers`) can alternatively be set via environment variables.

### Minimum Configuration

The only parameter value that is required and has no default value to fall back on is `training_data_path`. The user must have a training data file, with appropriate data formatting (see above). Since this parameter can be set with an environment variable, the config file can be left out entirely. In this case, the default values specified in [CONFIG](CONFIG.md) are used for all remaining parameters.

See [CONFIG](CONFIG.md) for further configuration details.

### Training statistics and index compatibility

CSV statistics are collected while building chronological sessions, before
posting pruning or deduplication, without rereading the training file. Event
counts include **all raw interactions**, including repeated visits. Duration
is `max(timestamp) - min(timestamp)`; single-event sessions have zero duration.
Fractional duration seconds are rounded to the nearest second for the existing
integer statistics fields; timestamp extrema retain fractional seconds.
Counts, distinct sessions/items, timestamp extrema and p100 are exact.
Empty in-memory inputs yield zero counts/percentiles and Unix epoch timestamps;
an empty training CSV is rejected because it cannot produce a serving index.

Duration and raw event-count quantiles are p05, p25, p50, p75, p90, p95, p99,
p99.5 and p100. Up to 4096 sessions use exact linear interpolation at
`q * (n - 1)`, rounded to the nearest integer. Larger datasets use two streaming
t-digests (1000 centroids, batches of 4096 sessions), keeping percentile storage
bounded independently of event/session count. Approximation can vary slightly
with input order; synthetic tests check <=1% value error on smooth distributions
and <=0.1 percentage-point rank error on discrete distributions, including the
p99.5 pruning boundary. These are validation tolerances, not formal guarantees
for every distribution. p100 always uses the observed maximum.

CSV construction uses one event array and the chronology sort required by the
algorithms; it does not sort a second event corpus for statistics. Exact distinct
item counting uses a temporary set proportional to catalog size. The standalone
`determine_training_data_statistics` helper supports arbitrary unsorted slices
using one summary per session and a distinct-item set; the CSV path avoids that
session map by consuming its existing groups.

The posting-pruning threshold is the **raw event-count p99.5**. The quantity
compared to it remains the number of distinct historical items, preserving the
existing posting-filter rule. Raw event count, stored sequence length and
distinct item count are different quantities. Rebuilt indexes may have different
postings and recommendations because this threshold is now data-dependent.

`build_index CONFIG OUTPUT.vmis [training-stats.bin]` can persist the same
aggregates during an existing CSV index build. Snapshots now use
`SERENADE-VMIS-003`; older snapshots require rebuilding from raw data. No
per-click timestamps are added to the serving index.

Avro's session-level timestamp and unique item list cannot recover raw counts,
minimum event time or durations. Place the compact, versioned
`training-stats.bin` beside `itemindex/` and `sessionindex/`, generated from the
**same original, unpruned corpus** using the optional build_index argument (or
`TrainingDataStats::write_metadata` during upstream preparation). Publish the
metadata with its matching Avro artifacts. It describes the raw corpus, so its
counts can exceed the pruned serving index. Absent, invalid or unsupported
metadata causes a clear load error; no values are inferred or fabricated.
If original events are unavailable, accurate duration statistics cannot be
reconstructed and the old index cannot be migrated. Positional algorithms and
reference IDF still require chronological CSV/snapshots, as described in
[compatibility](knn-compatibility.md).
