# Scalable session kNN validation

Serenade is the recommendation system; VMIS-kNN is its original algorithm; VMISIndex is the shared infrastructure. S-SKNN and WSKNN are scalable adaptations of established algorithms, not new algorithm definitions. All seven algorithms share the same recent-candidate index and persistence format.

## Definitions, references and scope

S-SKNN uses SKNN neighbor selection, then multiplies each neighbor's item contribution by a function of the distance from the latest current-session item shared with that neighbor. The [2018 paper](https://doi.org/10.1007/s11257-018-9209-6) ([open manuscript](https://arxiv.org/abs/1803.09587), equation 6) describes a linear position factor. The original SeqContextKNN implementation instead defaults to inverse distance, and supplies five weighting functions. Serenade follows that executable reference, including its SKNN similarity formulas and reminder behavior; `weighting = "div"` is the default. Thus default scores are implementation-compatible rather than a literal reproduction of equation 6.

WSKNN follows [Moli?ski, JOSS 2023](https://doi.org/10.21105/joss.05639), [documentation](https://wsknn.readthedocs.io/en/latest/) and the original source. Current-session position weights determine neighbor similarity (weighted intersection divided by the number of unique current items). A second weight, based on the latest shared current position, multiplies all items from a neighbor. Repeated historical events contribute repeatedly. Supported current weighting: linear, log, quadratic; item ranking: linear, inverse (`div`/`inv`), log, quadratic. Upstream's linear current weight `(position + 1) / length`, with positions starting at one, is preserved, including its off-by-one behavior. The reference does not assign separate historical-position weights; existing STAN variants retain that behavior.

Only recent sampling is supported. No event metadata beyond item order and session timestamps is required. WSKNN reuses the chronological sequences already stored for STAN; old legacy snapshots without these sequences must be rebuilt. S-SKNN also works with the set representation. Current snapshots save and load all seven configurations without a format change. Configuration defaults, parameter domains and limitations are in [CONFIG.md](CONFIG.md).

Optional algorithms were not added. The available t-KNN references did not establish an unambiguous original session-based specification and validated reference distinct from existing temporal methods. The [2018 idf-knn music paper](https://web-ainf.aau.at/pub/jannach/files/Workshop_RecSys_Challenge_2018.pdf) defines cosine similarity over TF-IDF playlist vectors, including playlist-length smoothing; this is playlist completion, not a validated chronological session method. It must not be conflated with VS-KNN's existing IDF weighting options. Neither satisfied the inclusion criteria sufficiently to justify another named algorithm.

## Reference provenance and reproducibility

Python methods are executed unmodified from pinned source files through an AST loader; imports for unrelated training, dataframe and array operations are excluded. The validation adapter supplies in-memory maps and calls the original candidate sampling, similarity and item scoring methods. No Python dependencies enter production Rust.

* SeqContextKNN: [session-rec commit 5dcd583](https://github.com/rn5l/session-rec/blob/5dcd583cbd8d44703a5248b9a308945f24b91390/backup/algorithms/aware_backup/knn_aware/scknn.py). No repository license file was found in that pinned tree. The source is downloaded for local validation only, not redistributed; Rust implements the formulas independently.
* WSKNN: [commit bd06b56](https://github.com/nokaut/wsknn/tree/bd06b567a15df256503ed3aa0de607712ab912e3), BSD-3-Clause, copyright Szymon Moli?ski, Sales Intelligence 2023. The [complete notice](third-party/wsknn-LICENSE.md) is included, also in release packages.

Run from the repository root (Python standard library only):

```text
python tests/reference/download_extensions.py
python tests/reference/generate_extensions.py
python tests/reference/evaluate_extensions.py target/example/train.txt target/example/valid.txt target/reference-predictions.json
cargo run --offline --release --example reference_metrics -- target/example/train.txt target/reference-predictions.json
cargo test --offline --all-targets
python -m unittest discover -s scripts -p test_package_release.py
```

The checked-in 68 additional scoring cases cover all supported new weighting combinations, repeated items, unknown items and short queries, supplementing the existing algorithm fixtures. Rust tests compare item scores and neighbors to these fixtures. Snapshot round trips and reused/fresh index equivalence cover all seven algorithms; bounded-candidate tests compare recency cutoffs against exhaustive retrieval including timestamp ties. HPO tests exercise all algorithms. The extracted release smoke check additionally ran one HPO trial per algorithm, evaluation, generated config reloads, live HTTP serving, seed reproducibility, disabled algorithms and legacy configuration.

## Accuracy protocol

The bundled example training split has 23,753 sessions, 113,601 events and 18,009 items. Its validation split has 330 sessions and 1,451 events, yielding 1,121 chronological nonempty prefix/remaining-suffix pairs. Both implementations use recent m=500, k=100, last 10 current events, cutoff 20, default algorithm settings, no business filters and final current-item exclusion. Rust's existing EvaluationReporter evaluates both prediction streams, with the same training popularity and catalog. F1 is the existing F1@20 reporter field (F1Score@20).

The Python adapter recomputes candidate unions from the same ten-event window as Rust, rather than retaining earlier items from a stateful predict call. Python ties use ascending item ID; Rust uses its existing deterministic ordering. Python WSKNN sorts historical timestamp lists lexicographically (primarily session start), whereas VMISIndex samples by final session timestamp. Timestamp-cutoff ties can select different sessions. Unknown current items and repeated events retain each implementation's native representation. These differences explain small aggregate discrepancies; scoring fixtures isolate formula correctness. No hyperparameters were tuned on this validation comparison, and this is not a claim of statistical superiority.

### SSKNN accuracy

| Metric | Python | Rust | Relative difference |
|---|---:|---:|---:|
| MRR@20 | 0.315910 | 0.316350 | +0.14% |
| NDCG@20 | 0.376801 | 0.375396 | -0.37% |
| HitRate@20 | 0.669938 | 0.662801 | -1.07% |
| Precision@20 | 0.092774 | 0.090500 | -2.45% |
| Recall@20 | 0.452798 | 0.452078 | -0.16% |
| F1@20 | 0.153996 | 0.150809 | -2.07% |
| Coverage@20 | 0.274418 | 0.274807 | +0.14% |
| Popularity@20 | 0.059614 | 0.060216 | +1.01% |

### WSKNN accuracy

| Metric | Python | Rust | Relative difference |
|---|---:|---:|---:|
| MRR@20 | 0.289383 | 0.286596 | -0.96% |
| NDCG@20 | 0.359028 | 0.350071 | -2.49% |
| HitRate@20 | 0.656557 | 0.648528 | -1.22% |
| Precision@20 | 0.091079 | 0.088671 | -2.64% |
| Recall@20 | 0.444562 | 0.443096 | -0.33% |
| F1@20 | 0.151185 | 0.147770 | -2.26% |
| Coverage@20 | 0.260592 | 0.261425 | +0.32% |
| Popularity@20 | 0.061346 | 0.062305 | +1.56% |

Across the eight metrics, relative differences are within 2.65%. The largest changes are precision and F1 (about 2?3%), which are sensitive to recommendation-cutoff tie membership. MRR and recall stay within 1% for both methods. Together with exact local score fixtures, this supports comparable behavior without changing scalable retrieval to force identical reference rankings.

## Scalability and pruning

Candidate collection is unchanged: reverse-current-item traversal, at most m postings per unique item, a timestamp heap and a map bounded by m sessions. Each item posting list is ordered newest first. Once full, an unseen candidate no newer than the oldest retained timestamp cannot improve the recent sample; all subsequent postings in that list are at least as old, so traversal may stop. Truncating each item list to m cannot remove a strictly newer member of the global top-m union: that item alone already supplies m candidates at least as recent. Equal-time membership follows legacy traversal and is deterministic for fixed input. This argument depends only on timestamps, not similarity or item-score weights, and therefore remains valid for both new algorithms.

Similarity is recomputed on the bounded candidates before exact top-k neighbor selection. No VMIS-specific similarity upper bound is applied to S-SKNN or WSKNN. Item scores are computed only from selected neighbors. Query cost is bounded by current history, m and neighbor sequence lengths; it does not scan all historical sessions. WSKNN adds no historical vectors or index fields; S-SKNN reuses SKNN scoring. The owned-heap estimate measures index allocations, not process RSS or transient query allocations.

Benchmarks use the existing consolidated CSV implementation: release build, m=500, k=100, history=10, cutoff=20, all default configurations, one complete warm-up iteration and three independent measured index-build iterations. Reported values are medians of per-iteration values; latency percentiles are first computed within each iteration, not pooled. Three prediction passes run per iteration. Build/save/load timing and byte estimates come from that same runner.

Host: Windows x86_64, Intel Core i9-10900KF 3.70 GHz (10 cores / 20 logical processors), Rust 1.89.0. The existing scaled local dataset contains 2,375,300 sessions, 11,360,100 events and 180,090 unique items; validation has 112,100 prefixes. It replicates the example sessions 100 times with 10 catalog namespaces, retaining timestamps. It stresses posting/index size but does not establish generalization to a genuinely different dataset. The host was not CPU-isolated; a large SKNN scaled iteration outlier is visible in the raw measurements. Medians limit that effect, but these local figures are not controlled cross-machine performance claims.

Reproduce after extracting the example zip into target/example (and copying its train/valid/test files into target/example-source):

```text
python scripts/scale_local_datasets.py --input-dir target/example-source --output target/scaled-local-100x-10x-items --factor 100 --catalog-factor 10
cargo run --offline --release --example knn_benchmark -- target/example/train.txt target/example/valid.txt all --iterations 3 --warmup 1 --output target/extension-benchmark.csv --raw target/extension-benchmark-raw.csv --snapshot-dir target/extension-snapshots --dataset scaled-100x-10x-items target/scaled-local-100x-10x-items/train.tsv target/scaled-local-100x-10x-items/valid.tsv --quiet
```

### example performance

All times in milliseconds; heap in MiB.

| Algorithm | Build | Heap | p50 | p90 | p99 | Save | Load |
|---|---:|---:|---:|---:|---:|---:|---:|
| vmis | 41.9698 | 3.7944 | 0.0040 | 0.0119 | 0.0255 | 7.9718 | 15.7065 |
| sknn | 40.0580 | 3.7944 | 0.0079 | 0.0272 | 0.0589 | 16.8517 | 12.4975 |
| ssknn | 38.3103 | 3.7944 | 0.0076 | 0.0261 | 0.0561 | 8.5913 | 12.2582 |
| wsknn | 38.8955 | 3.7944 | 0.0056 | 0.0189 | 0.0475 | 15.7729 | 10.6508 |
| vsknn | 39.5500 | 3.7944 | 0.0081 | 0.0276 | 0.0655 | 7.9937 | 11.6574 |
| stan | 38.2773 | 3.7944 | 0.0079 | 0.0290 | 0.0681 | 6.0684 | 13.8826 |
| vsknn_stan | 40.4328 | 3.7944 | 0.0087 | 0.0312 | 0.0733 | 5.8877 | 13.6133 |

### scaled-100x-10x-items performance

All times in milliseconds; heap in MiB.

| Algorithm | Build | Heap | p50 | p90 | p99 | Save | Load |
|---|---:|---:|---:|---:|---:|---:|---:|
| vmis | 5514.4272 | 242.5890 | 0.0216 | 0.0463 | 0.0737 | 411.1462 | 1739.3077 |
| sknn | 5091.4284 | 242.5890 | 0.0411 | 0.1043 | 0.1807 | 429.2559 | 1495.9581 |
| ssknn | 5133.7067 | 242.5890 | 0.0406 | 0.1028 | 0.1771 | 459.0171 | 1503.9770 |
| wsknn | 5156.0166 | 242.5890 | 0.0323 | 0.0970 | 0.1694 | 453.4862 | 1419.3103 |
| vsknn | 4997.1217 | 242.5890 | 0.0444 | 0.1210 | 0.2315 | 408.7684 | 1413.5054 |
| stan | 4880.3537 | 242.5890 | 0.0476 | 0.1175 | 0.1864 | 404.4917 | 1339.7359 |
| vsknn_stan | 5051.6898 | 242.5890 | 0.0503 | 0.1216 | 0.1917 | 392.3003 | 1335.8315 |

S-SKNN and WSKNN have exactly the same owned index heap and serialized bytes as the other variants: 3,978,692 heap bytes / 2,267,412 snapshot bytes on the example, and 254,372,976 / 158,315,082 on the scaled dataset. Their build times are consistent with shared construction. S-SKNN p99 is comparable to SKNN; WSKNN remains below 0.2 ms at the scaled median p99. Neither introduces a memory regression. Differences among build/save/load measurements reflect repeated construction and host variability, not separate index layouts.

Complete results, including all eight recommendation metrics for all seven algorithms on both datasets, are retained in [consolidated benchmark CSV](validation/benchmark.csv). [Raw iterations](validation/benchmark-raw.csv) include warm-up rows, excluded by the consolidated aggregation. [Run metadata](validation/benchmark.metadata.json), [scaled data manifest](validation/scaled-manifest.json), [accuracy CSV](validation/accuracy.csv) and [Python metric arrays](validation/python-metrics.json) retain precision beyond these rounded tables. Python arrays are ordered MRR, NDCG, HitRate, Popularity, Precision, Coverage, Recall, F1.
