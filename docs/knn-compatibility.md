# Shared VMIS kNN implementation audit

Reference: [rn5l/session-rec, algorithms/knn](https://github.com/rn5l/session-rec/tree/5dcd583cbd8d44703a5248b9a308945f24b91390/algorithms/knn), pinned to `5dcd583cbd8d44703a5248b9a308945f24b91390`.
All eight implementations were inspected. SKNN, VS-KNN, STAN and VSKNN_STAN are implemented alongside the backwards-compatible default VMIS scorer. They share candidate retrieval, the existing index, output ranking, business filtering and serving.

| Algorithm | Required shared data | Additional data | Reuse and constraints | Decision |
| --- | --- | --- | --- | --- |
| SKNN (`sknn.py`) | Item postings, historical item sets | None for default scoring | VMIS retrieval, top-k, output filtering; set similarity and summed neighbor scores | Implemented |
| VS-KNN (`vsknn.py`) | Same, session timestamps, IDF | Raw training event frequencies for reference IDF | Last current occurrence determines position; match-position decay defaults to reciprocal; IDF score multiplier is `1 + idf * weight` | Implemented separately from legacy VMIS |
| STAN (`stan.py`) | Same, chronological historical sequences | Prediction timestamp | Weighted cosine, age decay in seconds, last historical positions; shares positional scoring with VSKNN_STAN | Implemented |
| VSKNN_STAN (`vstan.py`) | Same, chronological historical sequences, IDF | Prediction timestamp, raw training event frequencies | STAN plus latest matching current-position decay and additive IDF weighting | Implemented |
| USTAN (`ustan.py`) | STAN data | Stable user identity in training and requests, session-to-user mapping, recent user items; optional reminder state | Lightweight metadata would suffice for core extension/own-session boosting, but required identity is absent | Exclude for current input/API contract |
| UVSKNN (`uvsknn.py`) | VS-KNN data | Same user data; optional past-neighbor/reminder activity | Cannot implement user extension, own-session boost or past-user neighbors without users | Exclude for current input/API contract |
| U_VSKNN_STAN (`uvstan.py`) | VSKNN_STAN data | Same user data and reminder activity | Same missing inputs; disabling user features is not an implementation | Exclude for current input/API contract |
| Item-KNN (`iknn.py`) | Precomputed item-to-item similarity matrix | Different index | Cannot use item-to-session retrieval as its similarity representation | Exclude |

Training currently accepts `(session_id, item_id, timestamp)` only. Recommendation requests identify a session, an item and user consent; consent is not a user identity. There is no cross-session user history. User-aware support would require an explicit data/API expansion, beyond lightweight metadata derived from today's inputs.

## Sequence storage and legacy behavior

The existing `Vec<Vec<u64>>` must retain events sorted by session and timestamp, preserving input order for timestamp ties and retaining repeated items. No positional index or per-event timestamp storage is needed after construction. The historical field name `session_to_items_sorted` is legacy terminology, not a guarantee of item-ID order.

VMIS handles historical items as a set during inference. Each neighbor contributes once per distinct item. Inverted postings contain each session once, so repeated events cannot consume the candidate budget or multiply similarity. Legacy IDF counts distinct items per session. Posting pruning compares distinct session items to the data-derived p99.5 of raw interaction counts (including repeats). This threshold intentionally differs from previous builds. Reference IDF instead uses `ln(number_of_sessions / raw_item_event_count)`, which can be negative; the two must not be conflated.

The CSV reader deliberately corrects the old grouping logic, which lost the final event/session and failed to update timestamps for repeat visits. Correcting those cases can change recommendations for affected training data; it is separate from the tested invariance of scoring a fixed historical session with repetitions.

## Python fidelity constraints

* SKNN uses sets. Its cosine code is `intersection / sqrt(neighbor_size) * sqrt(current_size)`, despite the conventional formula being different. Binary similarity is `2 * intersection / (2 * intersection + neighbor_size + current_size)`.
* VS-KNN maps repeated current items to their last position. Session IDF weighting replaces position weights. Match-position scoring defaults to `1 / step`, not legacy VMIS's linear decay. Optional age weighting uses Python's rounded days and `(7/8)^days`.
* STAN and VSKNN_STAN use unique-item weighted cosine. Historical positions are last-occurrence positions; the anchor is the final historical position containing any current item, rather than the position of the most recently clicked current item.
* VSKNN_STAN adds `score * idf * lambda_idf` and multiplies by the latest matching current item's exponential position weight.
* VMIS pruning remains bounded by recent-session postings. Parity tests must fix candidates independently of retrieval. Python's accumulated streaming candidate history, random sampling and unrestricted candidate pools do not describe VMIS retrieval.
* Serving filters (including removal of the latest item) apply after scoring and need to be distinguished from Python raw score parity.

## Persistence

Existing Avro `item_ids_asc` contains sorted unique IDs. It remains sufficient for legacy VMIS and set-based algorithms without reference IDF, but cannot recover event order, repeated visits or raw event frequencies. Loading positional algorithms or reference IDF from Avro is rejected; rebuild from event data. No migration can infer the lost information.

`build_index CONFIG OUTPUT.vmis` writes a bincode snapshot prefixed with `SERENADE-VMIS-003`. Training paths accept CSV, Avro directories with raw statistics metadata, or this format. Versions 001/002 must be rebuilt from original events because their statistics may be fabricated; durations cannot be recovered from saved sequences. See [Preparation](Preparation.md). Snapshots record chronological availability and posting capacity. Loading with a larger candidate budget is rejected because truncated postings cannot recover older sessions. Unknown snapshot versions are rejected. A chronological snapshot can switch between reference and legacy IDF: the single IDF map is rebuilt from its event lists. No redundant positional or IDF index is stored. Snapshots deserialize through a buffered reader without an additional full-file copy.

## Supported scope and differences

Reference defaults and scoring options are exposed in [CONFIG.md](CONFIG.md). SKNN reminder/popularity boosts are included. VS-KNN supports positional and match weights, time decay, score IDF, session IDF, reminder filtering and normalization. STAN variants share positional cosine and historical-distance scoring.

The Python online `extend`, dwell-time weighting, VS-KNN `push_reminders`/`add_reminders`, random sampling, and training-window convenience options are not implemented. Explicit configuration of these options is rejected. Online training needs mutable indexes; dwell time needs click-time state; the reminder postprocessors operate on catalog-sized score vectors beyond the shared sparse scoring pipeline. These are optional extensions disabled by the upstream defaults, not replacements for missing user-aware functionality. Use Serenade's current-session limit and preprocess training windows when needed.

VMIS bounded candidate pruning replaces Python's streaming relevant-session cache/sampling. Identical-candidate tests isolate scoring from that retrieval difference. Session timestamps retain Serenade's rounded `u32` seconds; Python's fractional historical timestamps can therefore differ by at most half a second before age decay. Current evaluation timestamps preserve fractions. Equal-score item order remains unspecified, as in existing Serenade. New neighbor ties retain candidate order. The shared output stage removes the latest item and applies business filters. Normalization avoids division by zero when no positive maximum exists rather than returning Python NaNs.

## Validation status

The initial baseline required enabling console's `std` feature for indicatif 0.16. With that build repair, main passed 18 library tests. The implementation passes 30 tests across all targets, including 46 deterministic Python reference cases comparing similarities, top-k neighbors, complete item scores and filtered final rankings at tolerance 1e-12. Tests cover repeats, negative IDF, all supported weighting functions, time decay, disabled decay components, snapshot migration and legacy VMIS repeat invariance. See [reference instructions](../tests/reference/README.md) and [benchmark results](knn-benchmark.md). The fixtures do not constitute an exhaustive proof over every possible dataset or parameter combination.
