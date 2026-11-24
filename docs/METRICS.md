# Offline metrics

The evaluator reports fifteen metrics at K=20. Console output separates ranking
quality from catalog exposure, diversity and completeness; CSV retains full metric
names and precision. HPO accepts the names below as `hyperparam.objective = "name@k"`
with any positive K, independently of the serving recommendation count.

| Metric / HPO name | Definition | Optimize |
| --- | --- | --- |
| MAP / `map` | Mean of summed precision at relevant ranks, divided by min(number of distinct remaining target items, K). | Maximize |
| Gini / `gini` | Gini coefficient of recommendation frequencies across the catalog, including zero-exposure items. | Minimize |
| ShannonEntropy / `shannonentropy` | `-sum(p * ln(p)) / number_of_distinct_exposed_items`, where p is an item's share of all recommendations. | Maximize |
| LongTailShare / `longtailshare` | Mean number of tail recommendations divided by K. Tail is the least-frequent 10% of distinct training items, with at least one item; ties use ascending item ID. | Maximize |
| IntraListDiversity / `intralistdiversity` | Mean pairwise cosine distance of binary item-to-training-session vectors, averaged across prediction cases. | Maximize |
| EmptyResultRate / `emptyresultrate` | Fraction of prediction cases with no recommendations. | Minimize |
| FillRate / `fillrate` | Mean number of distinct returned items divided by K. | Maximize |

All metrics inspect only the first K recommendation positions. Repeated prediction
IDs count once; MAP preserves the original ranks. Repeated target events count once
for MAP. Empty recommendations contribute zero to MAP, tail share, diversity and
fill rate, and one to empty-result rate. With no evaluation cases, all seven new
metrics return zero. Missing recommendation slots contribute zero to tail share
and fill rate. Unknown external prediction IDs count toward exposure and fill, but
are not tail items; pairs without training-session vectors are omitted from
diversity. Lists with fewer than two known distinct items have diversity zero.
Gini's catalog includes distinct training IDs plus any unknown predicted IDs.
Entropy and Gini aggregate frequencies globally across workers, rather than
averaging worker-level coefficients. With no exposure, both return zero.

## RecBole compatibility

MAP, Gini, ShannonEntropy and LongTailShare follow the implementations of
`MAP`, `GiniIndex`, `ShannonEntropy` and `TailPercentage` in
[RecBole commit 2d715841](https://github.com/RUCAIBox/RecBole/blob/2d7158418d1181cd2b9f4fca94923d0aa96bf2ad/recbole/evaluator/metrics.py).
In particular, entropy follows the upstream implementation's natural logarithm
and division by distinct exposed items; it is not the usual unnormalized Shannon
entropy. Tail share matches RecBole's default `tail_ratio = 0.1`.

Comparisons require the same unique top-K lists, target sets, training interaction
counts, item-ID tie order, and catalog size. RecBole's `data.num_items` must equal
the catalog size used here: its usual internal padding ID can otherwise change
Gini's denominator. Do not compare different ID mappings without aligning ties.
Serenade keeps full precision; round to RecBole's `metric_decimal_place` for printed
comparisons. Tests use upstream results before rounding with tolerance 1e-12.

RecBole expects rectangular full-length lists and nonempty target sets. The
short-list, empty-list, duplicate-ID and unknown-ID behavior above is a Serenade
extension. RecBole has no implementations of the three diversity/completeness
extensions in this pinned source. Existing metrics retain historical behavior:
MRR and HitRate assess the immediate next item, NDCG uses Serenade's discount,
and popularity is normalized. They are not claimed to match RecBole.

`tests/reference/generate_recobole.py` executes the four original upstream
classes to produce `tests/reference/recobole.json`. Rust tests compare all cutoffs,
including merged worker accumulators and selected HPO accumulators. Regeneration
requires Python with NumPy and network access; normal Rust tests use the checked-in
fixtures without either dependency.
