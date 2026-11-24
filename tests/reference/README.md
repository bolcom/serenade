# Python reference fixtures

All eight originals were audited at [session-rec commit 5dcd583](https://github.com/rn5l/session-rec/tree/5dcd583cbd8d44703a5248b9a308945f24b91390/algorithms/knn).

Download sknn.py, vsknn.py, stan.py and vstan.py from that pinned directory alongside generate.py, then run `python tests/reference/generate.py` from the repository root. Local upstream copies are ignored. Only Python standard library is needed. The generator AST-loads original class definitions, avoiding pandas/numpy imports, and executes unmodified reference methods. SKNN reminder/popularity blocks are extracted from the original prediction method AST. No formulas are translated into the generator.

scores.tsv contains 46 cases with identical fixed candidates, repeated items, k=2/10, similarity/weighting choices, negative IDF, recency, reminder/popularity boosts and disabled STAN components. `scoring_matches_upstream_python_on_identical_candidates` compares neighbors, similarities, all raw item scores and final filtered rankings at 1e-12 tolerance. Final ranking applies Serenade normalization/latest-item removal; ties allow either order.

## RecBole metric fixtures

`generate_recobole.py` downloads pinned RecBole commit 2d7158418d1181cd2b9f4fca94923d0aa96bf2ad and executes its MAP, GiniIndex, ShannonEntropy and TailPercentage classes unchanged (AST-loaded to avoid PyTorch dependencies). It needs NumPy; `np.float` is restored as an alias for compatibility with modern NumPy. Run `python tests/reference/generate_recobole.py` from the root. The generated `recobole.json` records the source URL and SHA256. Rust tests verify the unrounded results at every fixture cutoff, including merged workers and selected objectives. See [metric semantics](../../docs/METRICS.md) for comparison requirements and extensions.
