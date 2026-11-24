# Data preparation

The binary package includes example training, validation, and test data. To use your own data, create three tab-separated text files with a header and these columns in this order:

```text
SessionId	ItemId	Time
10036	14957	1592337718
10036	14713	1592337765
10037	7267	1592337800
```

Use actual tab characters between columns.

| Column | Value |
| --- | --- |
| `SessionId` | Nonnegative integer fitting an unsigned 32-bit integer |
| `ItemId` | Nonnegative integer fitting an unsigned 64-bit integer |
| `Time` | Finite, nonnegative Unix timestamp in seconds; fractional seconds accepted |

Historical timestamps are rounded to unsigned 32-bit seconds. Events are sorted within each session; equal timestamps preserve file order. Repeated clicks are supported.

Split complete sessions into training, validation, and test sets, preferably in chronological order. Keep each session in only one split. Training builds the historical index, validation selects hyperparameters, and test measures held-out recommendation quality. Holdout sessions are never added to the historical index.

Place `train.txt`, `validation.txt`, and `test.txt` in the workspace directory,
or leave large datasets on their current volume and use the explicit path overrides
in [the workflow guide](BINARIES.md). HPO requires only training and validation;
evaluation requires only training and test; serving requires only training or an index.
Generated model files contain no dataset paths.

Use the tab-separated format to get started with all seven algorithms. Allow enough RAM for the training index; requirements depend on catalog size, session history, and candidate budget.

Next: [optimize, evaluate, and serve](BINARIES.md).
