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

Update the existing `[hyperparam]` section in `configs/hpo.toml`:

```toml
[hyperparam]
training_data_path = "data/train.txt"
validation_data_path = "data/valid.txt"
test_data_path = "data/test.txt"
```

Keep the remaining HPO settings. Run commands from the extracted package directory, with paths relative to that directory, or use absolute paths. Generated model configurations retain the dataset paths for evaluation and serving.

Use the tab-separated format to get started with all five algorithms. Allow enough RAM for the training index; requirements depend on catalog size, session history, and candidate budget.

Next: [optimize, evaluate, and serve](BINARIES.md).
