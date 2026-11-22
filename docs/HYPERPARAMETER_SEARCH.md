Hyperparameter Search
===

TPE (`src/bin/tpe_hyperparameter_optm.rs`) is the only supported optimizer.
Run `cargo run --release --bin tpe_hyperparameter_optm -- CONFIG.toml`.
The shared objective evaluates the selected `[model] algorithm`: VMIS-kNN, SKNN,
VS-KNN, STAN or V-STAN (`vsknn_stan`). Existing hyperparameter ranges control
candidate budget, neighborhood size, current history length and IDF weight.
The IDF axis maps to VS-KNN's `idf_weighting` or V-STAN's `lambda_idf`; it has
no scoring effect for SKNN/STAN. Other scoring hyperparameters are configured
in `[model]` and stay fixed for a search. See [CONFIG](CONFIG.md).

The [TPE crate](https://docs.rs/tpe/0.2.0/tpe/) minimizes its objective. Serenade supplies negative MRR to maximize recommendation quality, while displaying positive MRR. One optimizer tracks each search axis; sampled configurations are evaluated by the shared objective and observations guide subsequent trials.
6. Repeat steps 2-5 with a fixed number of trials or until the time limit is reached.
