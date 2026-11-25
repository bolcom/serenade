//! # Installation
//!
//! Add the recommendation library without application dependencies:
//!
//! ```sh
//! cargo add serenade --no-default-features
//! ```
//!
//! # Quick start
//!
//! Load training interactions, configure a model, and predict from a session
//! ordered oldest to newest. Training files have a header followed by three
//! tab-separated columns: `SessionId ItemId Time` (Unix seconds).
//!
//! ```
//! use serenade::{ModelConfig, Recommender, VMISIndex};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let root = std::env::temp_dir().join(format!("serenade-quickstart-{}", std::process::id()));
//! # std::fs::create_dir_all(&root)?;
//! # let training = root.join("train.txt");
//! # std::fs::write(&training, "SessionId\tItemId\tTime\n1\t13598\t100\n1\t2835\t101\n1\t12068\t102\n2\t13598\t103\n2\t2835\t104\n2\t42\t105\n")?;
//! # let index = VMISIndex::load(&training)?;
//! # /*
//! let index = VMISIndex::load("train.txt")?;
//! # */
//! let recommender = Recommender::new(&index, ModelConfig::default())?;
//! let recommendations = recommender.predict(&[13598, 2835], 20)?;
//! for item in &recommendations {
//!     println!("{}: {}", item.id, item.score);
//! }
//! # assert!(!recommendations.is_empty());
//! # std::fs::remove_dir_all(root)?;
//! # Ok(()) }
//! ```
//!
//! [`Recommendation`] has an item `id: u64` and a `score: f64`. Results are ranked
//! by descending score, then descending item ID for ties. Scores retain each
//! algorithm's semantics; they are not probabilities or comparable across models.
//! Empty or entirely unknown sessions and a zero cutoff return no results.
//! Only the configured [`ModelConfig::max_items_in_session`] most recent items
//! are used. [`Recommender::predict_at`] accepts an explicit Unix timestamp for
//! reproducible evaluation; [`Recommender::predict`] uses the current time.
//!
//! # Configuration and algorithms
//!
//! [`Recommender::from_config`] reads the portable TOML model files exported by
//! Serenade's hyperparameter optimizer and used by its evaluator and server.
//! Dataset paths and server settings are not required:
//!
//! ```
//! use serenade::{Recommender, VMISIndex};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let root = std::env::temp_dir().join(format!("serenade-config-doc-{}", std::process::id()));
//! # std::fs::create_dir_all(&root)?;
//! # std::fs::write(root.join("train.txt"), "SessionId\tItemId\tTime\n1\t1\t100\n1\t2\t101\n2\t1\t102\n2\t3\t103\n")?;
//! # std::fs::write(root.join("best-vmis.toml"), "[model]\nalgorithm=\"vmis\"\n")?;
//! # let index = VMISIndex::load(root.join("train.txt"))?;
//! # let model_path = root.join("best-vmis.toml");
//! # /*
//! let index = VMISIndex::load("train.txt")?;
//! let model_path = "best-vmis.toml";
//! # */
//! let model = Recommender::from_config(&index, model_path)?;
//! let recommendations = model.predict(&[1], 20)?;
//! # assert!(!recommendations.is_empty());
//! # std::fs::remove_dir_all(root)?;
//! # Ok(()) }
//! ```
//!
//! For programmatic configuration, start with [`ModelConfig::for_algorithm`]
//! and override its fields. [`KnnConfig`] contains scoring options, including
//! [`Similarity`], [`Weighting`], IDF, normalization, reminders and decay settings.
//! Use the algorithm's defaults as a starting point: valid combinations differ.
//! Constructors return errors for invalid parameters or incompatible indexes.
//!
//! | [`Algorithm`] | Scoring approach | TOML `model.algorithm` |
//! | --- | --- | --- |
//! | [`Vmis`](Algorithm::Vmis) (default) | VMIS session similarity and IDF-weighted item scores | `vmis` |
//! | [`Sknn`](Algorithm::Sknn) | Session nearest neighbors | `sknn` |
//! | [`Ssknn`](Algorithm::Ssknn) | Sequential session nearest neighbors | `ssknn` |
//! | [`Wsknn`](Algorithm::Wsknn) | Position- and distance-weighted session neighbors | `wsknn` |
//! | [`Vsknn`](Algorithm::Vsknn) | Vector-weighted session nearest neighbors | `vsknn` |
//! | [`Stan`](Algorithm::Stan) | Session position, neighbor age and item distance decay | `stan` |
//! | [`VsknnStan`](Algorithm::VsknnStan) | Vector weighting with STAN decay and optional IDF | `vsknn_stan` |
//!
//! # Index sharing, capacity and concurrency
//!
//! All algorithms use the same immutable [`VMISIndex`]. Recommenders borrow it
//! and cannot outlive it; constructing another model does not copy sessions or
//! postings. The default training-data capacity is 2500 recent sessions per
//! posting. Set it explicitly with [`IndexBuilder::capacity`]; models never
//! expand it automatically. A model's `m_most_recent_sessions` must fit it.
//!
//! This example configures two algorithms and shares both models across workers:
//!
//! ```
//! use serenade::{Algorithm, ModelConfig, Recommender, VMISIndex};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let root = std::env::temp_dir().join(format!("serenade-sharing-doc-{}", std::process::id()));
//! # std::fs::create_dir_all(&root)?;
//! # let training = root.join("train.txt");
//! # std::fs::write(&training, "SessionId\tItemId\tTime\n1\t1\t100\n1\t2\t101\n2\t1\t102\n2\t3\t103\n")?;
//! # /*
//! let training = "train.txt";
//! # */
//! let index = VMISIndex::builder().capacity(1000).load(training)?;
//! let mut config = ModelConfig::for_algorithm(Algorithm::Wsknn);
//! config.m_most_recent_sessions = 500;
//! config.neighborhood_size_k = 100;
//! config.max_items_in_session = 10;
//! let wsknn = Recommender::new(&index, config)?;
//! let vmis = Recommender::new(&index, ModelConfig::default())?;
//!
//! std::thread::scope(|scope| {
//!     for model in [&vmis, &wsknn] {
//!         scope.spawn(move || {
//!             let recommendations = model.predict(&[1], 20).unwrap();
//!             assert!(!recommendations.is_empty());
//!         });
//!     }
//! });
//! # std::fs::remove_dir_all(root)?;
//! # Ok(()) }
//! ```
//!
//! Both index and recommender are `Send + Sync`. For independently owned workers,
//! put the index in [`std::sync::Arc`] and construct a borrowing recommender
//! inside each worker. Predictions reuse thread-local scratch buffers without
//! a shared prediction lock. [`Recommender::predict_with_scratch`] accepts a
//! worker-owned [`PredictionScratch`] for explicit reuse. Scratch memory follows
//! the worker's largest query; it does not allocate a buffer for every corpus
//! session. Mixing IDF formulas may lazily cache alternate item-sized statistics
//! once per index; sessions and postings remain shared.
//!
//! # Persisting an index
//!
//! [`VMISIndex::save`] writes a versioned snapshot; [`VMISIndex::load`] detects
//! and restores it, retaining its stored capacity and chronological events.
//! Loading snapshots does not rebuild or enlarge their postings. Model
//! configuration stays separate so multiple models can reuse the snapshot.
//!
//! ```
//! use serenade::{ModelConfig, Recommender, VMISIndex};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let root = std::env::temp_dir().join(format!("serenade-snapshot-doc-{}", std::process::id()));
//! # std::fs::create_dir_all(&root)?;
//! # let training = root.join("train.txt");
//! # std::fs::write(&training, "SessionId\tItemId\tTime\n1\t1\t100\n1\t2\t101\n2\t1\t102\n2\t3\t103\n")?;
//! # let snapshot = root.join("index.vmis");
//! # let snapshot = snapshot.to_str().unwrap();
//! # /*
//! let training = "train.txt";
//! let snapshot = "index.vmis";
//! # */
//! let index = VMISIndex::load(training)?;
//! index.save(snapshot)?;
//! let restored = VMISIndex::load(snapshot)?;
//! let model = Recommender::new(&restored, ModelConfig::default())?;
//! let recommendations = model.predict_at(&[1], 20, 104.0)?;
//! # assert!(!recommendations.is_empty());
//! # std::fs::remove_dir_all(root)?;
//! # Ok(()) }
//! ```
//!
//! Existing legacy Avro index directories are also accepted. They may lack the
//! raw chronology or statistics needed by some algorithms; model construction
//! rejects incompatible combinations upfront. They preserve their baked-in
//! VMIS weights. Prefer loading training data or a current snapshot for access
//! to all algorithms.
//!
//! # Cargo features
//!
//! No feature is required for the recommendation API. With
//! `default-features = false`, Actix, RocksDB, Prometheus and TPE are excluded.
//! docs.rs builds this minimal configuration.
//!
//! | Feature | Enables |
//! | --- | --- |
//! | `applications` (default) | `serving` and `hpo`, producing all three executables |
//! | `serving` | HTTP server, session database and metrics integration |
//! | `evaluation` | Offline evaluator executable |
//! | `hpo` | TPE optimizer executable and `evaluation` |
//!
//! Start with the root-level types above. The other public modules retain
//! lower-level APIs and application integrations for existing users.
mod recommender;
pub use config::ModelConfig;
pub use recommender::{IndexBuilder, Recommender};
pub use vmisknn::algorithms::{Algorithm, KnnConfig, Similarity, Weighting};
pub use vmisknn::vmis_index::VMISIndex;
pub use vmisknn::{ItemScore as Recommendation, PredictionScratch};

pub mod config;
pub mod config_processors;
pub mod dataframeutils;
#[cfg(feature = "serving")]
pub mod endpoints;
pub mod io;
pub mod metrics;
pub mod objective;
mod serialization;
#[cfg(feature = "serving")]
pub mod sessions;
pub mod stopwatch;
pub mod vmisknn;

#[cfg(feature = "serving")]
pub mod executor;

pub mod offline;

pub mod paths;
