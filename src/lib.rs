//! Session recommendations using one immutable index shared by borrowed models.
//!
//! Disable default features for an embedded library without the HTTP server,
//! session database or optimizer dependencies. See [`Recommender`] for usage.
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
