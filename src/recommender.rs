use crate::{
    config::{load_model, validate_model},
    vmisknn::{self, scoring::ScoringView},
    ModelConfig, PredictionScratch, Recommendation, VMISIndex,
};
use std::{cell::RefCell, path::Path, time::SystemTime};

thread_local! {
    static SCRATCH: RefCell<PredictionScratch> = RefCell::new(PredictionScratch::default());
}

/// Construction options independent of algorithm selection.
///
/// CSV/TSV inputs default to 2500 recent sessions per posting, just like the
/// applications. Snapshots retain their stored capacity. No model can expand it.
#[derive(Default, Debug, Clone)]
pub struct IndexBuilder {
    capacity: Option<usize>,
}

impl IndexBuilder {
    /// Set a positive posting capacity. For snapshots this must not exceed the
    /// stored capacity; loading never rebuilds or expands a snapshot.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    /// Load training interactions, a versioned snapshot, or a legacy Avro index
    /// directory. Reports invalid paths, data and capacities as errors.
    pub fn load(self, path: impl AsRef<Path>) -> anyhow::Result<VMISIndex> {
        anyhow::ensure!(self.capacity != Some(0), "index capacity must be positive");
        let path = path.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("index path must be valid UTF-8"))?;
        VMISIndex::load_with_capacity(path, self.capacity, true).map_err(anyhow::Error::msg)
    }
}

impl VMISIndex {
    /// Load training data with the application default capacity (2500), or an
    /// existing snapshot at its stored capacity. Loading is algorithm independent.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::builder().load(path)
    }

    /// Customize index construction without selecting a recommendation algorithm.
    pub fn builder() -> IndexBuilder {
        IndexBuilder::default()
    }
}

/// A validated model borrowing one immutable index, without copying its corpus.
///
/// Multiple models can borrow the same index. Both index and recommender are
/// `Send + Sync`; share them across scoped threads or put the index in an `Arc`
/// and construct models inside each owner. Models cannot outlive their index.
/// Construction validates retrieval capacity and snapshot capabilities upfront.
/// When mixing IDF formulas, one alternate item-sized statistics map is lazily
/// built and shared by the index. No sessions or postings are duplicated.
/// Older weighted CSV indexes may also cache unit legacy statistics; model weights
/// are applied independently. Use [`VMISIndex::load`] or [`VMISIndex::builder`]
/// to avoid this extra normalization cache.
/// Legacy Avro indexes preserve their baked-in VMIS weights and support only
/// algorithms whose required raw-event information is present.
///
/// Predictions reuse thread-local scratch buffers across models and calls. Their
/// retained memory follows the worker's largest query, not the entire corpus.
/// Use [`Self::predict_with_scratch`] to control scratch ownership explicitly.
/// Business filtering is disabled by default; algorithm-specific `remind` and
/// normalization settings still apply.
///
/// ```no_run
/// use serenade::{Algorithm, ModelConfig, Recommender, VMISIndex};
/// # fn main() -> anyhow::Result<()> {
/// let index = VMISIndex::load("train.txt")?;
/// let vmis = Recommender::from_config(&index, "best-vmis.toml")?;
/// let wsknn = Recommender::new(&index, ModelConfig::for_algorithm(Algorithm::Wsknn))?;
/// let recommendations = vmis.predict(&[13598, 2835, 12068], 20)?;
/// for item in recommendations {
///     println!("{}: {}", item.id, item.score);
/// }
/// # Ok(()) }
/// ```
pub struct Recommender<'a> {
    index: &'a VMISIndex,
    view: ScoringView<'a>,
    model: ModelConfig,
    business: bool,
}

impl<'a> Recommender<'a> {
    /// Configure a model programmatically. Invalid parameters and incompatible
    /// indexes return errors before any prediction is attempted.
    pub fn new(index: &'a VMISIndex, model: ModelConfig) -> anyhow::Result<Self> {
        validate_model(&model, usize::MAX)?;
        let view = ScoringView::new(index, &model).map_err(anyhow::Error::msg)?;
        Ok(Self {
            index,
            view,
            model,
            business: false,
        })
    }

    /// Load the same portable model file produced by HPO and read by the evaluator
    /// and serving applications. No server or dataset settings are required.
    pub fn from_config(index: &'a VMISIndex, path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::new(index, load_model(path.as_ref())?)
    }

    /// Enable the existing attribute-based business filters, when desired.
    pub fn with_business_logic(mut self, enabled: bool) -> Self {
        self.business = enabled;
        self
    }

    /// Recommend up to `limit` items, best score first, then descending item ID
    /// for ties, matching the applications. Scores retain native algorithm semantics.
    ///
    /// The session is ordered oldest to newest. Only its last configured
    /// `max_items_in_session` items are used. Empty/unknown sessions and a zero
    /// limit return an empty vector. Time-based models use the current Unix time;
    /// use [`Self::predict_at`] for replayable predictions. The file's
    /// `num_items_to_recommend` is overridden by `limit`.
    pub fn predict(&self, session: &[u64], limit: usize) -> anyhow::Result<Vec<Recommendation>> {
        self.predict_at(
            session,
            limit,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs_f64(),
        )
    }

    /// Predict at a finite, nonnegative Unix timestamp in seconds, using
    /// worker-local scratch. Use the last interaction's timestamp for evaluation.
    pub fn predict_at(
        &self,
        session: &[u64],
        limit: usize,
        timestamp: f64,
    ) -> anyhow::Result<Vec<Recommendation>> {
        SCRATCH.with(|scratch| {
            self.predict_with_scratch(session, limit, timestamp, &mut scratch.borrow_mut())
        })
    }

    /// Predict with buffers owned by the calling worker, avoiding global locks
    /// and per-call scratch allocation. Buffers may be reused across models.
    pub fn predict_with_scratch(
        &self,
        session: &[u64],
        limit: usize,
        timestamp: f64,
        scratch: &mut PredictionScratch,
    ) -> anyhow::Result<Vec<Recommendation>> {
        anyhow::ensure!(
            timestamp.is_finite() && timestamp >= 0.0,
            "prediction timestamp must be finite and nonnegative"
        );
        let session = &session[session
            .len()
            .saturating_sub(self.model.max_items_in_session)..];
        Ok(vmisknn::predict_with_model_scratch(
            self.index,
            &self.view,
            session,
            self.model
                .neighborhood_size_k
                .min(self.model.m_most_recent_sessions),
            self.model.m_most_recent_sessions,
            limit,
            self.business,
            &self.model.knn,
            timestamp,
            scratch,
        )
        .into_sorted_vec())
    }
}
