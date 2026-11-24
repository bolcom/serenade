//! Parallel objective evaluation over borrowed, immutable corpus data.
use super::EvaluationData;
use crate::{
    config::ModelConfig,
    metrics::{
        hitrate::HitRate,
        mrr::Mrr,
        ndcg::Ndcg,
        objective::{Objective, RankingMetric},
        SessionMetric,
    },
    vmisknn::{self, scoring::ScoringView, vmis_index::VMISIndex, PredictionScratch},
};
use rayon::prelude::*;
use std::sync::Mutex;

pub struct Execution {
    pool: rayon::ThreadPool,
    buffers: Vec<Mutex<Buffers>>,
}
pub struct RunResult {
    pub predictions: usize,
    pub objective: f64,
}
#[derive(Default)]
struct Buffers {
    scratch: PredictionScratch,
    items: Vec<u64>,
    ranked: Vec<u64>,
}
// Select exactly one accumulator; no catalog statistics, predictions or timings
// are retained. Each metric keeps its established recommendation semantics.
enum Metric {
    Mrr(Mrr),
    Ndcg(Ndcg),
    HitRate(HitRate),
}
impl Metric {
    fn new(objective: Objective) -> Self {
        match objective.metric() {
            RankingMetric::Mrr => Self::Mrr(Mrr::new(objective.cutoff())),
            RankingMetric::Ndcg => Self::Ndcg(Ndcg::new(objective.cutoff())),
            RankingMetric::HitRate => Self::HitRate(HitRate::new(objective.cutoff())),
        }
    }
    fn add(&mut self, ranked: &[u64], target: &[u64]) {
        match self {
            Self::Mrr(m) => m.add(ranked, target),
            Self::Ndcg(m) => m.add(ranked, target),
            Self::HitRate(m) => m.add(ranked, target),
        }
    }
    fn merge(&mut self, other: Self) {
        match (self, other) {
            (Self::Mrr(a), Self::Mrr(b)) => a.merge(b),
            (Self::Ndcg(a), Self::Ndcg(b)) => a.merge(b),
            (Self::HitRate(a), Self::HitRate(b)) => a.merge(b),
            _ => unreachable!("one objective per evaluation"),
        }
    }
    fn result(&self) -> f64 {
        match self {
            Self::Mrr(m) => m.result(),
            Self::Ndcg(m) => m.result(),
            Self::HitRate(m) => m.result(),
        }
    }
}
impl Execution {
    /// Zero selects the CPU availability reported by the operating system.
    pub fn new(workers: usize) -> anyhow::Result<Self> {
        let workers = if workers == 0 {
            std::thread::available_parallelism()?.get()
        } else {
            workers
        };
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()?,
            buffers: (0..workers)
                .map(|_| Mutex::new(Buffers::default()))
                .collect(),
        })
    }
    pub fn workers(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// The existing index must have been prepared for this algorithm with unit
    /// legacy IDF weight, once before its sequential trials. This method never
    /// changes the index or scans the historical corpus.
    pub fn run(
        &self,
        index: &VMISIndex,
        data: &EvaluationData,
        model: &ModelConfig,
        business: bool,
        objective: Objective,
    ) -> anyhow::Result<RunResult> {
        let view = ScoringView::new(index, model).map_err(anyhow::Error::msg)?;
        let cutoff = objective.cutoff();
        // At most one active chunk per worker. Results carry only accumulators;
        // prediction allocations stay in their worker slots across trials.
        let chunk = data.sessions.len().div_ceil(self.workers()).max(1);
        let (predictions, metric) = self.pool.install(|| {
            data.sessions
                .par_chunks(chunk)
                .map(|sessions| {
                    let id = rayon::current_thread_index().unwrap();
                    // Uncontended worker-local lock, acquired outside prediction.
                    let mut buffers = self.buffers[id].lock().unwrap();
                    let Buffers {
                        scratch,
                        items,
                        ranked,
                    } = &mut *buffers;
                    let mut metric = Metric::new(objective);
                    let mut predictions = 0;
                    for events in sessions {
                        items.clear();
                        items.extend(events.iter().map(|e| e.0));
                        for state in 1..items.len() {
                            let current =
                                &items[state.saturating_sub(model.max_items_in_session)..state];
                            let recommendations = vmisknn::predict_with_model_scratch(
                                index,
                                &view,
                                current,
                                model.neighborhood_size_k,
                                model.m_most_recent_sessions,
                                cutoff,
                                business,
                                &model.knn,
                                events[state - 1].1,
                                scratch,
                            );
                            ranked.clear();
                            ranked.extend(recommendations.into_sorted_vec().iter().map(|s| s.id));
                            metric.add(ranked, &items[state..]);
                            predictions += 1;
                        }
                    }
                    (predictions, metric)
                })
                .reduce(
                    || (0, Metric::new(objective)),
                    |(n, mut a), (m, b)| {
                        a.merge(b);
                        (n + m, a)
                    },
                )
        });
        Ok(RunResult {
            predictions,
            objective: metric.result(),
        })
    }
}
