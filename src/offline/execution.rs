//! Shared parallel prediction execution over borrowed, immutable corpus data.
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
// HPO selects one objective accumulator without timings.
// Quality shares catalog statistics; performance passes retain no metric state.
enum Metric {
    Full(crate::metrics::evaluation_reporter::EvaluationReporter),
    None,
    Mrr(Mrr),
    Ndcg(Ndcg),
    HitRate(HitRate),
    Beyond(crate::metrics::beyond_accuracy::BeyondAccuracy, usize),
}
impl Metric {
    fn new(
        objective: Objective,
        catalog: Option<std::sync::Arc<crate::metrics::beyond_accuracy::Catalog>>,
    ) -> Self {
        match objective.metric() {
            RankingMetric::Mrr => Self::Mrr(Mrr::new(objective.cutoff())),
            RankingMetric::Ndcg => Self::Ndcg(Ndcg::new(objective.cutoff())),
            RankingMetric::HitRate => Self::HitRate(HitRate::new(objective.cutoff())),
            RankingMetric::Map => Self::Beyond(
                crate::metrics::beyond_accuracy::BeyondAccuracy::selected(
                    catalog.unwrap(),
                    objective.cutoff(),
                    0,
                ),
                0,
            ),
            RankingMetric::Gini => Self::Beyond(
                crate::metrics::beyond_accuracy::BeyondAccuracy::selected(
                    catalog.unwrap(),
                    objective.cutoff(),
                    1,
                ),
                1,
            ),
            RankingMetric::ShannonEntropy => Self::Beyond(
                crate::metrics::beyond_accuracy::BeyondAccuracy::selected(
                    catalog.unwrap(),
                    objective.cutoff(),
                    2,
                ),
                2,
            ),
            RankingMetric::LongTailShare => Self::Beyond(
                crate::metrics::beyond_accuracy::BeyondAccuracy::selected(
                    catalog.unwrap(),
                    objective.cutoff(),
                    3,
                ),
                3,
            ),
            RankingMetric::IntraListDiversity => Self::Beyond(
                crate::metrics::beyond_accuracy::BeyondAccuracy::selected(
                    catalog.unwrap(),
                    objective.cutoff(),
                    4,
                ),
                4,
            ),
            RankingMetric::EmptyResultRate => Self::Beyond(
                crate::metrics::beyond_accuracy::BeyondAccuracy::selected(
                    catalog.unwrap(),
                    objective.cutoff(),
                    5,
                ),
                5,
            ),
            RankingMetric::FillRate => Self::Beyond(
                crate::metrics::beyond_accuracy::BeyondAccuracy::selected(
                    catalog.unwrap(),
                    objective.cutoff(),
                    6,
                ),
                6,
            ),
        }
    }
    fn add(&mut self, ranked: &[u64], target: &[u64]) {
        match self {
            Self::Full(m) => m.add(ranked, target),
            Self::Beyond(m, _) => m.add(ranked, target),
            Self::None => {}
            Self::Mrr(m) => m.add(ranked, target),
            Self::Ndcg(m) => m.add(ranked, target),
            Self::HitRate(m) => m.add(ranked, target),
        }
    }
    fn merge(&mut self, other: Self) {
        match (self, other) {
            (Self::Full(a), Self::Full(b)) => a.merge(b),
            (Self::Beyond(a, _), Self::Beyond(b, _)) => a.merge(b),
            (Self::None, Self::None) => {}
            (Self::Mrr(a), Self::Mrr(b)) => a.merge(b),
            (Self::Ndcg(a), Self::Ndcg(b)) => a.merge(b),
            (Self::HitRate(a), Self::HitRate(b)) => a.merge(b),
            _ => unreachable!("one objective per evaluation"),
        }
    }
    fn result(&self) -> f64 {
        match self {
            Self::Beyond(m, i) => m.scores()[*i],
            Self::Full(_) | Self::None => 0.,
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
    /// changes the index. Use run_with_catalog to reuse catalog statistics across trials.
    pub fn run(
        &self,
        index: &VMISIndex,
        data: &EvaluationData,
        model: &ModelConfig,
        business: bool,
        objective: Objective,
    ) -> anyhow::Result<RunResult> {
        self.run_with_catalog(
            index,
            data,
            model,
            business,
            objective,
            Self::objective_catalog(index, objective),
        )
    }
    /// Build immutable objective statistics once, outside the HPO trial loop.
    pub fn objective_catalog(
        index: &VMISIndex,
        objective: Objective,
    ) -> Option<std::sync::Arc<crate::metrics::beyond_accuracy::Catalog>> {
        use crate::metrics::beyond_accuracy::Catalog;
        match objective.metric() {
            RankingMetric::Mrr | RankingMetric::Ndcg | RankingMetric::HitRate => None,
            RankingMetric::IntraListDiversity => Some(std::sync::Arc::new(Catalog::from_sessions(
                index.training_sessions(),
            ))),
            RankingMetric::Gini | RankingMetric::LongTailShare => Some(std::sync::Arc::new(
                Catalog::from_items(index.training_items()),
            )),
            _ => Some(std::sync::Arc::new(Catalog::default())),
        }
    }
    pub fn run_with_catalog(
        &self,
        index: &VMISIndex,
        data: &EvaluationData,
        model: &ModelConfig,
        business: bool,
        objective: Objective,
        catalog: Option<std::sync::Arc<crate::metrics::beyond_accuracy::Catalog>>,
    ) -> anyhow::Result<RunResult> {
        let (predictions, metric) = self.execute(
            index,
            data,
            model,
            business,
            objective.cutoff(),
            &|| Metric::new(objective, catalog.clone()),
            None,
        )?;
        Ok(RunResult {
            predictions,
            objective: metric.result(),
        })
    }
    pub fn quality(
        &self,
        index: &VMISIndex,
        data: &EvaluationData,
        model: &ModelConfig,
        business: bool,
        cutoff: usize,
    ) -> anyhow::Result<(usize, [f64; 15])> {
        let template = crate::metrics::evaluation_reporter::EvaluationReporter::from_sessions(
            index.training_sessions(),
            cutoff,
        );
        let (n, metric) = self.execute(
            index,
            data,
            model,
            business,
            cutoff,
            &|| Metric::Full(template.empty(cutoff)),
            None,
        )?;
        let Metric::Full(reporter) = metric else {
            unreachable!()
        };
        Ok((n, reporter.all_scores()))
    }
    /// Latencies require a dedicated single-worker executor. Throughput retains no samples.
    pub fn predictions(
        &self,
        index: &VMISIndex,
        data: &EvaluationData,
        model: &ModelConfig,
        business: bool,
        cutoff: usize,
        latencies: Option<&Mutex<Vec<u64>>>,
    ) -> anyhow::Result<usize> {
        anyhow::ensure!(
            latencies.is_none() || self.workers() == 1,
            "latency measurement requires one worker"
        );
        Ok(self
            .execute(
                index,
                data,
                model,
                business,
                cutoff,
                &|| Metric::None,
                latencies,
            )?
            .0)
    }
    fn execute(
        &self,
        index: &VMISIndex,
        data: &EvaluationData,
        model: &ModelConfig,
        business: bool,
        cutoff: usize,
        make: &(impl Fn() -> Metric + Sync),
        latencies: Option<&Mutex<Vec<u64>>>,
    ) -> anyhow::Result<(usize, Metric)> {
        let view = ScoringView::new(index, model).map_err(anyhow::Error::msg)?;
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
                    let mut samples = latencies.map(|s| s.lock().unwrap());
                    let mut metric = make();
                    let mut predictions = 0;
                    for events in sessions {
                        items.clear();
                        items.extend(events.iter().map(|e| e.0));
                        for state in 1..items.len() {
                            let current =
                                &items[state.saturating_sub(model.max_items_in_session)..state];
                            let started = latencies.map(|_| std::time::Instant::now());
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
                            if let Some(samples) = &mut samples {
                                samples.push(started.unwrap().elapsed().as_nanos() as u64);
                            }
                            if matches!(metric, Metric::None) {
                                std::hint::black_box(recommendations);
                                predictions += 1;
                                continue;
                            }
                            ranked.clear();
                            ranked.extend(recommendations.into_sorted_vec().iter().map(|s| s.id));
                            metric.add(ranked, &items[state..]);
                            predictions += 1;
                        }
                    }
                    (predictions, metric)
                })
                .reduce(
                    || (0, make()),
                    |(n, mut a), (m, b)| {
                        a.merge(b);
                        (n + m, a)
                    },
                )
        });
        Ok((predictions, metric))
    }
}
