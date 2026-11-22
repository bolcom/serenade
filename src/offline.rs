//! Shared offline prediction points, metrics and algorithm selection.
use crate::{
    config::ModelConfig,
    io,
    metrics::evaluation_reporter::EvaluationReporter,
    vmisknn::{
        self,
        algorithms::{Algorithm, KnnConfig},
        vmis_index::VMISIndex,
    },
};
use serde::Serialize;
use std::time::Instant;

pub struct EvaluationData {
    sessions: Vec<Vec<(u64, f64)>>,
}
impl EvaluationData {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let mut sessions: Vec<_> = io::try_read_test_data_timed(path)?.into_iter().collect();
        sessions.sort_by_key(|(id, _)| *id);
        anyhow::ensure!(
            sessions.iter().any(|(_, events)| events.len() > 1),
            "dataset has no prediction points: {}",
            path
        );
        Ok(Self {
            sessions: sessions.into_iter().map(|(_, events)| events).collect(),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct EvaluationResult {
    pub algorithm: String,
    pub predictions: usize,
    pub mrr_at_20: f64,
    pub ndcg_at_20: f64,
    pub hitrate_at_20: f64,
    pub coverage_at_20: f64,
    pub duration_seconds: f64,
    pub prediction_mean_us: f64,
    pub prediction_p95_us: f64,
}

pub fn evaluate(
    index: &VMISIndex,
    data: &EvaluationData,
    model: &ModelConfig,
    business: bool,
) -> EvaluationResult {
    let started = Instant::now();
    let mut reporter = EvaluationReporter::from_items(index.training_items(), 20);
    let mut latencies = Vec::new();
    for events in &data.sessions {
        let items: Vec<_> = events.iter().map(|e| e.0).collect();
        for state in 1..items.len() {
            let session = &items[state.saturating_sub(model.max_items_in_session)..state];
            let start = Instant::now();
            let recommendations = vmisknn::predict_with_model(
                index,
                session,
                model.neighborhood_size_k,
                model.m_most_recent_sessions,
                20,
                business,
                &model.knn,
                events[state - 1].1,
            )
            .into_sorted_vec();
            latencies.push(start.elapsed().as_secs_f64() * 1e6);
            let recommended: Vec<_> = recommendations.iter().map(|s| s.id).collect();
            reporter.add(&recommended, &items[state..]);
        }
    }
    let [mrr, ndcg, hitrate, coverage] = reporter.scores();
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    latencies.sort_by(f64::total_cmp);
    EvaluationResult {
        algorithm: model.knn.algorithm.name().to_owned(),
        predictions: latencies.len(),
        mrr_at_20: mrr,
        ndcg_at_20: ndcg,
        hitrate_at_20: hitrate,
        coverage_at_20: coverage,
        duration_seconds: started.elapsed().as_secs_f64(),
        prediction_mean_us: mean,
        prediction_p95_us: latencies[(latencies.len() - 1) * 95 / 100],
    }
}

pub fn selected_models(
    base: &ModelConfig,
    selection: Option<&str>,
) -> anyhow::Result<Vec<ModelConfig>> {
    let algorithms = match selection {
        Some("all") => Algorithm::ALL.to_vec(),
        Some(name) => vec![name.parse::<Algorithm>().map_err(anyhow::Error::msg)?],
        None => vec![base.knn.algorithm],
    };
    Ok(algorithms
        .into_iter()
        .map(|algorithm| {
            let mut model = base.clone();
            if algorithm != base.knn.algorithm {
                model.knn = KnnConfig::for_algorithm(algorithm);
            }
            model
        })
        .collect())
}

pub fn print_table(results: &[EvaluationResult]) {
    println!("algorithm\tMRR@20\tNDCG@20\tHitRate@20\tCoverage@20\tseconds\tmean_us\tp95_us");
    for r in results {
        println!(
            "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.3}\t{:.1}\t{:.1}",
            r.algorithm,
            r.mrr_at_20,
            r.ndcg_at_20,
            r.hitrate_at_20,
            r.coverage_at_20,
            r.duration_seconds,
            r.prediction_mean_us,
            r.prediction_p95_us
        );
    }
}
