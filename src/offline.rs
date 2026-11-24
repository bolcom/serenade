//! Shared offline prediction points, metrics and algorithm selection.
pub mod execution;
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
    pub fn prediction_count(&self) -> usize {
        self.sessions
            .iter()
            .map(|s| s.len().saturating_sub(1))
            .sum()
    }
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
    let result = evaluate_at_cutoff(
        index,
        data,
        model,
        business,
        std::num::NonZeroUsize::new(20).unwrap(),
    );
    EvaluationResult {
        algorithm: result.algorithm,
        predictions: result.predictions,
        mrr_at_20: result.mrr,
        ndcg_at_20: result.ndcg,
        hitrate_at_20: result.hitrate,
        coverage_at_20: result.coverage,
        duration_seconds: result.duration_seconds,
        prediction_mean_us: result.prediction_mean_us,
        prediction_p95_us: result.prediction_p95_us,
    }
}

/// Ranking metrics use the existing definitions at `cutoff`; serving length is independent.
#[derive(Debug, Serialize)]
pub struct RankingEvaluationResult {
    pub algorithm: String,
    pub predictions: usize,
    pub cutoff: usize,
    pub mrr: f64,
    pub ndcg: f64,
    pub hitrate: f64,
    pub coverage: f64,
    pub duration_seconds: f64,
    pub prediction_mean_us: f64,
    pub prediction_p95_us: f64,
}

impl RankingEvaluationResult {
    pub fn score(&self, objective: crate::metrics::objective::Objective) -> f64 {
        use crate::metrics::objective::RankingMetric;
        assert_eq!(
            self.cutoff,
            objective.cutoff(),
            "objective and evaluation cutoff must match"
        );
        match objective.metric() {
            RankingMetric::Mrr => self.mrr,
            RankingMetric::Ndcg => self.ndcg,
            RankingMetric::HitRate => self.hitrate,
        }
    }
}

pub fn evaluate_at_cutoff(
    index: &VMISIndex,
    data: &EvaluationData,
    model: &ModelConfig,
    business: bool,
    cutoff: std::num::NonZeroUsize,
) -> RankingEvaluationResult {
    let started = Instant::now();
    let mut reporter = EvaluationReporter::from_items(index.training_items(), cutoff.get());
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
                cutoff.get(),
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
    RankingEvaluationResult {
        algorithm: model.knn.algorithm.name().to_owned(),
        predictions: latencies.len(),
        cutoff: cutoff.get(),
        mrr,
        ndcg,
        hitrate,
        coverage,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn cutoff_controls_predictions_and_metrics_and_preserves_default() {
        let path = std::env::temp_dir().join(format!("serenade-cutoff-{}.tsv", std::process::id()));
        let mut training = String::from("SessionId\tItemId\tTime\n");
        for item in 1..=30 {
            training.push_str(&format!("1\t{item}\t{item}\n"));
        }
        std::fs::write(&path, training).unwrap();
        let knn = KnnConfig::for_algorithm(Algorithm::Sknn);
        let index = VMISIndex::load_for_model(path.to_str().unwrap(), 10, 1.0, &knn).unwrap();
        let model = ModelConfig {
            knn,
            m_most_recent_sessions: 10,
            neighborhood_size_k: 10,
            num_items_to_recommend: 1,
            max_items_in_session: 2,
            idf_weighting: 1.0,
        };
        let recommendations =
            vmisknn::predict_with_model(&index, &[1], 10, 10, 30, false, &model.knn, 100.0)
                .into_sorted_vec();
        assert_eq!(recommendations.len(), 29);
        for rank in [2, 21] {
            let target = recommendations[rank - 1].id;
            let data = EvaluationData {
                sessions: vec![vec![(1, 100.0), (target, 101.0)]],
            };
            let below = evaluate_at_cutoff(
                &index,
                &data,
                &model,
                false,
                NonZeroUsize::new(rank - 1).unwrap(),
            );
            let included = evaluate_at_cutoff(
                &index,
                &data,
                &model,
                false,
                NonZeroUsize::new(rank).unwrap(),
            );
            assert_eq!((below.mrr, below.ndcg, below.hitrate), (0.0, 0.0, 0.0));
            assert_eq!(included.mrr, 1.0 / rank as f64);
            assert_eq!(included.hitrate, 1.0);
            // The established NDCG discount is 1/log2(rank) after rank one.
            assert_eq!(included.ndcg, 1.0 / (rank as f64).log2());
            let old = evaluate(&index, &data, &model, false);
            let default =
                evaluate_at_cutoff(&index, &data, &model, false, NonZeroUsize::new(20).unwrap());
            assert_eq!(
                old.mrr_at_20,
                if rank <= 20 { 1.0 / rank as f64 } else { 0.0 }
            );
            assert_eq!(
                [
                    old.mrr_at_20,
                    old.ndcg_at_20,
                    old.hitrate_at_20,
                    old.coverage_at_20
                ],
                [default.mrr, default.ndcg, default.hitrate, default.coverage]
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}
