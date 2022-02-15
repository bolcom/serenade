use crate::io::{ItemId, Time, TrainingSessionId};
use crate::metrics::coverage::Coverage;
use crate::metrics::f1score::F1score;
use crate::metrics::hitrate::HitRate;
use crate::metrics::mrr::Mrr;
use crate::metrics::ndcg::Ndcg;
use crate::metrics::popularity::Popularity;
use crate::metrics::precision::Precision;
use crate::metrics::recall::Recall;
use crate::metrics::SessionMetric;

pub struct EvaluationReporter {
    mrr: Mrr,
    ndcg: Ndcg,
    hitrate: HitRate,
    popularity: Popularity,
    precision: Precision,
    coverage: Coverage,
    recall: Recall,
    f1: F1score,
}

impl EvaluationReporter {}

impl EvaluationReporter {
    pub fn new(
        training_df: &[(TrainingSessionId, ItemId, Time)],
        length: usize,
    ) -> EvaluationReporter {
        // If we define these metrics as mutable as we expect it to be,
        // the compilation reports a warning that these variables dont need to be mutable.
        let mrr = Mrr::new(length);
        let ndcg = Ndcg::new(length);
        let hitrate = HitRate::new(length);
        let popularity = Popularity::new(training_df, length);
        let precision = Precision::new(length);
        let coverage = Coverage::new(training_df, length);
        let recall = Recall::new(length);
        let f1 = F1score::new(length);

        EvaluationReporter {
            mrr,
            ndcg,
            hitrate,
            popularity,
            precision,
            coverage,
            recall,
            f1,
        }
    }
}

impl EvaluationReporter {
    pub fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        self.mrr.add(recommendations, next_items);
        self.ndcg.add(recommendations, next_items);
        self.hitrate.add(recommendations, next_items);
        self.popularity.add(recommendations, next_items);
        self.precision.add(recommendations, next_items);
        self.coverage.add(recommendations, next_items);
        self.recall.add(recommendations, next_items);
        self.f1.add(recommendations, next_items);
    }

    pub fn result(&self) -> String {
        let mrr_score = format!("{:.4}", self.mrr.result());
        let ndcg_score = format!("{:.4}", self.ndcg.result());
        let hitrate_score = format!("{:.4}", self.hitrate.result());
        let popularity_score = format!("{:.4}", self.popularity.result());
        let precision_score = format!("{:.4}", self.precision.result());
        let coverage_score = format!("{:.4}", self.coverage.result());
        let recall_score = format!("{:.4}", self.recall.result());
        let f1_score = format!("{:.4}", self.f1.result());
        format!(
            "{},{},{},{},{},{},{},{}",
            mrr_score,
            ndcg_score,
            hitrate_score,
            popularity_score,
            precision_score,
            coverage_score,
            recall_score,
            f1_score
        )
    }

    pub fn get_name(&self) -> String {
        let mrr_name = self.mrr.get_name();
        let ndcg_name = self.ndcg.get_name();
        let hitrate_name = self.hitrate.get_name();
        let popularity_name = self.popularity.get_name();
        let precision_name = self.precision.get_name();
        let coverage_name = self.coverage.get_name();
        let recall_name = self.recall.get_name();
        let f1_name = self.f1.get_name();
        format!(
            "{},{},{},{},{},{},{},{}",
            mrr_name,
            ndcg_name,
            hitrate_name,
            popularity_name,
            precision_name,
            coverage_name,
            recall_name,
            f1_name
        )
    }
}
