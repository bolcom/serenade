use super::beyond_accuracy::{BeyondAccuracy, Catalog, NAMES};
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
use std::sync::Arc;

pub struct EvaluationReporter {
    n: i32,
    cutoff: usize,
    pub beyond: BeyondAccuracy,
    mrr: Mrr,
    ndcg: Ndcg,
    hitrate: HitRate,
    popularity: Popularity,
    precision: Precision,
    coverage: Coverage,
    recall: Recall,
    f1: F1score,
}

impl EvaluationReporter {
    pub fn empty(&self, cutoff: usize) -> Self {
        Self {
            n: 0,
            cutoff,
            beyond: self.beyond.empty(cutoff),
            mrr: Mrr::new(cutoff),
            ndcg: Ndcg::new(cutoff),
            hitrate: HitRate::new(cutoff),
            popularity: self.popularity.empty(),
            precision: Precision::new(cutoff),
            coverage: self.coverage.empty(),
            recall: Recall::new(cutoff),
            f1: F1score::new(cutoff),
        }
    }
    pub fn merge(&mut self, other: Self) {
        self.n += other.n;
        self.beyond.merge(other.beyond);
        self.mrr.merge(other.mrr);
        self.ndcg.merge(other.ndcg);
        self.hitrate.merge(other.hitrate);
        self.popularity.merge(other.popularity);
        self.precision.merge(other.precision);
        self.coverage.merge(other.coverage);
        self.recall.merge(other.recall);
        self.f1.merge(other.f1);
    }

    /// Full-precision values in MRR, NDCG, hit rate, popularity, precision,
    /// coverage, recall, F1, then beyond_accuracy::NAMES order.
    /// Formatting is the caller's responsibility.
    pub fn all_scores(&self) -> [f64; 15] {
        let mut scores = [0.; 15];
        scores[..8].copy_from_slice(&[
            self.mrr.result(),
            self.ndcg.result(),
            self.hitrate.result(),
            self.popularity.result(),
            self.precision.result(),
            self.coverage.result(),
            self.recall.result(),
            self.f1.result(),
        ]);
        scores[8..].copy_from_slice(&self.beyond.scores());
        scores
    }

    pub fn scores(&self) -> [f64; 4] {
        [
            self.mrr.result(),
            self.ndcg.result(),
            self.hitrate.result(),
            self.coverage.result(),
        ]
    }
}

impl EvaluationReporter {
    pub fn new(
        training_df: &[(TrainingSessionId, ItemId, Time)],
        length: usize,
    ) -> EvaluationReporter {
        let mut sessions = std::collections::BTreeMap::<u32, Vec<u64>>::new();
        for &(session, item, _) in training_df {
            sessions.entry(session).or_default().push(item);
        }
        Self::from_sessions(sessions.values().map(|s| s.as_slice()), length)
    }

    pub fn from_sessions<'a>(
        sessions: impl Iterator<Item = &'a [u64]> + Clone,
        length: usize,
    ) -> Self {
        let catalog = Arc::new(Catalog::from_sessions(sessions.clone()));
        Self::with_catalog(sessions.flat_map(|s| s.iter().copied()), catalog, length)
    }

    /// Without session membership, intra-list diversity is zero. Prefer from_sessions.
    pub fn from_items(items: impl Iterator<Item = u64> + Clone, length: usize) -> Self {
        let catalog = Arc::new(Catalog::from_items(items.clone()));
        Self::with_catalog(items, catalog, length)
    }
    fn with_catalog(
        items: impl Iterator<Item = u64> + Clone,
        catalog: Arc<Catalog>,
        length: usize,
    ) -> Self {
        let n = 0;
        let beyond = BeyondAccuracy::new(catalog, length);
        let mrr = Mrr::new(length);
        let ndcg = Ndcg::new(length);
        let hitrate = HitRate::new(length);
        let popularity = Popularity::from_items(items.clone(), length);
        let precision = Precision::new(length);
        let coverage = Coverage::from_items(items, length);
        let recall = Recall::new(length);
        let f1 = F1score::new(length);

        EvaluationReporter {
            n,
            cutoff: length,
            beyond,
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
        self.n += 1;
        self.beyond.add(recommendations, next_items);
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
        let n = format!("{}", self.n);
        let mrr_score = format!("{:.4}", self.mrr.result());
        let ndcg_score = format!("{:.4}", self.ndcg.result());
        let hitrate_score = format!("{:.4}", self.hitrate.result());
        let popularity_score = format!("{:.4}", self.popularity.result());
        let precision_score = format!("{:.4}", self.precision.result());
        let coverage_score = format!("{:.4}", self.coverage.result());
        let recall_score = format!("{:.4}", self.recall.result());
        let f1_score = format!("{:.4}", self.f1.result());
        let legacy = format!(
            "{},{},{},{},{},{},{},{},{}",
            n,
            mrr_score,
            ndcg_score,
            hitrate_score,
            popularity_score,
            precision_score,
            coverage_score,
            recall_score,
            f1_score
        );
        format!(
            "{},{}",
            legacy,
            self.beyond
                .scores()
                .iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    pub fn get_name(&self) -> String {
        let n_name = "qty_evaluations".to_string();
        let mrr_name = self.mrr.get_name();
        let ndcg_name = self.ndcg.get_name();
        let hitrate_name = self.hitrate.get_name();
        let popularity_name = self.popularity.get_name();
        let precision_name = self.precision.get_name();
        let coverage_name = self.coverage.get_name();
        let recall_name = self.recall.get_name();
        let f1_name = self.f1.get_name();
        let legacy = format!(
            "{},{},{},{},{},{},{},{},{}",
            n_name,
            mrr_name,
            ndcg_name,
            hitrate_name,
            popularity_name,
            precision_name,
            coverage_name,
            recall_name,
            f1_name
        );
        format!(
            "{},{}",
            legacy,
            NAMES
                .iter()
                .map(|n| format!("{n}@{}", self.cutoff))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}
