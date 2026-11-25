//! A borrowed scoring view with item statistics shared across models.
use super::{
    similarity_indexed::SimilarityComputationNew,
    vmis_index::{ProductAttributes, VMISIndex},
    NeighborScratch, SessionScore,
};
use crate::config::ModelConfig;
use std::collections::BinaryHeap;

pub(crate) struct ScoringView<'a> {
    index: &'a VMISIndex,
    weight: f64,
    idf: &'a hashbrown::HashMap<u64, f64>,
}
impl<'a> ScoringView<'a> {
    /// Borrow the existing IDF representation or cache the alternate formula.
    /// Model weights are applied to unit statistics, even for older weighted indexes.
    pub(crate) fn new(index: &'a VMISIndex, model: &ModelConfig) -> Result<Self, String> {
        model.knn.validate()?;
        if !model.idf_weighting.is_finite() {
            return Err("IDF weighting must be finite".into());
        }
        if index
            .candidate_capacity
            .is_some_and(|m| model.m_most_recent_sessions > m)
        {
            return Err("Requested m exceeds fixed index retrieval capacity".into());
        }
        if !index.chronological_events
            && (model.knn.positional() || model.knn.needs_reference_idf())
        {
            return Err(
                "This index lacks chronological raw events; rebuild from training CSV".into(),
            );
        }
        let reference = model.knn.needs_reference_idf();
        let legacy = model.knn.algorithm == super::algorithms::Algorithm::Vmis;
        let idf = if reference && !index.reference_idf {
            index
                .reference_idf_scores
                .get_or_init(|| index.compute_idf(true))
        } else if legacy
            && index.chronological_events
            && (index.reference_idf || !index.unit_legacy_idf)
        {
            index.legacy_idf.get_or_init(|| index.compute_idf(false))
        } else {
            &index.item_to_idf_score
        };
        Ok(Self {
            index,
            idf,
            weight: if legacy && index.chronological_events {
                model.idf_weighting
            } else {
                1.0
            },
        })
    }
}
impl SimilarityComputationNew for ScoringView<'_> {
    fn items_for_session(&self, id: &u32) -> &[u64] {
        self.index.items_for_session(id)
    }
    fn session_timestamp(&self, id: u32) -> u32 {
        self.index.session_timestamp(id)
    }
    fn idf(&self, item: &u64) -> f64 {
        self.idf.get(item).copied().unwrap_or(0.0) * self.weight
    }
    fn find_neighbors(&self, current: &[u64], k: usize, m: usize) -> BinaryHeap<SessionScore> {
        self.index.find_neighbors(current, k, m)
    }
    fn find_neighbors_with_scratch(
        &self,
        current: &[u64],
        k: usize,
        m: usize,
        scratch: &mut NeighborScratch,
    ) -> BinaryHeap<SessionScore> {
        self.index
            .find_neighbors_with_scratch(current, k, m, scratch)
    }
    fn find_attributes(&self, id: &u64) -> Option<&ProductAttributes> {
        self.index.find_attributes(id)
    }
}
