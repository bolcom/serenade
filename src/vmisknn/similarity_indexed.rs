extern crate hashbrown;

use crate::vmisknn::offline_index::ProductAttributes;
use crate::vmisknn::SessionScore;
use std::collections::BinaryHeap;


pub trait SimilarityComputationNew {
    fn items_for_session(&self, session_idx: &u32) -> &[u64];

    fn idf(&self, item_id: &u64) -> f64;

    /// find neighboring sessions for the given evolving_session.
    /// param m select the 'm' most recent historical sessions
    /// param k defines the top 'k' scored historical sessions out of the 'm' historical sessions.
    fn find_neighbors(
        &self,
        evolving_session: &[u64],
        k: usize,
        m: usize,
    ) -> BinaryHeap<SessionScore>;

    fn find_attributes(&self, item_id: &u64) -> Option<&ProductAttributes>;
}
