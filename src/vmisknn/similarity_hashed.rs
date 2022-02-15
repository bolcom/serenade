extern crate hashbrown;

use std::collections::BinaryHeap;

use hashbrown::hash_map::DefaultHashBuilder as RandomState;
use hashbrown::HashSet;

use crate::vmisknn::SessionScore;

pub trait SimilarityComputationHash {
    fn items_for_session(&self, session: &u32) -> &HashSet<u64, RandomState>;

    fn idf(&self, item_id: &u64) -> f64;

    fn find_neighbors(
        &self,
        evolving_session: &[u64],
        k: usize,
        m: usize,
    ) -> BinaryHeap<SessionScore>;
}

pub(crate) fn idf(num_sessions_total: usize, num_session_with_item: usize) -> f64 {
    (num_sessions_total as f64 / num_session_with_item as f64).ln()
}


#[cfg(test)]
mod sessiontime_test {
    use crate::vmisknn::SessionTime;

    use super::*;

    #[test]
    fn handle_reverse_ordering_sessiontime() {
        let largest = SessionTime::new(123, 5000);
        let middle = SessionTime::new(234, 100);
        let smallest = SessionTime::new(543, 1);
        let items = vec![largest, smallest, middle];

        let how_many = 2;
        let mut top_items: BinaryHeap<SessionTime> = BinaryHeap::with_capacity(how_many);

        for sessiontime in items.into_iter() {
            if top_items.len() < how_many {
                top_items.push(sessiontime);
            } else {
                let mut reverse_top = top_items.peek_mut().unwrap();
                if sessiontime.time > reverse_top.time {
                    // ordering is reverse thus, item score is larger than the reverse top.
                    *reverse_top = sessiontime;
                }
            }
        }
        // the results are the top `how_many` in reverse order
        assert_eq!(234, top_items.pop().unwrap().session_id);
        assert_eq!(123, top_items.pop().unwrap().session_id);
    }
}