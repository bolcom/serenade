use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};

use crate::vmisknn::similarity_indexed::SimilarityComputationNew;
use crate::vmisknn::vmis_index::ProductAttributes;

pub mod algorithms;
pub mod similarity_indexed;
pub mod vmis_index;

#[derive(PartialEq, Debug)]
pub struct SessionScore {
    pub id: u32,
    pub score: f64,
}

impl SessionScore {
    fn new(id: u32, score: f64) -> Self {
        SessionScore { id, score }
    }
}

impl Eq for SessionScore {}

impl Ord for SessionScore {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Less) => Ordering::Greater,
            Some(Ordering::Greater) => Ordering::Less,
            _ => Ordering::Equal,
        }
    }
}

impl PartialOrd for SessionScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(PartialEq, Debug)]
pub struct ItemScore {
    pub id: u64,
    pub score: f64,
}

impl ItemScore {
    fn new(id: u64, score: f64) -> Self {
        ItemScore { id, score }
    }
}

impl Eq for ItemScore {}

impl Ord for ItemScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // reverse order by score
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Less) => Ordering::Greater,
            Some(Ordering::Greater) => Ordering::Less,
            _ => Ordering::Equal,
        }
    }
}

impl PartialOrd for ItemScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Eq, Debug)]
pub struct SessionTime {
    pub session_id: u32,
    pub time: u32,
}

impl SessionTime {
    pub fn new(session_id: u32, time: u32) -> Self {
        SessionTime { session_id, time }
    }
}

impl Ord for SessionTime {
    fn cmp(&self, other: &Self) -> Ordering {
        // reverse order by time
        other.time.cmp(&self.time)
    }
}

impl PartialOrd for SessionTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SessionTime {
    fn eq(&self, other: &Self) -> bool {
        // == is defined as being based on the contents of an object.
        self.session_id == other.session_id
    }
}

fn linear_score(pos: usize) -> f64 {
    if pos < 100 {
        1.0 - (0.1 * pos as f64)
    } else {
        0.0
    }
}

/// Predict recommendations. Empty histories or zero k, m, or result limits return an empty heap.
pub fn predict<I: SimilarityComputationNew + Send + Sync>(
    index: &I,
    evolving_session: &[u64],
    k: usize,
    m: usize,
    how_many: usize,
    enable_business_logic: bool,
) -> BinaryHeap<ItemScore> {
    predict_with_scratch(
        index,
        evolving_session,
        k,
        m,
        how_many,
        enable_business_logic,
        &mut PredictionScratch::default(),
    )
}

/// Buffers exclusively owned by one caller or prediction worker.
#[derive(Default)]
pub struct PredictionScratch {
    pub(crate) neighbors: NeighborScratch,
    item_scores: HashMap<u64, f64>,
}

#[derive(Default)]
pub struct NeighborScratch {
    pub(crate) timestamps: dary_heap::OctonaryHeap<SessionTime>,
    pub(crate) similarities: HashMap<u32, f64>,
    pub(crate) items: HashMap<u64, bool>,
}

pub fn predict_with_scratch<I: SimilarityComputationNew + Send + Sync>(
    index: &I,
    evolving_session: &[u64],
    k: usize,
    m: usize,
    how_many: usize,
    enable_business_logic: bool,
    scratch: &mut PredictionScratch,
) -> BinaryHeap<ItemScore> {
    if evolving_session.is_empty() || k == 0 || m == 0 || how_many == 0 {
        return BinaryHeap::new();
    }
    let neighbors =
        index.find_neighbors_with_scratch(evolving_session, k, m, &mut scratch.neighbors);

    let item_scores = &mut scratch.item_scores;
    item_scores.clear();
    // Match the original scoring capacity; reused buffers keep their allocation.
    item_scores.reserve(1000);

    for scored_session in neighbors.into_iter() {
        let training_item_ids: &[u64] = index.items_for_session(&scored_session.id);

        let (first_match_index, _) = evolving_session
            .iter()
            .rev()
            .enumerate()
            .find(|(_, item_id)| training_item_ids.contains(*item_id))
            .unwrap();

        let first_match_pos = first_match_index + 1;

        let session_weight = linear_score(first_match_pos);

        // Historical sequences preserve repeated visits for positional algorithms.
        // VMIS scores each distinct item only once per neighboring session.
        // Short sessions avoid a per-neighbor allocation. Bound the quadratic
        // scan; imported indexes can contain much longer sessions.
        let mut seen =
            (training_item_ids.len() > 32).then(|| HashSet::with_capacity(training_item_ids.len()));
        for (position, item_id) in training_item_ids.iter().enumerate() {
            let unique = match &mut seen {
                Some(seen) => seen.insert(*item_id),
                None => !training_item_ids[..position].contains(item_id),
            };
            if !unique {
                continue;
            }
            let item_idf = index.idf(item_id);
            if item_idf > 0.0 {
                *item_scores.entry(*item_id).or_insert(0.0) +=
                    session_weight * item_idf * scored_session.score;
            } else {
                *item_scores.entry(*item_id).or_insert(0.0) +=
                    session_weight * scored_session.score;
            }
        }
    }

    rank_items(
        index,
        evolving_session,
        item_scores,
        how_many,
        enable_business_logic,
    )
}

/// Algorithm selection shares retrieval, filtering and serving output with legacy VMIS.
pub fn predict_with_model(
    index: &vmis_index::VMISIndex,
    current: &[u64],
    k: usize,
    m: usize,
    how_many: usize,
    enable_business_logic: bool,
    model: &algorithms::KnnConfig,
    timestamp: f64,
) -> BinaryHeap<ItemScore> {
    if model.algorithm == algorithms::Algorithm::Vmis {
        return predict(index, current, k, m, how_many, enable_business_logic);
    }
    if current.is_empty() || k == 0 || m == 0 || how_many == 0 {
        return BinaryHeap::new();
    }
    let candidates = index.candidate_sessions(current, m);
    let neighbors = algorithms::score_neighbors(index, current, &candidates, timestamp, k, model);
    let mut scores = algorithms::score_items(index, current, &neighbors, model);
    if model.normalize {
        let max = scores.values().copied().fold(0.0_f64, f64::max);
        if max > 0.0 {
            for score in scores.values_mut() {
                *score /= max;
            }
        }
    }
    rank_items(index, current, &mut scores, how_many, enable_business_logic)
}

fn rank_items<I: SimilarityComputationNew>(
    index: &I,
    evolving_session: &[u64],
    item_scores: &mut HashMap<u64, f64>,
    how_many: usize,
    enable_business_logic: bool,
) -> BinaryHeap<ItemScore> {
    // Remove most recent item if it has been scored as well
    let most_recent_item = *evolving_session.last().unwrap();
    if let Entry::Occupied(entry) = item_scores.entry(most_recent_item) {
        entry.remove_entry();
    }

    fn passes_business_rules(
        current_item_attribs: Option<&ProductAttributes>,
        reco_item_attribs: Option<&ProductAttributes>,
    ) -> bool {
        if reco_item_attribs.is_none() {
            return false;
        }
        let reco_attribs = reco_item_attribs.unwrap();
        if reco_attribs.is_for_sale {
            if reco_attribs.is_adult {
                if let Some(current_attribs) = current_item_attribs {
                    return current_attribs.is_adult;
                } else {
                    return false;
                }
            } else {
                return true;
            }
        }
        false
    }

    // Return the proper amount of recommendations and filter them using business rules.
    let mut top_items: BinaryHeap<ItemScore> = BinaryHeap::with_capacity(how_many);
    let current_item_attribs: Option<&ProductAttributes> = index.find_attributes(&most_recent_item);
    for (reco_item_id, reco_item_score) in item_scores.drain() {
        let scored_item = ItemScore::new(reco_item_id, reco_item_score);

        if top_items.len() < how_many {
            if enable_business_logic {
                let reco_item_attribs: Option<&ProductAttributes> =
                    index.find_attributes(&reco_item_id);
                if passes_business_rules(current_item_attribs, reco_item_attribs) {
                    top_items.push(scored_item);
                }
            } else {
                top_items.push(scored_item);
            }
        } else {
            let mut bottom = top_items.peek_mut().unwrap();
            if scored_item.score > bottom.score {
                if enable_business_logic {
                    let reco_item_attribs = index.find_attributes(&reco_item_id);
                    if passes_business_rules(current_item_attribs, reco_item_attribs) {
                        *bottom = scored_item;
                    }
                } else {
                    *bottom = scored_item;
                }
            }
        }
    }

    top_items
}

#[cfg(test)]
mod vmisknn_test {

    use crate::vmisknn::vmis_index::prepare_hashmap;
    use crate::vmisknn::vmis_index::VMISIndex;

    use super::*;
    use dary_heap::OctonaryHeap;

    #[test]
    fn empty_and_zero_limits_do_not_access_index() {
        struct UnusedIndex;
        impl SimilarityComputationNew for UnusedIndex {
            fn items_for_session(&self, _: &u32) -> &[u64] {
                panic!("unexpected access")
            }
            fn idf(&self, _: &u64) -> f64 {
                panic!("unexpected access")
            }
            fn find_neighbors(&self, _: &[u64], _: usize, _: usize) -> BinaryHeap<SessionScore> {
                panic!("unexpected access")
            }
            fn find_attributes(&self, _: &u64) -> Option<&ProductAttributes> {
                panic!("unexpected access")
            }
        }
        for (session, k, m, count) in [
            (&[][..], 1, 1, 1),
            (&[1][..], 0, 1, 1),
            (&[1][..], 1, 0, 1),
            (&[1][..], 1, 1, 0),
        ] {
            assert!(predict(&UnusedIndex, session, k, m, count, false).is_empty());
        }
    }

    #[test]
    fn should_train_and_predict() {
        let n_most_recent_sessions = 5;
        let k = 500;
        let m = 500;
        let how_many = 20;
        let enable_business_logic = false;
        let idf_weighting = 1.0;

        // 7 training data records
        let session1_items_ids: Vec<u64> = vec![920006, 920005, 920004];
        let session1_max_timestamp: u32 = 1;
        let session2_items_ids: Vec<u64> = vec![920005, 920004, 920003, 920002];
        let session2_max_timestamp: u32 = 1;
        let historical_sessions_train = vec![session1_items_ids, session2_items_ids];
        let historical_sessions_max_time_stamp =
            vec![session1_max_timestamp, session2_max_timestamp];

        let raw_events: Vec<_> = historical_sessions_train
            .iter()
            .enumerate()
            .flat_map(|(id, items)| items.iter().map(move |&item| (id as u32, item, 1)))
            .collect();
        let training_data_stats =
            crate::dataframeutils::determine_training_data_statistics("test", &raw_events);

        let (item_to_top_sessions_ordered, item_to_idf_score, item_to_product_attributes) =
            prepare_hashmap(
                &historical_sessions_train,
                &historical_sessions_max_time_stamp,
                n_most_recent_sessions,
                training_data_stats.qty_events_p99_5 as usize,
                idf_weighting,
            );

        let vmis_index = VMISIndex {
            chronological_events: true,
            candidate_capacity: Some(n_most_recent_sessions),
            reference_idf: false,
            item_to_top_sessions_ordered: item_to_top_sessions_ordered,
            session_to_max_time_stamp: historical_sessions_max_time_stamp,
            item_to_idf_score: item_to_idf_score,
            session_to_items_sorted: historical_sessions_train,
            training_data_stats: training_data_stats,
            item_to_product_attributes: item_to_product_attributes,
        };

        let session_items = vec![920005];

        let recommendations = predict(
            &vmis_index,
            &session_items,
            k,
            m,
            how_many,
            enable_business_logic,
        );

        // we expect the four other item_ids to be recommended
        assert_eq!(4, recommendations.len());

        let recommended_items: Vec<u64> = recommendations
            .into_sorted_vec()
            .iter()
            .map(|scored| scored.id)
            .collect();

        // item_id: 920004 should have the highest score and thus be the first result.
        assert_eq!(920004, recommended_items[0]);
    }

    #[test]
    fn repeated_historical_visits_do_not_change_vmis_scores() {
        struct SequenceIndex(Vec<u64>);
        impl SimilarityComputationNew for SequenceIndex {
            fn items_for_session(&self, _: &u32) -> &[u64] {
                &self.0
            }
            fn idf(&self, _: &u64) -> f64 {
                2.0
            }
            fn find_neighbors(&self, _: &[u64], _: usize, _: usize) -> BinaryHeap<SessionScore> {
                vec![SessionScore::new(0, 0.75)].into_iter().collect()
            }
            fn find_attributes(&self, _: &u64) -> Option<&ProductAttributes> {
                None
            }
        }
        let scores = |items| {
            let mut scores = predict(&SequenceIndex(items), &[8, 1], 10, 10, 10, false).into_vec();
            scores.sort_by_key(|score| score.id);
            scores
        };
        let expected = scores(vec![1, 2, 3]);
        let repeated = scores(vec![2, 1, 2, 3, 1, 3]);
        assert_eq!(expected, repeated);
        assert_eq!(repeated.len(), 2);
        assert!((repeated[0].score - 1.35).abs() < 1e-12);
    }

    #[test]
    fn handle_reverse_ordering_itemscore() {
        let largest = ItemScore::new(123, 5000 as f64);
        let middle = ItemScore::new(234, 100 as f64);
        let smallest = ItemScore::new(543, 1 as f64);
        let items = vec![largest, smallest, middle];

        let how_many = 2;
        let mut top_items: BinaryHeap<ItemScore> = BinaryHeap::with_capacity(how_many);

        for itemscore in items.into_iter() {
            if top_items.len() < how_many {
                top_items.push(itemscore);
            } else {
                let mut reverse_top = top_items.peek_mut().unwrap();
                if itemscore.score > reverse_top.score {
                    // ordering is reverse thus, item score is larger than the reverse top.
                    *reverse_top = itemscore;
                }
            }
        }
        // the results are the top `how_many` in reverse order
        assert_eq!(234, top_items.pop().unwrap().id);
        assert_eq!(123, top_items.pop().unwrap().id);
    }

    #[test]
    fn handle_vector_sort_ordering_itemscore() {
        let largest = ItemScore::new(123, 5000 as f64);
        let middle = ItemScore::new(234, 100 as f64);
        let smallest = ItemScore::new(543, 1 as f64);

        let mut recommendations: BinaryHeap<ItemScore> = BinaryHeap::new();
        recommendations.push(largest);
        recommendations.push(smallest);
        recommendations.push(middle);

        let recommended_items: Vec<u64> = recommendations
            .into_sorted_vec()
            .iter()
            .map(|scored| scored.id)
            .collect();
        let expected_items: Vec<u64> = vec![123, 234, 543];
        assert_eq!(expected_items, recommended_items);
    }

    #[test]
    fn handle_reverse_ordering_sessionscore() {
        let largest = SessionScore::new(123, 5000 as f64);
        let middle = SessionScore::new(234, 100 as f64);
        let smallest = SessionScore::new(543, 1 as f64);
        let items = vec![largest, smallest, middle];

        let how_many = 2;
        let mut top_items: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(how_many);

        for sessionscore in items.into_iter() {
            if top_items.len() < how_many {
                top_items.push(sessionscore);
            } else {
                let mut reverse_top = top_items.peek_mut().unwrap();
                if sessionscore.score > reverse_top.score {
                    // ordering is reverse thus, item score is larger than the reverse top.
                    *reverse_top = sessionscore;
                }
            }
        }
        // the results are the top `how_many` in reverse order
        assert_eq!(234, top_items.pop().unwrap().id);
        assert_eq!(123, top_items.pop().unwrap().id);
    }

    #[test]
    fn handle_ordering_sessiontime() {
        let large = SessionTime::new(123, 5000);
        let middle = SessionTime::new(234, 499);
        let small = SessionTime::new(345, 99);
        let tiny = SessionTime::new(456, 1);
        let items = vec![large, small, tiny, middle];

        let how_many = 2;
        let mut heap_timestamps = OctonaryHeap::<SessionTime>::with_capacity(how_many);

        for session in items.into_iter() {
            if heap_timestamps.len() < how_many {
                heap_timestamps.push(session);
            } else {
                let mut bottom = heap_timestamps.peek_mut().unwrap();
                if session.time > bottom.time {
                    // Set new minimum timestamp
                    *bottom = session
                }
            }
        }

        // the results are the top `how_many` in reverse order
        assert_eq!(234, heap_timestamps.pop().unwrap().session_id);
        assert_eq!(123, heap_timestamps.pop().unwrap().session_id);
    }
}
