extern crate hashbrown;

use std::collections::BinaryHeap;

use hashbrown::{HashMap, HashSet};
//use std::time::Instant;
use hashbrown::hash_map::DefaultHashBuilder as RandomState;
use itertools::Itertools;

use crate::io::{ItemId, Time, TrainingSessionId};
use crate::vmisknn::{SessionScore, SessionTime};
use crate::vmisknn::similarity_hashed::{idf, SimilarityComputationHash};

pub struct VSkNNIndex {
    session_index: HashMap<TrainingSessionId, HashSet<ItemId>>,
    session_max_order: HashMap<TrainingSessionId, Time>,
    item_index: HashMap<ItemId, HashSet<TrainingSessionId>>,
    item_idfs: HashMap<ItemId, f64>,
}

impl SimilarityComputationHash for VSkNNIndex {
    fn items_for_session(&self, session: &TrainingSessionId) -> &HashSet<ItemId, RandomState> {
        &self.session_index[session]
    }

    fn idf(&self, item: &ItemId) -> f64 {
        self.item_idfs[item]
    }

    fn find_neighbors(
        &self,
        evolving_session: &[ItemId],
        k: usize,
        m: usize,
    ) -> BinaryHeap<SessionScore> {
        let num_items_in_evolving_session = evolving_session.len();

        let mut most_recent_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(m);

        for session_item in evolving_session.iter() {
            if let Some(sessions) = self.sessions_for_item(session_item) {
                for session in sessions {
                    let max_order = self.max_order_for_session(session);

                    let session_with_age = SessionScore::new(*session, max_order as f64);

                    if most_recent_neighbors.len() < m {
                        most_recent_neighbors.push(session_with_age);
                    } else {
                        let mut top = most_recent_neighbors.peek_mut().unwrap();
                        if session_with_age.score > top.score {
                            *top = session_with_age;
                        }
                    }
                }
            }
        }

        let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);

        for neighbor_session in most_recent_neighbors.into_iter() {
            let mut similarity = 0_f64;

            let other_session_items = self.items_for_session(&neighbor_session.id);

            //            let start_time = Instant::now();
            // Decayed dot product
            for (pos, item_id) in evolving_session.iter().enumerate() {
                if other_session_items.contains(item_id) {
                    let decay_factor = (pos + 1) as f64 / num_items_in_evolving_session as f64;
                    similarity += decay_factor;
                }
            }
            // let duration = start_time.elapsed();
            // let duration_as_micros:f64 = duration.as_micros() as f64;
            // if duration_as_micros > 500_f64 {
            //     println!("slow session matching:{} micros, evolving_session.len():{}, other_session_items.len:{}", duration_as_micros, evolving_session.len(), other_session_items.len());
            // }

            if similarity > 0.0 {
                // Update heap holding top-n scored items for this item
                let scored_session = SessionScore::new(neighbor_session.id, similarity);

                if closest_neighbors.len() < k {
                    closest_neighbors.push(scored_session);
                } else {
                    let mut bottom = closest_neighbors.peek_mut().unwrap();
                    if scored_session.score > bottom.score {
                        *bottom = scored_session;
                    }
                }
            }
        }

        closest_neighbors
    }
}

impl VSkNNIndex {
    pub fn new(
        interactions: Vec<(TrainingSessionId, ItemId, Time)>,
        sample_size_m: usize,
        max_qty_session_items: usize,
    ) -> Self {
        // start only need to retain sample_size_m sessions per item
        let valid_session_ids: HashSet<u32> = interactions
            .iter()
            .cloned()
            .map(|(session_id, item_id, time)| (item_id, SessionTime::new(session_id, time as u32)))
            .into_group_map()
            .into_iter()
            .flat_map(|(_item_id, mut session_id_with_time)| {
                session_id_with_time.sort();
                session_id_with_time.dedup();
                session_id_with_time.sort_unstable_by(|left, right| {
                    // We keep the sessions with the largest time values
                    left.cmp(right).reverse()
                });
                if session_id_with_time.len() > sample_size_m {
                    // we remove the sessions per item with the lowest time values
                    session_id_with_time.truncate(sample_size_m);
                }
                if session_id_with_time.len() > max_qty_session_items {
                    // this training session has too many items and does not contribute to improving predictions
                    session_id_with_time.clear();
                }

                let session_ids: HashSet<u32> = session_id_with_time
                    .iter()
                    .map(|session_id_time| session_id_time.session_id)
                    .collect();
                session_ids
            })
            .collect();
        // end only need to retain sample_size_m sessions per item

        let mut historical_session_index: HashMap<TrainingSessionId, HashSet<ItemId>> =
            HashMap::new();
        let mut historical_session_max_order: HashMap<TrainingSessionId, Time> = HashMap::new();
        let mut historical_item_index: HashMap<ItemId, HashSet<TrainingSessionId>> = HashMap::new();

        //let mut ignored_training_rows = 0;
        for (session_id, item_id, order) in interactions.into_iter() {
            if !valid_session_ids.contains(&session_id) {
                //ignored_training_rows += 1;
                continue;
            }
            let session_items = historical_session_index
                .entry(session_id)
                .or_insert(HashSet::new());
            session_items.insert(item_id);

            let current_max_order = historical_session_max_order
                .entry(session_id)
                .or_insert(order);
            if order > *current_max_order {
                *current_max_order = order;
            }

            let item_sessions = historical_item_index
                .entry(item_id)
                .or_insert(HashSet::new());
            item_sessions.insert(session_id);
        }

        //println!("Ignored training events: {} ", ignored_training_rows);
        //println!("Sessions used: {}", valid_session_ids.len());
        //println!("Items used: {}", historical_item_index.len());

        let num_historical_sessions = historical_session_index.len();

        let item_idfs: HashMap<u64, f64> = historical_item_index
            .iter()
            .map(|(item, session_ids)| {
                let item_idf = idf(num_historical_sessions, session_ids.len());

                (*item, item_idf)
            })
            .collect();

        VSkNNIndex {
            session_index: historical_session_index,
            session_max_order: historical_session_max_order,
            item_index: historical_item_index,
            item_idfs,
        }
    }

    fn sessions_for_item(&self, item: &u64) -> Option<&HashSet<u32, RandomState>> {
        self.item_index.get(item) // move object and ownership to function that call us
    }

    fn max_order_for_session(&self, session: &TrainingSessionId) -> Time {
        self.session_max_order[session]
    }
}
