use std::collections::BinaryHeap;

use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::HashMap;
use hashbrown::HashSet;

use crate::vmisknn::{SessionScore, SessionTime};
use crate::vmisknn::vmisknn_index::prepare_hashmap;
use crate::vmisknn::similarity_hashed::SimilarityComputationHash;
use crate::vmisknn::vmisknn_index::read_from_file;

pub struct VMISSkNNIndexNoOpt {
    sessions_for_item: HashMap<u64, Vec<u32>>,
    historical_sessions_max_time_stamp: Vec<u32>,
}

impl VMISSkNNIndexNoOpt {
    pub fn new(path_train: &str, n_most_recent_sessions: usize) -> Self {
        //println!("Reading inputs for tree index from {}...", path_train);
        let data_train = read_from_file(path_train);
        let (
            historical_sessions_train,
            _historical_sessions_id_train,
            historical_sessions_max_time_stamp,
        ) = data_train.unwrap();

        //println!("Creating index...");
        let historical_sessions = prepare_hashmap(
            &historical_sessions_train,
            &historical_sessions_max_time_stamp,
            n_most_recent_sessions,
        );

        VMISSkNNIndexNoOpt {
            sessions_for_item: historical_sessions,
            historical_sessions_max_time_stamp,
        }
    }
}

impl SimilarityComputationHash for VMISSkNNIndexNoOpt {
    fn items_for_session(&self, _session: &u32) -> &HashSet<u64, DefaultHashBuilder> {
        unimplemented!()
    }

    fn idf(&self, _item_id: &u64) -> f64 {
        unimplemented!()
    }

    fn find_neighbors(
        &self,
        evolving_session: &[u64],
        k: usize,
        m: usize,
    ) -> BinaryHeap<SessionScore> {
        // We use a d-ary heap for the (timestamp, session_id) tuple and a hashmap for the (session_id, score) tuples.
        let mut heap_timestamps = BinaryHeap::<SessionTime>::with_capacity(m);
        let mut hash_similarities = HashMap::with_capacity(m);

        //  Loop over items in evolving session
        for (pos, item_id) in evolving_session.iter().enumerate() {
            if let Some(similar_sessions) = self.sessions_for_item.get(item_id) {
                let decay_factor = (pos + 1) as f64 / evolving_session.len() as f64;
                // Loop over all historical sessions with a match
                for session_id in similar_sessions {
                    match hash_similarities.get_mut(session_id) {
                        Some(value) => *value += decay_factor,
                        None => {
                            let session_time_stamp =
                                self.historical_sessions_max_time_stamp[*session_id as usize];
                            if hash_similarities.len() < m {
                                hash_similarities.insert(*session_id, decay_factor);
                                heap_timestamps
                                    .push(SessionTime::new(*session_id, session_time_stamp));
                            } else {
                                let mut bottom = heap_timestamps.peek_mut().unwrap();

                                if session_time_stamp > bottom.time {
                                    // Remove the the existing minimum time stamp.
                                    hash_similarities.remove_entry(&bottom.session_id);
                                    // Set new minimum timestamp
                                    hash_similarities.insert(*session_id, decay_factor);
                                    *bottom = SessionTime::new(*session_id, session_time_stamp);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return top-k
        let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
        for (session_id, score) in hash_similarities.iter() {
            if closest_neighbors.len() < k {
                let scored_session = SessionScore::new(*session_id, *score);
                closest_neighbors.push(scored_session);
            } else {
                let mut bottom = closest_neighbors.peek_mut().unwrap();
                if score > &bottom.score {
                    let scored_session = SessionScore::new(*session_id, *score);
                    *bottom = scored_session;
                } else if (score - bottom.score).abs() < f64::EPSILON
                    && (self.historical_sessions_max_time_stamp[*session_id as usize]
                    > self.historical_sessions_max_time_stamp[bottom.id as usize])
                {
                    let scored_session = SessionScore::new(*session_id, *score);
                    *bottom = scored_session;
                }
            }
        }

        closest_neighbors
    }
}
