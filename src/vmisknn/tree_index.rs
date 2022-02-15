use std::collections::BinaryHeap;
use std::error::Error;

use dary_heap::OctonaryHeap;
use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::HashMap;
use hashbrown::HashSet;

use crate::vmisknn::{SessionScore, SessionTime};
use crate::vmisknn::similarity_hashed::SimilarityComputationHash;

pub struct TreeIndex {
    historical_sessions_item_id_sorted: Vec<u64>,
    historical_sessions_session_id_sorted: Vec<u32>,
    historical_sessions_max_time_stamp: Vec<u32>,
}

impl TreeIndex {
    pub fn new(path_train: &str, n_most_recent_sessions: usize) -> Self {
        //println!("Reading inputs for tree index from {}...", path_train);
        let data_train = read_from_file(path_train);
        let (
            historical_sessions_train,
            _historical_sessions_id_train,
            historical_sessions_max_time_stamp,
        ) = data_train.unwrap();

        //println!("Creating tree index...");
        let (historical_sessions_item_id_sorted, historical_sessions_session_id_sorted) =
            prepare_binary_search(
                &historical_sessions_train,
                &historical_sessions_max_time_stamp,
                n_most_recent_sessions,
            );

        TreeIndex {
            historical_sessions_item_id_sorted,
            historical_sessions_session_id_sorted,
            historical_sessions_max_time_stamp,
        }
    }
}

fn read_from_file(
    path: &str,
) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<u32>), Box<dyn Error>> {
    // Creates a new csv `Reader` from a file
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(path)?;

    // Vector initialization
    let mut session_id: Vec<usize> = Vec::new();
    let mut item_id: Vec<usize> = Vec::new();
    let mut time: Vec<usize> = Vec::new();

    reader.deserialize().for_each(|result| {
        if result.is_ok() {
            let raw: (usize, usize, f64) = result.unwrap();
            let (a_session_id, a_item_id, a_time): (usize, usize, usize) =
                (raw.0, raw.1, raw.2.round() as usize);

            session_id.push(a_session_id);
            item_id.push(a_item_id);
            time.push(a_time);
        } else {
            eprintln!("Unable to parse input!");
        }
    });

    // Sort by session id - the data is unsorted
    let mut session_id_indices: Vec<usize> = (0..session_id.len()).into_iter().collect();
    session_id_indices.sort_by_key(|&i| &session_id[i]);
    let session_id_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| session_id[i])
        .collect();
    let item_id_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| item_id[i])
        .collect();
    let time_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| time[i])
        .collect();

    // Get unique session ids
    // let mut session_id_unique = session_id.clone();
    session_id.sort_unstable();
    session_id.dedup();

    // Create historical sessions array (deduplicated), historical sessions id array and array with max timestamps.
    //let mut i: usize = 0;
    let mut historical_sessions: Vec<Vec<usize>> = Vec::with_capacity(session_id.len());
    let mut historical_sessions_id: Vec<Vec<usize>> = Vec::with_capacity(session_id.len());
    let mut historical_sessions_max_time_stamp: Vec<u32> =
        Vec::with_capacity(session_id.len());
    let mut history_session: Vec<usize> = Vec::with_capacity(1000);
    let mut history_session_id: Vec<usize> = Vec::with_capacity(1000);
    let mut max_time_stamp: usize = time_sorted[0];
    // Push initial session and item id
    history_session.push(item_id_sorted[0]);
    history_session_id.push(item_id_sorted[0]);
    // Loop over length of data
    for i in 1..session_id_sorted.len() {
        if session_id_sorted[i] == session_id_sorted[i - 1] {
            if !history_session.contains(&item_id_sorted[i]) {
                history_session.push(item_id_sorted[i]);
                history_session_id.push(session_id_sorted[i]);
                if time_sorted[i] > max_time_stamp {
                    max_time_stamp = time_sorted[i];
                }
            }
        } else {
            let mut history_session_sorted = history_session.clone();
            history_session_sorted.sort_unstable();
            historical_sessions.push(history_session_sorted);
            historical_sessions_id.push(history_session_id.clone());
            historical_sessions_max_time_stamp.push(max_time_stamp as u32);
            history_session.clear();
            history_session_id.clear();
            history_session.push(item_id_sorted[i]);
            history_session_id.push(session_id_sorted[i]);
            max_time_stamp = time_sorted[i];
        }
    }

    Ok((
        historical_sessions,
        historical_sessions_id,
        historical_sessions_max_time_stamp,
    ))
}

fn prepare_binary_search(
    historical_sessions: &[Vec<usize>],
    timestamps: &[u32],
    n_most_recent_sessions: usize,
) -> (Vec<u64>, Vec<u32>) {
    // Initialize arrays
    let historical_sessions_length: usize = historical_sessions.iter().map(|x| x.len()).sum();
    let mut historical_sessions_values = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_session_indices = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_indices = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_timestamps = Vec::with_capacity(historical_sessions_length);
    let mut iterable = 0_usize;

    // Create (i) vector of historical sessions, (ii) vector of historical session indices, (iii) vector of session indices
    for (session_id, session) in historical_sessions.iter().enumerate() {
        for (item_id, _) in session.iter().enumerate() {
            historical_sessions_values.push(historical_sessions[session_id][item_id]);
            historical_sessions_indices.push(iterable);
            historical_sessions_session_indices.push(session_id);
            historical_sessions_timestamps.push(timestamps[session_id]);
            iterable += 1;
        }
    }

    // Sort historical session values and session indices array
    historical_sessions_indices.sort_by_key(|&i| historical_sessions_values[i]);
    let historical_sessions_values_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_values[i] as u64)
        .collect();
    let historical_sessions_session_indices_sorted: Vec<u32> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_session_indices[i] as u32)
        .collect();
    let historical_sessions_timestamps_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_timestamps[i] as u64)
        .collect();

    // Get unique item_ids
    let mut unique_items = historical_sessions_values_sorted.clone();
    unique_items.dedup();

    let mut historical_sessions_values_sorted_truncated: Vec<u64> =
        Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_session_indices_sorted_truncated: Vec<u32> =
        Vec::with_capacity(historical_sessions_length);

    // Loop over historical sessions to remove all sessions per item older than n_most_recent_sessions
    for current_item in unique_items.iter() {
        let left_index =
            binary_search_left(&historical_sessions_values_sorted, *current_item).unwrap();
        let right_index =
            binary_search_right(&historical_sessions_values_sorted, *current_item).unwrap();
        let mut current_item_similar_sessions: Vec<u64> =
            historical_sessions_values_sorted[left_index..right_index + 1].to_vec();
        let current_item_timestamps: Vec<u64> =
            historical_sessions_timestamps_sorted[left_index..right_index + 1].to_vec();
        let current_item_similar_sessions_ids: Vec<u32> =
            historical_sessions_session_indices_sorted[left_index..right_index + 1].to_vec();
        // Sort session ids by reverse timestamp and truncate to n_most_recent_sessions
        let mut timestamp_indices: Vec<usize> = (0..current_item_timestamps.len()).collect();
        timestamp_indices.sort_by_key(|&i| current_item_timestamps[i]);
        let mut current_item_similar_sessions_id_sorted: Vec<u32> = timestamp_indices
            .iter()
            .map(|&i| current_item_similar_sessions_ids[i] as u32)
            .collect();
        current_item_similar_sessions_id_sorted.reverse();
        current_item_similar_sessions_id_sorted.truncate(n_most_recent_sessions);
        current_item_similar_sessions.truncate(n_most_recent_sessions);
        // Store
        historical_sessions_values_sorted_truncated.append(&mut current_item_similar_sessions);
        historical_sessions_session_indices_sorted_truncated
            .append(&mut current_item_similar_sessions_id_sorted);
    }

    // Return sorted flattened historical sessions and sorted flattened historical session indices
    (
        historical_sessions_values_sorted_truncated,
        historical_sessions_session_indices_sorted_truncated,
    )
}

// Custom binary search because this is stable unlike the rust default (i.e. this always returns right-most index in case of duplicate entries instead of a random match)
fn binary_search_right(array: &[u64], key: u64) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if unsafe { array.get_unchecked(mid) } > &key {
            top = mid;
        } else {
            bottom = mid + 1;
        }
    }

    if top > 0 {
        if array[top - 1] == key {
            Ok(top - 1)
        } else {
            Err(top - 1)
        }
    } else {
        Err(top)
    }
}

fn binary_search_left(array: &[u64], key: u64) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if unsafe { array.get_unchecked(mid) } < &key {
            bottom = mid + 1;
        } else {
            top = mid;
        }
    }

    if top < array.len() {
        if array[top] == key {
            Ok(top)
        } else {
            Err(top)
        }
    } else {
        Err(top)
    }
}

impl SimilarityComputationHash for TreeIndex {
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
        // We use a binary heap for the (timestamp, session_id) tuple and a hashmap for the (session_id, score) tuples.
        let mut heap_timestamps = OctonaryHeap::<SessionTime>::with_capacity(m);
        let mut hash_similarities: HashMap<u32, f32> = HashMap::with_capacity(m + 1);

        //  Loop over items in evolving session
        for (pos, item_id) in evolving_session.iter().enumerate() {
            // Find index of current item in historical session vector using binary search
            let right_index_result =
                binary_search_right(&self.historical_sessions_item_id_sorted, *item_id);
            // If match found, find duplicate elements left to the match found;
            // this is because binary search returns the right most match.
            if let Ok(right_index) = right_index_result {
                let left_index =
                    binary_search_left(&self.historical_sessions_item_id_sorted, *item_id).unwrap();
                // Calculate position weight
                let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
                // Loop over all historical sessions with a match
                let similar_sessions =
                    &self.historical_sessions_session_id_sorted[left_index..right_index + 1];
                'session_loop: for session_id in similar_sessions {
                    let session_present = hash_similarities.contains_key(session_id);
                    if session_present {
                        let value = hash_similarities.get_mut(session_id).unwrap();
                        *value += decay_factor;
                    } else {
                        let session_time_stamp =
                            self.historical_sessions_max_time_stamp[*session_id as usize];
                        if hash_similarities.len() < m {
                            hash_similarities.insert(*session_id, decay_factor);
                            heap_timestamps.push(SessionTime::new(*session_id, session_time_stamp));
                        } else if session_time_stamp > heap_timestamps.peek().unwrap().time {
                            // Insert new key
                            hash_similarities.insert(*session_id, decay_factor);
                            // Remove the the existing minimum time stamp.
                            let session_id_remove = heap_timestamps.peek().unwrap().session_id;
                            hash_similarities.remove_entry(&session_id_remove);
                            // Set new minimum timestamp
                            let mut bottom = heap_timestamps.peek_mut().unwrap();
                            *bottom = SessionTime::new(*session_id, session_time_stamp);
                        } else {
                            break 'session_loop;
                        }
                    }
                }
            }
        }

        // Return top-k
        let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
        for (session_id, score) in hash_similarities.iter() {
            let scored_session = SessionScore::new(*session_id, *score as f64);
            if closest_neighbors.len() < k {
                closest_neighbors.push(scored_session);
            } else {
                let mut bottom = closest_neighbors.peek_mut().unwrap();
                if scored_session.score > bottom.score {
                    *bottom = scored_session;
                }
            }
        }

        closest_neighbors
    }
}
