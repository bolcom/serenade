#![allow(warnings)]

extern crate csv;
extern crate hashbrown;
extern crate num_format;
extern crate rand;

use float_cmp::*;
use hashbrown::HashSet;
use num_format::{Locale, ToFormattedString};
use rand::Rng;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::error::Error;
use std::time::{Duration, Instant};

fn main() {
    // Retrieve data
    let dataset = "50m";
    // Training set
    let path = "./data/private-clicks-";
    let path_train = format!("{}{}_train.txt", path, dataset);
    let max_events_train: usize = 100_000_000;
    let data_train = read_from_file(&path_train, max_events_train);
    let (
        historical_sessions_train,
        historical_sessions_id_train,
        historical_sessions_max_time_stamp_train,
    ) = data_train.unwrap();
    // println!("Number of sessions in training set: {}", historical_sessions_train.len().to_formatted_string(&Locale::en));
    let NUM_HISTORY_SESSIONS: usize = historical_sessions_train.len();

    // Test set
    let path_test = format!("{}{}_test.txt", path, dataset);
    let max_events_test: usize = 100_000_000;
    let data_test = read_from_file(&path_test, max_events_test);
    let (
        historical_sessions_test,
        historical_sessions_id_test,
        historical_sessions_max_time_stamp_test,
    ) = data_test.unwrap();
    // println!("Number of sessions in test set: {}", historical_sessions_test.len().to_formatted_string(&Locale::en));

    // Prepare algorithms
    let (historical_sessions_item_id_sorted, historical_sessions_session_id_sorted) =
        prepare_binary_search(&historical_sessions_train);
    // let historical_sessions_hashed = prepare_nested_loops(&historical_sessions_train);

    // Simple benchmark loop, to allow better control over inputs
    let n_iterations: usize = 10000;
    let n_most_recent_sessions: usize = 5000;
    let mut rng = rand::thread_rng();
    // let algorithms = ["binary_search", "nested_loops ", "linear_search"];
    // let algorithms = ["binary_search", "linear_search"];
    let algorithms = ["binary_search"];

    let mut duration: Vec<Duration> = vec![Duration::new(0, 0); n_iterations * algorithms.len()];

    println!("Task: for {} iterations ('evolving sessions'), find {} most recent similar sessions out of {} historical sessions ", n_iterations.to_formatted_string(&Locale::en), n_most_recent_sessions.to_formatted_string(&Locale::en), NUM_HISTORY_SESSIONS.to_formatted_string(&Locale::en));

    for i in 0..n_iterations {
        // Choose random evolving session
        let choice = rng.gen_range(0..historical_sessions_test.len());
        // let choice = 252711; // Longest evolving session in 50m test dataset (103 items, before deduplicating)
        // let choice = 290; // Longest evolving session in 100k dataset (339 items, before deduplicating)
        let evolving_session = &historical_sessions_test[choice];
        // Binary search
        let start = Instant::now();
        let (similarities_binary_search_index, similarities_binary_search_values) =
            calc_similarities_binary_search(
                &evolving_session,
                &historical_sessions_item_id_sorted,
                &historical_sessions_session_id_sorted,
                &historical_sessions_max_time_stamp_train,
                n_most_recent_sessions,
            );
        duration[i] += start.elapsed();
        // println!("{:?}", similarities_binary_search_index);
        // Nested loops
        // let start = Instant::now();
        // let similarities_nested_loops = calc_similarities_nested_loops(&evolving_session, &historical_sessions_hashed);
        // duration[i + n_iterations] += start.elapsed();
        // assert_eq!(similarities_binary_search, similarities_nested_loops);
        // Linear search
        // let start = Instant::now();
        // let (similarities_linear_search_index, similarities_linear_search_values) = calc_similarities_linear_search(&evolving_session, &historical_sessions_train, &historical_sessions_max_time_stamp_train, n_most_recent_sessions);
        // duration[i + 1 * n_iterations] += start.elapsed();
        // assert_eq!(similarities_binary_search_index, similarities_linear_search_index);
        // for (key, _) in similarities_binary_search_values.iter().enumerate(){
        // assert!( approx_eq!(f32, similarities_binary_search_values[key], similarities_linear_search_values[key], epsilon = 0.00001) );
        // }
    }
    for (i, algorithm) in algorithms.iter().enumerate() {
        print_timing(
            &mut duration[(i * n_iterations)..(i + 1) * n_iterations],
            &n_iterations,
            algorithm,
        );
        // println!("{:?}", &duration[(i * n_iterations)..(i + 1) * n_iterations]);
    }
}

fn print_timing(duration: &mut [Duration], n_iterations: &usize, name: &str) {
    duration.sort();
    let mut duration_total = Duration::new(0, 0);
    let mut duration_max = Duration::new(0, 0);
    let mut n_iterations_p90: usize;
    let mut duration_p90: Duration;
    if *n_iterations == 1 {
        n_iterations_p90 = *n_iterations as usize;
        duration_p90 = duration.iter().sum();
    } else {
        n_iterations_p90 = (0.9 * *n_iterations as f32) as usize;
        duration_p90 = duration[0..n_iterations_p90].iter().sum();
    }
    for timing in duration.iter() {
        duration_total += *timing;
        if timing > &duration_max {
            duration_max = *timing;
        }
    }
    let ns_avg = ((duration_total.as_secs() * 1_000_000_000
        + (duration_total.subsec_nanos() as u64))
        / *n_iterations as u64)
        .to_formatted_string(&Locale::en);
    let ns_max = (duration_max.as_secs() * 1_000_000_000 + (duration_max.subsec_nanos() as u64))
        .to_formatted_string(&Locale::en);
    let ns_p90 = ((duration_p90.as_secs() * 1_000_000_000 + (duration_p90.subsec_nanos() as u64))
        / n_iterations_p90 as u64)
        .to_formatted_string(&Locale::en);
    println!(
        "test {:}: {:>20} ns/iter (average) | {:>15} ns/iter (p90) | {:>15} ns/iter (max)",
        name, ns_avg, ns_p90, ns_max
    );
}

#[derive(Debug, Deserialize)]
struct Event {
    SessionId: usize,
    ItemId: usize,
    Time: usize,
}

fn read_from_file(
    path: &str,
    max_events: usize,
) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>), Box<dyn Error>> {
    // Creates a new csv `Reader` from a file
    let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    // Vector initialization
    let mut session_id: Vec<usize> = Vec::with_capacity(max_events);
    let mut item_id: Vec<usize> = Vec::with_capacity(max_events);
    let mut time: Vec<usize> = Vec::with_capacity(max_events);

    // Fill vectors
    for record in reader.deserialize() {
        let record: Event = record?;
        session_id.push(record.SessionId);
        item_id.push(record.ItemId);
        time.push(record.Time);
    }

    // Sort by session id - the data is unsorted
    let mut session_id_indices: Vec<usize> = (0..session_id.len()).into_iter().collect();
    session_id_indices.sort_by_key(|&i| &session_id[i]);
    let session_id_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| session_id[i].clone())
        .collect();
    let item_id_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| item_id[i].clone())
        .collect();
    let time_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| time[i].clone())
        .collect();

    // Get unique session ids
    let mut session_id_unique = session_id.clone();
    session_id_unique.sort();
    session_id_unique.dedup();

    // Create historical sessions array (deduplicated), historical sessions id array and array with max timestamps.
    let mut i: usize = 0;
    let mut historical_sessions: Vec<Vec<usize>> = Vec::with_capacity(session_id_unique.len());
    let mut historical_sessions_id: Vec<Vec<usize>> = Vec::with_capacity(session_id_unique.len());
    let mut historical_sessions_max_time_stamp: Vec<usize> =
        Vec::with_capacity(session_id_unique.len());
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
            history_session_sorted.sort();
            historical_sessions.push(history_session_sorted);
            historical_sessions_id.push(history_session_id.clone());
            historical_sessions_max_time_stamp.push(max_time_stamp.clone());
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

fn prepare_nested_loops(historical_sessions: &[Vec<usize>]) -> Vec<HashSet<usize>> {
    // Create hashtable from historical sessions
    let historical_sessions_hashed: Vec<HashSet<usize>> = (0..historical_sessions.len())
        .map(|i| {
            let mut history_session = HashSet::new();

            let session_length = historical_sessions[i].len();
            for j in 0..session_length {
                history_session.insert(historical_sessions[i][j]);
            }
            history_session
        })
        .collect();

    historical_sessions_hashed
}

fn calc_similarities_nested_loops(
    evolving_session: &[usize],
    historical_sessions: &[HashSet<usize>],
) -> Vec<f32> {
    let mut similarities = Vec::with_capacity(historical_sessions.len());

    for neighbor_session in historical_sessions.iter() {
        let mut similarity = 0_f32;

        for (pos, item_id) in evolving_session.iter().enumerate() {
            if neighbor_session.contains(&item_id) {
                let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
                similarity += decay_factor;
            }
        }
        similarities.push(similarity);
    }

    similarities
}

fn prepare_binary_search(historical_sessions: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>) {
    // Initialize arrays
    let historical_sessions_length: usize = historical_sessions.iter().map(|x| x.len()).sum();
    let mut historical_sessions_values = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_session_indices = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_indices = Vec::with_capacity(historical_sessions_length);
    let mut iterable = 0_usize;

    // Create (i) vector of historical sessions, (ii) vector of historical session indices, (iii) vector of session indices
    for (session_id, session) in historical_sessions.iter().enumerate() {
        for (item_id, _) in session.iter().enumerate() {
            historical_sessions_values.push(historical_sessions[session_id][item_id]);
            historical_sessions_indices.push(iterable);
            historical_sessions_session_indices.push(session_id);
            iterable += 1;
        }
    }

    // Sort historical session values and session indices array
    historical_sessions_indices.sort_by_key(|&i| historical_sessions_values[i]);
    let historical_sessions_values_sorted: Vec<usize> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_values[i].clone())
        .collect();
    let historical_sessions_session_indices_sorted: Vec<usize> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_session_indices[i].clone())
        .collect();

    // Return sorted flattened historical sessions and sorted flattened historical session indices
    (
        historical_sessions_values_sorted,
        historical_sessions_session_indices_sorted,
    )
}

// Custom binary search because this is stable unlike the rust default (i.e. this always returns right-most index in case of duplicate entries instead of a random match)
fn binary_search_right(array: &[usize], key: &usize) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if (unsafe { array.get_unchecked(mid) } > key) {
            top = mid;
        } else {
            bottom = mid + 1;
        }
    }

    if top > 0 {
        if array[top - 1] == *key {
            Ok(top - 1)
        } else {
            Err(top - 1)
        }
    } else {
        Err(top)
    }
}

fn binary_search_left(array: &[usize], key: &usize) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if (unsafe { array.get_unchecked(mid) } < key) {
            bottom = mid + 1;
        } else {
            top = mid;
        }
    }

    if top < array.len() {
        if array[top] == *key {
            return Ok(top);
        } else {
            return Err(top);
        }
    } else {
        return Err(top);
    }
}

fn binary_search_reverse(array: &[usize], key: &usize) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if (unsafe { array.get_unchecked(mid) } > key) {
            bottom = mid + 1;
        } else {
            top = mid;
        }
    }

    if top < array.len() {
        if array[top] == *key {
            return Ok(top);
        } else {
            return Err(top);
        }
    } else {
        return Err(top);
    }
}

#[derive(PartialEq, Debug, PartialOrd)]
struct SessionScore(usize, f32);

impl Ord for SessionScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Eq for SessionScore {}

#[derive(Eq, PartialEq, Debug)]
struct SessionTime(usize, usize);

impl Ord for SessionTime {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0).reverse()
    }
}

impl PartialOrd for SessionTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

fn calc_similarities_binary_search(
    evolving_session: &[usize],
    historical_sessions_item_id_sorted: &[usize],
    historical_sessions_session_id_sorted: &[usize],
    historical_sessions_max_time_stamp: &[usize],
    n_most_recent_sessions: usize,
) -> (Vec<usize>, Vec<f32>) {
    // We use a binary heap for the (timestamp, session_id) tuple and a BTreeMap for the (session_id, score) tuples.
    let mut heap_timestamps = BinaryHeap::<SessionTime>::with_capacity(n_most_recent_sessions);
    let mut bst_similarities = BTreeMap::new();

    //  Loop over items in evolving session
    for (pos, item_id) in evolving_session.iter().enumerate() {
        // Find index of current item in historical session vector using binary search
        let right_index_result = binary_search_right(historical_sessions_item_id_sorted, item_id);
        // If match found, find duplicate elements left to the match found; this is because binary search returns the right most match.
        if right_index_result.is_ok() {
            let left_index =
                binary_search_left(historical_sessions_item_id_sorted, item_id).unwrap();
            let right_index = right_index_result.unwrap();
            // Calculate position weight
            let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
            // Loop over all historical sessions with a match
            for session_id in
                historical_sessions_session_id_sorted[left_index..right_index + 1].iter()
            {
                let session_present = bst_similarities.contains_key(session_id);
                // If the current historical session is already in the output, just add decay factor to the similarity score.
                if session_present {
                    let mut old_value = bst_similarities.get_mut(session_id).unwrap();
                    *old_value += decay_factor;
                } else {
                    let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                    // Push similar sessions into the output until we reach n_most_recent_sessions
                    if bst_similarities.len() < n_most_recent_sessions {
                        heap_timestamps.push(SessionTime(session_time_stamp, *session_id));
                        bst_similarities.insert(*session_id, decay_factor);
                    }
                    // Once we reach n_most_recent_sessions in the output, only store new sessions if they are newer than the existing minimum timestamp.
                    else if session_time_stamp > heap_timestamps.peek().unwrap().0 {
                        bst_similarities.insert(*session_id, decay_factor);
                        // Remove the the existing minimum time stamp.
                        let session_id_remove = heap_timestamps.peek().unwrap().1;
                        bst_similarities.remove(&session_id_remove);
                        // Set new minimum timestamp in heap_timestamp
                        let mut bottom = heap_timestamps.peek_mut().unwrap();
                        *bottom = SessionTime(session_time_stamp, *session_id);
                    }
                }
            }
        }
    }
    // Return session_id and corresponding similarity score to evolving session for up to n_most_recent_sessions
    let similarities = bst_similarities.values().cloned().collect();
    let sessions = bst_similarities.keys().cloned().collect();
    (sessions, similarities)
}

fn calc_similarities_linear_search(
    evolving_session: &[usize],
    historical_sessions: &[Vec<usize>],
    historical_sessions_max_time_stamp: &[usize],
    n_most_recent_sessions: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut session_ids_sorted_by_session: Vec<usize> =
        Vec::with_capacity(n_most_recent_sessions + 1);
    let mut session_ids_sorted_by_timestamp: Vec<usize> =
        Vec::with_capacity(n_most_recent_sessions + 1);
    let mut similarity_score_sorted_by_session: Vec<f32> =
        Vec::with_capacity(n_most_recent_sessions + 1);
    let mut timestamps_sorted_by_timestamp: Vec<usize> =
        Vec::with_capacity(n_most_recent_sessions + 1);
    let mut similarity_score = 0_f32;

    for (session_id, session) in historical_sessions.iter().enumerate() {
        let session_time_stamp = historical_sessions_max_time_stamp[session_id];
        if session_ids_sorted_by_session.len() < n_most_recent_sessions {
            for (item_id, item) in session.iter().enumerate() {
                for (pos, item_evolving_session) in evolving_session.iter().enumerate() {
                    if item_evolving_session == item {
                        let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
                        similarity_score += decay_factor;
                    }
                }
            }
            if similarity_score > 0_f32 {
                let session_time_stamp_index =
                    binary_search_reverse(&timestamps_sorted_by_timestamp, &session_time_stamp)
                        .unwrap_or_else(|x| x);
                // Insert
                session_ids_sorted_by_session.push(session_id);
                similarity_score_sorted_by_session.push(similarity_score);
                timestamps_sorted_by_timestamp.insert(session_time_stamp_index, session_time_stamp);
                session_ids_sorted_by_timestamp.insert(session_time_stamp_index, session_id);
                similarity_score = 0_f32;
            }
        } else if session_time_stamp > timestamps_sorted_by_timestamp[n_most_recent_sessions - 1] {
            for (item_id, item) in session.iter().enumerate() {
                for (pos, item_evolving_session) in evolving_session.iter().enumerate() {
                    if item_evolving_session == item {
                        let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
                        similarity_score += decay_factor;
                    }
                }
            }
            if similarity_score > 0_f32 {
                let session_time_stamp_index =
                    binary_search_reverse(&timestamps_sorted_by_timestamp, &session_time_stamp)
                        .unwrap_or_else(|x| x);
                // Insert
                session_ids_sorted_by_session.push(session_id);
                similarity_score_sorted_by_session.push(similarity_score);
                timestamps_sorted_by_timestamp.insert(session_time_stamp_index, session_time_stamp);
                session_ids_sorted_by_timestamp.insert(session_time_stamp_index, session_id);
                // Remove
                let session_id_remove = session_ids_sorted_by_timestamp[n_most_recent_sessions];
                let session_index_remove = session_ids_sorted_by_session
                    .binary_search(&session_id_remove)
                    .unwrap();
                session_ids_sorted_by_session.remove(session_index_remove);
                similarity_score_sorted_by_session.remove(session_index_remove);
                timestamps_sorted_by_timestamp.pop();
                session_ids_sorted_by_timestamp.pop();
                similarity_score = 0_f32;
            }
        }
    }

    // Return session_id and corresponding similarity score to evolving session for up to n_most_recent_sessions
    (
        session_ids_sorted_by_session,
        similarity_score_sorted_by_session,
    )
}
