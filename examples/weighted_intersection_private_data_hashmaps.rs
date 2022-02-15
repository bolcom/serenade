#![allow(warnings)]

extern crate csv;
extern crate dary_heap;
extern crate hashbrown;
extern crate num_format;
extern crate rand;
extern crate rand_pcg;

use core::ptr;
use dary_heap::OctonaryHeap;
use float_cmp::*;
use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use num_format::{Locale, ToFormattedString};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
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

    // Simple benchmark loop, to allow better control over inputs
    let n_iterations: usize = historical_sessions_test.len();
    let n_most_recent_sessions: usize = 5000;
    let max_items_in_session: usize = 100000;
    let k: usize = 100;
    let mut rng = Pcg64::seed_from_u64(0);
    let algorithms = [
        "binary_search_heap+tree",
        "binary_search_heap+vecs",
        "binary_search_heap+hash",
        "hashmap+heap+hash      ",
        "hashmap+heap+hash_adj  ",
    ];

    // Prepare algorithms
    let (historical_sessions_item_id_sorted, historical_sessions_session_id_sorted) =
        prepare_binary_search(&historical_sessions_train);
    let (
        historical_sessions_item_id_sorted_truncated,
        historical_sessions_session_id_sorted_truncated,
    ) = prepare_binary_search_truncated(
        &historical_sessions_train,
        &historical_sessions_max_time_stamp_train,
        n_most_recent_sessions,
    );
    let historical_sessions_hashed = prepare_hashmap(
        &historical_sessions_item_id_sorted,
        &historical_sessions_session_id_sorted,
    );
    let historical_sessions_hashed_truncated = prepare_hashmap(
        &historical_sessions_item_id_sorted_truncated,
        &historical_sessions_session_id_sorted_truncated,
    );

    let mut duration: Vec<Duration> = vec![Duration::new(0, 0); n_iterations * algorithms.len()];

    println!("Task: for {} iterations ('evolving sessions'), find top-{} of {} most recent similar sessions out of {} historical sessions ", n_iterations.to_formatted_string(&Locale::en), k, n_most_recent_sessions.to_formatted_string(&Locale::en), NUM_HISTORY_SESSIONS.to_formatted_string(&Locale::en));

    for i in 0..n_iterations {
        // Choose random evolving session (we use a fixed seed for reproducibility)
        // let choice = rng.gen_range(0..historical_sessions_test.len());
        let evolving_session_base = &historical_sessions_test[i];
        let mut evolving_session: Vec<usize> = evolving_session_base.clone().into_iter().collect();
        evolving_session.truncate(max_items_in_session);
        // println!("{:?}", evolving_session);
        // Binary search + heap + treemap
        let start = Instant::now();
        let closest_neighbors_binary_search_heapplustree =
            calc_similarities_binary_search_heapplustree(
                &evolving_session,
                &historical_sessions_item_id_sorted_truncated,
                &historical_sessions_session_id_sorted_truncated,
                &historical_sessions_max_time_stamp_train,
                n_most_recent_sessions,
                k,
            );
        duration[i] += start.elapsed();
        // Binary search + heap + vectors
        let start = Instant::now();
        let closest_neighbors_binary_search_heapplusvec =
            calc_similarities_binary_search_heapplusvectors(
                &evolving_session,
                &historical_sessions_item_id_sorted_truncated,
                &historical_sessions_session_id_sorted_truncated,
                &historical_sessions_max_time_stamp_train,
                n_most_recent_sessions,
                k,
            );
        duration[i + n_iterations] += start.elapsed();
        // Binary search + heap + hashmap
        let start = Instant::now();
        let closest_neighbors_binary_search_heapplushash =
            calc_similarities_binary_search_heapplushashmap(
                &evolving_session,
                &historical_sessions_item_id_sorted_truncated,
                &historical_sessions_session_id_sorted_truncated,
                &historical_sessions_max_time_stamp_train,
                n_most_recent_sessions,
                k,
            );
        duration[i + 2 * n_iterations] += start.elapsed();
        // Hashmap + heap + hashmap
        let start = Instant::now();
        let closest_neighbors_hashheapplushash = calc_similarities_hashplusheapplushash(
            &evolving_session,
            &historical_sessions_hashed_truncated,
            &historical_sessions_max_time_stamp_train,
            n_most_recent_sessions,
            k,
        );
        duration[i + 3 * n_iterations] += start.elapsed();
        // Hashmap + heap + hashmap
        let start = Instant::now();
        let closest_neighbors_hashheapplushashadj = calc_similarities_hashplusheapplushashadj(
            &evolving_session,
            &historical_sessions_hashed_truncated,
            &historical_sessions_max_time_stamp_train,
            n_most_recent_sessions,
            k,
        );
        duration[i + 4 * n_iterations] += start.elapsed();
        // Assert equivalences between all solutions. NB: for the last two, make sure to sort the hashmap in the output!
        // let sessions_bsht = closest_neighbors_binary_search_heapplustree.into_sorted_vec();
        // let sessions_bshv = closest_neighbors_binary_search_heapplusvec.into_sorted_vec();
        // let sessions_bshh = closest_neighbors_binary_search_heapplushash.into_sorted_vec();
        // let sessions_hhh = closest_neighbors_hashheapplushash.into_sorted_vec();

        // println!("Baseline {:?}", sessions_bshv);
        // println!("Hash truncated {:?}", sessions_hhh);
        // println!("Timestamp session 4891555 {:?}", historical_sessions_max_time_stamp_train[4891555]);
        // println!("Timestamp session 5005933 {:?}", historical_sessions_max_time_stamp_train[5005933]);

        // for (idx, session) in sessions_bshv.iter().enumerate(){
        //     let id_left = session.id;
        //     let score_left = session.score;
        //     assert_eq!(id_left, sessions_bsht[idx].id);
        //     assert_eq!(id_left, sessions_bshh[idx].id);
        //     assert_eq!(id_left, sessions_hhh[idx].id);
        //     // assert_eq!(sessions_bshh[idx].id, sessions_hhh[idx].id);
        //     assert!(approx_eq!(f32, score_left, sessions_bsht[idx].score, epsilon = 0.00001));
        //     assert!(approx_eq!(f32, score_left, sessions_bshh[idx].score, epsilon = 0.00001));
        //     assert!(approx_eq!(f32, score_left, sessions_hhh[idx].score, epsilon = 0.00001));
        //     // assert!(approx_eq!(f32, sessions_bshh[idx].score, sessions_hhh[idx].score, epsilon = 0.00001));
        // }
    }
    for (i, algorithm) in algorithms.iter().enumerate() {
        print_timing(
            &mut duration[(i * n_iterations)..(i + 1) * n_iterations],
            &n_iterations,
            algorithm,
        );
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

fn prepare_hashmap(
    historical_sessions_item_id_sorted: &[usize],
    historical_sessions_session_id_sorted: &[usize],
) -> HashMap<usize, Vec<usize>> {
    // Create hashmap from historical sessions
    let mut historical_sessions_item_ids: Vec<usize> = historical_sessions_item_id_sorted.to_vec();
    historical_sessions_item_ids.dedup();
    let mut historical_sessions_hashed = HashMap::with_capacity(historical_sessions_item_ids.len());

    for (key, current_item) in historical_sessions_item_ids.iter().enumerate() {
        let left_index =
            binary_search_left(historical_sessions_item_id_sorted, current_item).unwrap();
        let right_index =
            binary_search_right(historical_sessions_item_id_sorted, current_item).unwrap();
        let sessions: Vec<usize> =
            historical_sessions_session_id_sorted[left_index..right_index + 1].to_vec();
        historical_sessions_hashed.insert(*current_item, sessions);
    }

    historical_sessions_hashed
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

fn prepare_binary_search_truncated(
    historical_sessions: &[Vec<usize>],
    timestamps: &[usize],
    n_most_recent_sessions: usize,
) -> (Vec<usize>, Vec<usize>) {
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
    let historical_sessions_values_sorted: Vec<usize> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_values[i].clone())
        .collect();
    let historical_sessions_session_indices_sorted: Vec<usize> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_session_indices[i].clone())
        .collect();
    let historical_sessions_timestamps_sorted: Vec<usize> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_timestamps[i].clone())
        .collect();

    // Get unique item_ids
    let mut unique_items = historical_sessions_values_sorted.clone();
    unique_items.dedup();

    let mut historical_sessions_values_sorted_truncated: Vec<usize> =
        Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_session_indices_sorted_truncated: Vec<usize> =
        Vec::with_capacity(historical_sessions_length);

    // Loop over historical sessions to remove all sessions per item older than n_most_recent_sessions
    for current_item in unique_items.iter() {
        let left_index =
            binary_search_left(&historical_sessions_values_sorted, current_item).unwrap();
        let right_index =
            binary_search_right(&historical_sessions_values_sorted, current_item).unwrap();
        let mut current_item_similar_sessions: Vec<usize> =
            historical_sessions_values_sorted[left_index..right_index + 1].to_vec();
        let mut current_item_timestamps: Vec<usize> =
            historical_sessions_timestamps_sorted[left_index..right_index + 1].to_vec();
        let mut current_item_similar_sessions_ids: Vec<usize> =
            historical_sessions_session_indices_sorted[left_index..right_index + 1].to_vec();
        // Sort session ids by reverse timestamp and truncate to n_most_recent_sessions
        let mut timestamp_indices: Vec<usize> = (0..current_item_timestamps.len()).collect();
        timestamp_indices.sort_by_key(|&i| current_item_timestamps[i]);
        let mut current_item_similar_sessions_id_sorted: Vec<usize> = timestamp_indices
            .iter()
            .map(|&i| current_item_similar_sessions_ids[i].clone())
            .collect();
        current_item_similar_sessions_id_sorted.reverse();
        // let mut truncation_length = n_most_recent_sessions;
        // if current_item_similar_sessions_id_sorted.len() > n_most_recent_sessions {
        // let mut current_item_timestamp_sorted: Vec<usize> = timestamp_indices.iter().map(|&i| current_item_timestamps[i].clone()).collect();
        // current_item_timestamp_sorted.reverse();
        // let last_time_stamp = current_item_timestamp_sorted[n_most_recent_sessions - 1];
        // current_item_timestamp_sorted.reverse();
        // let left_index = binary_search_left(&current_item_timestamp_sorted, &last_time_stamp).unwrap();
        // truncation_length = current_item_similar_sessions_id_sorted.len() - left_index;
        // println!("{:?}", truncation_length);
        // if truncation_length > n_most_recent_sessions {
        // }
        // }
        current_item_similar_sessions_id_sorted.truncate(n_most_recent_sessions);
        current_item_similar_sessions.truncate(n_most_recent_sessions);
        // current_item_similar_sessions_id_sorted.sort_unstable();
        // current_item_similar_sessions_id_sorted.reverse();
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
    // (historical_sessions_values_sorted, historical_sessions_session_indices_sorted)
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

// Custom binary search because this is stable unlike the rust default (i.e. this always returns right-most index in case of duplicate entries instead of a random match)
fn binary_search_right_reverse(array: &[usize], key: &usize) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if (unsafe { array.get_unchecked(mid) } < key) {
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

#[derive(PartialEq, Debug)]
pub struct SessionScore {
    pub id: usize,
    pub score: f32,
}

impl SessionScore {
    fn new(id: usize, score: f32) -> Self {
        SessionScore { id, score }
    }
}

/// Ordering for our max-heap, not that we must use a special implementation here as there is no
/// total order on floating point numbers.
fn cmp_reverse(scored_a: &SessionScore, scored_b: &SessionScore) -> Ordering {
    match scored_a.score.partial_cmp(&scored_b.score) {
        Some(Ordering::Less) => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Less,
        _ => Ordering::Equal,
    }
}

impl Eq for SessionScore {}

impl Ord for SessionScore {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_reverse(self, other)
    }
}

impl PartialOrd for SessionScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_reverse(self, other))
    }
}

// Fused insert-and-remove for vectors
fn insert_and_remove<T>(
    mut vector: Vec<T>,
    index_insert: usize,
    index_remove: usize,
    elem_insert: T,
) -> Vec<T> {
    let length = vector.len();
    let vec_ptr = vector.as_mut_ptr();
    assert!(index_insert <= length, "index out of bounds");

    unsafe {
        if index_insert < index_remove {
            ptr::copy(
                vec_ptr.offset(index_insert as isize),
                vec_ptr.offset(index_insert as isize + 1),
                index_remove - index_insert,
            );
            ptr::write(vec_ptr.offset(index_insert as isize), elem_insert);
        } else if index_insert > index_remove {
            ptr::copy(
                vec_ptr.offset(index_remove as isize + 1),
                vec_ptr.offset(index_remove as isize),
                index_insert - index_remove - 1,
            );
            ptr::write(vec_ptr.offset(index_insert as isize - 1), elem_insert);
        } else {
            ptr::write(vec_ptr.offset(index_insert as isize), elem_insert);
        }
    }

    vector
}

fn calc_similarities_binary_search_heapplustree(
    evolving_session: &[usize],
    historical_sessions_item_id_sorted: &[usize],
    historical_sessions_session_id_sorted: &[usize],
    historical_sessions_max_time_stamp: &[usize],
    n_most_recent_sessions: usize,
    k: usize,
) -> BinaryHeap<SessionScore> {
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
            let similar_sessions =
                &historical_sessions_session_id_sorted[left_index..right_index + 1];
            for session_id in similar_sessions {
                let session_present = bst_similarities.contains_key(session_id);
                // If the current historical session is already in the output, just add decay factor to the similarity score.
                if session_present {
                    let old_value = bst_similarities.get_mut(session_id).unwrap();
                    *old_value += decay_factor;
                } else {
                    // Push similar sessions into the output until we reach n_most_recent_sessions
                    if bst_similarities.len() < n_most_recent_sessions {
                        let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                        heap_timestamps.push(SessionTime(session_time_stamp, *session_id));
                        bst_similarities.insert(*session_id, decay_factor);
                    }
                    // Once we reach n_most_recent_sessions in the output, only store new sessions if they are newer than the existing minimum timestamp.
                    else {
                        let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                        if session_time_stamp > heap_timestamps.peek().unwrap().0 {
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
    }

    // Return top-k
    let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    for (session_id, score) in bst_similarities.iter() {
        let scored_session = SessionScore::new(*session_id, *score);
        if closest_neighbors.len() < k {
            closest_neighbors.push(scored_session);
        } else {
            let mut top = closest_neighbors.peek_mut().unwrap();
            if scored_session <= *top {
                *top = scored_session;
            }
        }
    }

    closest_neighbors
}

fn calc_similarities_binary_search_heapplusvectors(
    evolving_session: &[usize],
    historical_sessions_item_id_sorted: &[usize],
    historical_sessions_session_id_sorted: &[usize],
    historical_sessions_max_time_stamp: &[usize],
    n_most_recent_sessions: usize,
    k: usize,
) -> BinaryHeap<SessionScore> {
    // We use a binary heap for the (timestamp, session_id) tuple and two vectors for the (session_id, score) tuples.
    let mut heap_timestamps = BinaryHeap::<SessionTime>::with_capacity(n_most_recent_sessions);
    let mut session_ids_sorted_by_session: Vec<usize> =
        Vec::with_capacity(n_most_recent_sessions + 1);
    let mut similarity_score_sorted_by_session: Vec<f32> =
        Vec::with_capacity(n_most_recent_sessions + 1);

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
            let similar_sessions =
                &historical_sessions_session_id_sorted[left_index..right_index + 1];
            for session_id in similar_sessions {
                let session_index_result = session_ids_sorted_by_session.binary_search(session_id);
                // If the current historical session is already in the output, just add decay factor to the similarity score.
                if session_index_result.is_ok() {
                    let session_index = session_index_result.unwrap();
                    similarity_score_sorted_by_session[session_index] += decay_factor;
                } else {
                    // Push similar sessions into the output until we reach n_most_recent_sessions
                    if session_ids_sorted_by_session.len() < n_most_recent_sessions {
                        let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                        let session_index = session_index_result.unwrap_err();
                        session_ids_sorted_by_session.insert(session_index, *session_id);
                        similarity_score_sorted_by_session.insert(session_index, decay_factor);
                        heap_timestamps.push(SessionTime(session_time_stamp, *session_id));
                    }
                    // Once we reach n_most_recent_sessions in the output, only store new sessions if they are newer than the existing minimum timestamp.
                    else {
                        let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                        if session_time_stamp > heap_timestamps.peek().unwrap().0 {
                            let session_index = session_index_result.unwrap_err();
                            // Remove the the existing minimum time stamp.
                            let session_id_remove = heap_timestamps.peek().unwrap().1;
                            let session_index_remove = session_ids_sorted_by_session
                                .binary_search(&session_id_remove)
                                .unwrap();
                            // Fused insert-and-remove for the vectors reduces computation time
                            session_ids_sorted_by_session = insert_and_remove(
                                session_ids_sorted_by_session,
                                session_index,
                                session_index_remove,
                                *session_id,
                            );
                            similarity_score_sorted_by_session = insert_and_remove(
                                similarity_score_sorted_by_session,
                                session_index,
                                session_index_remove,
                                decay_factor,
                            );
                            // Set new minimum timestamp in heap_timestamp
                            let mut bottom = heap_timestamps.peek_mut().unwrap();
                            *bottom = SessionTime(session_time_stamp, *session_id);
                        }
                    }
                }
            }
        }
    }

    // Return top-k
    let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    for (key, session_id) in session_ids_sorted_by_session.iter().enumerate() {
        let score = similarity_score_sorted_by_session[key];
        let scored_session = SessionScore::new(*session_id, score);
        if closest_neighbors.len() < k {
            closest_neighbors.push(scored_session);
        } else {
            let mut top = closest_neighbors.peek_mut().unwrap();
            if scored_session <= *top {
                *top = scored_session;
            }
        }
    }

    closest_neighbors
}

fn calc_similarities_binary_search_heapplushashmap(
    evolving_session: &[usize],
    historical_sessions_item_id_sorted: &[usize],
    historical_sessions_session_id_sorted: &[usize],
    historical_sessions_max_time_stamp: &[usize],
    n_most_recent_sessions: usize,
    k: usize,
) -> BinaryHeap<SessionScore> {
    // We use a binary heap for the (timestamp, session_id) tuple and a hashmap for the (session_id, score) tuples.
    let mut heap_timestamps = BinaryHeap::<SessionTime>::with_capacity(n_most_recent_sessions);
    let mut hash_similarities: HashMap<usize, f32> =
        HashMap::with_capacity(n_most_recent_sessions + 1);

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
            let similar_sessions =
                &historical_sessions_session_id_sorted[left_index..right_index + 1];
            for session_id in similar_sessions {
                let session_present = hash_similarities.contains_key(session_id);
                if session_present {
                    let value = hash_similarities.get_mut(session_id).unwrap();
                    *value += decay_factor;
                } else {
                    let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                    if hash_similarities.len() < n_most_recent_sessions {
                        hash_similarities.insert(*session_id, decay_factor);
                        heap_timestamps.push(SessionTime(session_time_stamp, *session_id));
                    } else {
                        if session_time_stamp > heap_timestamps.peek().unwrap().0 {
                            // Insert new key
                            hash_similarities.insert(*session_id, decay_factor);
                            // Remove the the existing minimum time stamp.
                            let session_id_remove = heap_timestamps.peek().unwrap().1;
                            hash_similarities.remove_entry(&session_id_remove);
                            // Set new minimum timestamp
                            let mut bottom = heap_timestamps.peek_mut().unwrap();
                            *bottom = SessionTime(session_time_stamp, *session_id);
                        }
                    }
                }
            }
        }
    }

    // Return top-k
    let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    for (session_id, score) in hash_similarities.iter() {
        let scored_session = SessionScore::new(*session_id, *score);
        if closest_neighbors.len() < k {
            closest_neighbors.push(scored_session);
        } else {
            let mut top = closest_neighbors.peek_mut().unwrap();
            if scored_session < *top {
                *top = scored_session;
            }
        }
    }

    // Return top-k in sorted order (needed for asserting equivalence to other algos, but slower)
    // let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    // for session_id in hash_similarities.keys().sorted() {
    //     let score = hash_similarities.get(session_id).unwrap();
    //     let scored_session = SessionScore::new(*session_id, *score);
    //     if closest_neighbors.len() < k {
    //         closest_neighbors.push(scored_session);
    //     } else {
    //         let mut top = closest_neighbors.peek_mut().unwrap();
    //         if scored_session <= *top {
    //             *top = scored_session;
    //         }
    //     }
    // }

    closest_neighbors
}

fn calc_similarities_hashplusheapplushash(
    evolving_session: &[usize],
    historical_sessions: &HashMap<usize, Vec<usize>>,
    historical_sessions_max_time_stamp: &[usize],
    n_most_recent_sessions: usize,
    k: usize,
) -> BinaryHeap<SessionScore> {
    // We use a binary heap for the (timestamp, session_id) tuple and a hashmap for the (session_id, score) tuples.
    let mut heap_timestamps = BinaryHeap::<SessionTime>::with_capacity(n_most_recent_sessions);
    let mut hash_similarities = HashMap::with_capacity(n_most_recent_sessions);

    //  Loop over items in evolving session
    for (pos, item_id) in evolving_session.iter().enumerate() {
        // Check if item is in history
        let in_history = historical_sessions.contains_key(item_id);
        // If match found, return vec with session_ids
        if in_history {
            // Calculate position weight
            let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
            // Loop over all historical sessions with a match
            let similar_sessions = historical_sessions.get(item_id).unwrap();
            'session_loop: for session_id in similar_sessions {
                let session_present = hash_similarities.contains_key(session_id);
                if session_present {
                    let value = hash_similarities.get_mut(session_id).unwrap();
                    *value += decay_factor;
                } else {
                    let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                    if hash_similarities.len() < n_most_recent_sessions {
                        hash_similarities.insert(*session_id, decay_factor);
                        heap_timestamps.push(SessionTime(session_time_stamp, *session_id));
                    } else {
                        if session_time_stamp > heap_timestamps.peek().unwrap().0 {
                            // Remove the the existing minimum time stamp.
                            let session_id_remove = heap_timestamps.peek().unwrap().1;
                            hash_similarities.remove(&session_id_remove);
                            // Set new minimum timestamp
                            hash_similarities.insert(*session_id, decay_factor);
                            let mut bottom = heap_timestamps.peek_mut().unwrap();
                            *bottom = SessionTime(session_time_stamp, *session_id);
                        } else {
                            break 'session_loop;
                        }
                    }
                }
            }
        }
    }

    // Return top-k
    let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    for (session_id, score) in hash_similarities.iter() {
        let scored_session = SessionScore::new(*session_id, *score);
        if closest_neighbors.len() < k {
            closest_neighbors.push(scored_session);
        } else {
            let mut top = closest_neighbors.peek_mut().unwrap();
            if scored_session < *top {
                *top = scored_session;
            }
        }
    }

    // Return top-k in sorted order (needed for asserting equivalence to other algos, but slower)
    // let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    // for session_id in hash_similarities.keys().sorted() {
    //     let score = hash_similarities.get(session_id).unwrap();
    //     let scored_session = SessionScore::new(*session_id, *score);
    //     if closest_neighbors.len() < k {
    //         closest_neighbors.push(scored_session);
    //     } else {
    //         let mut top = closest_neighbors.peek_mut().unwrap();
    //         if scored_session <= *top {
    //             *top = scored_session;
    //         }
    //     }
    // }

    closest_neighbors
}

fn calc_similarities_hashplusheapplushashadj(
    evolving_session: &[usize],
    historical_sessions: &HashMap<usize, Vec<usize>>,
    historical_sessions_max_time_stamp: &[usize],
    n_most_recent_sessions: usize,
    k: usize,
) -> BinaryHeap<SessionScore> {
    // We use a binary heap for the (timestamp, session_id) tuple and a hashmap for the (session_id, score) tuples.
    let mut heap_timestamps = OctonaryHeap::<SessionTime>::with_capacity(n_most_recent_sessions);
    let mut hash_similarities = HashMap::with_capacity(n_most_recent_sessions);

    //  Loop over items in evolving session
    for (pos, item_id) in evolving_session.iter().enumerate() {
        // Check if item is in history
        let in_history = historical_sessions.contains_key(item_id);
        // If match found, return vec with session_ids
        if in_history {
            // Calculate position weight
            let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
            // Loop over all historical sessions with a match
            let similar_sessions = historical_sessions.get(item_id).unwrap();
            'session_loop: for session_id in similar_sessions {
                let session_present = hash_similarities.contains_key(session_id);
                if session_present {
                    let value = hash_similarities.get_mut(session_id).unwrap();
                    *value += decay_factor;
                } else {
                    let session_time_stamp = historical_sessions_max_time_stamp[*session_id];
                    if hash_similarities.len() < n_most_recent_sessions {
                        hash_similarities.insert(*session_id, decay_factor);
                        heap_timestamps.push(SessionTime(session_time_stamp, *session_id));
                    } else {
                        if session_time_stamp > heap_timestamps.peek().unwrap().0 {
                            // Remove the the existing minimum time stamp.
                            let session_id_remove = heap_timestamps.peek().unwrap().1;
                            hash_similarities.remove(&session_id_remove);
                            // Set new minimum timestamp
                            hash_similarities.insert(*session_id, decay_factor);
                            let mut bottom = heap_timestamps.peek_mut().unwrap();
                            *bottom = SessionTime(session_time_stamp, *session_id);
                        } else {
                            break 'session_loop;
                        }
                    }
                }
            }
        }
    }

    // Return top-k
    let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    for (session_id, score) in hash_similarities.iter() {
        let scored_session = SessionScore::new(*session_id, *score);
        if closest_neighbors.len() < k {
            closest_neighbors.push(scored_session);
        } else {
            let mut top = closest_neighbors.peek_mut().unwrap();
            if scored_session < *top {
                *top = scored_session;
            }
        }
    }

    // Return top-k in sorted order (needed for asserting equivalence to other algos, but slower)
    // let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
    // for session_id in hash_similarities.keys().sorted() {
    //     let score = hash_similarities.get(session_id).unwrap();
    //     let scored_session = SessionScore::new(*session_id, *score);
    //     if closest_neighbors.len() < k {
    //         closest_neighbors.push(scored_session);
    //     } else {
    //         let mut top = closest_neighbors.peek_mut().unwrap();
    //         if scored_session <= *top {
    //             *top = scored_session;
    //         }
    //     }
    // }

    closest_neighbors
}
