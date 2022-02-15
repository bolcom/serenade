#![allow(warnings)]

extern crate hashbrown;
extern crate num_format;
extern crate rand;

use hashbrown::HashSet;
use num_format::{Locale, ToFormattedString};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

// const NUM_ITEMS_IN_EVOLVING_SESSION: usize = 2;
// const MAX_ITEM_ID: usize = 5;
// const MAX_NUM_ITEMS_IN_HISTORY_SESSION: usize = 5;
const NUM_ITEMS_IN_EVOLVING_SESSION: usize = 10;
const MAX_ITEM_ID: usize = 22_278_380;
const MAX_NUM_ITEMS_IN_HISTORY_SESSION: usize = 38;

fn main() {
    // Retrieve evolving session and historical sessions
    let mut rng = rand::thread_rng();
    let NUM_HISTORY_SESSIONS: usize = 500;
    let historical_sessions = get_historical_sessions(&mut rng, NUM_HISTORY_SESSIONS);
    // println!("Historical sessions: {:?}", historical_sessions);
    // println!("Evolving session: {:?}", evolving_session);

    // Prepare algorithms
    let (historical_sessions_item_id_sorted, historical_sessions_session_id_sorted) =
        prepare_binary_search(&historical_sessions);
    let (historical_sessions_item_id_sorted_eytzinger, eytzinger_permutation) =
        prepare_eytzinger_search(&historical_sessions_item_id_sorted);
    let historical_sessions_hashed = prepare_nested_loops(&historical_sessions);

    // Simple benchmark loop, to allow better control over inputs
    let n_iterations: usize = 100_000;
    let mut rng = rand::thread_rng();
    let algorithms = [
        "binary_search",
        "eytzgr_search",
        "nested_loops ",
        "linear_search",
    ];
    // let algorithms = ["binary_search", "eytzgr_search"];
    let mut duration: Vec<Duration> = vec![Duration::new(0, 0); n_iterations * algorithms.len()];

    for i in 0..n_iterations {
        // Choose random evolving session
        let evolving_session = get_evolving_session(&mut rng);
        // Binary search
        let start = Instant::now();
        let similarities_binary_search = calc_similarities_binary_search(
            &evolving_session,
            &historical_sessions_item_id_sorted,
            &historical_sessions_session_id_sorted,
            NUM_HISTORY_SESSIONS,
        );
        duration[i] += start.elapsed();
        // Eytzinger search
        let start = Instant::now();
        let similarities_eytzinger_search = calc_similarities_eytzinger_search(
            &evolving_session,
            &historical_sessions_item_id_sorted,
            &historical_sessions_item_id_sorted_eytzinger,
            &eytzinger_permutation,
            &historical_sessions_session_id_sorted,
            NUM_HISTORY_SESSIONS,
        );
        duration[i + n_iterations] += start.elapsed();
        assert_eq!(similarities_binary_search, similarities_eytzinger_search);
        // Nested loops
        let start = Instant::now();
        let similarities_nested_loops =
            calc_similarities_nested_loops(&evolving_session, &historical_sessions_hashed);
        duration[i + 2 * n_iterations] += start.elapsed();
        assert_eq!(similarities_binary_search, similarities_nested_loops);
        // Linear search
        let start = Instant::now();
        let similarities_linear_search = calc_similarities_linear_search(
            &evolving_session,
            &historical_sessions_item_id_sorted,
            &historical_sessions_session_id_sorted,
            NUM_HISTORY_SESSIONS,
        );
        duration[i + 3 * n_iterations] += start.elapsed();
        assert_eq!(similarities_binary_search, similarities_linear_search);
    }
    for (i, algorithm) in algorithms.iter().enumerate() {
        print_timing(
            &duration[(i * n_iterations)..(i + 1) * n_iterations],
            &n_iterations,
            algorithm,
        );
    }
}

fn print_timing(duration: &[Duration], n_iterations: &usize, name: &str) {
    let duration_total: Duration = duration.iter().sum();
    let ns_per_iter = (duration_total.as_secs() * 1_000_000_000
        + (duration_total.subsec_nanos() as u64))
        / *n_iterations as u64;
    let ns_per_iter_string = ns_per_iter.to_formatted_string(&Locale::en);
    println!("test {:}: {:>20} ns/iter", name, ns_per_iter_string)
}

fn get_evolving_session(rng: &mut ThreadRng) -> Vec<usize> {
    let evolving_session: Vec<usize> = (0..NUM_ITEMS_IN_EVOLVING_SESSION)
        .map(|_| rng.gen_range(0..MAX_ITEM_ID))
        .collect();

    evolving_session
}

fn get_historical_sessions(rng: &mut ThreadRng, NUM_HISTORY_SESSIONS: usize) -> Vec<Vec<usize>> {
    let historical_sessions: Vec<Vec<usize>> = (0..NUM_HISTORY_SESSIONS)
        .map(|_| {
            let session_length = rng.gen_range(1..MAX_NUM_ITEMS_IN_HISTORY_SESSION);

            let mut history_session: Vec<usize> = Vec::with_capacity(session_length);

            for _ in 0..session_length {
                history_session.push(rng.gen_range(0..MAX_ITEM_ID));
            }
            history_session.sort();
            history_session.dedup();
            history_session
        })
        .collect();

    historical_sessions
}

// fn get_most_recent_neigbors(evolving_session: &[usize]){

//     let mut most_recent_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(m);

//     for session_item in evolving_session.iter() {
//         match index.sessions_for_item(session_item) {
//             Some(sessions) => {
//                 for session in sessions {
//                     let max_order = index.max_order_for_session(session);
//                     let session_with_age = SessionScore::new(*session, max_order as f64);
//                     if most_recent_neighbors.len() < m {
//                         most_recent_neighbors.push(session_with_age);
//                     } else {
//                         let mut top = most_recent_neighbors.peek_mut().unwrap();
//                         if session_with_age < *top {
//                             *top = session_with_age;
//                         }
//                     }
//                 }
//             },
//             None => (),
//         }
//     }
// }

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

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if (array[mid] > *key) {
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

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if (array[mid] < *key) {
            bottom = mid + 1;
        } else {
            top = mid;
        }
    }

    if array[top] == *key {
        Ok(top)
    } else {
        Err(top)
    }
}

fn calc_similarities_binary_search(
    evolving_session: &Vec<usize>,
    historical_sessions_item_id_sorted: &Vec<usize>,
    historical_sessions_session_id_sorted: &Vec<usize>,
    NUM_HISTORY_SESSIONS: usize,
) -> Vec<f32> {
    //  Loop over items in evolving session
    let mut similarities: Vec<f32> = vec![0.0; NUM_HISTORY_SESSIONS];

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
            // Fill similarities
            let current_similarity_index =
                &historical_sessions_session_id_sorted[left_index..right_index + 1];
            current_similarity_index
                .iter()
                .for_each(|i| similarities[*i] += decay_factor);
        }
    }
    similarities
}

fn calc_similarities_linear_search(
    evolving_session: &Vec<usize>,
    historical_sessions_item_id_sorted: &Vec<usize>,
    historical_sessions_session_id_sorted: &Vec<usize>,
    NUM_HISTORY_SESSIONS: usize,
) -> Vec<f32> {
    //  Loop over items in evolving session
    let mut similarities: Vec<f32> = vec![0.0; NUM_HISTORY_SESSIONS];
    for (pos, item_evolving_session) in evolving_session.iter().enumerate() {
        // Calculate position weight
        let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
        for (item_id, item) in historical_sessions_item_id_sorted.iter().enumerate() {
            if item_evolving_session == item {
                let current_similarity_index = historical_sessions_session_id_sorted[item_id];
                similarities[current_similarity_index] += decay_factor;
            }
        }
    }
    similarities
}

fn prepare_eytzinger_search(array: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let n = array.len();
    let mut array_e = vec![0; n + 1];
    let mut permutation = vec![0; n + 1];
    let mut ind_arr: usize = 0;
    let mut ind_eyt: usize = 1;
    // Create eytzinger layout
    eytzinger(&array, &mut array_e, &mut permutation, ind_arr, ind_eyt, n);

    (array_e, permutation)
}

// https://algorithmica.org/en/eytzinger
fn eytzinger(
    array: &Vec<usize>,
    array_e: &mut Vec<usize>,
    permutation: &mut Vec<usize>,
    mut ind_arr: usize,
    mut ind_eyt: usize,
    n: usize,
) -> (usize) {
    if (ind_eyt <= n) {
        ind_arr = eytzinger(array, array_e, permutation, ind_arr, 2 * ind_eyt, n);
        array_e[ind_eyt] = array[ind_arr];
        permutation[ind_eyt] = ind_arr;
        ind_arr += 1;
        ind_arr = eytzinger(array, array_e, permutation, ind_arr, 2 * ind_eyt + 1, n);
    }
    ind_arr
}

fn eytzinger_search(array_e: &Vec<usize>, key: &usize) -> Result<usize, usize> {
    let mut k = 1;
    let n = array_e.len() - 1;
    while k <= n {
        k = 2 * k + (array_e[k] < *key) as usize;
    }

    k >>= (!k).trailing_zeros() + 1;
    if array_e[k] == *key {
        Ok(k)
    } else {
        Err(k)
    }
}

fn calc_similarities_eytzinger_search(
    evolving_session: &Vec<usize>,
    historical_sessions_item_id_sorted: &Vec<usize>,
    historical_sessions_item_id_sorted_eytzinger: &Vec<usize>,
    eytzinger_permutation: &Vec<usize>,
    historical_sessions_session_id_sorted: &Vec<usize>,
    NUM_HISTORY_SESSIONS: usize,
) -> Vec<f32> {
    //  Loop over items in evolving session
    let mut similarities: Vec<f32> = vec![0.0; NUM_HISTORY_SESSIONS];
    for (pos, item_id) in evolving_session.iter().enumerate() {
        // Find index of current item in historical session vector using binary search
        let index = eytzinger_search(historical_sessions_item_id_sorted_eytzinger, item_id);
        // If match found, find duplicate elements left to the match found; this is because binary search returns the right most match.
        if index.is_ok() {
            // Calculate position weight
            let decay_factor = (pos + 1) as f32 / evolving_session.len() as f32;
            // Fill first similarity
            let first_index_eytzinger = index.unwrap();
            let first_index = eytzinger_permutation[first_index_eytzinger];
            let current_similarity_index = historical_sessions_session_id_sorted[first_index];
            similarities[current_similarity_index] += decay_factor;
            // Set next index, first left and then right
            let mut next_index = first_index - 1;
            // Count duplicate elements left to first index;
            while next_index < first_index {
                if historical_sessions_item_id_sorted[next_index] == *item_id {
                    let current_similarity_index =
                        historical_sessions_session_id_sorted[next_index];
                    similarities[current_similarity_index] += decay_factor;
                    next_index -= 1;
                } else {
                    next_index = first_index;
                }
            }
            next_index = first_index + 1;
            // Count duplicate elements right to first index; the eytzinger search is not stable
            while (next_index < historical_sessions_item_id_sorted.len()) {
                if historical_sessions_item_id_sorted[next_index] == *item_id {
                    let current_similarity_index =
                        historical_sessions_session_id_sorted[next_index];
                    similarities[current_similarity_index] += decay_factor;
                    next_index += 1;
                } else {
                    next_index = historical_sessions_item_id_sorted.len()
                }
            }
        }
    }
    similarities
}
