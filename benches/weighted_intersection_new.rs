#[macro_use]
extern crate bencher;
extern crate hashbrown;
extern crate rand;

use bencher::Bencher;
use hashbrown::HashSet;
use rand::rngs::ThreadRng;
use rand::Rng;

const NUM_ITEMS_IN_EVOLVING_SESSION: usize = 10;
const MAX_ITEM_ID: u64 = 22_278_380;
const MAX_NUM_ITEMS_IN_HISTORY_SESSION: usize = 38;
const NUM_HISTORY_SESSIONS: usize = 500;

benchmark_group!(
    benches,
    bench_nested_loops,
    bench_binary_search,
    bench_linear_search
);
benchmark_main!(benches);

fn bench_nested_loops(bench: &mut Bencher) {
    // Retrieve evolving session and historical sessions
    let mut rng = rand::thread_rng();
    let historical_sessions = get_historical_sessions(&mut rng);
    let evolving_session = get_evolving_session(&mut rng);

    // Prepare algorithms
    let historical_sessions_hashed = prepare_nested_loops(&historical_sessions);

    // Run bencher
    bench.iter(|| {
        bencher::black_box(nested_loops(&evolving_session, &historical_sessions_hashed));
    });
}

fn bench_binary_search(bench: &mut Bencher) {
    // Retrieve evolving session and historical sessions
    let mut rng = rand::thread_rng();
    let historical_sessions = get_historical_sessions(&mut rng);
    let evolving_session = get_evolving_session(&mut rng);

    // Prepare algorithms
    let (historical_sessions_values_sorted, historical_sessions_session_indices_sorted) =
        prepare_binary_search(&historical_sessions);

    // Run bencher
    bench.iter(|| {
        bencher::black_box(binary_search(
            &evolving_session,
            &historical_sessions_values_sorted,
            &historical_sessions_session_indices_sorted,
        ));
    });
}

fn bench_linear_search(bench: &mut Bencher) {
    // Retrieve evolving session and historical sessions
    let mut rng = rand::thread_rng();
    let historical_sessions = get_historical_sessions(&mut rng);
    let evolving_session = get_evolving_session(&mut rng);

    // Prepare algorithms
    let (historical_sessions_values_sorted, historical_sessions_session_indices_sorted) =
        prepare_binary_search(&historical_sessions);

    // Run bencher
    bench.iter(|| {
        bencher::black_box(linear_search(
            &evolving_session,
            &historical_sessions_values_sorted,
            &historical_sessions_session_indices_sorted,
        ));
    });
}

fn get_evolving_session(rng: &mut ThreadRng) -> Vec<u64> {
    let evolving_session: Vec<u64> = (0..NUM_ITEMS_IN_EVOLVING_SESSION)
        .map(|_| rng.gen_range(0..MAX_ITEM_ID))
        .collect();

    evolving_session
}

fn get_historical_sessions(rng: &mut ThreadRng) -> Vec<Vec<u64>> {
    let historical_sessions: Vec<Vec<u64>> = (0..NUM_HISTORY_SESSIONS)
        .map(|_| {
            let session_length = rng.gen_range(1..MAX_NUM_ITEMS_IN_HISTORY_SESSION);

            let mut history_session = Vec::with_capacity(session_length);

            for _ in 0..session_length {
                history_session.push(rng.gen_range(0..MAX_ITEM_ID));
            }
            history_session
        })
        .collect();

    historical_sessions
}

fn prepare_nested_loops(historical_sessions: &Vec<Vec<u64>>) -> Vec<HashSet<u64>> {
    // Create hashtable from historical sessions
    let historical_sessions_hashed: Vec<HashSet<u64>> = (0..NUM_HISTORY_SESSIONS)
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

fn nested_loops(evolving_session: &[u64], historical_sessions: &Vec<HashSet<u64>>) -> Vec<f64> {
    let mut similarities = Vec::with_capacity(NUM_HISTORY_SESSIONS);

    for neighbor_session in historical_sessions.iter() {
        let mut similarity = 0_f64;

        for (pos, item_id) in evolving_session.iter().enumerate() {
            if neighbor_session.contains(&item_id) {
                let decay_factor = (pos + 1) as f64 / evolving_session.len() as f64;
                similarity += decay_factor;
            }
        }
        similarities.push(similarity);
    }

    similarities
}

fn prepare_binary_search(historical_sessions: &Vec<Vec<u64>>) -> (Vec<u64>, Vec<u64>) {
    // Initialize arrays
    let historical_sessions_length: usize = historical_sessions.iter().map(|x| x.len()).sum();
    let mut historical_sessions_values = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_session_indices = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_indices = Vec::with_capacity(historical_sessions_length);
    let mut iterable = 0_u64;

    // Create (i) vector of historical sessions, (ii) vector of historical session indices, (iii) vector of session indices
    for (session_id, session) in historical_sessions.iter().enumerate() {
        for (item_id, _) in session.iter().enumerate() {
            historical_sessions_values.push(historical_sessions[session_id][item_id]);
            historical_sessions_indices.push(iterable);
            historical_sessions_session_indices.push(session_id as u64);
            iterable += 1;
        }
    }

    // Sort historical session values and session indices array
    historical_sessions_indices.sort_by_key(|&i| &historical_sessions_values[i as usize]);
    let historical_sessions_values_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_values[i as usize].clone())
        .collect();
    let historical_sessions_session_indices_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_session_indices[i as usize].clone())
        .collect();

    // Return sorted flattened historical sessions and sorted flattened historical session indices
    (
        historical_sessions_values_sorted,
        historical_sessions_session_indices_sorted,
    )
}

fn binary_search(
    evolving_session: &Vec<u64>,
    historical_sessions_values_sorted: &Vec<u64>,
    historical_sessions_session_indices_sorted: &Vec<u64>,
) -> Vec<f64> {
    // Sort evolving session (should be part of benchmark, imho)
    // let mut evolving_session_indices: Vec<u64> = (0..(NUM_ITEMS_IN_EVOLVING_SESSION as u64)).into_iter().collect();
    // evolving_session_indices.sort_unstable_by_key(|&i| &evolving_session[i as usize]);
    // let evolving_session_sorted: Vec<u64> = evolving_session_indices.iter().map(|&i| evolving_session[i as usize].clone()).collect();

    let mut similarities = vec![0.0; NUM_HISTORY_SESSIONS];
    //  Loop over items in evolving session
    for (pos, item_id) in evolving_session.iter().enumerate() {
        // Find index of current item in historical session vector using binary search
        let index = historical_sessions_values_sorted.binary_search(&item_id);
        // If match found, find duplicate elements left and right to the match found; this is because binary search returns a random match (unlike the Torch / Numpy functions where we can indicate that we'd like the left or right first match - this is an open issue on Rust Git)
        if index.is_ok() {
            // Calculate position weight
            let decay_factor = (pos + 1) as f64 / NUM_ITEMS_IN_EVOLVING_SESSION as f64;
            // Fill first similarity
            let first_index = index.unwrap();
            let current_similarity_index =
                historical_sessions_session_indices_sorted[first_index] as usize;
            similarities[current_similarity_index] += decay_factor;
            // Set left and right indices
            let mut left_index = (first_index - 1) as i64;
            let mut right_index = first_index + 1;
            // Count duplicate elements left to first index;
            while left_index >= 0 {
                if historical_sessions_values_sorted[left_index as usize] == *item_id {
                    let current_similarity_index =
                        historical_sessions_session_indices_sorted[left_index as usize] as usize;
                    similarities[current_similarity_index] += decay_factor;
                }
                left_index -= 1;
            }
            // Count duplicate elements right to first index - note that empirically it seems Rust's binary search always returns the most right index, causing the below loop never to be triggered;
            while right_index < historical_sessions_values_sorted.len() {
                if historical_sessions_values_sorted[right_index] == *item_id {
                    let current_similarity_index =
                        historical_sessions_session_indices_sorted[right_index] as usize;
                    similarities[current_similarity_index] += decay_factor;
                }
                right_index += 1;
            }
        }
    }
    similarities
}

fn linear_search(
    evolving_session: &Vec<u64>,
    historical_sessions_values_sorted: &Vec<u64>,
    historical_sessions_session_indices_sorted: &Vec<u64>,
) -> Vec<f64> {
    let mut similarities = vec![0.0; NUM_HISTORY_SESSIONS];
    for (pos, item_evolving_session) in evolving_session.iter().enumerate() {
        // Calculate position weight
        let decay_factor = (pos + 1) as f64 / NUM_ITEMS_IN_EVOLVING_SESSION as f64;
        for (item_id, item) in historical_sessions_values_sorted.iter().enumerate() {
            if *item_evolving_session == *item {
                let current_similarity_index =
                    historical_sessions_session_indices_sorted[item_id] as usize;
                similarities[current_similarity_index] += decay_factor;
            }
        }
    }
    similarities
}
