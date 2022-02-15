#[macro_use]
extern crate bencher;
extern crate hashbrown;
extern crate rand;
extern crate sprs;
extern crate tch;

use bencher::Bencher;
use rand::Rng;

use sprs::CsMat;
use sprs::TriMat;

use hashbrown::HashSet;
use tch::{IndexOp, Kind, Tensor};

// benchmark_group!(benches, bench_nested_loops, bench_sprs_matrix_vector_multiplication,
//  bench_torch_matrix_vector_multiplication, bench_binarysearch);
// benchmark_group!(benches, bench_nested_loops, bench_intersection, bench_binarysearch);
benchmark_group!(benches, bench_nested_loops, bench_binarysearch);
benchmark_main!(benches);

const NUM_ITEMS_IN_EVOLVING_SESSION: usize = 10;
const MAX_ITEM_ID: u64 = 2_2278_380;

const MAX_NUM_ITEMS_IN_HISTORY_SESSION: usize = 38;
const NUM_HISTORY_SESSIONS: usize = 500;

fn bench_nested_loops(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();

    let evolving_session: Vec<u64> = (0..NUM_ITEMS_IN_EVOLVING_SESSION)
        .map(|_| rng.gen_range(0..MAX_ITEM_ID))
        .collect();

    let historical_sessions: Vec<HashSet<u64>> = (0..NUM_HISTORY_SESSIONS)
        .map(|_| {
            let mut history_session = HashSet::new();

            let session_length = rng.gen_range(1..MAX_NUM_ITEMS_IN_HISTORY_SESSION);
            for _ in 0..session_length {
                history_session.insert(rng.gen_range(0..MAX_ITEM_ID));
            }
            history_session
        })
        .collect();

    bench.iter(|| {
        bencher::black_box(nested_loops(&evolving_session, &historical_sessions));
    });
}

fn nested_loops(evolving_session: &[u64], history_sessions: &Vec<HashSet<u64>>) -> Vec<f64> {
    let mut similarities = Vec::with_capacity(NUM_HISTORY_SESSIONS);

    for neighbor_session in history_sessions.iter() {
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

fn bench_sprs_matrix_vector_multiplication(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();

    let mut evolving_session = TriMat::new((MAX_ITEM_ID as usize, 1));

    for _ in 0..NUM_ITEMS_IN_EVOLVING_SESSION {
        let item_id = rng.gen_range(0..MAX_ITEM_ID) as usize;
        evolving_session.add_triplet(item_id, 0, 1.0);
    }

    let mut historical_sessions = TriMat::new((NUM_HISTORY_SESSIONS, MAX_ITEM_ID as usize));

    for history_session in 0..NUM_HISTORY_SESSIONS {
        let session_length = rng.gen_range(1..MAX_NUM_ITEMS_IN_HISTORY_SESSION);
        for _ in 0..session_length {
            let item_id = rng.gen_range(0..MAX_ITEM_ID) as usize;
            historical_sessions.add_triplet(history_session, item_id, 1.0);
        }
    }

    let evolving_session = evolving_session.to_csc();
    let historical_sessions = historical_sessions.to_csr();

    bench.iter(|| {
        bencher::black_box(sprs_matrix_vector_multiplication(
            &evolving_session,
            &historical_sessions,
        ));
    });
}

fn sprs_matrix_vector_multiplication(
    evolving_session: &CsMat<f64>,
    historical_session: &CsMat<f64>,
) -> CsMat<f64> {
    historical_session * evolving_session
}

/*
import torch
import time
sessions = 500
items = 1000000
items_session = 100
device = torch.device(0)
hist_sessions = torch.randint(0, 2, (sessions, items), device=device)
evolving_session = torch.randperm(items, device=device)[:items_session]
start = time.perf_counter()
weights = 1 / torch.arange(1, items_session + 1, device=device)
similarity = (weights * hist_sessions.index_select(1, evolving_session)).sum()
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Time: {end-start:.5f}s')
*/

fn bench_torch_matrix_vector_multiplication(bench: &mut Bencher) {
    let sessions = NUM_HISTORY_SESSIONS as i64;
    let items = MAX_ITEM_ID as i64;
    let cpu = tch::Device::Cpu;

    let historical_sessions = tch::Tensor::randint(2, &[items, sessions], (Kind::Bool, cpu));
    let weights =
        tch::Tensor::arange(NUM_ITEMS_IN_EVOLVING_SESSION as f64, (Kind::Float, cpu)).unsqueeze(-1);
    let evolving_session = tch::Tensor::randperm(MAX_ITEM_ID as i64, (Kind::Int64, cpu))
        .i(0..(NUM_ITEMS_IN_EVOLVING_SESSION as i64));

    bench.iter(|| {
        bencher::black_box(torch_matrix_vector_multiplication(
            &evolving_session,
            &historical_sessions,
            &weights,
        ))
    })
}

fn torch_matrix_vector_multiplication(
    evolving_session: &Tensor,
    historical_sessions: &Tensor,
    weights: &Tensor,
) -> Tensor {
    (weights * historical_sessions.index_select(0_i64, evolving_session)).sum1(
        &[0_i64],
        false,
        Kind::Float,
    )
}

fn bench_intersection(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();

    let mut sorted_evolving_session: Vec<u64> = (0..NUM_ITEMS_IN_EVOLVING_SESSION)
        .map(|_| rng.gen_range(0..MAX_ITEM_ID))
        .collect();

    sorted_evolving_session.sort();

    // Dummy
    let position_weights: Vec<f64> = (0..NUM_ITEMS_IN_EVOLVING_SESSION)
        .map(|pos| 1.0_f64 / pos as f64)
        .collect();

    let historical_sessions: Vec<Vec<u64>> = (0..NUM_HISTORY_SESSIONS)
        .map(|_| {
            let mut history_session = Vec::new();

            let session_length = rng.gen_range(1..MAX_NUM_ITEMS_IN_HISTORY_SESSION);
            for _ in 0..session_length {
                history_session.push(rng.gen_range(0..MAX_ITEM_ID));
            }
            history_session.sort();
            history_session
        })
        .collect();

    bench.iter(|| {
        bencher::black_box(intersection(
            &sorted_evolving_session,
            &position_weights,
            &historical_sessions,
        ));
    });
}

fn intersection(
    sorted_evolving_session: &[u64],
    position_weights: &[f64],
    sorted_history_sessions: &Vec<Vec<u64>>,
) -> Vec<f64> {
    let mut similarities = Vec::with_capacity(NUM_HISTORY_SESSIONS);

    for neighbor_session in sorted_history_sessions.iter() {
        let mut similarity = 0_f64;

        let m = sorted_evolving_session.len();
        let n = sorted_history_sessions.len();

        let mut i: usize = 0;
        let mut j: usize = 0;

        while i < m && j < n {
            let i_smaller_j = unsafe { sorted_evolving_session.get_unchecked(i) }
                < unsafe { neighbor_session.get_unchecked(j) };
            let j_smaller_i = unsafe { neighbor_session.get_unchecked(j) }
                < unsafe { sorted_evolving_session.get_unchecked(i) };
            let i_equal_j = !i_smaller_j && !j_smaller_i;

            similarity = similarity
                + (i_equal_j as usize) as f64 * unsafe { position_weights.get_unchecked(i) };

            i = i + i_smaller_j as usize + i_equal_j as usize;
            j = j + j_smaller_i as usize + i_equal_j as usize;
        }

        similarities.push(similarity);
    }

    similarities
}

fn bench_binarysearch(bench: &mut Bencher) {
    let sessions = NUM_HISTORY_SESSIONS as i64;
    let items_hist_session = MAX_NUM_ITEMS_IN_HISTORY_SESSION as i64;
    let items_session = NUM_ITEMS_IN_EVOLVING_SESSION as i64;
    let items = MAX_ITEM_ID as i64;
    let cpu = tch::Device::Cpu;

    // Set random number generator
    let mut rng = rand::thread_rng();

    // Create random evolving session
    let evolving_session: Vec<i64> = (0..items_session)
        .map(|_| rng.gen_range(0..items))
        .collect::<Vec<i64>>();

    // Create random historical sessions value and session indices array
    let historical_sessions_length = MAX_NUM_ITEMS_IN_HISTORY_SESSION * NUM_HISTORY_SESSIONS;
    let mut historical_sessions_values = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_session_indices = Vec::with_capacity(historical_sessions_length);
    for session in 0..NUM_HISTORY_SESSIONS {
        for item in 0..MAX_NUM_ITEMS_IN_HISTORY_SESSION {
            historical_sessions_values.push(rng.gen_range(0..items));
            historical_sessions_session_indices.push(session as i64);
        }
    }

    // Sort historical session values and session indices array
    let mut historical_sessions_indices =
        (0..historical_sessions_length as i64).collect::<Vec<i64>>();
    historical_sessions_indices.sort_by_key(|&i| &historical_sessions_values[i as usize]);
    let historical_sessions_values_sorted = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_values[i as usize].clone())
        .collect();
    let historical_sessions_session_indices_sorted = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_session_indices[i as usize].clone())
        .collect();

    bench.iter(|| {
        bencher::black_box(binarysearch(
            &historical_sessions_values_sorted,
            &historical_sessions_session_indices_sorted,
            &evolving_session,
        ))
    })
}

fn binarysearch(
    historical_sessions_values_sorted: &Vec<i64>,
    historical_sessions_session_indices_sorted: &Vec<i64>,
    evolving_session: &Vec<i64>,
) -> Vec<f64> {
    let mut similarities = vec![0.0; NUM_HISTORY_SESSIONS];
    //  Loop over items in evolving session
    for (pos, item_id) in evolving_session.iter().enumerate() {
        // Find index of current item in historical session vector using binary search
        let index = historical_sessions_values_sorted.binary_search(&item_id);
        // println!("{}", index.unwrap());
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
                if (historical_sessions_values_sorted[left_index as usize] == *item_id) {
                    let current_similarity_index =
                        historical_sessions_session_indices_sorted[left_index as usize] as usize;
                    similarities[current_similarity_index] += decay_factor;
                }
                left_index -= 1;
            }
            // Count duplicate elements right to first index - note that empirically it seems Rust's binary search always returns the most right index, causing the below loop never to be triggered;
            while right_index < historical_sessions_values_sorted.len() {
                if (historical_sessions_values_sorted[right_index] == *item_id) {
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
