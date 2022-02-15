#[macro_use]
extern crate bencher;
extern crate rand;
extern crate serenade_optimized;

use bencher::Bencher;
use hashbrown::{HashMap, HashSet as HashSetBrown, HashSet};
use itertools::Itertools;
use rand::Rng;
use std::collections::BTreeSet as BTreeSetStd;
use std::collections::HashSet as HashSetStd;

use rayon::prelude::*;

use serenade_optimized::io::{read_test_data_evolving, read_training_data};
use serenade_optimized::io::{ItemId, Time, TrainingSessionId};
use serenade_optimized::vmisknn::SessionTime;

benchmark_group!(benches, zipper);
benchmark_main!(benches);

const TRAIN_DATA_FILE: &str = "../serenade/datasets/private-clicks-100k_train.txt";
const TEST_DATA_FILE: &str = "../serenade/datasets/private-clicks-100k_test.txt";
const SAMPLE_SIZE_M: usize = 5000;
const MAX_SESSION_ITEMS: usize = 27;

fn zipper(bench: &mut Bencher) {
    let training_data: Vec<(TrainingSessionId, ItemId, Time)> = read_training_data(TRAIN_DATA_FILE);
    let training_data = sanitize_training_data(training_data, SAMPLE_SIZE_M, MAX_SESSION_ITEMS);
    println!("{}", training_data.len());
    let test_data: HashMap<TrainingSessionId, Vec<ItemId>> =
        read_test_data_evolving(TEST_DATA_FILE);
}

fn sanitize_training_data(
    training_data: Vec<(TrainingSessionId, ItemId, Time)>,
    sample_size_m: usize,
    max_qty_session_items: usize,
) -> Vec<(TrainingSessionId, ItemId, Time)> {
    let session_ids: HashSet<TrainingSessionId> = training_data
        .iter()
        .map(|(session_id, item_id, _time)| (session_id, item_id))
        .unique()
        .into_group_map()
        .into_iter()
        .filter(|(_session_id, item_ids)| item_ids.len() <= max_qty_session_items)
        .map(|(session_id, _item_ids)| *session_id)
        .collect::<HashSet<TrainingSessionId>>();

    let filtered_training_data: Vec<(TrainingSessionId, ItemId, Time)> = training_data
        .par_iter()
        .filter(|(session_id, _item_id, _time)| session_ids.contains(session_id))
        .map(|(session_id, item_id, time)| (session_id.clone(), item_id.clone(), time.clone()))
        .collect();

    let valid_session_ids: HashSet<u32> = filtered_training_data
        .iter()
        .map(|(session_id, item_id, time)| {
            (item_id, SessionTime::new(session_id.clone(), time.clone()))
        })
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

            let session_ids: HashSet<u32> = session_id_with_time
                .iter()
                .map(|session_id_time| session_id_time.session_id)
                .collect();
            session_ids
        })
        .collect();

    training_data
        .into_iter()
        .filter(|training_event| valid_session_ids.contains(&training_event.0))
        .cloned()
        .collect()
}
