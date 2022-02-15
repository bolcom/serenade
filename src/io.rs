use hashbrown::HashMap;
use itertools::Itertools;
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub type TrainingSessionId = u32;
pub type ItemId = u64;
pub type Time = usize;

pub fn read_training_data(training_data_path: &str) -> Vec<(TrainingSessionId, ItemId, Time)> {
    let mut line_iterator = create_buffered_line_reader(training_data_path).unwrap();
    line_iterator.next(); // skip header
    let training_data = line_iterator.par_bridge().filter_map(move |result| {
        if let Ok(rawline) = result {
            let parts = rawline.split_whitespace().take(3).collect::<Vec<_>>();
            let (session_id, item_id, time) = (
                parts.get(0).unwrap().parse::<TrainingSessionId>().unwrap(),
                parts.get(1).unwrap().parse::<ItemId>().unwrap(),
                parts.get(2).unwrap().parse::<f64>().unwrap(),
            );
            Some((session_id, item_id, time.round() as Time))
        } else {
            None
        }
    });
    training_data.collect()
}

fn create_buffered_line_reader<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn read_test_data_evolving(test_data_file: &str) -> HashMap<TrainingSessionId, Vec<ItemId>> {
    let training_data = read_training_data(test_data_file);
    let evolving_sessions: HashMap<TrainingSessionId, Vec<ItemId>> = training_data
        .into_iter()
        .map(|(session_id, item_id, time)| (session_id, (item_id, time)))
        .into_group_map()
        .into_iter()
        .map(|(session_id, mut item_ids_with_order)| {
            item_ids_with_order.sort_unstable_by(|(_, time_a), (_, time_b)| time_a.cmp(time_b));
            let session_items: Vec<ItemId> = item_ids_with_order
                .into_iter()
                .map(|(item, _order)| item)
                .collect();

            (session_id, session_items)
        })
        .collect();

    evolving_sessions
}
