use hashbrown::HashMap;
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

/// Chronological events with stable input order for equal timestamps.
/// Keep fractional timestamps until prediction, so session-age weighting is reproducible.
pub fn read_test_data_timed(
    test_data_file: &str,
) -> HashMap<TrainingSessionId, Vec<(ItemId, f64)>> {
    let mut sessions = HashMap::<TrainingSessionId, Vec<(ItemId, f64)>>::new();
    for line in create_buffered_line_reader(test_data_file)
        .expect("Cannot read test data")
        .skip(1)
    {
        let line = line.expect("Cannot read test event");
        let mut fields = line.split_whitespace();
        let session = fields.next().unwrap().parse().unwrap();
        let item = fields.next().unwrap().parse().unwrap();
        let time: f64 = fields.next().unwrap().parse().unwrap();
        assert!(time.is_finite(), "Event timestamp must be finite");
        sessions.entry(session).or_default().push((item, time));
    }
    for events in sessions.values_mut() {
        events.sort_by(|a, b| a.1.total_cmp(&b.1));
    }
    sessions
}

pub fn read_test_data_evolving(test_data_file: &str) -> HashMap<TrainingSessionId, Vec<ItemId>> {
    read_test_data_timed(test_data_file)
        .into_iter()
        .map(|(id, events)| (id, events.into_iter().map(|(item, _)| item).collect()))
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluation_order_preserves_fractional_times_ties_and_repeats() {
        let path = std::env::temp_dir().join(format!("serenade-timed-{}.tsv", std::process::id()));
        std::fs::write(
            &path,
            "SessionId\tItemId\tTime\n1\t4\t2.8\n1\t2\t2.1\n1\t2\t2.1\n1\t3\t1\n",
        )
        .unwrap();
        let sessions = super::read_test_data_timed(path.to_str().unwrap());
        assert_eq!(sessions[&1], vec![(3, 1.0), (2, 2.1), (2, 2.1), (4, 2.8)]);
        std::fs::remove_file(path).unwrap();
    }
}
