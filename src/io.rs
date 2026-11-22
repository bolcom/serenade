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
    try_read_test_data_timed(test_data_file).expect("Cannot read test data")
}

pub fn try_read_test_data_timed(
    test_data_file: &str,
) -> anyhow::Result<HashMap<TrainingSessionId, Vec<(ItemId, f64)>>> {
    let mut sessions = HashMap::<TrainingSessionId, Vec<(ItemId, f64)>>::new();
    for (line_number, line) in create_buffered_line_reader(test_data_file)?
        .enumerate()
        .skip(1)
    {
        let line = line?;
        let fields: Vec<_> = line.split_whitespace().collect();
        anyhow::ensure!(
            fields.len() == 3,
            "{}:{}: expected SessionId ItemId Time",
            test_data_file,
            line_number + 1
        );
        let session = fields[0].parse()?;
        let item = fields[1].parse()?;
        let time: f64 = fields[2].parse()?;
        anyhow::ensure!(
            time.is_finite() && time >= 0.0 && time.round() <= u32::MAX as f64,
            "{}:{}: timestamp must fit u32",
            test_data_file,
            line_number + 1
        );
        sessions.entry(session).or_default().push((item, time));
    }
    for events in sessions.values_mut() {
        events.sort_by(|a, b| a.1.total_cmp(&b.1));
    }
    Ok(sessions)
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
