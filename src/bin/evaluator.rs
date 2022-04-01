use std::path::Path;
use serenade::{io, vmisknn};

use serenade::vmisknn::vmis_index::VMISIndex;
use serenade::config::AppConfig;
use serenade::metrics::evaluation_reporter::EvaluationReporter;
use serenade::stopwatch::Stopwatch;

fn main() {
    let config_path = std::env::args().nth(1).unwrap_or_default();
    let config = AppConfig::new(config_path);

    let m_most_recent_sessions = config.model.m_most_recent_sessions;
    let neighborhood_size_k = config.model.neighborhood_size_k;
    let num_items_to_recommend = config.model.num_items_to_recommend;
    let max_items_in_session = config.model.max_items_in_session;
    let enable_business_logic = config.logic.enable_business_logic;

    let training_data_path = Path::new(&config.data.training_data_path);
    let vmis_index = if training_data_path.is_dir() {
        // By default we use an index that is computed offline on billions of user-item interactions.
        VMISIndex::new(&config.data.training_data_path)
    } else if training_data_path.is_file() {
        // The following line creates an index directly from a csv file as input.
        VMISIndex::new_from_csv(
            &config.data.training_data_path,
            config.model.m_most_recent_sessions,
            config.model.idf_weighting as f64,
        )
    } else {
        panic!(
            "Training data file does not exist: {}",
            &config.data.training_data_path
        )
    };

    let test_data_file = config.hyperparam.test_data_path;
    println!("test_data_file:{}", test_data_file);

    let ordered_test_sessions = io::read_test_data_evolving(&*test_data_file);

    let mut reporter = EvaluationReporter::new(&io::read_training_data(&*config.data.training_data_path), num_items_to_recommend);

    let mut stopwatch = Stopwatch::new();

    ordered_test_sessions
        .iter()
        .for_each(|(_session_id, evolving_session_items)| {
            for session_state in 1..evolving_session_items.len() {
                // use last x items of evolving session
                let start_index = if session_state > max_items_in_session {
                    session_state - max_items_in_session
                } else {
                    0
                };
                let session: &[u64] = &evolving_session_items[start_index..session_state];
                stopwatch.start();
                let recommendations = vmisknn::predict(
                    &vmis_index,
                    &session,
                    neighborhood_size_k,
                    m_most_recent_sessions,
                    num_items_to_recommend,
                    enable_business_logic,
                );
                stopwatch.stop(&start_index);
                let recommended_items = recommendations
                    .into_sorted_vec()
                    .iter()
                    .map(|scored| scored.id)
                    .collect::<Vec<u64>>();

                let actual_next_items = Vec::from(&evolving_session_items[session_state..]);
                reporter.add(&recommended_items, &actual_next_items);
            }
        });
    println!("===============================================================");
    println!("===               START EVALUATING TEST FILE               ====");
    println!("===============================================================");
    println!("{}", reporter.get_name());
    println!("{}", reporter.result());
    println!("Qty test evaluations: {}", stopwatch.get_n());
    println!("Prediction latency");
    println!("p90 (microseconds): {}", stopwatch.get_percentile_in_micros(90.0));
    println!("p95 (microseconds): {}", stopwatch.get_percentile_in_micros(95.0));
    println!("p99.5 (microseconds): {}", stopwatch.get_percentile_in_micros(99.5));
}
