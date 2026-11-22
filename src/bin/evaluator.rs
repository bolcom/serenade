use serenade::{io, vmisknn};

use serenade::config::AppConfig;
use serenade::metrics::evaluation_reporter::EvaluationReporter;
use serenade::stopwatch::Stopwatch;
use serenade::vmisknn::vmis_index::VMISIndex;

fn main() {
    let config_path = std::env::args().nth(1).unwrap_or_default();
    let config = AppConfig::new(config_path);

    let m_most_recent_sessions = config.model.m_most_recent_sessions;
    let neighborhood_size_k = config.model.neighborhood_size_k;
    let num_items_to_recommend = config.model.num_items_to_recommend;
    let max_items_in_session = config.model.max_items_in_session;
    let enable_business_logic = config.logic.enable_business_logic;

    let vmis_index = VMISIndex::load_for_model(
        &config.data.training_data_path,
        m_most_recent_sessions,
        config.model.idf_weighting as f64,
        &config.model.knn,
    )
    .expect("Cannot load selected kNN model");

    let test_data_file = config.hyperparam.test_data_path;
    println!("test_data_file:{}", test_data_file);

    let ordered_test_sessions = io::read_test_data_timed(&*test_data_file);

    let mut reporter =
        EvaluationReporter::from_items(vmis_index.training_items(), num_items_to_recommend);

    let mut stopwatch = Stopwatch::new();

    ordered_test_sessions
        .iter()
        .for_each(|(_session_id, events)| {
            let evolving_session_items: Vec<_> = events.iter().map(|(item, _)| *item).collect();
            for session_state in 1..evolving_session_items.len() {
                // use last x items of evolving session
                let start_index = if session_state > max_items_in_session {
                    session_state - max_items_in_session
                } else {
                    0
                };
                let session: &[u64] = &evolving_session_items[start_index..session_state];
                stopwatch.start();
                let recommendations = vmisknn::predict_with_model(
                    &vmis_index,
                    &session,
                    neighborhood_size_k,
                    m_most_recent_sessions,
                    num_items_to_recommend,
                    enable_business_logic,
                    &config.model.knn,
                    events[session_state - 1].1,
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
    println!(
        "p25 (microseconds): {}",
        stopwatch.get_percentile_in_micros(25.0)
    );
    println!(
        "p50 (microseconds): {}",
        stopwatch.get_percentile_in_micros(50.0)
    );
    println!(
        "p75 (microseconds): {}",
        stopwatch.get_percentile_in_micros(75.0)
    );
    println!(
        "p90 (microseconds): {}",
        stopwatch.get_percentile_in_micros(90.0)
    );
    println!(
        "p95 (microseconds): {}",
        stopwatch.get_percentile_in_micros(95.0)
    );
    println!(
        "p99 (microseconds): {}",
        stopwatch.get_percentile_in_micros(99.0)
    );
    println!(
        "p99.5 (microseconds): {}",
        stopwatch.get_percentile_in_micros(99.5)
    );
}
