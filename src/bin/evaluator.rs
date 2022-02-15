use serenade_optimized::{io, vmisknn};

use serenade_optimized::metrics::mrr::Mrr;
use serenade_optimized::metrics::SessionMetric;
use serenade_optimized::vmisknn::offline_index::OfflineIndex;

fn main() {
    // hyper-parameters
    let n_most_recent_sessions = 1500;
    let neighborhood_size_k = 500;
    let last_items_in_session = 3;
    let enable_business_logic = false;

    let path_to_training = std::env::args()
        .nth(1)
        .expect("Training data file not specified!");

    println!("training_data_file:{}", path_to_training);

    let test_data_file = std::env::args()
        .nth(2)
        .expect("Test data file not specified!");
    println!("test_data_file:{}", test_data_file);

    let offline_index = OfflineIndex::new_from_csv(&*path_to_training, n_most_recent_sessions);

    let ordered_test_sessions = io::read_test_data_evolving(&*test_data_file);

    let qty_max_reco_results = 20;
    let mut mymetric = Mrr::new(qty_max_reco_results);

    ordered_test_sessions
        .iter()
        .for_each(|(_session_id, evolving_session_items)| {
            for session_state in 1..evolving_session_items.len() {
                // use last x items of evolving session
                let start_index = if session_state > last_items_in_session {
                    session_state - last_items_in_session
                } else {
                    0
                };
                let session: &[u64] = &evolving_session_items[start_index..session_state];
                let recommendations = vmisknn::predict(
                    &offline_index,
                    &session,
                    neighborhood_size_k,
                    n_most_recent_sessions,
                    qty_max_reco_results,
                    enable_business_logic,
                );

                let recommended_items = recommendations
                    .into_sorted_vec()
                    .iter()
                    .map(|scored| scored.id)
                    .collect::<Vec<u64>>();

                let actual_next_items = Vec::from(&evolving_session_items[session_state..]);
                mymetric.add(&recommended_items, &actual_next_items);
            }
        });

    println!("{}: {}", mymetric.get_name(), mymetric.result());
}
