use serenade_optimized::hyperparameter::hyperparamgrid::HyperParamGrid;
use serenade_optimized::io::read_training_data;
use serenade_optimized::metrics::evaluation_reporter::EvaluationReporter;
use serenade_optimized::vmisknn::offline_index::OfflineIndex;
use serenade_optimized::{io, vmisknn};
use std::collections::HashMap;

fn main() {
    let mut param_grid = HashMap::new();
    param_grid.insert(
        "sample_size".to_string(),
        vec![35, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 10000],
    );
    param_grid.insert("k".to_string(), vec![50, 100, 500, 1000, 1500]);
    param_grid.insert("last_items_in_session".to_string(), vec![2]);

    let qty_max_reco_results = 21;
    let enable_business_logic = false;

    let path_to_training = std::env::args()
        .nth(1)
        .expect("Training data file not specified!");
    println!("result:training_data_file:{}", path_to_training);

    let test_data_file = std::env::args()
        .nth(2)
        .expect("Test data file not specified!");
    println!("result:test_data_file:{}", test_data_file);

    let hyper_parametergrid = HyperParamGrid { param_grid };

    let training_df = read_training_data(&*path_to_training);

    let evaluation_reporter = EvaluationReporter::new(&training_df, 20);
    println!(
        "result:n_most_recent_sessions,neighborhood_size_k,max_items_in_session,{}",
        evaluation_reporter.get_name()
    );

    let chosen_hyperparameters = hyper_parametergrid.get_all_combinations();
    for hyperparams in chosen_hyperparameters {
        let max_items_in_session = *hyperparams.get("last_items_in_session").unwrap();
        let neighborhood_size_k = *hyperparams.get("k").unwrap();
        let n_most_recent_sessions = *hyperparams.get("sample_size").unwrap();

        if neighborhood_size_k <= n_most_recent_sessions {
            let vsknn_index = OfflineIndex::new_from_csv(&*path_to_training, n_most_recent_sessions);
            let ordered_test_sessions = io::read_test_data_evolving(&*test_data_file);
            let mut evaluation_reporter = EvaluationReporter::new(&training_df, 20);

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
                        let recommendations = vmisknn::predict(
                            &vsknn_index,
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

                        evaluation_reporter.add(&recommended_items, &actual_next_items);
                    }
                });
            println!(
                "result:{},{},{},{}",
                n_most_recent_sessions,
                neighborhood_size_k,
                max_items_in_session,
                evaluation_reporter.result()
            );
        }
    }
}
