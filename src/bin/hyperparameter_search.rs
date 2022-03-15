use serenade_optimized::hyperparameter::hyperparamgrid::HyperParamGrid;
use serenade_optimized::metrics::mrr::Mrr;
use serenade_optimized::metrics::SessionMetric;
use serenade_optimized::vmisknn::vmis_index::VMISIndex;
use serenade_optimized::{io, vmisknn};
use std::collections::HashMap;

use indicatif::ProgressBar;

fn main() {
    let mut param_grid = HashMap::new();
    param_grid.insert(
        "m_most_recent_sessions".to_string(),
        vec![100, 250, 500, 750, 1000, 2500],
    );
    param_grid.insert(
        "neighborhood_size_k".to_string(),
        vec![50, 100, 500, 1000, 1500],
    );
    param_grid.insert("max_items_in_session".to_string(), vec![1, 2, 3, 5, 7, 15, 100]);
    param_grid.insert("idf_weighting".to_string(), vec![1, 2, 3]);

    let qty_max_reco_results = 21;

    let path_to_training = std::env::args()
        .nth(1)
        .expect("Training data file not specified!");
    println!("training_data_file:{}", path_to_training);

    let test_data_file = std::env::args()
        .nth(2)
        .expect("Test data file not specified!");
    println!("test_data_file:{}", test_data_file);

    let hyper_parametergrid = HyperParamGrid { param_grid };

    let mut best_score = 0.0;
    let mut best_params = HashMap::new();
    let main_metric_name = Mrr::new(20).get_name();

    let chosen_hyperparameters = hyper_parametergrid.get_n_random_combinations(150);// Progress bar
    // Progress bar - HPO
    let pb_hpo = ProgressBar::new(chosen_hyperparameters.len() as u64);
    for hyperparams in chosen_hyperparameters {
        // increment progress bar
        pb_hpo.inc(1);
        let max_items_in_session = *hyperparams.get("max_items_in_session").unwrap();
        let neighborhood_size_k = *hyperparams.get("neighborhood_size_k").unwrap();
        let m_most_recent_sessions = *hyperparams.get("m_most_recent_sessions").unwrap();
        let idf_weighting = *hyperparams.get("idf_weighting").unwrap() as f64;
        let enable_business_logic = false;

        if neighborhood_size_k <= m_most_recent_sessions {
            let vmis_index = VMISIndex::new_from_csv(&*path_to_training, m_most_recent_sessions, idf_weighting);
            let ordered_test_sessions = io::read_test_data_evolving(&*test_data_file);
            let mut mymetric = Mrr::new(20);
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
                            &vmis_index,
                            &session,
                            neighborhood_size_k,
                            m_most_recent_sessions,
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
            if mymetric.result() > best_score {
                best_score = mymetric.result();
                best_params = hyperparams.clone();
            }
            println!(
                "HPO,{},{},{},{}",
                m_most_recent_sessions,
                neighborhood_size_k,
                max_items_in_session,
                mymetric.result()
            );
        }
    }
    println!(
        "Best hyperparameter values found:,{:?} with {}:{}",
        best_params, main_metric_name, best_score
    );
}
