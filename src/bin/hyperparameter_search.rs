use serenade::hyperparameter::hyperparamgrid::HyperParamGrid;
use serenade::metrics::mrr::Mrr;
use serenade::metrics::SessionMetric;
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
    param_grid.insert(
        "max_items_in_session".to_string(),
        vec![1, 2, 3, 5, 7, 15, 100],
    );
    param_grid.insert("idf_weighting".to_string(), vec![1, 2, 3]);

    let model = std::env::args()
        .nth(3)
        .map(|path| serenade::config::AppConfig::new(path).model.knn)
        .unwrap_or_default();

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

    let chosen_hyperparameters = hyper_parametergrid.get_n_random_combinations(150); // Progress bar
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
            let score = serenade::objective::objective_with_model(
                path_to_training.clone(),
                test_data_file.clone(),
                m_most_recent_sessions as i32,
                neighborhood_size_k as i32,
                max_items_in_session as i32,
                idf_weighting,
                enable_business_logic,
                &model,
            );
            if score > best_score {
                best_score = score;
                best_params = hyperparams.clone();
            }
            println!(
                "HPO,{},{},{},{}",
                m_most_recent_sessions, neighborhood_size_k, max_items_in_session, score
            );
        }
    }
    println!(
        "Best hyperparameter values found:,{:?} with {}:{}",
        best_params, main_metric_name, best_score
    );
}
