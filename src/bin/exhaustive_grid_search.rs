use serenade_optimized::config::AppConfig;
use serenade_optimized::objective;

use indicatif::ProgressBar;

extern crate csv;
use csv::Writer;

fn main() -> anyhow::Result<()>{
    // get params from config file
    let config_path = std::env::args().
        nth(1).
        expect("Config file not specified!");
    let config = AppConfig::new(config_path);
    let train_data_path = config.hyperparam.training_data_path;
    let test_data_path = config.hyperparam.test_data_path;
    let save_records = config.hyperparam.save_records;
    let out_path = config.hyperparam.out_path;
    let enable_business_logic = config.hyperparam.enable_business_logic;

    // Possible values for hyperparameters
    let n_most_recent_sessions_choices = [100, 500, 1000, 2500];
    let neighborhood_size_k_choices = [50, 100, 500, 1000, 1500];
    let last_items_in_session_choices = [1, 2, 3, 5, 7, 10];
    let idf_weighting_choices = [1, 2, 3, 5, 7, 10];

    // Progress bar
    let total_num_iterations = n_most_recent_sessions_choices.len() * neighborhood_size_k_choices.len() * last_items_in_session_choices.len() * idf_weighting_choices.len();
    let pb = ProgressBar::new(total_num_iterations as u64);

    let mut wtr = Writer::from_path(out_path)?;    
    if save_records {
        // csv writer for storing all values of the whole procedure
        wtr.write_record(&[
            "iteration",
            "n_most_recent_sessions",
            "neighborhood_size_k",
            "last_items_in_session",
            "idf_weighting",
            "MRR@20"
        ])?;
    }

    // mutable variables
    let mut iteration = 0;
    let mut best_value = std::f64::NEG_INFINITY;
    let mut best_n_most_recent_sessions = -1;
    let mut best_neighborhood_size_k = -1;
    let mut best_last_items_in_session = -1;
    let mut best_idf_weighting = -1;
    // let mut rng = rand::rngs::StdRng::from_seed(Default::default());
    
    // exhaustive grid search
    for n_most_recent_sessions in n_most_recent_sessions_choices {
        for neighborhood_size_k in neighborhood_size_k_choices {
            for last_items_in_session in last_items_in_session_choices {
                for idf_weighting in idf_weighting_choices {
                    // increment progress bar
                    pb.inc(1);
                    // get the result of the object function
                    // with current combination of hyperparameters
                    let v = objective::objective(
                        train_data_path.clone(),
                        test_data_path.clone(),
                        n_most_recent_sessions,
                        neighborhood_size_k,
                        last_items_in_session,
                        idf_weighting as f64,
                        enable_business_logic
                    );

                    if save_records {
                        // Save current values
                        wtr.write_record(&[
                            (iteration as i32).to_string(),
                            n_most_recent_sessions.to_string(),
                            neighborhood_size_k.to_string(),
                            last_items_in_session.to_string(),
                            (idf_weighting as i32).to_string(),
                            v.to_string()
                        ])?;
                    }
                    // update current best values
                    if v > best_value {
                        best_value = v;
                        best_n_most_recent_sessions = n_most_recent_sessions;
                        best_neighborhood_size_k = neighborhood_size_k;
                        best_last_items_in_session = last_items_in_session;
                        best_idf_weighting = idf_weighting;
                    }
                    iteration = iteration + 1;

                }
            }
        }
    }
    // print best value for each hyperparameter    
    println!("Best n_most_recent_sessions: {}", 
        best_n_most_recent_sessions);
    println!("Best neighborhood_size_k: {}", 
        best_neighborhood_size_k);
    println!("Best last_items_in_session: {}", 
        best_last_items_in_session);
    println!("Best idf_weighting: {}",
             best_idf_weighting);
    if enable_business_logic {
        println!("Business logic were enabled.");
    } else {
        println!("Business logic were disabled.");
    }

    println!("Best value for the goal metric: {}", best_value);
    
    wtr.flush()?;
    
    Ok(())
}