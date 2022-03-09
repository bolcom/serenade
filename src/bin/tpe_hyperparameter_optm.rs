use rand::SeedableRng as _;
use serenade_optimized::objective;
use serenade_optimized::config::AppConfig;
use std::io::{stdout, Write};

use indicatif::ProgressBar;

extern crate csv;
use csv::Writer;


pub fn convert_string_to_vec_i32(s: String) -> Vec<i32> {
    return s
        .replace(&['[', ']'][..], "")
        .chars().filter(|c| !c.is_whitespace())
        .collect::<String>()
        .split(",")
        .map(|s| s.parse().unwrap())
        .collect();
}

fn main() -> anyhow::Result<()>{
    // get params from config file
    let config_path = std::env::args().
        nth(1).
        expect("Config file not specified!");
    let config = AppConfig::new(config_path);
    let training_data_path = config.hyperparam.training_data_path;
    let test_data_path = config.hyperparam.test_data_path;
    let validation_data_path = config.hyperparam.validation_data_path;
    let num_iterations = config.hyperparam.num_iterations;
    let save_records = config.hyperparam.save_records;
    let out_path = config.hyperparam.out_path;
    let enable_business_logic = config.hyperparam.enable_business_logic;
    let n_most_recent_sessions_range = convert_string_to_vec_i32(
        config.hyperparam.n_most_recent_sessions_range);
    let neighborhood_size_k_range = convert_string_to_vec_i32(
        config.hyperparam.neighborhood_size_k_range);
    let last_items_in_session_range = convert_string_to_vec_i32(
        config.hyperparam.last_items_in_session_range);

    // Progress bar
    let pb = ProgressBar::new(num_iterations as u64);

    let mut wtr = Writer::from_path(out_path)?;    
    if save_records {
        // csv writer for storing all values of the whole procedure
        wtr.write_record(&[
            "iteration",
            "n_most_recent_sessions",
            "neighborhood_size_k",
            "last_items_in_session",
            "Mean_reciprocal_rank"
        ])?;
    }
    /* 
    Each instance of TpeOptimizer tries to search out the value
    which could minimize/maximize the evaluation result 
    for such hyperparameter.
    */
    let mut optim0 =
            // n most recent sessions
            tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(n_most_recent_sessions_range.first().unwrap().clone() as f64, n_most_recent_sessions_range.last().unwrap().clone() as f64)?);

    let mut optim1 =
            // neighbourhood size k
            tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(neighborhood_size_k_range.first().unwrap().clone() as f64, neighborhood_size_k_range.last().unwrap().clone() as f64)?);

    let mut optim2 =
            // last items from session
            tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(last_items_in_session_range.first().unwrap().clone() as f64, last_items_in_session_range.last().unwrap().clone() as f64)?);

    println!("===============================================================");
    println!("===           START HYPER PARAMETER OPTIMIZATION           ====");
    println!("===============================================================");

    // mutable variables
    let mut best_value = std::f64::NEG_INFINITY;
    let mut rng = rand::rngs::StdRng::from_seed(Default::default());
    
    // optimization loop for num_iterations
    for i in 0..num_iterations {
        // increment progress bar
        pb.inc(1);
        // ask() gets the next value of the optimization target 
        // hyperparameter to be evaluated
        let n_most_recent_sessions = optim0.ask(&mut rng)?;
        let neighborhood_size_k = optim1.ask(&mut rng)?;
        let last_items_in_session = optim2.ask(&mut rng)?;
        // get the result of the object function
        // with current combination of hyperparameters
        let v = objective::objective(
            training_data_path.clone(),
            test_data_path.clone(),
            n_most_recent_sessions as i32,
            neighborhood_size_k as i32,
            last_items_in_session as i32,
            enable_business_logic
        );

        if save_records {
            // Save current values
            wtr.write_record(&[
                i.to_string(),
                n_most_recent_sessions.to_string(),
                neighborhood_size_k.to_string(),
                last_items_in_session.to_string(),
                v.to_string()
            ])?;
        }
        
        // Tells the evaluation result of a hyperparameter
        // value to the optimizer
        optim0.tell(n_most_recent_sessions, v)?;
        optim1.tell(neighborhood_size_k, v)?;
        optim2.tell(last_items_in_session, v)?;

        // update current best_value
        best_value = best_value.max(v);
        
    }

    // print best value for each hyperparameter
    // discussion showing that it is faster to "iterate by hand"
    // https://www.reddit.com/r/rust/comments/31syce/using_iterators_to_find_the_index_of_the_min_or/cq4r6xw/
    println!("Considering {} iterations for hyper parameter optimization...", num_iterations);

    let mut n_most_recent_sessions = 0;
    let mut neighborhood_size_k = 0;
    let mut last_items_in_session = 0;
    for (a, b) in optim0.trials() {
        if b == best_value {
            n_most_recent_sessions = a as i32;
        }
    }
    for (a, b) in optim1.trials() {
        if b == best_value {
            neighborhood_size_k = a as i32;
        }
    }
    for (a, b) in optim2.trials() {
        if b == best_value {
            last_items_in_session = a as i32;
        }
    }

    let test_score = objective::objective(
        training_data_path.clone(),
        validation_data_path.clone(),
        n_most_recent_sessions,
        neighborhood_size_k,
        last_items_in_session,
        enable_business_logic
    );
    println!("===============================================================");
    println!("===          HYPER PARAMETER OPTIMIZATION RESULTS          ====");
    println!("===============================================================");
    println!("MRR@20 for validation data: {:.4}", best_value);
    println!("MRR@20 for test data: {:.4}", test_score);

    println!("enabled business_logic for evaluation:{}", enable_business_logic);
    println!("best hyperparameter values:");
    println!("n_most_recent_sessions:{}", n_most_recent_sessions);
    println!("neighborhood_size_k:{}", neighborhood_size_k);
    println!("last_items_in_session:{}", last_items_in_session);

    println!("HPO done");
    println!("HPO done");
    stdout().flush()?;
    wtr.flush()?;

    Ok(())
}