use std::cmp::min;
use rand::{SeedableRng as _, thread_rng};
use serenade::objective;
use serenade::config::AppConfig;
use std::io::{stdout, Write};
use std::sync::{Arc, Mutex};

use indicatif::ProgressBar;

extern crate csv;
use csv::Writer;
use itertools::Itertools;
use rayon::prelude::*;

pub fn convert_string_to_vec_i32(s: String) -> Vec<i32> {
    return s
        .replace(&['[', ']'][..], "")
        .chars().filter(|c| !c.is_whitespace())
        .collect::<String>()
        .split(",")
        .map(|s| s.parse().unwrap())
        .collect();
}

fn main() {
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
    let idf_weighting_range = convert_string_to_vec_i32(
        config.hyperparam.idf_weighting_range);

    // Progress bar
    let pb = ProgressBar::new(num_iterations as u64);

    let mut wtr = Writer::from_path(out_path).unwrap();
    if save_records {
        // csv writer for storing all values of the whole procedure
        wtr.write_record(&[
            "n_most_recent_sessions",
            "neighborhood_size_k",
            "last_items_in_session",
            "idf_weighting",
            "MRR@20"
        ]).unwrap();
    }
    // Wrap the CSV writer in a Mutex and then an Arc
    let wtr_mutex = Arc::new(Mutex::new(wtr));
    /*
    Each instance of TpeOptimizer tries to search out the value
    which could minimize/maximize the evaluation result
    for such hyperparameter.
    */
    let mut optim0 =
            // n most recent sessions
        Arc::new(Mutex::new(tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(n_most_recent_sessions_range.first().unwrap().clone() as f64, n_most_recent_sessions_range.last().unwrap().clone() as f64).unwrap())));

    let mut optim1 =
            // neighbourhood size k
        Arc::new(Mutex::new(tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(neighborhood_size_k_range.first().unwrap().clone() as f64, neighborhood_size_k_range.last().unwrap().clone() as f64).unwrap())));

    let mut optim2 =
            // last items from session
        Arc::new(Mutex::new(tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(last_items_in_session_range.first().unwrap().clone() as f64, last_items_in_session_range.last().unwrap().clone() as f64).unwrap())));

    let mut optim3 =
        // last items from session
        Arc::new(Mutex::new(tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(idf_weighting_range.first().unwrap().clone() as f64, idf_weighting_range.last().unwrap().clone() as f64).unwrap())));

    println!("===============================================================");
    println!("===           START HYPER PARAMETER OPTIMIZATION           ====");
    println!("===============================================================");

    // mutable variables
    let mut best_value = Arc::new(Mutex::new(f64::NEG_INFINITY));

    // optimization loop for num_iterations
    (0..num_iterations).into_par_iter().for_each(|_| {
        let optim0_clone = Arc::clone(&optim0);
        let optim1_clone = Arc::clone(&optim1);
        let optim2_clone = Arc::clone(&optim2);
        let optim3_clone = Arc::clone(&optim3);
        let wtr_mutex_clone = Arc::clone(&wtr_mutex);
        // Generate a random number using thread's local RNG
        let mut rng = thread_rng();

        // increment progress bar
        pb.inc(1);
        // ask() gets the next value of the optimization target
        // hyperparameter to be evaluated
        let n_most_recent_sessions = optim0_clone.lock().unwrap().ask(&mut rng).unwrap();
        let neighborhood_size_k = optim1_clone.lock().unwrap().ask(&mut rng).unwrap();
        let last_items_in_session = optim2_clone.lock().unwrap().ask(&mut rng).unwrap();
        let idf_weighting = optim3_clone.lock().unwrap().ask(&mut rng).unwrap().floor();
        let best_value_clone = Arc::clone(&best_value);

        // Unlock the mutexes to allow other threads to access optim's
        drop(optim0_clone);
        drop(optim1_clone);
        drop(optim2_clone);
        drop(optim3_clone);

        // get the result of the object function
        // with current combination of hyperparameters
        let v = objective::objective(
            training_data_path.clone(),
            validation_data_path.clone(),
            n_most_recent_sessions as i32,
            neighborhood_size_k as i32,
            last_items_in_session as i32,
            idf_weighting,
            enable_business_logic
        );



        if save_records {

            // Each thread writes to the file
            let mut wtr_locked = wtr_mutex_clone.lock().unwrap();
            wtr_locked.write_record(&[
                n_most_recent_sessions.to_string(),
                neighborhood_size_k.to_string(),
                last_items_in_session.to_string(),
                idf_weighting.to_string(),
                v.to_string()
            ])
                .unwrap();
            // Ensure the write is flushed to the file
            wtr_locked.flush().unwrap();
        }

        // Tells the evaluation result of a hyperparameter
        // value to the optimizer
        let optim0_clone = Arc::clone(&optim0);
        let optim1_clone = Arc::clone(&optim1);
        let optim2_clone = Arc::clone(&optim2);
        let optim3_clone = Arc::clone(&optim3);

        optim0_clone.lock().unwrap().tell(n_most_recent_sessions, v).unwrap();
        optim1_clone.lock().unwrap().tell(neighborhood_size_k, v).unwrap();
        optim2_clone.lock().unwrap().tell(last_items_in_session, v).unwrap();
        optim3_clone.lock().unwrap().tell(idf_weighting, v).unwrap();


        // update current best_value
        let mut best_value_locked = best_value_clone.lock().unwrap();
        if v > *best_value_locked {
            *best_value_locked = v;
        }

    });

    println!("Considering {} iterations for hyper parameter optimization...", num_iterations);

    // Access the final best_value after all iterations have completed
    let final_best_value = *best_value.lock().unwrap();

    // Accessing all parameters after all iterations have completed
    let n_most_recent_sessions = optim0.lock().unwrap().trials().collect_vec().into_iter()
        .find(|(_value, score)| score == &final_best_value).map(|(value, _score)| value as i32).unwrap();
    let neighborhood_size_k = optim1.lock().unwrap().trials().collect_vec().into_iter()
        .find(|(_value, score)| score == &final_best_value).map(|(value, _score)| value as i32).unwrap();
    let neighborhood_size_k = min(neighborhood_size_k, n_most_recent_sessions);
    let last_items_in_session = optim2.lock().unwrap().trials().collect_vec().into_iter()
        .find(|(_value, score)| score == &final_best_value).map(|(value, _score)| value as i32).unwrap();
    let idf_weighting = optim3.lock().unwrap().trials().collect_vec().into_iter()
        .find(|(_value, score)| score == &final_best_value).map(|(value, _score)| value.floor()).unwrap();

    let test_score = objective::objective(
        training_data_path.clone(),
        test_data_path.clone(),
        n_most_recent_sessions,
        neighborhood_size_k,
        last_items_in_session,
        idf_weighting ,
        enable_business_logic,
    );



    let evaluation_length = 20;
    println!("===============================================================");
    println!("===          HYPER PARAMETER OPTIMIZATION RESULTS          ====");
    println!("===============================================================");
    println!("MRR@{} for validation data: {:.4}", evaluation_length, final_best_value);
    println!("MRR@{} for test data: {:.4}", evaluation_length, test_score);

    println!("enabled business_logic for evaluation:{}", enable_business_logic);
    println!("best hyperparameter values:");
    println!("n_most_recent_sessions:{}", n_most_recent_sessions);
    println!("neighborhood_size_k:{}", neighborhood_size_k);
    println!("idf_weighting:{}", idf_weighting);
    println!("last_items_in_session:{}", last_items_in_session);

    println!("HPO done");
    stdout().flush().unwrap();


}