use rand::SeedableRng as _;
use serenade_optimized::objective;
use serenade_optimized::config::AppConfig;

use indicatif::ProgressBar;

extern crate csv;
use csv::Writer;


fn main() -> anyhow::Result<()>{
    // get params from config file
    let config_path = std::env::args().
        nth(1).
        expect("Config file not specified!");
    let config = AppConfig::new(config_path);
    let training_data_path = config.hyperparam.training_data_path;
    let test_data_path = config.hyperparam.test_data_path;
    let num_iterations = config.hyperparam.num_iterations;
    let save_records = config.hyperparam.save_records;
    let out_path = config.hyperparam.out_path;
    let enable_business_logic = config.hyperparam.enable_business_logic;
    
    // Possible values for hyperparameters
    let n_most_recent_sessions_choices = [100, 500, 1000, 2500];
    let neighborhood_size_k_choices = [50, 100, 500, 1000, 1500];
    let last_items_in_session_choices = [1, 2, 3, 5, 7, 10];
    
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
        tpe::TpeOptimizer::new(
            tpe::histogram_estimator(), 
            tpe::categorical_range(
                n_most_recent_sessions_choices.len())?
        );
    let mut optim1 =
        tpe::TpeOptimizer::new(
            tpe::histogram_estimator(), 
            tpe::categorical_range(
                neighborhood_size_k_choices.len())?
        );
    let mut optim2 =
        tpe::TpeOptimizer::new(
            tpe::histogram_estimator(), 
            tpe::categorical_range(
                last_items_in_session_choices.len())?
        );

    // mutable variables
    let mut best_value = std::f64::NEG_INFINITY;
    let mut rng = rand::rngs::StdRng::from_seed(Default::default());
    
    // optimization loop for num_iterations
    for i in 0..num_iterations {
        // increment progress bar
        pb.inc(1);
        // ask() gets the next value of the optimization target 
        // hyperparameter to be evaluated
        let n_most_recent_sessions_index = optim0.ask(&mut rng)?;
        let neighborhood_size_k_index = optim1.ask(&mut rng)?;
        let last_items_in_session_index = optim2.ask(&mut rng)?;
        let n_most_recent_sessions = n_most_recent_sessions_choices[
            n_most_recent_sessions_index as usize];
        let neighborhood_size_k = neighborhood_size_k_choices[
            neighborhood_size_k_index as usize];   
        let last_items_in_session = last_items_in_session_choices[
            last_items_in_session_index as usize];
        // get the result of the object function
        // with current combination of hyperparameters
        let v = objective::objective(
            training_data_path.clone(), 
            test_data_path.clone(), 
            n_most_recent_sessions, 
            neighborhood_size_k, 
            last_items_in_session,
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
        optim0.tell(n_most_recent_sessions_index, v)?;
        optim1.tell(neighborhood_size_k_index, v)?;
        optim2.tell(last_items_in_session_index, v)?;

        // update current best_value
        best_value = best_value.max(v);
        
    }

    // print best value for each hyperparameter
    // discussion showing that it is faster to "iterate by hand"
    // https://www.reddit.com/r/rust/comments/31syce/using_iterators_to_find_the_index_of_the_min_or/cq4r6xw/
    println!("Considering {} iterations...", num_iterations);
    for (a, b) in optim0.trials() {
        if b == best_value {
            println!("Best n_most_recent_sessions: {}", 
                n_most_recent_sessions_choices[a as usize]);
        }
    }
    for (a, b) in optim1.trials() {
        if b == best_value {
            println!("Best neighborhood_size_k: {}", 
            neighborhood_size_k_choices[a as usize]);
        }
    }
    for (a, b) in optim1.trials() {
        if b == best_value {
            println!("Best last_items_in_session: {}", 
            last_items_in_session_choices[a as usize]);
        }
    }
    if enable_business_logic {
        println!("Business logic were enabled.");
    } else {
        println!("Business logic were disabled.");
    }

    println!("Best value for the goal metric: {}", best_value);
    
    wtr.flush()?;

    Ok(())
}