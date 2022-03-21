use itertools::Itertools;
use serenade_optimized::io::read_training_data;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use serenade_optimized::metrics::evaluation_reporter::EvaluationReporter;

fn main() {
    // This tool can evaluate predictions made by computational models and stored as a file.
    // Its needs access to the training data to determine the metrics 'popularity' and 'coverage'.
    let training_data_path = std::env::args().nth(1).unwrap_or_default();
    let predictions_file = std::env::args().nth(2).unwrap_or_default();

    let training_df = read_training_data(&*training_data_path);
    let length = 20;
    let mut reporter = EvaluationReporter::new(&training_df, length);

    if let Ok(lines) = read_lines(&*predictions_file) {
        // Consumes the iterator, returns an (Optional) String
        for result in lines {
            if let Ok(line) = result {
                let recos_w_predictions = line.split(";").collect_vec();
                let recos = *unsafe { recos_w_predictions.get_unchecked(0) };
                let recos = recos
                    .split(",")
                    .collect_vec()
                    .into_iter()
                    .filter(|str| str.len() > 0)
                    .map(|x| x.clone().parse::<u64>().unwrap())
                    .collect_vec();
                let next_items = *unsafe { recos_w_predictions.get_unchecked(1) };
                let next_items = next_items
                    .split(",")
                    .collect_vec()
                    .into_iter()
                    .map(|x| x.clone().parse::<u64>().unwrap())
                    .collect_vec();
                // recos.iter
                reporter.add(&recos, &next_items);
            }
        }
    }

    println!("===============================================================");
    println!("===              EVALUATING PREDICTONS BY FILE             ====");
    println!("===============================================================");
    println!("training data: {}", training_data_path);
    println!("predictions file: {}", predictions_file);
    println!("{}", reporter.get_name());
    println!("{}", reporter.result());

}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
