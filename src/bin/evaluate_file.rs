use itertools::Itertools;
use serenade_optimized::io::read_training_data;
use serenade_optimized::metrics::coverage::Coverage;
use serenade_optimized::metrics::hitrate::HitRate;
use serenade_optimized::metrics::mrr::Mrr;
use serenade_optimized::metrics::ndcg::Ndcg;
use serenade_optimized::metrics::popularity::Popularity;
use serenade_optimized::metrics::SessionMetric;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn main() {
    // This tool can evaluate predictions made by computational models and stored as a file.
    // Its needs access to the training data for the metrics 'popularity' and 'coverage'.
    let training_data_path = "../datasets/private-clicks-1m_train.txt";
    let predictions_file = "../java_impl/java_vsknn_predictions.txt";

    let training_df = read_training_data(&*training_data_path);

    let mut ndcg = Ndcg::new(20);
    let mut mrr = Mrr::new(20);
    let mut hitrate = HitRate::new(20);
    let mut coverage = Coverage::new(&training_df, 20);
    let mut popularity = Popularity::new(&training_df, 20);

    if let Ok(lines) = read_lines(predictions_file) {
        // Consumes the iterator, returns an (Optional) String
        for result in lines {
            if let Ok(line) = result {
                let recos_w_predictions = line.split(";").collect_vec();
                let recos = *unsafe { recos_w_predictions.get_unchecked(0) };
                let recos = recos
                    .split(",")
                    .collect_vec()
                    .into_iter()
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
                ndcg.add(&recos, &next_items);
                mrr.add(&recos, &next_items);
                hitrate.add(&recos, &next_items);
                coverage.add(&recos, &next_items);
                popularity.add(&recos, &next_items);
            }
        }
    }

    println!("{}: {:.4}", ndcg.get_name(), ndcg.result());
    println!("{}: {:.4}", mrr.get_name(), mrr.result());
    println!("{}: {:.4}", hitrate.get_name(), hitrate.result());
    println!("{}: {:.4}", coverage.get_name(), coverage.result());
    println!("{}: {:.4}", popularity.get_name(), popularity.result());
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
