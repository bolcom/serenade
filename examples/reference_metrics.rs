//! Evaluate externally generated recommendation lists with Serenade's metrics.
use serenade::{metrics::evaluation_reporter::EvaluationReporter, vmisknn::vmis_index::VMISIndex};
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 3,
        "usage: reference_metrics TRAIN PREDICTIONS.json"
    );
    let index = VMISIndex::new_from_csv(&args[1], 500, 1.0);
    let predictions: BTreeMap<String, Vec<(Vec<u64>, Vec<u64>)>> =
        serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let mut results = BTreeMap::new();
    for (algorithm, points) in predictions {
        let mut reporter = EvaluationReporter::from_items(index.training_items(), 20);
        for (recommendations, targets) in points {
            reporter.add(&recommendations, &targets);
        }
        results.insert(algorithm, reporter.all_scores());
    }
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
