use serenade::{
    config::AppConfig,
    offline::{self, EvaluationData},
    vmisknn::vmis_index::VMISIndex,
};
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        (2..=4).contains(&args.len()),
        "usage: evaluator CONFIG [ALGORITHM|all] [RESULTS.json] or evaluator OUTPUT_DIRECTORY"
    );
    if std::path::Path::new(&args[1]).is_dir() {
        anyhow::ensure!(
            args.len() == 2,
            "directory evaluation usage: evaluator OUTPUT_DIRECTORY"
        );
        let mut paths = std::fs::read_dir(&args[1])?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.sort();
        let mut count = 0;
        for path in paths
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
        {
            let config = AppConfig::try_new(path.to_string_lossy().into_owned())?;
            evaluate_config(
                config,
                None,
                &path.with_extension("evaluation.json").to_string_lossy(),
            )?;
            count += 1;
        }
        anyhow::ensure!(count > 0, "no TOML configurations in directory");
        return Ok(());
    }
    let config = AppConfig::try_new(args[1].clone())?;
    let output = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("evaluation-results.json");
    evaluate_config(config, args.get(2).map(String::as_str), output)
}

fn evaluate_config(config: AppConfig, selection: Option<&str>, output: &str) -> anyhow::Result<()> {
    let models = offline::selected_models(&config.model, selection)?;
    let data = EvaluationData::load(&config.hyperparam.test_data_path)?;
    let mut index = VMISIndex::load_for_model(
        &config.data.training_data_path,
        config.model.m_most_recent_sessions,
        config.model.idf_weighting,
        &config.model.knn,
    )
    .map_err(anyhow::Error::msg)?;
    let mut results = Vec::new();
    let model_configs = models.clone();
    for model in models {
        index
            .prepare_for_model(
                model.m_most_recent_sessions,
                model.idf_weighting,
                &model.knn,
            )
            .map_err(anyhow::Error::msg)?;
        results.push(offline::evaluate(
            &index,
            &data,
            &model,
            config.logic.enable_business_logic,
        ));
    }
    offline::print_table(&results);
    std::fs::write(
        output,
        serde_json::to_string_pretty(
            &serde_json::json!({"training_data": config.data.training_data_path, "test_data": config.hyperparam.test_data_path, "cutoff":20, "business_logic": config.logic.enable_business_logic, "model_configurations":model_configs, "results":results}),
        )?,
    )?;
    println!("Results: {}", output);
    Ok(())
}
