use serenade::{
    config::AppConfig,
    offline::{self, EvaluationData},
    vmisknn::vmis_index::VMISIndex,
};
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        (2..=4).contains(&args.len()),
        "usage: evaluator CONFIG [ALGORITHM|all] [RESULTS.json]"
    );
    let config = AppConfig::try_new(args[1].clone())?;
    let models = offline::selected_models(&config.model, args.get(2).map(String::as_str))?;
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
    let output = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("evaluation-results.json");
    std::fs::write(
        output,
        serde_json::to_string_pretty(
            &serde_json::json!({"training_data": config.data.training_data_path, "test_data": config.hyperparam.test_data_path, "cutoff":20, "business_logic": config.logic.enable_business_logic, "model_configurations":model_configs, "results":results}),
        )?,
    )?;
    println!("Results: {}", output);
    Ok(())
}
