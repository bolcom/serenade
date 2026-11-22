use rand::{rngs::StdRng, SeedableRng};
use serde_json::json;
use serenade::{
    config::{AppConfig, ModelConfig},
    offline::{self, EvaluationData},
    vmisknn::{algorithms::Algorithm, vmis_index::VMISIndex},
};
use std::{collections::BTreeMap, path::Path};

struct Axis {
    name: &'static str,
    integer: bool,
    optimizer: tpe::TpeOptimizer,
}
fn range(text: &str, integer: bool, positive: bool) -> anyhow::Result<(f64, f64)> {
    let values: Vec<f64> = text
        .trim()
        .trim_matches(&['[', ']'][..])
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(
        values.len() == 2,
        "range must contain two endpoints: {}",
        text
    );
    let (low, high) = (values[0], values[1]);
    anyhow::ensure!(
        low.is_finite() && high.is_finite() && low < high && low >= 0.0 && (!positive || low > 0.0),
        "invalid range: {}",
        text
    );
    anyhow::ensure!(
        !integer || (low.fract() == 0.0 && high.fract() == 0.0 && high <= usize::MAX as f64),
        "integer range required: {}",
        text
    );
    Ok((low, high))
}
struct Search {
    axes: Vec<Axis>,
    max_m: usize,
    ranges: BTreeMap<String, (f64, f64)>,
}

fn axes(config: &AppConfig, algorithm: Algorithm) -> anyhow::Result<Search> {
    let h = &config.hyperparam;
    let mut specs = vec![
        (
            "m_most_recent_sessions",
            true,
            true,
            h.n_most_recent_sessions_range.as_str(),
        ),
        (
            "neighborhood_size_k",
            true,
            true,
            h.neighborhood_size_k_range.as_str(),
        ),
        (
            "max_items_in_session",
            true,
            true,
            h.last_items_in_session_range.as_str(),
        ),
    ];
    match algorithm {
        Algorithm::Vmis | Algorithm::Vsknn => {
            specs.push(("idf_weighting", false, false, &h.idf_weighting_range))
        }
        Algorithm::Sknn => specs.push((
            "pop_boost",
            false,
            false,
            h.search_ranges
                .get("pop_boost")
                .map(String::as_str)
                .unwrap_or("[0, 5]"),
        )),
        _ => {}
    }
    if matches!(algorithm, Algorithm::Stan | Algorithm::VsknnStan) {
        for name in ["lambda_spw", "lambda_snh", "lambda_inh"] {
            specs.push((
                name,
                false,
                true,
                h.search_ranges
                    .get(name)
                    .map(String::as_str)
                    .unwrap_or("[0.1, 10]"),
            ));
        }
    }
    if algorithm == Algorithm::VsknnStan {
        for name in ["lambda_ipw", "lambda_idf"] {
            specs.push((
                name,
                false,
                name != "lambda_idf",
                h.search_ranges
                    .get(name)
                    .map(String::as_str)
                    .unwrap_or("[0.1, 10]"),
            ));
        }
    }
    let mut axes = Vec::new();
    let mut ranges = BTreeMap::new();
    let mut max_m = 0;
    for (name, integer, positive, default) in specs {
        let text = h.algorithms[algorithm.name()]
            .ranges
            .get(name)
            .map(String::as_str)
            .unwrap_or(default);
        let (low, high) = range(text, integer, positive)?;
        if name == "m_most_recent_sessions" {
            max_m = high as usize;
        }
        ranges.insert(name.to_owned(), (low, high));
        axes.push(Axis {
            name,
            integer,
            optimizer: tpe::TpeOptimizer::new(
                tpe::parzen_estimator(),
                tpe::range(low, if integer { high + 1.0 } else { high })?,
            ),
        });
    }
    Ok(Search {
        axes,
        max_m,
        ranges,
    })
}
fn apply(model: &mut ModelConfig, name: &str, value: f64) {
    match name {
        "m_most_recent_sessions" => model.m_most_recent_sessions = value as usize,
        "neighborhood_size_k" => model.neighborhood_size_k = value as usize,
        "max_items_in_session" => model.max_items_in_session = value as usize,
        "idf_weighting" if model.knn.algorithm == Algorithm::Vmis => model.idf_weighting = value,
        "idf_weighting" => model.knn.idf_weighting = value,
        "pop_boost" => model.knn.pop_boost = value,
        "lambda_spw" => model.knn.lambda_spw = Some(value),
        "lambda_snh" => model.knn.lambda_snh = Some(value),
        "lambda_inh" => model.knn.lambda_inh = Some(value),
        "lambda_ipw" => model.knn.lambda_ipw = Some(value),
        "lambda_idf" => model.knn.lambda_idf = Some(value),
        _ => unreachable!(),
    }
}
fn model_snippet(m: &ModelConfig) -> String {
    let k = &m.knn;
    let mut text = format!("[model]\nalgorithm = \"{}\"\nm_most_recent_sessions = {}\nneighborhood_size_k = {}\nmax_items_in_session = {}\nnum_items_to_recommend = 20\nidf_weighting = {}\n\n[model.knn]\nsimilarity = \"{:?}\"\nweighting = \"{:?}\"\nweighting_score = \"{:?}\"\nweighting_time = {}\nidf_weighting = {}\nidf_weighting_session = {}\nremind = {}\nnormalize = {}\npop_boost = {}\n", k.algorithm.name(), m.m_most_recent_sessions, m.neighborhood_size_k, m.max_items_in_session, m.idf_weighting, k.similarity, k.weighting, k.weighting_score, k.weighting_time, k.idf_weighting, k.idf_weighting_session, k.remind, k.normalize, k.pop_boost).to_lowercase();
    for (name, value) in [
        ("lambda_spw", k.lambda_spw),
        ("lambda_snh", k.lambda_snh),
        ("lambda_inh", k.lambda_inh),
        ("lambda_ipw", k.lambda_ipw),
        ("lambda_idf", k.lambda_idf),
    ] {
        text.push_str(&format!(
            "{} = {}\n",
            name,
            value
                .map(|v| v.to_string())
                .unwrap_or_else(|| "\"none\"".into())
        ));
    }
    text
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        (2..=4).contains(&args.len()),
        "usage: tpe_hyperparameter_optm CONFIG [ALGORITHM|all] [OUTPUT_DIRECTORY]"
    );
    let config = AppConfig::try_new(args[1].clone())?;
    anyhow::ensure!(
        config.hyperparam.num_trials > 0,
        "hyperparam.num_trials must be positive"
    );
    let selection = args
        .get(2)
        .map(String::as_str)
        .or(if config.hyperparam.multi_algorithm {
            Some("all")
        } else {
            None
        });
    let mut models = offline::selected_models(&config.model, selection)?;
    if selection != Some("all") {
        anyhow::ensure!(
            config.hyperparam.algorithms[models[0].knn.algorithm.name()].enabled,
            "selected algorithm is disabled"
        );
    }
    models.retain(|m| config.hyperparam.algorithms[m.knn.algorithm.name()].enabled);
    anyhow::ensure!(!models.is_empty(), "no enabled algorithms");
    let searches = models
        .iter()
        .map(|m| axes(&config, m.knn.algorithm))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let max_m = searches.iter().map(|s| s.max_m).max().unwrap();
    let training = if config.hyperparam.training_data_path.is_empty() {
        &config.data.training_data_path
    } else {
        &config.hyperparam.training_data_path
    };
    anyhow::ensure!(
        !training.is_empty(),
        "hyperparam.training_data_path is required"
    );
    let data = EvaluationData::load(&config.hyperparam.validation_data_path)?;
    let mut index = VMISIndex::load_for_model(
        training,
        max_m,
        config.model.idf_weighting,
        &config.model.knn,
    )
    .map_err(anyhow::Error::msg)?;
    let output = args
        .get(3)
        .map(String::as_str)
        .unwrap_or(&config.hyperparam.output_dir);
    std::fs::create_dir_all(output)?;
    println!("algorithm\tbest validation MRR@20\thyperparameters\tapplication configuration");
    let mut comparison = Vec::new();
    for (base, search) in models.into_iter().zip(searches) {
        let Search {
            mut axes, ranges, ..
        } = search;
        let algorithm = base.knn.algorithm.name();
        let mut rng = StdRng::seed_from_u64(config.hyperparam.seed);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_model = base.clone();
        best_model.num_items_to_recommend = 20;
        let mut best_parameters = BTreeMap::new();
        let mut trials = Vec::new();
        let mut prepared = None;
        let trials_path = Path::new(output).join(format!("{}-trials.jsonl", algorithm));
        let mut records = std::io::BufWriter::new(std::fs::File::create(trials_path)?);
        use std::io::Write;
        for trial in 0..config.hyperparam.num_trials {
            let mut model = base.clone();
            model.num_items_to_recommend = 20;
            let mut samples = Vec::new();
            let mut parameters = BTreeMap::new();
            for axis in &mut axes {
                let sample = axis.optimizer.ask(&mut rng)?;
                let value = if axis.integer { sample.floor() } else { sample };
                apply(&mut model, axis.name, value);
                parameters.insert(
                    axis.name,
                    if axis.integer {
                        json!(value as usize)
                    } else {
                        json!(value)
                    },
                );
                samples.push(sample);
            }
            // A neighborhood cannot exceed the retrieved candidate budget.
            model.neighborhood_size_k = model.neighborhood_size_k.min(model.m_most_recent_sessions);
            parameters.insert("neighborhood_size_k", json!(model.neighborhood_size_k));
            model.knn.validate().map_err(anyhow::Error::msg)?;
            // The only changing index statistics are VMIS's IDF weight and the reference-IDF mode.
            let key = (
                model.knn.needs_reference_idf(),
                model.knn.algorithm == Algorithm::Vmis,
                model.idf_weighting.to_bits(),
            );
            if prepared != Some(key) {
                index
                    .prepare_for_model(max_m, model.idf_weighting, &model.knn)
                    .map_err(anyhow::Error::msg)?;
                prepared = Some(key);
            }
            let result = offline::evaluate(
                &index,
                &data,
                &model,
                config.hyperparam.enable_business_logic,
            );
            let score = result.mrr_at_20;
            for (axis, sample) in axes.iter_mut().zip(samples) {
                axis.optimizer.tell(sample, -score)?;
            }
            let record =
                json!({"trial":trial, "hyperparameters":parameters, "validation_mrr_at_20":score});
            writeln!(records, "{}", record)?;
            records.flush()?;
            trials.push(record);
            if score > best_score {
                best_score = score;
                best_model = model;
                best_parameters = parameters;
            }
        }
        let snippet = model_snippet(&best_model);
        let application = format!("[hyperparam]\ntraining_data_path = {:?}\nvalidation_data_path = {:?}\ntest_data_path = {:?}\n\n[server]\nhost = \"{}\"\nport = {}\nnum_workers = {}\n\n[data]\ntraining_data_path = {:?}\n\n[logic]\nenable_business_logic = {}\nexample_item_id = {}\n\n{}", training, config.hyperparam.validation_data_path, config.hyperparam.test_data_path, config.server.host, config.server.port, config.server.num_workers, training, config.hyperparam.enable_business_logic, config.logic.example_item_id, snippet);
        let application_path = Path::new(output).join(format!("{}.toml", algorithm));
        std::fs::write(&application_path, application)?;
        let reloaded = AppConfig::try_new(application_path.to_string_lossy().into_owned())?;
        anyhow::ensure!(
            serde_json::to_value(&reloaded.model)? == serde_json::to_value(&best_model)?,
            "generated configuration did not reload"
        );
        let result = json!({"algorithm":algorithm, "best_hyperparameters":best_parameters, "best_validation_mrr_at_20":best_score, "optimization":{"optimizer":"TPE", "iterations":config.hyperparam.num_trials, "seed":config.hyperparam.seed, "ranges":ranges, "objective":"maximize MRR@20"}, "evaluation":{"training_data":training, "validation_data":config.hyperparam.validation_data_path, "test_data":config.hyperparam.test_data_path, "cutoff":20, "business_logic":config.hyperparam.enable_business_logic}, "model_config":snippet, "trials":trials});
        std::fs::write(
            Path::new(output).join(format!("{}.json", algorithm)),
            serde_json::to_string_pretty(&result)?,
        )?;
        println!(
            "{}\t{:.6}\t{}\t{}",
            algorithm,
            best_score,
            serde_json::to_string(&best_parameters)?,
            application_path.display()
        );
        comparison.push(result);
    }
    std::fs::write(
        Path::new(output).join("comparison.json"),
        serde_json::to_string_pretty(&comparison)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_search_types_and_constraints() {
        assert_eq!(range("[0.25, 2.05]", false, true).unwrap(), (0.25, 2.05));
        for text in ["[0, 2]", "[1, 1]", "[2, 1]", "[NaN, 5]", "[1, inf]", "[1]"] {
            assert!(range(text, false, true).is_err(), "{}", text);
        }
        assert!(range("[1.5, 5]", true, true).is_err());
        assert!(range("[0, 5]", false, false).is_ok());
    }

    #[test]
    fn algorithm_ranges_override_shared_defaults_and_exclude_unused_axes() {
        let mut config = AppConfig::try_new("configs/hpo.toml".into()).unwrap();
        assert!(config.hyperparam.multi_algorithm);
        assert!(config.hyperparam.algorithms.values().all(|a| a.enabled));
        config
            .hyperparam
            .algorithms
            .get_mut("stan")
            .unwrap()
            .ranges
            .insert("lambda_spw".into(), "[0.25, 2.05]".into());
        config
            .hyperparam
            .algorithms
            .get_mut("stan")
            .unwrap()
            .ranges
            .insert("m_most_recent_sessions".into(), "[10, 30]".into());
        // An inapplicable parameter is not passed to TPE or range validation.
        config
            .hyperparam
            .algorithms
            .get_mut("stan")
            .unwrap()
            .ranges
            .insert("pop_boost".into(), "unused".into());
        let stan = axes(&config, Algorithm::Stan).unwrap();
        assert_eq!(stan.ranges["lambda_spw"], (0.25, 2.05));
        assert_eq!(stan.max_m, 30);
        assert!(!stan.ranges.contains_key("pop_boost"));
        assert!(!stan.ranges.contains_key("idf_weighting"));
        assert_eq!(axes(&config, Algorithm::Sknn).unwrap().max_m, 2500);
        config
            .hyperparam
            .algorithms
            .get_mut("stan")
            .unwrap()
            .ranges
            .insert("lambda_inh".into(), "[0, 2]".into());
        assert!(axes(&config, Algorithm::Stan).is_err());
    }

    #[test]
    fn generated_snippets_roundtrip_all_model_settings() {
        let path =
            std::env::temp_dir().join(format!("serenade-snippet-{}.toml", std::process::id()));
        for algorithm in Algorithm::ALL {
            let mut knn = serenade::vmisknn::algorithms::KnnConfig::for_algorithm(algorithm);
            knn.lambda_snh = None;
            knn.idf_weighting = 1.75;
            let model = ModelConfig {
                knn,
                m_most_recent_sessions: 200,
                neighborhood_size_k: 40,
                max_items_in_session: 5,
                num_items_to_recommend: 20,
                idf_weighting: 0.75,
            };
            std::fs::write(&path, format!("[data]\ntraining_data_path = \"train.tsv\"\n[logic]\nenable_business_logic = true\n{}", model_snippet(&model))).unwrap();
            let actual = AppConfig::try_new(path.to_string_lossy().into_owned()).unwrap();
            assert_eq!(
                serde_json::to_value(&actual.model).unwrap(),
                serde_json::to_value(&model).unwrap()
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}
