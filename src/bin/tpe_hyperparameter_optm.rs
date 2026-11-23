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
    choices: Vec<serde_json::Value>,
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
    choices: BTreeMap<String, Vec<serde_json::Value>>,
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
            choices: Vec::new(),
            optimizer: tpe::TpeOptimizer::new(
                tpe::parzen_estimator(),
                tpe::range(low, if integer { high + 1.0 } else { high })?,
            ),
        });
    }
    let mut choices = BTreeMap::new();
    for (name, allowed) in categorical_specs(algorithm) {
        let configured = h.algorithms[algorithm.name()].choices.get(name);
        // Normalization only rescales scores; keep it fixed unless explicitly requested.
        if name == "normalize" && configured.is_none() {
            continue;
        }
        let values = if let Some(text) = configured {
            let values: Vec<serde_json::Value> = serde_json::from_str(text)
                .map_err(|e| anyhow::anyhow!("{}.{}_choices: {}", algorithm.name(), name, e))?;
            anyhow::ensure!(!values.is_empty(), "{}_choices must not be empty", name);
            for (i, value) in values.iter().enumerate() {
                anyhow::ensure!(
                    allowed.contains(value),
                    "invalid {} choice: {}",
                    name,
                    value
                );
                anyhow::ensure!(
                    !values[..i].contains(value),
                    "duplicate {} choice: {}",
                    name,
                    value
                );
            }
            values
        } else {
            allowed
        };
        axes.push(Axis {
            name,
            integer: false,
            choices: values.clone(),
            optimizer: tpe::TpeOptimizer::new(
                tpe::histogram_estimator(),
                tpe::categorical_range(values.len())?,
            ),
        });
        choices.insert(name.to_owned(), values);
    }
    Ok(Search {
        choices,
        axes,
        max_m,
        ranges,
    })
}
fn categorical_specs(algorithm: Algorithm) -> Vec<(&'static str, Vec<serde_json::Value>)> {
    let strings = |values: &[&str]| values.iter().map(|v| json!(v)).collect();
    let mut specs = Vec::new();
    if algorithm != Algorithm::Vmis {
        specs.push(("normalize", vec![json!(false), json!(true)]));
        specs.push(("remind", vec![json!(false), json!(true)]));
    }
    match algorithm {
        Algorithm::Sknn => specs.push((
            "similarity",
            strings(&["jaccard", "cosine", "tanimoto", "binary"]),
        )),
        Algorithm::Vsknn => {
            specs.push(("similarity", strings(&["vec", "cosine"])));
            for name in ["weighting", "weighting_score"] {
                specs.push((
                    name,
                    strings(&["same", "linear", "div", "log", "quadratic"]),
                ));
            }
            for name in ["weighting_time", "idf_weighting_session"] {
                specs.push((name, vec![json!(false), json!(true)]));
            }
        }
        _ => {}
    }
    specs
}

fn apply_choice(model: &mut ModelConfig, name: &str, value: &serde_json::Value) {
    match name {
        "similarity" => model.knn.similarity = value.as_str().unwrap().parse().unwrap(),
        "weighting" => model.knn.weighting = value.as_str().unwrap().parse().unwrap(),
        "weighting_score" => model.knn.weighting_score = value.as_str().unwrap().parse().unwrap(),
        "weighting_time" => model.knn.weighting_time = value.as_bool().unwrap(),
        "idf_weighting_session" => model.knn.idf_weighting_session = value.as_bool().unwrap(),
        "remind" => model.knn.remind = value.as_bool().unwrap(),
        "normalize" => model.knn.normalize = value.as_bool().unwrap(),
        _ => unreachable!(),
    }
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
    let mut text = format!("[model]\nalgorithm = \"{}\"\nm_most_recent_sessions = {}\nneighborhood_size_k = {}\nmax_items_in_session = {}\nnum_items_to_recommend = {}\nidf_weighting = {}\n\n[model.knn]\nsimilarity = \"{:?}\"\nweighting = \"{:?}\"\nweighting_score = \"{:?}\"\nweighting_time = {}\nidf_weighting = {}\nidf_weighting_session = {}\nremind = {}\nnormalize = {}\npop_boost = {}\n", k.algorithm.name(), m.m_most_recent_sessions, m.neighborhood_size_k, m.max_items_in_session, m.num_items_to_recommend, m.idf_weighting, k.similarity, k.weighting, k.weighting_score, k.weighting_time, k.idf_weighting, k.idf_weighting_session, k.remind, k.normalize, k.pop_boost).to_lowercase();
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
    let objective = config.hyperparam.objective;
    println!("Objective: {}", objective);
    println!(
        "algorithm\tbest validation {}\thyperparameters\tapplication configuration",
        objective
    );
    let mut comparison = Vec::new();
    for (base, search) in models.into_iter().zip(searches) {
        let Search {
            mut axes,
            ranges,
            choices,
            ..
        } = search;
        let algorithm = base.knn.algorithm.name();
        let mut rng = StdRng::seed_from_u64(config.hyperparam.seed);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_model = base.clone();
        let mut best_parameters = BTreeMap::new();
        let mut trials = Vec::new();
        let mut prepared = None;
        let trials_path = Path::new(output).join(format!("{}-trials.jsonl", algorithm));
        let mut records = std::io::BufWriter::new(std::fs::File::create(trials_path)?);
        use std::io::Write;
        for trial in 0..config.hyperparam.num_trials {
            let mut model = base.clone();
            let mut samples = Vec::new();
            let mut parameters = BTreeMap::new();
            for axis in &mut axes {
                let sample = axis.optimizer.ask(&mut rng)?;
                if axis.choices.is_empty() {
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
                } else {
                    let value = axis.choices[sample as usize].clone();
                    apply_choice(&mut model, axis.name, &value);
                    parameters.insert(axis.name, value);
                }
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
            let result = offline::evaluate_at_cutoff(
                &index,
                &data,
                &model,
                config.hyperparam.enable_business_logic,
                std::num::NonZeroUsize::new(objective.cutoff()).unwrap(),
            );
            let score = result.score(objective);
            for (axis, sample) in axes.iter_mut().zip(samples) {
                axis.optimizer.tell(sample, -score)?;
            }
            let mut record = json!({"trial":trial, "hyperparameters":parameters,
                "objective":objective.to_string(), "validation_score":score, "validation_metrics":{"cutoff":result.cutoff, "mrr":result.mrr, "ndcg":result.ndcg, "hitrate":result.hitrate, "coverage":result.coverage}});
            if objective == Default::default() {
                record["validation_mrr_at_20"] = json!(score);
            }
            println!(
                "{} trial {}/{}: {} = {:.6}",
                algorithm,
                trial + 1,
                config.hyperparam.num_trials,
                objective,
                score
            );
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
        let application = format!("[hyperparam]\nobjective = {:?}\ntraining_data_path = {:?}\nvalidation_data_path = {:?}\ntest_data_path = {:?}\n\n[server]\nhost = \"{}\"\nport = {}\nnum_workers = {}\n\n[data]\ntraining_data_path = {:?}\n\n[logic]\nenable_business_logic = {}\nexample_item_id = {}\n\n{}", objective.to_string().to_lowercase(), training, config.hyperparam.validation_data_path, config.hyperparam.test_data_path, config.server.host, config.server.port, config.server.num_workers, training, config.hyperparam.enable_business_logic, config.logic.example_item_id, snippet);
        let application_path = Path::new(output).join(format!("{}.toml", algorithm));
        std::fs::write(&application_path, application)?;
        let reloaded = AppConfig::try_new(application_path.to_string_lossy().into_owned())?;
        anyhow::ensure!(
            serde_json::to_value(&reloaded.model)? == serde_json::to_value(&best_model)?,
            "generated configuration did not reload"
        );
        let mut result = json!({"algorithm":algorithm, "objective":objective.to_string(), "best_hyperparameters":best_parameters, "best_validation_score":best_score, "optimization":{"optimizer":"TPE", "iterations":config.hyperparam.num_trials, "seed":config.hyperparam.seed, "ranges":ranges, "choices":choices, "objective":format!("maximize {}", objective)}, "evaluation":{"training_data":training, "validation_data":config.hyperparam.validation_data_path, "test_data":config.hyperparam.test_data_path, "cutoff":objective.cutoff(), "business_logic":config.hyperparam.enable_business_logic}, "model_config":snippet, "trials":trials});
        if objective == Default::default() {
            result["best_validation_mrr_at_20"] = json!(best_score);
        }
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
    println!("\nAlgorithm\tBest {}", objective);
    for result in &comparison {
        println!(
            "{}\t{:.6}",
            result["algorithm"].as_str().unwrap(),
            result["best_validation_score"].as_f64().unwrap()
        );
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
    fn categorical_search_validates_samples_and_preserves_types() {
        let path =
            std::env::temp_dir().join(format!("serenade-choices-{}.toml", std::process::id()));
        std::fs::write(
            &path,
            r#"[hyperparam]
training_data_path = "train.tsv"
[hyperparam.algorithms.vsknn]
similarity_choices = ["cosine"]
normalize_choices = [false, true]
"#,
        )
        .unwrap();
        let mut config = AppConfig::try_new(path.to_string_lossy().into_owned()).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(config.hyperparam.multi_algorithm);
        let mut search = axes(&config, Algorithm::Vsknn).unwrap();
        assert_eq!(search.choices["similarity"], vec![json!("cosine")]);
        let mut rng = StdRng::seed_from_u64(42);
        let mut seen = BTreeMap::<&str, Vec<serde_json::Value>>::new();
        let mut model = config.model.clone();
        model.knn = serenade::vmisknn::algorithms::KnnConfig::for_algorithm(Algorithm::Vsknn);
        for trial in 0..50 {
            for axis in &mut search.axes {
                let sample = axis.optimizer.ask(&mut rng).unwrap();
                if !axis.choices.is_empty() {
                    let value = &axis.choices[sample as usize];
                    apply_choice(&mut model, axis.name, value);
                    let values = seen.entry(axis.name).or_default();
                    if !values.contains(value) {
                        values.push(value.clone());
                    }
                }
                axis.optimizer.tell(sample, trial as f64).unwrap();
            }
            model.knn.validate().unwrap();
        }
        assert_eq!(seen["normalize"].len(), 2);
        assert_eq!(seen["weighting"].len(), 5);
        assert_eq!(
            model.knn.similarity,
            serenade::vmisknn::algorithms::Similarity::Cosine
        );
        for text in [
            "[]",
            "[true]",
            "[\"jaccard\"]",
            "[\"vec\", \"vec\"]",
            "invalid",
        ] {
            config
                .hyperparam
                .algorithms
                .get_mut("vsknn")
                .unwrap()
                .choices
                .insert("similarity".into(), text.into());
            assert!(axes(&config, Algorithm::Vsknn).is_err(), "{}", text);
        }
        for algorithm in Algorithm::ALL {
            let specs = categorical_specs(algorithm);
            if !matches!(algorithm, Algorithm::Sknn | Algorithm::Vsknn) {
                assert!(!specs.iter().any(|(name, _)| *name == "similarity"));
            }
        }
    }

    #[test]
    fn objectives_select_different_trials_using_existing_metrics() {
        use serenade::metrics::evaluation_reporter::EvaluationReporter;
        use serenade::offline::RankingEvaluationResult;
        // Trial 0 gets the next interaction; trial 1 gets more of the remaining session.
        let trials: Vec<_> = [[1, 99, 98, 97], [99, 2, 3, 4]]
            .iter()
            .map(|recommendations| {
                let mut reporter = EvaluationReporter::from_items(1..=100, 8);
                reporter.add(recommendations, &[1, 2, 3, 4]);
                let [mrr, ndcg, hitrate, coverage] = reporter.scores();
                RankingEvaluationResult {
                    algorithm: "sknn".into(),
                    predictions: 1,
                    cutoff: 8,
                    mrr,
                    ndcg,
                    hitrate,
                    coverage,
                    duration_seconds: 0.0,
                    prediction_mean_us: 0.0,
                    prediction_p95_us: 0.0,
                }
            })
            .collect();
        for (text, expected) in [("mrr@8", 0), ("ndcg@8", 1), ("hitrate@8", 0)] {
            let objective = text.parse().unwrap();
            let mut best_score = f64::NEG_INFINITY;
            let mut best_trial = None;
            for (trial, result) in trials.iter().enumerate() {
                let score = result.score(objective);
                if score > best_score {
                    best_score = score;
                    best_trial = Some(trial);
                }
            }
            assert_eq!(best_trial, Some(expected));
            // TPE minimizes: the negative of the best metric must have the lowest loss.
            assert!(trials
                .iter()
                .all(|result| -best_score <= -result.score(objective)));
        }
    }

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
        let path =
            std::env::temp_dir().join(format!("serenade-range-test-{}.toml", std::process::id()));
        std::fs::write(&path, "[hyperparam]\ntraining_data_path = \"train.tsv\"\n[hyperparam.algorithms.stan]\nenabled = true\n").unwrap();
        let mut config = AppConfig::try_new(path.to_string_lossy().into_owned()).unwrap();
        std::fs::remove_file(path).unwrap();
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
                num_items_to_recommend: 37,
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
