//! Small end-to-end runs check shared objectives, best-trial reporting and generated configs.
use serde_json::Value;
use serenade::{
    metrics::objective::Objective,
    offline::{self, EvaluationData},
    vmisknn::vmis_index::VMISIndex,
};
use std::{path::Path, process::Command};

fn run(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn all_algorithms_optimize_selected_metric_and_generate_usable_models() {
    let root = std::env::temp_dir().join(format!("serenade-hpo-objective-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let train = root.join("train.tsv");
    let holdout = root.join("holdout.tsv");
    let mut events = String::from("SessionId\tItemId\tTime\n");
    for session in 0..30 {
        for (position, item) in [1, 2 + session % 5, 8 + session % 3, 1].iter().enumerate() {
            events.push_str(&format!(
                "{}\t{}\t{}\n",
                session,
                item,
                100 + session + position
            ));
        }
    }
    std::fs::write(&train, events).unwrap();
    std::fs::write(&holdout, "SessionId\tItemId\tTime\n1\t1\t200\n1\t3\t201\n1\t8\t202\n2\t2\t200\n2\t1\t201\n2\t10\t202\n").unwrap();
    let config_path = root.join("hpo.toml");
    for text in ["mrr@20", "ndcg@8", "hitrate@10"] {
        let objective: Objective = text.parse().unwrap();
        let output = root.join(text.replace('@', "-"));
        // JSON strings also provide suitable quoting for these TOML paths.
        std::fs::write(&config_path, format!("[hyperparam]\nobjective = {text:?}\ntraining_data_path = {}\nvalidation_data_path = {}\ntest_data_path = {}\nnum_trials = 5\nseed = 42\nenable_business_logic = false\nn_most_recent_sessions_range = [5, 30]\nneighborhood_size_k_range = [2, 15]\nlast_items_in_session_range = [1, 3]\n[model]\nnum_items_to_recommend = 37\n", serde_json::to_string(&train.to_string_lossy().replace('\\', "/")).unwrap(), serde_json::to_string(&holdout.to_string_lossy().replace('\\', "/")).unwrap(), serde_json::to_string(&holdout.to_string_lossy().replace('\\', "/")).unwrap())).unwrap();
        let console = run(Command::new(env!("CARGO_BIN_EXE_tpe_hyperparameter_optm"))
            .arg(&config_path)
            .arg("all")
            .arg(&output));
        assert!(console.contains(&format!("Objective: {objective}")));
        assert!(console.contains(&format!("Best {objective}")));
        let comparison = json(output.join("comparison.json"));
        let results = comparison.as_array().unwrap();
        assert_eq!(
            results.len(),
            serenade::vmisknn::algorithms::Algorithm::ALL.len()
        );
        for result in results {
            assert_eq!(result["objective"], objective.to_string());
            assert_eq!(result["evaluation"]["cutoff"], objective.cutoff());
            let algorithm = result["algorithm"].as_str().unwrap();
            assert_eq!(json(output.join(format!("{algorithm}.json"))), *result);
            let trials = result["trials"].as_array().unwrap();
            assert_eq!(trials.len(), 5);
            let metric = text.split('@').next().unwrap();
            let mut best = &trials[0];
            for trial in trials {
                assert_eq!(trial["objective"], objective.to_string());
                assert_eq!(trial["validation_metrics"]["cutoff"], objective.cutoff());
                assert_eq!(
                    trial["validation_score"],
                    trial["validation_metrics"][metric]
                );
                if trial["validation_score"].as_f64().unwrap()
                    > best["validation_score"].as_f64().unwrap()
                {
                    best = trial;
                }
                assert_eq!(
                    trial.get("validation_mrr_at_20").is_some(),
                    text == "mrr@20"
                );
            }
            assert_eq!(result["best_validation_score"], best["validation_score"]);
            assert_eq!(result["best_hyperparameters"], best["hyperparameters"]);
            assert_eq!(
                result.get("best_validation_mrr_at_20").is_some(),
                text == "mrr@20"
            );
            let records: Vec<Value> =
                std::fs::read_to_string(output.join(format!("{algorithm}-trials.jsonl")))
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
            assert_eq!(&records, trials);
            // Both serving and evaluator use this same loader and model contract.
            let model =
                serenade::config::load_model(&output.join(format!("best-{algorithm}.toml")))
                    .unwrap();
            assert_eq!(model.num_items_to_recommend, 37);
            for trial in trials {
                assert_eq!(trial["validation_metrics"].as_object().unwrap().len(), 2);
            }
            let selected = serde_json::to_value(&model.knn).unwrap();
            for (name, choices) in result["optimization"]["choices"].as_object().unwrap() {
                assert_eq!(selected[name], best["hyperparameters"][name]);
                for trial in trials {
                    assert!(choices
                        .as_array()
                        .unwrap()
                        .contains(&trial["hyperparameters"][name]));
                }
            }

            let index = VMISIndex::load_for_model(
                train.to_str().unwrap(),
                model.m_most_recent_sessions,
                model.idf_weighting,
                &model.knn,
            )
            .unwrap();
            let validation = EvaluationData::load(holdout.to_str().unwrap()).unwrap();
            let reproduced = offline::evaluate_at_cutoff(
                &index,
                &validation,
                &model,
                false,
                std::num::NonZeroUsize::new(objective.cutoff()).unwrap(),
            );
            assert_eq!(
                reproduced.score(objective),
                result["best_validation_score"].as_f64().unwrap()
            );
        }
        let csv_path = output.join("evaluation.csv");
        let mut evaluation = Command::new(env!("CARGO_BIN_EXE_evaluator"));
        for result in results {
            evaluation.arg("--model").arg(output.join(format!(
                "best-{}.toml",
                result["algorithm"].as_str().unwrap()
            )));
        }
        run(evaluation
            .arg("--train")
            .arg(&train)
            .arg("--test")
            .arg(&holdout)
            .arg("--output")
            .arg(&csv_path));
        let mut reader = csv::Reader::from_path(csv_path).unwrap();
        let header = reader.headers().unwrap().clone();
        let rows: Vec<_> = reader.records().map(Result::unwrap).collect();
        assert_eq!(rows.len(), results.len());
        let column = header.iter().position(|h| h == "MRR@20").unwrap();
        for row in rows {
            assert!(row[column].parse::<f64>().unwrap().is_finite());
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_search_above_fixed_capacity_before_loading_data() {
    let root = std::env::temp_dir().join(format!("serenade-hpo-capacity-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let config = root.join("hpo.toml");
    std::fs::write(&config, "[data]\nindex_capacity = 29\n[hyperparam]\ntraining_data_path = \"missing-training.tsv\"\nvalidation_data_path = \"missing-validation.tsv\"\nn_most_recent_sessions_range = [5, 30]\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_tpe_hyperparameter_optm"))
        .arg(&config)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(
        error.contains("exceeds fixed data.index_capacity 29"),
        "{error}"
    );
    assert!(!root.join("hpo-results").exists());
    std::fs::remove_dir_all(root).unwrap();
}
