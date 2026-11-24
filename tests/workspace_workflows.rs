//! Configuration paths and portable exports exercised from an unrelated working directory.
use serenade::{config, vmisknn::vmis_index::VMISIndex};
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "serenade workspace {}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("workspace")).unwrap();
        std::fs::create_dir_all(root.join("deployment")).unwrap();
        let events = "SessionId\tItemId\tTime\n1\t10\t100\n1\t20\t101\n1\t30\t102\n2\t20\t110\n2\t40\t111\n3\t30\t112\n3\t40\t113\n";
        for name in ["train.txt", "validation.txt", "test.txt"] {
            std::fs::write(root.join("workspace").join(name), events).unwrap();
        }
        std::fs::write(root.join("workspace/hpo.toml"), "[hyperparam]\nnum_trials = 2\nenable_business_logic = false\nn_most_recent_sessions_range = [2, 5]\nneighborhood_size_k_range = [1, 3]\nlast_items_in_session_range = [1, 3]\n").unwrap();
        std::fs::write(
            root.join("deployment/serenade.toml"),
            "[server]\nport = 8081\n[data]\nindex_capacity = 2500\n",
        )
        .unwrap();
        Self(root)
    }
    fn run(&self, binary: &str, args: &[&str]) -> Output {
        Command::new(binary)
            .args(args)
            .current_dir(&self.0)
            .output()
            .unwrap()
    }
    fn ok(&self, binary: &str, args: &[&str]) {
        let o = self.run(binary, args);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    fn optimize(&self) {
        self.ok(
            env!("CARGO_BIN_EXE_tpe_hyperparameter_optm"),
            &["--config", "workspace/hpo.toml"],
        );
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn rows(path: &Path) -> usize {
    csv::Reader::from_path(path).unwrap().records().count()
}

#[test]
fn default_workflow_portability_multiple_models_and_benchmark() {
    let f = Fixture::new();
    // HPO must not open or need the test set.
    let test = std::fs::read(f.0.join("workspace/test.txt")).unwrap();
    std::fs::remove_file(f.0.join("workspace/test.txt")).unwrap();
    f.optimize();
    assert_eq!(
        std::fs::read_dir(f.0.join("workspace")).unwrap().count(),
        10
    ); // 2 datasets, HPO, 7 models
    let generated = std::fs::read_to_string(f.0.join("workspace/best-vmis.toml")).unwrap();
    for forbidden in [
        "[data]",
        "[hyperparam]",
        "[server]",
        "[logic]",
        "training_data_path",
        "index_capacity",
    ] {
        assert!(!generated.contains(forbidden));
    }
    let moved = f.0.join("moved models");
    std::fs::rename(f.0.join("workspace"), &moved).unwrap();
    std::fs::write(moved.join("test.txt"), test).unwrap();
    std::fs::remove_file(moved.join("hpo.toml")).unwrap();
    let model = moved.join("best-vmis.toml");
    let model_arg = model.to_str().unwrap();
    f.ok(env!("CARGO_BIN_EXE_evaluator"), &["--model", model_arg]);
    assert_eq!(rows(&moved.join("evaluation.csv")), 1);
    f.ok(
        env!("CARGO_BIN_EXE_evaluator"),
        &[
            "--model",
            model_arg,
            "--model",
            model_arg,
            "--model",
            moved.join("best-wsknn.toml").to_str().unwrap(),
            "--benchmark",
            "--iterations",
            "2",
            "--warmup",
            "0",
        ],
    );
    assert_eq!(rows(&moved.join("benchmark.csv")), 3);
    let serving = config::load_serving(&f.0.join("deployment/serenade.toml"), &model).unwrap();
    assert_eq!(
        Path::new(&serving.data.training_data_path),
        moved.join("train.txt")
    );
}

#[test]
fn overrides_ambiguity_missing_files_and_capacity() {
    let f = Fixture::new();
    f.optimize();
    std::fs::copy(
        f.0.join("workspace/best-vmis.toml"),
        f.0.join("deployment/best-vmis.toml"),
    )
    .unwrap();
    let eval = env!("CARGO_BIN_EXE_evaluator");
    let models = [
        "--model",
        "workspace/best-vmis.toml",
        "--model",
        "deployment/best-vmis.toml",
    ];
    let o = f.run(eval, &models);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("different directories"));
    let mut args = models.to_vec();
    args.extend(["--dataset-dir", "workspace"]);
    f.ok(eval, &args);
    assert_eq!(rows(&f.0.join("workspace/evaluation.csv")), 2);
    let mut args = models.to_vec();
    args.extend([
        "--train",
        "workspace/train.txt",
        "--test",
        "workspace/test.txt",
    ]);
    let o = f.run(eval, &args);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--output"));
    args.extend(["--output", "explicit output.csv"]);
    f.ok(eval, &args);
    let o = f.run(
        eval,
        &[
            "--model",
            "workspace/best-vmis.toml",
            "--test",
            "missing test.txt",
        ],
    );
    assert!(!o.status.success());
    assert!(
        String::from_utf8_lossy(&o.stderr).contains(f.0.join("missing test.txt").to_str().unwrap())
    );
    let o = f.run(
        eval,
        &[
            "--model",
            "workspace/best-vmis.toml",
            "--index-capacity",
            "1",
        ],
    );
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("exceeds index capacity"));
    std::fs::write(f.0.join("workspace/hpo.toml"), "[hyperparam]\nnum_trials = 1\ntraining_data_path = \"../workspace/train.txt\"\nvalidation_data_path = \"../workspace/validation.txt\"\noutput_dir = \"../config output\"\nn_most_recent_sessions_range = [2, 5]\n").unwrap();
    f.optimize();
    assert!(f.0.join("config output/best-vmis.toml").is_file());
    f.ok(
        env!("CARGO_BIN_EXE_tpe_hyperparameter_optm"),
        &[
            "--config",
            "workspace/hpo.toml",
            "--train",
            "workspace/train.txt",
            "--validation",
            "workspace/validation.txt",
            "--output",
            "cli output",
            "--algorithm",
            "vmis",
        ],
    );
    assert!(f.0.join("cli output/best-vmis.toml").is_file());
}

#[test]
fn serving_overrides_and_persisted_capacity_are_independent() {
    let f = Fixture::new();
    f.optimize();
    let model = f.0.join("workspace/best-vmis.toml");
    let serving = f.0.join("deployment/serenade.toml");
    std::fs::write(&serving, "[data]\ntraining_data_path = \"../workspace/train.txt\"\nindex_capacity = 2500\n[hyperparam]\nobjective = \"invalid-irrelevant-hpo-setting\"\n").unwrap();
    let c = config::load_serving(&serving, &model).unwrap();
    assert_eq!(
        serenade::paths::corpus(&c.data.training_data_path).unwrap(),
        serenade::paths::input(f.0.join("workspace/train.txt")).unwrap()
    );
    let index = VMISIndex::load_corpus(&c.data.training_data_path, 2500).unwrap();
    let snapshot = f.0.join("deployment/index.vmis");
    index.save(snapshot.to_str().unwrap()).unwrap();
    std::fs::write(
        &serving,
        "[data]\nindex_path = \"index.vmis\"\nindex_capacity = 2500\n",
    )
    .unwrap();
    let c = config::load_serving(&serving, &model).unwrap();
    assert_eq!(Path::new(&c.data.training_data_path), snapshot);
    let mut loaded =
        VMISIndex::load_corpus(&c.data.training_data_path, c.data.index_capacity).unwrap();
    loaded
        .prepare_for_model(
            c.model.m_most_recent_sessions,
            c.model.idf_weighting,
            &c.model.knn,
        )
        .unwrap();
    assert!(VMISIndex::load_corpus(&c.data.training_data_path, 2501).is_err());
}
