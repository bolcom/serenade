//! Unified quality, benchmark and CSV regression checks on every supported algorithm.
use serenade::{
    config::AppConfig,
    metrics::evaluation_reporter::EvaluationReporter,
    offline::{execution::Execution, EvaluationData},
    vmisknn::{self, algorithms::Algorithm, vmis_index::VMISIndex},
};
use std::{path::PathBuf, process::Command, sync::Mutex};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "serenade-unified-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("train.tsv"),"SessionId\tItemId\tTime\n1\t10\t100\n1\t20\t101\n1\t10\t102\n1\t30\t103\n2\t20\t110\n2\t40\t111\n3\t30\t112\n3\t40\t113\n").unwrap();
        std::fs::write(root.join("test.tsv"),"SessionId\tItemId\tTime\n1\t10\t120\n1\t20\t121\n1\t30\t122\n2\t20\t120\n2\t40\t121\n3\t30\t120\n3\t40\t121\n").unwrap();
        Self(root)
    }
    fn config(&self, name: &str, algorithm: Algorithm, m: usize) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path,format!("[data]\ntraining_data_path = \"{}\"\nindex_capacity = 10\n[hyperparam]\ntest_data_path = \"{}\"\nevaluation_workers = 2\n[model]\nalgorithm = \"{}\"\nm_most_recent_sessions = {m}\nneighborhood_size_k = 5\nmax_items_in_session = 3\nidf_weighting = 0.7\n[logic]\nenable_business_logic = false\n",self.0.join("train.tsv").to_string_lossy().replace('\\',"/"),self.0.join("test.tsv").to_string_lossy().replace('\\',"/"),algorithm.name())).unwrap();
        path
    }
    fn evaluate(
        &self,
        configs: &[PathBuf],
        flags: &[&str],
        filename: &str,
    ) -> Vec<std::collections::BTreeMap<String, String>> {
        let output = self.0.join(filename);
        let result = Command::new(env!("CARGO_BIN_EXE_evaluator"))
            .args(configs)
            .args(flags)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let console = String::from_utf8(result.stdout).unwrap();
        assert!(!console.contains(r"\\?\"));
        for name in METRICS {
            assert!(console.contains(name), "missing console metric {name}");
        }
        let mut reader = csv::Reader::from_path(output).unwrap();
        let headers = reader.headers().unwrap().clone();
        let rows: Vec<std::collections::BTreeMap<String, String>> = reader
            .records()
            .map(|r| {
                headers
                    .iter()
                    .map(str::to_string)
                    .zip(r.unwrap().iter().map(str::to_string))
                    .collect()
            })
            .collect();
        for (title, names, offset) in [
            (
                "Recommendation quality",
                vec![
                    "MRR@20",
                    "NDCG@20",
                    "HitRate@20",
                    "Precision@20",
                    "Recall@20",
                    "F1@20",
                    "MAP@20",
                ],
                3,
            ),
            (
                "Catalog exposure, diversity and result completeness",
                vec![
                    "Popularity@20",
                    "Coverage@20",
                    "Gini@20",
                    "ShannonEntropy@20",
                    "LongTailShare@20",
                    "IntraListDiversity@20",
                    "EmptyResultRate@20",
                    "FillRate@20",
                ],
                2,
            ),
        ] {
            let section = console.split(&format!("{title}\n")).nth(1).unwrap();
            let lines: Vec<_> = section.lines().skip(2).take(rows.len()).collect();
            assert_eq!(lines.len(), rows.len());
            for (row, line) in rows.iter().zip(lines) {
                let cells: Vec<_> = line.split_whitespace().collect();
                assert_eq!(cells[0], row["configuration_id"]);
                assert_eq!(cells[1], row["algorithm"]);
                for (i, name) in names.iter().enumerate() {
                    assert_eq!(
                        cells[i + offset],
                        format!("{:.4}", row[*name].parse::<f64>().unwrap())
                    );
                }
            }
        }
        assert_eq!(console.matches("Quality: per-case").count(), 1);
        assert_eq!(console.matches("Dataset 1:").count(), 1);
        if flags.contains(&"--benchmark") {
            assert_eq!(console.matches("Benchmark: medians").count(), 1);
            for heading in [
                "p50 (ms)",
                "p90 (ms)",
                "p95 (ms)",
                "p99 (ms)",
                "Predictions/s",
                "Wall (ms)",
                "Build (ms)",
                "Heap est. (bytes)",
                "Save (ms)",
                "Load (ms)",
                "Snapshot (bytes)",
            ] {
                assert!(console.contains(heading), "missing {heading}");
            }
        }
        rows
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let result = std::fs::remove_dir_all(&self.0);
        if !std::thread::panicking() {
            result.unwrap();
        }
    }
}
const METRICS: [&str; 15] = [
    "MRR@20",
    "NDCG@20",
    "HitRate@20",
    "Popularity@20",
    "Precision@20",
    "Coverage@20",
    "Recall@20",
    "F1@20",
    "MAP@20",
    "Gini@20",
    "ShannonEntropy@20",
    "LongTailShare@20",
    "IntraListDiversity@20",
    "EmptyResultRate@20",
    "FillRate@20",
];
#[test]
fn all_metrics_match_sequential_reference_and_parallel_workers() {
    let f = Fixture::new();
    for algorithm in Algorithm::ALL {
        let path = f.config("config.toml", algorithm, 5);
        let c = AppConfig::try_new(path.to_string_lossy().into_owned()).unwrap();
        let data = EvaluationData::load(f.0.join("test.tsv").to_str().unwrap()).unwrap();
        let mut index =
            VMISIndex::load_corpus(f.0.join("train.tsv").to_str().unwrap(), 10).unwrap();
        index.prepare_for_model(5, 1., &c.model.knn).unwrap();
        let mut reference = EvaluationReporter::from_sessions(index.training_sessions(), 20);
        let sessions = [
            vec![(10, 120.), (20, 121.), (30, 122.)],
            vec![(20, 120.), (40, 121.)],
            vec![(30, 120.), (40, 121.)],
        ];
        // Independently prepared legacy predictor applies the same non-unit VMIS weight.
        let legacy = VMISIndex::load_for_model(
            f.0.join("train.tsv").to_str().unwrap(),
            10,
            c.model.idf_weighting,
            &c.model.knn,
        )
        .unwrap();
        for events in sessions {
            let items: Vec<_> = events.iter().map(|e| e.0).collect();
            for state in 1..items.len() {
                let ranked = vmisknn::predict_with_model(
                    &legacy,
                    &items[..state],
                    5,
                    5,
                    20,
                    false,
                    &c.model.knn,
                    events[state - 1].1,
                )
                .into_sorted_vec()
                .iter()
                .map(|s| s.id)
                .collect::<Vec<_>>();
                reference.add(&ranked, &items[state..]);
            }
        }
        for workers in [1, 2, 4] {
            let execution = Execution::new(workers).unwrap();
            let (n, scores) = execution
                .quality(&index, &data, &c.model, false, 20)
                .unwrap();
            assert_eq!(n, 4);
            for (actual, expected) in scores.into_iter().zip(reference.all_scores()) {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{algorithm:?}: {actual} != {expected}"
                );
            }
            assert_eq!(
                execution
                    .predictions(&index, &data, &c.model, false, 20, None)
                    .unwrap(),
                n
            );
            let samples = Mutex::new(Vec::new());
            let timed = execution.predictions(&index, &data, &c.model, false, 20, Some(&samples));
            assert_eq!(timed.is_ok(), workers == 1);
            if workers == 1 {
                assert_eq!(samples.into_inner().unwrap().len(), n);
            }
        }
    }
}
#[test]
fn multiple_configs_all_algorithms_benchmark_and_missing_cells() {
    let f = Fixture::new();
    let mut configs: Vec<_> = Algorithm::ALL
        .into_iter()
        .map(|a| f.config(&format!("{}.toml", a.name()), a, 5))
        .collect();
    configs.push(f.config("second-vmis.toml", Algorithm::Vmis, 3));
    let baseline = f.evaluate(&configs, &["--workers", "1"], "single.csv");
    let parallel = f.evaluate(&[f.0.clone()], &["--workers", "4"], "parallel.csv");
    assert_eq!(baseline.len(), 8);
    assert_eq!(parallel.len(), 8);
    let snapshots = f.0.join("snapshots");
    let measured = f.evaluate(
        &configs,
        &[
            "--benchmark",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--workers",
            "2",
            "--snapshot-dir",
            snapshots.to_str().unwrap(),
        ],
        "measured.csv",
    );
    for rows in [&parallel, &measured] {
        for row in rows.iter() {
            let original = baseline
                .iter()
                .find(|r| r["configuration"] == row["configuration"])
                .unwrap();
            for name in METRICS {
                assert!(
                    (original[name].parse::<f64>().unwrap() - row[name].parse::<f64>().unwrap())
                        .abs()
                        < 1e-12
                );
            }
        }
    }
    for row in baseline {
        assert_eq!(row["idf_weighting"], "0.7");
        assert_eq!(row["knn_idf_weighting"], "0.0");
        assert!(row["index_build_ms_median"].is_empty());
        assert!(row["warmup_iterations"].is_empty());
    }
    for row in measured {
        assert_eq!(row["measured_iterations"], "2");
        assert_eq!(row["warmup_iterations"], "1");
        for (name, value) in row {
            if name.ends_with("_median") {
                assert!(value.parse::<f64>().unwrap() >= 0., "{name}");
            }
        }
    }
    let single = f.evaluate(&configs[..1], &[], "only.csv");
    assert_eq!(single.len(), 1);
    let no_snapshot = f.evaluate(
        &configs[..1],
        &["--benchmark", "--warmup", "0", "--iterations", "1"],
        "no-snapshot.csv",
    );
    assert!(no_snapshot[0]["snapshot_save_ms_median"].is_empty());
}
#[test]
fn capacity_rejected_before_data_loading_and_inputs_protected() {
    let f = Fixture::new();
    let config = f.config("bad.toml", Algorithm::Vmis, 11);
    std::fs::remove_file(f.0.join("train.tsv")).unwrap();
    let output = f.0.join("bad.csv");
    let result = Command::new(env!("CARGO_BIN_EXE_evaluator"))
        .arg(&config)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("exceeds index capacity"));
    assert!(!output.exists());
    let f = Fixture::new();
    let config = f.config("good.toml", Algorithm::Vmis, 5);
    for input in [&config, &f.0.join("train.tsv"), &f.0.join("test.tsv")] {
        let before = std::fs::read(input).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_evaluator"))
            .arg(&config)
            .arg("--output")
            .arg(input)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("overwrite an input"));
        assert_eq!(std::fs::read(input).unwrap(), before);
    }
}

#[test]
fn external_predictions_preserve_metrics_and_reject_malformed_rows() {
    let f = Fixture::new();
    let predictions = f.0.join("external predictions.txt");
    std::fs::write(&predictions, "10,20,;20,30\n;40\n40,30;30\n").unwrap();
    let training = serenade::io::read_training_data(f.0.join("train.tsv").to_str().unwrap());
    let mut reference = EvaluationReporter::new(&training, 20);
    reference.add(&[10, 20], &[20, 30]);
    reference.add(&[], &[40]);
    reference.add(&[40, 30], &[30]);
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_evaluator"))
            .arg("--train")
            .arg(f.0.join("train.tsv"))
            .arg("--predictions")
            .arg(&predictions)
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout)
            .unwrap()
            .replace("\r\n", "\n"),
        format!("{}\n{}\n", reference.get_name(), reference.result())
    );
    for bad in ["10,20", "10;", "invalid;20", "10;20;30"] {
        std::fs::write(&predictions, format!("10;20\n{bad}\n")).unwrap();
        let result = run();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("external predictions.txt:2"));
        assert!(result.stdout.is_empty());
    }
    std::fs::remove_file(&predictions).unwrap();
    assert!(!run().status.success());
}
