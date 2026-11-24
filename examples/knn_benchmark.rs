//! Single-thread kNN benchmark. See docs/knn-benchmark.md.
#[path = "support/benchmark_output.rs"]
mod output;
use anyhow::{anyhow, ensure, Context, Result};
use output::{aggregate, number, percentile, write_row, Row, MEASUREMENTS};
use serenade::{
    io,
    metrics::evaluation_reporter::EvaluationReporter,
    vmisknn::{
        self,
        algorithms::{Algorithm, KnnConfig},
        vmis_index::VMISIndex,
    },
};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const QUALITY_METRICS: [&str; 8] = [
    "MRR",
    "NDCG",
    "HitRate",
    "Popularity",
    "Precision",
    "Coverage",
    "Recall",
    "F1",
];

#[derive(Debug)]
struct Dataset {
    name: String,
    train: String,
    valid: String,
}
#[derive(Debug)]
struct Options {
    datasets: Vec<Dataset>,
    algorithms: Vec<Algorithm>,
    iterations: usize,
    warmup: usize,
    passes: usize,
    cutoff: usize,
    output: PathBuf,
    raw: Option<PathBuf>,
    snapshot: Option<PathBuf>,
    config: Option<String>,
    quiet: bool,
}
impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        ensure!(args.len() >= 3, "usage: knn_benchmark TRAIN VALID [ALGORITHM|all] [SNAPSHOT] [CONFIG] [--iterations N] [--warmup N] [--passes N] [--output FILE] [--raw FILE] [--snapshot-dir DIR] [--config FILE] [--cutoff N] [--dataset NAME TRAIN VALID] [--quiet]");
        let name = Path::new(&args[1])
            .parent()
            .and_then(Path::file_name)
            .and_then(|s| s.to_str())
            .unwrap_or("dataset")
            .to_owned();
        let mut options = Self {
            datasets: vec![Dataset {
                name,
                train: args[1].clone(),
                valid: args[2].clone(),
            }],
            algorithms: Algorithm::ALL.to_vec(),
            iterations: 3,
            warmup: 1,
            passes: 3,
            cutoff: 20,
            output: "target/knn-benchmark/results.csv".into(),
            raw: None,
            snapshot: None,
            config: None,
            quiet: false,
        };
        let mut i = 3;
        if args.get(i).is_some_and(|s| !s.starts_with("--")) {
            if !args[i].eq_ignore_ascii_case("all") {
                options.algorithms = args[i]
                    .split(',')
                    .map(|s| s.parse().map_err(|e: String| anyhow!(e)))
                    .collect::<Result<_>>()?;
            }
            i += 1;
        }
        // Retain the original positional snapshot prefix and config syntax.
        if args.get(i).is_some_and(|s| !s.starts_with("--")) {
            options.snapshot = Some(args[i].clone().into());
            i += 1;
        }
        if args.get(i).is_some_and(|s| !s.starts_with("--")) {
            options.config = Some(args[i].clone());
            i += 1;
        }
        while i < args.len() {
            let flag = args[i].as_str();
            if flag == "--quiet" {
                options.quiet = true;
                i += 1;
                continue;
            }
            let value = args
                .get(i + 1)
                .with_context(|| format!("missing value for {flag}"))?;
            match flag {
                "--iterations" => options.iterations = value.parse()?,
                "--warmup" => options.warmup = value.parse()?,
                "--passes" => options.passes = value.parse()?,
                "--cutoff" => options.cutoff = value.parse()?,
                "--output" => options.output = value.into(),
                "--raw" => options.raw = Some(value.into()),
                "--snapshot-dir" => options.snapshot = Some(Path::new(value).join("index")),
                "--config" => options.config = Some(value.clone()),
                "--dataset" => {
                    ensure!(i + 3 < args.len(), "--dataset requires NAME TRAIN VALID");
                    options.datasets.push(Dataset {
                        name: value.clone(),
                        train: args[i + 2].clone(),
                        valid: args[i + 3].clone(),
                    });
                    i += 2;
                }
                _ => return Err(anyhow!("unknown option {flag}")),
            }
            i += 2;
        }
        ensure!(
            options.iterations > 0 && options.passes > 0 && options.cutoff > 0,
            "iterations, passes and cutoff must be positive"
        );
        options
            .iterations
            .checked_add(options.warmup)
            .context("iteration count overflow")?;
        for dataset in &options.datasets {
            for path in [&dataset.train, &dataset.valid] {
                ensure!(
                    Path::new(path).is_file(),
                    "dataset file does not exist: {path}"
                );
            }
        }
        for (i, a) in options.algorithms.iter().enumerate() {
            ensure!(!options.algorithms[..i].contains(a), "duplicate algorithm");
        }
        Ok(options)
    }
    fn progress(&self, text: impl std::fmt::Display) {
        if !self.quiet {
            eprintln!("{text}");
        }
    }
}

#[derive(Clone, Copy)]
struct Limits {
    k: usize,
    m: usize,
    history: usize,
    business: bool,
    legacy_idf: f64,
}
impl Limits {
    fn load(path: Option<&String>) -> Self {
        path.map(|p| {
            let c = serenade::config::AppConfig::new(p.clone());
            Self {
                k: c.model.neighborhood_size_k,
                m: c.model.m_most_recent_sessions,
                history: c.model.max_items_in_session,
                business: c.logic.enable_business_logic,
                legacy_idf: c.model.idf_weighting as f64,
            }
        })
        .unwrap_or(Self {
            k: 100,
            m: 500,
            history: 10,
            business: false,
            legacy_idf: 1.0,
        })
    }
}

fn create_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
fn command(program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program).args(args).output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_owned())
}
fn json_cell(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        _ => value.to_string(),
    }
}
fn file_metadata(path: &str) -> Result<serde_json::Value> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut digest = md5::Context::new();
    let mut buffer = [0u8; 65536];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        digest.consume(&buffer[..size]);
    }
    let checksum = format!("{:x}", digest.finalize());
    let metadata = fs::metadata(path)?;
    Ok(
        serde_json::json!({"path": fs::canonicalize(path)?, "bytes": metadata.len(), "content_md5": checksum,
        "modified_unix_seconds": metadata.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs())}),
    )
}

// Exact, linearly interpolated quantiles of raw event counts (including repeats).
fn dataset_statistics(training: &[(io::TrainingSessionId, io::ItemId, io::Time)]) -> Row {
    let mut sessions = HashMap::new();
    let mut items = HashMap::new();
    for &(session, item, _) in training {
        *sessions.entry(session).or_insert(0u64) += 1;
        *items.entry(item).or_insert(0u64) += 1;
    }
    let mut row = Row::from([
        ("training_sessions_count".into(), sessions.len().to_string()),
        (
            "training_unique_items_count".into(),
            items.len().to_string(),
        ),
    ]);
    for (name, counts) in [
        ("session_length", sessions.into_values().collect::<Vec<_>>()),
        ("item_popularity", items.into_values().collect::<Vec<_>>()),
    ] {
        let mut values = counts;
        values.sort_unstable();
        for (label, q) in [
            ("q0", 0.),
            ("q10", 0.1),
            ("q50", 0.5),
            ("q90", 0.9),
            ("q95", 0.95),
            ("q99", 0.99),
            ("q100", 1.),
        ] {
            let value = if values.is_empty() {
                None
            } else {
                let position = (values.len() - 1) as f64 * q;
                let lo = values[position.floor() as usize] as f64;
                let hi = values[position.ceil() as usize] as f64;
                Some(lo + (hi - lo) * position.fract())
            };
            row.insert(format!("training_{name}_events_{label}"), number(value));
        }
    }
    row
}

fn overview(rows: &[Row]) -> String {
    let mut text = String::new();
    // Split quality and performance into readable tables with the same row order.
    for chunk in rows.chunk_by(|a, b| a["dataset_id"] == b["dataset_id"]) {
        let first = &chunk[0];
        text.push_str(&format!(
            "\nDataset: {} | sessions: {} | items: {} | interactions: {}\n",
            first["dataset_name"],
            first["training_sessions_count"],
            first["training_unique_items_count"],
            first["training_interactions_count"]
        ));
        for (heading, columns) in [
            (
                "Quality (fractions)",
                vec![
                    ("MRR", "MRR", 1., 4),
                    ("NDCG", "NDCG", 1., 4),
                    ("HitRate", "HitRate", 1., 4),
                    ("Popularity", "Popularity", 1., 4),
                    ("Precision", "Precision", 1., 4),
                    ("Coverage", "Coverage", 1., 4),
                    ("Recall", "Recall", 1., 4),
                    ("F1", "F1", 1., 4),
                ],
            ),
            (
                "Performance (measured-iteration medians)",
                vec![
                    ("Build ms", "index_build_ms_median", 1., 1),
                    ("Save ms", "snapshot_save_ms_median", 1., 1),
                    ("Load ms", "snapshot_load_ms_median", 1., 1),
                    ("p50 ms", "prediction_latency_p50_ms_median", 1., 4),
                    ("p90 ms", "prediction_latency_p90_ms_median", 1., 4),
                    ("p95 ms", "prediction_latency_p95_ms_median", 1., 4),
                    ("p99 ms", "prediction_latency_p99_ms_median", 1., 4),
                    (
                        "Heap MiB",
                        "index_owned_heap_estimate_bytes_median",
                        1048576.,
                        1,
                    ),
                    ("Disk MiB", "snapshot_size_bytes_median", 1048576., 1),
                ],
            ),
        ] {
            text.push_str(&format!("\n{heading}\n{:<12}", "Algorithm"));
            for (label, _, _, _) in &columns {
                let label = if heading.starts_with("Quality") {
                    format!("{label}@{}", first["recommendation_cutoff"])
                } else {
                    label.to_string()
                };
                text.push_str(&format!(" {label:>14}"));
            }
            text.push('\n');
            for row in chunk {
                text.push_str(&format!("{:<12}", row["algorithm"]));
                for (_, key, scale, decimals) in &columns {
                    let key = if heading.starts_with("Quality") {
                        format!("{key}@{}", row["recommendation_cutoff"])
                    } else {
                        key.to_string()
                    };
                    let value = row
                        .get(&key)
                        .and_then(|s| s.parse::<f64>().ok())
                        .map(|v| format!("{:.*}", decimals, v / scale))
                        .unwrap_or_else(|| "-".into());
                    text.push_str(&format!(" {value:>14}"));
                }
                text.push('\n');
            }
        }
    }
    text
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    run(args)
}

fn run(args: Vec<String>) -> Result<()> {
    let options = Options::parse(&args)?;
    let limits = Limits::load(options.config.as_ref());
    let metadata_path = options.output.with_extension("metadata.json");
    // Refuse colliding artifacts before opening any output files.
    let mut paths = vec![options.output.clone(), metadata_path.clone()];
    paths.extend(options.raw.iter().cloned());
    if let Some(prefix) = &options.snapshot {
        for d in 0..options.datasets.len() {
            for a in &options.algorithms {
                paths.push(PathBuf::from(format!(
                    "{}.{}.{}.vmis",
                    prefix.display(),
                    d + 1,
                    a.name()
                )));
            }
        }
    }
    // Canonicalize parents after creating them so relative aliases also collide.
    let mut resolved = HashSet::new();
    let mut inputs: HashSet<_> = options
        .datasets
        .iter()
        .flat_map(|d| [&d.train, &d.valid])
        .map(fs::canonicalize)
        .collect::<std::io::Result<_>>()?;
    if let Some(config) = &options.config {
        inputs.insert(fs::canonicalize(config)?);
    }
    for dataset in &options.datasets {
        let manifest = Path::new(&dataset.train).with_file_name("manifest.json");
        if manifest.is_file() {
            inputs.insert(fs::canonicalize(manifest)?);
        }
    }
    for path in &paths {
        create_parent(path)?;
        let absolute = fs::canonicalize(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?
        .join(path.file_name().context("output requires a filename")?);
        let absolute = if path.exists() {
            fs::canonicalize(path)?
        } else {
            absolute
        };
        ensure!(
            !inputs.contains(&absolute) && resolved.insert(absolute),
            "output paths must be distinct and must not overwrite datasets"
        );
    }
    let mut results = csv::Writer::from_path(&options.output)?;
    let mut result_columns = None;
    let mut raw = options
        .raw
        .as_ref()
        .map(csv::Writer::from_path)
        .transpose()?;
    let mut raw_columns = None;
    let git_commit = command("git", &["rev-parse", "HEAD"]);
    let dirty = command("git", &["status", "--porcelain"]).map(|s| !s.is_empty());
    let run_id = format!(
        "{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        std::process::id()
    );
    let mut metadata = serde_json::json!({
        "schema_version": 3, "run_id": run_id, "status": "running", "command": args,
        "git_commit": git_commit, "git_worktree_dirty": dirty,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "rustc": command("rustc", &["-vV"]), "os": std::env::consts::OS, "architecture": std::env::consts::ARCH,
        "logical_cpus": std::thread::available_parallelism().ok().map(|n| n.get()),
        "processor": std::env::var("PROCESSOR_IDENTIFIER").ok(),
        "hostname": command("hostname", &[]),
        "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(),
        "prediction_threads": 1, "benchmark_random_seed": serde_json::Value::Null,
        "randomness": "No benchmark sampling or shuffling. Algorithm/index hash maps retain their production seeding.",
        "config": options.config.as_ref().map(fs::read_to_string).transpose()?,
        "datasets": [], "completed_rows": 0,
        "methodology": "Fresh index per iteration; OS file cache is not flushed. Warmups excluded. Three prediction passes by default. Quality once in separate pass. Per-iteration nearest-rank-by-rounded-index percentiles, then arithmetic median across iterations. See docs/knn-benchmark.md."
    });
    fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)?;
    let mut completed_rows = 0;
    let mut overview_rows = Vec::new();
    for (dataset_index, dataset) in options.datasets.iter().enumerate() {
        options.progress(format_args!(
            "[Dataset {}/{}] {}",
            dataset_index + 1,
            options.datasets.len(),
            dataset.name
        ));
        let training = io::read_training_data(&dataset.train);
        let statistics = dataset_statistics(&training);
        let mut sessions: Vec<_> = io::try_read_test_data_timed(&dataset.valid)?
            .into_iter()
            .collect();
        sessions.sort_by_key(|(id, _)| *id);
        let session_items: Vec<Vec<_>> = sessions
            .iter()
            .map(|(_, events)| events.iter().map(|r| r.0).collect())
            .collect();
        let evaluation_events: usize = sessions.iter().map(|(_, e)| e.len()).sum();
        let prefixes: usize = sessions
            .iter()
            .map(|(_, e)| e.len().saturating_sub(1))
            .sum();
        ensure!(prefixes > 0, "validation data has no prediction prefixes");
        let count = prefixes
            .checked_mul(options.passes)
            .context("too many prediction samples")?;
        let mut timings = vec![0u64; count];
        let manifest_path = Path::new(&dataset.train).with_file_name("manifest.json");
        let manifest: Option<serde_json::Value> = if manifest_path.is_file() {
            Some(serde_json::from_slice(&fs::read(&manifest_path)?)?)
        } else {
            None
        };
        metadata["datasets"].as_array_mut().unwrap().push(serde_json::json!({
            "dataset_id": dataset_index+1, "name": dataset.name, "training": file_metadata(&dataset.train)?,
            "validation": file_metadata(&dataset.valid)?, "scaling_manifest": manifest,
        }));
        fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)?;
        for (algorithm_index, &algorithm) in options.algorithms.iter().enumerate() {
            options.progress(format_args!(
                "  [Algorithm {}/{}] {}",
                algorithm_index + 1,
                options.algorithms.len(),
                algorithm.name()
            ));
            let model = KnnConfig::for_algorithm(algorithm);
            let mut base = statistics.clone();
            for (key, value) in [
                ("run_id", run_id.clone()),
                ("dataset_id", (dataset_index + 1).to_string()),
                ("dataset_name", dataset.name.clone()),
                ("training_path", dataset.train.clone()),
                ("evaluation_path", dataset.valid.clone()),
                ("training_interactions_count", training.len().to_string()),
                ("evaluation_sessions_count", sessions.len().to_string()),
                ("evaluation_events_count", evaluation_events.to_string()),
                ("evaluation_prefixes_count", prefixes.to_string()),
                ("predictions_per_iteration_count", count.to_string()),
                (
                    "prediction_passes_per_iteration",
                    options.passes.to_string(),
                ),
                ("measured_iterations", options.iterations.to_string()),
                ("warmup_iterations", options.warmup.to_string()),
                ("recommendation_cutoff", options.cutoff.to_string()),
                ("neighborhood_size_k", limits.k.to_string()),
                ("candidate_sessions_m", limits.m.to_string()),
                ("max_history_items", limits.history.to_string()),
                ("business_logic_enabled", limits.business.to_string()),
                ("legacy_idf_weight", limits.legacy_idf.to_string()),
                (
                    "evaluation_order",
                    "session_id_ascending_chronological_prefixes".into(),
                ),
                (
                    "evaluation_targets",
                    "remaining_suffix_existing_serenade_metrics".into(),
                ),
            ] {
                base.insert(key.into(), value);
            }
            for (key, value) in serde_json::to_value(&model)?.as_object().unwrap() {
                base.insert(
                    if key == "lambda_snh" {
                        "lambda_snh_days".into()
                    } else {
                        key.clone()
                    },
                    json_cell(value),
                );
            }
            for (column, key) in [
                ("session_scaling_factor", "factor"),
                ("item_catalog_scaling_factor", "catalog_factor"),
                ("dataset_random_seed", "seed"),
            ] {
                base.insert(
                    column.into(),
                    manifest
                        .as_ref()
                        .and_then(|m| m.get(key))
                        .map(json_cell)
                        .unwrap_or_default(),
                );
            }
            let snapshot = options.snapshot.as_ref().map(|p| {
                format!(
                    "{}.{}.{}.vmis",
                    p.display(),
                    dataset_index + 1,
                    algorithm.name()
                )
            });
            let mut measured = Vec::with_capacity(options.iterations);
            let mut quality = None;
            for iteration in 0..options.warmup + options.iterations {
                let warmup = iteration < options.warmup;
                let iteration_number = if warmup {
                    iteration + 1
                } else {
                    iteration - options.warmup + 1
                };
                options.progress(format_args!(
                    "    {} {}/{}",
                    if warmup { "Warm-up" } else { "Iteration" },
                    iteration_number,
                    if warmup {
                        options.warmup
                    } else {
                        options.iterations
                    }
                ));
                let start = Instant::now();
                let index =
                    VMISIndex::load_for_model(&dataset.train, limits.m, limits.legacy_idf, &model)
                        .map_err(|e| anyhow!(e))?;
                let build_ms = start.elapsed().as_secs_f64() * 1000.0;
                let heap = index.estimated_heap_bytes();
                let (mut save_ms, mut load_ms, mut snapshot_bytes) = (None, None, None);
                if let Some(path) = &snapshot {
                    let start = Instant::now();
                    let saved = index.save(path);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    saved.map_err(|e| anyhow!(e.to_string()))?;
                    save_ms = Some(elapsed);
                    snapshot_bytes = Some(fs::metadata(path)?.len() as f64);
                    let start = Instant::now();
                    let restored =
                        VMISIndex::load_for_model(path, limits.m, limits.legacy_idf, &model)
                            .map_err(|e| anyhow!(e))?;
                    load_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                    std::hint::black_box(&restored);
                    drop(restored);
                }
                let mut timing_index = 0;
                for _ in 0..options.passes {
                    for ((_, events), items) in sessions.iter().zip(&session_items) {
                        for state in 1..items.len() {
                            let current = &items[state.saturating_sub(limits.history)..state];
                            let start = Instant::now();
                            let recommendations = vmisknn::predict_with_model(
                                &index,
                                current,
                                limits.k,
                                limits.m,
                                options.cutoff,
                                limits.business,
                                &model,
                                events[state - 1].1,
                            );
                            let elapsed = start.elapsed().as_nanos() as u64;
                            timings[timing_index] = elapsed;
                            timing_index += 1;
                            std::hint::black_box(recommendations);
                        }
                    }
                }
                timings.sort_unstable();
                let values = [
                    Some(build_ms),
                    save_ms,
                    load_ms,
                    Some(percentile(&timings, 0.50)),
                    Some(percentile(&timings, 0.90)),
                    Some(percentile(&timings, 0.95)),
                    Some(percentile(&timings, 0.99)),
                    Some(heap as f64),
                    snapshot_bytes,
                ];
                // Metric evaluation has its own untimed pass and happens exactly once.
                if !warmup && quality.is_none() {
                    options.progress("    Quality evaluation (once, untimed)");
                    let mut reporter = EvaluationReporter::new(&training, options.cutoff);
                    let mut ranked = Vec::with_capacity(options.cutoff);
                    for ((_, events), items) in sessions.iter().zip(&session_items) {
                        for state in 1..items.len() {
                            let recommendations = vmisknn::predict_with_model(
                                &index,
                                &items[state.saturating_sub(limits.history)..state],
                                limits.k,
                                limits.m,
                                options.cutoff,
                                limits.business,
                                &model,
                                events[state - 1].1,
                            );
                            ranked.clear();
                            ranked.extend(recommendations.into_sorted_vec().iter().map(|s| s.id));
                            reporter.add(&ranked, &items[state..]);
                        }
                    }
                    quality = Some(reporter.all_scores());
                }
                if let Some(raw) = &mut raw {
                    let mut row = base.clone();
                    row.insert(
                        "phase".into(),
                        if warmup { "warmup" } else { "measured" }.into(),
                    );
                    row.insert("iteration".into(), iteration_number.to_string());
                    for (key, value) in MEASUREMENTS.iter().zip(values) {
                        row.insert((*key).into(), number(value));
                    }
                    write_row(raw, &mut raw_columns, &row)?;
                }
                if !warmup {
                    measured.push(values);
                }
                drop(index);
            }
            let mut row = base;
            for (j, metric) in MEASUREMENTS.iter().enumerate() {
                let values: Vec<_> = measured.iter().map(|m| m[j]).collect();
                for (suffix, value) in ["median", "min", "max"]
                    .into_iter()
                    .zip(aggregate(&values)?)
                {
                    row.insert(format!("{metric}_{suffix}"), number(value));
                }
            }
            for (name, value) in QUALITY_METRICS
                .into_iter()
                .zip(quality.context("quality not evaluated")?)
            {
                row.insert(format!("{name}@{}", options.cutoff), number(Some(value)));
            }
            write_row(&mut results, &mut result_columns, &row)?;
            overview_rows.push(row);
            completed_rows += 1;
            metadata["completed_rows"] = completed_rows.into();
            fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)?;
        }
    }
    metadata["status"] = "complete".into();
    fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)?;
    options.progress(overview(&overview_rows));
    options.progress(format_args!("Results: {}", options.output.display()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_dataset_quantiles_count_repeated_events() {
        let training = [(1, 10, 0), (1, 10, 1), (1, 20, 2), (2, 20, 3), (3, 30, 4)];
        let row = dataset_statistics(&training);
        for (q, expected) in [("q0", 1.), ("q95", 2.8), ("q99", 2.96), ("q100", 3.)] {
            let actual: f64 = row[&format!("training_session_length_events_{q}")]
                .parse()
                .unwrap();
            assert!((actual - expected).abs() < 1e-12);
        }

        assert_eq!(row["training_sessions_count"], "3");
        assert_eq!(row["training_unique_items_count"], "3");
        for (name, expected) in [
            ("session_length", [1., 1., 2.6]),
            ("item_popularity", [1.2, 2., 2.]),
        ] {
            for (q, value) in ["q10", "q50", "q90"].into_iter().zip(expected) {
                assert_eq!(
                    row[&format!("training_{name}_events_{q}")]
                        .parse::<f64>()
                        .unwrap(),
                    value
                );
            }
        }
        assert_eq!(
            dataset_statistics(&[])["training_session_length_events_q50"],
            ""
        );
    }
    #[test]
    fn overview_handles_multiple_datasets_and_missing_values() {
        let base = Row::from([
            ("dataset_id".into(), "1".into()),
            ("dataset_name".into(), "sample".into()),
            ("training_sessions_count".into(), "2".into()),
            ("training_unique_items_count".into(), "4".into()),
            ("training_interactions_count".into(), "5".into()),
            ("recommendation_cutoff".into(), "20".into()),
            ("algorithm".into(), "vmis".into()),
        ]);
        let mut second = base.clone();
        second.insert("dataset_id".into(), "2".into());
        second.insert("recommendation_cutoff".into(), "5".into());
        let text = overview(&[base, second]);
        assert_eq!(text.matches("Dataset: sample").count(), 2);
        assert_eq!(text.matches("Algorithm").count(), 4);
        assert!(!text.contains("cutoff:"));
        for metric in QUALITY_METRICS {
            assert!(text.contains(&format!("{metric}@20")));
            assert!(text.contains(&format!("{metric}@5")));
        }
        assert!(text.contains("p99 ms") && text.contains("Heap MiB") && text.contains("-"));
    }
    #[test]
    fn rejects_zero_iterations_before_accessing_data() {
        let args = ["bench", "missing", "missing", "all", "--iterations", "0"].map(String::from);
        assert!(Options::parse(&args)
            .unwrap_err()
            .to_string()
            .contains("must be positive"));
    }
    #[test]
    fn end_to_end_all_algorithms_and_snapshot_missing_values() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "serenade-bench-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        fs::create_dir_all(&root)?;
        let train = root.join("train.tsv");
        let valid = root.join("valid.tsv");
        fs::write(
            &train,
            "SessionId\tItemId\tTime\n1\t10\t100\n1\t20\t101\n1\t30\t102\n2\t20\t110\n2\t40\t111\n",
        )?;
        fs::write(
            &valid,
            "SessionId\tItemId\tTime\n3\t10\t120\n3\t20\t121\n3\t30\t122\n",
        )?;
        for snapshots in [false, true] {
            let result = root.join(if snapshots {
                "snapshot.csv"
            } else {
                "results.csv"
            });
            let raw = root.join("raw.csv");
            let mut args = vec![
                "bench".into(),
                train.display().to_string(),
                valid.display().to_string(),
                if snapshots { "vmis" } else { "all" }.into(),
                "--iterations".into(),
                "2".into(),
                "--warmup".into(),
                "1".into(),
                "--passes".into(),
                "1".into(),
                "--quiet".into(),
                "--output".into(),
                result.display().to_string(),
                "--raw".into(),
                raw.display().to_string(),
            ];
            if snapshots {
                args.extend(["--cutoff".into(), "5".into()]);
                args.extend([
                    "--snapshot-dir".into(),
                    root.join("snapshots").display().to_string(),
                ]);
            } else {
                args.extend([
                    "--dataset".into(),
                    "second,\"dataset\"".into(),
                    train.display().to_string(),
                    valid.display().to_string(),
                ]);
            }
            run(args)?;
            let mut reader = csv::Reader::from_path(&result)?;
            let header = reader.headers()?.clone();
            let rows: Vec<_> = reader.records().collect::<std::result::Result<_, _>>()?;
            assert_eq!(
                rows.len(),
                if snapshots {
                    1
                } else {
                    2 * Algorithm::ALL.len()
                }
            );
            let column = |key: &str| header.iter().position(|h| h == key).unwrap();
            let mut raw_reader = csv::Reader::from_path(&raw)?;
            let raw_header = raw_reader.headers()?.clone();
            let raw_rows: Vec<_> = raw_reader
                .records()
                .collect::<std::result::Result<_, _>>()?;
            assert_eq!(raw_rows.len(), rows.len() * 3);
            let raw_column = |key: &str| raw_header.iter().position(|h| h == key).unwrap();
            for row in &rows {
                assert_eq!(&row[column("evaluation_prefixes_count")], "2");
                for metric in QUALITY_METRICS {
                    let cutoff = if snapshots { 5 } else { 20 };
                    assert!(!row[column(&format!("{metric}@{cutoff}"))].is_empty());
                }
                assert_eq!(&row[column("training_session_length_events_q50")], "2.5");
                assert_eq!(&row[column("training_item_popularity_events_q50")], "1");
                assert_eq!(&row[column("session_scaling_factor")], "");
                assert_eq!(
                    row[column("snapshot_save_ms_median")].is_empty(),
                    !snapshots
                );
                assert_eq!(
                    row[column("snapshot_size_bytes_median")].is_empty(),
                    !snapshots
                );
                let algorithm = &row[column("algorithm")];
                for metric in MEASUREMENTS {
                    let values: Vec<_> = raw_rows
                        .iter()
                        .filter(|r| {
                            &r[raw_column("algorithm")] == algorithm
                                && r[raw_column("dataset_id")] == row[column("dataset_id")]
                                && &r[raw_column("phase")] == "measured"
                        })
                        .map(|r| {
                            if r[raw_column(metric)].is_empty() {
                                None
                            } else {
                                Some(r[raw_column(metric)].parse::<f64>().unwrap())
                            }
                        })
                        .collect();
                    for (suffix, value) in ["median", "min", "max"]
                        .into_iter()
                        .zip(aggregate(&values)?)
                    {
                        assert_eq!(&row[column(&format!("{metric}_{suffix}"))], number(value));
                    }
                }
            }
            let metadata: serde_json::Value =
                serde_json::from_slice(&fs::read(result.with_extension("metadata.json"))?)?;
            assert_eq!(metadata["status"], "complete");
            assert_eq!(metadata["completed_rows"], rows.len());
            assert!(metadata["datasets"][0]["training"]["content_md5"].is_string());
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
