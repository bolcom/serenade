//! Unified offline quality evaluation and optional controlled benchmarking.
#[path = "support/benchmark_output.rs"]
mod output;
use anyhow::{ensure, Context, Result};
use output::{aggregate, number, percentile, write_row, Row};
use serenade::{
    config::AppConfig,
    offline::{self, execution::Execution, EvaluationData},
    vmisknn::vmis_index::VMISIndex,
};
use std::{path::PathBuf, sync::Mutex, time::Instant};

fn main() -> Result<()> {
    run(std::env::args().skip(1).collect())
}
fn run(args: Vec<String>) -> Result<()> {
    let mut paths = Vec::new();
    let (mut benchmark, mut iterations, mut warmup, mut workers) = (false, 3usize, 1usize, None);
    let mut output = PathBuf::from("evaluation-results.csv");
    let mut snapshot = None;
    let mut selection = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--benchmark" => benchmark = true,
            "--iterations" | "--warmup" | "--workers" | "--output" | "--snapshot-dir"
            | "--algorithm" => {
                i += 1;
                let value = args.get(i).context("missing option value")?;
                match arg.as_str() {
                    "--iterations" => iterations = value.parse()?,
                    "--warmup" => warmup = value.parse()?,
                    "--workers" => workers = Some(value.parse::<usize>()?),
                    "--output" => output = value.into(),
                    "--snapshot-dir" => snapshot = Some(PathBuf::from(value)),
                    _ => selection = Some(value.clone()),
                }
            }
            _ => {
                ensure!(!arg.starts_with("--"), "unknown option {arg}");
                let path = PathBuf::from(arg);
                if path.is_dir() {
                    let mut entries = std::fs::read_dir(path)?
                        .map(|e| e.map(|e| e.path()))
                        .collect::<std::io::Result<Vec<_>>>()?;
                    entries.sort();
                    paths.extend(
                        entries
                            .into_iter()
                            .filter(|p| p.extension().is_some_and(|e| e == "toml")),
                    );
                } else if arg == "all"
                    || arg
                        .parse::<serenade::vmisknn::algorithms::Algorithm>()
                        .is_ok()
                {
                    selection = Some(arg.clone());
                } else {
                    paths.push(path);
                }
            }
        }
        i += 1;
    }
    ensure!(!paths.is_empty(), "usage: evaluator CONFIG...|DIRECTORY [--output FILE.csv] [--algorithm NAME|all] [--benchmark --iterations N --warmup N --snapshot-dir DIR] [--workers N]");
    ensure!(iterations > 0, "iterations must be positive");
    iterations
        .checked_add(warmup)
        .context("iteration count overflow")?;
    ensure!(
        benchmark || snapshot.is_none(),
        "--snapshot-dir requires --benchmark"
    );
    // Validate every configuration before loading any corpus or opening output.
    let mut jobs = Vec::new();
    for path in &paths {
        let config = AppConfig::try_new(path.to_string_lossy().into_owned())?;
        for model in offline::selected_models(&config.model, selection.as_deref())? {
            ensure!(
                model.m_most_recent_sessions <= config.data.index_capacity,
                "{}: model m exceeds data.index_capacity",
                path.display()
            );
            model.knn.validate().map_err(anyhow::Error::msg)?;
            jobs.push((
                path.clone(),
                config.data.training_data_path.clone(),
                config.hyperparam.test_data_path.clone(),
                config.data.index_capacity,
                model,
                config.logic.enable_business_logic,
                workers.unwrap_or(config.hyperparam.evaluation_workers),
            ));
        }
    }
    // Group equal datasets/capacities to reuse one corpus and one evaluation dataset.
    jobs.sort_by(|a, b| (&a.1, a.3, &a.2).cmp(&(&b.1, b.3, &b.2)));
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    fn resolved(path: &std::path::Path) -> Result<PathBuf> {
        if path.exists() {
            return Ok(std::fs::canonicalize(path)?);
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        Ok(std::fs::canonicalize(parent)?
            .join(path.file_name().context("output requires filename")?))
    }
    let absolute_output = resolved(&output)?;
    for path in paths.iter().chain(
        jobs.iter()
            .flat_map(|j| [PathBuf::from(&j.1), PathBuf::from(&j.2)])
            .collect::<Vec<_>>()
            .iter(),
    ) {
        ensure!(
            resolved(path)? != absolute_output,
            "output must not overwrite an input"
        );
    }
    if let Some(dir) = &snapshot {
        std::fs::create_dir_all(dir)?;
        for job_id in 0..jobs.len() {
            let file = resolved(&dir.join(format!("index-{job_id}.vmis")))?;
            ensure!(file != absolute_output, "snapshot and CSV paths collide");
            for input in paths.iter().cloned().chain(
                jobs.iter()
                    .flat_map(|j| [PathBuf::from(&j.1), PathBuf::from(&j.2)]),
            ) {
                ensure!(
                    resolved(&input)? != file,
                    "snapshot must not overwrite an input"
                );
            }
        }
    }
    let mut writer = csv::Writer::from_path(&output)?;
    let mut columns = None;
    let mut cached_corpus = None;
    let mut cached_data = None;
    let mut corpus = None;
    let mut data = None;
    for (job_id, (path, train, test, capacity, model, business, workers)) in
        jobs.into_iter().enumerate()
    {
        eprintln!(
            "Dataset {test}, algorithm {}, configuration {}",
            model.knn.algorithm.name(),
            path.display()
        );
        let key = (train.clone(), capacity);
        if cached_data.as_ref() != Some(&test) {
            data = Some(EvaluationData::load(&test)?);
            cached_data = Some(test.clone());
        }
        if cached_corpus.as_ref() != Some(&key) {
            drop(corpus.take()); // Drop the previous corpus before loading another.
            if !benchmark {
                corpus =
                    Some(VMISIndex::load_corpus(&train, capacity).map_err(anyhow::Error::msg)?);
            }
            cached_corpus = Some(key);
        }
        let data = data.as_ref().unwrap();
        let execution = Execution::new(workers)?;
        let serial = if benchmark {
            Some(Execution::new(1)?)
        } else {
            None
        };
        let mut row = Row::from([
            ("dataset".into(), test.clone()),
            ("training_path".into(), train.clone()),
            ("configuration".into(), path.display().to_string()),
            ("configuration_id".into(), (job_id + 1).to_string()),
            ("algorithm".into(), model.knn.algorithm.name().into()),
            ("index_capacity".into(), capacity.to_string()),
            ("evaluation_workers".into(), execution.workers().to_string()),
            ("business_logic_enabled".into(), business.to_string()),
            (
                "measured_iterations".into(),
                if benchmark {
                    iterations.to_string()
                } else {
                    String::new()
                },
            ),
            (
                "warmup_iterations".into(),
                if benchmark {
                    warmup.to_string()
                } else {
                    String::new()
                },
            ),
        ]);
        for (name, value) in serde_json::to_value(&model)?.as_object().unwrap() {
            if name == "knn" {
                for (key, value) in value.as_object().unwrap() {
                    if key != "algorithm" {
                        row.insert(
                            format!("knn_{key}"),
                            if value.is_null() {
                                String::new()
                            } else {
                                value.to_string()
                            },
                        );
                    }
                }
            } else {
                row.insert(name.clone(), value.to_string());
            }
        }
        let mut measurements = Vec::new();
        let mut quality = None;
        let count = if benchmark { warmup + iterations } else { 1 };
        for iteration in 0..count {
            if benchmark {
                eprintln!(
                    "  {test} / {} / {}: {} iteration {}",
                    model.knn.algorithm.name(),
                    path.display(),
                    if iteration < warmup {
                        "warm-up"
                    } else {
                        "measured"
                    },
                    if iteration < warmup {
                        iteration + 1
                    } else {
                        iteration - warmup + 1
                    }
                );
            }
            let total = Instant::now();
            let build = Instant::now();
            if benchmark {
                corpus =
                    Some(VMISIndex::load_corpus(&train, capacity).map_err(anyhow::Error::msg)?);
            }
            let index = corpus.as_mut().unwrap();
            index
                .prepare_for_model(model.m_most_recent_sessions, 1., &model.knn)
                .map_err(anyhow::Error::msg)?;
            let build_ms = build.elapsed().as_secs_f64() * 1000.;
            let (mut save, mut load, mut bytes) = (None, None, None);
            if let Some(dir) = &snapshot {
                std::fs::create_dir_all(dir)?;
                let file = dir.join(format!("index-{job_id}.vmis"));
                let start = Instant::now();
                index
                    .save(file.to_str().context("snapshot path is not UTF-8")?)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                save = Some(start.elapsed().as_secs_f64() * 1000.);
                bytes = Some(std::fs::metadata(&file)?.len() as f64);
                // Release the original before loading the snapshot: never duplicate the corpus.
                drop(corpus.take());
                let start = Instant::now();
                corpus = Some(
                    VMISIndex::load_corpus(file.to_str().unwrap(), capacity)
                        .map_err(anyhow::Error::msg)?,
                );
                load = Some(start.elapsed().as_secs_f64() * 1000.);
            }
            let index = corpus.as_ref().unwrap();
            let quality_start = Instant::now();
            if quality.is_none() && (!benchmark || iteration >= warmup) {
                quality = Some(execution.quality(index, data, &model, business, 20)?);
            }
            let quality_duration = quality_start.elapsed();
            if benchmark {
                let samples = Mutex::new(Vec::with_capacity(data.prediction_count()));
                serial.as_ref().unwrap().predictions(
                    index,
                    data,
                    &model,
                    business,
                    20,
                    Some(&samples),
                )?;
                let mut samples = samples.into_inner().unwrap();
                samples.sort_unstable();
                let start = Instant::now();
                let n = execution.predictions(index, data, &model, business, 20, None)?;
                let throughput = n as f64 / start.elapsed().as_secs_f64();
                let values = [
                    Some(build_ms),
                    save,
                    load,
                    Some(percentile(&samples, 0.5)),
                    Some(percentile(&samples, 0.9)),
                    Some(percentile(&samples, 0.95)),
                    Some(percentile(&samples, 0.99)),
                    Some(index.estimated_heap_bytes() as f64),
                    bytes,
                    Some(throughput),
                    Some((total.elapsed() - quality_duration).as_secs_f64() * 1000.),
                ];
                if iteration >= warmup {
                    measurements.push(values);
                }
                drop(corpus.take());
            }
        }
        let (predictions, scores) = quality.context("quality not evaluated")?;
        row.insert("predictions".into(), predictions.to_string());
        for (name, value) in [
            "MRR",
            "NDCG",
            "HitRate",
            "Popularity",
            "Precision",
            "Coverage",
            "Recall",
            "F1",
        ]
        .into_iter()
        .zip(scores)
        {
            row.insert(format!("{name}@20"), number(Some(value)));
        }
        for (j, name) in output::MEASUREMENTS
            .iter()
            .chain(
                [
                    "prediction_throughput_predictions_per_second",
                    "total_evaluation_wall_ms",
                ]
                .iter(),
            )
            .enumerate()
        {
            let value = if benchmark {
                aggregate(&measurements.iter().map(|v| v[j]).collect::<Vec<_>>())?[0]
            } else {
                None
            };
            row.insert(format!("{name}_median"), number(value));
        }
        write_row(&mut writer, &mut columns, &row)?;
    }
    eprintln!("Results: {}", output.display());
    Ok(())
}
