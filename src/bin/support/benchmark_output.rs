use anyhow::{ensure, Result};
use std::{collections::BTreeMap, io::Write};

// Alphabetical column order is stable across algorithms, datasets and runs.
pub type Row = BTreeMap<String, String>;
pub const MEASUREMENTS: [&str; 9] = [
    "index_build_ms",
    "snapshot_save_ms",
    "snapshot_load_ms",
    "prediction_latency_p50_ms",
    "prediction_latency_p90_ms",
    "prediction_latency_p95_ms",
    "prediction_latency_p99_ms",
    "index_owned_heap_estimate_bytes",
    "snapshot_size_bytes",
];

pub fn number(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| v.to_string())
        .unwrap_or_default()
}

pub fn percentile(sorted: &[u64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize] as f64 / 1_000_000.0
}

pub fn aggregate(values: &[Option<f64>]) -> Result<[Option<f64>; 3]> {
    ensure!(!values.is_empty(), "cannot aggregate zero iterations");
    if values.iter().all(Option::is_none) {
        return Ok([None; 3]);
    }
    ensure!(
        values.iter().all(|v| v.is_some_and(f64::is_finite)),
        "measurements must be finite and present in every iteration, or absent in all"
    );
    let mut sorted: Vec<_> = values.iter().flatten().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 0 {
        sorted[n / 2 - 1] / 2.0 + sorted[n / 2] / 2.0
    } else {
        sorted[n / 2]
    };
    Ok([Some(median), Some(sorted[0]), Some(sorted[n - 1])])
}

pub fn write_row<W: Write>(
    writer: &mut csv::Writer<W>,
    columns: &mut Option<Vec<String>>,
    row: &Row,
) -> Result<()> {
    let keys: Vec<_> = row.keys().cloned().collect();
    if let Some(expected) = columns {
        ensure!(*expected == keys, "CSV columns changed between rows");
    } else {
        writer.write_record(&keys)?;
        *columns = Some(keys);
    }
    writer.write_record(row.values())?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medians_and_missing_measurements() {
        assert_eq!(
            aggregate(&[Some(9.), Some(1.), Some(5.)]).unwrap(),
            [Some(5.), Some(1.), Some(9.)]
        );
        assert_eq!(aggregate(&[Some(9.), Some(1.)]).unwrap()[0], Some(5.));
        assert_eq!(aggregate(&[None, None]).unwrap(), [None; 3]);
        assert!(aggregate(&[]).is_err());
        assert!(aggregate(&[Some(1.), None]).is_err());
        assert!(aggregate(&[Some(f64::NAN)]).is_err());
        assert_eq!(number(None), "");
        assert_eq!(number(Some(0.)), "0");
    }

    #[test]
    fn latency_quantiles_convert_nanoseconds_to_milliseconds() {
        let samples: Vec<_> = (0..=100).map(|n| n * 1_000_000).collect();
        for (q, expected) in [(0.5, 50.), (0.9, 90.), (0.95, 95.), (0.99, 99.)] {
            assert_eq!(percentile(&samples, q), expected);
        }
    }

    #[test]
    fn aggregate_iteration_percentiles_without_pooling() {
        let iterations = [vec![1000, 2000, 3000], vec![10000, 20000, 90000]];
        let p90: Vec<_> = iterations
            .iter()
            .map(|s| Some(percentile(s, 0.9)))
            .collect();
        assert_eq!(aggregate(&p90).unwrap()[0], Some(0.0465));
        let pooled = [1000, 2000, 3000, 10000, 20000, 90000];
        assert_ne!(aggregate(&p90).unwrap()[0], Some(percentile(&pooled, 0.9)));
    }

    #[test]
    fn csv_has_one_header_stable_order_escaping_and_empty_values() {
        let mut writer = csv::Writer::from_writer(Vec::new());
        let mut columns = None;
        let row = Row::from([
            ("z_missing".into(), number(None)),
            ("a_name".into(), "a,\"b\"\nline".into()),
        ]);
        write_row(&mut writer, &mut columns, &row).unwrap();
        write_row(&mut writer, &mut columns, &row).unwrap();
        let bytes = writer.into_inner().unwrap();
        let mut reader = csv::Reader::from_reader(bytes.as_slice());
        assert_eq!(
            reader.headers().unwrap(),
            &csv::StringRecord::from(vec!["a_name", "z_missing"])
        );
        let records: Vec<_> = reader.records().map(Result::unwrap).collect();
        assert_eq!(records.len(), 2);
        assert_eq!(&records[0][0], "a,\"b\"\nline");
        assert_eq!(&records[0][1], "");
        let wrong = Row::from([("different".into(), "1".into())]);
        let mut writer = csv::Writer::from_writer(Vec::new());
        assert!(write_row(&mut writer, &mut columns, &wrong).is_err());
    }
}

/// Render the exact stored values, with widths calculated from headers and cells.
pub fn table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let widths: Vec<_> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            rows.iter()
                .map(|r| r[i].chars().count())
                .chain([h.chars().count()])
                .max()
                .unwrap()
        })
        .collect();
    let line = |cells: &[String]| {
        cells
            .iter()
            .zip(&widths)
            .map(|(cell, width)| format!("{cell:width$}"))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_owned()
    };
    let mut lines = vec![line(
        &headers.iter().map(|h| h.to_string()).collect::<Vec<_>>(),
    )];
    lines.push(
        widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("  "),
    );
    lines.extend(rows.iter().map(|r| line(r)));
    lines.join("\n")
}

/// Console presentation of the same full-precision rows written to CSV.
pub fn result_tables(rows: &[Row], benchmark: bool) -> String {
    let Some(first) = rows.first() else {
        return String::new();
    };
    let mut sections = Vec::new();
    let mut datasets = Vec::new();
    for row in rows {
        let dataset = (&row["training_path"], &row["dataset"]);
        if !datasets.contains(&dataset) {
            datasets.push(dataset);
        }
    }
    for (i, (train, test)) in datasets.iter().enumerate() {
        sections.push(format!(
            "Dataset {}: training {}\n           test {}",
            i + 1,
            serenade::paths::display(train),
            serenade::paths::display(test)
        ));
    }
    for row in rows {
        let dataset = datasets
            .iter()
            .position(|d| *d == (&row["training_path"], &row["dataset"]))
            .unwrap()
            + 1;
        sections.push(format!(
            "Model {}: {} (dataset {})",
            row["configuration_id"],
            serenade::paths::display(&row["configuration"]),
            dataset
        ));
    }
    sections.push("Quality: unitless means over prediction cases; Coverage = distinct recommended items / distinct training items.".into());
    let render = |title: &str, specs: &[(&str, &str, usize)]| {
        let mut headers = vec!["Model", "Algorithm"];
        headers.extend(specs.iter().map(|s| s.1));
        let cells: Vec<_> = rows
            .iter()
            .map(|row| {
                let mut cells = vec![row["configuration_id"].clone(), row["algorithm"].clone()];
                cells.extend(specs.iter().map(|(key, _, precision)| {
                    let value = &row[*key];
                    if value.is_empty() {
                        "n/a".into()
                    } else if let Ok(value) = value.parse::<f64>() {
                        format!("{value:.precision$}")
                    } else {
                        value.clone()
                    }
                }));
                cells
            })
            .collect();
        format!("{title}\n{}", table(&headers, &cells))
    };
    sections.push(render(
        "Recommendation quality",
        &[
            ("predictions", "Cases", 0),
            ("MRR@20", "MRR@20", 4),
            ("NDCG@20", "NDCG@20", 4),
            ("HitRate@20", "HitRate@20", 4),
            ("Popularity@20", "Popularity@20", 4),
            ("Precision@20", "Precision@20", 4),
            ("Coverage@20", "Coverage@20", 4),
            ("Recall@20", "Recall@20", 4),
            ("F1@20", "F1@20", 4),
        ],
    ));
    if benchmark {
        sections.push(format!("Benchmark: medians across {} measured iterations; {} warmup iterations excluded.\nLatency percentiles use a serial prediction pass per iteration before taking medians; throughput uses the listed workers. n/a = not measured.", first["measured_iterations"], first["warmup_iterations"]));
        sections.push(render(
            "Performance",
            &[
                ("evaluation_workers", "Workers", 0),
                ("prediction_latency_p50_ms_median", "p50 (ms)", 4),
                ("prediction_latency_p90_ms_median", "p90 (ms)", 4),
                ("prediction_latency_p95_ms_median", "p95 (ms)", 4),
                ("prediction_latency_p99_ms_median", "p99 (ms)", 4),
                (
                    "prediction_throughput_predictions_per_second_median",
                    "Predictions/s",
                    1,
                ),
                ("total_evaluation_wall_ms_median", "Wall (ms)", 2),
            ],
        ));
        sections.push(render(
            "Index and persistence",
            &[
                ("index_build_ms_median", "Build (ms)", 2),
                (
                    "index_owned_heap_estimate_bytes_median",
                    "Heap est. (bytes)",
                    0,
                ),
                ("snapshot_save_ms_median", "Save (ms)", 2),
                ("snapshot_load_ms_median", "Load (ms)", 2),
                ("snapshot_size_bytes_median", "Snapshot (bytes)", 0),
            ],
        ));
    }
    sections.join("\n\n")
}

#[cfg(test)]
mod console_tests {
    #[test]
    fn columns_align_for_long_headers_and_values() {
        let rendered = super::table(
            &["Metric", "Value"],
            &[
                vec!["short".into(), "0.123456789".into()],
                vec!["long_metric_name".into(), "n/a".into()],
            ],
        );
        let lines: Vec<_> = rendered.lines().collect();
        let column = lines[0].find("Value").unwrap();
        assert_eq!(lines[2].find("0.123456789"), Some(column));
        assert_eq!(lines[3].find("n/a"), Some(column));
    }
}
