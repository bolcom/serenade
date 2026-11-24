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

pub fn print_result(row: &Row, benchmark: bool) {
    println!(
        "\nResult {}: {} ({} predictions)",
        row["configuration_id"], row["algorithm"], row["predictions"]
    );
    println!("Quality: mean over prediction cases; Coverage is distinct recommended items / distinct training items.");
    let quality: Vec<_> = row
        .iter()
        .filter(|(name, _)| name.contains('@'))
        .map(|(name, value)| vec![name.clone(), value.clone()])
        .collect();
    println!(
        "{}",
        table(&["Quality metric", "Value (unitless)"], &quality)
    );
    if benchmark {
        println!(
            "\nPerformance: median across {} measured iterations; {} warmup iterations excluded.",
            row["measured_iterations"], row["warmup_iterations"]
        );
        println!("Latency percentiles are computed per iteration before taking medians; throughput uses {} workers.", row["evaluation_workers"]);
        let performance: Vec<_> = row
            .iter()
            .filter(|(name, _)| name.ends_with("_median"))
            .map(|(name, value)| {
                vec![
                    name.clone(),
                    if name.contains("_ms_") {
                        "ms"
                    } else if name.contains("_bytes_") {
                        "bytes"
                    } else {
                        "predictions/s"
                    }
                    .into(),
                    if value.is_empty() {
                        "n/a".into()
                    } else {
                        value.clone()
                    },
                ]
            })
            .collect();
        println!(
            "{}",
            table(&["Performance metric", "Unit", "Value"], &performance)
        );
    }
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
