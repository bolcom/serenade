use chrono::NaiveDateTime;
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use tdigest::TDigest;

use crate::sessions::RocksDBSessionStore;
use crate::vmisknn::vmis_index::VMISIndex;

pub struct SharedHandlesAndConfig {
    pub knn: crate::vmisknn::algorithms::KnnConfig,
    pub session_store: Arc<RocksDBSessionStore>,
    pub vmis_index: Arc<VMISIndex>,
    pub m_most_recent_sessions: usize,
    pub neighborhood_size_k: usize,
    pub num_items_to_recommend: usize,
    pub max_items_in_session: usize,
    pub qty_workers: usize,
    pub db_compaction_ttl_in_secs: usize,
    pub enable_business_logic: bool,
    pub example_item_id: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainingDataStats {
    pub descriptive_name: String,
    pub qty_records: usize,
    pub qty_unique_session_ids: usize,
    pub qty_unique_item_ids: usize,
    pub min_time_date_time: NaiveDateTime,
    pub max_time_date_time: NaiveDateTime,
    pub session_duration_p05: u64,
    pub session_duration_p25: u64,
    pub session_duration_p50: u64,
    pub session_duration_p75: u64,
    pub session_duration_p90: u64,
    pub session_duration_p95: u64,
    pub session_duration_p99: u64,
    pub session_duration_p99_5: u64,
    pub session_duration_p100: u64,
    pub qty_events_p05: u64,
    pub qty_events_p25: u64,
    pub qty_events_p50: u64,
    pub qty_events_p75: u64,
    pub qty_events_p90: u64,
    pub qty_events_p95: u64,
    pub qty_events_p99: u64,
    pub qty_events_p99_5: u64,
    pub qty_events_p100: u64,
}

// Exact linear interpolation for <=4096 sessions; larger inputs use bounded
// batches and a 1000-centroid t-digest. Extrema never come from the digest.
const BATCH: usize = 4096;
struct Percentiles {
    digest: TDigest,
    pending: Vec<f64>,
    max: u64,
}
impl Percentiles {
    fn new() -> Self {
        Self {
            digest: TDigest::new_with_size(1000),
            pending: Vec::with_capacity(BATCH),
            max: 0,
        }
    }
    fn push(&mut self, value: u64) {
        if self.pending.len() == BATCH {
            self.digest = self.digest.merge_unsorted(std::mem::replace(
                &mut self.pending,
                Vec::with_capacity(BATCH),
            ));
        }
        self.pending.push(value as f64);
        self.max = self.max.max(value);
    }
    fn finish(mut self) -> [u64; 9] {
        let exact = self.digest.count() == 0.0;
        self.pending.sort_unstable_by(f64::total_cmp);
        if !exact {
            self.digest = self.digest.merge_sorted(std::mem::take(&mut self.pending));
        }
        let mut result = [0; 9];
        for (i, q) in [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]
            .iter()
            .enumerate()
        {
            result[i] = if exact && self.pending.is_empty() {
                0
            } else if exact {
                let pos = q * (self.pending.len() - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                (self.pending[lo] + (self.pending[hi] - self.pending[lo]) * pos.fract()).round()
                    as u64
            } else {
                self.digest.estimate_quantile(*q).round() as u64
            };
        }
        result[8] = self.max;
        result
    }
}

/// Consumes completed RAW sessions before deduplication or posting pruning.
/// No per-event storage: two bounded percentile buffers plus scalar counters.
pub(crate) struct TrainingStatsCollector {
    durations: Percentiles,
    events: Percentiles,
    records: usize,
    sessions: usize,
    min_time: f64,
    max_time: f64,
}
impl TrainingStatsCollector {
    pub(crate) fn new() -> Self {
        Self {
            durations: Percentiles::new(),
            events: Percentiles::new(),
            records: 0,
            sessions: 0,
            min_time: f64::INFINITY,
            max_time: 0.0,
        }
    }
    pub(crate) fn session(&mut self, count: usize, min: f64, max: f64) {
        assert!(count > 0 && min.is_finite() && max.is_finite() && min <= max);
        self.records += count;
        self.sessions += 1;
        self.min_time = self.min_time.min(min);
        self.max_time = self.max_time.max(max);
        self.durations.push((max - min).round() as u64);
        self.events.push(count as u64);
    }
    pub(crate) fn finish(self, name: &str, unique_items: usize) -> TrainingDataStats {
        let durations = self.durations.finish();
        let events = self.events.finish();
        let timestamp = |time: f64| {
            chrono::DateTime::from_timestamp(time.floor() as i64, 0)
                .unwrap()
                .naive_utc()
                + chrono::TimeDelta::nanoseconds((time.fract() * 1_000_000_000.0).round() as i64)
        };
        TrainingDataStats {
            descriptive_name: name.into(),
            qty_records: self.records,
            qty_unique_session_ids: self.sessions,
            qty_unique_item_ids: unique_items,
            min_time_date_time: timestamp(if self.sessions == 0 {
                0.0
            } else {
                self.min_time
            }),
            max_time_date_time: timestamp(self.max_time),
            session_duration_p05: durations[0],
            session_duration_p25: durations[1],
            session_duration_p50: durations[2],
            session_duration_p75: durations[3],
            session_duration_p90: durations[4],
            session_duration_p95: durations[5],
            session_duration_p99: durations[6],
            session_duration_p99_5: durations[7],
            session_duration_p100: durations[8],
            qty_events_p05: events[0],
            qty_events_p25: events[1],
            qty_events_p50: events[2],
            qty_events_p75: events[3],
            qty_events_p90: events[4],
            qty_events_p95: events[5],
            qty_events_p99: events[6],
            qty_events_p99_5: events[7],
            qty_events_p100: events[8],
        }
    }
}

/// Statistics for arbitrary unsorted in-memory events. The CSV loader instead
/// reuses its existing session grouping, avoiding this additional session map.
/// Empty input has zero counts/percentiles and Unix epoch timestamp sentinels.
/// Events per session counts every interaction, including repeated item visits.
pub fn determine_training_data_statistics(
    name: &str,
    data: &[(u32, u64, u32)],
) -> TrainingDataStats {
    let mut sessions = HashMap::<u32, (usize, u32, u32)>::new();
    let mut items = HashSet::new();
    for &(session, item, time) in data {
        items.insert(item);
        let summary = sessions.entry(session).or_insert((0, time, time));
        summary.0 += 1;
        summary.1 = summary.1.min(time);
        summary.2 = summary.2.max(time);
    }
    let mut collector = TrainingStatsCollector::new();
    for (_, (count, min, max)) in sessions {
        collector.session(count, min as f64, max as f64);
    }
    collector.finish(name, items.len())
}

impl TrainingDataStats {
    /// Version 1: raw interaction counts, seconds, pre-pruning aggregates.
    pub fn write_metadata(&self, path: &std::path::Path) -> anyhow::Result<()> {
        use std::io::Write;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        writer.write_all(b"SERENADE-STATS-001")?;
        bincode::serialize_into(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }
    pub fn read_metadata(path: &std::path::Path) -> anyhow::Result<Self> {
        use std::io::Read;
        let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut header = [0; 18];
        // The format tag has 18 bytes, including its version.
        reader.read_exact(&mut header)?;
        anyhow::ensure!(
            &header == b"SERENADE-STATS-001",
            "unsupported training statistics metadata version"
        );
        let stats: Self = bincode::deserialize_from(reader)?;
        anyhow::ensure!(
            stats.qty_records >= stats.qty_unique_session_ids
                && stats.qty_unique_session_ids > 0
                && stats.qty_unique_item_ids > 0
                && stats.min_time_date_time <= stats.max_time_date_time,
            "invalid training statistics metadata"
        );
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const QS: [f64; 9] = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 1.0];
    fn exact(values: &mut [u64]) -> [u64; 9] {
        values.sort_unstable();
        QS.map(|q| {
            let pos = q * (values.len() - 1) as f64;
            let lo = values[pos.floor() as usize] as f64;
            let hi = values[pos.ceil() as usize] as f64;
            (lo + (hi - lo) * pos.fract()).round() as u64
        })
    }
    fn event_quantiles(s: &TrainingDataStats) -> [u64; 9] {
        [
            s.qty_events_p05,
            s.qty_events_p25,
            s.qty_events_p50,
            s.qty_events_p75,
            s.qty_events_p90,
            s.qty_events_p95,
            s.qty_events_p99,
            s.qty_events_p99_5,
            s.qty_events_p100,
        ]
    }
    fn duration_quantiles(s: &TrainingDataStats) -> [u64; 9] {
        [
            s.session_duration_p05,
            s.session_duration_p25,
            s.session_duration_p50,
            s.session_duration_p75,
            s.session_duration_p90,
            s.session_duration_p95,
            s.session_duration_p99,
            s.session_duration_p99_5,
            s.session_duration_p100,
        ]
    }
    #[test]
    fn empty_and_singletons() {
        let empty = determine_training_data_statistics("empty", &[]);
        assert_eq!(
            (
                empty.qty_records,
                empty.qty_unique_session_ids,
                empty.qty_unique_item_ids
            ),
            (0, 0, 0)
        );
        assert_eq!(event_quantiles(&empty), [0; 9]);
        assert_eq!(duration_quantiles(&empty), [0; 9]);
        assert_eq!(empty.min_time_date_time.and_utc().timestamp(), 0);
        let single = determine_training_data_statistics("one", &[(42, 9, 123)]);
        assert_eq!(event_quantiles(&single), [1; 9]);
        assert_eq!(duration_quantiles(&single), [0; 9]);
        assert_eq!(single.min_time_date_time, single.max_time_date_time);
        assert_eq!(single.min_time_date_time.and_utc().timestamp(), 123);
    }
    #[test]
    fn unsorted_repeats_out_of_order_and_extrema() {
        let data = [
            (3, 7, 100),
            (1, 7, 9),
            (2, 8, 20),
            (1, 7, 1),
            (3, 9, 0),
            (3, 7, u32::MAX),
            (2, 8, 30),
            (3, 7, 5),
            (3, 7, 10),
        ];
        let stats = determine_training_data_statistics("raw", &data);
        assert_eq!(
            (
                stats.qty_records,
                stats.qty_unique_session_ids,
                stats.qty_unique_item_ids
            ),
            (9, 3, 3)
        );
        assert_eq!(event_quantiles(&stats), exact(&mut [2, 2, 5]));
        assert_eq!(
            duration_quantiles(&stats),
            exact(&mut [8, 10, u32::MAX as u64])
        );
        assert_eq!(stats.min_time_date_time.and_utc().timestamp(), 0);
        assert_eq!(
            stats.max_time_date_time.and_utc().timestamp(),
            u32::MAX as i64
        );
    }
    #[test]
    fn exact_boundaries_and_large_counts() {
        for size in [1, 2, 3, 100, BATCH] {
            let mut values: Vec<u64> = (0..size as u64).map(|n| n * n).collect();
            let mut p = Percentiles::new();
            for &value in values.iter().rev() {
                p.push(value);
            }
            assert_eq!(p.finish(), exact(&mut values));
        }
        let mut c = TrainingStatsCollector::new();
        c.session(1_000_000_000, 0.0, u32::MAX as f64);
        let stats = c.finish("extreme", 1);
        assert_eq!(event_quantiles(&stats), [1_000_000_000; 9]);
        assert_eq!(duration_quantiles(&stats), [u32::MAX as u64; 9]);
    }
    #[test]
    fn streaming_accuracy_uniform_skewed_and_discrete_tails() {
        for kind in 0..3 {
            let mut values: Vec<u64> = (0..200_000_u64)
                .map(|n| match kind {
                    0 => n + 1,
                    1 => (n / 100 + 1).pow(3),
                    _ => {
                        if n < 198_000 {
                            2
                        } else if n < 199_500 {
                            40
                        } else {
                            1000
                        }
                    }
                })
                .collect();
            let reference = exact(&mut values);
            // Deterministic permutation exercises unsorted batches.
            let mut p = Percentiles::new();
            for i in 0..values.len() {
                p.push(values[(i * 104729) % values.len()]);
            }
            let approximate = p.finish();
            assert_eq!(approximate[8], reference[8]);
            for i in 0..8 {
                if kind < 2 {
                    let tolerance = (reference[i] as f64 * 0.01).max(2.0);
                    assert!(
                        (approximate[i] as f64 - reference[i] as f64).abs() <= tolerance,
                        "kind={kind} q={} actual={} exact={}",
                        QS[i],
                        approximate[i],
                        reference[i]
                    );
                } else {
                    // On discontinuities, value error is unbounded; verify rank error.
                    let below = values.partition_point(|&v| v < approximate[i]) as f64
                        / values.len() as f64;
                    let above = values.partition_point(|&v| v <= approximate[i]) as f64
                        / values.len() as f64;
                    assert!(below <= QS[i] + 0.001 && above >= QS[i] - 0.001);
                }
            }
            if kind == 2 {
                assert_eq!(
                    approximate[7], 40,
                    "p99.5 must preserve this pruning boundary"
                );
            }
        }
    }
    #[test]
    #[ignore = "manual streaming throughput benchmark; run in release mode"]
    fn benchmark_ten_million_session_summaries() {
        let start = std::time::Instant::now();
        let mut collector = TrainingStatsCollector::new();
        for n in 0..10_000_000 {
            collector.session(1 + n % 100, n as f64, (n + n % 10000) as f64);
        }
        let stats = collector.finish("synthetic", 100_000);
        assert_eq!(stats.qty_unique_session_ids, 10_000_000);
        assert_eq!(stats.qty_records, 505_000_000);
        println!(
            "10M session summaries (505M raw events represented): {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }
    #[test]
    fn metadata_roundtrip_and_reject_missing_invalid_versions() {
        let path = std::env::temp_dir().join(format!("serenade-stats-{}.bin", std::process::id()));
        let stats = determine_training_data_statistics("raw", &[(1, 7, 1), (1, 7, 5)]);
        stats.write_metadata(&path).unwrap();
        assert_eq!(TrainingDataStats::read_metadata(&path).unwrap(), stats);
        std::fs::write(&path, b"SERENADE-STATS-000").unwrap();
        assert!(TrainingDataStats::read_metadata(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(TrainingDataStats::read_metadata(&path).is_err());
    }
}
