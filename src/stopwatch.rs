use crate::stopwatch;
use std::time::Instant;
use tdigest::TDigest;

#[derive(Clone)]
pub struct Stopwatch {
    start_time: Instant,
    prediction_durations: Vec<PositionDurationMicros>,
}

pub type PositionDurationMicros = (u32, f64);

impl Default for stopwatch::Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}

impl Stopwatch {
    pub fn new() -> Stopwatch {
        Stopwatch {
            start_time: Instant::now(),
            prediction_durations: Vec::new(),
        }
    }

    pub fn start(&mut self) {
        self.start_time = Instant::now();
    }

    pub fn stop(&mut self, position_in_session: &usize) {
        let duration = self.start_time.elapsed();
        let duration_as_micros: f64 = duration.as_micros() as f64;
        let tuple: PositionDurationMicros = (*position_in_session as u32, duration_as_micros);
        self.prediction_durations.push(tuple);
    }

    pub fn get_n(&mut self) -> usize {
        self.prediction_durations.len()
    }

    pub fn get_percentile_in_micros(&mut self, q: f64) -> f64 {
        let t_digest = TDigest::new_with_size(100);
        let durations = self
            .prediction_durations
            .iter()
            .map(|tuple| tuple.1)
            .collect();
        let sorted_digest = t_digest.merge_unsorted(durations);
        sorted_digest.estimate_quantile(q / 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentiles_use_percentage_units_from_evaluator() {
        let mut watch = Stopwatch::new();
        watch.prediction_durations = (0..101).map(|n| (0, n as f64)).collect();
        assert!((watch.get_percentile_in_micros(50.0) - 50.0).abs() < 2.0);
        assert!((watch.get_percentile_in_micros(90.0) - 90.0).abs() < 2.0);
        assert!((watch.get_percentile_in_micros(99.0) - 99.0).abs() < 2.0);
    }
}
