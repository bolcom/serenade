use crate::io::{ItemId, Time, TrainingSessionId};
use crate::metrics::SessionMetric;

use itertools::Itertools;
use itertools::__std_iter::FromIterator;
use std::cmp;
use std::collections::{HashMap, HashSet};

pub struct Popularity {
    sum_of_scores: f64,
    qty: usize,
    popularity_scores: HashMap<u64, i32>,
    length: usize,
    max_frequency: i32,
}

impl Popularity {}

impl Popularity {
    pub fn new(training_df: &[(TrainingSessionId, ItemId, Time)], length: usize) -> Popularity {
        let mut popularity_scores = HashMap::with_capacity(training_df.len());
        let mut max_frequency = 0;
        for (_session_id, item_id, _time) in training_df.iter() {
            let counter = popularity_scores.entry(*item_id).or_insert(0);
            *counter += 1;
            max_frequency = cmp::max(*counter, max_frequency);
        }

        Popularity {
            sum_of_scores: 0.0,
            qty: 0,
            popularity_scores,
            length,
            max_frequency,
        }
    }
}

impl SessionMetric for Popularity {
    fn add(&mut self, recommendations: &[u64], _next_items: &[u64]) {
        let items: HashSet<&u64> = HashSet::from_iter(
            recommendations
                .iter()
                .take(cmp::min(recommendations.len(), self.length))
                .collect_vec()
                .clone(),
        );
        self.qty += 1;
        if !items.is_empty() {
            let mut sum = 0_f64;
            for item in items.iter() {
                if let Some(item_freq) = self.popularity_scores.get(item) { sum += *item_freq as f64 / self.max_frequency as f64 }
            }
            self.sum_of_scores += sum / items.len() as f64;
        }
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Popularity@{}", self.length)
    }
}
