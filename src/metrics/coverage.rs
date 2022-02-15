use crate::io::{ItemId, Time, TrainingSessionId};
use crate::metrics::SessionMetric;

use hashbrown::HashSet;
use itertools::Itertools;
use std::cmp;

pub struct Coverage {
    unique_training_items: usize,
    test_items: HashSet<u64>,
    length: usize,
}

impl Coverage {}

impl Coverage {
    pub fn new(training_df: &[(TrainingSessionId, ItemId, Time)], length: usize) -> Coverage {
        let mut distinct_item_ids = training_df.iter().map(|record| record.1).collect_vec();
        distinct_item_ids.sort_unstable();
        distinct_item_ids.dedup();
        Coverage {
            unique_training_items: distinct_item_ids.len(),
            test_items: HashSet::new(),
            length,
        }
    }
}

impl SessionMetric for Coverage {
    fn add(&mut self, recommendations: &[u64], _next_items: &[u64]) {
        let top_recos = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .collect_vec();
        for item_id in top_recos.into_iter() {
            self.test_items.insert(*item_id);
        }
    }

    fn result(&self) -> f64 {
        if self.unique_training_items > 0 {
            self.test_items.len() as f64 / self.unique_training_items as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Coverage@{}", self.length)
    }
}
