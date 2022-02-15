use crate::metrics::SessionMetric;

use itertools::Itertools;
use std::cmp;

pub struct HitRate {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl HitRate {}

impl HitRate {
    pub fn new(length: usize) -> HitRate {
        HitRate {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl SessionMetric for HitRate {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        self.qty += 1;
        let top_recos = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .collect_vec();
        let next_item = next_items[0];
        let index = top_recos.iter().position(|&item_id| item_id == &next_item);
        if let Some(_rank) = index { self.sum_of_scores += 1_f64 }
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("HitRate@{}", self.length)
    }
}


#[cfg(test)]
mod hitrate_test {
    use super::*;

    #[test]
    fn should_happyflow_hitrate() {
        let mut undertest = HitRate::new(20);
        let recommendations: Vec<u64> = vec![1, 2];
        let actual_next_items: Vec<u64> = vec![2, 3];
        undertest.add(&recommendations, &actual_next_items);
        assert!((1.0 - undertest.result()).abs() < f64::EPSILON);
        assert_eq!("HitRate@20", undertest.get_name());
    }

    #[test]
    fn should_handle_divide_by_zero() {
        let undertest = HitRate::new(20);
        assert!((0.0 - undertest.result()).abs() < f64::EPSILON);
    }
}
