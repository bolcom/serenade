use crate::metrics::SessionMetric;
use itertools::Itertools;
use std::cmp;

pub struct Mrr {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl Mrr {}

impl Mrr {
    pub fn new(length: usize) -> Mrr {
        Mrr {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl SessionMetric for Mrr {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        self.qty += 1;
        let top_recos = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .collect_vec();
        let next_item = next_items[0];
        let index = top_recos.iter().position(|&&item_id| item_id == next_item);
        if let Some(rank) = index { self.sum_of_scores += 1_f64 / (rank as f64 + 1_f64) }
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Mrr@{}", self.length)
    }
}

#[cfg(test)]
mod mrr_test {
    use super::*;

    #[test]
    fn should_calculate_mrr() {
        let mut mymetric = Mrr::new(20);
        let recommendations: Vec<u64> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ];
        let actual_next_items: Vec<u64> = vec![3, 55, 3, 4];
        mymetric.add(&recommendations, &actual_next_items);
        assert_eq!(0.3333333333333333, mymetric.result());
        assert_eq!("Mrr@20", mymetric.get_name());
    }
}
