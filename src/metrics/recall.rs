use crate::metrics::SessionMetric;
use std::cmp;
use std::collections::HashSet;

pub struct Recall {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl Recall {}

impl Recall {
    /// Returns a Recall evaluation metric.
    /// Recall quantifies the number of positive recommendations made out
    /// of all interacted items.
    ///
    /// # Arguments
    ///
    /// * `length` - the length aka 'k' that will be used for evaluation.
    ///

    pub fn new(length: usize) -> Recall {
        Recall {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl SessionMetric for Recall {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        self.qty += 1;
        let top_recos: HashSet<&u64> = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .collect();

        let unique_next_items: HashSet<&u64> = next_items.iter().collect();

        let intersection = top_recos.intersection(&unique_next_items);

        self.sum_of_scores += intersection.count() as f64 / next_items.len() as f64
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Recall@{}", self.length)
    }
}

#[cfg(test)]
mod recall_test {
    use super::*;

    #[test]
    fn should_calculate_recall() {
        let length = 20;
        let mut under_test = Recall::new(length);
        let recommendations: Vec<u64> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ];
        let actual_next_items: Vec<u64> = vec![3, 55, 3, 4];
        under_test.add(&recommendations, &actual_next_items);
        assert!((0.6666666666666666 - under_test.result()).abs() < f64::EPSILON);
        assert_eq!("Recall@20", under_test.get_name());
    }
}
