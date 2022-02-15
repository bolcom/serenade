use crate::metrics::SessionMetric;
use std::cmp;
use std::collections::HashSet;

pub struct Precision {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl Precision {}

impl Precision {
    /// Returns a Precision evaluation metric.
    /// Precision quantifies the number of positive class predictions that
    /// actually belong to the positive class
    ///
    /// # Arguments
    ///
    /// * `length` - the length aka 'k' that will be used for evaluation.
    ///
    pub fn new(length: usize) -> Precision {
        Precision {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl SessionMetric for Precision {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        self.qty += 1;
        let top_recos: HashSet<&u64> = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .collect();

        let next_items: HashSet<&u64> = next_items.iter().collect();

        let intersection = top_recos.intersection(&next_items);

        self.sum_of_scores += intersection.count() as f64 / self.length as f64
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Precision@{}", self.length)
    }
}

#[cfg(test)]
mod precision_test {
    use super::*;

    #[test]
    fn should_calculate_precision() {
        let length = 20;
        let mut mymetric = Precision::new(length);
        let recommendations: Vec<u64> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ];
        let actual_next_items: Vec<u64> = vec![3, 55, 3, 4];
        mymetric.add(&recommendations, &actual_next_items);
        assert_eq!(2.0 / length as f64, mymetric.result());
        assert_eq!("Precision@20", mymetric.get_name());
    }
}
