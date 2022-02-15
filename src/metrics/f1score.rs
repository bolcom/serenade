use crate::metrics::precision::Precision;
use crate::metrics::recall::Recall;
use crate::metrics::SessionMetric;

pub struct F1score {
    precision: Precision,
    recall: Recall,
    length: usize,
}

impl F1score {}

impl F1score {
    pub fn new(length: usize) -> F1score {
        F1score {
            precision: Precision::new(length),
            recall: Recall::new(length),
            length,
        }
    }
}

impl SessionMetric for F1score {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        self.precision.add(recommendations, next_items);
        self.recall.add(recommendations, next_items);
    }

    fn result(&self) -> f64 {
        let precision_score = self.precision.result();
        let recall_score = self.recall.result();
        let f1score: f64 =
            2.0 * (precision_score * recall_score) / (precision_score + recall_score);
        if f1score.is_nan() {
            0.0
        } else {
            f1score
        }
    }

    fn get_name(&self) -> String {
        format!("F1score@{}", self.length)
    }
}

#[cfg(test)]
mod f1score_test {
    use super::*;

    #[test]
    fn should_happyflow_f1score() {
        let mut undertest = F1score::new(20);
        let recommendations: Vec<u64> = vec![1, 2];
        let actual_next_items: Vec<u64> = vec![2, 3];
        undertest.add(&recommendations, &actual_next_items);
        assert!((0.09090909090909091 - undertest.result()).abs() < f64::EPSILON);
        assert_eq!("F1score@20", undertest.get_name());
    }

    #[test]
    fn should_handle_divide_by_zero() {
        let undertest = F1score::new(20);
        assert!((0.0 - undertest.result()).abs() < f64::EPSILON);
    }
}
