use crate::metrics::SessionMetric;
use std::collections::HashSet;

pub struct Ndcg {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
    relevant: HashSet<u64>,
}

impl Ndcg {
    //
    /// Calculate Ndcg for predicted recommendations and the given next items that will be interacted with.
    pub fn new(length: usize) -> Ndcg {
        Ndcg {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
            relevant: HashSet::new(),
        }
    }
}

impl SessionMetric for Ndcg {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        self.relevant.clear();
        self.relevant.extend(next_items.iter().copied());
        let gain = |rank: usize| {
            if rank == 0 {
                1.0
            } else {
                1.0 / (rank as f64 + 1.0).log2()
            }
        };
        let dcg: f64 = recommendations
            .iter()
            .take(self.length)
            .enumerate()
            .filter(|(_, item)| self.relevant.contains(item))
            .map(|(rank, _)| gain(rank))
            .sum();
        // Preserve the historical definition, including repeated target events.
        let ideal: f64 = (0..next_items.len().min(self.length)).map(gain).sum();
        self.sum_of_scores += dcg / ideal;
        self.qty += 1;
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Ndcg@{}", self.length)
    }
}

#[cfg(test)]
mod ndcg_test {
    use super::*;

    #[test]
    fn should_calculate_ndcg() {
        let mut mymetric = Ndcg::new(20);
        let recommendations: Vec<u64> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ];
        let actual_next_items: Vec<u64> = vec![3, 55, 88, 4];
        mymetric.add(&recommendations, &actual_next_items);
        assert_eq!(0.36121211352040195, mymetric.result());
        assert_eq!("Ndcg@20", mymetric.get_name());
    }
}

impl Ndcg {
    pub(crate) fn merge(&mut self, other: Self) {
        self.sum_of_scores += other.sum_of_scores;
        self.qty += other.qty;
    }
}
