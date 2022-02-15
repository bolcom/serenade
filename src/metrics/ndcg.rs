use crate::metrics::SessionMetric;
use itertools::Itertools;
use std::cmp;
use std::collections::HashSet;

pub struct Ndcg {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl Ndcg {
    fn dcg(&self, top_recos: &[&u64], next_items: &[&u64]) -> f64 {
        let mut result = 0_f64;
        let next_items_set: HashSet<&u64> = next_items.iter().cloned().collect::<HashSet<_>>();
        for (index, _item_id) in top_recos.iter().enumerate() {
            if next_items_set.contains(top_recos[index]) {
                if index == 0 {
                    result += 1_f64;
                } else {
                    result += 1_f64 / ((index as f64) + 1_f64).log2();
                }
            }
        }
        result
    }
}

impl Ndcg {
    //
    /// Calculate Ndcg for predicted recommendations and the given next items that will be interacted with.
    pub fn new(length: usize) -> Ndcg {
        Ndcg {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl SessionMetric for Ndcg {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]) {
        let top_recos = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .collect_vec();
        let top_next_items = next_items
            .iter()
            .take(cmp::min(next_items.len(), self.length))
            .collect_vec();
        let next_items = next_items.iter().collect_vec();
        let dcg: f64 = self.dcg(&top_recos, &next_items);
        let dcg_max: f64 = self.dcg(&top_next_items, &next_items);
        self.sum_of_scores += dcg / dcg_max;
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
