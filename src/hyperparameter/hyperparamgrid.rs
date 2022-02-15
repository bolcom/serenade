use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;

pub struct HyperParamGrid {
    pub param_grid: HashMap<String, Vec<usize>>,
}

impl HyperParamGrid {
    /// Returns 'n' unique random combinations from all hyperparameter combinations or less depending on the amount of combinations possible.
    /// This function assumes that the given values for a parameter are unique (ofcourse)
    ///
    /// # Arguments
    ///
    /// * `n` - the requested amount of random results returned.
    pub fn get_n_random_combinations(
        &self,
        n: usize,
    ) -> Vec<HashMap<String, usize, RandomState>> {
        let mut all_combinations = self.get_all_combinations();
        all_combinations.shuffle(&mut thread_rng());
        all_combinations.into_iter().take(n).collect()
    }

    pub fn get_all_combinations(&self) -> Vec<HashMap<String, usize, RandomState>> {
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for (k, vs) in self.param_grid.iter() {
            keys.push(k.clone());
            values.push(vs.clone());
        }
        let all_pairs = HyperParamGrid::cartesian_product(&values);

        let mut result = Vec::new();
        for param_values in all_pairs.iter() {
            let mut combi = HashMap::new();
            for (index, param_name) in keys.iter().enumerate() {
                combi.insert(param_name.clone(), *param_values.get(index).unwrap());
            }
            result.push(combi);
        }
        result
    }

    fn cartesian_product(lists: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let mut res = vec![];

        let mut list_iter = lists.iter();
        if let Some(first_list) = list_iter.next() {
            for &i in first_list {
                res.push(vec![i]);
            }
        }
        for l in list_iter {
            let mut tmp = vec![];
            for r in res {
                for &el in l {
                    let mut tmp_el = r.clone();
                    tmp_el.push(el);
                    tmp.push(tmp_el);
                }
            }
            res = tmp;
        }
        res
    }

    pub fn get_qty_combinations(&self) -> usize {
        // determine all hyperparameter combinations
        let mut total_combinations = 0;
        for valid_values in self.param_grid.values() {
            if total_combinations == 0 {
                total_combinations = valid_values.len();
            } else {
                total_combinations *= valid_values.len();
            }
        }
        total_combinations
    }
}

#[cfg(test)]
mod config_test {
    use super::*;

    #[test]
    fn should_get_expected_results() {
        let mut param_grid = HashMap::new();
        param_grid.insert("sample_size".to_string(), vec![1000]);
        param_grid.insert("k".to_string(), vec![500]);
        param_grid.insert("last_items_in_session".to_string(), vec![10]);

        let hyper_parametergrid = HyperParamGrid { param_grid };
        let random_parameter_values = hyper_parametergrid.get_all_combinations();
        assert_eq!(1, random_parameter_values.len());
        let hyperparameters = random_parameter_values[0].clone();
        assert_eq!(1000, hyperparameters.get("sample_size").unwrap().clone());
        assert_eq!(500, hyperparameters.get("k").unwrap().clone());
        assert_eq!(
            10,
            hyperparameters
                .get("last_items_in_session")
                .unwrap()
                .clone()
        );
    }

    #[test]
    fn should_determine_qty_combinations() {
        let mut param_grid = HashMap::new();
        param_grid.insert("sample_size".to_string(), vec![500, 750, 1000, 2500, 5000]);
        param_grid.insert("k".to_string(), vec![50, 100, 500, 1000, 1500]);
        param_grid.insert("last_items_in_session".to_string(), vec![1, 2, 3, 5, 10]);

        let hyper_parametergrid = HyperParamGrid { param_grid };
        assert_eq!(5 * 5 * 5, hyper_parametergrid.get_qty_combinations());
    }

    #[test]
    fn should_get_all_combinations() {
        let mut param_grid = HashMap::new();
        param_grid.insert("sample_size".to_string(), vec![500, 750, 1000, 2500, 5000]);
        param_grid.insert("k".to_string(), vec![50, 100, 500, 1000, 1500]);
        param_grid.insert("last_items_in_session".to_string(), vec![1, 2, 3, 5, 10]);

        let hyper_parametergrid = HyperParamGrid { param_grid };
        let combinations = hyper_parametergrid.get_all_combinations();
        assert_eq!(5 * 5 * 5, combinations.len());
        assert_eq!(3, combinations.get(0).unwrap().len());
    }

    #[test]
    fn should_get_n_random_combinations() {
        let mut param_grid = HashMap::new();
        param_grid.insert("sample_size".to_string(), vec![500, 750, 1000, 2500, 5000]);
        param_grid.insert("k".to_string(), vec![50, 100, 500, 1000, 1500]);
        param_grid.insert("last_items_in_session".to_string(), vec![1, 2, 3, 5, 10]);

        let hyper_parametergrid = HyperParamGrid { param_grid };
        let combinations = hyper_parametergrid.get_n_random_combinations(100000000);
        assert_eq!(5 * 5 * 5, combinations.len());

        let n_random_combinations = hyper_parametergrid.get_n_random_combinations(10);
        assert_eq!(10, n_random_combinations.len());
    }
}
