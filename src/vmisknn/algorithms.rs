//! Reference scoring over candidates retrieved by the VMIS index.
use hashbrown::{HashMap, HashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug)]
struct RankedNeighbor {
    neighbor: SessionScore,
    order: usize,
}
impl PartialEq for RankedNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for RankedNeighbor {}
impl PartialOrd for RankedNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .neighbor
            .score
            .total_cmp(&self.neighbor.score)
            .then(self.order.cmp(&other.order))
    }
}
use super::{similarity_indexed::SimilarityComputationNew, SessionScore};

/// Scoring algorithm; all variants use the same bounded VMIS candidate index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Vmis,
    Sknn,
    Ssknn,
    Wsknn,
    Vsknn,
    Stan,
    VsknnStan,
}
impl std::str::FromStr for Algorithm {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "vmis" | "vmisknn" => Ok(Self::Vmis),
            "sknn" => Ok(Self::Sknn),
            "ssknn" | "s-sknn" => Ok(Self::Ssknn),
            "wsknn" => Ok(Self::Wsknn),
            "vsknn" | "vs-knn" => Ok(Self::Vsknn),
            "stan" => Ok(Self::Stan),
            "vsknn_stan" | "vstan" => Ok(Self::VsknnStan),
            _ => Err(format!("unsupported kNN algorithm: {}", s)),
        }
    }
}
impl Algorithm {
    pub const ALL: [Self; 7] = [
        Self::Vmis,
        Self::Sknn,
        Self::Ssknn,
        Self::Wsknn,
        Self::Vsknn,
        Self::Stan,
        Self::VsknnStan,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Vmis => "vmis",
            Self::Sknn => "sknn",
            Self::Ssknn => "ssknn",
            Self::Wsknn => "wsknn",
            Self::Vsknn => "vsknn",
            Self::Stan => "stan",
            Self::VsknnStan => "vsknn_stan",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
/// Session similarity measure used by the selected algorithm.
pub enum Similarity {
    Jaccard,
    Cosine,
    Tanimoto,
    Binary,
    Vec,
}
impl std::str::FromStr for Similarity {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "jaccard" => Ok(Self::Jaccard),
            "cosine" => Ok(Self::Cosine),
            "tanimoto" => Ok(Self::Tanimoto),
            "binary" => Ok(Self::Binary),
            "vec" => Ok(Self::Vec),
            _ => Err(format!("unsupported similarity: {}", s)),
        }
    }
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
/// Position or matched-distance weighting, using the reference algorithm formulas.
pub enum Weighting {
    Same,
    Linear,
    Div,
    Log,
    Quadratic,
}
impl std::str::FromStr for Weighting {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "same" | "none" => Ok(Self::Same),
            "linear" => Ok(Self::Linear),
            "div" | "inv" => Ok(Self::Div),
            "log" => Ok(Self::Log),
            "quadratic" => Ok(Self::Quadratic),
            _ => Err(format!("unsupported weighting: {}", s)),
        }
    }
}
impl Weighting {
    fn position(self, i: usize, length: usize) -> f64 {
        let (i, length) = (i as f64, length as f64);
        match self {
            Self::Same => 1.0,
            Self::Div => i / length,
            Self::Linear => {
                if i <= 10.0 {
                    1.0 - 0.1 * (length - i)
                } else {
                    0.0
                }
            }
            Self::Log => 1.0 / (length - i + 1.7).log10(),
            Self::Quadratic => (i / length).powi(2),
        }
    }
    fn matched(self, step: usize) -> f64 {
        let i = step as f64;
        match self {
            Self::Same => 1.0,
            Self::Div => 1.0 / i,
            Self::Linear => {
                if i <= 100.0 {
                    1.0 - 0.1 * i
                } else {
                    0.0
                }
            }
            Self::Log => 1.0 / (i + 1.7).log10(),
            Self::Quadratic => 1.0 / i.powi(2),
        }
    }
}
#[derive(Clone, Debug, serde::Serialize)]
/// Algorithm-specific scoring settings shared with portable TOML models.
/// Start with [`Self::for_algorithm`] to get the same defaults as the applications.
/// Irrelevant settings are ignored by algorithms that do not use them.
pub struct KnnConfig {
    /// Scoring algorithm.
    pub algorithm: Algorithm,
    /// Session similarity measure.
    pub similarity: Similarity,
    /// Current-session position weighting.
    pub weighting: Weighting,
    /// Matched-distance item score weighting.
    pub weighting_score: Weighting,
    /// Apply reference recency weighting to neighbors.
    pub weighting_time: bool,
    /// Reference item IDF multiplier (legacy VMIS uses `ModelConfig::idf_weighting`).
    pub idf_weighting: f64,
    /// Apply reference IDF to current-session similarity weights.
    pub idf_weighting_session: bool,
    /// Allow recommendations of items already in the input session.
    pub remind: bool,
    /// Normalize item scores using the selected algorithm's reference semantics.
    pub normalize: bool,
    /// Popularity boost for supported reference algorithms.
    pub pop_boost: f64,
    /// STAN session-position decay; positive when enabled, `None` disables it.
    pub lambda_spw: Option<f64>,
    /// STAN neighbor-age decay in days; positive when enabled, `None` disables it.
    pub lambda_snh: Option<f64>,
    /// STAN matched-distance decay; positive when enabled.
    pub lambda_inh: Option<f64>,
    /// VSKNN-STAN item-position decay; positive when enabled.
    pub lambda_ipw: Option<f64>,
    /// VSKNN-STAN IDF multiplier; finite when enabled.
    pub lambda_idf: Option<f64>,
}
impl Default for KnnConfig {
    fn default() -> Self {
        Self::for_algorithm(Algorithm::Vmis)
    }
}
impl KnnConfig {
    pub fn for_algorithm(algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            similarity: match algorithm {
                Algorithm::Sknn | Algorithm::Ssknn => Similarity::Jaccard,
                Algorithm::Vsknn | Algorithm::Wsknn => Similarity::Vec,
                _ => Similarity::Cosine,
            },
            weighting: if algorithm == Algorithm::Wsknn {
                Weighting::Linear
            } else {
                Weighting::Div
            },
            weighting_score: if algorithm == Algorithm::Wsknn {
                Weighting::Linear
            } else {
                Weighting::Div
            },
            weighting_time: false,
            idf_weighting: 0.0,
            idf_weighting_session: false,
            remind: !matches!(algorithm, Algorithm::Sknn | Algorithm::Ssknn),
            normalize: matches!(
                algorithm,
                Algorithm::Sknn | Algorithm::Ssknn | Algorithm::Vsknn
            ),
            pop_boost: 0.0,
            lambda_spw: Some(1.02),
            lambda_snh: Some(5.0),
            lambda_inh: Some(2.05),
            lambda_ipw: Some(1.02),
            lambda_idf: Some(5.0),
        }
    }
    pub fn positional(&self) -> bool {
        matches!(
            self.algorithm,
            Algorithm::Stan | Algorithm::VsknnStan | Algorithm::Wsknn
        )
    }
    fn stan(&self) -> bool {
        matches!(self.algorithm, Algorithm::Stan | Algorithm::VsknnStan)
    }
    pub fn needs_reference_idf(&self) -> bool {
        (self.algorithm == Algorithm::Vsknn
            && (self.idf_weighting != 0.0 || self.idf_weighting_session))
            || (self.algorithm == Algorithm::VsknnStan && self.lambda_idf.is_some())
    }
    pub fn validate(&self) -> Result<(), String> {
        if matches!(self.algorithm, Algorithm::Sknn | Algorithm::Ssknn)
            && self.similarity == Similarity::Vec
        {
            return Err("SKNN does not support vec similarity".into());
        }
        if self.algorithm == Algorithm::Vsknn
            && !matches!(self.similarity, Similarity::Vec | Similarity::Cosine)
        {
            return Err("VS-KNN supports vec and weighted cosine similarity".into());
        }
        if self.algorithm == Algorithm::Wsknn
            && (!matches!(
                self.weighting,
                Weighting::Linear | Weighting::Log | Weighting::Quadratic
            ) || !matches!(
                self.weighting_score,
                Weighting::Linear | Weighting::Log | Weighting::Quadratic | Weighting::Div
            ) || self.similarity != Similarity::Vec)
        {
            return Err("WSKNN requires vec similarity, linear/log/quadratic weighting and linear/div/log/quadratic scoring".into());
        }
        for v in [
            self.lambda_spw,
            self.lambda_snh,
            self.lambda_inh,
            self.lambda_ipw,
        ]
        .iter()
        .flatten()
        {
            if !v.is_finite() || *v <= 0.0 {
                return Err("decay lambdas must be positive or none".into());
            }
        }
        if !self.idf_weighting.is_finite()
            || !self.pop_boost.is_finite()
            || self.lambda_idf.map_or(false, |v| !v.is_finite())
        {
            return Err("weights must be finite".into());
        }
        Ok(())
    }
}

/// Build once per request; duplicate current items overwrite earlier positions.
fn current_weights(
    current: &[u64],
    config: &KnnConfig,
    index: &(impl SimilarityComputationNew + ?Sized),
) -> HashMap<u64, f64> {
    current
        .iter()
        .enumerate()
        .map(|(i, &item)| {
            let weight = if config.algorithm == Algorithm::Wsknn {
                let count = (i + 1) as f64;
                let length = current.len() as f64;
                match config.weighting {
                    Weighting::Linear => (count + 1.0) / length,
                    Weighting::Log => 2.7_f64.log10() / (length - count + 1.7).log10(),
                    Weighting::Quadratic => (count / length).powi(2),
                    _ => unreachable!("validated WSKNN weighting"),
                }
            } else if config.stan() {
                1.0
            } else {
                config.weighting.position(i + 1, current.len())
            };
            let weight = if config.stan() {
                config
                    .lambda_spw
                    .map_or(1.0, |l| (((i + 1) as f64 - current.len() as f64) / l).exp())
            } else if config.idf_weighting_session {
                index.idf(&item)
            } else {
                weight
            };
            (item, weight)
        })
        .collect()
}

/// Score identical candidates independently of VMIS pruning. Ties retain candidate order.
pub fn score_neighbors(
    index: &(impl SimilarityComputationNew + ?Sized),
    current: &[u64],
    candidates: &[u32],
    timestamp: f64,
    k: usize,
    config: &KnnConfig,
) -> Vec<SessionScore> {
    if current.is_empty() || k == 0 {
        return Vec::new();
    }
    let weights = current_weights(current, config, index);
    let mut ordered_weights: Vec<_> = weights.iter().collect();
    ordered_weights.sort_unstable_by_key(|(item, _)| **item);
    let norm = ordered_weights
        .iter()
        .map(|(_, v)| *v * *v)
        .sum::<f64>()
        .sqrt();
    let mut top = BinaryHeap::<RankedNeighbor>::with_capacity(k);
    for (order, &id) in candidates.iter().enumerate() {
        let mut items = index.items_for_session(&id).to_vec();
        items.sort_unstable();
        items.dedup();
        let intersection = items.iter().filter(|i| weights.contains_key(*i)).count() as f64;
        let (a, b) = (items.len() as f64, weights.len() as f64);
        if a == 0.0 {
            continue;
        }
        let mut score = if matches!(config.algorithm, Algorithm::Sknn | Algorithm::Ssknn) {
            match config.similarity {
                Similarity::Jaccard | Similarity::Tanimoto => intersection / (a + b - intersection),
                // This order of operations intentionally follows sknn.py.
                Similarity::Cosine => intersection / a.sqrt() * b.sqrt(),
                Similarity::Binary => 2.0 * intersection / (2.0 * intersection + a + b),
                Similarity::Vec => unreachable!(),
            }
        } else {
            let dot = items.iter().filter_map(|i| weights.get(i)).sum::<f64>();
            if matches!(config.algorithm, Algorithm::Vsknn | Algorithm::Wsknn)
                && config.similarity == Similarity::Vec
            {
                dot / b
            } else if norm > 0.0 {
                dot / (norm * a.sqrt())
            } else {
                0.0
            }
        };
        if !config.stan() && score <= 0.0 {
            continue;
        }
        let age = timestamp - index.session_timestamp(id) as f64;
        if config.stan() {
            if let Some(days) = config.lambda_snh {
                score *= (-age / (days * 86400.0)).exp();
            }
        } else if config.algorithm == Algorithm::Vsknn && config.weighting_time {
            score *= (7.0_f64 / 8.0).powf((age / 86400.0).round_ties_even());
        }
        let neighbor = RankedNeighbor {
            neighbor: SessionScore::new(id, score),
            order,
        };
        if top.len() < k {
            top.push(neighbor);
        } else if score > top.peek().unwrap().neighbor.score {
            *top.peek_mut().unwrap() = neighbor;
        }
    }
    let mut selected = top.into_vec();
    selected.sort_by(|a, b| {
        b.neighbor
            .score
            .total_cmp(&a.neighbor.score)
            .then(a.order.cmp(&b.order))
    });
    selected.into_iter().map(|n| n.neighbor).collect()
}

/// Raw reference item scores, before shared serving filters and normalization.
pub fn score_items(
    index: &(impl SimilarityComputationNew + ?Sized),
    current: &[u64],
    neighbors: &[SessionScore],
    config: &KnnConfig,
) -> HashMap<u64, f64> {
    let mut scores = HashMap::with_capacity(1000);
    score_items_into(index, current, neighbors, config, &mut scores);
    scores
}

/// Reuse caller-owned scores without retaining predictions between interactions.
pub(crate) fn score_items_into(
    index: &(impl SimilarityComputationNew + ?Sized),
    current: &[u64],
    neighbors: &[SessionScore],
    config: &KnnConfig,
    scores: &mut HashMap<u64, f64>,
) {
    scores.clear();
    let current_set: HashSet<_> = current.iter().copied().collect();
    for neighbor in neighbors {
        let sequence = index.items_for_session(&neighbor.id);
        if config.algorithm == Algorithm::Wsknn {
            let Some(latest) = current.iter().rposition(|item| sequence.contains(item)) else {
                continue;
            };
            let step = current.len() - latest;
            let decay = match config.weighting_score {
                Weighting::Linear => {
                    if step < 10 {
                        (1.0 - 0.1 * step as f64) / 0.9
                    } else {
                        0.0
                    }
                }
                Weighting::Log => 2.7_f64.log10() / (step as f64 + 1.7).log10(),
                other => other.matched(step),
            };
            for &item in sequence {
                if config.remind || !current_set.contains(&item) {
                    *scores.entry(item).or_insert(0.0) += neighbor.score * decay;
                }
            }
            continue;
        }
        let mut positions = HashMap::with_capacity(sequence.len());
        let mut anchor = 0;
        for (i, &item) in sequence.iter().enumerate() {
            positions.insert(item, i + 1);
            if current_set.contains(&item) {
                anchor = i + 1;
            }
        }
        let latest_match = current.iter().rposition(|i| positions.contains_key(i));
        let Some(latest_match) = latest_match else {
            continue;
        };
        let decay = match config.algorithm {
            Algorithm::Ssknn => config.weighting.matched(current.len() - latest_match),
            Algorithm::Vsknn => config.weighting_score.matched(current.len() - latest_match),
            Algorithm::VsknnStan => 1.0,
            _ => 1.0,
        };
        let decay = if config.algorithm == Algorithm::VsknnStan {
            config.lambda_ipw.map_or(1.0, |l| {
                (((latest_match + 1) as f64 - current.len() as f64) / l).exp()
            })
        } else {
            decay
        };
        for (item, position) in positions {
            // SKNN's remind flag adds reminders after scoring; it does not exclude current items.
            if !matches!(config.algorithm, Algorithm::Sknn | Algorithm::Ssknn)
                && !config.remind
                && current_set.contains(&item)
            {
                continue;
            }
            let mut score = neighbor.score;
            if config.stan() {
                if let Some(l) = config.lambda_inh {
                    score *= (-(position.abs_diff(anchor) as f64) / l).exp();
                }
            }
            let idf_weight = match config.algorithm {
                Algorithm::Vsknn => config.idf_weighting,
                Algorithm::VsknnStan => config.lambda_idf.unwrap_or(0.0),
                _ => 0.0,
            };
            if idf_weight != 0.0 {
                score += score * index.idf(&item) * idf_weight;
            }
            *scores.entry(item).or_insert(0.0) += score * decay;
        }
    }
    if matches!(config.algorithm, Algorithm::Sknn | Algorithm::Ssknn) {
        if config.remind {
            for (i, item) in current
                .iter()
                .skip(current.len().saturating_sub(3))
                .enumerate()
            {
                let old = scores.get(item).copied().unwrap_or(0.0);
                scores.insert(*item, (old + 5.0) * 5.0 + (i + 1) as f64 / 100.0);
            }
        }
        if config.pop_boost > 0.0 {
            let mut counts = HashMap::<u64, usize>::new();
            for n in neighbors {
                for item in index
                    .items_for_session(&n.id)
                    .iter()
                    .collect::<HashSet<_>>()
                {
                    *counts.entry(*item).or_default() += 1;
                }
            }
            let max = counts.values().copied().max().unwrap_or(1) as f64;
            for (item, score) in scores.iter_mut() {
                *score += config.pop_boost * counts.get(item).copied().unwrap_or(0) as f64 / max;
            }
        }
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use crate::vmisknn::vmis_index::VMISIndex;
    fn close(actual: f64, expected: f64, context: &str) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{}: actual {}, expected {}",
            context,
            actual,
            expected
        );
    }
    fn pairs(value: &str) -> Vec<(u64, f64)> {
        value
            .split(',')
            .map(|pair| {
                let (id, score) = pair.split_once(':').unwrap();
                (id.parse().unwrap(), score.parse().unwrap())
            })
            .collect()
    }
    #[test]
    fn scoring_matches_upstream_python_on_identical_candidates() {
        let sequences = vec![
            vec![1, 2, 1, 4],
            vec![2, 3, 2, 5],
            vec![1, 3, 6],
            vec![1, 2, 1, 4],
        ];
        let path =
            std::env::temp_dir().join(format!("serenade-reference-{}.tsv", std::process::id()));
        let mut data = String::from("SessionId\tItemId\tTime\n");
        for (session, items) in sequences.iter().enumerate() {
            for (pos, item) in items.iter().enumerate() {
                data.push_str(&format!("{}\t{}\t{}\n", session, item, pos + 1));
            }
        }
        std::fs::write(&path, data).unwrap();
        let mut index = VMISIndex::new_from_csv(path.to_str().unwrap(), 100, 1.0);
        std::fs::remove_file(path).unwrap();
        index.session_to_max_time_stamp = vec![100, 86500, 172900, 100];
        assert_eq!(index.session_to_items_sorted, sequences);
        let mut frequencies = HashMap::<u64, usize>::new();
        for item in sequences.iter().flatten() {
            *frequencies.entry(*item).or_default() += 1;
        }
        index.item_to_idf_score = frequencies
            .into_iter()
            .map(|(item, count)| (item, (4.0 / count as f64).ln()))
            .collect();
        for line in concat!(
            include_str!("../../tests/reference/scores.tsv"),
            include_str!("../../tests/reference/extensions.tsv")
        )
        .lines()
        {
            let f: Vec<_> = line.split('\t').collect();
            let mut cfg = KnnConfig::for_algorithm(f[0].parse().unwrap());
            cfg.similarity = f[1].parse().unwrap();
            cfg.weighting = f[2].parse().unwrap();
            cfg.weighting_score = f[3].parse().unwrap();
            cfg.weighting_time = f[4].parse().unwrap();
            cfg.idf_weighting = f[5].parse().unwrap();
            cfg.idf_weighting_session = f[6].parse().unwrap();
            cfg.remind = f[7].parse().unwrap();
            cfg.pop_boost = f[17].parse().unwrap();
            let optional = |s: &str| {
                if s == "none" {
                    None
                } else {
                    Some(s.parse::<f64>().unwrap())
                }
            };
            cfg.lambda_spw = optional(f[8]);
            cfg.lambda_snh = optional(f[9]);
            cfg.lambda_inh = optional(f[10]);
            cfg.lambda_ipw = optional(f[11]);
            cfg.lambda_idf = optional(f[12]);
            cfg.validate().unwrap();
            let current: Vec<u64> = f[14].split(',').map(|s| s.parse().unwrap()).collect();
            let neighbors = score_neighbors(
                &index,
                &current,
                &[0, 1, 2, 3],
                216100.0,
                f[13].parse().unwrap(),
                &cfg,
            );
            let expected_neighbors = pairs(f[15]);
            assert_eq!(neighbors.len(), expected_neighbors.len(), "{}", line);
            for (actual, (id, score)) in neighbors.iter().zip(expected_neighbors) {
                assert_eq!(actual.id as u64, id, "{}", line);
                close(actual.score, score, line);
            }
            let scores = score_items(&index, &current, &neighbors, &cfg);
            let expected_scores = pairs(f[16]);
            assert_eq!(scores.len(), expected_scores.len(), "{}", line);
            for (item, expected) in expected_scores {
                close(scores[&item], expected, line);
            }
            let mut ranked_scores = scores.clone();
            if cfg.normalize {
                let max = ranked_scores.values().copied().fold(0.0_f64, f64::max);
                if max > 0.0 {
                    for value in ranked_scores.values_mut() {
                        *value /= max;
                    }
                }
            }
            let ranked = super::super::rank_items(&index, &current, &mut ranked_scores, 20, false)
                .into_sorted_vec();
            let expected: Vec<u64> = f[18].split(',').map(|s| s.parse().unwrap()).collect();
            assert_eq!(ranked.len(), expected.len());
            // Python and the serving heap use different deterministic item tie orders.
            // Every rank must have the reference score; either tied item is valid.
            for (actual, expected_id) in ranked.iter().zip(&expected) {
                close(scores[&actual.id], scores[expected_id], line);
            }
            let mut actual_ids: Vec<_> = ranked.iter().map(|s| s.id).collect();
            actual_ids.sort_unstable();
            let mut expected_ids = expected;
            expected_ids.sort_unstable();
            assert_eq!(actual_ids, expected_ids);
        }
    }
    #[test]
    fn selected_models_share_retrieval_top_n_and_business_filters() {
        use super::super::vmis_index::ProductAttributes;
        use super::super::{predict, predict_with_model};
        let path =
            std::env::temp_dir().join(format!("serenade-serving-{}.tsv", std::process::id()));
        std::fs::write(
            &path,
            "SessionId\tItemId\tTime\n1\t1\t1\n1\t2\t2\n1\t4\t3\n2\t1\t4\n2\t5\t5\n2\t6\t6\n",
        )
        .unwrap();
        let mut index = VMISIndex::new_from_csv(path.to_str().unwrap(), 100, 1.0);
        std::fs::remove_file(path).unwrap();
        for item in [1, 2, 4, 5, 6] {
            index.item_to_product_attributes.insert(
                item,
                ProductAttributes {
                    is_adult: item == 5,
                    is_for_sale: item != 6,
                },
            );
        }
        for algorithm in Algorithm::ALL {
            let model = KnnConfig::for_algorithm(algorithm);
            let full = predict_with_model(&index, &[1], 100, 100, 20, false, &model, 10.0)
                .into_sorted_vec();
            assert_eq!(full.len(), 4);
            assert!(full.iter().all(|i| i.id != 1 && i.score.is_finite()));
            let limited = predict_with_model(&index, &[1], 100, 100, 1, false, &model, 10.0)
                .into_sorted_vec();
            assert_eq!(limited.len(), 1);
            close(limited[0].score, full[0].score, "top-N");
            let filtered = predict_with_model(&index, &[1], 100, 100, 20, true, &model, 10.0)
                .into_sorted_vec();
            assert_eq!(filtered.len(), 2);
            assert!(filtered.iter().all(|i| [2, 4].contains(&i.id)));
            if algorithm == Algorithm::Vmis {
                let legacy = predict(&index, &[1], 100, 100, 20, false).into_sorted_vec();
                for (a, b) in full.iter().zip(legacy) {
                    close(a.score, b.score, "legacy dispatch");
                }
            }
        }
    }
}
