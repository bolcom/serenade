//! Catalog statistics and mergeable beyond-accuracy evaluation.
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

pub const NAMES: [&str; 7] = [
    "MAP",
    "Gini",
    "ShannonEntropy",
    "LongTailShare",
    "IntraListDiversity",
    "EmptyResultRate",
    "FillRate",
];

#[derive(Default)]
pub struct Catalog {
    counts: HashMap<u64, usize>,
    tail: HashSet<u64>,
    sessions: HashMap<u64, Vec<usize>>,
}
impl Catalog {
    pub fn from_items(items: impl Iterator<Item = u64>) -> Self {
        let mut result = Self::default();
        for item in items {
            *result.counts.entry(item).or_default() += 1;
        }
        let mut ranked: Vec<_> = result.counts.iter().map(|(&i, &n)| (n, i)).collect();
        ranked.sort_unstable();
        result.tail = ranked
            .iter()
            .take((ranked.len() / 10).max(1))
            .map(|&(_, i)| i)
            .collect();
        result
    }
    pub fn from_sessions<'a>(sessions: impl Iterator<Item = &'a [u64]> + Clone) -> Self {
        let mut result = Self::from_items(sessions.clone().flat_map(|s| s.iter().copied()));
        for (id, items) in sessions.enumerate() {
            let unique: HashSet<_> = items.iter().copied().collect();
            for item in unique {
                result.sessions.entry(item).or_default().push(id);
            }
        }
        result
    }
    fn distance(&self, a: u64, b: u64) -> Option<f64> {
        let a = self.sessions.get(&a)?;
        let b = self.sessions.get(&b)?;
        let (mut i, mut j, mut common) = (0, 0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    common += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        Some(1. - common as f64 / (a.len() as f64 * b.len() as f64).sqrt())
    }
}

pub struct BeyondAccuracy {
    pub catalog: Arc<Catalog>,
    k: usize,
    cases: usize,
    sums: [f64; 7],
    exposure: HashMap<u64, usize>,
    selected: Option<usize>,
}
impl BeyondAccuracy {
    pub fn new(catalog: Arc<Catalog>, k: usize) -> Self {
        assert!(k > 0);
        Self {
            catalog,
            k,
            cases: 0,
            sums: [0.; 7],
            exposure: HashMap::new(),
            selected: None,
        }
    }
    /// Retain only the state and work needed for one HPO objective.
    pub fn selected(catalog: Arc<Catalog>, k: usize, metric: usize) -> Self {
        assert!(metric < NAMES.len());
        let mut result = Self::new(catalog, k);
        result.selected = Some(metric);
        result
    }
    fn needs(&self, metric: usize) -> bool {
        self.selected.is_none_or(|i| i == metric)
    }
    pub fn empty(&self, k: usize) -> Self {
        let mut result = Self::new(self.catalog.clone(), k);
        result.selected = self.selected;
        result
    }
    pub fn add(&mut self, recommendations: &[u64], targets: &[u64]) {
        let mut seen = HashSet::new();
        let items: Vec<_> = recommendations
            .iter()
            .take(self.k)
            .copied()
            .filter(|i| seen.insert(*i))
            .collect();
        if self.needs(0) {
            let relevant: HashSet<_> = targets.iter().copied().collect();
            let (mut hits, mut ap) = (0, 0.);
            // Original ranks are retained even for malformed duplicate predictions.
            let mut seen = HashSet::new();
            for (rank, item) in recommendations.iter().take(self.k).enumerate() {
                if seen.insert(*item) && relevant.contains(item) {
                    hits += 1;
                    ap += hits as f64 / (rank + 1) as f64;
                }
            }
            if !relevant.is_empty() {
                self.sums[0] += ap / relevant.len().min(self.k) as f64;
            }
        }
        if self.needs(1) || self.needs(2) {
            for item in &items {
                *self.exposure.entry(*item).or_default() += 1;
            }
        }
        if self.needs(3) && !items.is_empty() {
            self.sums[3] += items
                .iter()
                .filter(|i| self.catalog.tail.contains(i))
                .count() as f64
                / self.k as f64;
        }
        if self.needs(4) {
            let (mut distance, mut pairs) = (0., 0);
            for (i, &a) in items.iter().enumerate() {
                for &b in &items[i + 1..] {
                    if let Some(d) = self.catalog.distance(a, b) {
                        distance += d;
                        pairs += 1;
                    }
                }
            }
            if pairs > 0 {
                self.sums[4] += distance / pairs as f64;
            }
        }
        self.sums[5] += usize::from(items.is_empty()) as f64;
        self.sums[6] += items.len() as f64 / self.k as f64;
        self.cases += 1;
    }
    pub fn merge(&mut self, other: Self) {
        assert_eq!(self.k, other.k);
        assert_eq!(self.selected, other.selected);
        self.cases += other.cases;
        for (a, b) in self.sums.iter_mut().zip(other.sums) {
            *a += b;
        }
        for (item, count) in other.exposure {
            *self.exposure.entry(item).or_default() += count;
        }
    }
    pub fn scores(&self) -> [f64; 7] {
        let mut scores = self.sums.map(|s| {
            if self.cases == 0 {
                0.
            } else {
                s / self.cases as f64
            }
        });
        if !self.needs(1) && !self.needs(2) {
            return scores;
        }
        // Zero exposures need not be materialized: offset the exposed-item ranks.
        let num_items = self.catalog.counts.len()
            + self
                .exposure
                .keys()
                .filter(|i| !self.catalog.counts.contains_key(i))
                .count();
        let mut counts: Vec<_> = self.exposure.values().copied().collect();
        counts.sort_unstable();
        let total = counts.iter().sum::<usize>() as f64;
        if total > 0. {
            let n = num_items as f64;
            let zero_items = num_items - counts.len();
            if self.needs(1) {
                scores[1] = counts
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| (2. * (zero_items + i + 1) as f64 - n - 1.) * c as f64)
                    .sum::<f64>()
                    / (n * total);
            }
            if self.needs(2) {
                // Match RecBole's implementation (natural log, divided by exposed IDs).
                scores[2] = counts
                    .iter()
                    .filter(|&&c| c > 0)
                    .map(|&c| {
                        let p = c as f64 / total;
                        -p * p.ln()
                    })
                    .sum::<f64>()
                    / self.exposure.len() as f64;
            }
        }
        scores
    }
}
