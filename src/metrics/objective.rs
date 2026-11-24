//! The shared HPO criterion, parsed once when loading configuration.
use std::{fmt, num::NonZeroUsize, str::FromStr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankingMetric {
    Mrr,
    Ndcg,
    HitRate,
    Map,
    Gini,
    ShannonEntropy,
    LongTailShare,
    IntraListDiversity,
    EmptyResultRate,
    FillRate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Objective {
    metric: RankingMetric,
    cutoff: NonZeroUsize,
}

impl Objective {
    pub fn minimize(self) -> bool {
        matches!(
            self.metric,
            RankingMetric::Gini | RankingMetric::EmptyResultRate
        )
    }
    pub fn loss(self, score: f64) -> f64 {
        if self.minimize() {
            score
        } else {
            -score
        }
    }
    pub fn name(self) -> &'static str {
        match self.metric {
            RankingMetric::Mrr => "mrr",
            RankingMetric::Ndcg => "ndcg",
            RankingMetric::HitRate => "hitrate",
            RankingMetric::Map => "map",
            RankingMetric::Gini => "gini",
            RankingMetric::ShannonEntropy => "shannonentropy",
            RankingMetric::LongTailShare => "longtailshare",
            RankingMetric::IntraListDiversity => "intralistdiversity",
            RankingMetric::EmptyResultRate => "emptyresultrate",
            RankingMetric::FillRate => "fillrate",
        }
    }
    pub fn metric(self) -> RankingMetric {
        self.metric
    }
    pub fn cutoff(self) -> usize {
        self.cutoff.get()
    }
}

impl Default for Objective {
    fn default() -> Self {
        Self {
            metric: RankingMetric::Mrr,
            cutoff: NonZeroUsize::new(20).unwrap(),
        }
    }
}

impl FromStr for Objective {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let invalid = || {
            anyhow::anyhow!("hyperparam.objective must be metric@k (a supported metric with a positive integer cutoff), got {:?}", value)
        };
        let (metric, cutoff) = value.split_once('@').ok_or_else(invalid)?;
        let metric = match metric.to_ascii_lowercase().as_str() {
            "mrr" => RankingMetric::Mrr,
            "ndcg" => RankingMetric::Ndcg,
            "hitrate" => RankingMetric::HitRate,
            "map" => RankingMetric::Map,
            "gini" => RankingMetric::Gini,
            "shannonentropy" => RankingMetric::ShannonEntropy,
            "longtailshare" => RankingMetric::LongTailShare,
            "intralistdiversity" => RankingMetric::IntraListDiversity,
            "emptyresultrate" => RankingMetric::EmptyResultRate,
            "fillrate" => RankingMetric::FillRate,
            _ => anyhow::bail!("unsupported hyperparam.objective metric {:?}; supported metrics: mrr, ndcg, hitrate, map, gini, shannonentropy, longtailshare, intralistdiversity, emptyresultrate, fillrate", metric),
        };
        anyhow::ensure!(
            !cutoff.is_empty() && cutoff.bytes().all(|b| b.is_ascii_digit()),
            "{}",
            invalid()
        );
        let cutoff = cutoff
            .parse::<usize>()
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(invalid)?;
        Ok(Self { metric, cutoff })
    }
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.metric {
            RankingMetric::Mrr => "MRR",
            RankingMetric::Ndcg => "NDCG",
            RankingMetric::HitRate => "HitRate",
            RankingMetric::Map => "MAP",
            RankingMetric::Gini => "Gini",
            RankingMetric::ShannonEntropy => "ShannonEntropy",
            RankingMetric::LongTailShare => "LongTailShare",
            RankingMetric::IntraListDiversity => "IntraListDiversity",
            RankingMetric::EmptyResultRate => "EmptyResultRate",
            RankingMetric::FillRate => "FillRate",
        };
        write!(f, "{}@{}", name, self.cutoff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_supported_objectives() {
        for (text, metric, k) in [
            ("mrr@20", RankingMetric::Mrr, 20),
            ("ndcg@8", RankingMetric::Ndcg, 8),
            ("hitrate@10", RankingMetric::HitRate, 10),
            ("NdCg@8", RankingMetric::Ndcg, 8),
            ("MRR@5", RankingMetric::Mrr, 5),
            ("MAP@4", RankingMetric::Map, 4),
            ("Gini@2", RankingMetric::Gini, 2),
            ("ShannonEntropy@2", RankingMetric::ShannonEntropy, 2),
            ("LongTailShare@2", RankingMetric::LongTailShare, 2),
            ("IntraListDiversity@2", RankingMetric::IntraListDiversity, 2),
            ("EmptyResultRate@2", RankingMetric::EmptyResultRate, 2),
            ("FillRate@2", RankingMetric::FillRate, 2),
        ] {
            let objective: Objective = text.parse().unwrap();
            assert_eq!(objective.metric(), metric);
            assert_eq!(objective.cutoff(), k);
            assert_eq!(
                objective.minimize(),
                matches!(metric, RankingMetric::Gini | RankingMetric::EmptyResultRate)
            );
            assert_eq!(
                objective.loss(0.25),
                if objective.minimize() { 0.25 } else { -0.25 }
            );
            assert_eq!(
                objective.to_string().parse::<Objective>().unwrap(),
                objective
            );
        }
        assert_eq!(Objective::default(), "mrr@20".parse().unwrap());
    }
    #[test]
    fn rejects_invalid_objectives_clearly() {
        for text in [
            "recall@10",
            "",
            "mrr",
            "mrr@",
            "mrr@0",
            "mrr@-1",
            "mrr@+1",
            "mrr@1.5",
            "mrr@8@2",
            "mrr@@8",
            "mrr@ 8",
            "mrr@999999999999999999999999999999999",
        ] {
            let error = text.parse::<Objective>().unwrap_err().to_string();
            assert!(error.contains("hyperparam.objective"), "{text}: {error}");
            assert!(
                error.contains("supported metrics") || error.contains("positive integer cutoff"),
                "{error}"
            );
        }
    }
}
