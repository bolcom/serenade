pub mod coverage;
pub mod evaluation_reporter;
pub mod f1score;
pub mod hitrate;
pub mod mrr;
pub mod ndcg;
pub mod popularity;
pub mod precision;
pub mod recall;

pub trait SessionMetric {
    fn add(&mut self, recommendations: &[u64], next_items: &[u64]);
    fn result(&self) -> f64;
    fn get_name(&self) -> String;
}
