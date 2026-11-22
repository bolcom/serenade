//! Reproducible single-thread benchmark. See docs/knn-benchmark.md.
use serenade::metrics::evaluation_reporter::EvaluationReporter;
use serenade::vmisknn::{algorithms::KnnConfig, vmis_index::VMISIndex};
use serenade::{io, vmisknn};
use std::time::Instant;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 4,
        "usage: knn_benchmark TRAIN VALID ALGORITHM [SNAPSHOT] [CONFIG]"
    );
    let model: KnnConfig = KnnConfig::for_algorithm(args[3].parse().expect("algorithm"));
    // Optional application config supplies shared retrieval/serving limits.
    // The requested algorithm retains its own reference scoring defaults.
    let config = args
        .get(5)
        .map(|path| serenade::config::AppConfig::new(path.clone()));
    let (k, m, history, business, legacy_idf) = config
        .as_ref()
        .map(|c| {
            (
                c.model.neighborhood_size_k,
                c.model.m_most_recent_sessions,
                c.model.max_items_in_session,
                c.logic.enable_business_logic,
                c.model.idf_weighting as f64,
            )
        })
        .unwrap_or((100, 500, 10, false, 1.0));
    let start = Instant::now();
    let index = VMISIndex::load_for_model(&args[1], m, legacy_idf, &model).unwrap();
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let heap = index.estimated_heap_bytes();
    let mut save_ms = 0.0;
    let mut load_ms = 0.0;
    let mut snapshot_bytes = 0;
    if let Some(path) = args.get(4) {
        let start = Instant::now();
        index.save(path).unwrap();
        save_ms = start.elapsed().as_secs_f64() * 1000.0;
        snapshot_bytes = std::fs::metadata(path).unwrap().len();
        let start = Instant::now();
        let restored = VMISIndex::load_for_model(path, m, legacy_idf, &model).unwrap();
        load_ms = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(restored);
    }
    let mut sessions: Vec<_> = io::read_test_data_timed(&args[2]).into_iter().collect();
    sessions.sort_by_key(|(id, _)| *id);
    let training = io::read_training_data(&args[1]);
    let mut reporter = EvaluationReporter::new(&training, 20);
    let mut timings = Vec::new();
    for pass in 0..3 {
        for (_, events) in &sessions {
            let items: Vec<_> = events.iter().map(|(item, _)| *item).collect();
            for state in 1..items.len() {
                let current = &items[state.saturating_sub(history)..state];
                let start = Instant::now();
                let recommendations = vmisknn::predict_with_model(
                    &index,
                    current,
                    k,
                    m,
                    20,
                    business,
                    &model,
                    events[state - 1].1,
                );
                timings.push(start.elapsed().as_nanos() as u64);
                let ranked: Vec<_> = recommendations
                    .into_sorted_vec()
                    .iter()
                    .map(|s| s.id)
                    .collect();
                if pass == 0 {
                    reporter.add(&ranked, &items[state..]);
                }
                std::hint::black_box(ranked);
            }
        }
    }
    assert!(
        !timings.is_empty(),
        "validation data has no prediction prefixes"
    );
    timings.sort_unstable();
    let percentile =
        |p: f64| timings[((timings.len() - 1) as f64 * p).round() as usize] as f64 / 1000.0;
    println!("BENCH_MODEL,{}", args[3]);
    println!("{}", reporter.get_name());
    println!("{}", reporter.result());
    println!("build_ms,save_ms,load_ms,owned_heap_estimate_bytes,snapshot_bytes,p50_us,p90_us,p99_us,predictions");
    println!(
        "{:.3},{:.3},{:.3},{},{},{:.3},{:.3},{:.3},{}",
        build_ms,
        save_ms,
        load_ms,
        heap,
        snapshot_bytes,
        percentile(0.50),
        percentile(0.90),
        percentile(0.99),
        timings.len()
    );
}
