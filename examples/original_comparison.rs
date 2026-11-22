use serenade::vmisknn::{self, vmis_index::VMISIndex};
use std::{
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};
fn main() -> anyhow::Result<()> {
    let path = std::env::var("BENCH_TRAINING_DATA")?;
    let queries: Arc<Vec<Vec<u64>>> = Arc::new(
        std::fs::read_to_string(std::env::var("BENCH_QUERIES")?)?
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect()
            })
            .collect(),
    );
    let build = Instant::now();
    let index = Arc::new(VMISIndex::new_from_csv(&path, 500, 1.0));
    eprintln!("index_build_ms={}", build.elapsed().as_millis());
    println!("history,concurrency,successful_per_second,p50_us,p95_us,p99_us");
    for clients in [1, 4, 16] {
        for history in [20usize, 0] {
            let barrier = Arc::new(Barrier::new(clients));
            let mut handles = Vec::new();
            for client in 0..clients {
                let (index, queries, barrier) = (index.clone(), queries.clone(), barrier.clone());
                handles.push(std::thread::spawn(move || {
                    let mut samples = Vec::new();
                    barrier.wait();
                    let start = Instant::now();
                    let mut request = client;
                    while start.elapsed() < Duration::from_secs(4) {
                        let query = &queries[request % queries.len()];
                        let items = query[if history == 0 {
                            0
                        } else {
                            query.len().saturating_sub(history)
                        }..]
                            .to_vec();
                        let measured = start.elapsed() >= Duration::from_secs(1);
                        let before = Instant::now();
                        std::hint::black_box(vmisknn::predict(
                            index.as_ref(),
                            &items,
                            100,
                            500,
                            21,
                            false,
                        ));
                        if measured {
                            samples.push(before.elapsed().as_nanos() as u64);
                        }
                        request = request.wrapping_add(7919);
                    }
                    samples
                }));
            }
            let mut samples = Vec::new();
            for handle in handles {
                samples.extend(handle.join().unwrap());
            }
            samples.sort_unstable();
            let percentile = |p: usize| samples[(samples.len() - 1) * p / 100] as f64 / 1000.0;
            println!(
                "{},{},{:.1},{:.2},{:.2},{:.2}",
                history,
                clients,
                samples.len() as f64 / 3.0,
                percentile(50),
                percentile(95),
                percentile(99)
            );
        }
    }
    Ok(())
}
