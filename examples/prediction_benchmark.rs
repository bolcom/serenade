//! Synthetic prediction/executor comparison. See PERFORMANCE.md.
use serenade::{
    executor::BlockingExecutor,
    vmisknn::{self, vmis_index::VMISIndex, PredictionScratch},
};
use std::{
    io::Write,
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};
thread_local! { static SCRATCH: std::cell::RefCell<PredictionScratch> = std::cell::RefCell::new(PredictionScratch::default()); }
fn main() -> anyhow::Result<()> {
    let seconds: u64 = std::env::var("BENCH_SECONDS")
        .unwrap_or("30".into())
        .parse()?;
    let warmup: u64 = std::env::var("BENCH_WARMUP")
        .unwrap_or("10".into())
        .parse()?;
    let repeats: usize = std::env::var("BENCH_REPEATS")
        .unwrap_or("5".into())
        .parse()?;
    // Zero means retain the complete held-out prefix, including its long tail.
    let history_limits: Vec<u64> = std::env::var("BENCH_HISTORY_LIMITS")
        .unwrap_or("1,20".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let concurrency: Vec<usize> = std::env::var("BENCH_CONCURRENCY")
        .unwrap_or("1,2,4,8,16,32,64".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(
        seconds > 0
            && repeats > 0
            && !concurrency.is_empty()
            && !history_limits.is_empty()
            && concurrency.iter().all(|n| *n > 0),
        "positive benchmark parameters required"
    );
    let external = std::env::var("BENCH_TRAINING_DATA").ok();
    let path = external
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("serenade-benchmark-{}.tsv", std::process::id()))
        });
    if external.is_none() {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        writeln!(file, "session\titem\ttime")?;
        for session in 0..20_000u64 {
            for offset in 0..20u64 {
                writeln!(
                    file,
                    "{}\t{}\t{}",
                    session,
                    (session * 17 + offset * 997) % 100_003,
                    session + offset
                )?;
            }
        }
        file.flush()?;
    }
    let queries: Option<Arc<Vec<Vec<u64>>>> = std::env::var("BENCH_QUERIES")
        .ok()
        .map(|path| -> anyhow::Result<_> {
            let text = std::fs::read_to_string(path)?;
            let queries = text
                .lines()
                .map(|line| {
                    line.split_whitespace()
                        .map(str::parse)
                        .collect::<Result<Vec<u64>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            anyhow::ensure!(
                !queries.is_empty() && queries.iter().all(|q| !q.is_empty()),
                "queries must contain nonempty histories"
            );
            Ok(Arc::new(queries))
        })
        .transpose()?;
    anyhow::ensure!(
        external.is_some() == queries.is_some(),
        "BENCH_TRAINING_DATA and BENCH_QUERIES must be provided together"
    );
    anyhow::ensure!(
        !history_limits.contains(&0) || queries.is_some(),
        "uncapped histories require BENCH_QUERIES"
    );
    let build = Instant::now();
    let index = Arc::new(VMISIndex::try_new_from_csv(
        path.to_str().unwrap(),
        500,
        1.0,
    )?);
    eprintln!("index_build_ms={}", build.elapsed().as_millis());
    if external.is_none() {
        std::fs::remove_file(path)?;
    }
    println!("mode,scratch,history,concurrency,repeat,successful_per_second,p50_us,p95_us,p99_us,rejected");
    for &clients in &concurrency {
        for blocking in [false, true] {
            for reuse in [false, true] {
                for &history in &history_limits {
                    let pool = if blocking {
                        Some(Arc::new(BlockingExecutor::new(4, 8)?))
                    } else {
                        None
                    };
                    for repeat in 0..repeats {
                        let barrier = Arc::new(Barrier::new(clients));
                        let mut handles = Vec::new();
                        for client in 0..clients {
                            let index = index.clone();
                            let pool = pool.clone();
                            let queries = queries.clone();
                            let barrier = barrier.clone();
                            handles.push(std::thread::spawn(move || {
                                let mut samples = Vec::new();
                                let mut rejected = 0;
                                barrier.wait();
                                let start = Instant::now();
                                let mut request = client as u64;
                                while start.elapsed() < Duration::from_secs(warmup + seconds) {
                                    let items: Vec<u64> = if let Some(queries) = &queries {
                                        let query = &queries[request as usize % queries.len()];
                                        query[if history == 0 {
                                            0
                                        } else {
                                            query.len().saturating_sub(history as usize)
                                        }..]
                                            .to_vec()
                                    } else {
                                        (0..if history == 0 { 20 } else { history })
                                            .map(|offset| {
                                                ((request % 20_000) * 17 + offset * 997) % 100_003
                                            })
                                            .collect()
                                    };
                                    let index = index.clone();
                                    let work = move || {
                                        if reuse {
                                            SCRATCH.with(|scratch| {
                                                vmisknn::predict_with_scratch(
                                                    index.as_ref(),
                                                    &items,
                                                    100,
                                                    500,
                                                    21,
                                                    false,
                                                    &mut scratch.borrow_mut(),
                                                )
                                            })
                                        } else {
                                            vmisknn::predict(
                                                index.as_ref(),
                                                &items,
                                                100,
                                                500,
                                                21,
                                                false,
                                            )
                                        }
                                    };
                                    let measured = start.elapsed() >= Duration::from_secs(warmup);
                                    let before = Instant::now();
                                    let success = match &pool {
                                        Some(pool) => match pool.submit(work) {
                                            Ok(result) => {
                                                futures::executor::block_on(result).is_ok()
                                            }
                                            Err(_) => false,
                                        },
                                        None => {
                                            std::hint::black_box(work());
                                            true
                                        }
                                    };
                                    if measured {
                                        if success {
                                            samples.push(before.elapsed().as_nanos() as u64);
                                        } else {
                                            rejected += 1;
                                        }
                                    }
                                    if !success {
                                        std::thread::yield_now();
                                    }
                                    request = request.wrapping_add(7919);
                                }
                                (samples, rejected)
                            }));
                        }
                        let mut samples = Vec::new();
                        let mut rejected = 0;
                        for handle in handles {
                            let (mut values, failures) = handle.join().unwrap();
                            samples.append(&mut values);
                            rejected += failures;
                        }
                        samples.sort_unstable();
                        let percentile = |p: usize| {
                            samples
                                .get((samples.len().saturating_sub(1) * p) / 100)
                                .copied()
                                .unwrap_or(0) as f64
                                / 1000.0
                        };
                        println!(
                            "{},{},{},{},{},{:.1},{:.2},{:.2},{:.2},{}",
                            if blocking { "blocking" } else { "inline" },
                            reuse,
                            history,
                            clients,
                            repeat,
                            samples.len() as f64 / seconds as f64,
                            percentile(50),
                            percentile(95),
                            percentile(99),
                            rejected
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
