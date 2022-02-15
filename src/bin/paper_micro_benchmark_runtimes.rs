#![allow(warnings)]

extern crate bencher;
extern crate itertools;
extern crate num_format;
extern crate rand_pcg;
extern crate rayon;
extern crate serenade_optimized;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::ops::Add;
use std::time::{Duration, Instant};

use bencher::black_box;
use inc_stats::DerefCopy;
use num_format::{Locale, ToFormattedString};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use tdigest::TDigest;

use serenade_optimized::io;
use serenade_optimized::vmisknn::similarity_hashed::SimilarityComputationHash;
use serenade_optimized::vmisknn::similarity_indexed::SimilarityComputationNew;
use serenade_optimized::vmisknn::tree_index::TreeIndex;
use serenade_optimized::vmisknn::vmisknn_index::VMISSkNNIndex;
use serenade_optimized::vmisknn::vmisknn_index_noopt::VMISSkNNIndexNoOpt;
use serenade_optimized::vmisknn::vsknn_index::VSkNNIndex;

#[derive(Debug)]
struct MicroBenchmarkLine<'a> {
    index: &'a str,
    dataset: &'a str,
    m: usize,
    k: usize,
    duration_ns: f64,
}

fn main() {
    // Benchmark with different Rust-based variants of our index
    // and similarity computation to validate the design choices of our index.
    let num_threads = 6;

    println!("index,dataset,m,k,duration_(ns)");

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let algorithms = vec!["vmis_noopt", "vmis", "vsknn"];

    for dataset in &["1m"] {
        for m in &[100, 250, 500, 1000] {
            let path_train = format!("data/private-clicks-{}_train.txt", dataset);
            let path_test = format!("data/private-clicks-{}_test.txt", dataset);

            let historical_sessions = io::read_training_data(&path_train);

            let vmis_noopt_index = VMISSkNNIndexNoOpt::new(&path_train, *m);
            let vmis_index = VMISSkNNIndex::new(&path_train, *m);
            let vsknn_index = VSkNNIndex::new(historical_sessions, *m, 1_000_000);
            // let tree_index = TreeIndex::new(&path_train, *m);

            let mut test_sessions: HashMap<u32, Vec<u64>> = HashMap::new();

            for (session, item, _) in io::read_training_data(&path_test).iter() {
                if !test_sessions.contains_key(session) {
                    test_sessions.insert(*session, Vec::new());
                }

                test_sessions.get_mut(session).unwrap().push(*item);
            }

            let num_repetitions = 10;

            for i in 0..num_repetitions {
                let mut rng_k = Pcg64::seed_from_u64(i);
                let mut topk = [20, 50, 100];
                topk.shuffle(&mut rng_k);

                for k in &topk {
                    let mut rng_random = thread_rng();
                    black_box::<Vec<u64>>(
                        (0..10_000_000)
                            .map(|i| rng_random.gen_range(0..10_000_000))
                            .collect(),
                    );

                    let duration_array = test_sessions
                        .par_iter()
                        .map(|(key, items)| {
                            let mut rng_length = Pcg64::seed_from_u64(*key as u64);
                            let length = rng_length.gen_range(1..items.len());
                            let session = &items[0..length];
                            let mut duration_sample: Vec<MicroBenchmarkLine> =
                                Vec::with_capacity(algorithms.len());

                            let start = Instant::now();
                            black_box(vmis_noopt_index.find_neighbors(&session, *k, *m));
                            let elapsed = start.elapsed();
                            let result = MicroBenchmarkLine {
                                index: algorithms.get(0).unwrap().deref_copy(),
                                dataset: dataset,
                                m: *m,
                                k: *k,
                                duration_ns: elapsed.as_nanos() as f64,
                            };
                            duration_sample.push(result);

                            let start = Instant::now();
                            black_box(vmis_index.find_neighbors(&session, *k, *m));
                            let elapsed = start.elapsed();
                            let result = MicroBenchmarkLine {
                                index: algorithms.get(1).unwrap().deref_copy(),
                                dataset: dataset,
                                m: *m,
                                k: *k,
                                duration_ns: elapsed.as_nanos() as f64,
                            };
                            duration_sample.push(result);

                            let start = Instant::now();
                            black_box(vsknn_index.find_neighbors(&session, *k, *m));
                            let elapsed = start.elapsed();
                            let result = MicroBenchmarkLine {
                                index: algorithms.get(2).unwrap().deref_copy(),
                                dataset: dataset,
                                m: *m,
                                k: *k,
                                duration_ns: elapsed.as_nanos() as f64,
                            };
                            duration_sample.push(result);

                            // let start = Instant::now();
                            // black_box(
                            //     tree_index.find_neighbors(&session, *k, *m)
                            // );
                            // duration_sample[3] = start.elapsed();

                            duration_sample
                        })
                        .collect::<Vec<_>>();

                    for lines in duration_array.iter() {
                        for result in lines.iter() {
                            println!(
                                "{},{},{},{},{}",
                                result.index,
                                result.dataset,
                                result.m,
                                result.k,
                                result.duration_ns
                            );
                        }
                    }
                }
            }
        }
    }
}
