use serenade::vmisknn::{self, vmis_index::VMISIndex};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    io::Write,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
struct CountingAllocator;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, size)
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
fn main() -> anyhow::Result<()> {
    let index = VMISIndex::new_from_csv(&std::env::var("BENCH_TRAINING_DATA")?, 500, 1.0);
    let queries: Vec<Vec<u64>> = std::fs::read_to_string(std::env::var("BENCH_QUERIES")?)?
        .lines()
        .map(|line| line.split_whitespace().map(str::parse).collect())
        .collect::<Result<_, _>>()?;
    ENABLED.store(true, Ordering::Relaxed);
    for query in &queries {
        std::hint::black_box(vmisknn::predict(&index, query, 100, 500, 21, false));
    }
    ENABLED.store(false, Ordering::Relaxed);
    println!(
        "queries={},allocation_calls={},cumulative_requested_bytes={}",
        queries.len(),
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed)
    );
    if let Ok(path) = std::env::var("BENCH_SNAPSHOT") {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for (query_id, query) in queries.iter().enumerate() {
            // Include all scored items to avoid arbitrary ties at the recommendation cutoff.
            let mut scores = vmisknn::predict(&index, query, 100, 500, 18_009, false).into_vec();
            scores.sort_unstable_by_key(|item| item.id);
            for score in scores {
                writeln!(file, "{}\t{}\t{:.17e}", query_id, score.id, score.score)?;
            }
        }
    }
    Ok(())
}
