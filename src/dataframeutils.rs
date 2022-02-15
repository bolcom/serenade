//#[macro_use] extern crate serde_derive;
// use itertools::Itertools;
use chrono::NaiveDateTime;
// use tdigest::TDigest;
use rayon::prelude::*;
use std::sync::Arc;

use crate::sessions::RocksDBSessionStore;
use crate::vmisknn::offline_index::OfflineIndex;

pub struct SharedHandlesAndConfig {
    pub session_store: Arc<RocksDBSessionStore>,
    pub vsknn_index: Arc<OfflineIndex>,
    pub m_most_recent_sessions: usize,
    pub neighborhood_size_k: usize,
    pub num_items_to_recommend: usize,
    pub max_items_in_session: usize,
    pub qty_workers: usize,
    pub db_compaction_ttl_in_secs: usize,
    pub enable_business_logic: bool,
}

pub struct TrainingDataStats {
    pub descriptive_name: String,
    pub qty_records: usize,
    pub qty_unique_session_ids: usize,
    pub qty_unique_item_ids: usize,
    pub min_time_date_time: NaiveDateTime,
    pub max_time_date_time: NaiveDateTime,
    pub session_duration_p05: u64,
    pub session_duration_p25: u64,
    pub session_duration_p50: u64,
    pub session_duration_p75: u64,
    pub session_duration_p90: u64,
    pub session_duration_p95: u64,
    pub session_duration_p99: u64,
    pub session_duration_p99_5: u64,
    pub session_duration_p100: u64,
    pub qty_events_p05: u64,
    pub qty_events_p25: u64,
    pub qty_events_p50: u64,
    pub qty_events_p75: u64,
    pub qty_events_p90: u64,
    pub qty_events_p95: u64,
    pub qty_events_p99: u64,
    pub qty_events_p99_5: u64,
    pub qty_events_p100: u64,
}

pub fn determine_training_data_statistics(
    descriptive_name: &str,
    training_data: &[(u32, u64, u32)],
) -> TrainingDataStats {
    let qty_records = training_data.len();

    let mut session_ids: Vec<u32> = training_data
        .into_par_iter()
        .map(|(session_id, _item_id, _time)| *session_id)
        .collect();
    session_ids.par_sort_unstable();
    session_ids.dedup();
    let qty_unique_session_ids = session_ids.len();

    let mut item_ids: Vec<u64> = training_data
        .into_par_iter()
        .map(|(_session_id, item_id, _time)| *item_id)
        .collect();
    item_ids.par_sort_unstable();
    item_ids.dedup();
    let qty_unique_item_ids = item_ids.len();

    let min_time = training_data
        .par_iter()
        .map(|(_session_id, _item_id, time)| time)
        .min()
        .unwrap();
    let max_time = training_data
        .par_iter()
        .map(|(_session_id, _item_id, time)| time)
        .max()
        .unwrap();

    // │     Session duration percentiles (secs):  p5=14 p25=77 p50=248 p75=681 p90=1316 p95=1862 p99=3359 p99.5=4087 p100=539931
    println!("Using hardcoded session duration percentiles.");
    let session_duration_p05 = 14_u64;
    let session_duration_p25 = 77_u64;
    let session_duration_p50 = 248_u64;
    let session_duration_p75 = 681_u64;
    let session_duration_p90 = 1316_u64;
    let session_duration_p95 = 1862_u64;
    let session_duration_p99 = 3359_u64;
    let session_duration_p99_5 = 4087_u64;
    let session_duration_p100 = 539931_u64;

    // Session qty event percentiles:  p5=2 p25=2 p50=3 p75=6 p90=10 p95=14 p99=27 p99.5=34 p100=9408
    println!("Using hardcoded qty event percentiles.");
    let qty_events_p05 = 2_u64;
    let qty_events_p25 = 2_u64;
    let qty_events_p50 = 3_u64;
    let qty_events_p75 = 6_u64;
    let qty_events_p90 = 10_u64;
    let qty_events_p95 = 14_u64;
    let qty_events_p99 = 27_u64;
    let qty_events_p99_5 = 34_u64;
    let qty_events_p100 = 9408_u64;

    let min_time_date_time = NaiveDateTime::from_timestamp(*min_time as i64, 0);
    let max_time_date_time = NaiveDateTime::from_timestamp(*max_time as i64, 0);

    println!("Loaded {}", descriptive_name);
    println!("\tEvents: {}", qty_records);
    println!("\tSessions: {}", qty_unique_session_ids);
    println!("\tItems: {}", qty_unique_item_ids);
    println!("\tSpan: {} / {}", min_time_date_time, max_time_date_time);
    print!("\tSession duration percentiles (secs): ");
    print!(" p5={}", &session_duration_p05);
    print!(" p25={}", &session_duration_p25);
    print!(" p50={}", &session_duration_p50);
    print!(" p75={}", &session_duration_p75);
    print!(" p90={}", &session_duration_p90);
    print!(" p95={}", &session_duration_p95);
    print!(" p99={}", &session_duration_p99);
    print!(" p99.5={}", &session_duration_p99_5);
    println!(" p100={}", &session_duration_p100);
    print!("\tSession qty event percentiles: ");
    print!(" p5={}", &qty_events_p05);
    print!(" p25={}", &qty_events_p25);
    print!(" p50={}", &qty_events_p50);
    print!(" p75={}", &qty_events_p75);
    print!(" p90={}", &qty_events_p90);
    print!(" p95={}", &qty_events_p95);
    print!(" p99={}", &qty_events_p99);
    print!(" p99.5={}", &qty_events_p99_5);
    println!(" p100={}", &qty_events_p100);

    TrainingDataStats {
        descriptive_name: descriptive_name.to_string(),
        qty_records,
        qty_unique_session_ids,
        qty_unique_item_ids,
        min_time_date_time,
        max_time_date_time,
        session_duration_p05,
        session_duration_p25,
        session_duration_p50,
        session_duration_p75,
        session_duration_p90,
        session_duration_p95,
        session_duration_p99,
        session_duration_p99_5,
        session_duration_p100,
        qty_events_p05,
        qty_events_p25,
        qty_events_p50,
        qty_events_p75,
        qty_events_p90,
        qty_events_p95,
        qty_events_p99,
        qty_events_p99_5,
        qty_events_p100,
    }
}
