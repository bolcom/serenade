use crate::dataframeutils::{TrainingDataStats, TrainingStatsCollector};
use crate::vmisknn::similarity_indexed::SimilarityComputationNew;
use crate::vmisknn::SessionScore;
use crate::vmisknn::SessionTime;
use dary_heap::OctonaryHeap;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use avro_rs::from_value;
use avro_rs::Reader;
use itertools::Itertools;

#[derive(Serialize, Deserialize)]
pub struct ProductAttributes {
    pub(crate) is_adult: bool,
    pub(crate) is_for_sale: bool,
}

#[derive(Serialize, Deserialize)]
pub struct VMISIndex {
    pub(crate) chronological_events: bool,
    pub(crate) candidate_capacity: Option<usize>,
    pub(crate) reference_idf: bool,
    pub(crate) item_to_top_sessions_ordered: HashMap<u64, Vec<u32>>,
    pub(crate) session_to_max_time_stamp: Vec<u32>,
    pub(crate) item_to_idf_score: HashMap<u64, f64>,
    pub(crate) session_to_items_sorted: Vec<Vec<u64>>,
    pub(crate) training_data_stats: TrainingDataStats,
    pub(crate) item_to_product_attributes: HashMap<u64, ProductAttributes>,
}

impl VMISIndex {
    fn candidate_similarities(&self, evolving_session: &[u64], m: usize) -> HashMap<u32, f64> {
        if m == 0 {
            return HashMap::new();
        }
        // We use a d-ary heap for the (timestamp, session_id) tuple, a hashmap for the (session_id, score) tuples, and a hashmap for the unique items in the evolving session
        let mut heap_timestamps = OctonaryHeap::<SessionTime>::with_capacity(m);
        let mut session_similarities = HashMap::with_capacity(m);
        let len_evolving_session = evolving_session.len();
        let mut unique = evolving_session.iter().clone().collect_vec();
        unique.sort_unstable();
        unique.dedup();

        let qty_unique_session_items = unique.len() as f64;

        let mut hash_items = HashMap::with_capacity(len_evolving_session);

        //  Loop over items in evolving session in reverse order
        for (pos, item_id) in evolving_session.iter().rev().enumerate() {
            // Duplicate items: only calculate similarity score for the item in the farthest position in the evolving session
            match hash_items.insert(*item_id, pos) {
                Some(_) => {}
                None => {
                    // Find similar sessions in training data
                    if let Some(similar_sessions) = self.item_to_top_sessions_ordered.get(item_id) {
                        let decay_factor =
                            (len_evolving_session - pos) as f64 / qty_unique_session_items;
                        // Loop over all similar sessions.
                        'session_loop: for session_id in similar_sessions.iter().take(m) {
                            match session_similarities.get_mut(session_id) {
                                Some(similarity) => *similarity += decay_factor,
                                None => {
                                    let session_time_stamp =
                                        self.session_to_max_time_stamp[*session_id as usize];
                                    if session_similarities.len() < m {
                                        session_similarities.insert(*session_id, decay_factor);
                                        heap_timestamps.push(SessionTime::new(
                                            *session_id,
                                            session_time_stamp,
                                        ));
                                    } else {
                                        let mut bottom = heap_timestamps.peek_mut().unwrap();
                                        if session_time_stamp > bottom.time {
                                            // Remove the the existing minimum time stamp.
                                            session_similarities.remove_entry(&bottom.session_id);
                                            // Set new minimum timestamp
                                            session_similarities.insert(*session_id, decay_factor);
                                            *bottom =
                                                SessionTime::new(*session_id, session_time_stamp);
                                        } else {
                                            break 'session_loop;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        session_similarities
    }

    /// The same bounded recent-session retrieval used by legacy VMIS.
    pub fn candidate_sessions(&self, current: &[u64], m: usize) -> Vec<u32> {
        let mut ids: Vec<_> = self
            .candidate_similarities(current, m)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        ids.sort_by_key(|&id| {
            (
                std::cmp::Reverse(self.session_to_max_time_stamp[id as usize]),
                id,
            )
        });
        ids
    }

    /// Load a model using the shared VMIS index. Legacy Avro lacks chronology
    /// and raw-event IDF; reject those modes rather than approximate their scores.
    pub fn load_for_model(
        path: &str,
        m: usize,
        legacy_idf_weight: f64,
        config: &super::algorithms::KnnConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        if Path::new(path).is_dir() {
            if config.positional() || config.needs_reference_idf() {
                return Err("Legacy Avro has no chronological events or raw-event IDF. Rebuild from the original chronological training CSV for this model.".into());
            }
            return Self::try_new(path).map_err(|e| e.to_string());
        }
        if !Path::new(path).is_file() {
            return Err(format!("Training data does not exist: {}", path));
        }
        use std::io::Read;
        let mut reader = std::io::BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        let mut header = [0_u8; 17];
        let mut index: Self = if reader.read_exact(&mut header).is_ok()
            && &header == b"SERENADE-VMIS-003"
        {
            bincode::deserialize_from(reader)
                .map_err(|e| format!("Invalid VMIS snapshot: {}", e))?
        } else {
            if header.starts_with(b"SERENADE-VMIS") {
                return Err("Unsupported VMIS snapshot version; rebuild from training data".into());
            }
            Self::try_new_from_csv(path, m, legacy_idf_weight).map_err(|e| e.to_string())?
        };
        index.prepare_for_model(m, legacy_idf_weight, config)?;
        Ok(index)
    }

    /// Reconfigure scoring statistics without rebuilding postings or loading events.
    pub fn prepare_for_model(
        &mut self,
        m: usize,
        legacy_idf_weight: f64,
        config: &super::algorithms::KnnConfig,
    ) -> Result<(), String> {
        config.validate()?;
        if self
            .candidate_capacity
            .map_or(false, |capacity| m > capacity)
        {
            return Err(
                "Requested m exceeds snapshot posting capacity; rebuild from training CSV".into(),
            );
        }
        if config.positional() && !self.chronological_events {
            return Err(
                "This snapshot lacks chronological repeated events; rebuild from training CSV"
                    .into(),
            );
        }
        if config.needs_reference_idf() && !self.reference_idf {
            if !self.chronological_events {
                return Err(
                    "This snapshot lacks raw events for reference IDF; rebuild from training CSV"
                        .into(),
                );
            }
            self.set_reference_idf();
        } else if config.algorithm == super::algorithms::Algorithm::Vmis
            && self.chronological_events
        {
            // Restore the legacy formula without adding a second permanent IDF map.
            let mut counts = HashMap::<u64, usize>::new();
            for sequence in &self.session_to_items_sorted {
                let unique: hashbrown::HashSet<_> = sequence.iter().copied().collect();
                if unique.len() <= self.training_data_stats.qty_events_p99_5 as usize {
                    for item in unique {
                        *counts.entry(item).or_default() += 1;
                    }
                }
            }
            let total = counts.values().sum::<usize>() as f64;
            self.item_to_idf_score = counts
                .into_iter()
                .map(|(item, count)| (item, (total / count as f64).ln() * legacy_idf_weight))
                .collect();
            self.reference_idf = false;
        }
        self.validate().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Historical events for evaluation statistics; no second event corpus is stored.
    pub fn training_items(&self) -> impl Iterator<Item = u64> + Clone + '_ {
        self.session_to_items_sorted.iter().flatten().copied()
    }

    /// Owned allocation estimate, excluding hash-table control bytes and allocator overhead.
    pub fn estimated_heap_bytes(&self) -> usize {
        use std::mem::size_of;
        self.session_to_items_sorted.capacity() * size_of::<Vec<u64>>()
            + self
                .session_to_items_sorted
                .iter()
                .map(|v| v.capacity() * size_of::<u64>())
                .sum::<usize>()
            + self.session_to_max_time_stamp.capacity() * size_of::<u32>()
            + self.item_to_top_sessions_ordered.capacity() * size_of::<(u64, Vec<u32>)>()
            + self
                .item_to_top_sessions_ordered
                .values()
                .map(|v| v.capacity() * size_of::<u32>())
                .sum::<usize>()
            + self.item_to_idf_score.capacity() * size_of::<(u64, f64)>()
            + self.item_to_product_attributes.capacity() * size_of::<(u64, ProductAttributes)>()
    }

    fn set_reference_idf(&mut self) {
        let mut counts = HashMap::<u64, usize>::new();
        for item in self.session_to_items_sorted.iter().flatten() {
            *counts.entry(*item).or_default() += 1;
        }
        let sessions = self.session_to_items_sorted.len() as f64;
        self.item_to_idf_score = counts
            .into_iter()
            .map(|(item, count)| (item, (sessions / count as f64).ln()))
            .collect();
        self.reference_idf = true;
    }

    /// Persist aggregates alongside externally prepared Avro, from the SAME raw corpus.
    pub fn save_training_statistics(&self, path: &Path) -> anyhow::Result<()> {
        self.training_data_stats.write_metadata(path)
    }

    /// Versioned snapshot retaining chronological repetitions and IDF semantics.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        use std::io::Write;
        let mut writer = std::io::BufWriter::new(File::create(path)?);
        writer.write_all(b"SERENADE-VMIS-003")?;
        bincode::serialize_into(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn new_from_csv(
        path_to_training: &str,
        m_most_recent_sessions: usize,
        idf_weighting: f64,
    ) -> Self {
        Self::try_new_from_csv(path_to_training, m_most_recent_sessions, idf_weighting)
            .expect("load CSV prediction index")
    }

    pub fn try_new_from_csv(
        path_to_training: &str,
        m_most_recent_sessions: usize,
        idf_weighting: f64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(idf_weighting.is_finite(), "IDF weighting must be finite");
        let start_time = Instant::now();
        println!(
            "reading training data, determine items per training session {}",
            &path_to_training
        );
        let data_train = read_from_file(path_to_training);
        let (historical_sessions_train, historical_sessions_max_time_stamp, training_data_stats) =
            data_train.map_err(|error| {
                anyhow::anyhow!("load training data {}: {}", path_to_training, error)
            })?;
        println!(
            "reading training data, determine items per training session:{} micros",
            start_time.elapsed().as_micros()
        );

        let start_time = Instant::now();
        println!("prepare indexes");
        let (item_to_top_sessions_ordered, item_to_idf_score, item_to_product_attributes) =
            prepare_hashmap(
                &historical_sessions_train,
                &historical_sessions_max_time_stamp,
                m_most_recent_sessions,
                training_data_stats.qty_events_p99_5 as usize,
                idf_weighting,
            );
        println!(
            "prepare indexes:{} micros",
            start_time.elapsed().as_micros()
        );

        let index = VMISIndex {
            chronological_events: true,
            candidate_capacity: Some(m_most_recent_sessions),
            reference_idf: false,
            item_to_top_sessions_ordered,
            session_to_max_time_stamp: historical_sessions_max_time_stamp,
            item_to_idf_score,
            session_to_items_sorted: historical_sessions_train,
            training_data_stats,
            item_to_product_attributes,
        };
        index.validate()?;
        Ok(index)
    }

    /// Check all references used by prediction before accepting requests.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.item_to_top_sessions_ordered.is_empty(),
            "prediction index is empty"
        );
        anyhow::ensure!(
            self.session_to_items_sorted.len() == self.session_to_max_time_stamp.len(),
            "session items and timestamps have different lengths"
        );
        for (item, idf) in &self.item_to_idf_score {
            anyhow::ensure!(idf.is_finite(), "item {} has nonfinite IDF", item);
        }
        let mut checked_sessions = hashbrown::HashSet::new();
        for (&item, sessions) in &self.item_to_top_sessions_ordered {
            let mut previous_time = u32::MAX;
            for &session in sessions {
                let items = self
                    .session_to_items_sorted
                    .get(session as usize)
                    .ok_or_else(|| {
                        anyhow::anyhow!("item {} references missing session {}", item, session)
                    })?;
                anyhow::ensure!(
                    !items.is_empty(),
                    "item {} references empty session {}",
                    item,
                    session
                );
                anyhow::ensure!(
                    items.contains(&item),
                    "item {} references session {} without that item",
                    item,
                    session
                );
                let timestamp = self.session_to_max_time_stamp[session as usize];
                anyhow::ensure!(
                    timestamp <= previous_time,
                    "item {} sessions are not timestamp descending",
                    item
                );
                previous_time = timestamp;
                if !checked_sessions.insert(session) {
                    continue;
                }
                for candidate in items {
                    let idf = self.item_to_idf_score.get(candidate).ok_or_else(|| {
                        anyhow::anyhow!("session {} item {} has no IDF entry", session, candidate)
                    })?;
                    anyhow::ensure!(
                        idf.is_finite(),
                        "session {} item {} has nonfinite IDF",
                        session,
                        candidate
                    );
                }
            }
        }
        Ok(())
    }

    pub fn try_new(base_path: &str) -> anyhow::Result<Self> {
        use anyhow::Context;
        // Session Avro has neither original event counts nor minimum timestamps.
        // Require aggregates computed from the original, unpruned training events.
        let metadata = Path::new(base_path).join("training-stats.bin");
        let training_data_stats = TrainingDataStats::read_metadata(&metadata)
            .with_context(|| format!("{}: missing or invalid raw training statistics; regenerate training-stats.bin from the original training CSV or rebuild as a .vmis snapshot", metadata.display()))?;

        println!(
            "reading training data, determine items per training session {}",
            &base_path
        );
        let start_time = Instant::now();
        let (item_to_top_sessions_ordered, item_to_idf_score, item_to_product_attributes) =
            create_item_indices_from_avro(&*(base_path.to_owned() + "/itemindex/"))?;
        println!(
            "indexing item indices: {} secs",
            start_time.elapsed().as_secs()
        );
        let start_time = Instant::now();
        let (session_to_items_sorted, session_to_max_time_stamp) =
            create_session_indices_from_avro(&*(base_path.to_owned() + "/sessionindex/"))?;
        anyhow::ensure!(
            !item_to_top_sessions_ordered.is_empty()
                && session_to_items_sorted
                    .iter()
                    .any(|items| !items.is_empty()),
            "{}: empty prediction index",
            base_path
        );
        println!(
            "indexing session indices: {} secs",
            start_time.elapsed().as_secs()
        );

        #[allow(non_snake_case)]
        #[derive(Debug, Deserialize)]
        struct ItemIdexAvroSchema {
            ItemId: i64,
            session_indices_time_ordered: Vec<i32>,
            idf: f64,
            ForSale: bool,
            IsAdult: bool,
        }
        fn create_item_indices_from_avro(
            dir: &str,
        ) -> anyhow::Result<(
            HashMap<u64, Vec<u32>>,
            HashMap<u64, f64>,
            HashMap<u64, ProductAttributes>,
        )> {
            let dir_entry =
                fs::read_dir(dir).with_context(|| format!("read index directory {}", dir))?;
            let mut item_to_top_sessions_ordered = HashMap::new();
            let mut item_to_idf = HashMap::new();
            let mut item_to_product_attributes = HashMap::new();
            for path in dir_entry {
                let full_path_to_file = path
                    .with_context(|| format!("read directory entry {}", dir))?
                    .path()
                    .display()
                    .to_string();
                if full_path_to_file.ends_with(".avro") {
                    let file = File::open(&Path::new(&full_path_to_file))
                        .with_context(|| format!("open {}", full_path_to_file))?;
                    let reader = Reader::new(file)
                        .with_context(|| format!("read Avro header {}", full_path_to_file))?;
                    for (record, value) in reader.enumerate() {
                        let parse_result =
                            from_value::<ItemIdexAvroSchema>(&value.with_context(|| {
                                format!("{} record {}", full_path_to_file, record + 1)
                            })?);
                        match parse_result {
                            Ok(item_index) => {
                                anyhow::ensure!(
                                    item_index.ItemId >= 0
                                        && item_index.idf.is_finite()
                                        && item_index
                                            .session_indices_time_ordered
                                            .iter()
                                            .all(|id| *id >= 0),
                                    "{} record {}: invalid item ID, session reference or IDF",
                                    full_path_to_file,
                                    record + 1
                                );
                                let top_sessions_ordered = item_index
                                    .session_indices_time_ordered
                                    .iter()
                                    .map(|x| *x as u32)
                                    .collect_vec();
                                item_to_top_sessions_ordered
                                    .insert(item_index.ItemId as u64, top_sessions_ordered);
                                item_to_idf.insert(item_index.ItemId as u64, item_index.idf);
                                let attributes = ProductAttributes {
                                    is_adult: item_index.IsAdult,
                                    is_for_sale: item_index.ForSale,
                                };
                                item_to_product_attributes
                                    .insert(item_index.ItemId as u64, attributes);
                            }
                            Err(err) => {
                                return Err(anyhow::anyhow!(
                                    "{} record {}: {}",
                                    full_path_to_file,
                                    record + 1,
                                    err
                                ));
                            }
                        }
                    }
                }
            }
            println!(
                " item_to_top_sessions_ordered.len():{}",
                item_to_top_sessions_ordered.len()
            );
            Ok((
                item_to_top_sessions_ordered,
                item_to_idf,
                item_to_product_attributes,
            ))
        }

        #[allow(non_snake_case)]
        #[derive(Debug, Deserialize)]
        struct SessionIdexAvroSchema {
            SessionIndex: i32,
            item_ids_asc: Vec<i64>,
            Time: i32,
        }
        fn create_session_indices_from_avro(
            dir: &str,
        ) -> anyhow::Result<(Vec<Vec<u64>>, Vec<u32>)> {
            let mut session_to_items_sorted = Vec::new();
            let mut timestamps = Vec::new();
            let dir_entry =
                fs::read_dir(dir).with_context(|| format!("read index directory {}", dir))?;
            for path in dir_entry {
                let full_path_to_file = path
                    .with_context(|| format!("read directory entry {}", dir))?
                    .path()
                    .display()
                    .to_string();
                if full_path_to_file.ends_with(".avro") {
                    let file = File::open(&Path::new(&full_path_to_file))
                        .with_context(|| format!("open {}", full_path_to_file))?;
                    let reader = Reader::new(file)
                        .with_context(|| format!("read Avro header {}", full_path_to_file))?;
                    for (record, value) in reader.enumerate() {
                        let parse_result =
                            from_value::<SessionIdexAvroSchema>(&value.with_context(|| {
                                format!("{} record {}", full_path_to_file, record + 1)
                            })?);
                        match parse_result {
                            Ok(session_index) => {
                                anyhow::ensure!(
                                    session_index.SessionIndex >= 0
                                        && session_index.Time >= 0
                                        && session_index.item_ids_asc.iter().all(|id| *id >= 0),
                                    "{} record {}: invalid session ID, item ID or timestamp",
                                    full_path_to_file,
                                    record + 1
                                );
                                let session_items_asc = session_index
                                    .item_ids_asc
                                    .iter()
                                    .map(|x| *x as u64)
                                    .collect_vec();
                                let session_id = session_index.SessionIndex as usize;
                                if session_id >= session_to_items_sorted.len() {
                                    let new_size = session_id + 1;
                                    session_to_items_sorted.resize(new_size, Vec::new());
                                    timestamps.resize(new_size, 0);
                                }
                                session_to_items_sorted[session_id] = session_items_asc;
                                timestamps[session_id] = session_index.Time as u32;
                            }
                            Err(err) => {
                                return Err(anyhow::anyhow!(
                                    "{} record {}: {}",
                                    full_path_to_file,
                                    record + 1,
                                    err
                                ));
                            }
                        }
                    }
                }
            }
            Ok((session_to_items_sorted, timestamps))
        }

        let index = VMISIndex {
            chronological_events: false,
            candidate_capacity: None,
            reference_idf: false,
            item_to_top_sessions_ordered,
            session_to_max_time_stamp,
            item_to_idf_score,
            session_to_items_sorted,
            training_data_stats,
            item_to_product_attributes,
        };
        index.validate()?;
        Ok(index)
    }
}

impl SimilarityComputationNew for VMISIndex {
    fn items_for_session(&self, session: &u32) -> &[u64] {
        &self.session_to_items_sorted[*session as usize]
    }

    fn idf(&self, item: &u64) -> f64 {
        self.item_to_idf_score[item]
    }

    fn find_neighbors(
        &self,
        evolving_session: &[u64],
        k: usize,
        m: usize,
    ) -> BinaryHeap<SessionScore> {
        self.find_neighbors_with_scratch(
            evolving_session,
            k,
            m,
            &mut crate::vmisknn::NeighborScratch::default(),
        )
    }

    fn find_neighbors_with_scratch(
        &self,
        evolving_session: &[u64],
        k: usize,
        m: usize,
        scratch: &mut crate::vmisknn::NeighborScratch,
    ) -> BinaryHeap<SessionScore> {
        if evolving_session.is_empty() || k == 0 || m == 0 {
            return BinaryHeap::new();
        }
        // We use a d-ary heap for the (timestamp, session_id) tuple, a hashmap for the (session_id, score) tuples, and a hashmap for the unique items in the evolving session
        let heap_timestamps = &mut scratch.timestamps;
        heap_timestamps.clear();
        heap_timestamps.reserve(m);
        let session_similarities = &mut scratch.similarities;
        session_similarities.clear();
        session_similarities.reserve(m);
        let len_evolving_session = evolving_session.len();
        let hash_items = &mut scratch.items;
        hash_items.clear();
        hash_items.reserve(len_evolving_session);
        for item in evolving_session {
            hash_items.insert(*item, false);
        }
        let qty_unique_session_items = hash_items.len() as f64;

        //  Loop over items in evolving session in reverse order
        for (pos, item_id) in evolving_session.iter().rev().enumerate() {
            // Duplicate items: only calculate similarity score for the item in the farthest position in the evolving session
            match hash_items.insert(*item_id, true) {
                Some(true) => {}
                _ => {
                    // Find similar sessions in training data
                    if let Some(similar_sessions) = self.item_to_top_sessions_ordered.get(item_id) {
                        let decay_factor =
                            (len_evolving_session - pos) as f64 / qty_unique_session_items;
                        // Loop over all similar sessions.
                        'session_loop: for session_id in similar_sessions.iter().take(m) {
                            match session_similarities.get_mut(session_id) {
                                Some(similarity) => *similarity += decay_factor,
                                None => {
                                    let session_time_stamp =
                                        self.session_to_max_time_stamp[*session_id as usize];
                                    if session_similarities.len() < m {
                                        session_similarities.insert(*session_id, decay_factor);
                                        heap_timestamps.push(SessionTime::new(
                                            *session_id,
                                            session_time_stamp,
                                        ));
                                    } else {
                                        let mut bottom = heap_timestamps.peek_mut().unwrap();
                                        if session_time_stamp > bottom.time {
                                            // Remove the the existing minimum time stamp.
                                            session_similarities.remove_entry(&bottom.session_id);
                                            // Set new minimum timestamp
                                            session_similarities.insert(*session_id, decay_factor);
                                            *bottom =
                                                SessionTime::new(*session_id, session_time_stamp);
                                        } else {
                                            break 'session_loop;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return top-k
        let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
        // Stable traversal also fixes accumulation order when neighbor scores tie.
        let mut similarities: Vec<_> = session_similarities.iter().collect();
        similarities.sort_unstable_by_key(|(id, _)| **id);
        for (session_id, score) in similarities {
            if closest_neighbors.len() < k {
                let scored_session = SessionScore::new(*session_id, *score);
                closest_neighbors.push(scored_session);
            } else {
                let mut bottom = closest_neighbors.peek_mut().unwrap();
                if score > &bottom.score {
                    let scored_session = SessionScore::new(*session_id, *score);
                    *bottom = scored_session;
                } else if (score - bottom.score).abs() < f64::EPSILON
                    && ((
                        self.session_to_max_time_stamp[*session_id as usize],
                        std::cmp::Reverse(*session_id),
                    ) > (
                        self.session_to_max_time_stamp[bottom.id as usize],
                        std::cmp::Reverse(bottom.id),
                    ))
                {
                    let scored_session = SessionScore::new(*session_id, *score);
                    *bottom = scored_session;
                }
            }
        }
        // Closest neigbours contain unique session_ids and corresponding top-k similarity scores
        closest_neighbors
    }

    fn find_attributes(&self, item_id: &u64) -> Option<&ProductAttributes> {
        self.item_to_product_attributes.get(item_id)
    }
}

pub(crate) fn prepare_hashmap(
    historical_sessions: &[Vec<u64>],
    timestamps: &[u32],
    m_most_recent_sessions: usize,
    max_training_session_length: usize,
    idf_weighting: f64,
) -> (
    HashMap<u64, Vec<u32>>,
    HashMap<u64, f64>,
    HashMap<u64, ProductAttributes>,
) {
    let mut interactions = Vec::new();
    for (session_id, session) in historical_sessions.iter().enumerate() {
        let unique: hashbrown::HashSet<_> = session.iter().copied().collect();
        if unique.len() <= max_training_session_length {
            for item in unique {
                interactions.push((item, timestamps[session_id], session_id as u32));
            }
        }
    }
    interactions.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
    });
    let mut postings = HashMap::new();
    let mut idfs = HashMap::new();
    let mut attributes = HashMap::new();
    let mut start = 0;
    while start < interactions.len() {
        let item = interactions[start].0;
        let mut end = start + 1;
        while end < interactions.len() && interactions[end].0 == item {
            end += 1;
        }
        postings.insert(
            item,
            interactions[start..end]
                .iter()
                .take(m_most_recent_sessions)
                .map(|record| record.2)
                .collect(),
        );
        idfs.insert(
            item,
            (interactions.len() as f64 / (end - start) as f64).ln() * idf_weighting,
        );
        attributes.insert(
            item,
            ProductAttributes {
                is_adult: false,
                is_for_sale: true,
            },
        );
        start = end;
    }
    (postings, idfs, attributes)
}

pub fn read_from_file(
    path: &str,
) -> Result<(Vec<Vec<u64>>, Vec<u32>, TrainingDataStats), Box<dyn Error>> {
    // Creates a new csv `Reader` from a file
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(path)?;

    // One event array, sorted only for the index's required chronology. Stable
    // sorting preserves input order at equal timestamps; no statistics-only sort.
    let mut events = Vec::<(usize, u64, f64)>::new();
    let mut unique_items = hashbrown::HashSet::new();
    for (record, result) in reader.deserialize().enumerate() {
        let raw: (usize, u64, f64) = result.map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{} record {}: {}", path, record + 1, error),
            )
        })?;
        if !raw.2.is_finite() || raw.2 < 0.0 || raw.2.round() > u32::MAX as f64 {
            return Err(format!("{} record {}: timestamp must fit u32", path, record + 1).into());
        }
        unique_items.insert(raw.1);
        events.push(raw);
    }
    if events.is_empty() {
        return Err(format!("{}: training data is empty", path).into());
    }
    let unique_count = unique_items.len();
    drop(unique_items);
    events.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.total_cmp(&b.2)));
    let mut collector = TrainingStatsCollector::new();
    let mut sessions = Vec::new();
    let mut timestamps = Vec::new();
    let mut start = 0;
    while start < events.len() {
        let mut end = start + 1;
        while end < events.len() && events[end].0 == events[start].0 {
            end += 1;
        }
        collector.session(end - start, events[start].2, events[end - 1].2);
        sessions.push(events[start..end].iter().map(|event| event.1).collect());
        timestamps.push(events[end - 1].2.round() as u32);
        start = end;
    }
    Ok((sessions, timestamps, collector.finish(path, unique_count)))
}

#[cfg(test)]
mod index_build_tests {
    use super::*;

    #[test]
    fn grouped_interactions_preserve_filtering_idf_and_timestamp_ties() {
        let sessions = vec![vec![1, 2], vec![1, 3], vec![1, 4], vec![1, 2, 3, 4]];
        let (postings, idfs, _) = prepare_hashmap(&sessions, &[10, 20, 20, 99], 2, 2, 1.0);
        assert_eq!(postings[&1], vec![2, 1]);
        assert_eq!(postings[&2], vec![0]);
        assert!((idfs[&1] - 2.0_f64.ln()).abs() < 1e-12);
        let (postings, idfs, _) = prepare_hashmap(&sessions, &[10, 20, 20, 99], 0, 2, 0.0);
        assert!(postings.values().all(Vec::is_empty));
        assert!(idfs.values().all(|idf| *idf == 0.0));
    }
}

#[cfg(test)]
mod csv_loading_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn fixture(contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "serenade-{}-{}.tsv",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn final_session_and_final_event_are_retained() {
        let path = fixture("session\titem\ttime\n1\t10\t100\n2\t20\t200\n2\t21\t201\n");
        let (items, timestamps, _) = read_from_file(path.to_str().unwrap()).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(items, vec![vec![10], vec![20, 21]]);
        assert_eq!(timestamps, vec![100, 201]);
    }

    #[test]
    fn rejects_missing_sessions_and_idf_before_prediction() {
        let path = fixture("session\titem\ttime\n1\t10\t100\n1\t20\t101\n");
        let mut index = VMISIndex::try_new_from_csv(path.to_str().unwrap(), 10, 1.0).unwrap();
        fs::remove_file(path).unwrap();
        index.item_to_idf_score.remove(&20);
        assert!(index.validate().unwrap_err().to_string().contains("IDF"));
        index.item_to_idf_score.insert(20, 1.0);
        index.item_to_top_sessions_ordered.insert(10, vec![99]);
        assert!(index
            .validate()
            .unwrap_err()
            .to_string()
            .contains("missing session"));
    }

    #[test]
    fn scratch_and_fresh_predictions_have_equal_scores() {
        let path = fixture("session\titem\ttime\n1\t10\t100\n1\t20\t101\n2\t10\t102\n2\t30\t103\n");
        let index = VMISIndex::try_new_from_csv(path.to_str().unwrap(), 10, 1.0).unwrap();
        fs::remove_file(path).unwrap();
        let mut scratch = crate::vmisknn::PredictionScratch::default();
        for items in [vec![10], vec![10, 20, 10], vec![999], vec![], vec![30]] {
            let scores = |heap: BinaryHeap<crate::vmisknn::ItemScore>| {
                let mut scores: Vec<_> =
                    heap.into_iter().map(|item| (item.id, item.score)).collect();
                scores.sort_by_key(|item| item.0);
                scores
            };
            assert_eq!(
                scores(crate::vmisknn::predict(&index, &items, 10, 10, 10, false)),
                scores(crate::vmisknn::predict_with_scratch(
                    &index,
                    &items,
                    10,
                    10,
                    10,
                    false,
                    &mut scratch
                ))
            );
        }
    }

    #[test]
    fn invalid_rows_and_timestamps_fail_loading() {
        for row in [
            "1\tbad\t10",
            "1\t2\tNaN",
            "1\t2\t-1",
            "1\t2\t-0.1",
            "1\t2\t4294967296",
            "",
        ] {
            let path = fixture(&format!("session\titem\ttime\n{}\n", row));
            assert!(read_from_file(path.to_str().unwrap()).is_err());
            fs::remove_file(path).unwrap();
        }
    }
}

#[cfg(test)]
mod sequence_tests {
    use super::*;

    #[test]
    fn training_keeps_chronological_repeats_and_final_events() {
        let path =
            std::env::temp_dir().join(format!("serenade-sequences-{}.tsv", std::process::id()));
        fs::write(&path, "SessionId\tItemId\tTime\n2\t9\t12\n1\t4\t3.4\n1\t7\t1\n1\t6\t3.1\n1\t7\t4\n2\t8\t11\n2\t9\t13\n").unwrap();
        let (sessions, times, _) = read_from_file(path.to_str().unwrap()).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(sessions, vec![vec![7, 6, 4, 7], vec![8, 9, 9]]);
        assert_eq!(times, vec![4, 13]);
    }

    #[test]
    fn snapshots_preserve_sequences_and_convert_idf_without_a_second_index() {
        use crate::vmisknn::algorithms::{Algorithm, KnnConfig};
        let base = std::env::temp_dir().join(format!("serenade-snapshot-{}", std::process::id()));
        let csv = base.with_extension("tsv");
        let snapshot = base.with_extension("vmis");
        fs::write(
            &csv,
            "SessionId\tItemId\tTime\n1\t1\t1\n1\t2\t2\n1\t1\t3\n2\t1\t4\n2\t3\t5\n",
        )
        .unwrap();
        let legacy = KnnConfig::default();
        let mut reference = KnnConfig::for_algorithm(Algorithm::Vsknn);
        reference.idf_weighting = 2.0;
        let index = VMISIndex::load_for_model(csv.to_str().unwrap(), 10, 1.0, &reference).unwrap();
        assert!(index.reference_idf);
        assert!((index.idf(&1) - (2.0_f64 / 3.0).ln()).abs() < 1e-12);
        index.save(snapshot.to_str().unwrap()).unwrap();
        let restored =
            VMISIndex::load_for_model(snapshot.to_str().unwrap(), 10, 1.0, &reference).unwrap();
        assert_eq!(
            restored.session_to_items_sorted,
            vec![vec![1, 2, 1], vec![1, 3]]
        );
        assert_eq!(
            restored.item_to_top_sessions_ordered,
            index.item_to_top_sessions_ordered
        );
        assert_eq!(restored.item_to_idf_score, index.item_to_idf_score);
        let original = VMISIndex::load_for_model(csv.to_str().unwrap(), 10, 2.0, &legacy).unwrap();
        let converted =
            VMISIndex::load_for_model(snapshot.to_str().unwrap(), 10, 2.0, &legacy).unwrap();
        assert_eq!(converted.item_to_idf_score, original.item_to_idf_score);
        assert!(VMISIndex::load_for_model(snapshot.to_str().unwrap(), 11, 1.0, &legacy).is_err());
        let mut old = restored;
        old.chronological_events = false;
        old.save(snapshot.to_str().unwrap()).unwrap();
        assert!(VMISIndex::load_for_model(
            snapshot.to_str().unwrap(),
            10,
            1.0,
            &KnnConfig::for_algorithm(Algorithm::Stan)
        )
        .is_err());
        fs::write(&snapshot, b"SERENADE-VMIS-001").unwrap();
        assert!(VMISIndex::load_for_model(snapshot.to_str().unwrap(), 10, 1.0, &legacy).is_err());
        fs::remove_file(csv).unwrap();
        fs::remove_file(snapshot).unwrap();
    }

    #[test]
    fn postings_and_legacy_idf_count_sessions_once_per_item() {
        let sessions = vec![vec![1, 2, 1], vec![1, 3, 1], vec![1, 4]];
        let (postings, idf, _) = prepare_hashmap(&sessions, &[5, 5, 4], 3, 2, 1.0);
        assert_eq!(postings[&1], vec![1, 0, 2]);
        assert!((idf[&1] - 2.0_f64.ln()).abs() < 1e-12);
        assert_eq!(sessions[0], vec![1, 2, 1]);
    }
}

#[cfg(test)]
mod statistics_integration_tests {
    use super::*;
    #[test]
    fn raw_statistics_control_pruning_before_deduplication() {
        let path =
            std::env::temp_dir().join(format!("serenade-pruning-{}.tsv", std::process::id()));
        let mut text = String::from("SessionId\tItemId\tTime\n");
        // 199 sessions with three raw clicks but one distinct item; one with four.
        for session in 0..199 {
            for time in [9, 1, 5] {
                text.push_str(&format!("{session}\t1\t{time}\n"));
            }
        }
        for item in 1..=4 {
            text.push_str(&format!("199\t{item}\t{item}\n"));
        }
        fs::write(&path, text).unwrap();
        let index = VMISIndex::try_new_from_csv(path.to_str().unwrap(), 500, 1.0).unwrap();
        fs::remove_file(&path).unwrap();
        let stats = &index.training_data_stats;
        assert_eq!(stats.qty_records, 601);
        assert_eq!(stats.qty_unique_session_ids, 200);
        assert_eq!(stats.qty_events_p99_5, 3);
        assert_eq!(stats.qty_events_p100, 4);
        assert_eq!(stats.session_duration_p100, 8);
        assert_eq!(stats.qty_unique_item_ids, 4); // includes pruned-only items
        assert!(!index.item_to_top_sessions_ordered.contains_key(&4));
        assert_eq!(index.session_to_items_sorted[199], vec![1, 2, 3, 4]);
        assert_eq!(index.session_to_items_sorted[0], vec![1, 1, 1]);
        // The old fixed 34 cutoff retains the four-item session.
        let (old, _, _) = prepare_hashmap(
            &index.session_to_items_sorted,
            &index.session_to_max_time_stamp,
            500,
            34,
            1.0,
        );
        assert_eq!(old[&4], vec![199]);
    }
    #[test]
    fn csv_fractional_durations_use_original_timestamps() {
        let path = std::env::temp_dir().join(format!(
            "serenade-fraction-stats-{}.tsv",
            std::process::id()
        ));
        fs::write(
            &path,
            "SessionId\tItemId\tTime\n2\t3\t11.1\n1\t1\t1.4\n1\t1\t2.6\n2\t4\t10.4\n",
        )
        .unwrap();
        let (_, _, stats) = read_from_file(path.to_str().unwrap()).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(stats.qty_records, 4);
        assert_eq!(stats.qty_events_p100, 2);
        assert_eq!(stats.session_duration_p100, 1); // round(max-min), not round(max)-round(min)
        assert_eq!(stats.min_time_date_time.and_utc().timestamp(), 1);
        assert_eq!(stats.max_time_date_time.and_utc().timestamp(), 11);
        assert_eq!(
            stats.min_time_date_time.and_utc().timestamp_subsec_nanos(),
            400_000_000
        );
        assert_eq!(
            stats.max_time_date_time.and_utc().timestamp_subsec_nanos(),
            100_000_000
        );
    }
    #[test]
    fn avro_requires_raw_metadata_and_preserves_it_exactly() {
        use avro_rs::{to_value, Schema, Writer};
        #[allow(non_snake_case)]
        #[derive(Serialize)]
        struct Session {
            SessionIndex: i32,
            item_ids_asc: Vec<i64>,
            Time: i32,
        }
        #[allow(non_snake_case)]
        #[derive(Serialize)]
        struct Item {
            ItemId: i64,
            session_indices_time_ordered: Vec<i32>,
            idf: f64,
            ForSale: bool,
            IsAdult: bool,
        }
        let base = std::env::temp_dir().join(format!("serenade-avro-stats-{}", std::process::id()));
        fs::create_dir_all(base.join("sessionindex")).unwrap();
        fs::create_dir_all(base.join("itemindex")).unwrap();
        let error = VMISIndex::try_new(base.to_str().unwrap()).err().unwrap();
        assert!(error.to_string().contains("original training CSV"));
        let schema = Schema::parse_str(r#"{"type":"record","name":"Session","fields":[{"name":"SessionIndex","type":"int"},{"name":"item_ids_asc","type":{"type":"array","items":"long"}},{"name":"Time","type":"int"}]}"#).unwrap();
        let mut writer = Writer::new(
            &schema,
            File::create(base.join("sessionindex/part.avro")).unwrap(),
        );
        writer
            .append(
                to_value(Session {
                    SessionIndex: 0,
                    item_ids_asc: vec![7],
                    Time: 10,
                })
                .unwrap(),
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);
        let schema = Schema::parse_str(r#"{"type":"record","name":"Item","fields":[{"name":"ItemId","type":"long"},{"name":"session_indices_time_ordered","type":{"type":"array","items":"int"}},{"name":"idf","type":"double"},{"name":"ForSale","type":"boolean"},{"name":"IsAdult","type":"boolean"}]}"#).unwrap();
        let mut writer = Writer::new(
            &schema,
            File::create(base.join("itemindex/part.avro")).unwrap(),
        );
        writer
            .append(
                to_value(Item {
                    ItemId: 7,
                    session_indices_time_ordered: vec![0],
                    idf: 0.0,
                    ForSale: true,
                    IsAdult: false,
                })
                .unwrap(),
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);
        let csv = base.join("raw.tsv");
        fs::write(&csv, "SessionId\tItemId\tTime\n1\t7\t2\n1\t7\t10\n").unwrap();
        let (_, _, stats) = read_from_file(csv.to_str().unwrap()).unwrap();
        stats
            .write_metadata(&base.join("training-stats.bin"))
            .unwrap();
        let index = VMISIndex::try_new(base.to_str().unwrap()).unwrap();
        assert_eq!(index.training_data_stats, stats);
        assert_eq!(index.training_data_stats.qty_records, 2);
        assert_eq!(index.training_data_stats.session_duration_p100, 8);
        assert_eq!(index.session_to_items_sorted, vec![vec![7]]);
        drop(index);
        fs::remove_dir_all(base).unwrap();
    }
}
