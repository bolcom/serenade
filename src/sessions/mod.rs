use rocksdb::{DB, Options};
use bincode;
use std::time::{Duration, SystemTime};
use crate::io::ItemId;
use serde::{Serialize, Deserialize};

pub struct RocksDBSessionStore {
    rocks_db: DB,
    max_session_idle_duration_in_seconds: u64,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct DBValue {
    session_items: Vec<ItemId>,
    epoch_secs: u64,
}


impl RocksDBSessionStore {
    pub fn new(database_file: &str, ttl: Duration) -> Self {
        let mut options = Options::default();
        options.create_if_missing(true);
        options.optimize_for_point_lookup(5000);
        options.set_allow_mmap_reads(true);
        options.set_allow_mmap_writes(true);

        let rocks_db =
            DB::open_with_ttl(
                &options,
                database_file,
                ttl,
            )
                .unwrap();

        Self { rocks_db, max_session_idle_duration_in_seconds: 60 * 20  }
    }

    pub fn get_session_items(&self, evolving_session_id: &u128) -> Vec<u64> {
        let serialized_session_id =
            bincode::serialize(&evolving_session_id).unwrap();

        let bytes = self.rocks_db.get(&serialized_session_id).unwrap();

        let session_items: Vec<u64> = match bytes {
            Some(bytes) => {
                let payload:DBValue = bincode::deserialize(&bytes).unwrap();
                let now = self.get_seconds_since_epoch();
                let seconds_since_last_event = now - payload.epoch_secs;
                if seconds_since_last_event <= self.max_session_idle_duration_in_seconds {
                    payload.session_items
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        };
        session_items
    }

    pub fn update_session_items(&self, evolving_session_id: &u128,
                                session_items: &[u64]) {
        let serialized_session_id =
            bincode::serialize(evolving_session_id).unwrap();
        let now = self.get_seconds_since_epoch();
        let payload = DBValue {
            session_items: Vec::from(session_items),
            epoch_secs: now,
        };
        let bytes = bincode::serialize(&payload).unwrap();

        let _ = self.rocks_db.put(&serialized_session_id, &bytes).unwrap();
    }

    fn get_seconds_since_epoch(&self) -> u64 {
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    }
}
