use crate::io::ItemId;
use crate::serialization as bincode;
use rocksdb::{Options, DB};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

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
        Self::try_new(database_file, ttl).expect("open session database")
    }

    pub fn try_new(database_file: &str, ttl: Duration) -> anyhow::Result<Self> {
        use anyhow::Context;
        let mut options = Options::default();
        options.create_if_missing(true);
        options.optimize_for_point_lookup(5000);
        options.set_allow_mmap_reads(true);
        options.set_allow_mmap_writes(true);

        let rocks_db = DB::open_with_ttl(&options, database_file, ttl)
            .with_context(|| format!("open session database {}", database_file))?;

        Ok(Self {
            rocks_db,
            max_session_idle_duration_in_seconds: 60 * 20,
        })
    }

    pub fn try_get_session_items(&self, evolving_session_id: &u128) -> anyhow::Result<Vec<u64>> {
        use anyhow::Context;
        let serialized_session_id =
            bincode::serialize(&evolving_session_id).context("serialize session key")?;

        let bytes = self
            .rocks_db
            .get(&serialized_session_id)
            .context("read session database")?;

        let session_items: Vec<u64> = match bytes {
            Some(bytes) => {
                let payload: DBValue =
                    bincode::deserialize(&bytes).context("decode session payload")?;
                let now = self.get_seconds_since_epoch()?;
                let seconds_since_last_event =
                    now.checked_sub(payload.epoch_secs).unwrap_or(u64::MAX);
                if seconds_since_last_event <= self.max_session_idle_duration_in_seconds {
                    payload.session_items
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        };
        Ok(session_items)
    }

    pub fn try_update_session_items(
        &self,
        evolving_session_id: &u128,
        session_items: &[u64],
    ) -> anyhow::Result<()> {
        use anyhow::Context;
        let serialized_session_id =
            bincode::serialize(evolving_session_id).context("serialize session key")?;
        let now = self.get_seconds_since_epoch()?;
        let payload = DBValue {
            session_items: Vec::from(session_items),
            epoch_secs: now,
        };
        let bytes = bincode::serialize(&payload).context("serialize session payload")?;

        self.rocks_db
            .put(&serialized_session_id, &bytes)
            .context("write session database")?;
        Ok(())
    }

    fn get_seconds_since_epoch(&self) -> anyhow::Result<u64> {
        use anyhow::Context;
        Ok(SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .context("system clock precedes Unix epoch")?
            .as_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupt_records_return_errors_and_future_records_expire() {
        let path =
            std::env::temp_dir().join(format!("serenade-session-test-{}", std::process::id()));
        {
            let store =
                RocksDBSessionStore::try_new(path.to_str().unwrap(), Duration::from_secs(1800))
                    .unwrap();
            let key = bincode::serialize(&1u128).unwrap();
            store.rocks_db.put(&key, b"corrupt").unwrap();
            assert!(store
                .try_get_session_items(&1)
                .unwrap_err()
                .to_string()
                .contains("decode"));
            let payload = DBValue {
                session_items: vec![10],
                epoch_secs: u64::MAX,
            };
            store
                .rocks_db
                .put(&key, bincode::serialize(&payload).unwrap())
                .unwrap();
            assert!(store.try_get_session_items(&1).unwrap().is_empty());
            store.try_update_session_items(&1, &[10, 20]).unwrap();
            assert_eq!(store.try_get_session_items(&1).unwrap(), vec![10, 20]);
        }
        std::fs::remove_dir_all(path).unwrap();
    }
}
