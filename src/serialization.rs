//! Keep the bincode 1 on-disk format for snapshots, metadata, and session records.
use serde::{de::DeserializeOwned, Serialize};
use std::io::{Read, Write};

pub(crate) fn serialize<T: Serialize + ?Sized>(
    value: &T,
) -> Result<Vec<u8>, bincode::error::EncodeError> {
    bincode::serde::encode_to_vec(value, bincode::config::legacy())
}

pub(crate) fn deserialize<T: DeserializeOwned>(
    bytes: &[u8],
) -> Result<T, bincode::error::DecodeError> {
    bincode::serde::decode_from_slice(bytes, bincode::config::legacy()).map(|(value, _)| value)
}

pub(crate) fn serialize_into<T: Serialize + ?Sized>(
    mut writer: impl Write,
    value: &T,
) -> Result<(), bincode::error::EncodeError> {
    bincode::serde::encode_into_std_write(value, &mut writer, bincode::config::legacy()).map(|_| ())
}

pub(crate) fn deserialize_from<T: DeserializeOwned>(
    mut reader: impl Read,
) -> Result<T, bincode::error::DecodeError> {
    bincode::serde::decode_from_std_read(&mut reader, bincode::config::legacy())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_legacy_session_bytes() {
        // bincode 1 uses little-endian fixed-width integers, including lengths.
        let bytes = [2u64, 10, 20, 1234]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>();
        let payload = (vec![10u64, 20], 1234u64);
        assert_eq!(serialize(&payload).unwrap(), bytes);
        assert_eq!(deserialize::<(Vec<u64>, u64)>(&bytes).unwrap(), payload);
        let mut written = Vec::new();
        serialize_into(&mut written, &payload).unwrap();
        assert_eq!(written, bytes);
        assert_eq!(
            deserialize_from::<(Vec<u64>, u64)>(bytes.as_slice()).unwrap(),
            payload
        );
        assert_eq!(serialize(&1u128).unwrap(), 1u128.to_le_bytes());
    }
}
