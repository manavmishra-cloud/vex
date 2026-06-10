//! On-disk persistence for vex indices.
//!
//! Indices are serialized with [`bincode`] (compact binary format) for
//! speed and storage efficiency. The file layout is just a `bincode`-
//! serialized `IndexFile`:
//!
//! ```text
//! [magic: 4 bytes "VEXF"]
//! [version: u16]
//! [payload: bincode(IndexFile)]
//! ```
//!
//! The magic + version bytes let us evolve the format later without
//! silently misinterpreting old files.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::index::flat::FlatIndex;
use crate::index::hnsw::HnswIndex;

const MAGIC: &[u8; 4] = b"VEXF";
const VERSION: u16 = 1;

/// An on-disk index file. The single source of truth for what gets
/// persisted; new index variants are added here. The size mismatch
/// between variants is intentional — see `server::AnyIndex`.
#[derive(Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum IndexFile {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
}

/// Errors specific to on-disk persistence.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic bytes; not a vex index file")]
    BadMagic,
    #[error("unsupported file version: got {got}, expected {expected}")]
    UnsupportedVersion { got: u16, expected: u16 },
    #[error("serialization: {0}")]
    Serialize(Box<bincode::ErrorKind>),
}

impl From<Box<bincode::ErrorKind>> for StorageError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        StorageError::Serialize(e)
    }
}

pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Save an index to disk.
pub fn save<P: AsRef<Path>>(path: P, index: &IndexFile) -> StorageResult<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    bincode::serialize_into(&mut w, index)?;
    w.flush()?;
    Ok(())
}

/// Load an index from disk.
pub fn load<P: AsRef<Path>>(path: P) -> StorageResult<IndexFile> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(StorageError::BadMagic);
    }

    let mut version_bytes = [0u8; 2];
    r.read_exact(&mut version_bytes)?;
    let version = u16::from_le_bytes(version_bytes);
    if version != VERSION {
        return Err(StorageError::UnsupportedVersion {
            got: version,
            expected: VERSION,
        });
    }

    let idx: IndexFile = bincode::deserialize_from(&mut r)?;
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Distance;
    use crate::index::hnsw::HnswParams;
    use crate::index::Index;
    use tempfile::TempDir;

    #[test]
    fn flat_index_round_trip() {
        let mut idx = FlatIndex::new(3, Distance::L2);
        idx.insert(&[1.0, 2.0, 3.0]).unwrap();
        idx.insert(&[4.0, 5.0, 6.0]).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("flat.vex");
        save(&path, &IndexFile::Flat(idx)).unwrap();

        let loaded = load(&path).unwrap();
        match loaded {
            IndexFile::Flat(f) => {
                assert_eq!(f.len(), 2);
                let hits = f.search(&[1.0, 2.0, 3.0], 1).unwrap();
                assert_eq!(hits[0].id, 0);
            }
            _ => panic!("expected Flat"),
        }
    }

    #[test]
    fn hnsw_index_round_trip() {
        let mut idx = HnswIndex::new(4, Distance::L2, HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.insert(&[0.0, 0.0, 1.0, 0.0]).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("hnsw.vex");
        save(&path, &IndexFile::Hnsw(idx)).unwrap();

        let loaded = load(&path).unwrap();
        match loaded {
            IndexFile::Hnsw(h) => {
                assert_eq!(h.len(), 3);
                let hits = h.search(&[0.9, 0.1, 0.0, 0.0], 1).unwrap();
                assert_eq!(hits[0].id, 0);
            }
            _ => panic!("expected Hnsw"),
        }
    }

    #[test]
    fn bad_magic_is_detected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.vex");
        std::fs::write(&path, b"NOTVEX_garbage").unwrap();
        let err = load(&path).unwrap_err();
        assert!(matches!(err, StorageError::BadMagic));
    }
}
