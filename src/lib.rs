//! `vex` — a vector database from scratch in Rust.
//!
//! Vex is an educational, performance-oriented vector database that
//! implements approximate nearest-neighbor search from primitives.
//! It is designed to be:
//!
//! - **Readable.** Every algorithm is implemented in clean Rust with
//!   detailed comments — you can read the source and understand HNSW.
//! - **Fast enough to compare.** Honest benchmarks against Qdrant and
//!   FAISS, no benchmark games.
//! - **Visualizable.** A companion web UI lets you watch the HNSW
//!   graph being constructed and queries traversing it in real time.
//!
//! # Quick example
//!
//! ```
//! use vex::distance::Distance;
//! use vex::index::{Index, flat::FlatIndex};
//!
//! let mut index = FlatIndex::new(4, Distance::L2);
//! index.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
//! index.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
//! index.insert(&[0.0, 0.0, 1.0, 0.0]).unwrap();
//!
//! let hits = index.search(&[0.9, 0.1, 0.0, 0.0], 2).unwrap();
//! assert_eq!(hits[0].id, 0);
//! ```
//!
//! # Roadmap
//!
//! - **v0.1** (current): Flat brute-force index, three distance metrics, tests.
//! - **v0.2**: HNSW index from scratch.
//! - **v0.3**: HTTP REST API + on-disk persistence.
//! - **v0.4**: Benchmarks against Qdrant and FAISS.
//! - **v0.5**: Interactive web visualizer (HNSW graph + animated queries).
//! - **v0.6**: SIMD-accelerated distance kernels, quantization (PQ, scalar).
//!
//! See the project README for the full plan.

pub mod distance;
pub mod error;
pub mod index;
pub mod server;
pub mod storage;

pub use distance::Distance;
pub use error::{Result, VexError};
pub use index::{
    flat::FlatIndex,
    hnsw::{HnswIndex, HnswParams},
    Index, SearchResult,
};
