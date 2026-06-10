//! Index implementations for nearest-neighbor search.
//!
//! Today `vex` ships a [`flat::FlatIndex`] — a brute-force baseline that
//! computes the distance from a query to every stored point. It is the
//! reference implementation we compare every other index against for
//! correctness.
//!
//! The Hierarchical Navigable Small World (HNSW) index lands in v0.2.

pub mod flat;
pub mod hnsw;

use crate::error::Result;

/// A search hit: an index ID paired with the distance from the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// Index-assigned ID for the stored vector.
    pub id: u64,
    /// Distance from the query under the index's configured metric.
    pub distance: f32,
}

/// A trait every index implementation must satisfy.
///
/// Concrete implementations: [`flat::FlatIndex`].
pub trait Index {
    /// Dimensionality of the vectors this index stores.
    fn dim(&self) -> usize;

    /// Number of vectors currently in the index.
    fn len(&self) -> usize;

    /// True if no vectors have been inserted yet.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a single vector. Returns the assigned ID.
    fn insert(&mut self, vector: &[f32]) -> Result<u64>;

    /// Insert multiple vectors. Returns the assigned IDs in input order.
    fn insert_many(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(vectors.len());
        for v in vectors {
            ids.push(self.insert(v)?);
        }
        Ok(ids)
    }

    /// Return the top-`k` nearest neighbors of `query`, sorted ascending by
    /// distance (closest first).
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
}
