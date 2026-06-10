//! Flat (brute-force) index — the reference implementation.
//!
//! `FlatIndex` keeps every inserted vector in a contiguous `Vec<f32>` and
//! computes the distance from a query to every stored vector at search
//! time. This is `O(n · d)` per query and is the slowest possible index,
//! but it returns **exact** nearest neighbors and is therefore the
//! gold-standard reference we use to measure the recall of approximate
//! indices like HNSW.
//!
//! # Example
//!
//! ```
//! use vex::distance::Distance;
//! use vex::index::{Index, flat::FlatIndex};
//!
//! let mut idx = FlatIndex::new(3, Distance::L2);
//! idx.insert(&[1.0, 0.0, 0.0]).unwrap();
//! idx.insert(&[0.0, 1.0, 0.0]).unwrap();
//! idx.insert(&[0.0, 0.0, 1.0]).unwrap();
//!
//! let hits = idx.search(&[0.9, 0.1, 0.0], 1).unwrap();
//! assert_eq!(hits[0].id, 0);
//! ```

use serde::{Deserialize, Serialize};

use super::{Index, SearchResult};
use crate::distance::Distance;
use crate::error::{Result, VexError};

/// A flat (brute-force) vector index.
///
/// All vectors are stored interleaved in a single `Vec<f32>` of length
/// `n * dim`. Vector `i` occupies the slice `[i*dim .. (i+1)*dim]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex {
    dim: usize,
    metric: Distance,
    data: Vec<f32>, // flattened: n * dim
    n: usize,       // number of vectors stored
}

impl FlatIndex {
    /// Create an empty flat index for vectors of `dim` dimensions, using
    /// `metric` for distance comparisons.
    pub fn new(dim: usize, metric: Distance) -> Self {
        Self {
            dim,
            metric,
            data: Vec::new(),
            n: 0,
        }
    }

    /// Pre-allocate storage for `capacity` vectors. Optional optimization.
    pub fn with_capacity(dim: usize, metric: Distance, capacity: usize) -> Self {
        Self {
            dim,
            metric,
            data: Vec::with_capacity(capacity * dim),
            n: 0,
        }
    }

    /// View the stored vector with the given `id`, or `None` if it doesn't exist.
    pub fn get(&self, id: u64) -> Option<&[f32]> {
        let i = id as usize;
        if i >= self.n {
            return None;
        }
        Some(&self.data[i * self.dim..(i + 1) * self.dim])
    }

    /// The metric this index uses.
    pub fn metric(&self) -> Distance {
        self.metric
    }
}

impl Index for FlatIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.n
    }

    fn insert(&mut self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dim {
            return Err(VexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let id = self.n as u64;
        self.data.extend_from_slice(vector);
        self.n += 1;
        Ok(id)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VexError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if self.n == 0 {
            return Err(VexError::EmptyIndex);
        }
        if k > self.n {
            return Err(VexError::NotEnoughPoints {
                requested: k,
                available: self.n,
            });
        }

        // Compute distance to every stored vector. For a flat index this
        // is O(n·d) and we make no attempt to short-circuit it; the
        // important thing is *exactness* — we'll compare HNSW recall
        // against these results.
        let mut hits: Vec<SearchResult> = (0..self.n)
            .map(|i| {
                let v = &self.data[i * self.dim..(i + 1) * self.dim];
                SearchResult {
                    id: i as u64,
                    distance: self.metric.compute(query, v),
                }
            })
            .collect();

        // Partial sort: only the first k elements need to be ordered.
        hits.select_nth_unstable_by(k - 1, |a, b| a.distance.partial_cmp(&b.distance).unwrap());
        hits.truncate(k);
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn empty_index_errors_on_search() {
        let idx = FlatIndex::new(3, Distance::L2);
        let err = idx.search(&[0.0, 0.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, VexError::EmptyIndex));
    }

    #[test]
    fn dim_mismatch_errors_on_insert() {
        let mut idx = FlatIndex::new(3, Distance::L2);
        let err = idx.insert(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            VexError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn nearest_neighbor_in_3d() {
        let mut idx = FlatIndex::new(3, Distance::L2);
        idx.insert(&[1.0, 0.0, 0.0]).unwrap(); // id 0
        idx.insert(&[0.0, 1.0, 0.0]).unwrap(); // id 1
        idx.insert(&[0.0, 0.0, 1.0]).unwrap(); // id 2

        // Query is closest to id 0
        let hits = idx.search(&[0.9, 0.1, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].distance < hits[1].distance);
    }

    #[test]
    fn cosine_distance_ranks_correctly() {
        let mut idx = FlatIndex::new(2, Distance::Cosine);
        idx.insert(&[1.0, 0.0]).unwrap();
        idx.insert(&[0.0, 1.0]).unwrap();
        idx.insert(&[1.0, 1.0]).unwrap();

        let hits = idx.search(&[1.0, 0.0], 3).unwrap();
        // Identical vector first, then 45° (1,1), then orthogonal (0,1)
        assert_eq!(hits[0].id, 0);
        assert_eq!(hits[1].id, 2);
        assert_eq!(hits[2].id, 1);
        assert_abs_diff_eq!(hits[0].distance, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_ranks_largest_inner_product_first() {
        let mut idx = FlatIndex::new(2, Distance::DotProduct);
        idx.insert(&[1.0, 0.0]).unwrap();
        idx.insert(&[2.0, 0.0]).unwrap();
        idx.insert(&[3.0, 0.0]).unwrap();

        // dot-product "distance" is negated, so largest dot becomes smallest dist.
        let hits = idx.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(hits[0].id, 2);
        assert_eq!(hits[1].id, 1);
        assert_eq!(hits[2].id, 0);
    }

    #[test]
    fn insert_many_assigns_sequential_ids() {
        let mut idx = FlatIndex::new(2, Distance::L2);
        let vs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let ids = idx.insert_many(&vs).unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn requesting_too_many_neighbors_errors() {
        let mut idx = FlatIndex::new(2, Distance::L2);
        idx.insert(&[0.0, 0.0]).unwrap();
        let err = idx.search(&[1.0, 1.0], 5).unwrap_err();
        assert!(matches!(
            err,
            VexError::NotEnoughPoints {
                requested: 5,
                available: 1
            }
        ));
    }
}
