//! Error types for the vex library.

use thiserror::Error;

/// All errors that can be produced by the public API.
#[derive(Debug, Error)]
pub enum VexError {
    /// Vector inserted into the index had the wrong number of dimensions.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Number of dimensions the index was created with.
        expected: usize,
        /// Number of dimensions in the offending input vector.
        got: usize,
    },

    /// Querying an empty index — there are no points to return.
    #[error("index is empty; insert points before querying")]
    EmptyIndex,

    /// Requested more neighbors than the index contains.
    #[error("requested k={requested} neighbors but index only has {available} points")]
    NotEnoughPoints {
        /// Number of neighbors requested.
        requested: usize,
        /// Number of points in the index.
        available: usize,
    },

    /// An ID was requested that does not exist in the index.
    #[error("id {id} not found in index")]
    IdNotFound {
        /// The requested ID.
        id: u64,
    },
}

/// Result alias for fallible vex operations.
pub type Result<T> = std::result::Result<T, VexError>;
