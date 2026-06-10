//! Distance and similarity metrics for vector queries.
//!
//! Three metrics are provided:
//!
//! - [`Distance::Cosine`]:    cosine *distance*, in `[0, 2]` (lower is more similar).
//!   Normalized vectors recover `1 - dot(a, b)` exactly.
//! - [`Distance::L2`]:        Euclidean L2 *squared* distance (we never need the sqrt
//!   for ranking; nearest-by-L2-squared is the same as nearest-by-L2).
//! - [`Distance::DotProduct`]:    negative dot product. For maximum inner product
//!   search (MIPS), use this and the smallest "distance" is the best match.
//!
//! All metrics are implemented as plain Rust today. Hand-written SIMD via
//! `std::simd` is a planned v0.3 optimization.

use serde::{Deserialize, Serialize};

/// Distance metric used by an index for similarity comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Distance {
    /// Cosine distance: `1 - cosine_similarity(a, b)`. Range `[0, 2]`.
    Cosine,
    /// Squared Euclidean distance (no `sqrt`). Range `[0, +∞)`.
    L2,
    /// Negative dot product. Use for maximum inner product search.
    DotProduct,
}

impl Distance {
    /// Compute the metric between two vectors of equal dimensionality.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`. This is a programming error; the
    /// public API of [`crate::index`] validates dimensions before calling.
    #[inline]
    pub fn compute(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dim mismatch");
        match self {
            Distance::Cosine => cosine_distance(a, b),
            Distance::L2 => l2_squared(a, b),
            Distance::DotProduct => -dot(a, b),
        }
    }
}

/// Squared Euclidean distance.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Cosine distance: `1 - cos_sim`.
///
/// Returns 0.0 if either vector has zero norm (defined-but-degenerate case).
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot_ab, mut norm_a, mut norm_b) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        dot_ab += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - dot_ab / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn dot_basic() {
        assert_abs_diff_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn l2_squared_basic() {
        assert_abs_diff_eq!(l2_squared(&[1.0, 2.0], &[4.0, 6.0]), 25.0);
    }

    #[test]
    fn cosine_identical_vectors_are_zero() {
        assert_abs_diff_eq!(
            cosine_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn cosine_orthogonal_vectors_are_one() {
        assert_abs_diff_eq!(
            cosine_distance(&[1.0, 0.0], &[0.0, 1.0]),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn cosine_zero_vector_is_zero() {
        // Degenerate: defined-but-degenerate. We return 0 to keep ranking well-defined.
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn distance_enum_dispatch() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert_abs_diff_eq!(Distance::Cosine.compute(&a, &b), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(Distance::L2.compute(&a, &b), 2.0);
        assert_abs_diff_eq!(Distance::DotProduct.compute(&a, &b), 0.0);
    }
}
