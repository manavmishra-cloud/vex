//! Basic usage example: insert 10k random vectors and run a few queries.
//!
//! Run with: `cargo run --release --example basic`

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::time::Instant;
use vex::{Distance, FlatIndex, Index};

fn main() {
    const N: usize = 10_000;
    const D: usize = 128;
    const K: usize = 10;

    println!("Building flat index with {N} vectors of {D} dimensions ...");

    let mut rng = StdRng::seed_from_u64(0);
    let mut idx = FlatIndex::with_capacity(D, Distance::Cosine, N);

    let t0 = Instant::now();
    for _ in 0..N {
        let v: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
        idx.insert(&v).unwrap();
    }
    println!("  insert: {N} vectors in {:.2?}", t0.elapsed());

    println!("\nRunning 100 random queries, k={K} ...");
    let mut total_query_time = std::time::Duration::ZERO;

    for _ in 0..100 {
        let q: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let t = Instant::now();
        let hits = idx.search(&q, K).unwrap();
        total_query_time += t.elapsed();
        // Sanity-check: hits sorted ascending by distance
        for w in hits.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    println!(
        "  100 queries total: {:.2?}  ({:.3} ms/query average)",
        total_query_time,
        total_query_time.as_secs_f64() * 1000.0 / 100.0
    );
}
