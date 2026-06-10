//! Compare flat (brute-force) vs HNSW index on a realistic workload.
//!
//! Run with: `cargo run --release --example hnsw_vs_flat`

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::time::Instant;
use vex::index::hnsw::{HnswIndex, HnswParams};
use vex::{Distance, FlatIndex, Index};

const N: usize = 20_000;
const D: usize = 128;
const K: usize = 10;
const N_QUERIES: usize = 200;

fn random_unit_vec(rng: &mut StdRng) -> Vec<f32> {
    let v: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / norm.max(1e-12)).collect()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let dataset: Vec<Vec<f32>> = (0..N).map(|_| random_unit_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|_| random_unit_vec(&mut rng)).collect();

    println!("=== vex: flat vs HNSW ===");
    println!("Dataset: {N} unit vectors in {D} dimensions");
    println!("Queries: {N_QUERIES}, top-{K} per query, L2 distance\n");

    // -------- FLAT --------
    let t0 = Instant::now();
    let mut flat = FlatIndex::with_capacity(D, Distance::L2, N);
    for v in &dataset {
        flat.insert(v).unwrap();
    }
    let flat_build = t0.elapsed();

    let t = Instant::now();
    let flat_results: Vec<HashSet<u64>> = queries
        .iter()
        .map(|q| {
            flat.search(q, K)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();
    let flat_query = t.elapsed();

    println!(
        "FLAT  build={:>7.2?}  query={:>7.2?} total  ({:>6.3} ms/query)",
        flat_build,
        flat_query,
        flat_query.as_secs_f64() * 1000.0 / N_QUERIES as f64
    );

    // -------- HNSW --------
    // ef_search controls recall-vs-latency tradeoff at query time;
    // 200 gives high recall on this 20k x 128d workload while still
    // being substantially faster than the flat brute-force scan.
    let params = HnswParams {
        ef_search: 200,
        ..HnswParams::default()
    };
    let t0 = Instant::now();
    let mut hnsw = HnswIndex::new(D, Distance::L2, params);
    for v in &dataset {
        hnsw.insert(v).unwrap();
    }
    let hnsw_build = t0.elapsed();

    let t = Instant::now();
    let hnsw_results: Vec<HashSet<u64>> = queries
        .iter()
        .map(|q| {
            hnsw.search(q, K)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();
    let hnsw_query = t.elapsed();

    println!(
        "HNSW  build={:>7.2?}  query={:>7.2?} total  ({:>6.3} ms/query)",
        hnsw_build,
        hnsw_query,
        hnsw_query.as_secs_f64() * 1000.0 / N_QUERIES as f64
    );

    // -------- RECALL --------
    let mut recall_sum = 0.0;
    for (f, h) in flat_results.iter().zip(hnsw_results.iter()) {
        let inter = f.intersection(h).count();
        recall_sum += inter as f64 / K as f64;
    }
    let recall = recall_sum / N_QUERIES as f64;

    let speedup = flat_query.as_secs_f64() / hnsw_query.as_secs_f64();
    println!("\nRECALL@{K} of HNSW vs flat ground truth: {:.3}", recall);
    println!("QUERY SPEEDUP (flat / HNSW)              : {:.1}x", speedup);
}
