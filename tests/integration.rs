//! Integration tests: full end-to-end correctness of the public API.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use vex::{Distance, FlatIndex, Index};

/// Generate `n` random unit vectors in `d` dimensions for repeatable testing.
fn random_unit_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm.max(1e-12)).collect()
        })
        .collect()
}

#[test]
fn flat_index_round_trip_1000_vectors_128d() {
    let vectors = random_unit_vectors(1000, 128, 42);
    let mut idx = FlatIndex::with_capacity(128, Distance::L2, 1000);
    let ids = idx.insert_many(&vectors).unwrap();

    assert_eq!(ids.len(), 1000);
    assert_eq!(idx.len(), 1000);

    // Query each inserted vector — itself should be the nearest neighbor (distance ~= 0).
    for (i, v) in vectors.iter().enumerate() {
        let hits = idx.search(v, 1).unwrap();
        assert_eq!(hits[0].id as usize, i, "self-query failed for id {i}");
        assert!(hits[0].distance < 1e-6, "self-distance not ~0 for id {i}");
    }
}

#[test]
fn flat_index_cosine_top_k_matches_known_answer() {
    let mut idx = FlatIndex::new(3, Distance::Cosine);
    // Construct a deliberate ordering:
    idx.insert(&[1.0, 0.0, 0.0]).unwrap(); // id 0: same as query
    idx.insert(&[1.0, 0.1, 0.0]).unwrap(); // id 1: very close
    idx.insert(&[1.0, 1.0, 0.0]).unwrap(); // id 2: 45 deg
    idx.insert(&[0.0, 0.0, 1.0]).unwrap(); // id 3: orthogonal

    let hits = idx.search(&[1.0, 0.0, 0.0], 4).unwrap();
    assert_eq!(hits[0].id, 0);
    assert_eq!(hits[1].id, 1);
    assert_eq!(hits[2].id, 2);
    assert_eq!(hits[3].id, 3);
}

#[test]
fn get_returns_stored_vector() {
    let mut idx = FlatIndex::new(3, Distance::L2);
    let id = idx.insert(&[1.0, 2.0, 3.0]).unwrap();
    let v = idx.get(id).unwrap();
    assert_eq!(v, &[1.0, 2.0, 3.0]);
}
