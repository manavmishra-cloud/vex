//! Criterion benchmark for the flat index, the baseline against which
//! future HNSW improvements will be measured.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use vex::{Distance, FlatIndex, Index};

fn make_random_index(n: usize, d: usize, seed: u64) -> FlatIndex {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx = FlatIndex::with_capacity(d, Distance::L2, n);
    for _ in 0..n {
        let v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        idx.insert(&v).unwrap();
    }
    idx
}

fn bench_flat_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_search");
    let d = 128;
    for &n in &[1_000usize, 10_000, 50_000] {
        let idx = make_random_index(n, d, 0);
        let mut rng = StdRng::seed_from_u64(1);
        let q: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let hits = idx.search(black_box(&q), 10).unwrap();
                black_box(hits);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_flat_search);
criterion_main!(benches);
