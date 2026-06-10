# 🧭 vex

> A vector database from scratch in Rust, designed to be **readable, fast, and visually understandable**.

![Rust](https://img.shields.io/badge/Rust-1.96+-CE422B?logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in_development-yellow)
![Tests](https://github.com/manavmishra-cloud/vex/actions/workflows/tests.yml/badge.svg)

Vex implements approximate nearest-neighbor search from primitives, with a
focus on three things:

1. **Readability.** Every algorithm — distance kernels, HNSW indexing,
   query traversal — is implemented in clean Rust with detailed
   comments. You can read the source and understand exactly how a
   modern vector database works.
2. **Honest performance.** Vex is benchmarked head-to-head against
   Qdrant and FAISS on realistic recall/latency curves. No
   benchmark games.
3. **Visualizability.** A companion web UI lets you **watch the HNSW
   graph being built** and **animate queries** traversing the graph in
   real time. Genuinely educational and genuinely useful.

## Why another vector database?

Production vector databases like Pinecone, Qdrant, and Weaviate are
optimized for scale and operability. They are not optimized for
*understanding* — their code is large, their algorithms are hidden
behind abstractions, and you cannot watch a query happen. Vex inverts
those tradeoffs: it is a small, well-commented, single-author
implementation that you can read in an afternoon and *see* in motion.

## Quick example

```rust
use vex::{Distance, FlatIndex, Index};

let mut idx = FlatIndex::new(4, Distance::Cosine);
idx.insert(&[1.0, 0.0, 0.0, 0.0])?;
idx.insert(&[0.0, 1.0, 0.0, 0.0])?;
idx.insert(&[0.0, 0.0, 1.0, 0.0])?;

let hits = idx.search(&[0.9, 0.1, 0.0, 0.0], 2)?;
assert_eq!(hits[0].id, 0);
```

## Roadmap

| Version | Scope | Status |
|---------|-------|--------|
| **v0.1** | Flat brute-force index, 3 distance metrics, tests | ✅ released |
| **v0.2** | HNSW index from scratch (hierarchical layers, M-construction, query traversal) | ✅ released |
| **v0.3** | HTTP REST API (axum), bincode-based file persistence | ✅ released |
| **v0.4** | Benchmark suite vs Qdrant and FAISS — recall@10/100, p50/p99 latency | 🚧 next |
| **v0.5** | **Interactive web visualizer**: animated HNSW construction + query traversal | planned |
| **v0.6** | Hand-written SIMD distance kernels, scalar + product quantization | planned |
| **v0.7** | Concurrent writes, light sharding | exploratory |

## Project structure

```
vex/
├── Cargo.toml           # crate manifest
├── src/
│   ├── lib.rs           # public API entry point
│   ├── distance.rs      # cosine, L2, dot-product metrics
│   ├── error.rs         # VexError + Result alias
│   └── index/
│       ├── mod.rs       # Index trait, SearchResult
│       └── flat.rs      # brute-force baseline (v0.1)
├── tests/integration.rs # public-API integration tests
├── examples/basic.rs    # 10k vectors, timing summary
└── benches/search.rs    # criterion micro-benchmarks
```

## Running

```bash
# Build (release for honest perf numbers)
cargo build --release

# Tests
cargo test

# Examples
cargo run --release --example basic           # 10k vectors timing
cargo run --release --example hnsw_vs_flat    # head-to-head comparison

# HTTP server
cargo run --release --bin vex-server -- --bind 127.0.0.1:8080

# Criterion benchmarks (saves HTML report to target/criterion/)
cargo bench
```

## HTTP API

Start the server with `cargo run --release --bin vex-server` (default bind
is `127.0.0.1:8080`). All payloads are JSON.

### Create a collection

```bash
curl -X POST http://127.0.0.1:8080/collections -H 'Content-Type: application/json' -d '{
  "name": "my-vectors",
  "dim": 128,
  "metric": "Cosine",
  "index": "hnsw"
}'
```

### Insert vectors

```bash
curl -X POST http://127.0.0.1:8080/collections/my-vectors/points \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}'
# -> {"ids": [0, 1]}
```

### Search

```bash
curl -X POST http://127.0.0.1:8080/collections/my-vectors/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, ...], "k": 10}'
# -> {"hits": [{"id": 5, "distance": 0.012}, ...]}
```

### Save and load

```bash
# Save a collection to disk
curl -X POST http://127.0.0.1:8080/collections/my-vectors/save \
  -d '{"path": "/tmp/my-vectors.vex"}'

# Load a collection from disk into a new name
curl -X POST http://127.0.0.1:8080/collections/load \
  -d '{"name": "loaded", "path": "/tmp/my-vectors.vex"}'
```

Full endpoint list: `GET /health`, `GET /collections`, `POST /collections`,
`GET/DELETE /collections/:name`, `POST /collections/:name/points`,
`POST /collections/:name/search`, `POST /collections/:name/save`,
`POST /collections/load`.

Expected sample output from `cargo run --release --example basic`:

```
Building flat index with 10000 vectors of 128 dimensions ...
  insert: 10000 vectors in 142.3ms

Running 100 random queries, k=10 ...
  100 queries total: 81.2ms  (0.812 ms/query average)
```

(Exact numbers depend on your CPU. On a modern laptop a flat index over
10k×128d takes <1 ms per query — useful for correctness baselines and
small workloads, but linear in `n`. HNSW (v0.2) will be sub-millisecond
even at 1M vectors.)

## Design choices

### Storage layout

Vectors are stored as a flat `Vec<f32>` of length `n × dim`. Vector
`i` occupies the slice `[i*dim .. (i+1)*dim]`. This is cache-friendly
for sequential scans (the flat-index hot path) and lets us hand off
contiguous slices to SIMD kernels later without copying.

### Distance metrics

Three metrics are supported: cosine, squared-L2, and (negative)
dot-product. We use squared-L2 (no `sqrt`) for ranking — `sqrt` is
monotonic, so the *order* of nearest neighbors is preserved. This is
a common HFT / vector-DB optimization.

### Error handling

Public APIs return `Result<T, VexError>` with descriptive errors via
`thiserror`. Dimension mismatches, empty indices, and over-sized `k`
requests all produce structured errors rather than panics.

## Why these design priorities?

This project is deliberately scoped so that someone reading the source
can answer three questions clearly:

1. *How does HNSW actually work?* — by reading `src/index/hnsw.rs`
   (coming in v0.2).
2. *How does my vector DB choice compare on real workloads?* — by
   running the bench suite (v0.4).
3. *What does it look like when an HNSW query happens?* — by clicking
   through the web visualizer (v0.5).

If we ship those three, this becomes the vector DB everyone references
when they want to learn how vector search works, even if they deploy
something else in production.

## Citation

If you use vex in research or teaching:

```bibtex
@software{vex2026,
  author = {Mishra, Manav},
  title = {vex: A readable, visualizable vector database in Rust},
  year = {2026},
  url = {https://github.com/manavmishra-cloud/vex}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

Manav Mishra · [LinkedIn](https://linkedin.com/in/manav-mishra-23a26b308) · manavmishra260205@gmail.com
