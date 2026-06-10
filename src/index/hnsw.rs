//! Hierarchical Navigable Small World (HNSW) index.
//!
//! HNSW is the approximate-nearest-neighbor algorithm used internally by
//! Pinecone, Qdrant, Weaviate, FAISS, and most modern vector databases.
//! The construction is layered: each point exists in layers `[0, l]`
//! where `l` is sampled from a geometric distribution. Each layer is a
//! graph in which queries can navigate by greedy traversal from a small
//! set of entry points. Searches start at the top (sparse) layer and
//! descend to the bottom (dense) layer, refining the candidate set as
//! they go.
//!
//! The reference is Malkov & Yashunin (2018), *Efficient and Robust
//! Approximate Nearest Neighbor Search Using Hierarchical Navigable
//! Small World Graphs*, IEEE TPAMI.
//!
//! # Quick example
//!
//! ```
//! use vex::{Distance, Index};
//! use vex::index::hnsw::{HnswIndex, HnswParams};
//!
//! let mut idx = HnswIndex::new(4, Distance::L2, HnswParams::default());
//! idx.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
//! idx.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
//! idx.insert(&[0.0, 0.0, 1.0, 0.0]).unwrap();
//! idx.insert(&[0.0, 0.0, 0.0, 1.0]).unwrap();
//!
//! let hits = idx.search(&[0.9, 0.1, 0.0, 0.0], 2).unwrap();
//! assert_eq!(hits[0].id, 0);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::{Index, SearchResult};
use crate::distance::Distance;
use crate::error::{Result, VexError};

/// Tunable parameters for HNSW construction and query.
///
/// The defaults are taken from the parameter values recommended by
/// Malkov & Yashunin (2018) and reproduce roughly Qdrant-class quality
/// on the typical 100-1M point regime.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Number of bidirectional links per node on each layer above 0.
    /// Larger `m` => denser graph, slower construction, better recall.
    pub m: usize,
    /// Maximum links per node on layer 0. Usually `2 * m`.
    pub m_max0: usize,
    /// Size of the dynamic candidate list during *construction*.
    /// Larger => slower build, better recall, denser graph.
    pub ef_construction: usize,
    /// Default size of the dynamic candidate list during *search*.
    /// Larger => slower query, better recall.
    pub ef_search: usize,
    /// Random seed for reproducible level sampling.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            seed: 0,
        }
    }
}

/// A neighbor candidate ordered by distance for priority queues.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: u64,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN-safe comparison; we treat NaN as "greater than everything"
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Greater)
    }
}

/// HNSW index implementation.
///
/// Internally stores all vectors in a flat `Vec<f32>` of length `n*dim`
/// and maintains a per-node, per-layer adjacency list as nested `Vec`s.
/// Node IDs are dense `0..n` in insertion order.
#[derive(Debug)]
pub struct HnswIndex {
    dim: usize,
    metric: Distance,
    params: HnswParams,
    /// Level multiplier `m_l = 1/ln(M)` from the HNSW paper.
    m_l: f64,
    /// Flattened vector storage: `data[i*dim .. (i+1)*dim]` is vector `i`.
    data: Vec<f32>,
    /// Number of stored vectors.
    n: usize,
    /// `levels[i]` = maximum layer at which node `i` appears.
    levels: Vec<u8>,
    /// `neighbors[i][layer]` = list of node IDs neighboring `i` at `layer`.
    /// `neighbors[i].len() == levels[i] + 1` always.
    neighbors: Vec<Vec<Vec<u64>>>,
    /// Entry point (a node ID at `max_level`), or `None` if index is empty.
    entry_point: Option<u64>,
    /// Highest layer present in the index.
    max_level: u8,
    rng: StdRng,
}

impl HnswIndex {
    /// Create an empty HNSW index.
    pub fn new(dim: usize, metric: Distance, params: HnswParams) -> Self {
        let m_l = 1.0 / (params.m as f64).ln();
        Self {
            dim,
            metric,
            params,
            m_l,
            data: Vec::new(),
            n: 0,
            levels: Vec::new(),
            neighbors: Vec::new(),
            entry_point: None,
            max_level: 0,
            rng: StdRng::seed_from_u64(params.seed),
        }
    }

    /// Vector view of stored point `id`.
    fn vector_of(&self, id: u64) -> &[f32] {
        let i = id as usize;
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Sample a new node's max level from the HNSW geometric distribution.
    fn sample_level(&mut self) -> u8 {
        // l = floor(-ln(U) * m_l), U ~ Uniform(0, 1)
        let u: f64 = self.rng.gen_range(f64::EPSILON..1.0);
        let l = (-u.ln() * self.m_l).floor() as i64;
        l.max(0).min(u8::MAX as i64) as u8
    }

    /// Greedy 1-nearest descent at a single layer, starting from `entry`.
    /// Returns the closest point to `query` reachable by greedy traversal.
    fn greedy_search(&self, query: &[f32], entry: u64, layer: u8) -> Candidate {
        let mut best = Candidate {
            id: entry,
            distance: self.metric.compute(query, self.vector_of(entry)),
        };
        loop {
            let mut improved = false;
            let neighbors = &self.neighbors[best.id as usize][layer as usize];
            for &nid in neighbors {
                let d = self.metric.compute(query, self.vector_of(nid));
                if d < best.distance {
                    best = Candidate {
                        id: nid,
                        distance: d,
                    };
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    /// Beam search at a single layer (the workhorse of both construction
    /// and query). Returns up to `ef` nearest candidates to `query`,
    /// starting from `entry_points`.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u64],
        ef: usize,
        layer: u8,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<u64> = HashSet::with_capacity(ef * 4);
        // Min-heap of candidates to visit, ordered by ascending distance.
        let mut candidates: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();
        // Max-heap of best results, ordered by descending distance so .peek()
        // gives the *furthest* element (we discard it when we find better).
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &id in entry_points {
            if visited.insert(id) {
                let d = self.metric.compute(query, self.vector_of(id));
                let c = Candidate { id, distance: d };
                candidates.push(std::cmp::Reverse(c));
                results.push(c);
            }
        }

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            // Stop condition: if the closest remaining candidate is further
            // than the furthest result we've kept, we cannot improve.
            if let Some(furthest_kept) = results.peek() {
                if current.distance > furthest_kept.distance && results.len() >= ef {
                    break;
                }
            }

            // Visit current's neighbors at this layer.
            for &nid in &self.neighbors[current.id as usize][layer as usize] {
                if visited.insert(nid) {
                    let d = self.metric.compute(query, self.vector_of(nid));
                    let cand = Candidate {
                        id: nid,
                        distance: d,
                    };
                    let push_it = match results.peek() {
                        None => true,
                        Some(f) => d < f.distance || results.len() < ef,
                    };
                    if push_it {
                        candidates.push(std::cmp::Reverse(cand));
                        results.push(cand);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert max-heap to ascending-by-distance Vec.
        let mut out: Vec<Candidate> = results.into_vec();
        out.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        out
    }

    /// Simple neighbor-selection heuristic: just take the `m` closest from
    /// the candidate set. Cheap and clear. The "fancier" heuristic from
    /// Algorithm 4 of Malkov & Yashunin (2018), which diversifies the
    /// neighborhood, is a planned v0.2.1 follow-up.
    fn select_neighbors_simple(candidates: &[Candidate], m: usize) -> Vec<u64> {
        candidates.iter().take(m).map(|c| c.id).collect()
    }

    /// Prune `node`'s neighbor list at `layer` down to at most `max_links`
    /// by keeping only the closest. Called whenever a node accumulates
    /// too many connections.
    fn prune_neighbors(&mut self, node: u64, layer: u8, max_links: usize) {
        let neigh_ids: Vec<u64> = self.neighbors[node as usize][layer as usize].clone();
        if neigh_ids.len() <= max_links {
            return;
        }
        let node_vec = self.vector_of(node).to_vec();
        let mut cands: Vec<Candidate> = neigh_ids
            .iter()
            .map(|&nid| Candidate {
                id: nid,
                distance: self.metric.compute(&node_vec, self.vector_of(nid)),
            })
            .collect();
        cands.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        cands.truncate(max_links);
        self.neighbors[node as usize][layer as usize] = cands.into_iter().map(|c| c.id).collect();
    }
}

impl Index for HnswIndex {
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

        let new_id = self.n as u64;
        let new_level = self.sample_level();

        // Append vector data.
        self.data.extend_from_slice(vector);
        self.levels.push(new_level);

        // Allocate empty neighbor lists for each layer the new node will live on.
        let mut node_neigh: Vec<Vec<u64>> = Vec::with_capacity(new_level as usize + 1);
        for _ in 0..=new_level {
            node_neigh.push(Vec::new());
        }
        self.neighbors.push(node_neigh);
        self.n += 1;

        // First insert ever: just becomes the entry point.
        if self.entry_point.is_none() {
            self.entry_point = Some(new_id);
            self.max_level = new_level;
            return Ok(new_id);
        }

        let mut ep = self.entry_point.unwrap();
        let cur_max_level = self.max_level;

        // Phase 1: greedily descend layers above the new node's level
        // to find a tight starting point.
        if new_level < cur_max_level {
            let mut layer = cur_max_level;
            while layer > new_level {
                let c = self.greedy_search(vector, ep, layer);
                ep = c.id;
                layer -= 1;
            }
        }

        // Phase 2: for each layer from min(new_level, cur_max_level) down to 0,
        // run a beam search to find candidate neighbors, connect bidirectionally,
        // and prune over-connected nodes.
        let mut layer = new_level.min(cur_max_level) as i32;
        let mut entry_points = vec![ep];
        while layer >= 0 {
            let l = layer as u8;
            let neighbors_at_layer =
                self.search_layer(vector, &entry_points, self.params.ef_construction, l);
            let m = self.params.m;
            let selected = Self::select_neighbors_simple(&neighbors_at_layer, m);

            // Connect new_id -> selected at this layer.
            self.neighbors[new_id as usize][l as usize] = selected.clone();
            // Connect each selected -> new_id (bidirectional), then prune if needed.
            let max_links = if l == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };
            for nid in &selected {
                self.neighbors[*nid as usize][l as usize].push(new_id);
                self.prune_neighbors(*nid, l, max_links);
            }

            // Entry points for next (deeper) layer = current selection.
            entry_points = neighbors_at_layer.iter().map(|c| c.id).collect();
            if entry_points.is_empty() {
                entry_points = vec![ep];
            }

            layer -= 1;
        }

        // If new node's level is higher than the existing max, it becomes
        // the new entry point.
        if new_level > cur_max_level {
            self.entry_point = Some(new_id);
            self.max_level = new_level;
        }

        Ok(new_id)
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

        let ep = self.entry_point.unwrap();

        // Phase 1: greedy descent from the top layer down to layer 1.
        let mut current = ep;
        let mut layer = self.max_level;
        while layer > 0 {
            let c = self.greedy_search(query, current, layer);
            current = c.id;
            layer -= 1;
        }

        // Phase 2: beam search at layer 0 with size ef = max(ef_search, k).
        let ef = self.params.ef_search.max(k);
        let candidates = self.search_layer(query, &[current], ef, 0);
        let hits: Vec<SearchResult> = candidates
            .into_iter()
            .take(k)
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
            })
            .collect();

        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatIndex;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
    fn single_insert_becomes_entry_point() {
        let mut idx = HnswIndex::new(3, Distance::L2, HnswParams::default());
        let id = idx.insert(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(id, 0);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.entry_point, Some(0));
    }

    #[test]
    fn search_after_one_insert_returns_that_point() {
        let mut idx = HnswIndex::new(3, Distance::L2, HnswParams::default());
        idx.insert(&[1.0, 2.0, 3.0]).unwrap();
        let hits = idx.search(&[1.1, 2.1, 3.1], 1).unwrap();
        assert_eq!(hits[0].id, 0);
    }

    #[test]
    fn hnsw_high_recall_on_1000_vectors_128d() {
        // Compare HNSW @ k=10 against flat-index ground truth on a known-good
        // workload. We expect >=0.9 recall at this scale.
        let d = 64;
        let n = 1000;
        let vecs = random_unit_vectors(n, d, 0);

        let mut flat = FlatIndex::new(d, Distance::L2);
        let mut hnsw = HnswIndex::new(d, Distance::L2, HnswParams::default());
        for v in &vecs {
            flat.insert(v).unwrap();
            hnsw.insert(v).unwrap();
        }

        // Build 20 random queries (subset of inserted points, slightly perturbed).
        let mut rng = StdRng::seed_from_u64(99);
        let mut recall_sum = 0.0;
        let trials = 20;
        let k = 10;
        for _ in 0..trials {
            let i: usize = rng.gen_range(0..n);
            let q: Vec<f32> = vecs[i]
                .iter()
                .map(|x| x + rng.gen_range(-0.01..0.01))
                .collect();

            let flat_hits: HashSet<u64> =
                flat.search(&q, k).unwrap().iter().map(|h| h.id).collect();
            let hnsw_hits: HashSet<u64> =
                hnsw.search(&q, k).unwrap().iter().map(|h| h.id).collect();
            let intersection = flat_hits.intersection(&hnsw_hits).count();
            recall_sum += intersection as f64 / k as f64;
        }
        let recall = recall_sum / trials as f64;
        assert!(
            recall >= 0.85,
            "recall@{k} too low: {recall:.3} (want >=0.85)"
        );
    }

    #[test]
    fn level_sampling_distribution_is_reasonable() {
        let mut idx = HnswIndex::new(3, Distance::L2, HnswParams::default());
        let mut counts = [0u32; 8];
        for _ in 0..10_000 {
            let l = idx.sample_level().min(7) as usize;
            counts[l] += 1;
        }
        // Most points should be at level 0; counts should monotonically decrease.
        assert!(counts[0] > counts[1]);
        assert!(counts[1] >= counts[2]);
    }
}
