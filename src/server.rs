//! HTTP REST API for vex.
//!
//! Built on [`axum`]. Exposes endpoints to create collections, insert
//! vectors, search nearest neighbors, save/load to disk, and inspect
//! state. State is held in-process behind an `Arc<RwLock<_>>`.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::distance::Distance;
use crate::index::flat::FlatIndex;
use crate::index::hnsw::{HnswIndex, HnswParams};
use crate::index::Index;
use crate::storage::{self, IndexFile};

/// Enum dispatch over the two index variants.
///
/// We deliberately do not box the larger HNSW variant — the index lives
/// behind an `RwLock` in `AppState`, the enum itself is never moved
/// after construction, and adding a heap indirection for every access
/// would hurt query throughput far more than the size mismatch hurts
/// memory layout.
#[derive(Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum AnyIndex {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
}

impl AnyIndex {
    pub fn dim(&self) -> usize {
        match self {
            AnyIndex::Flat(f) => f.dim(),
            AnyIndex::Hnsw(h) => h.dim(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            AnyIndex::Flat(f) => f.len(),
            AnyIndex::Hnsw(h) => h.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn insert(&mut self, v: &[f32]) -> crate::error::Result<u64> {
        match self {
            AnyIndex::Flat(f) => f.insert(v),
            AnyIndex::Hnsw(h) => h.insert(v),
        }
    }
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
    ) -> crate::error::Result<Vec<crate::index::SearchResult>> {
        match self {
            AnyIndex::Flat(f) => f.search(q, k),
            AnyIndex::Hnsw(h) => h.search(q, k),
        }
    }
}

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub collections: Arc<RwLock<HashMap<String, AnyIndex>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// Build the full router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route(
            "/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/collections/:name",
            get(get_collection).delete(drop_collection),
        )
        .route("/collections/:name/points", post(insert_points))
        .route("/collections/:name/search", post(search))
        .route("/collections/:name/save", post(save_collection))
        .route("/collections/load", post(load_collection))
        .with_state(state)
}

// ----------------------------------------------------------------------
// Error handling: any error becomes a JSON {error: ...} body with a 4xx/5xx.
// ----------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
}

#[derive(Debug)]
struct ApiErrResp(StatusCode, String);

impl IntoResponse for ApiErrResp {
    fn into_response(self) -> Response {
        (self.0, Json(ApiError { error: self.1 })).into_response()
    }
}

// ----------------------------------------------------------------------
// Handlers
// ----------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct HealthResp {
    status: &'static str,
}

async fn health() -> Json<HealthResp> {
    Json(HealthResp { status: "ok" })
}

#[derive(Debug, Deserialize)]
struct CreateCollectionReq {
    name: String,
    dim: usize,
    #[serde(default = "default_metric")]
    metric: Distance,
    #[serde(default = "default_index_kind")]
    index: IndexKind,
    #[serde(default)]
    hnsw: Option<HnswParams>,
}

fn default_metric() -> Distance {
    Distance::Cosine
}
fn default_index_kind() -> IndexKind {
    IndexKind::Flat
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum IndexKind {
    Flat,
    Hnsw,
}

#[derive(Debug, Serialize)]
struct CollectionInfo {
    name: String,
    dim: usize,
    size: usize,
}

async fn list_collections(State(state): State<AppState>) -> Json<Vec<CollectionInfo>> {
    let map = state.collections.read().await;
    let mut out: Vec<_> = map
        .iter()
        .map(|(name, idx)| CollectionInfo {
            name: name.clone(),
            dim: idx.dim(),
            size: idx.len(),
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Json(out)
}

async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionReq>,
) -> Result<StatusCode, ApiErrResp> {
    let mut map = state.collections.write().await;
    if map.contains_key(&req.name) {
        return Err(ApiErrResp(
            StatusCode::CONFLICT,
            format!("collection '{}' already exists", req.name),
        ));
    }
    let idx = match req.index {
        IndexKind::Flat => AnyIndex::Flat(FlatIndex::new(req.dim, req.metric)),
        IndexKind::Hnsw => AnyIndex::Hnsw(HnswIndex::new(
            req.dim,
            req.metric,
            req.hnsw.unwrap_or_default(),
        )),
    };
    map.insert(req.name, idx);
    Ok(StatusCode::CREATED)
}

async fn get_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, ApiErrResp> {
    let map = state.collections.read().await;
    let idx = map
        .get(&name)
        .ok_or_else(|| ApiErrResp(StatusCode::NOT_FOUND, format!("no collection '{name}'")))?;
    Ok(Json(CollectionInfo {
        name,
        dim: idx.dim(),
        size: idx.len(),
    }))
}

async fn drop_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiErrResp> {
    let mut map = state.collections.write().await;
    if map.remove(&name).is_none() {
        return Err(ApiErrResp(
            StatusCode::NOT_FOUND,
            format!("no collection '{name}'"),
        ));
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
struct InsertReq {
    vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct InsertResp {
    ids: Vec<u64>,
}

async fn insert_points(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<InsertReq>,
) -> Result<(StatusCode, Json<InsertResp>), ApiErrResp> {
    let mut map = state.collections.write().await;
    let idx = map
        .get_mut(&name)
        .ok_or_else(|| ApiErrResp(StatusCode::NOT_FOUND, format!("no collection '{name}'")))?;

    let mut ids = Vec::with_capacity(req.vectors.len());
    for v in &req.vectors {
        let id = idx
            .insert(v)
            .map_err(|e| ApiErrResp(StatusCode::BAD_REQUEST, e.to_string()))?;
        ids.push(id);
    }
    Ok((StatusCode::CREATED, Json(InsertResp { ids })))
}

#[derive(Debug, Deserialize)]
struct SearchReq {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct HitJson {
    id: u64,
    distance: f32,
}

#[derive(Debug, Serialize)]
struct SearchResp {
    hits: Vec<HitJson>,
}

async fn search(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SearchReq>,
) -> Result<Json<SearchResp>, ApiErrResp> {
    let map = state.collections.read().await;
    let idx = map
        .get(&name)
        .ok_or_else(|| ApiErrResp(StatusCode::NOT_FOUND, format!("no collection '{name}'")))?;
    let hits = idx
        .search(&req.vector, req.k)
        .map_err(|e| ApiErrResp(StatusCode::BAD_REQUEST, e.to_string()))?;
    let body = SearchResp {
        hits: hits
            .into_iter()
            .map(|h| HitJson {
                id: h.id,
                distance: h.distance,
            })
            .collect(),
    };
    Ok(Json(body))
}

#[derive(Debug, Deserialize)]
struct SaveReq {
    path: String,
}

async fn save_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SaveReq>,
) -> Result<StatusCode, ApiErrResp> {
    let map = state.collections.read().await;
    let idx = map
        .get(&name)
        .ok_or_else(|| ApiErrResp(StatusCode::NOT_FOUND, format!("no collection '{name}'")))?;
    let file = match idx {
        AnyIndex::Flat(f) => IndexFile::Flat(f.clone()),
        AnyIndex::Hnsw(_) => {
            return Err(ApiErrResp(
                StatusCode::NOT_IMPLEMENTED,
                "HNSW save in-place requires Clone; use a fresh load/save cycle".into(),
            ));
        }
    };
    storage::save(&req.path, &file)
        .map_err(|e| ApiErrResp(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(StatusCode::OK)
}

#[derive(Debug, Deserialize)]
struct LoadReq {
    name: String,
    path: String,
}

async fn load_collection(
    State(state): State<AppState>,
    Json(req): Json<LoadReq>,
) -> Result<StatusCode, ApiErrResp> {
    let file =
        storage::load(&req.path).map_err(|e| ApiErrResp(StatusCode::BAD_REQUEST, e.to_string()))?;
    let idx = match file {
        IndexFile::Flat(f) => AnyIndex::Flat(f),
        IndexFile::Hnsw(h) => AnyIndex::Hnsw(h),
    };
    let mut map = state.collections.write().await;
    map.insert(req.name, idx);
    Ok(StatusCode::CREATED)
}
