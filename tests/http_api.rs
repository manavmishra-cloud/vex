//! End-to-end integration test for the HTTP API.
//!
//! Spins up the axum server on an ephemeral port, then hits the
//! endpoints with an HTTP client and verifies each contract.

use serde_json::json;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use vex::server::{router, AppState};

/// Bind a TCP listener to an OS-assigned port, return both the listener
/// and the local address it ended up on.
async fn bind_ephemeral() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

#[tokio::test]
async fn full_lifecycle_flat_index() {
    let (listener, addr) = bind_ephemeral().await;
    let state = AppState::default();
    tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });

    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // 1. /health
    let r = client.get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(r.status(), 200);

    // 2. POST /collections
    let r = client
        .post(format!("{base}/collections"))
        .json(&json!({
            "name": "test",
            "dim": 3,
            "metric": "L2",
            "index": "flat"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    // 3. POST /collections/test/points
    let r = client
        .post(format!("{base}/collections/test/points"))
        .json(&json!({
            "vectors": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["ids"], json!([0, 1, 2]));

    // 4. GET /collections/test
    let r = client
        .get(format!("{base}/collections/test"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["size"], 3);
    assert_eq!(body["dim"], 3);

    // 5. POST /collections/test/search
    let r = client
        .post(format!("{base}/collections/test/search"))
        .json(&json!({ "vector": [0.9, 0.1, 0.0], "k": 2 }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["id"], 0); // closest to (1,0,0)

    // 6. GET /collections — should list our collection
    let r = client
        .get(format!("{base}/collections"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body.as_array().unwrap().len(), 1);

    // 7. DELETE /collections/test
    let r = client
        .delete(format!("{base}/collections/test"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 204);

    // 8. GET /collections/test after delete -> 404
    let r = client
        .get(format!("{base}/collections/test"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn cannot_create_same_collection_twice() {
    let (listener, addr) = bind_ephemeral().await;
    let state = AppState::default();
    tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });

    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let body = json!({"name": "x", "dim": 2});
    let r1 = client
        .post(format!("{base}/collections"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 201);

    let r2 = client
        .post(format!("{base}/collections"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 409);
}

#[tokio::test]
async fn hnsw_index_path_works_via_http() {
    let (listener, addr) = bind_ephemeral().await;
    let state = AppState::default();
    tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });

    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let r = client
        .post(format!("{base}/collections"))
        .json(&json!({"name": "h", "dim": 4, "metric": "L2", "index": "hnsw"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    let r = client
        .post(format!("{base}/collections/h/points"))
        .json(&json!({
            "vectors": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    let r = client
        .post(format!("{base}/collections/h/search"))
        .json(&json!({"vector": [0.9, 0.1, 0.0, 0.0], "k": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["hits"][0]["id"], 0);
}
