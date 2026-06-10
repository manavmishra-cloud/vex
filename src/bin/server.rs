//! `vex-server` — the vex HTTP daemon.
//!
//! Run with: `cargo run --release --bin vex-server -- --bind 127.0.0.1:8080`
//!
//! Endpoints:
//!   GET    /health
//!   GET    /collections
//!   POST   /collections
//!   GET    /collections/:name
//!   DELETE /collections/:name
//!   POST   /collections/:name/points
//!   POST   /collections/:name/search
//!   POST   /collections/:name/save
//!   POST   /collections/load
//!
//! All payloads are JSON. The server keeps collections in memory only
//! unless you call `/save` and `/load` explicitly.

use std::net::SocketAddr;
use std::str::FromStr;

use tracing_subscriber::EnvFilter;
use vex::server::{router, AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    // Parse --bind <addr> argument; default 127.0.0.1:8080
    let mut bind_addr = "127.0.0.1:8080".to_string();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--bind" && i + 1 < args.len() {
            bind_addr = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }

    let addr = SocketAddr::from_str(&bind_addr)?;
    let app = router(AppState::default());

    tracing::info!("vex-server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
