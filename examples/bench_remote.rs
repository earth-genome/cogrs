//! Remote-COG latency/throughput benchmark.
//!
//! Prints one JSON line per run. Run each scenario in a fresh process (the decompressed-tile
//! cache is process-wide), at least five times, and aggregate the lines.
//!
//! ```text
//! AWS_SKIP_SIGNATURE=true AWS_REGION=us-west-2 \
//!   cargo run --release --example bench_remote -- <scenario> <s3|https> [blocking_threads] [workers]
//! ```
//!
//! Scenarios:
//! * `single`          open + one z14 tile (cold)
//! * `z12`             open + one z12 tile
//! * `scene`           open + one low-zoom tile covering the whole scene (`BENCH_TILE=z/x/y`,
//!                     default 8/73/97); `transparent_pct` is the share of fill pixels
//! * `concurrent`      one shared reader, 64 concurrent distinct z14 tiles
//! * `concurrent_open` 64 concurrent requests that each open the COG and extract a z14 tile
//!
//! `BENCH_HINT=all` opens with `OverviewQualityHint::AllUsable` (no overview sampling at
//! open); `BENCH_TRACE=1` prints each network request (new code only). The same file builds
//! against the pre-async revision with `RUSTFLAGS="--cfg baseline"`, which is how the numbers
//! in the async-I/O change were compared. Requests and bytes are counted at the reader
//! (reads served from the open-time prefix are not requests).
#![allow(unexpected_cfgs)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cogrs::{CogReader, OverviewQualityHint, TileData, TileExtractor};

const KEY: &str = "2026-09-01_2026-10-01/18SUJ_2026-09-01_2026-10-01/TCI.tif";
const BUCKET: &str = "ei-imagery-sentinel2-prd";
const REGION: &str = "us-west-2";

fn url(kind: &str) -> String {
    match kind {
        "s3" => format!("s3://{BUCKET}/{KEY}"),
        "https" => format!("https://{BUCKET}.s3.{REGION}.amazonaws.com/{KEY}"),
        other => panic!("unknown source kind {other}"),
    }
}

/// `BENCH_HINT=all` skips the overview-quality sampling at open (a server would pass a stored
/// hint); the default samples tiles from the coarsest overview, like `CogReader::open_async`.
fn hint() -> OverviewQualityHint {
    match std::env::var("BENCH_HINT").as_deref() {
        Ok("all") => OverviewQualityHint::AllUsable,
        _ => OverviewQualityHint::ComputeAtRuntime,
    }
}

/// Counts the HTTP requests a reader makes (open request included).
struct Counters {
    requests: AtomicUsize,
    started: Instant,
}

impl Default for Counters {
    fn default() -> Self {
        Self { requests: AtomicUsize::new(0), started: Instant::now() }
    }
}

#[cfg(not(baseline))]
mod counted {
    use super::*;
    use bytes::Bytes;
    use cogrs::{AsyncRangeReader, IoOptions, ObjectStoreRangeReader};
    use futures::future::BoxFuture;

    struct Counting {
        inner: ObjectStoreRangeReader,
        counters: Arc<Counters>,
    }

    impl AsyncRangeReader for Counting {
        fn read_range(&self, offset: u64, len: usize) -> BoxFuture<'_, Result<Bytes, Box<dyn std::error::Error + Send + Sync>>> {
            // Reads inside the 16 KiB prefix are served from memory.
            if offset + len as u64 > 16 * 1024 {
                self.counters.requests.fetch_add(1, Ordering::Relaxed);
                if std::env::var_os("BENCH_TRACE").is_some() {
                    eprintln!("[{:?}] request {offset}+{len}", self.counters.started.elapsed());
                }
            }
            self.inner.read_range(offset, len)
        }
        fn size(&self) -> u64 {
            self.inner.size()
        }
        fn identifier(&self) -> &str {
            self.inner.identifier()
        }
        fn is_local(&self) -> bool {
            false
        }
        fn io_options(&self) -> &IoOptions {
            self.inner.io_options()
        }
    }

    pub async fn open(url: &str, counters: Arc<Counters>) -> CogReader {
        let inner = ObjectStoreRangeReader::open(url).await.expect("open");
        counters.requests.fetch_add(1, Ordering::Relaxed); // the open request itself
        CogReader::from_async_reader_with_hint(Arc::new(Counting { inner, counters }), hint()).await.expect("parse")
    }
}

#[cfg(baseline)]
mod counted {
    use super::*;
    use cogrs::{create_range_reader, RangeReader};

    struct Counting {
        inner: Arc<dyn RangeReader>,
        counters: Arc<Counters>,
    }

    impl RangeReader for Counting {
        fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
            self.counters.requests.fetch_add(1, Ordering::Relaxed);
            self.inner.read_range(offset, length)
        }
        fn size(&self) -> u64 {
            self.inner.size()
        }
        fn identifier(&self) -> &str {
            self.inner.identifier()
        }
    }

    /// Same as `CogReader::open_async`, with the reader wrapped in a request counter.
    pub async fn open(url: &str, counters: Arc<Counters>) -> CogReader {
        let url = url.to_string();
        tokio::task::spawn_blocking(move || {
            let inner = create_range_reader(&url).expect("open");
            counters.requests.fetch_add(1, Ordering::Relaxed); // the HEAD
            CogReader::from_reader_with_hint(Arc::new(Counting { inner, counters }), hint()).expect("parse")
        })
        .await
        .expect("join")
    }
}

struct Stats {
    means: [f64; 3],
    transparent_pct: f64,
}

fn stats(t: &TileData) -> Stats {
    let (mut sums, mut n, mut transparent, mut total) = ([0f64; 3], 0usize, 0usize, 0usize);
    for px in t.pixels.chunks(t.bands) {
        total += 1;
        let nan = px.iter().any(|v| v.is_nan());
        let nodata = t.nodata.is_some_and(|nd| px.iter().all(|v| f64::from(*v) == nd));
        if nan || nodata {
            transparent += 1;
        } else {
            n += 1;
            for (s, v) in sums.iter_mut().zip(px) {
                *s += f64::from(*v);
            }
        }
    }
    // Means over the whole tile with transparent pixels counted as 0.
    let total = total.max(1) as f64;
    let _ = n;
    Stats { means: [sums[0] / total, sums[1] / total, sums[2] / total], transparent_pct: 100.0 * transparent as f64 / total }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn z14_neighbourhood() -> Vec<(u32, u32, u32)> {
    let (x0, y0) = (4713, 6225);
    (0..8).flat_map(|dy| (0..8).map(move |dx| (14, x0 + dx, y0 + dy))).collect()
}

async fn extract(reader: &CogReader, (z, x, y): (u32, u32, u32)) -> TileData {
    TileExtractor::new(reader).xyz(z, x, y).size(256).extract().await.expect("extract")
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

async fn run(scenario: &str, kind: &str) {
    let url = url(kind);
    let counters = Arc::new(Counters::default());
    let start = Instant::now();

    match scenario {
        "single" | "z12" | "scene" => {
            let reader = counted::open(&url, counters.clone()).await;
            let opened = start.elapsed();
            let tile_id = match scenario {
                "single" => (14, 4717, 6229),
                "z12" => (12, 1179, 1557),
                _ => {
                    let spec = std::env::var("BENCH_TILE").unwrap_or_else(|_| "8/73/97".to_string());
                    let p: Vec<u32> = spec.split('/').map(|v| v.parse().expect("BENCH_TILE=z/x/y")).collect();
                    (p[0], p[1], p[2])
                }
            };
            let t0 = Instant::now();
            let tile = extract(&reader, tile_id).await;
            let tile_time = t0.elapsed();
            let s = stats(&tile);
            if let Ok(path) = std::env::var("BENCH_DUMP") {
                // raw interleaved u8 samples (NaN -> 0), e.g. to compare with a gdalwarp reference
                let raw: Vec<u8> = tile.pixels.iter().map(|v| if v.is_nan() { 0 } else { v.round().clamp(0.0, 255.0) as u8 }).collect();
                std::fs::write(path, raw).expect("dump");
            }
            println!(
                "{{\"scenario\":\"{scenario}\",\"kind\":\"{kind}\",\"open_ms\":{:.1},\"tile_ms\":{:.1},\"total_ms\":{:.1},\
                 \"requests\":{},\"tiles_read\":{},\"tile_bytes\":{},\"overview\":{:?},\
                 \"means\":[{:.2},{:.2},{:.2}],\"transparent_pct\":{:.2}}}",
                ms(opened),
                ms(tile_time),
                ms(start.elapsed()),
                counters.requests.load(Ordering::Relaxed),
                tile.tiles_read,
                tile.bytes_fetched,
                tile.overview_used,
                s.means[0],
                s.means[1],
                s.means[2],
                s.transparent_pct
            );
        }
        "concurrent" | "concurrent_open" => {
            let shared = if scenario == "concurrent" { Some(Arc::new(counted::open(&url, counters.clone()).await)) } else { None };
            let setup = start.elapsed();
            let t0 = Instant::now();
            let tiles = z14_neighbourhood();
            let handles: Vec<_> = tiles
                .into_iter()
                .map(|id| {
                    let shared = shared.clone();
                    let url = url.clone();
                    let counters = counters.clone();
                    tokio::spawn(async move {
                        let began = Instant::now();
                        let tile = match &shared {
                            Some(reader) => extract(reader, id).await,
                            None => {
                                let reader = counted::open(&url, counters).await;
                                extract(&reader, id).await
                            }
                        };
                        (began.elapsed(), tile)
                    })
                })
                .collect();
            let mut latencies = Vec::new();
            let (mut bytes, mut tiles_read) = (0usize, 0usize);
            for h in handles {
                let (d, tile) = h.await.expect("task");
                latencies.push(ms(d));
                bytes += tile.bytes_fetched;
                tiles_read += tile.tiles_read;
            }
            let wall = t0.elapsed();
            latencies.sort_by(f64::total_cmp);
            println!(
                "{{\"scenario\":\"{scenario}\",\"kind\":\"{kind}\",\"setup_ms\":{:.1},\"wall_ms\":{:.1},\"rps\":{:.1},\
                 \"p50_ms\":{:.1},\"p95_ms\":{:.1},\"max_ms\":{:.1},\"requests\":{},\"tiles_read\":{tiles_read},\"tile_bytes\":{bytes}}}",
                ms(setup),
                ms(wall),
                latencies.len() as f64 / wall.as_secs_f64(),
                percentile(&latencies, 0.5),
                percentile(&latencies, 0.95),
                latencies.last().copied().unwrap_or(0.0),
                counters.requests.load(Ordering::Relaxed),
            );
        }
        other => panic!("unknown scenario {other}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scenario = args.get(1).map_or("single", String::as_str);
    let kind = args.get(2).map_or("s3", String::as_str);
    let blocking: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
    let workers: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(4);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(blocking)
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(run(scenario, kind));
}
