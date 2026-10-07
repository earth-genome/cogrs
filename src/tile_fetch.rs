//! Fetching the source tiles an output tile needs.
//!
//! [`fetch_tiles`] resolves a set of source tiles of one COG level into decoded pixel data:
//!
//! 1. Tiles already in the process-wide decompressed-tile cache, and sparse tiles (never
//!    written), are resolved immediately.
//! 2. For the rest, each tile is *claimed* in a process-wide in-flight table. The caller that
//!    claims a tile (the leader) fetches it; concurrent callers that need the same tile wait for
//!    the leader's result instead of fetching it again.
//! 3. The leader fetches all its tiles' byte ranges at once through
//!    [`AsyncRangeReader::read_ranges`](crate::AsyncRangeReader::read_ranges), which merges
//!    nearby ranges and issues the requests concurrently, then decodes each tile on tokio's
//!    blocking pool, inserts it into the tile cache and publishes it to waiting followers.
//!
//! Cancellation is safe: if a leader is dropped before publishing, its followers notice and
//! re-resolve the tile (one of them becomes the new leader).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, LazyLock};

use ahash::AHashMap;
use parking_lot::Mutex;
use tokio::sync::watch;

use crate::async_io::RangeFetchError;
use crate::cog_reader::{CogReader, TileRef, TileSpan};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Outcome of fetching one tile, as shared with followers. Errors are shared as text.
type TileResult = Result<Arc<Vec<f32>>, Arc<str>>;

/// Decoded source tiles plus I/O statistics for the tiles this call fetched itself.
#[derive(Default)]
pub(crate) struct FetchedTiles {
    /// Decoded tiles by index within the requested level.
    pub tiles: AHashMap<usize, Arc<Vec<f32>>>,
    /// Compressed bytes fetched by this call (0 for cache hits and tiles fetched by others).
    pub bytes_fetched: usize,
    /// Number of tiles fetched by this call.
    pub tiles_read: usize,
}

/// `(source identifier, overview level, tile index)`
type InflightKey = (String, Option<usize>, usize);

/// Publishing end of a tile being fetched; followers subscribe to it.
type InflightSender = Arc<watch::Sender<Option<TileResult>>>;

static INFLIGHT: LazyLock<Mutex<HashMap<InflightKey, InflightSender>>> = LazyLock::new(Mutex::default);

/// The caller responsible for fetching a tile. Dropping it unregisters the tile.
struct Leader {
    key: InflightKey,
    tx: InflightSender,
}

impl Leader {
    fn publish(&self, result: TileResult) {
        self.tx.send_replace(Some(result));
    }
}

impl Drop for Leader {
    fn drop(&mut self) {
        INFLIGHT.lock().remove(&self.key);
    }
}

enum Claim {
    Leader(Leader),
    Follower(watch::Receiver<Option<TileResult>>),
}

fn claim(identifier: &str, tile: TileRef) -> Claim {
    let key = (identifier.to_string(), tile.overview, tile.index);
    let mut inflight = INFLIGHT.lock();
    if let Some(tx) = inflight.get(&key) {
        return Claim::Follower(tx.subscribe());
    }
    let tx = Arc::new(watch::channel(None).0);
    inflight.insert(key.clone(), Arc::clone(&tx));
    Claim::Leader(Leader { key, tx })
}

/// Number of tiles of `identifier` currently being fetched by some caller (tests).
#[cfg(test)]
pub(crate) fn inflight_count(identifier: &str) -> usize {
    INFLIGHT.lock().keys().filter(|(id, _, _)| id == identifier).count()
}

fn tile_error(index: usize, overview: Option<usize>, cause: impl std::fmt::Display) -> BoxError {
    format!("Failed to read source tile {index} (overview {overview:?}): {cause}").into()
}

/// Fetch and decode `needed` tiles (indexes within `overview`'s level; `None` = full
/// resolution).
///
/// Any read or decode error aborts with `Failed to read source tile {index} (overview {..}):
/// {cause}`. Sparse tiles come back as all-NaN with 0 bytes.
pub(crate) async fn fetch_tiles(
    reader: &CogReader,
    overview: Option<usize>,
    needed: &[usize],
) -> Result<FetchedTiles, BoxError> {
    let mut out = FetchedTiles::default();
    let mut pending: Vec<usize> = needed.to_vec();

    while !pending.is_empty() {
        let mut leaders: Vec<(usize, TileSpan, Leader)> = Vec::new();
        let mut followers: Vec<(usize, watch::Receiver<Option<TileResult>>)> = Vec::new();

        for index in pending.drain(..) {
            let tile = TileRef { overview, index };
            if let Some(data) = reader.cached_tile(tile) {
                out.tiles.insert(index, data);
                continue;
            }
            let span = reader.tile_span(tile).map_err(|e| tile_error(index, overview, e))?;
            if span.len == 0 {
                out.tiles.insert(index, reader.sparse_tile(&span));
                continue;
            }
            match claim(reader.identifier(), tile) {
                Claim::Leader(leader) => {
                    // Another caller may have published and unregistered it since the first check.
                    if let Some(data) = reader.cached_tile(tile) {
                        leader.publish(Ok(Arc::clone(&data)));
                        out.tiles.insert(index, data);
                    } else {
                        leaders.push((index, span, leader));
                    }
                }
                Claim::Follower(rx) => followers.push((index, rx)),
            }
        }

        // Fetch our tiles and wait for the ones others are fetching at the same time.
        let (led, followed) = futures::join!(
            fetch_and_decode(reader, overview, leaders),
            wait_for_followers(followers),
        );

        for (index, data, bytes) in led? {
            out.tiles.insert(index, data);
            out.bytes_fetched += bytes;
            out.tiles_read += 1;
        }
        for (index, outcome) in followed {
            match outcome {
                Followed::Ready(data) => {
                    out.tiles.insert(index, data);
                }
                Followed::Failed(message) => return Err(tile_error(index, overview, message)),
                // The leader went away without a result: resolve the tile again.
                Followed::Abandoned => pending.push(index),
            }
        }
    }

    Ok(out)
}

/// Fetch `leaders`' byte ranges in one coalesced concurrent request set, then decode each tile
/// on the blocking pool, cache it and publish it. Returns `(index, data, compressed bytes)`.
async fn fetch_and_decode(
    reader: &CogReader,
    overview: Option<usize>,
    leaders: Vec<(usize, TileSpan, Leader)>,
) -> Result<Vec<(usize, Arc<Vec<f32>>, usize)>, BoxError> {
    if leaders.is_empty() {
        return Ok(Vec::new());
    }

    let ranges: Vec<Range<u64>> = leaders.iter().map(|(_, span, _)| span.offset..span.offset + span.len as u64).collect();
    let compressed = match reader.io().read_ranges(&ranges).await {
        Ok(compressed) => compressed,
        Err(e) => {
            // Name the tile the failed request was for (the first of its tiles if the failed
            // request merged several).
            let culprit = e
                .downcast_ref::<RangeFetchError>()
                .and_then(|failed| {
                    leaders
                        .iter()
                        .zip(&ranges)
                        .find(|(_, r)| r.start < failed.range.end && failed.range.start < r.end)
                        .map(|((index, _, _), _)| *index)
                })
                .unwrap_or(leaders[0].0);
            let message: Arc<str> = e.to_string().into();
            for (_, _, leader) in &leaders {
                leader.publish(Err(Arc::clone(&message)));
            }
            return Err(tile_error(culprit, overview, message));
        }
    };

    let decodes = leaders.into_iter().zip(compressed).map(|((index, span, leader), bytes)| {
        let tile = TileRef { overview, index };
        let this = reader.clone();
        async move {
            let decoded = tokio::task::spawn_blocking({
                let this = this.clone();
                move || this.decode_tile(&span, &bytes)
            })
            .await
            .map_err(|e| format!("Task join error: {e}"))
            .and_then(|r| r.map_err(|e| e.to_string()));
            match decoded {
                Ok(data) => {
                    let data = Arc::new(data);
                    this.cache_tile(tile, Arc::clone(&data));
                    leader.publish(Ok(Arc::clone(&data)));
                    Ok((index, data, span.len))
                }
                Err(message) => {
                    let message: Arc<str> = message.into();
                    leader.publish(Err(Arc::clone(&message)));
                    Err(tile_error(index, overview, message))
                }
            }
        }
    });
    futures::future::try_join_all(decodes).await
}

enum Followed {
    Ready(Arc<Vec<f32>>),
    Failed(Arc<str>),
    Abandoned,
}

async fn wait_for_followers(
    followers: Vec<(usize, watch::Receiver<Option<TileResult>>)>,
) -> Vec<(usize, Followed)> {
    futures::future::join_all(followers.into_iter().map(|(index, mut rx)| async move {
        let outcome = match rx.wait_for(Option::is_some).await {
            Ok(value) => match value.as_ref() {
                Some(Ok(data)) => Followed::Ready(Arc::clone(data)),
                Some(Err(message)) => Followed::Failed(Arc::clone(message)),
                None => Followed::Abandoned,
            },
            Err(_) => Followed::Abandoned,
        };
        (index, outcome)
    }))
    .await
}
