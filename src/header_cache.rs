//! Process-wide cache of parsed COG headers for remote sources.
//!
//! A tile server opens the same image for every request. Opening costs a ranged `GET` for the
//! header plus, for COGs with many tiles, extra `GET`s for the tile offset arrays, and the default
//! overview-quality hint samples tiles on top of that. [`CogCache`] remembers what an open
//! learned, so the next open of the same source sends no request at all.
//!
//! # What is cached
//!
//! One entry per source: the parsed [`CogMetadata`] and [`OverviewMetadata`] (shared `Arc`s),
//! what the server reported about the object (size, ETag, last-modified), and the computed
//! overview-quality result. Not cached: the object store client (shared per bucket or host by
//! the store registry), the S3 region (cached per bucket), and decoded tiles (the byte-bounded
//! tile cache, keyed by the same identity, see [`tile_cache`]).
//!
//! An entry weighs about 1.5 KiB plus 16 bytes per tile (offset and byte count, all levels): a
//! 10980 x 10980 COG with 512 px tiles and three overviews is about 12 KB, a 100k x 100k one
//! 0.8 MB (512 px tiles) to 3.3 MB (256 px tiles). The cache is bounded by the sum of the
//! weights ([`CacheConfig::capacity_bytes`], 64 MiB by default, least recently used first).
//!
//! # Keeping it correct
//!
//! The lookup key is known before any I/O: the origin (S3 bucket, endpoint and credentials, or
//! HTTP scheme, host and port), the object path and the [`IoOptions`]. The query string of an
//! HTTP URL is not part of it, so presigned URLs of one object share an entry. What the cache
//! stored is only valid for the version of the object it was learned from, and
//! [`Validation`] decides how that is enforced:
//!
//! - [`Validation::IfMatch`] (default): every read of the object is conditional on its ETag
//!   (`If-Match`, or `If-Unmodified-Since` without a strong ETag). A replaced object answers
//!   `412`; the entry and the old version's decoded tiles are dropped and the operation runs
//!   again once against a fresh open. Entries live for [`CacheConfig::ttl`]; entries whose
//!   server gives no validator at all expire after [`CacheConfig::ttl_unvalidated`].
//! - [`Validation::Ttl`]: no conditions; entries expire after [`CacheConfig::ttl`].
//! - [`Validation::Immutable`]: no conditions and no expiry, for objects that are never
//!   overwritten (date-versioned keys).
//!
//! [`CogCache::invalidate`] drops a source explicitly, [`CogCache::clear`] everything.
//!
//! Concurrent opens of the same source share one request ("single flight"), and a missing object
//! (`404`) is remembered for [`CacheConfig::negative_ttl`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use futures::future::{BoxFuture, FutureExt, Shared, WeakShared};
use lru::LruCache;
use parking_lot::Mutex;
use tokio::time::Instant;

use crate::async_io::{AsyncRangeReader, IoOptions, is_source_changed};
use crate::cog_reader::{CogMetadata, CogReader, OverviewMetadata, OverviewQualityHint, is_remote_source, parse_cog_structure};
use crate::remote::{ObjectIdentity, ObjectStoreRangeReader, SourceKeyParts, SourceNotFound, Validation, source_key_parts};
use crate::tile_cache::{self, TileCacheStats};
use crate::tiff_utils::AnyResult;

/// Fixed part of an entry's weight (structs, strings, allocator overhead), in bytes.
const ENTRY_BASE_BYTES: usize = 1536;
/// Most missing sources remembered at once.
const NEGATIVE_CAPACITY: usize = 1024;

/// Tuning of a [`CogCache`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    /// Upper bound on the summed weight of the entries, in bytes (64 MiB).
    pub capacity_bytes: usize,
    /// How long an entry that reads can be validated against (see [`Validation`]) is trusted
    /// (1 hour). Entries are also dropped as soon as a read finds the object changed.
    pub ttl: Duration,
    /// How long an entry is trusted when the server gave no usable validator, so a replaced
    /// object cannot be detected on read (60 seconds). Only used with [`Validation::IfMatch`].
    pub ttl_unvalidated: Duration,
    /// How long a `404` is remembered (5 seconds; zero turns negative caching off). Only "not
    /// found" is remembered, never `401`/`403`, server errors or timeouts.
    pub negative_ttl: Duration,
    /// How reads are tied to the cached version.
    pub validation: Validation,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity_bytes: 64 * 1024 * 1024,
            ttl: Duration::from_secs(3600),
            ttl_unvalidated: Duration::from_secs(60),
            negative_ttl: Duration::from_secs(5),
            validation: Validation::IfMatch,
        }
    }
}

/// Whether an open may use and fill the header cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheMode {
    /// Use a cached header, or open and cache it (default).
    #[default]
    Use,
    /// Open the source from scratch and do not cache anything.
    Bypass,
    /// Drop what is cached for the source, open it from scratch and cache the result.
    Refresh,
}

/// Point-in-time statistics of the header cache. Counters only grow; they are atomics and cost
/// nothing on the hot path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeaderCacheStats {
    pub entries: usize,
    /// Summed weight of the entries
    pub bytes: usize,
    pub capacity_bytes: usize,
    /// Opens served from a cached entry (no request)
    pub hits: u64,
    /// Opens that found no usable entry and no open in flight (a request was made)
    pub misses: u64,
    /// Entries evicted to stay within the capacity
    pub evictions: u64,
    /// Entries dropped because their time to live ran out
    pub expirations: u64,
    /// Opens that waited for an identical open already in flight instead of requesting
    pub coalesced_opens: u64,
    /// Entries dropped because a read found the object changed (`412`)
    pub stale_evictions: u64,
    /// Opens answered from a remembered `404`
    pub negative_hits: u64,
}

/// Statistics of both caches, for server metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub headers: HeaderCacheStats,
    /// The process-wide decoded-tile cache, shared by every [`CogCache`].
    pub tiles: TileCacheStats,
}

/// Lookup key: known before any I/O.
#[derive(Clone, PartialEq, Eq, Hash)]
struct HeaderKey {
    origin: String,
    path: String,
    options: IoOptions,
}

/// What an open learned about one version of a source.
struct Entry {
    identifier: String,
    metadata: Arc<CogMetadata>,
    overviews: Arc<[OverviewMetadata]>,
    identity: ObjectIdentity,
    /// `identity`'s version token
    version: String,
    expires_at: Option<Instant>,
    weight: usize,
    quality: QualityCell,
}

impl Entry {
    fn expired(&self, now: Instant) -> bool {
        self.expires_at.is_some_and(|at| now >= at)
    }
}

type QualityFuture = Shared<BoxFuture<'static, (Option<usize>, bool)>>;

/// The overview-quality result of an entry, computed once: concurrent first requests share one
/// computation, and an incomplete one (some sample tile failed to read) is used by the callers
/// that waited for it but never stored.
enum QualityState {
    Empty,
    Computing(WeakShared<BoxFuture<'static, (Option<usize>, bool)>>),
    Done(Option<usize>),
}

struct QualityCell(Mutex<QualityState>);

impl QualityCell {
    async fn get(&self, reader: &CogReader) -> Option<usize> {
        let shared: QualityFuture = {
            let mut state = self.0.lock();
            if let QualityState::Done(value) = &*state {
                return *value;
            }
            let running = match &*state {
                QualityState::Computing(weak) => weak.upgrade(),
                _ => None,
            };
            if let Some(shared) = running {
                shared
            } else {
                let sampler = reader.clone();
                let shared = async move { sampler.analyze_overview_quality_async().await }.boxed().shared();
                *state = QualityState::Computing(shared.downgrade().expect("not polled yet"));
                shared
            }
        };
        let (value, complete) = shared.await;
        if complete {
            *self.0.lock() = QualityState::Done(value);
        }
        value
    }
}

/// An open that failed, shared by every caller that waited for it.
struct OpenFailure {
    message: String,
    not_found: bool,
}

impl OpenFailure {
    fn of(error: &(dyn std::error::Error + 'static)) -> Self {
        Self { message: error.to_string(), not_found: error.is::<SourceNotFound>() }
    }

    fn error(&self) -> Box<dyn std::error::Error + Send + Sync> {
        if self.not_found { Box::new(SourceNotFound::new(self.message.clone())) } else { self.message.clone().into() }
    }
}

type OpenOutcome = Result<Arc<Entry>, Arc<OpenFailure>>;
type OpenFuture = Shared<BoxFuture<'static, OpenOutcome>>;
type InflightOpens = HashMap<HeaderKey, WeakShared<BoxFuture<'static, OpenOutcome>>>;

struct NegativeEntry {
    identifier: String,
    message: String,
    until: Instant,
}

struct Headers {
    entries: LruCache<HeaderKey, Arc<Entry>>,
    bytes: usize,
    negatives: LruCache<HeaderKey, NegativeEntry>,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
    coalesced_opens: AtomicU64,
    stale_evictions: AtomicU64,
    negative_hits: AtomicU64,
}

struct Inner {
    config: CacheConfig,
    enabled: bool,
    headers: Mutex<Headers>,
    inflight: Mutex<InflightOpens>,
    counters: Counters,
}

/// Where a [`CogReader`] came from, so a stale one can be reopened (see
/// [`CogReader::reopen_after_change`](crate::CogReader)).
pub(crate) struct ReaderOrigin {
    pub cache: CogCache,
    /// The source as the caller gave it (including any presigned query string)
    pub source: String,
    pub options: IoOptions,
}

/// Cache of parsed COG headers; cheap to clone (clones share one cache).
///
/// [`CogCache::global`] is the process-wide default every `CogReader::open*` call uses for remote
/// sources. Create your own with [`CogCache::new`] and pass it to
/// [`CogReader::builder`](crate::CogReader::builder), or use [`CogCache::disabled`] to turn
/// caching off. Decoded tiles live in one process-wide cache regardless of the `CogCache` used
/// (see [`tile_cache`]); [`invalidate`](Self::invalidate),
/// [`clear`](Self::clear) and [`stats`](Self::stats) cover both.
///
/// The global cache is disabled by `COGRS_HEADER_CACHE=off` and sized by
/// `COGRS_HEADER_CACHE_MB`.
#[derive(Clone)]
pub struct CogCache {
    inner: Arc<Inner>,
}

static GLOBAL: LazyLock<CogCache> = LazyLock::new(|| {
    let disabled = std::env::var("COGRS_HEADER_CACHE")
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "disabled"));
    if disabled {
        return CogCache::disabled();
    }
    let mut config = CacheConfig::default();
    if let Some(mb) = std::env::var("COGRS_HEADER_CACHE_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        config.capacity_bytes = mb.saturating_mul(1024 * 1024);
    }
    CogCache::new(config)
});

static DISABLED: LazyLock<CogCache> = LazyLock::new(|| CogCache::build(CacheConfig::default(), false));

impl CogCache {
    /// A cache with its own entries and `config`.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self::build(config, true)
    }

    fn build(config: CacheConfig, enabled: bool) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                enabled,
                headers: Mutex::new(Headers {
                    entries: LruCache::unbounded(),
                    bytes: 0,
                    negatives: LruCache::new(std::num::NonZeroUsize::new(NEGATIVE_CAPACITY).expect("non-zero")),
                }),
                inflight: Mutex::new(HashMap::new()),
                counters: Counters::default(),
            }),
        }
    }

    /// The process-wide default cache.
    #[must_use]
    pub fn global() -> Self {
        GLOBAL.clone()
    }

    /// A cache that caches nothing: opens through it always read the source (the decoded-tile
    /// cache is unaffected). Reads are still conditional on the opened version.
    #[must_use]
    pub fn disabled() -> Self {
        DISABLED.clone()
    }

    /// This cache's configuration.
    #[must_use]
    pub fn config(&self) -> &CacheConfig {
        &self.inner.config
    }

    /// Statistics of the header cache and of the decoded-tile cache.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let (entries, bytes) = {
            let headers = self.inner.headers.lock();
            (headers.entries.len(), headers.bytes)
        };
        let c = &self.inner.counters;
        CacheStats {
            headers: HeaderCacheStats {
                entries,
                bytes,
                capacity_bytes: self.inner.config.capacity_bytes,
                hits: c.hits.load(Ordering::Relaxed),
                misses: c.misses.load(Ordering::Relaxed),
                evictions: c.evictions.load(Ordering::Relaxed),
                expirations: c.expirations.load(Ordering::Relaxed),
                coalesced_opens: c.coalesced_opens.load(Ordering::Relaxed),
                stale_evictions: c.stale_evictions.load(Ordering::Relaxed),
                negative_hits: c.negative_hits.load(Ordering::Relaxed),
            },
            tiles: tile_cache::snapshot(),
        }
    }

    /// Forget everything cached for `source` (an `s3://` or `http(s)://` URL, or a path): its
    /// header entries (every origin and version), a remembered `404`, and its decoded tiles.
    /// The next open reads the source. Returns the number of header entries dropped.
    pub fn invalidate(&self, source: &str) -> usize {
        let identifier = source_key_parts(source).map_or_else(|_| source.to_string(), |parts| parts.identifier);
        let removed = {
            let mut headers = self.inner.headers.lock();
            let keys: Vec<HeaderKey> =
                headers.entries.iter().filter(|(_, e)| e.identifier == identifier).map(|(k, _)| k.clone()).collect();
            for key in &keys {
                headers.remove(key);
            }
            let negatives: Vec<HeaderKey> =
                headers.negatives.iter().filter(|(_, n)| n.identifier == identifier).map(|(k, _)| k.clone()).collect();
            for key in &negatives {
                headers.negatives.pop(key);
            }
            keys.len()
        };
        tile_cache::invalidate_source(&identifier);
        removed
    }

    /// Drop every header entry and remembered `404` of this cache, and the decoded tiles of the
    /// sources they describe. Tiles of other sources stay; empty the whole tile cache with
    /// [`tile_cache::clear`].
    pub fn clear(&self) {
        let identifiers: Vec<String> = {
            let mut headers = self.inner.headers.lock();
            let ids = headers
                .entries
                .iter()
                .map(|(_, e)| e.identifier.clone())
                .chain(headers.negatives.iter().map(|(_, n)| n.identifier.clone()))
                .collect();
            headers.entries.clear();
            headers.negatives.clear();
            headers.bytes = 0;
            ids
        };
        for identifier in identifiers {
            tile_cache::invalidate_source(&identifier);
        }
    }

    /// Open `source` through this cache.
    pub(crate) async fn open(
        &self,
        source: &str,
        hint: OverviewQualityHint,
        options: &IoOptions,
        mode: CacheMode,
    ) -> AnyResult<CogReader> {
        let validation = self.inner.config.validation;
        let origin = is_remote_source(source).then(|| {
            Arc::new(ReaderOrigin { cache: self.clone(), source: source.to_string(), options: options.clone() })
        });
        if !self.inner.enabled || mode == CacheMode::Bypass || origin.is_none() {
            return CogReader::open_uncached(source, hint, options, validation, origin).await;
        }

        let parts = source_key_parts(source)?;
        let key = HeaderKey { origin: parts.origin.clone(), path: parts.path.clone(), options: options.clone() };
        if mode == CacheMode::Refresh {
            self.drop_source(&key, &parts.identifier);
        }
        let entry = self.entry(source, &key, &parts, options).await?;
        self.reader_from(source, options, &entry, hint, origin).await
    }

    /// Reopen `source` after a read found it changed: drop the entry for `stale_version` (unless
    /// someone already replaced it with a newer one), then open through the cache. Concurrent
    /// callers share one fresh open.
    pub(crate) async fn reopen_stale(
        &self,
        source: &str,
        hint: OverviewQualityHint,
        options: &IoOptions,
        stale_version: Option<&str>,
    ) -> AnyResult<CogReader> {
        if let Ok(parts) = source_key_parts(source) {
            let key = HeaderKey { origin: parts.origin, path: parts.path, options: options.clone() };
            let mut headers = self.inner.headers.lock();
            if headers.entries.peek(&key).is_some_and(|e| stale_version.is_none_or(|v| e.version == v)) {
                headers.remove(&key);
                self.inner.counters.stale_evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.open(source, hint, options, CacheMode::Use).await
    }

    /// A cached entry, or the result of one open shared by every concurrent caller.
    async fn entry(
        &self,
        source: &str,
        key: &HeaderKey,
        parts: &SourceKeyParts,
        options: &IoOptions,
    ) -> AnyResult<Arc<Entry>> {
        let counters = &self.inner.counters;
        if let Some(entry) = self.lookup(key) {
            counters.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(entry);
        }
        if let Some(message) = self.remembered_not_found(key) {
            counters.negative_hits.fetch_add(1, Ordering::Relaxed);
            return Err(Box::new(SourceNotFound::new(message)));
        }

        let open: OpenFuture = {
            let mut inflight = self.inner.inflight.lock();
            if let Some(shared) = inflight.get(key).and_then(WeakShared::upgrade) {
                counters.coalesced_opens.fetch_add(1, Ordering::Relaxed);
                shared
            } else {
                counters.misses.fetch_add(1, Ordering::Relaxed);
                let (cache, owned_key, identifier) = (self.clone(), key.clone(), parts.identifier.clone());
                let (source, options) = (source.to_string(), options.clone());
                let shared = async move {
                    let result = cache.open_entry(&source, &owned_key, &identifier, &options).await;
                    cache.inner.inflight.lock().remove(&owned_key);
                    result
                }
                .boxed()
                .shared();
                inflight.insert(key.clone(), shared.downgrade().expect("not polled yet"));
                shared
            }
        };
        open.await.map_err(|failure| failure.error())
    }

    /// Read the source and cache what was learned. A conditional read that finds the object
    /// replaced while the header is being parsed (a torn open) is tried once more.
    async fn open_entry(
        &self,
        source: &str,
        key: &HeaderKey,
        identifier: &str,
        options: &IoOptions,
    ) -> Result<Arc<Entry>, Arc<OpenFailure>> {
        let mut attempt = 0;
        let result = loop {
            attempt += 1;
            match self.read_entry(source, key, identifier, options).await {
                Err(e) if attempt == 1 && is_source_changed(&*e) => {}
                other => break other,
            }
        };
        match result {
            Ok(entry) => {
                let entry = Arc::new(entry);
                self.insert(key, Arc::clone(&entry));
                Ok(entry)
            }
            Err(e) => {
                let failure = OpenFailure::of(&*e);
                if failure.not_found {
                    self.remember_not_found(key, identifier, &failure.message);
                }
                Err(Arc::new(failure))
            }
        }
    }

    async fn read_entry(
        &self,
        source: &str,
        key: &HeaderKey,
        identifier: &str,
        options: &IoOptions,
    ) -> AnyResult<Entry> {
        let validation = self.inner.config.validation;
        let reader = ObjectStoreRangeReader::open_with_validation(source, options, validation).await?;
        let identity = reader.object_identity().clone();
        let structure = parse_cog_structure(&reader).await?;
        let (metadata, overviews) = (Arc::new(structure.metadata), Arc::<[OverviewMetadata]>::from(structure.overviews));

        let config = &self.inner.config;
        let ttl = match validation {
            Validation::Immutable => None,
            Validation::Ttl => Some(config.ttl),
            Validation::IfMatch if identity.is_validatable(validation) => Some(config.ttl),
            Validation::IfMatch => Some(config.ttl_unvalidated),
        };
        let tiles: usize = metadata.tile_offsets.len()
            + metadata.tile_byte_counts.len()
            + overviews.iter().map(|o| o.tile_offsets.len() + o.tile_byte_counts.len()).sum::<usize>();
        let weight = ENTRY_BASE_BYTES
            + tiles * std::mem::size_of::<u64>()
            + overviews.len() * std::mem::size_of::<OverviewMetadata>()
            + identifier.len()
            + key.origin.len()
            + key.path.len();
        Ok(Entry {
            identifier: identifier.to_string(),
            metadata,
            overviews,
            version: identity.version(),
            identity,
            expires_at: ttl.map(|ttl| Instant::now() + ttl),
            weight,
            quality: QualityCell(Mutex::new(QualityState::Empty)),
        })
    }

    /// A reader for `source` backed by `entry`: no request is sent.
    async fn reader_from(
        &self,
        source: &str,
        options: &IoOptions,
        entry: &Arc<Entry>,
        hint: OverviewQualityHint,
        origin: Option<Arc<ReaderOrigin>>,
    ) -> AnyResult<CogReader> {
        let reader =
            ObjectStoreRangeReader::open_known(source, options, self.inner.config.validation, &entry.identity).await?;
        let io: Arc<dyn AsyncRangeReader> = Arc::new(reader);
        let mut cog = CogReader::from_cached(io, Arc::clone(&entry.metadata), Arc::clone(&entry.overviews), hint, origin);
        if matches!(hint, OverviewQualityHint::ComputeAtRuntime) {
            cog.min_usable_overview = entry.quality.get(&cog).await;
        }
        Ok(cog)
    }

    /// A fresh entry for `key`, dropping it if its time to live ran out.
    fn lookup(&self, key: &HeaderKey) -> Option<Arc<Entry>> {
        let mut headers = self.inner.headers.lock();
        let entry = Arc::clone(headers.entries.get(key)?);
        if entry.expired(Instant::now()) {
            headers.remove(key);
            self.inner.counters.expirations.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(entry)
    }

    fn insert(&self, key: &HeaderKey, entry: Arc<Entry>) {
        let capacity = self.inner.config.capacity_bytes;
        let mut headers = self.inner.headers.lock();
        headers.negatives.pop(key);
        headers.remove(key);
        if entry.weight > capacity {
            return;
        }
        while headers.bytes + entry.weight > capacity {
            let Some((_, evicted)) = headers.entries.pop_lru() else { break };
            headers.bytes = headers.bytes.saturating_sub(evicted.weight);
            self.inner.counters.evictions.fetch_add(1, Ordering::Relaxed);
        }
        headers.bytes += entry.weight;
        headers.entries.put(key.clone(), entry);
    }

    /// Remove the entry and any remembered `404` for `key`, and every decoded tile of the source.
    fn drop_source(&self, key: &HeaderKey, identifier: &str) {
        {
            let mut headers = self.inner.headers.lock();
            headers.remove(key);
            headers.negatives.pop(key);
        }
        tile_cache::invalidate_source(identifier);
    }

    fn remember_not_found(&self, key: &HeaderKey, identifier: &str, message: &str) {
        let ttl = self.inner.config.negative_ttl;
        if ttl.is_zero() {
            return;
        }
        let entry = NegativeEntry { identifier: identifier.to_string(), message: message.to_string(), until: Instant::now() + ttl };
        self.inner.headers.lock().negatives.put(key.clone(), entry);
    }

    /// The message of a remembered `404` for `key` that has not expired.
    fn remembered_not_found(&self, key: &HeaderKey) -> Option<String> {
        let mut headers = self.inner.headers.lock();
        match headers.negatives.get(key) {
            Some(n) if Instant::now() < n.until => Some(n.message.clone()),
            Some(_) => {
                headers.negatives.pop(key);
                None
            }
            None => None,
        }
    }
}

impl Headers {
    fn remove(&mut self, key: &HeaderKey) {
        if let Some(entry) = self.entries.pop(key) {
            self.bytes = self.bytes.saturating_sub(entry.weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cog_reader::OverviewQualityHint as Hint;
    use crate::remote::SourceNotFound;
    use crate::test_support::{CogSpec, ObjectServer, Sample, ServedObject, build_cog};

    fn pattern_a(_: usize, x: usize, y: usize) -> f64 {
        ((x * 31 + y * 17) % 4000) as f64 + 1.0
    }

    fn pattern_b(_: usize, x: usize, y: usize) -> f64 {
        ((x * 7 + y * 13) % 3000) as f64 + 1001.0
    }

    /// Uncompressed 64 px tiles with two overviews: the tile data lies far beyond the 16 KiB
    /// prefix, so tile reads are real requests.
    fn spec(size: usize, pixel: fn(usize, usize, usize) -> f64) -> CogSpec {
        CogSpec {
            width: size,
            height: size,
            tile: 64,
            bands: 1,
            sample: Sample::U16,
            deflate: false,
            predictor: false,
            epsg: 3857,
            origin: (0.0, 6400.0),
            pixel_size: (10.0, 10.0),
            nodata: None,
            overviews: 2,
            sparse: vec![],
            corrupt: vec![],
            pixel,
        }
    }

    fn object(size: usize, pixel: fn(usize, usize, usize) -> f64, etag: &str) -> ServedObject {
        ServedObject::new(build_cog(&spec(size, pixel))).etag(Some(etag))
    }

    fn server(object: ServedObject) -> ObjectServer {
        ObjectServer::start(Some(object), Duration::ZERO)
    }

    fn options() -> IoOptions {
        IoOptions { max_retries: 0, ..IoOptions::default() }
    }

    fn cache_with(config: impl FnOnce(&mut CacheConfig)) -> CogCache {
        let mut c = CacheConfig::default();
        config(&mut c);
        CogCache::new(c)
    }

    async fn open(cache: &CogCache, url: &str, hint: Hint) -> AnyResult<CogReader> {
        CogReader::builder(url).cache(cache).hint(hint).io_options(options()).open_async().await
    }

    fn requests_for(server: &ObjectServer, path: &str) -> usize {
        server.requests().iter().filter(|r| r.path == path).count()
    }

    #[tokio::test]
    async fn the_second_open_sends_no_request() {
        let server = server(object(512, pattern_a, "\"a1\""));
        let url = format!("{}/a.tif", server.base());
        let cache = cache_with(|_| {});

        let first = open(&cache, &url, Hint::ComputeAtRuntime).await.unwrap();
        let after_first = server.requests().len();
        assert!(after_first > 1, "the open reads the header and samples tiles: {after_first}");
        assert_eq!(first.min_usable_overview, Some(1));

        let second = open(&cache, &url, Hint::ComputeAtRuntime).await.unwrap();
        assert_eq!(server.requests().len(), after_first, "a cache hit must not touch the network");
        assert!(Arc::ptr_eq(&first.metadata, &second.metadata));
        assert!(Arc::ptr_eq(&first.overviews, &second.overviews));
        assert_eq!(second.min_usable_overview, Some(1), "the overview quality is computed once per source");

        let stats = cache.stats().headers;
        assert_eq!((stats.entries, stats.hits, stats.misses), (1, 1, 1));
        assert!(stats.bytes > 1536 && stats.bytes < 64 * 1024, "weight {}", stats.bytes);

        // Explicit hints apply on top of the cached structure, still without a request.
        assert_eq!(open(&cache, &url, Hint::AllUsable).await.unwrap().min_usable_overview, Some(1));
        assert_eq!(open(&cache, &url, Hint::NoneUsable).await.unwrap().min_usable_overview, None);
        assert_eq!(open(&cache, &url, Hint::MinUsable(0)).await.unwrap().min_usable_overview, Some(0));
        assert_eq!(server.requests().len(), after_first);

        // The reader works: a tile read is a conditional request for the cached version.
        let tile = second.read_tile_async(0).await.unwrap();
        assert!((tile[0] - pattern_a(0, 0, 0) as f32).abs() < 1e-3);
        let last = server.requests().pop().unwrap();
        assert_eq!(last.if_match.as_deref(), Some("\"a1\""));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_cold_opens_share_one_request() {
        let server = ObjectServer::start(Some(object(512, pattern_a, "\"a1\"")), Duration::from_millis(60));
        let cache = cache_with(|_| {});
        // What one sequential open costs
        open(&cache, &format!("{}/reference.tif", server.base()), Hint::AllUsable).await.unwrap();
        let single = requests_for(&server, "/reference.tif");

        let url = format!("{}/shared.tif", server.base());
        let opens = futures::future::join_all((0..32).map(|_| open(&cache, &url, Hint::AllUsable))).await;
        assert!(opens.iter().all(Result::is_ok));
        assert_eq!(requests_for(&server, "/shared.tif"), single, "32 concurrent opens, one open's requests");
        let stats = cache.stats().headers;
        assert_eq!(stats.misses, 2, "reference + shared");
        assert_eq!(stats.coalesced_opens + stats.hits, 31);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_cold_opens_compute_the_quality_once() {
        let server = ObjectServer::start(Some(object(512, pattern_a, "\"a1\"")), Duration::from_millis(40));
        let url = format!("{}/q.tif", server.base());
        let cache = cache_with(|_| {});
        // Header plus the sample tiles of one open, measured on its own
        open(&cache, &format!("{}/ref.tif", server.base()), Hint::ComputeAtRuntime).await.unwrap();
        let single = requests_for(&server, "/ref.tif");

        let opens = futures::future::join_all((0..16).map(|_| open(&cache, &url, Hint::ComputeAtRuntime))).await;
        assert!(opens.iter().all(|r| r.as_ref().unwrap().min_usable_overview == Some(1)));
        assert_eq!(requests_for(&server, "/q.tif"), single);
    }

    #[tokio::test]
    async fn evicts_least_recently_used_entries_by_bytes() {
        let server = server(object(512, pattern_a, "\"a1\""));
        let url = |name: &str| format!("{}/{name}.tif", server.base());
        let probe = cache_with(|_| {});
        open(&probe, &url("probe"), Hint::AllUsable).await.unwrap();
        let weight = probe.stats().headers.bytes;

        // Room for two entries, not three
        let cache = cache_with(|c| c.capacity_bytes = 2 * weight + weight / 2);
        for name in ["one", "two", "three"] {
            open(&cache, &url(name), Hint::AllUsable).await.unwrap();
        }
        let stats = cache.stats().headers;
        assert_eq!((stats.entries, stats.evictions), (2, 1));
        assert!(stats.bytes <= stats.capacity_bytes);

        let before = server.requests().len();
        open(&cache, &url("three"), Hint::AllUsable).await.unwrap();
        open(&cache, &url("two"), Hint::AllUsable).await.unwrap();
        assert_eq!(server.requests().len(), before, "the two most recent are still cached");
        open(&cache, &url("one"), Hint::AllUsable).await.unwrap();
        assert!(server.requests().len() > before, "the least recently used entry was evicted");

        // An entry larger than the whole cache is not kept
        let tiny = cache_with(|c| c.capacity_bytes = weight / 2);
        open(&tiny, &url("tiny"), Hint::AllUsable).await.unwrap();
        assert_eq!(tiny.stats().headers.entries, 0);
    }

    #[tokio::test]
    async fn entries_expire_after_their_time_to_live() {
        let ttl = Duration::from_millis(60);
        let wait = Duration::from_millis(150);

        // A validated source uses `ttl`
        let validated = server(object(512, pattern_a, "\"a1\""));
        let url = format!("{}/v.tif", validated.base());
        let cache = cache_with(|c| {
            c.ttl = ttl;
            c.ttl_unvalidated = Duration::from_secs(3600);
        });
        open(&cache, &url, Hint::AllUsable).await.unwrap();
        tokio::time::sleep(wait).await;
        let before = validated.requests().len();
        open(&cache, &url, Hint::AllUsable).await.unwrap();
        assert!(validated.requests().len() > before);
        assert_eq!(cache.stats().headers.expirations, 1);

        // A source without any validator uses `ttl_unvalidated`
        let blind = server(ServedObject::new(build_cog(&spec(512, pattern_a))).etag(None).modified(None));
        let url = format!("{}/b.tif", blind.base());
        let cache = cache_with(|c| {
            c.ttl = Duration::from_secs(3600);
            c.ttl_unvalidated = ttl;
        });
        open(&cache, &url, Hint::AllUsable).await.unwrap();
        tokio::time::sleep(wait).await;
        let before = blind.requests().len();
        open(&cache, &url, Hint::AllUsable).await.unwrap();
        assert!(blind.requests().len() > before);

        // Immutable sources never expire (and send no conditions)
        let fixed = server(object(512, pattern_a, "\"a1\""));
        let url = format!("{}/i.tif", fixed.base());
        let cache = cache_with(|c| {
            c.ttl = ttl;
            c.ttl_unvalidated = ttl;
            c.validation = Validation::Immutable;
        });
        open(&cache, &url, Hint::AllUsable).await.unwrap().read_tile_async(0).await.unwrap();
        tokio::time::sleep(wait).await;
        let before = fixed.requests().len();
        let reader = open(&cache, &url, Hint::AllUsable).await.unwrap();
        assert_eq!(fixed.requests().len(), before);
        reader.read_tile_async(1).await.unwrap();
        assert!(fixed.requests().iter().all(|r| r.if_match.is_none()));
    }

    #[tokio::test]
    async fn reopening_a_replaced_object_evicts_the_stale_entry_once() {
        let server = server(object(512, pattern_a, "\"v1\""));
        let url = format!("{}/r.tif", server.base());
        let cache = cache_with(|_| {});
        let old = open(&cache, &url, Hint::AllUsable).await.unwrap();
        assert_eq!(old.metadata.width, 512);

        server.replace_default(Some(object(576, pattern_b, "\"v2\"")));
        let err = old.io().read_range(30_000, 16).await.unwrap_err();
        assert!(crate::async_io::is_source_changed(&*err), "{err}");
        // The cache still believes in v1 until a read has shown otherwise
        assert_eq!(open(&cache, &url, Hint::AllUsable).await.unwrap().metadata.width, 512);

        let before = server.requests().len();
        let fresh = futures::future::join_all((0..8).map(|_| old.reopen_after_change())).await;
        let fresh: Vec<CogReader> = fresh.into_iter().map(Result::unwrap).collect();
        assert!(fresh.iter().all(|r| r.metadata.width == 576));
        assert!(fresh.iter().all(|r| Arc::ptr_eq(&r.metadata, &fresh[0].metadata)));
        assert_eq!(server.requests().len() - before, 1, "8 concurrent reopens, one request");
        let stats = cache.stats().headers;
        assert_eq!((stats.stale_evictions, stats.entries), (1, 1));

        let tile = fresh[0].read_tile_async(0).await.unwrap();
        assert!((tile[0] - pattern_b(0, 0, 0) as f32).abs() < 1e-3);
    }

    #[tokio::test]
    async fn a_missing_object_is_remembered_briefly() {
        let server = ObjectServer::start(None, Duration::ZERO);
        let url = format!("{}/missing.tif", server.base());
        let cache = cache_with(|c| c.negative_ttl = Duration::from_millis(150));

        let err = open(&cache, &url, Hint::AllUsable).await.err().unwrap();
        assert!(err.downcast_ref::<SourceNotFound>().is_some(), "{err}");
        assert!(err.to_string().contains("Failed to open"), "{err}");
        assert_eq!(server.requests().len(), 1);

        let again = open(&cache, &url, Hint::AllUsable).await.err().unwrap();
        assert!(again.downcast_ref::<SourceNotFound>().is_some());
        assert_eq!(again.to_string(), err.to_string());
        assert_eq!(server.requests().len(), 1, "remembered: no request");
        assert_eq!(cache.stats().headers.negative_hits, 1);

        // After the TTL the object is asked for again, and once it exists it is cached
        tokio::time::sleep(Duration::from_millis(250)).await;
        server.replace_default(Some(object(512, pattern_a, "\"a1\"")));
        open(&cache, &url, Hint::AllUsable).await.unwrap();
        assert_eq!(server.requests().len(), 2);
        open(&cache, &url, Hint::AllUsable).await.unwrap();
        assert_eq!(server.requests().len(), 2);

        // A zero TTL turns it off
        let off = cache_with(|c| c.negative_ttl = Duration::ZERO);
        let gone = ObjectServer::start(None, Duration::ZERO);
        let url = format!("{}/gone.tif", gone.base());
        for _ in 0..2 {
            assert!(open(&off, &url, Hint::AllUsable).await.is_err());
        }
        assert_eq!(gone.requests().len(), 2);
    }

    #[tokio::test]
    async fn an_incomplete_quality_analysis_is_not_remembered() {
        let server = server(object(512, pattern_a, "\"a1\""));
        let url = format!("{}/flaky.tif", server.base());
        let cache = cache_with(|_| {});

        // Tile data (beyond the header prefix) fails: the open still succeeds, as it always did,
        // with the sparse-looking fallback for this reader only.
        server.fail_from(Some(crate::remote::PREFIX_BYTES as usize));
        let degraded = open(&cache, &url, Hint::ComputeAtRuntime).await.unwrap();
        assert_eq!(degraded.min_usable_overview, None);

        server.fail_from(None);
        let before = server.requests().len();
        let healthy = open(&cache, &url, Hint::ComputeAtRuntime).await.unwrap();
        assert_eq!(healthy.min_usable_overview, Some(1));
        assert!(server.requests().len() > before, "the analysis ran again");

        let before = server.requests().len();
        assert_eq!(open(&cache, &url, Hint::ComputeAtRuntime).await.unwrap().min_usable_overview, Some(1));
        assert_eq!(server.requests().len(), before, "and its complete result is cached");
    }

    #[tokio::test]
    async fn invalidate_clear_bypass_refresh_and_disabled() {
        let server = server(object(512, pattern_a, "\"a1\""));
        let url = format!("{}/m.tif", server.base());
        let cache = cache_with(|_| {});

        let reader = open(&cache, &url, Hint::AllUsable).await.unwrap();
        reader.read_tile_async(0).await.unwrap();
        assert_eq!(cache.stats().headers.entries, 1);
        assert_eq!(cache.invalidate(&format!("{url}?ignored=1")), 1, "the query string is not part of the source");
        assert_eq!(cache.stats().headers.entries, 0);
        let n = server.requests().len();
        open(&cache, &url, Hint::AllUsable).await.unwrap();
        assert!(server.requests().len() > n, "invalidated: the next open reads the source");

        // Bypass neither reads nor fills the cache
        let n = server.requests().len();
        let bypass = CogReader::builder(&url).cache(&cache).cache_mode(CacheMode::Bypass).io_options(options());
        bypass.clone().hint(Hint::AllUsable).open_async().await.unwrap();
        assert!(server.requests().len() > n);
        assert_eq!(cache.stats().headers.hits, 0);

        // Refresh reads the source and replaces the entry
        let n = server.requests().len();
        let refresh = CogReader::builder(&url).cache(&cache).cache_mode(CacheMode::Refresh).io_options(options());
        refresh.hint(Hint::AllUsable).open_async().await.unwrap();
        assert!(server.requests().len() > n);
        assert_eq!(cache.stats().headers.entries, 1);

        cache.clear();
        assert_eq!(cache.stats().headers.entries, 0);
        assert_eq!(cache.stats().headers.bytes, 0);

        // A disabled cache caches nothing, and its readers still validate their reads
        let off = CogCache::disabled();
        for tile in [2, 3] {
            let n = server.requests().len();
            let reader = open(&off, &url, Hint::AllUsable).await.unwrap();
            assert!(server.requests().len() > n);
            reader.read_tile_async(tile).await.unwrap();
        }
        assert_eq!(off.stats().headers.entries, 0);
        assert_eq!(server.requests().last().unwrap().if_match.as_deref(), Some("\"a1\""));
    }

    #[tokio::test]
    async fn presigned_urls_share_an_entry_and_each_read_uses_its_own_query() {
        let server = server(object(512, pattern_a, "\"a1\""));
        let url = |sig: &str| format!("{}/p.tif?sig={sig}", server.base());
        let cache = cache_with(|_| {});

        open(&cache, &url("one"), Hint::AllUsable).await.unwrap();
        let before = server.requests().len();
        let second = open(&cache, &url("two"), Hint::AllUsable).await.unwrap();
        assert_eq!(server.requests().len(), before, "same object, same entry");
        second.read_tile_async(0).await.unwrap();
        let read = server.requests().pop().unwrap();
        assert_eq!(read.query.as_deref(), Some("sig=two"), "{read:?}");
    }

    #[tokio::test]
    async fn local_files_are_not_cached() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("local.tif");
        std::fs::write(&path, build_cog(&spec(256, pattern_a))).unwrap();
        let cache = cache_with(|_| {});
        for _ in 0..2 {
            let reader = CogReader::builder(path.to_str().unwrap()).cache(&cache).hint(Hint::NoneUsable).open_async().await.unwrap();
            assert_eq!(reader.metadata.width, 256);
        }
        let stats = cache.stats().headers;
        assert_eq!((stats.entries, stats.hits, stats.misses), (0, 0, 0));
    }
}
