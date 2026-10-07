//! Process-wide cache of decoded (decompressed, f32) source tiles.
//!
//! Byte-bounded LRU: 512 MiB by default (`COGRS_TILE_CACHE_MB` overrides it at first use, or
//! call [`set_capacity`]). Tiles are keyed by the *source identity*, the object's identifier plus
//! its version token (see [`source_id`]), so an object that was replaced can never be served
//! from tiles decoded from its predecessor. [`invalidate_source`] drops every tile of a source,
//! whatever its version.
//!
//! Hit, miss and eviction counters are atomics outside the lock.

use lru::LruCache;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

const DEFAULT_CAPACITY_BYTES: usize = 512 * 1024 * 1024;

/// Separates the identifier from the version in a source identity.
const VERSION_SEPARATOR: char = '\u{1f}';

/// Source identity used as the tile-cache key: `identifier`, or `identifier` + separator +
/// `version` for sources that report a version (ETag, modification time).
#[must_use]
pub fn source_id(identifier: &str, version: Option<&str>) -> Arc<str> {
    match version {
        Some(version) => format!("{identifier}{VERSION_SEPARATOR}{version}").into(),
        None => identifier.into(),
    }
}

/// Key for cached decompressed tiles
#[derive(Clone, Eq, PartialEq, Hash)]
struct TileKey {
    /// Source identity ([`source_id`])
    source: Arc<str>,
    /// Tile index within the IFD
    tile_index: u32,
    /// Overview index (None = full resolution, Some(n) = overview n)
    overview_idx: Option<u16>,
}

impl TileKey {
    fn new(source: &Arc<str>, tile_index: usize, overview_idx: Option<usize>) -> Self {
        // Casts are safe: tile counts and overview counts are typically small
        #[allow(clippy::cast_possible_truncation)]
        TileKey {
            source: Arc::clone(source),
            tile_index: tile_index as u32,
            overview_idx: overview_idx.map(|i| {
                #[allow(clippy::cast_possible_truncation)]
                { i as u16 }
            }),
        }
    }
}

struct CacheEntry {
    data: Arc<Vec<f32>>,
    size_bytes: usize,
}

/// The tile cache's entries and byte accounting (behind the process-wide lock).
pub struct TileCache {
    current_bytes: usize,
    capacity_bytes: usize,
    entries: LruCache<TileKey, CacheEntry>,
}

impl TileCache {
    fn new(capacity_bytes: usize) -> Self {
        TileCache { current_bytes: 0, capacity_bytes, entries: LruCache::unbounded() }
    }

    fn get(&mut self, key: &TileKey) -> Option<Arc<Vec<f32>>> {
        self.entries.get(key).map(|entry| Arc::clone(&entry.data))
    }

    fn contains(&self, key: &TileKey) -> bool {
        self.entries.contains(key)
    }

    /// Evict least recently used entries until `needed` more bytes fit.
    fn make_room(&mut self, needed: usize) {
        while self.current_bytes + needed > self.capacity_bytes {
            let Some((_key, entry)) = self.entries.pop_lru() else { break };
            self.current_bytes = self.current_bytes.saturating_sub(entry.size_bytes);
            EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn insert(&mut self, key: TileKey, data: Arc<Vec<f32>>, size_bytes: usize) {
        if size_bytes > self.capacity_bytes {
            return;
        }
        if let Some(old) = self.entries.pop(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.size_bytes);
        }
        self.make_room(size_bytes);
        self.current_bytes = self.current_bytes.saturating_add(size_bytes);
        self.entries.put(key, CacheEntry { data, size_bytes });
    }

    /// Remove every entry whose source is `identifier` or a version of it.
    fn remove_source(&mut self, identifier: &str) -> usize {
        let versioned = format!("{identifier}{VERSION_SEPARATOR}");
        let doomed: Vec<TileKey> = self
            .entries
            .iter()
            .filter(|(key, _)| &*key.source == identifier || key.source.starts_with(&versioned))
            .map(|(key, _)| key.clone())
            .collect();
        for key in &doomed {
            if let Some(entry) = self.entries.pop(key) {
                self.current_bytes = self.current_bytes.saturating_sub(entry.size_bytes);
            }
        }
        doomed.len()
    }
}

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);

static TILE_CACHE: LazyLock<Mutex<TileCache>> = LazyLock::new(|| {
    let capacity = std::env::var("COGRS_TILE_CACHE_MB")
        .ok()
        .and_then(|mb| mb.trim().parse::<usize>().ok())
        .map_or(DEFAULT_CAPACITY_BYTES, |mb| mb.saturating_mul(1024 * 1024));
    Mutex::new(TileCache::new(capacity))
});

/// Get a cached tile by source identity ([`source_id`]), tile index, and optional overview index
/// (`None` for full resolution, `Some(n)` for overview n).
pub(crate) fn get_shared(source: &Arc<str>, tile_index: usize, overview_idx: Option<usize>) -> Option<Arc<Vec<f32>>> {
    let key = TileKey::new(source, tile_index, overview_idx);
    let found = TILE_CACHE.lock().get(&key);
    if found.is_some() {
        HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        MISSES.fetch_add(1, Ordering::Relaxed);
    }
    found
}

/// Insert a decompressed tile under a source identity.
pub(crate) fn insert_shared(source: &Arc<str>, tile_index: usize, overview_idx: Option<usize>, data: Arc<Vec<f32>>) {
    let size_bytes = data.len() * std::mem::size_of::<f32>();
    let key = TileKey::new(source, tile_index, overview_idx);
    TILE_CACHE.lock().insert(key, data, size_bytes);
}

/// Get a cached tile by source identifier, tile index, and optional overview index
/// - `source`: File path or URL identifying the COG (version-less identity)
/// - `tile_index`: Tile index within the IFD
/// - `overview_idx`: None for full resolution, Some(n) for overview n
pub fn get(source: &str, tile_index: usize, overview_idx: Option<usize>) -> Option<Arc<Vec<f32>>> {
    get_shared(&Arc::from(source), tile_index, overview_idx)
}

/// Check if a tile is cached
pub fn contains(source: &str, tile_index: usize, overview_idx: Option<usize>) -> bool {
    let key = TileKey::new(&Arc::from(source), tile_index, overview_idx);
    TILE_CACHE.lock().contains(&key)
}

/// Insert a decompressed tile into the cache
pub fn insert(source: &str, tile_index: usize, overview_idx: Option<usize>, data: Arc<Vec<f32>>) {
    insert_shared(&Arc::from(source), tile_index, overview_idx, data);
}

// ============================================================================
// Path-based API for tiff_chunked.rs and lzw_fallback.rs
// ============================================================================

/// Get cached tile by file path (used by tiff_chunked and lzw_fallback modules)
#[must_use]
pub fn get_by_path(path: &Path, index: usize) -> Option<Arc<Vec<f32>>> {
    let source = path.to_string_lossy();
    get(&source, index, None)
}

/// Check if tile is cached by file path
#[must_use]
pub fn contains_by_path(path: &Path, index: usize) -> bool {
    let source = path.to_string_lossy();
    contains(&source, index, None)
}

/// Insert tile into cache by file path
pub fn insert_by_path(path: &Path, index: usize, data: Arc<Vec<f32>>) {
    let source = path.to_string_lossy();
    insert(&source, index, None, data);
}

/// Drop every cached tile of the source with this `identifier`, whatever its version. Returns
/// the number of tiles removed.
pub fn invalidate_source(identifier: &str) -> usize {
    TILE_CACHE.lock().remove_source(identifier)
}

/// Drop every cached tile.
pub fn clear() {
    let mut cache = TILE_CACHE.lock();
    cache.entries.clear();
    cache.current_bytes = 0;
}

/// Set the capacity in bytes, evicting least recently used tiles if the cache is now over it.
pub fn set_capacity(capacity_bytes: usize) {
    let mut cache = TILE_CACHE.lock();
    cache.capacity_bytes = capacity_bytes;
    cache.make_room(0);
}

/// Point-in-time statistics of the tile cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileCacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub capacity_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    /// Tiles evicted to stay within the capacity.
    pub evictions: u64,
}

/// Current statistics. Only the entry count and byte total take the cache lock briefly; the
/// counters are atomics.
#[must_use]
pub fn snapshot() -> TileCacheStats {
    let (entries, bytes, capacity_bytes) = {
        let cache = TILE_CACHE.lock();
        (cache.entries.len(), cache.current_bytes, cache.capacity_bytes)
    };
    TileCacheStats {
        entries,
        bytes,
        capacity_bytes,
        hits: HITS.load(Ordering::Relaxed),
        misses: MISSES.load(Ordering::Relaxed),
        evictions: EVICTIONS.load(Ordering::Relaxed),
    }
}

/// Cache statistics: (`entry_count`, `current_bytes`, `capacity_bytes`, hits, misses)
#[must_use]
pub fn stats() -> (usize, usize, usize, u64, u64) {
    let s = snapshot();
    (s.entries, s.bytes, s.capacity_bytes, s.hits, s.misses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(len: usize) -> Arc<Vec<f32>> {
        Arc::new(vec![1.0; len])
    }

    #[test]
    fn versions_are_separate_identities() {
        let (a, b) = (source_id("tc://obj", Some("\"v1\"")), source_id("tc://obj", Some("\"v2\"")));
        insert_shared(&a, 0, None, tile(4));
        assert!(get_shared(&a, 0, None).is_some());
        assert!(get_shared(&b, 0, None).is_none(), "a new version must not see the old version's tiles");
        assert!(get_shared(&source_id("tc://obj", None), 0, None).is_none());
    }

    #[test]
    fn invalidate_source_removes_all_versions_and_only_that_source() {
        let (v1, v2) = (source_id("tc://inv/a", Some("1")), source_id("tc://inv/a", Some("2")));
        let other = source_id("tc://inv/ab", Some("1"));
        for id in [&v1, &v2, &other] {
            insert_shared(id, 3, Some(1), tile(8));
        }
        assert_eq!(invalidate_source("tc://inv/a"), 2);
        assert!(get_shared(&v1, 3, Some(1)).is_none() && get_shared(&v2, 3, Some(1)).is_none());
        assert!(get_shared(&other, 3, Some(1)).is_some(), "a source sharing the prefix is untouched");
    }

    #[test]
    fn evicts_least_recently_used_by_bytes() {
        let mut cache = TileCache::new(100 * 4);
        let id: Arc<str> = "tc://evict".into();
        let key = |i| TileKey::new(&id, i, None);
        for i in 0..3 {
            cache.insert(key(i), tile(40), 40 * 4);
        }
        // 40 + 40 fit, the third evicted the oldest
        assert_eq!(cache.entries.len(), 2);
        assert!(!cache.contains(&key(0)) && cache.contains(&key(1)) && cache.contains(&key(2)));
        assert!(cache.get(&key(1)).is_some()); // 1 is now the most recent
        cache.insert(key(3), tile(40), 40 * 4);
        assert!(cache.contains(&key(1)) && !cache.contains(&key(2)));
        cache.insert(key(9), tile(1000), 1000 * 4); // larger than the capacity: skipped
        assert!(!cache.contains(&key(9)));
        assert_eq!(cache.current_bytes, 2 * 40 * 4);
    }

    #[test]
    fn counters_and_snapshot_track_hits_and_misses() {
        let id = source_id("tc://counters", Some("x"));
        let before = snapshot();
        assert!(get_shared(&id, 0, None).is_none());
        insert_shared(&id, 0, None, tile(2));
        assert!(get_shared(&id, 0, None).is_some());
        let after = snapshot();
        assert!(after.misses > before.misses && after.hits > before.hits);
        assert!(after.entries >= 1 && after.bytes >= 8);
    }
}
