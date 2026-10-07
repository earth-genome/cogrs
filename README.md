# cogrs

Pure Rust COG (Cloud Optimized `GeoTIFF`) reader library.

## Features

- [Local, HTTP, and S3 sources](#sources)
- [Point queries](#point-queries)
- [XYZ tile extraction](#tile-extraction)
- [Header and tile caching](#caching)
- [Coordinate transforms](#coordinate-transforms)
- [Compression: DEFLATE, LZW, ZSTD, JPEG, WebP](#compression)

## Quick Start

```rust,no_run
use cogrs::{CogReader, PointQuery, TileExtractor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let reader = CogReader::open_async("path/to/file.tif").await?;

    // Point query (async: the containing tile is fetched without blocking a thread)
    let result = reader.sample_lonlat_async(-122.4, 37.8).await?;

    // XYZ tile extraction (async)
    let tile = TileExtractor::new(&reader)
        .xyz(10, 163, 395)
        .extract()
        .await?;

    Ok(())
}
```

## Sources

Remote I/O (S3, HTTP(S)) is natively asynchronous. From async code (axum/tokio handlers, etc.)
open with `CogReader::open_async` and extract with `TileExtractor`: waiting on the network
never occupies a thread, and a tile that needs several source tiles fetches them
concurrently (adjacent byte ranges are merged into one request, identical in-flight tiles
are shared between concurrent requests). Only decoding and resampling use tokio's blocking
pool. Works on `multi_thread` and `current_thread` runtimes.

```rust,no_run
use cogrs::CogReader;
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
// Local file
let reader = CogReader::open_async("path/to/file.tif").await?;

// HTTP
let reader = CogReader::open_async("https://example.com/file.tif").await?;

// S3 (uses AWS_* environment variables for credentials)
let reader = CogReader::open_async("s3://bucket/path/to/file.tif").await?;
# Ok(())
# }
```

S3 configuration: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for credentials,
`AWS_SKIP_SIGNATURE=true` for anonymous access to public buckets, `AWS_ENDPOINT_URL` /
`AWS_ALLOW_HTTP` for MinIO and other S3-compatible stores. The region is taken from
`AWS_REGION`, then `AWS_DEFAULT_REGION`; if neither is set it is detected from the bucket
(once per bucket per process) unless a custom endpoint is configured.

**BigTIFF.** COGs larger than 4 GiB are stored as BigTIFF (TIFF version 43: 8-byte offsets,
20-byte IFD entries, `LONG8` tile offsets); `CogReader` reads them exactly like classic TIFF
COGs, local or remote, sync or async, through the same header and tile caches. A BigTIFF's
tile arrays are twice as large, so opening one reads about 1.4x the header bytes of the same
COG as a classic TIFF (and, once the arrays outgrow the 16 KiB open request, up to one more
request); the header cache makes that a one-time cost. The local-file `TiffChunkedRasterSource`
and `LzwRasterSource` parse classic TIFF only and return an error naming BigTIFF.

Opening a remote COG is one ranged request (header, IFDs, size and ETag together), and every
COG in the same bucket or host shares one HTTP client. Pass an `OverviewQualityHint` (see
`open_async_with_hint`) to skip sampling tiles from the coarsest overview at open, and
`IoOptions` (see `open_async_with_options`) to tune the concurrency limits, range
coalescing, retries and timeouts:

```rust,no_run
use cogrs::{CogReader, IoOptions, OverviewQualityHint};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let options = IoOptions { max_concurrent_requests: 8, ..IoOptions::default() };
let reader = CogReader::open_async_with_options(
    "s3://bucket/path/to/file.tif",
    OverviewQualityHint::AllUsable,
    &options,
)
.await?;
// `CogReader` is cheap to clone (metadata is shared), so keep one per source and clone it
// into handlers.
let for_handler = reader.clone();
# Ok(())
# }
```

Plain threads (no async runtime) can use the synchronous API: `CogReader::open`,
`read_tile`, `sample_lonlat`, ... Remote sources work there too; the request runs on a private
I/O runtime while the calling thread blocks. On an async worker thread those calls block that
worker, so use the `*_async` methods (or `reader.spawn_blocking(..)` for other sync work).

## Point Queries

Async code uses `sample_lonlat_async` / `sample_crs_async` / `sample_points_lonlat_async`;
the synchronous `PointQuery` trait below is for plain threads.

```rust,no_run
use cogrs::{CogReader, PointQuery};
# fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open("elevation.tif")?;

// Sample at lon/lat
let result = reader.sample_lonlat(-122.4, 37.8)?;
for (band, value) in &result.values {
    println!("Band {band}: {value}");
}

// Sample in specific CRS (e.g., UTM zone 10N)
let result = reader.sample_crs(32610, 551000.0, 4185000.0)?;
# Ok(())
# }
```

## Tile Extraction

```rust,no_run
use cogrs::{CogReader, TileExtractor, ResamplingMethod};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open_async("imagery.tif").await?;

// Simple extraction (256x256)
let tile = TileExtractor::new(&reader)
    .xyz(10, 163, 395)
    .extract()
    .await?;

// With options
let tile = TileExtractor::new(&reader)
    .xyz(10, 163, 395)
    .output_size(512, 512)
    .resampling(ResamplingMethod::Bilinear)
    .bands(&[0, 1, 2])
    .extract()
    .await?;
# Ok(())
# }
```

## Caching

A tile server opens the same image for every request, so two caches sit behind
`CogReader::open*` / `TileExtractor`:

- **Header cache** (`CogCache`, remote sources only): the parsed header of each source, its
  overview metadata, what the server reported about the object (size, ETag, last-modified)
  and the computed overview-quality hint. Opening a cached source sends **no request**, and
  the sampling that `OverviewQualityHint::ComputeAtRuntime` does runs once per source, not once
  per open. Concurrent first opens of one source share a single request. Local files are not
  cached (parsing them is cheap).
- **Decoded tile cache** (process-wide, byte-bounded LRU): decompressed source tiles, keyed by
  the source *and its version* (ETag, or size and modification time for local files), so a
  replaced object can never be served from tiles decoded from its predecessor.

| Setting | Default | |
|---|---|---|
| `CacheConfig::capacity_bytes` | 64 MiB | header entries weigh about 1.5 KiB + 16 B per tile: ~12 KB for a 10980 x 10980 COG with 512 px tiles, 0.8 to 3.3 MB for 100k x 100k |
| `CacheConfig::ttl` | 1 hour | entries whose reads can be validated (below) |
| `CacheConfig::ttl_unvalidated` | 60 seconds | entries from servers that give no usable validator |
| `CacheConfig::negative_ttl` | 5 seconds | how long a `404` is remembered (`0` = off); `401`/`403`, server errors and timeouts never are |
| `CacheConfig::validation` | `Validation::IfMatch` | see below |
| tile cache capacity | 512 MiB | `tile_cache::set_capacity`, or `COGRS_TILE_CACHE_MB` |

**When the object is overwritten.** Cached headers are only valid for the version they were read
from. With the default `Validation::IfMatch`, every range request after the open carries the
object's ETag as `If-Match` (or `If-Unmodified-Since` when the server has a `Last-Modified` but no
strong ETag; weak ETags are never sent). A replaced object answers `412`, and the operation
recovers on its own: the stale header entry and the old version's decoded tiles are dropped, the
object is opened again (once, shared by all concurrent callers), and the extraction or point query
runs again against the new header, once. The conditions travel with the requests that are sent
anyway; they cost no round trip. `S3` honours them (verified against the real bucket by an
ignored test), and so do HTTP servers that implement RFC 9110 `If-Match`.

What it cannot detect: a server that ignores `If-Match`, or gives neither ETag nor
`Last-Modified`, can only be caught when the entry's time to live runs out (`ttl_unvalidated`,
60 s, for the latter). For objects that are never overwritten (date-versioned keys),
`Validation::Immutable` sends no conditions and never expires entries; `Validation::Ttl`
sends no conditions and expires entries after `ttl`. Low-level `read_tile*` calls report
`SourceChanged` instead of retrying (a tile index only means something for the header it came
from); a reader you keep for a long time stays pinned to the version it opened, like an open file
handle, and every operation on it then costs one `412` plus a cached reopen: reopen the source
(`open_async`) when you know it changed, or call `reader.reopen_after_change()`.

```rust,no_run
use cogrs::{CacheConfig, CacheMode, CogCache, CogReader, Validation};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
// The global cache is used automatically; `COGRS_HEADER_CACHE=off` disables it, and
// `COGRS_HEADER_CACHE_MB` sizes it.
let reader = CogReader::open_async("s3://bucket/dated/2026-09-01/tci.tif").await?;

// Your own cache (sized, immutable keys), passed explicitly; or `CogCache::disabled()`
let cache = CogCache::new(CacheConfig { validation: Validation::Immutable, ..CacheConfig::default() });
let reader = CogReader::builder("s3://bucket/dated/2026-09-01/tci.tif").cache(&cache).open_async().await?;
// `CacheMode::Bypass` / `CacheMode::Refresh` per open
let fresh = CogReader::builder("s3://bucket/live.tif").cache_mode(CacheMode::Refresh).open_async().await?;

// You overwrote the object and want the next open to see it: drop it
CogCache::global().invalidate("s3://bucket/live.tif");

// Metrics: header hits/misses/evictions/expirations/coalesced opens/412 evictions/negative hits,
// decoded-tile hits/misses/evictions, and network requests and bytes
let stats = CogCache::global().stats();
println!("{} header hits, {} tile hits", stats.headers.hits, stats.tiles.hits);
println!("{:?}", cogrs::io_stats());
# Ok(())
# }
```

`stats()` costs a couple of atomic loads and one short lock; the counters are atomics, so reading
them or counting a hit adds nothing to the hot path beyond the locks the caches already take.

**Presigned and SAS URLs.** The query string is not part of a source's identity: two presigned
URLs for one object share one header entry and the same decoded tiles, and each reader sends
*its own* query string with its requests. That is what makes caching useful with per-request
signed URLs, but it also means a cached header or tile can be served to a caller whose URL has
expired or was never valid; access control stays with whoever hands out the URLs (S3 sources
include the access key in the cache key, so credentials are not shared). The decoded tile cache
has always worked this way. Use `CacheMode::Bypass` or a separate `CogCache` where that is not
acceptable.

## WebP Output

Extracted tiles encode to lossless RGBA8 WebP (pure Rust). Pixels equal to the
COG's nodata value (in all bands) or NaN become transparent; 1-band tiles are
expanded to gray, 3-band to RGB, 4-band keeps its alpha. Values are rounded and
clamped to `0..=255`; use `WebpOptions::rescale` for 16-bit/float data.
Areas of a tile outside the COG extent are filled with the COG's nodata value
(or `NaN` if it declares none), so edge tiles come out transparent there.

```rust,no_run
use cogrs::{CogReader, TileExtractor, WebpOptions};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open_async("imagery.tif").await?;
let tile = TileExtractor::new(&reader).xyz(10, 163, 395).extract().await?;

let webp: Vec<u8> = tile.to_webp()?;

// 16-bit source: map 0..3000 linearly onto 0..255, override nodata
let opts = WebpOptions { rescale: Some((0.0, 3000.0)), nodata: Some(0.0) };
let webp = tile.to_webp_with(&opts)?;
# Ok(())
# }
```

## Coordinate Transforms

```rust
use cogrs::{CoordTransformer, project_point};

// One-off transform
let (x, y) = project_point(4326, 3857, -122.4, 37.8)?;

// Reusable transformer
let transformer = CoordTransformer::new(4326, 32610)?;
let (utm_x, utm_y) = transformer.transform(-122.4, 37.8)?;
# Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
```

## Compression

Supported formats are detected automatically:

- DEFLATE
- LZW (8/16-bit, predictors 1-3)
- ZSTD
- JPEG
- WebP

## License

MIT
