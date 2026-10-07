# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `ResamplingMethod::Cubic`: Catmull-Rom cubic convolution (Keys, a = -0.5), the kernel of
  `gdalwarp -r cubic`, both for the fixed 4x4 footprint and, stretched by the downsampling
  ratio, when an output pixel covers several source pixels. Masking is the same as for
  `Bicubic`. Against `gdalwarp -r cubic` the output is 100% bit-identical on a same-CRS
  synthetic raster downsampled by 1.5 to 4, and within 0.0005 (float rounding) when
  upsampling. `ResamplingMethod::Bicubic` is unchanged (Mitchell-Netravali, B = C = 1/3; no
  `gdalwarp` method uses it) and its docs now say so. `ResamplingMethod` is not
  `#[non_exhaustive]`, so an exhaustive `match` on it needs a `Cubic` arm: a minor breaking
  change for such callers
- `TileData::to_webp()` / `TileData::to_webp_with(&WebpOptions)`: lossless RGBA8 WebP
  encoding of extracted tiles (pure Rust, via the `image` crate). Supports 1 (gray),
  3 (RGB) and 4 (RGBA) bands; nodata/NaN pixels become transparent; values are rounded
  and clamped to `0..=255`, with an optional explicit `(min, max)` linear rescale
- `WebpOptions` (`nodata` override, `rescale`), re-exported from the crate root
- `TileData::nodata`: the source COG's nodata value, set at extraction time
- Native async remote I/O. `AsyncRangeReader` (object-safe, `BoxFuture`-based) is the core
  I/O trait; `ObjectStoreRangeReader` serves `s3://` and `http(s)://` through `object_store`
  (one implementation for both, retries and timeouts included), `create_async_range_reader`
  builds one for any source, and `SyncToAsync` / `AsyncToSync` adapt between it and the
  synchronous `RangeReader`. Remote I/O no longer occupies a tokio blocking thread while
  waiting on the network
- `IoOptions`: fan-out cap per fetch (16), in-flight cap per bucket/host (128), range
  coalescing (gap 32 KiB, max 4 MiB), and retry/timeout settings (3 retries within 15 s,
  5 s connect, 20 s request). Pass it with `CogReader::open_async_with_options`
- Opening a remote COG is one ranged `GET` (header, IFDs, size, ETag; no `HEAD`, which also
  makes presigned URLs usable) followed by one concurrent round trip for large tile-offset
  arrays. HTTP(S) URLs may carry a query string (sent with every request; such URLs get a
  private client). One `object_store` client is shared per bucket/host, so repeated opens
  reuse connections. Requests are spawned on a private multi-thread I/O runtime (2-8
  threads) and awaited from the caller's runtime, so the shared client's connections do not
  depend on any caller runtime staying alive, and dropping a request future aborts the request
- `TileExtractor`/`Reprojector` fetch all source tiles an output tile needs concurrently:
  cache lookups, then one coalesced request set (adjacent tiles merge into one request),
  with identical in-flight tiles de-duplicated across concurrent callers (cancellation
  safe). Only decode and resampling run on the blocking pool
- `CogReader` is `Clone` and cheap to clone: `metadata` is `Arc<CogMetadata>` and
  `overviews` is `Arc<[OverviewMetadata]>`, so parsed metadata can be cached per source and
  re-attached with `CogReader::from_parts(reader, metadata, overviews, min_usable_overview)`
- `CogReader::open_async()` / `open_async_with_hint()` / `open_async_with_options()`,
  `CogReader::from_async_reader()` / `from_async_reader_with_hint()`,
  `CogReader::read_tile_async()` / `read_overview_tile_async()`,
  `CogReader::compute_overview_quality_hint_async()`, `CogReader::identifier()` and
  `CogReader::is_remote()`. Overview-quality sampling at open fetches its tiles concurrently
- Async point queries: `CogReader::sample_async()`, `sample_crs_async()`,
  `sample_lonlat_async()`, `sample_band_crs_async()`, `sample_points_crs_async()` and
  `sample_points_lonlat_async()` fetch the one containing tile without blocking a thread
- `CogReader::spawn_blocking(|reader| ..)`: run any sync operation (point queries, tile
  reads) on an opened reader on the blocking pool
- `RangeReader::reads_inline()` (default `false`): marks readers that never block (memory)
- `S3ScanOptions::skip_signature` (default from `AWS_SKIP_SIGNATURE`), `concurrency` and
  `io`
- `OverviewMetadata::scale_x` / `scale_y`: exact per-axis ratio of full-resolution size to
  overview size (see Fixed). **Breaking:** `OverviewMetadata` has two new public fields, and its
  integer `scale` is now `scale_x` rounded to the nearest integer (informational; it used to be
  floored). Construct `OverviewMetadata` through the reader, not with a struct literal
- S3 bucket region auto-detection: when no region is configured and no custom endpoint is
  set, the region is detected with an unauthenticated `HeadBucket` request and cached per
  bucket for the life of the process. Failure to detect falls back to `us-east-1` with a
  warning
- S3 open failures without credentials now mention `AWS_SKIP_SIGNATURE=true` for public buckets
- Header cache: `CogCache` remembers the parsed header, overview metadata, object identity
  (size, ETag, last-modified) and the computed overview-quality hint of remote COGs, so
  re-opening a source sends no request and `ComputeAtRuntime` sampling runs once per source.
  Concurrent first opens of one source share one request. Bounded by entry weight (64 MiB by
  default, about 1.5 KiB plus 16 B per tile per entry), with `CacheConfig` for capacity, `ttl`
  (1 h), `ttl_unvalidated` (60 s for servers without a usable validator) and `negative_ttl`
  (5 s, `404` only). `CogCache::global()` is used by every `CogReader::open*` for `s3://` and
  `http(s)://` sources (`COGRS_HEADER_CACHE=off` disables it, `COGRS_HEADER_CACHE_MB` sizes
  it); `CogCache::new`, `CogCache::disabled` and `CogReader::builder(src).cache(..)
  .cache_mode(CacheMode::{Use, Bypass, Refresh})` select another. `stats()` (`CacheStats`:
  hits, misses, evictions, expirations, coalesced opens, 412 evictions, negative hits, plus the
  decoded-tile cache), `invalidate(url)` and `clear()`. See the README's "Caching" section
- `Validation { IfMatch, Ttl, Immutable }`: how reads are tied to the cached version. The
  default sends the ETag as `If-Match` on every range request after the open (or
  `If-Unmodified-Since` without a strong ETag); an overwritten object answers `412`, reported as
  the typed `SourceChanged` error (`is_source_changed`)
- `CogReader::reopen_after_change()`, and `AsyncRangeReader::reopen()` / `version()`
  (`RangeReader::version()`): a source's version token (ETag, or size and mtime)
- `SourceNotFound` (a `404` while opening), `io_stats()` (process-wide network requests and
  bytes of the remote readers), `tile_cache::{set_capacity, clear, invalidate_source,
  invalidate_identity, snapshot}` and `TileCacheStats`, `COGRS_TILE_CACHE_MB`
- `bench_remote cache`: 64 concurrent open-per-request tiles through the header cache
- BigTIFF (TIFF version 43) COGs: the header (8-byte offsets, byte size 8, reserved 0), IFD
  tables with 64-bit entry counts and 20-byte entries, values of up to 8 bytes stored inline,
  `LONG8` tile offsets and byte counts (and single `LONG8` values such as sizes), and 64-bit
  next-IFD offsets, for local files and remote sources, the async and the synchronous open, and
  the header cache. Compared with the classic encoding of the same COG, decoded tiles, geometry
  and extraction are identical (tested on synthetic COGs in both `LONG` and `LONG8` array types
  and on GDAL-made COGs with `BIGTIFF=YES`: DEFLATE, LZW, ZSTD, Float32 `PixelIsPoint`), and a
  real public BigTIFF COG (ArcticDEM 2 m mosaic tile, LZW with the floating-point predictor) opens
  and reads a tile identical to GDAL's

### Changed

- **Breaking:** `TileData` has a new public field `nodata`; code constructing it with a
  struct literal must set it
- Opening a COG reads the tile offset/byte-count arrays (and GeoTIFF/GDAL tag values) of all
  IFDs with one coalesced `read_ranges` call instead of one request per tag per IFD, so a
  remote open whose arrays outgrow the 16 KiB prefix costs the prefix plus one request when
  they are contiguous (previously 6-13 requests). A failed read of an overview's arrays now
  fails the open instead of silently dropping that overview
- Output pixels with no source data (outside the COG extent, entirely outside tiles,
  sparse source tiles) are now filled with the COG's nodata value, or `NaN` if none is
  declared, instead of `0.0`. Unchanged for COGs with `nodata = 0`
- Bilinear and bicubic resampling no longer blend `NaN`, nodata or unavailable samples
  into neighbouring pixels; if any sample in the interpolation window is invalid the
  nearest source sample is used. This removes dark fringes along nodata boundaries
- Failures reading or decoding a source tile during extraction now return an error
  instead of silently leaving a hole in the output (and, for overviews, instead of
  retrying at full resolution). Sparse tiles (zero byte count) are still valid and
  produce nodata/`NaN` pixels
- S3 region precedence is now: explicit `S3Config::region` / `S3ScanOptions::region` >
  `AWS_REGION` > `AWS_DEFAULT_REGION` > auto-detect. Previously `AWS_DEFAULT_REGION` was
  ignored and the region defaulted to `us-east-1`, which failed with "Received redirect
  without LOCATION" for buckets in other regions. `S3Config::region` /
  `S3ScanOptions::region` are still `Option<String>`, but `None` (what `from_url` /
  `Default` yield when no env var is set) now means "auto-detect" instead of being
  pre-filled with `us-east-1`. Custom endpoints (`AWS_ENDPOINT_URL`, MinIO) still default
  to `us-east-1` and never trigger detection
- `CogReader::open` for `s3://` / `http(s)://`, `HttpRangeReader` and `S3RangeReaderSync` are
  synchronous wrappers over the async backend: the request runs on a private I/O runtime and
  the calling thread blocks until it completes. They no longer need an ambient tokio
  runtime and never panic or deadlock inside one (on an async worker thread they block that
  worker; from async code use `open_async`)
- `S3CogSource::scan` reads object metadata with bounded concurrency (default 16) instead of
  one object at a time, and no longer downloads overview tiles per object (entries only need
  the IFDs)
- `S3CogSource::scan` and S3 opens share one client per bucket
- The synchronous and asynchronous point queries share one implementation (pixel lookup,
  per-band extraction, result construction). `PointQuery::sample_crs` and `CogReader::sample`
  now read a tile once and no longer copy it per band
- Remote reads are conditional by default (see `Validation`), and opening a remote source goes
  through the global header cache. Opt out with `COGRS_HEADER_CACHE=off` (no header caching;
  reads stay conditional) or `Validation::Ttl` / `Validation::Immutable` (no conditions)
- Extraction (`TileExtractor`, `Reprojector` and its streaming/chunk forms) and point queries
  (`sample*`, `*_async`) that find the object replaced since the reader was opened run once more
  against a fresh open instead of failing. The tile-index level `read_tile*` methods return
  `SourceChanged`
- Decoded tiles are keyed by the source's identifier *and version* (ETag; size and mtime for local
  files) instead of the identifier alone; `ObjectStoreRangeReader::last_modified_unix` is `None`
  when the server sent no `Last-Modified` (it was the epoch)
- The decoded tile cache uses `parking_lot` (no poisoning panics), counts hits, misses and
  evictions in atomics, and its capacity is configurable
- IFD tables are validated while parsing: an entry count above 65 535 and a tag value that lies
  outside the file are errors (they used to be read as is), and an IFD larger than the first
  4 KiB read is read in full instead of being truncated. The synchronous chunked and LZW file
  readers report BigTIFF files as unsupported (`Invalid TIFF version: N` for other versions)
- When bilinear/bicubic downsample (see Fixed), `NaN`/nodata source pixels inside the kernel
  are left out and the remaining weights renormalised, as `gdalwarp` does, instead of falling
  back to the nearest sample; a pixel whose own nearest source pixel is invalid still takes
  that sample. Upsampling keeps the nearest-sample fallback

### Removed

- **Breaking:** `PrefixCachedRangeReader` and `RangeReader::has_prefix_cache()`. The remote
  reader keeps the 16 KiB prefix it fetched on open
- **Breaking:** `CogReader::clone_for_async()`; use `Clone`
- **Breaking:** `CogReader::metadata` is now `Arc<CogMetadata>` and `CogReader::overviews`
  is `Arc<[OverviewMetadata]>`. Field reads (`reader.metadata.width`) compile unchanged;
  mutating them or passing them where owned values are expected does not
- **Breaking:** `S3RangeReaderSync::new` / `from_async` and `create_range_reader` no longer
  require a tokio runtime context
- **Breaking:** `S3ScanOptions` has new public fields (`skip_signature`, `concurrency`, `io`);
  construct it with `..Default::default()`
- The legacy `S3RangeReader` (plain HTTPS, size 0, one new client per read), the
  `catch_unwind`-based `blocking_call` guard and the `reqwest` `blocking` dependency

### Fixed

- Downsampling bilinear/bicubic/cubic of a multi-band source with nodata judged the kernel's
  centre band by band: a pixel that is nodata in some bands only (an RGB pixel with blue 0 at
  nodata 0) was passed through as nodata in those bands. As in `gdalwarp` reading the source's
  own nodata value (`UNIFIED_SRC_NODATA` unset, which `GDALWarpOperation::WarpRegionToBuffer`
  runs as `PARTIAL`: unified mask built, per-band masks kept), the centre pixel is now invalid
  only if it is nodata in every selected band (`TileExtractor::bands`, like `gdalwarp -b`);
  otherwise each band is interpolated from its own valid taps (tap validity stays per band,
  as in GDAL). Single-band and no-nodata output is unchanged; on a 3-band synthetic raster with
  partial nodata, 98.99% of samples at ratio 1.3 (bilinear) matched `gdalwarp`, now 100%, and
  every changed pixel was one that had nodata in some bands only. `gdalwarp -srcnodata 0`
  (`UNIFIED_SRC_NODATA=YES`, taps judged across all bands) is a different GDAL mode that
  cogrs does not follow, as it has no counterpart for a source's own nodata
- The 129th synchronous call made inside one tokio task never returned: the blocking adapter
  (`AsyncToSync`, which backs `CogReader::read_tile`, `sample*` and the other synchronous methods
  of remote and async-opened readers) waited on a join handle that tokio's cooperative budget
  kept answering "pending"
- A COG overwritten under the same name (or a local file rewritten in place) was served from
  decoded tiles of its previous contents; tiles are now keyed by version
- The tile offset arrays of a remote COG were read without checking they belonged to the same
  object version as the header (a torn open if the object changed in between); all reads after
  the first are now conditional, and an open that fails the check is retried once
- A failed S3 region probe (`AWS_REGION` unset) was repeated, up to 5 s, on every open; failures
  are now remembered for 60 s
- Bilinear and bicubic extraction fetch every source tile their interpolation taps read (the
  2×2 / 4×4 neighbourhood around each sample point), not only the tile of the nearest source
  pixel. Previously taps in an adjacent, unfetched tile were treated as missing and the pixel
  fell back to nearest, leaving faint seams along source-tile boundaries. Nearest planning is
  unchanged
- Rendered tiles were shifted by half a source pixel for `PixelIsArea` rasters (the common
  case): the renderer read the tiepoint-based coordinate as if integer values were pixel
  centres (that is the `PixelIsPoint` convention), so nearest took the pixel half a pixel to
  the south-east of the output pixel's centre and bilinear/bicubic interpolated half a pixel
  off. Source pixels are now sampled by the output pixel's centre (nearest = the source pixel
  containing it, interpolation between pixel centres), as `gdalwarp` does; `PixelIsPoint`
  rasters keep their half-pixel registration, including at overviews (whose geometry now
  derives from the pixel corner like GDAL's). Output pixels whose centre lies outside the
  raster (up to one source pixel past the right/bottom edge before) are now fill. Same-CRS and
  geographic-to-mercator nearest output is bit-identical to `gdalwarp -r near` (previously
  35.9% on a textured fixture); point queries were already consistent with this convention
- Reprojection through per-pixel transforms (UTM and other projected sources) took each output
  pixel's source row from the tile's left edge, so rows were skewed by the meridian
  convergence (about 4.6 source pixels across a z14 tile at 1.7 degrees from the central
  meridian, more at low zoom). Each pixel's source row now comes from its own transform;
  nearest output for a Sentinel-2 UTM 18N source is bit-identical to `gdalwarp -r near` on
  z14 and z8 tiles (14.8% and 0.6% before)
- Bilinear and bicubic extraction downsampled without anti-aliasing: an output pixel covering
  several source pixels (low zoom, or a source level coarser than the output grid) read a
  fixed 2x2 / 4x4 neighbourhood and ignored the rest, so tiles were aliased and differed
  strongly from `gdalwarp`. Like GDAL's warper (`alg/gdalwarpkernel.cpp`), the kernel is now
  stretched by the ratio of source pixels per output pixel on each axis (radius
  `ceil(support * ratio)`, weights `kernel((i - x) / ratio)` normalised by the weights used;
  ratios within 0.05 of a whole number snap to it; below a ratio of about 1.05 on both axes the
  fixed footprint is kept, so upsampling is unchanged), and source tile planning fetches the
  widened footprint. Same-CRS bilinear output is bit-identical to `gdalwarp -r bilinear` at
  ratios 1.5 to 4; on a Sentinel-2 UTM 18N z8 tile the share of identical pixels went from
  11% to 54% and the mean absolute difference from 6.8 to 0.6 (levels of 255). `bicubic`
  stays Mitchell-Netravali (B = C = 1/3) while `gdalwarp -r cubic` is Catmull-Rom, so cubic
  output still differs from GDAL's by the difference of the kernels. Cost grows with the
  square of the ratio (a 256x256 RGB tile at ratio 4 takes 24 ms bilinear / 73 ms bicubic,
  4 / 14 ms before); sources with few overviews pay it at low zooms
- `cargo bench` now works: benchmarks ported to the `TileExtractor` API, the library
  target sets `bench = false` so criterion flags are not passed to libtest, and the
  benchmarks use a synthetic COG generated at startup (or a file given by the
  `COGRS_BENCH_COG` environment variable) instead of a missing test file
- `S3CogSource::scan` ignored `AWS_SKIP_SIGNATURE` (anonymous listing of public buckets
  failed) and panicked when called from async code; it is now fully async
- `S3CogSource::scan` honors `AWS_DEFAULT_REGION` and bucket region detection like `S3Config`
- `TileExtractor` / `Reprojector` fetched too few source tiles when the raster covered only
  part of an output tile: the tiles to read were the bounding box of 9 sample points that
  happened to fall inside the raster, so a raster covering a small interior patch or a corner
  of the tile left most of its pixels as nodata/`NaN` (a low-zoom tile containing a whole scene
  rendered as a few tiles' worth of pixels). The source tiles are now derived from the source
  location of every output pixel, the same coordinates rendering samples through, so every
  output pixel that samples inside the raster has its tile fetched. Fully covered tiles read
  the same tiles as before. Output changes only for tiles that were previously missing data
  (nodata/`NaN` becomes data) and, with bilinear/bicubic resampling, for pixels that used the
  nearest-sample fallback because a neighbouring tile had not been fetched
- Overview pixel geometry used an integer scale (`full_width / overview_width`, floored), which
  is wrong whenever an overview's size is rounded up: a /8 level of a 10980 px raster is 1373
  px wide (ratio 7.997), so it was mapped with scale 7 and everything sampled from it was
  placed up to 12.5% off (a low-zoom tile covering a whole scene lost a strip of its data and
  misplaced the rest). Tile extraction and overview selection now use the exact per-axis ratios
  `full_width / width` and `full_height / height` (new `OverviewMetadata::scale_x` /
  `scale_y`), which is how GDAL derives an overview's geotransform. Output changes only for
  overviews whose size is not an exact division of the full size (power-of-two overviews of
  power-of-two-sized rasters are unchanged). Point queries read full resolution only and are
  unaffected

## [0.0.4] - 2025-12-10

### Added

- `TileExtractor` builder pattern for fluent XYZ tile extraction API
- `PointQuery` trait with `sample_lonlat()` and `sample_crs()` methods
- `CoordTransformer` for reusable CRS-to-CRS coordinate transforms
- `ResamplingMethod` enum with `Nearest`, `Bilinear`, and `Bicubic` options
- Band selection via `TileExtractor::bands(&[0, 1, 2])`
- Concurrent tile extraction with `extract_xyz_tiles_concurrent()`
- `JPEG` compression support for RGB COGs
- `LZW` 16-bit sample support
- Floating-point predictor (`predictor=3`) support
- `Point` struct in geometry module
- Export `LocalRangeReader`, `HttpRangeReader`, `create_range_reader`
- Criterion benchmarks for tile extraction and point queries
- RGB test COG (Natural Earth) with global coverage

### Changed

- Tile extraction is now async-only (removed sync API duplication)
- Reorganized `lib.rs` with clean categorized exports
- Improved documentation (README, rustdoc examples)

### Fixed

- `predictor=2` multi-band handling (was incorrectly accumulating across bands, causing striping artifacts)

## [0.0.3] - 2025-12-05

### Added

- `OverviewQualityHint` for pre-computed overview quality control

## [0.0.2] - 2025-12-05

### Added

- `MemoryRangeReader` for in-memory COG parsing
- Global LRU tile cache (512MB default) with overview index support
- GitHub Actions CI workflow
- Cache statistics retrieval

### Changed

- Renamed from `geocog` to `cogrs`
- Use `proj4rs` for all CRS transformations (pure Rust)
- Use `ahash` for faster `HashMap` lookups
- Optimized pixel loop with precomputed X and row-level Y transforms
- Added fast inline `EPSG:3857`↔`EPSG:4326` transform (2x speedup for `EPSG:4326` COGs)

### Fixed

- `merc_y_to_lat` formula (was using `PI/2` instead of `PI`)

## [0.0.1] - 2025-12-04

### Added

- Initial release extracted from `tileyolo`
- `CogReader` for reading Cloud Optimized GeoTIFFs
- Local and S3 range reader support
- `DEFLATE`, `LZW`, and `ZSTD` compression support
- XYZ tile extraction
- Basic coordinate projection utilities
