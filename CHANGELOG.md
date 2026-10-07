# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

- **Breaking:** `TileData` has a new public field `nodata`; code constructing it with a
  struct literal must set it
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
