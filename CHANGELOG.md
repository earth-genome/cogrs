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
- `PrefixCachedRangeReader`: serves reads inside the first 16 KiB of a source from
  memory. `CogReader::open` now wraps non-local readers with it, so opening a remote
  COG costs a single header request instead of many small sequential ones
- `RangeReader::has_prefix_cache()` (default method, returns `false`)
- `CogReader::open_async()` / `CogReader::open_async_with_hint()`: async entry points that
  run the blocking open (S3 region detection, header/IFD reads, overview analysis) on
  tokio's blocking pool. Safe on `multi_thread` and `current_thread` runtimes; use these
  for S3/HTTP sources from async code
- `CogReader::spawn_blocking(|reader| ..)`: run any sync operation (point queries, tile
  reads) on an opened reader on the blocking pool
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
- Blocking remote readers (`S3RangeReaderSync`, `HttpRangeReader`, legacy `S3RangeReader`,
  and therefore `CogReader::open` for `s3://` / `http(s)://`) called on a tokio worker
  thread now try to convert tokio's "Cannot start a runtime from within a runtime" panic
  into an error naming `CogReader::open_async` (best effort, string-matched). This works
  with unwinding panics only, and tokio's panic message is still printed by the panic hook;
  with `panic = "abort"` the process aborts as before; and reqwest's blocking client only
  detects the condition in debug builds, so in release builds `HttpRangeReader` may instead
  silently block the worker thread. Always use `open_async` from async code

### Fixed

- `cargo bench` now works: benchmarks ported to the `TileExtractor` API, the library
  target sets `bench = false` so criterion flags are not passed to libtest, and the
  benchmarks use a synthetic COG generated at startup (or a file given by the
  `COGRS_BENCH_COG` environment variable) instead of a missing test file
- `S3CogSource::scan` panicked when called from async code (it opened each COG with the
  blocking S3 reader on the runtime thread); metadata reads now run on the blocking pool
- `S3CogSource::scan` honors `AWS_DEFAULT_REGION` and bucket region detection like `S3Config`

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
