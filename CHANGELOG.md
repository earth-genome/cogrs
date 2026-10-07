# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `PrefixCachedRangeReader`: serves reads inside the first 16 KiB of a source from
  memory. `CogReader::open` now wraps non-local readers with it, so opening a remote
  COG costs a single header request instead of many small sequential ones
- `RangeReader::has_prefix_cache()` (default method, returns `false`)

### Fixed

- `cargo bench` now works: benchmarks ported to the `TileExtractor` API, the library
  target sets `bench = false` so criterion flags are not passed to libtest, and the
  benchmarks use a synthetic COG generated at startup (or a file given by the
  `COGRS_BENCH_COG` environment variable) instead of a missing test file

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
