#![doc = include_str!("../README.md")]
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`cog_reader`]: Core COG metadata parsing and tile reading
//! - [`point_query`]: Geographic coordinate sampling via [`PointQuery`] trait
//! - [`xyz_tile`]: XYZ tile extraction with [`TileExtractor`] builder
//! - [`geometry`]: Coordinate types ([`Point`], [`BoundingBox`]) and projections
//! - [`range_reader`]: I/O abstraction for local/HTTP/S3 sources
//! - [`source`]: COG discovery from directories and S3 buckets
//! - [`tile_cache`]: Global LRU cache for decompressed tiles
//! - [`s3`]: S3-compatible storage backend
//! - [`raster`]: Raster data abstraction trait
//! - [`geotiff_writer`]: Write reprojected rasters to GeoTIFF files
//! - [`webp`]: Lossless RGBA WebP encoding of extracted tiles ([`TileData::to_webp`])

// ============================================================================
// Public modules
// ============================================================================

pub mod async_io;
pub mod casting;
pub mod cog_reader;
pub mod geometry;
pub mod geotiff_writer;
pub mod lzw_fallback;
pub mod point_query;
mod point_query_async;
pub mod range_reader;
pub mod raster;
pub mod s3;
pub mod source;
pub mod tiff_chunked;
pub mod remote;
pub mod tiff_utils;
pub mod tile_cache;
mod tile_fetch;
pub mod xyz_tile;
pub mod webp;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod extract_tests;

// ============================================================================
// Core COG Types
// ============================================================================

pub use cog_reader::{
    CogReader,
    CogMetadata,
    CogDataType,
    Compression,
    GeoTransform,
    OverviewMetadata,
    OverviewQualityHint,
};

// ============================================================================
// Point Queries
// ============================================================================

pub use point_query::{
    PointQuery,
    PointQueryResult,
    sample_point,
    sample_point_crs,
};

// ============================================================================
// XYZ Tile Extraction
// ============================================================================
// Primary API: TileExtractor::new(&reader).xyz(...).extract().await

pub use xyz_tile::{
    TileData,
    TileExtractor,
    BoundingBox,
    CoordTransformer,
    ResamplingMethod,
};

pub use webp::WebpOptions;

// ============================================================================
// Raster Reprojection
// ============================================================================
// Primary API: Reprojector::new(&reader).to_crs(...).extract().await

pub use xyz_tile::{
    Reprojector,
    ReprojectedRaster,
    StreamingReprojector,
    StreamingOutputInfo,
    RasterChunk,
};

// ============================================================================
// Geometry & Projections
// ============================================================================

pub use geometry::Point;
pub use geometry::projection::{
    project_point,
    lon_lat_to_mercator,
    mercator_to_lon_lat,
    try_lon_lat_to_mercator,
    try_mercator_to_lon_lat,
    get_proj_string,
    is_geographic_crs,
};

// ============================================================================
// Range Readers (I/O Abstraction)
// ============================================================================

pub use range_reader::{
    RangeReader,
    LocalRangeReader,
    HttpRangeReader,
    MemoryRangeReader,
    create_range_reader,
};

pub use async_io::{
    AsyncRangeReader,
    AsyncToSync,
    IoOptions,
    RangeFetchError,
    SyncToAsync,
    fetch_ranges,
};
pub use remote::{ObjectStoreRangeReader, create_async_range_reader};

// ============================================================================
// S3 Support
// ============================================================================

pub use s3::{
    S3Config,
    S3RangeReaderAsync,
    S3RangeReaderSync,
};

// ============================================================================
// Source Discovery
// ============================================================================

pub use source::{
    CogSource,
    CogEntry,
    CogLocation,
    LocalCogSource,
    LocalScanOptions,
    LocalSourceStats,
    S3CogSource,
    S3ScanOptions,
    S3SourceStats,
};

// ============================================================================
// Caching
// ============================================================================

pub use tile_cache::TileCache;

// ============================================================================
// Raster Abstraction
// ============================================================================

pub use raster::RasterSource;

// ============================================================================
// GeoTIFF Writing
// ============================================================================

pub use geotiff_writer::{
    GeoTiffCompression,
    GeoTiffWriteError,
    GeoTiffWriter,
};
